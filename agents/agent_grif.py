import torch
import torch.nn as nn
import numpy as np

from torch.nn.utils import clip_grad_norm_

from networks.network_grif import GRIFNetwork
from algorithms.agent_babyai import CrossEn

class GRIFAgent():
    def __init__(self, config, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.device = device

        self.network = GRIFNetwork(device, config).to(self.device)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        self.align_loss_fct = CrossEn()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)

        self.align_weight = 0.5

    def get_action(self, state, goal, hidden_state=None, cell_state=None):
        state = torch.from_numpy(state['image']).to(self.device).to(torch.float32)
        goal = goal.to(self.device)
        if self.config.pixel_input:
            state = state / 255.0
            goal = goal / 255.0

        if len(state.shape) == 3:
            state = state.unsqueeze(dim=0)
        state = state.permute(0, 3, 1, 2)
        state = state.unsqueeze(dim=1) # (bs=1, seq_len=1, c, h, w)
        
        if len(goal.shape) == 1:
            goal = goal.unsqueeze(dim=0)

        with torch.no_grad():
            action_prob, hidden_state, cell_state = self.network.get_action(state, goal, 
                                                                   hidden_state, 
                                                                   cell_state, 
                                                                   single_step=True)
        action = np.argmax(action_prob.squeeze().cpu().data.numpy())

        return action, hidden_state, cell_state

    def learn(self, experiences, train=True):
        metrics = {}
        if train:
            self.network.train()
            self.optimizer.zero_grad()
        else:
            self.network.eval()
        states_seq, actions_seq, rewards_seq, next_states_seq, attn_masks, mcs, goals_lang, goals_lang_clip = experiences
        states_seq = states_seq.to(self.device).to(torch.float32)
        actions_seq = actions_seq.to(self.device)
        attn_masks = attn_masks.to(self.device)
        goals_lang = goals_lang.to(self.device)
        goals_lang_clip = goals_lang_clip.to(self.device)
        lengths = torch.sum(attn_masks, dim=1)

        # -------------------Action Loss-------------------
        goals_lang_clip_dict = {'clip': goals_lang_clip}
        lang_policy_output, prior_policy_output = self.network(states_seq, lengths, goals_lang_clip_dict)
        actions_pred_lang, _, _ = lang_policy_output
        actions_pred_prior, _, _ = prior_policy_output
        actions_target = torch.argmax(actions_seq, dim=-1)
        
        row_idx = torch.arange(actions_target.size(0)).unsqueeze(1)
        col_idx = torch.arange(actions_target.size(1)).unsqueeze(0).to(self.device)
        mask = col_idx >= lengths.unsqueeze(1)
        actions_target.masked_fill_(mask, -1)

        policy_loss_lang = self.loss_func(actions_pred_lang.reshape(-1, actions_pred_lang.shape[2]),
                                           actions_target.view(-1))
        policy_loss_prior = self.loss_func(actions_pred_prior.reshape(-1, actions_pred_prior.shape[2]),
                                            actions_target.view(-1))
        metrics['policy_loss_lang'] = policy_loss_lang.item()
        metrics['policy_loss_prior'] = policy_loss_prior.item()

        # -------------------Flitering the successful trajectories-------------------
        success_ids = torch.sum(rewards_seq, dim=1) > 0
        if torch.sum(success_ids) == 0:
            return {'total_loss':0}
        states_seq = states_seq[success_ids]
        actions_seq = actions_seq[success_ids]
        attn_masks = attn_masks[success_ids]
        goals_lang = goals_lang[success_ids]
        goals_lang_clip = goals_lang_clip[success_ids]
        lengths = torch.sum(attn_masks, dim=1)

        # -------------------Aligning Loss-------------------
        unique_goals_lang = torch.unique(goals_lang, dim=0, sorted=True)
        unique_positions = torch.tensor([torch.nonzero((goals_lang == element).all(dim=1))[0][0] for element in unique_goals_lang])

        unique_states_seq = states_seq[unique_positions]
        unique_goals_lang_clip = goals_lang_clip[unique_positions]
        unique_goals_lang_clip = {'clip': unique_goals_lang_clip}
        unique_lengths = lengths[unique_positions]

        video_embs, lang_embs, logit_scale = self.network.get_prior_lang_emb(unique_states_seq, unique_lengths, unique_goals_lang_clip)
        video_embs = video_embs / (video_embs.norm(dim=-1, keepdim=True) + 1e-6)
        lang_embs = lang_embs / (lang_embs.norm(dim=-1, keepdim=True) + 1e-6)

        retrieve_logits = logit_scale.exp() * torch.matmul(video_embs, lang_embs.T)

        sim_loss1 = self.align_loss_fct(retrieve_logits)
        sim_loss2 = self.align_loss_fct(retrieve_logits.T)
        sim_loss = self.align_weight * (sim_loss1 + sim_loss2) / 2
        metrics['align_loss'] = sim_loss.item()


        
        loss = policy_loss_lang + policy_loss_prior + sim_loss
        metrics['total_loss'] = loss.item()

        if train:
            loss.backward()
            clip_grad_norm_(self.network.parameters(), 1.)
            self.optimizer.step()

        return metrics