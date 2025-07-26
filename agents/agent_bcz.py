import torch
import torch.nn as nn
import numpy as np

from networks.network_bcz import BCZNetwork


class BCZAgent():
    def __init__(self, config, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.device = device

        self.network = BCZNetwork(device, config).to(self.device)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        self.video_loss_func = nn.CosineSimilarity()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
    
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
            action_prob, hidden_state, cell_state = self.network(state, goal, 
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
        if self.config.pixel_input:
            states_seq = states_seq / 255.0
        actions_seq = actions_seq.to(self.device)
        attn_masks = attn_masks.to(self.device)
        goals_lang = goals_lang.to(self.device)
        goals_lang_clip = goals_lang_clip.to(self.device)

        goals_lang_clip_dict = {'clip': goals_lang_clip}
        actions_pred, _, _ = self.network(states_seq, goals_lang_clip_dict)
        actions_target = torch.argmax(actions_seq, dim=-1)
        lengths = torch.sum(attn_masks, dim=1)
        row_idx = torch.arange(actions_target.size(0)).unsqueeze(1)
        col_idx = torch.arange(actions_target.size(1)).unsqueeze(0).to(self.device)
        mask = col_idx >= lengths.unsqueeze(1)
        actions_target.masked_fill_(mask, -1)

        policy_loss = self.loss_func(actions_pred.reshape(-1, actions_pred.shape[2]),
                                           actions_target.view(-1))
        metrics['policy_loss'] = policy_loss.item()

        # -------------------Flitering the successful trajectories-------------------
        success_ids = torch.sum(rewards_seq, dim=1) > 0
        if torch.sum(success_ids) == 0:
            return {'total_loss':0}
        states_seq = states_seq[success_ids]
        actions_seq = actions_seq[success_ids]
        attn_masks = attn_masks[success_ids]
        goals_lang = goals_lang[success_ids]
        goals_lang_clip = goals_lang_clip[success_ids]

        goals_lang_clip_dict = {'clip': goals_lang_clip}
        video_emb, lang_emb = self.network.get_video_lang_emb(states_seq, actions_seq, attn_masks, goals_lang_clip_dict)
        video_loss = 1 - self.video_loss_func(video_emb, lang_emb).mean()
        metrics['video_loss'] = video_loss.item()
        loss = policy_loss + video_loss
        metrics['total_loss'] = loss.item()
        
        if train:
            loss.backward()
            self.optimizer.step()
            return metrics
        else:
            return metrics, actions_pred