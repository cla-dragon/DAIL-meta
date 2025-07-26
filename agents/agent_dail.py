import torch
import torch.nn as nn
from networks.networks_clip import Network
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from collections import deque

import networks.CLIP.clip.clip as clip

EXP_ADV_MAX = 100.

class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

class CQLAgent():
    def __init__(self, env, action_size, hidden_size=64, device="cpu", config=None):
        self.config = config
        self.action_size = action_size
        self.device = device
        self.tau = 1e-3
        self.gamma = config.gamma
        self.last_action = 0
        
        self.loss_fct = CrossEn()
        
        self.ct = 0
        
        self.v_max = 20
        self.v_min = -20
        self.n_atoms = config.n_atoms
        
        self.atoms = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        
        self.net = Network(env=env,
                            atoms=self.atoms,
                            action_size=self.action_size,
                            config = config,
                            hidden_size=hidden_size
                            ).to(self.device)
        
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=config.q_learning_rate)
        self.optimizer_clip = optim.Adam(params=self.net.parameters(), lr=config.clip_learning_rate)
        
        self.clip, _ = clip.load("RN50", device=device)
        self.goal_tensor_record = {}
        
        self.target_action_gap = config.target_gap
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=config.q_learning_rate) 
        
    def reset(self):
        frame_num = self.config.history_frame
        self.state_buffer = deque([], maxlen=frame_num)
        for _ in range(frame_num):
            self.state_buffer.append(torch.zeros(3, 7, 7))
    
    def get_goal_tesnor(self, goals):
        goal_ids = "_".join([str(int(i)) for i in goals[0]])
        if goal_ids not in self.goal_tensor_record:
            with torch.no_grad():
                goal_tensor = self.clip.encode_text(goals.to(torch.int).to(torch.device(self.device))).to(torch.float32)
            self.goal_tensor_record[goal_ids] = goal_tensor[0].detach()
        return self.goal_tensor_record[goal_ids]
    
    def get_goal_tesnor_str(self, goals):
        if goals not in self.goal_tensor_record:
            with torch.no_grad():
                goal_tensor = self.clip.encode([goals]).to(torch.float32)
            self.goal_tensor_record[goals] = goal_tensor[0].detach()
        return self.goal_tensor_record[goals]
    
    def get_action(self, state, goals, hidden_state, cell_state):
        pass

    def cql_loss(self, q_values, current_action, masks):
        """Computes the CQL loss for a batch of Q-values and actions."""
        
        q_values = q_values[masks.to(torch.bool)]
        current_action = current_action[masks.to(torch.bool)]
        
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = (q_values * current_action).sum(dim=1).unsqueeze(1)
    
        return (logsumexp - q_a).mean()
    
    def asymmetric_l2_loss(self, u, tau):
        return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)
    
    def vae_loss_func(self, recon_x, x, mu, logvar):
        kl_weight = 0.00025
        BCE = F.mse_loss(recon_x, x)
        KLD = torch.mean(-0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1), 0)
        return BCE + KLD * kl_weight
    
    def compute_loss(states, next_states, actions, goals, rewards, dones):
        pass
    
    def slice(self, experiences):
        states, actions, rewards, next_states, dones, goals, masks = experiences
        
        masks = masks.bool()
        
        # batch_size seq_len stack_frames frame_shape
        frames_num = self.config.history_frame
        added_frames = torch.zeros((states.shape[0], frames_num-1, *(states.shape[2:])), device=self.device)
        states_added = torch.cat((added_frames, states), dim=1)
        
        states_added = states_added.unsqueeze(1)

        next_states = torch.zeros_like(states_added, device=self.device)
        next_states[:, :-1] = states_added[:, 1:]
        
        states = states_added[masks]
        next_states = next_states[masks]
        actions = actions[masks]
        
        rewards = rewards[masks]
        dones = dones[masks]
        
        padded_goals = []
        for i in range(masks.shape[0]):
            padded_goals.append(goals[i].unsqueeze(0).repeat(torch.sum(masks[i]), 1))
        
        padded_goals = torch.cat(padded_goals, dim=0)
        
        return states, actions, rewards, next_states, dones, padded_goals
    
    
    def learn_step(self, experiences, actor=False):
        states, actions, rewards, next_states, dones, goals = self.slice(experiences)
        
        actions = actions.to(torch.float32)
        states = states.to(torch.float32)
        next_states = next_states.to(torch.float32)
        
        if actor:
            metrics, q = self.actor_loss(states, next_states, actions, goals, rewards, dones)
        else:
            metrics, q = self.compute_loss(states, actions, goals, rewards, dones)
        
        self.optimizer.zero_grad()
        if actor:
            metrics['actor_loss'].backward()
        else:
            metrics['loss'].backward()
        clip_grad_norm_(self.net.parameters(), 1.)
        self.optimizer.step()
            
        # ------------------- update target network ------------------- #
        if self.ct % 1000 == 0:
            self.net.update()
        
        for key in metrics.keys():
            metrics[key] = metrics[key].detach().item()
        return metrics, q
        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
    
    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out
    
    def seq_add_stack_frames(self, states, actions, masks, goals):
        frames_num = self.config.history_frame
        added_frames = torch.zeros((states.shape[0], frames_num-1, *(states.shape[2:])), device=self.device)
        states_added = torch.cat((added_frames, states), dim=1)
        
        stacked_states = torch.zeros((states.shape[0], states.shape[1], frames_num, *(states.shape[2:])), device=self.device)
        
        for i in range(states.shape[1]):
            stacked_states[:, i, :] = states_added[:, i:i+frames_num]
        
        return stacked_states, actions, masks, goals
    
    def clip_learn(self, states_seq, actions_seq, attn_mask, goals):
        """
        states_seq: batch_size x sequence_len x *image_size
        actions_seq: batch_size x sequence_len
        attn_mask: batch_size x sequence_len
        goals: batch_size x goal_size
        """
        states_seq, actions_seq, attn_mask, goals = self.seq_add_stack_frames(states_seq, actions_seq, attn_mask, goals)
        states_seq = states_seq.to(torch.float32)
        
        visual_output = self.net.get_seq_emb(states_seq, actions_seq, attn_mask)
        goals = self.net.goal_encoder(goals)
        
        if not self.config.train_goal_clip:
            goals = goals.detach()
        
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)        
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, attn_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        
        goals = goals / goals.norm(dim=-1, keepdim=True)

        logit_scale = self.net.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(goals, visual_output.t())
        
        sim_loss1 = self.loss_fct(retrieve_logits)
        sim_loss2 = self.loss_fct(retrieve_logits.T)
        sim_loss = (sim_loss1 + sim_loss2) / 2
        loss = sim_loss
        
        self.optimizer_clip.zero_grad()
        loss.backward()
        clip_grad_norm_(self.net.parameters(), 1.)
        self.optimizer_clip.step()
        
        return loss.detach().item()
    
    def get_seq_emb(self, states, actions, masks):
        visual_output = self.net.get_seq_emb(states, actions, masks)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)        
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, masks)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        return visual_output
        
    def clip_loss(self, states, actions, masks, goals_):
        states = states.to(torch.float32)
        
        goals = self.net.goal_encoder(goals_)
        
        # For lstm
        masks = (masks.sum(dim=1)-1)[..., None, None].expand(masks.shape[0], 1, states.shape[-1])
        visual_output = states.gather(dim=1, index=masks.to(torch.int64)).squeeze(1)

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        
        goals_norm = goals / goals.norm(dim=-1, keepdim=True)

        logit_scale = self.net.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(goals_norm, visual_output.t())
        
        sim_loss1 = self.loss_fct(retrieve_logits)
        sim_loss2 = self.loss_fct(retrieve_logits.T)
        sim_loss = (sim_loss1 + sim_loss2) / 2
        loss = sim_loss

        return loss
    
    def clip_loss_grif(self, states, actions, masks, goals_):
        goals = self.net.goal_encoder(goals_)
        
        # For lstm
        masks = (masks.sum(dim=1)-1)[..., None, None].expand(masks.shape[0], 1, states.shape[-1])
        visual_t = states.gather(dim=1, index=masks.to(torch.int64)).squeeze(1)
        visual_0 = states[:, 0]
        visual_output = torch.cat([visual_0, visual_t], dim=-1)
        visual_output = self.net.align_net(visual_output)

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        
        goals_norm = goals / goals.norm(dim=-1, keepdim=True)

        logit_scale = self.net.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(goals_norm, visual_output.t())
        
        sim_loss1 = self.loss_fct(retrieve_logits)
        sim_loss2 = self.loss_fct(retrieve_logits.T)
        sim_loss = (sim_loss1 + sim_loss2) / 2
        loss = sim_loss

        return loss
    
    def get_next_states(self, states):
        states[:, :-1] = states[:, 1:].clone()
        return states
    
    def save_model(self, path, batches):
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'batches': batches,
        }, path)
    
    def load_model(self, path):
        d = torch.load(path)
        self.net.load_state_dict(d['model_state_dict'])
        self.optimizer.load_state_dict(d['optimizer_state_dict'])
        return d['batches']

class CQLAgentC51LSTM(CQLAgent):
    def __init__(self, env, action_size, hidden_size=64, device="cpu", config=None):
        super().__init__(env, action_size, hidden_size=hidden_size, device=device, config=config)
        self.max_time_step = 300
    
    def masked_loss(self, x, y, mask):
        x = x[mask.to(torch.bool)]
        y = y[mask.to(torch.bool)]
        
        loss = -(x * torch.log(y + 1e-6)).sum(-1).mean()

        return loss
    
    def encode_language(self, goals):
        if goals in self.goal_tensor_record:
            goal_tensor = self.goal_tensor_record[goals]
        else:
            # goal_tensor = self.get_goal_tesnor_str(goals)
            goal_tensor = self.get_goal_tesnor(goals)
            self.goal_tensor_record[goals] = goal_tensor

        goal_tensor = goal_tensor.to(torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        goals_emb = self.net.goal_encoder(goal_tensor)
        return goals_emb
    
    def get_m(self, next_values, next_dist, rewards, dones):
        bs = next_values.shape[0]
        seq_len = next_values.shape[1]
        
        next_values = next_values.view(bs * seq_len, -1)
        next_dist = next_dist.view(bs * seq_len, *(next_dist.shape[2:]))        
        rewards = rewards.view(bs * seq_len, -1)
        dones = dones.view(bs * seq_len, -1)
        
        next_actions = torch.argmax(next_values, dim=-1)
            
        next_actions = next_actions[..., None, None].expand(next_actions.shape[0], 1, self.n_atoms)

        next_chosen_dist = next_dist.gather(dim=1, index=next_actions).squeeze(1)
        target_dist = rewards.view(-1, 1).to(torch.float32) + self.gamma * (1-dones.view(-1, 1).to(torch.float32)) * self.atoms
        target_dist.clamp_(self.v_min, self.v_max)
        b = (target_dist - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        l[(l > 0) * (l == u)] -= 1
        u[(u < (self.atoms - 1)) * (l == u)] += 1

        m = torch.zeros(target_dist.size(), dtype=next_chosen_dist.dtype).to(self.device)
        
        offset = torch.linspace(0,
            (next_actions.shape[0] - 1) * self.n_atoms,
            next_actions.shape[0]
        ).long().unsqueeze(1).expand(next_actions.shape[0], self.n_atoms).to(self.device)
        
        m.view(-1).index_add_(0,
                                (l + offset).view(-1),
                                (next_chosen_dist * (u.to(torch.float32) - b)).view(-1)
                                )
        m.view(-1).index_add_(0,
                                (u + offset).view(-1),
                                (next_chosen_dist * (b - l.to(torch.float32))).view(-1)
                                )
        m = m.view(bs, seq_len, -1)
        return m
    
    def get_next_ht(self, state, action, ht):
        self.net.eval()
        state = torch.from_numpy(state['image']).to(torch.float32)
        self.state_buffer.append(state.permute(2, 0, 1))
        
        with torch.no_grad():
            states = torch.stack(list(self.state_buffer), dim=0).to(self.device)
            
            states = states.to(torch.float32).unsqueeze(0).unsqueeze(0)
            states = self.net.visual_encoder(states)
            
            out, ht = self.net.get_next_ht(states, action, ht)
            self.net.train()
            
            return out, ht

    def get_action(self, state, goals, ht, out=None, return_value=False, if_visualize=False):
        if out is None:
            out = self.net.get_init_out()
        
        self.net.eval()
        state = torch.from_numpy(state['image']).to(torch.float32)
        self.state_buffer.append(state.permute(2, 0, 1))
        
        with torch.no_grad():
            states = torch.stack(list(self.state_buffer), dim=0).to(self.device)
            
            goals_emb = self.encode_language(goals)
            
            states = states.to(torch.float32).unsqueeze(0).unsqueeze(0)
            states = self.net.visual_encoder(states)
            action_values_1, action_values_2, action_values_probs1, action_values_probs2 = self.net.get_q_values(states, goals_emb, out)
            action_values = (action_values_1 + action_values_2) / 2
            
        action = np.argmax(action_values.cpu().data.numpy(), axis=2).squeeze(0)
        out, ht = self.net.get_next_ht(states, action, ht)
        self.last_action = int(action)
        self.net.train()
        
        if if_visualize:
            return action, ((action_values_probs1+action_values_probs2)/2).cpu().numpy(), action_values.cpu().numpy(), out, ht
        elif return_value:
            return action, out, ht, action_values.detach().cpu(), ((action_values_probs1+action_values_probs2)/2).detach().cpu()
        else:
            return action, out, ht
        

    def compute_loss(self, states, actions, goals, rewards, dones, masks):
        actions_ = torch.argmax(actions, dim=2, keepdim=True)
        goals_emb = self.net.goal_encoder(goals)
        states_seq = self.net.visual_encoder(states)
        
        goals_emb = goals_emb.unsqueeze(1).expand(goals_emb.shape[0], states.shape[1], goals_emb.shape[1])
        ht = self.net.get_seq_emb(states_seq, actions, masks)
        init_out = self.net.init_out.expand(ht.shape[0], 1, ht.shape[-1])
        ht_ = torch.cat([init_out, ht[:, :-1]], dim=1)
        
        # ====== WARNING ======
        rewards = rewards
        
        with torch.no_grad():
            next_values_1, next_values_2, next_dist_1, next_dist_2 = self.net.get_q_values(states_seq, goals_emb, ht_, target=True)
            
            next_values_1 = self.get_next_states(next_values_1)
            next_values_2 = self.get_next_states(next_values_2)
            next_dist_1 = self.get_next_states(next_dist_1)
            next_dist_2 = self.get_next_states(next_dist_2)
            
            m_1 = self.get_m(next_values_1, next_dist_1, rewards, dones)
            m_2 = self.get_m(next_values_2, next_dist_2, rewards, dones)
            
            m = torch.min(m_1, m_2)
        
        current_values_1, current_values_2, current_dist_1, current_dist_2 = self.net.get_q_values(states_seq, goals_emb, ht_)
        actions_ = actions_[..., None].expand(*(actions_.shape[:2]), 1, self.n_atoms)
        current_dist_1 = current_dist_1.gather(dim=2, index=actions_).squeeze(2)
        current_dist_2 = current_dist_2.gather(dim=2, index=actions_).squeeze(2)
        
        bellman_error_1 = self.masked_loss(m, current_dist_1, masks)
        bellman_error_2 = self.masked_loss(m, current_dist_2, masks)
        
        cql_loss_1 = self.cql_loss(current_values_1, actions, masks)
        cql_loss_2 = self.cql_loss(current_values_2, actions, masks)
        
        metrics = {}
        if self.config.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
            cql_loss_1 = cql_alpha * (cql_loss_1 - self.target_action_gap)
            cql_loss_2 = cql_alpha * (cql_loss_2 - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql_loss_1 - cql_loss_2) * 0.5 
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()
        
            loss = bellman_error_1 + bellman_error_2 + cql_loss_1 + cql_loss_2
            metrics['alpha'] = cql_alpha
        else:
            loss = bellman_error_1 + bellman_error_2 + self.config.alpha * (cql_loss_1 + cql_loss_2)
        
        metrics['loss'] = loss
        metrics['cql_loss'] = (cql_loss_1 + cql_loss_2)/2
        metrics['q_loss'] = (bellman_error_1 + bellman_error_2)/2
        
        return metrics, ht, torch.min(current_values_1, current_values_2).detach()
    
       
    def learn_clip(self, experiences, actor=False):
        metrics = {}
        states, actions, rewards, dones, goals, masks = experiences
        
        frames_num = self.config.history_frame
        added_frames = torch.zeros((states.shape[0], frames_num-1, *(states.shape[2:])), device=self.device)
        states_added = torch.cat((added_frames, states), dim=1)
        
        stacked_states = torch.zeros((states.shape[0], states.shape[1], frames_num, *(states.shape[2:])), device=self.device)
        
        for i in range(states.shape[1]):
            stacked_states[:, i, :] = states_added[:, i:i+frames_num]
        
        states_seq = self.net.visual_encoder(stacked_states)
        ht = self.net.get_seq_emb(states_seq, actions, masks)
        
        clip_loss = self.clip_loss(ht, actions, masks, goals)
        metrics['clip_loss'] = clip_loss
        
        self.optimizer.zero_grad()
        loss = metrics['clip_loss']
        loss.backward()
        clip_grad_norm_(self.net.parameters(), 1.)
        self.optimizer.step()
            
        for key in metrics.keys():
            metrics[key] = metrics[key].detach().item()
        
        return metrics
    
    def learn_step(self, experiences, actor=False):
        metrics = {}
        states, actions, rewards, dones, goals, masks = experiences
        
        states = states.unsqueeze(2)
        metrics, ht, q = self.compute_loss(
            states, 
            actions, 
            goals, 
            rewards, 
            dones, 
            masks)
        
        metrics['loss'] = metrics['loss']
        clip_loss = self.clip_loss(ht, actions, masks, goals)
        metrics['clip_loss'] = clip_loss
        

        self.optimizer.zero_grad()
        if self.config.if_clip:
            loss = metrics['loss'] + 0.2 * clip_loss
        else:
            loss = metrics['loss']
        loss.backward()
        clip_grad_norm_(self.net.parameters(), 1.)
        self.optimizer.step()
            
        # ------------------- update target network ------------------- #
        if (self.ct+1) % self.config.update_frequency == 0:
            self.net.update()
        
        self.ct += 1
        
        for key in metrics.keys():
            metrics[key] = metrics[key].detach().item()
        return metrics, q
    
class CQLAgentNaiveLSTM(CQLAgentC51LSTM):
    def __init__(self, env, action_size, hidden_size=64, device="cpu", config=None):
        super().__init__(env, action_size, hidden_size=hidden_size, device=device, config=config)
    
    def masked_loss(self, x, y, mask):
        x = x[mask.to(torch.bool)]
        y = y[mask.to(torch.bool)]
        
        loss = F.mse_loss(x, y)

        return loss
    
    def get_action(self, state, goals, ht, out=None, return_value=False, forced_action=None, if_visualize=False):
        if out is None:
            out = self.net.get_init_out()
        
        self.net.eval()
        state = torch.from_numpy(state['image']).to(torch.float32)
        self.state_buffer.append(state.permute(2, 0, 1))
        
        with torch.no_grad():
            states = torch.stack(list(self.state_buffer), dim=0).to(self.device)
            
            goals_emb = self.encode_language(goals)
            
            states = states.to(torch.float32).unsqueeze(0).unsqueeze(0)
            states = self.net.visual_encoder(states)
            
            action_values_1, action_values_2 = self.net.get_q_values(states, goals_emb, out)
            action_values = (action_values_1 + action_values_2) / 2
            
        action = np.argmax(action_values.cpu().data.numpy(), axis=2).squeeze(0)
        
        if forced_action is None:
            out, ht = self.net.get_next_ht(states, action, ht)
        else:
            out, ht = self.net.get_next_ht(states, forced_action, ht)
            
        self.last_action = int(action)
        self.net.train()
        
        if if_visualize:
            return action, None, action_values.cpu().numpy(), out, ht
        elif return_value:
            return action, out, ht, action_values.detach().cpu(), None
        else:
            return action, out, ht

    def compute_loss(self, states, actions, goals, rewards, dones, masks):
        actions_ = torch.argmax(actions, dim=2, keepdim=True)
        goals_emb = self.net.goal_encoder(goals)
        states_seq = self.net.visual_encoder(states)
        
        goals_emb = goals_emb.unsqueeze(1).expand(goals_emb.shape[0], states.shape[1], goals_emb.shape[1])
        
        ht = self.net.get_seq_emb(states_seq, actions, masks)
        init_out = self.net.init_out.expand(ht.shape[0], 1, ht.shape[-1])
        ht_ = torch.cat([init_out, ht[:, :-1]], dim=1)
        
        with torch.no_grad():
            Q_targets_1, Q_targets_2 = self.net.get_q_values(states_seq, goals_emb, ht_, target=True)
            
            Q_targets_1 = self.get_next_states(Q_targets_1)
            Q_targets_2 = self.get_next_states(Q_targets_2)
            
            Q_targets = torch.min(Q_targets_1, Q_targets_2)
            
            Q_targets = Q_targets.detach().max(2, keepdim=True)[0]
            
            next_values = rewards.to(torch.float32)\
                + (self.gamma * Q_targets * (1 - dones.to(torch.float32)))
            
            next_values = next_values.squeeze(2)
        
        Q_1, Q_2 = self.net.get_q_values(states_seq, goals_emb, ht_)
        Q_1_ = torch.gather(Q_1, dim=2, index=actions_).squeeze(dim=2) #(bs,)
        Q_2_ = torch.gather(Q_2, dim=2, index=actions_).squeeze(dim=2) #(bs,)
        
        q_loss_1 = self.masked_loss(Q_1_, next_values, masks)
        q_loss_2 = self.masked_loss(Q_2_, next_values, masks)
        q_loss = q_loss_1 + q_loss_2
        
        cql_loss_1 = self.cql_loss(Q_1, actions, masks)
        cql_loss_2 = self.cql_loss(Q_2, actions, masks)
        
        loss = q_loss + self.config.alpha * (cql_loss_1 + cql_loss_2)
            
        metrics = {}
        metrics['loss'] = loss
        metrics['cql_loss'] = (cql_loss_1 + cql_loss_2) / 2
        metrics['q_loss'] = q_loss
        
        return metrics, ht, torch.min(Q_1, Q_2).detach()