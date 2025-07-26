import torch
import numpy as np
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from agents.agent_dail import CQLAgentC51LSTM

class CQLAgentNaiveLSTMMeta(CQLAgentC51LSTM):
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
    
    def focal_loss(self, goals, goals_emb, epsilon=1e-8):
        """
        计算focal loss矩阵
        
        Args:
            goals: (bs, n) 目标向量
            goals_emb: (bs, n_emb) 目标嵌入向量
            epsilon: 防止除零的小常数
            
        Returns:
            focal loss的均值
        """
        bs = goals.shape[0]
        
        # 计算所有pair之间的L2距离的平方
        # goals_emb: (bs, n_emb) -> (bs, 1, n_emb) 和 (1, bs, n_emb)
        goals_emb_i = goals_emb.unsqueeze(1)  # (bs, 1, n_emb)
        goals_emb_j = goals_emb.unsqueeze(0)  # (1, bs, n_emb)
        
        # 计算L2距离的平方: ||goal_emb_i - goal_emb_j||_2^2
        l2_dist_squared = torch.sum((goals_emb_i - goals_emb_j) ** 2, dim=-1)  # (bs, bs)
        
        # 判断goal_i == goal_j
        # goals: (bs, n) -> (bs, 1, n) 和 (1, bs, n)
        goals_i = goals.unsqueeze(1)  # (bs, 1, n)
        goals_j = goals.unsqueeze(0)  # (1, bs, n)
        
        # 检查goals是否相等 (所有维度都相等)
        goals_equal = torch.all(goals_i == goals_j, dim=-1)  # (bs, bs)
        
        # 构建loss矩阵
        # 如果goal_i == goal_j: Loss_ij = ||goal_emb_i - goal_emb_j||_2^2
        # 如果goal_i != goal_j: Loss_ij = 1 / (||goal_emb_i - goal_emb_j||_2^2 + epsilon)
        loss_matrix = torch.where(
            goals_equal,
            l2_dist_squared,
            1.0 / (l2_dist_squared + epsilon)
        )
        
        return torch.mean(loss_matrix)

    def csro_loss(self, states, actions, goal_embs):
        pass

    def compute_loss(self, states, actions, goals, rewards, dones, masks):
        actions_ = torch.argmax(actions, dim=2, keepdim=True)
        goals_emb = self.net.goal_encoder(goals)
        states_seq = self.net.visual_encoder(states)
        if self.config.meta_type == "focal":
            meta_loss = self.focal_loss(goals, goals_emb)

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
        
        loss = q_loss + self.config.alpha * (cql_loss_1 + cql_loss_2) + meta_loss
            
        metrics = {}
        metrics['loss'] = loss
        metrics['cql_loss'] = (cql_loss_1 + cql_loss_2) / 2
        metrics['q_loss'] = q_loss
        metrics[f'{self.config.meta_type}_loss'] = meta_loss

        return metrics, ht, torch.min(Q_1, Q_2).detach()