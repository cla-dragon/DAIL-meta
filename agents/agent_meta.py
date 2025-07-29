import torch
import numpy as np
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from agents.agent_dail import CQLAgentC51LSTM

from networks.networks_meta import StateActionEmbedding, RewardPredictor
from networks.networks_base_babyai import GoalEncoder

@torch.no_grad()
def momentum_update(model_q, model_k, m=0.99):
    """Update the key encoder using momentum"""
    for param_q, param_k in zip(model_q.layer.parameters(), model_k.layer.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)
class CQLAgentNaiveLSTMMeta(CQLAgentC51LSTM):
    def __init__(self, env, action_size, hidden_size=64, device="cpu", config=None):
        super().__init__(env, action_size, hidden_size=hidden_size, device=device, config=config)
        if config.meta_type == "csro":
            self.q_psi = StateActionEmbedding(config.feature_size, 128, config.feature_size).to(self.device)
            self.q_psi_optimizer = torch.optim.Adam(self.q_psi.parameters(), lr=config.q_learning_rate)
            self.meta_lambda = 1
        elif config.meta_type == "unicorn":
            self.reward_predictor = RewardPredictor(config.feature_size, 256, config.feature_size).to(self.device)
            self.reward_predictor_optimizer = torch.optim.Adam(self.reward_predictor.parameters(), lr=config.q_learning_rate)
        elif config.meta_type == "ccm":
            self.momentum_goal_encoder = GoalEncoder(256, config.device).to(self.device)
            for param_q, param_k in zip(self.momentum_goal_encoder.layer.parameters(), self.net.goal_encoder.layer.parameters()):
                param_q.data = param_k.data
    
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
    
    # def focal_loss(self, goals, goals_emb, epsilon=1e-8):
    #     """
    #     计算focal loss矩阵
        
    #     Args:
    #         goals: (bs, n) 目标向量
    #         goals_emb: (bs, n_emb) 目标嵌入向量
    #         epsilon: 防止除零的小常数
            
    #     Returns:
    #         focal loss的均值
    #     """
    #     bs = goals.shape[0]
        
    #     # 计算所有pair之间的L2距离的平方
    #     # goals_emb: (bs, n_emb) -> (bs, 1, n_emb) 和 (1, bs, n_emb)
    #     goals_emb_i = goals_emb.unsqueeze(1)  # (bs, 1, n_emb)
    #     goals_emb_j = goals_emb.unsqueeze(0)  # (1, bs, n_emb)
        
    #     # 计算L2距离的平方: ||goal_emb_i - goal_emb_j||_2^2
    #     l2_dist_squared = torch.sum((goals_emb_i - goals_emb_j) ** 2, dim=-1)  # (bs, bs)
        
    #     # 判断goal_i == goal_j
    #     # goals: (bs, n) -> (bs, 1, n) 和 (1, bs, n)
    #     goals_i = goals.unsqueeze(1)  # (bs, 1, n)
    #     goals_j = goals.unsqueeze(0)  # (1, bs, n)
        
    #     # 检查goals是否相等 (所有维度都相等)
    #     goals_equal = torch.all(goals_i == goals_j, dim=-1)  # (bs, bs)
        
    #     # 构建loss矩阵
    #     # 如果goal_i == goal_j: Loss_ij = ||goal_emb_i - goal_emb_j||_2^2
    #     # 如果goal_i != goal_j: Loss_ij = 1 / (||goal_emb_i - goal_emb_j||_2^2 + epsilon)
    #     loss_matrix = torch.where(
    #         goals_equal,
    #         l2_dist_squared,
    #         1.0 / (l2_dist_squared + epsilon)
    #     )
        
    #     return torch.mean(loss_matrix)

    def csro_loss(self, ht_seqs, unique_goal_embs, masks, k=3):
        bs, seq_len = masks.shape[:2]
        mask_counts = masks.sum(dim=1).to(torch.long)  # [bs]
        unique_goals_norm = unique_goal_embs / unique_goal_embs.norm(dim=-1, keepdim=True)
        # 选择每个序列mask==1的最后k个值
        valid_states_list = []
        selected_counts = []
        
        for i in range(bs):
            if mask_counts[i] > 0:
                # 找到当前序列中mask==1的位置
                valid_indices = torch.where(masks[i] == 1)[0]
                # 选择最后k个，如果不足k个就全选
                selected_indices = valid_indices[-min(k, len(valid_indices)):]
                selected_count = len(selected_indices)
                
                valid_states_list.append(ht_seqs[i, selected_indices])  # [selected_count, fs]
                selected_counts.append(selected_count)
        
        if len(valid_states_list) == 0:
            # 如果没有有效数据，返回零损失
            return torch.tensor(0.0, device=masks.device), torch.tensor(0.0, device=masks.device)
        
        valid_states = torch.cat(valid_states_list, dim=0)  # [new_bs, fs]
        
        # 构建对应的goals
        selected_counts_tensor = torch.tensor(selected_counts, dtype=torch.long, device=masks.device)
        valid_goal_indices = torch.cat([torch.full((count,), i, dtype=torch.long, device=masks.device) 
                                       for i, count in enumerate(selected_counts)])
        valid_goals = unique_goals_norm[valid_goal_indices]  # [new_bs, goal_emb_size]
        
        goals_log_prob_sa = self.q_psi(valid_states, valid_goals.detach())  # [new_bs, 1]
        q_psi_loss = - goals_log_prob_sa.mean()  # 取平均作为损失

        num_goals = unique_goals_norm.shape[0]
        new_bs = valid_states.shape[0]
        q_matrix = []
        for i in range(num_goals):
            goal_i = unique_goals_norm[i:i+1].expand(new_bs, -1)  # [new_bs, goal_emb_size]
            q_values_i = self.q_psi(valid_states, goal_i)  # [new_bs, 1]
            q_matrix.append(q_values_i.squeeze(-1))  # [new_bs]
        
        q_matrix = torch.stack(q_matrix, dim=1)  # [new_bs, num_goals]
        
        # 取出每行对应goal的值
        corresponding_values = q_matrix[torch.arange(new_bs), valid_goal_indices]  # [new_bs]
        
        # 减去每行的均值
        row_means = q_matrix.mean(dim=1)  # [new_bs]
        result = corresponding_values - row_means  # [new_bs]
        
        q_phi_loss = result.mean()  # 取负号和均值作为损失

        return q_psi_loss, q_phi_loss * self.meta_lambda

    def unicorn_loss(self, ht_seqs, unique_goal_embs, rewards, masks, k=3):
        bs, seq_len = masks.shape[:2]
        mask_counts = masks.sum(dim=1).to(torch.long)  # [bs]
        
        # 选择每个序列mask==1的最后k个值
        valid_states_list = []
        valid_rewards_list = []
        selected_counts = []
        
        for i in range(bs):
            if mask_counts[i] > 0:
                # 找到当前序列中mask==1的位置
                valid_indices = torch.where(masks[i] == 1)[0]
                # 选择最后k个，如果不足k个就全选
                selected_indices = valid_indices[-min(k, len(valid_indices)):]
                selected_count = len(selected_indices)
                
                valid_states_list.append(ht_seqs[i, selected_indices])  # [selected_count, fs]
                valid_rewards_list.append(rewards[i, selected_indices])
                selected_counts.append(selected_count)
        
        if len(valid_states_list) == 0:
            # 如果没有有效数据，返回零损失
            return torch.tensor(0.0, device=masks.device), torch.tensor(0.0, device=masks.device)
        
        valid_states = torch.cat(valid_states_list, dim=0)  # [new_bs, fs]
        valid_rewards = torch.cat(valid_rewards_list, dim=0)  # [new_bs]
        
        # 构建对应的goals
        selected_counts_tensor = torch.tensor(selected_counts, dtype=torch.long, device=masks.device)
        valid_goal_indices = torch.cat([torch.full((count,), i, dtype=torch.long, device=masks.device) 
                                       for i, count in enumerate(selected_counts)])
        valid_goals = unique_goal_embs[valid_goal_indices]  # [new_bs, goal_emb_size]

        rewards_predicted = self.reward_predictor(valid_states, valid_goals)  # [new_bs, 2]
        valid_rewards = (valid_rewards > 0).to(torch.long).squeeze(-1)
        loss = F.cross_entropy(rewards_predicted, valid_rewards, reduction='mean')  # [1]

        return loss
    
    def ccm_loss(self, goals, goals_emb, temperature=0.07):
            # Normalize to unit vectors
        with torch.no_grad():
            momentum_update(self.net.goal_encoder, self.momentum_goal_encoder)
            z_q = self.momentum_goal_encoder(goals)
        z_k = goals_emb
        z_q = F.normalize(z_q, dim=1)
        z_k = F.normalize(z_k, dim=1)

        # Compute similarity matrix [B, B]
        logits = torch.matmul(z_q, z_k.T) / temperature  # [B, B]

        # Positive samples are on the diagonal
        labels = torch.arange(z_q.size(0), device=z_q.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss

    def compute_loss(self, states, actions, goals, rewards, dones, masks):
        actions_ = torch.argmax(actions, dim=2, keepdim=True)
        goals_emb = self.net.goal_encoder(goals)
        states_seq = self.net.visual_encoder(states)
        if self.config.meta_type == "focal":
            meta_loss = self.focal_loss(goals, goals_emb)
        elif self.config.meta_type == "csro" or self.config.meta_type == "unicorn":
            focal_loss = self.focal_loss(goals, goals_emb)
        elif self.config.meta_type == "ccm":
            meta_loss = self.ccm_loss(goals, goals_emb)

        unique_goals_emb = goals_emb
        goals_emb = goals_emb.unsqueeze(1).expand(goals_emb.shape[0], states.shape[1], goals_emb.shape[1])
        
        ht = self.net.get_seq_emb(states_seq, actions, masks)
        init_out = self.net.init_out.expand(ht.shape[0], 1, ht.shape[-1])
        ht_ = torch.cat([init_out, ht[:, :-1]], dim=1)

        if self.config.meta_type == "csro":
            meta_psi_loss, meta_phi_loss = self.csro_loss(ht, unique_goals_emb, masks)
            meta_loss = focal_loss + meta_psi_loss + meta_phi_loss
        elif self.config.meta_type == "unicorn":
            unicorn_loss = self.unicorn_loss(ht, unique_goals_emb, rewards, masks)
            meta_loss = focal_loss + unicorn_loss
        
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
        metrics[f'all_{self.config.meta_type}_loss'] = meta_loss
        if self.config.meta_type == "csro":
            metrics['q_psi_loss'] = meta_psi_loss
            metrics['q_phi_loss'] = meta_phi_loss
        elif self.config.meta_type == "unicorn":
            metrics['unicorn_loss'] = unicorn_loss
            metrics['focal_loss'] = focal_loss

        return metrics, ht, torch.min(Q_1, Q_2).detach()