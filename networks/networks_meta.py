import torch
import torch.nn as nn

class StateActionEmbedding(nn.Module):
    '''
    calculate the log probability of p(z|s) where p(z|s) in [0,1]
    '''
    def __init__(self, state_emb_dim, hidden_dim, z_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_emb_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出概率在[0,1]
        )

    def forward(self, state_emb, z):
        h = torch.cat([state_emb, z], dim=-1)
        prob = self.mlp(h)  # [batch, 1] in [0,1]
        log_prob = torch.log(prob + 1e-8)  # 加小常数避免log(0)
        return log_prob.squeeze(-1)  # [batch] - log probability in (-∞, 0]

class RewardPredictor(nn.Module):
    def __init__(self, state_emb_dim, hidden_dim, z_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_emb_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # output reward 0/1
        )

    def forward(self, state_emb, z):
        # action = action.squeeze(1).to(torch.long)  # [batch]
        # state_emb = state_emb.reshape(state_emb.shape[0], -1, 7)  # [batch, feature_size, 7]
        # action_expanded = action.view(-1, 1, 1).expand(-1, state_emb.shape[1], 1)  # [bs, fs, 1]
        # s_a = torch.gather(state_emb, dim=2, index=action_expanded)  # [bs, fs, 1]
        # s_a = s_a.squeeze(2)
        s_a = state_emb
        h = torch.cat([s_a, z], dim=-1)  # concatenate state-action and goal embedding
        return self.mlp(h)  # output reward prediction