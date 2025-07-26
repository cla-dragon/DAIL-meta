import torch

state_emb = torch.randn(2519, 1792)  # Example state embedding
action = torch.randint(0, 7, (2519, 1))  # Example action tensor

action = action.squeeze(1).to(torch.long)  # [batch]
state_emb = state_emb.reshape(state_emb.shape[0], -1, 7)  # [batch, feature_size, 7]
action_expanded = action.view(-1, 1, 1).expand(-1, 256, 1)  # [2519, 256, 1]
s_a = torch.gather(state_emb, dim=2, index=action_expanded)  # [2519, 256, 1]
s_a = s_a.squeeze(2)  # [2519, 256]

print("s_a shape:", s_a.shape)  # Should be [batch, feature_size, 7]