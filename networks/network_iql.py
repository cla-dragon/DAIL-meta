import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.network_base import LangGoalEncoder, VisualEncoder, PureVisEncoder
from networks.network_seq import Transformer as TransformerSeq

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
class QNetwork(nn.Module):
    def __init__(self, action_size, config, hidden_size=512, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config = config
        self.action_size = action_size

        self.fc_0 = nn.Linear(config.state_emb_size, hidden_size)
        self.fc_1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc_2 = nn.Linear(int(hidden_size / 2), action_size)

    def forward(self, state_emb):
        x = state_emb
        x = self.fc_0(x)
        x = F.relu(x)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)

        return x
    
class ValueNetwork(nn.Module):
    def __init__(self, config, hidden_size=512, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

        self.fc_0 = nn.Linear(config.state_emb_size, hidden_size)
        self.fc_1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc_2 = nn.Linear(int(hidden_size / 2), 1)

    def forward(self, state_emb):
        x = state_emb
        x = self.fc_0(x)
        x = F.relu(x)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)

        return x

class ActorNetwork(nn.Module):
    def __init__(self, config, action_size, hidden_size=512, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.action_size = action_size

        self.fc_0 = nn.Linear(config.state_emb_size, hidden_size)
        self.fc_1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc_2 = nn.Linear(int(hidden_size / 2), action_size)
    
    def forward(self, state_emb):
        x = state_emb
        x = self.fc_0(x)
        x = F.relu(x)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)

        return x
    
class IQLNetwork(nn.Module):
    def __init__(self, device, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.device = device
        action_size = 7
        self.lang_goal_encoder = LangGoalEncoder(device, config.lang_emb_size)
        self.visual_encoder = VisualEncoder(config)

        self.q_network_1 = QNetwork(action_size, config)
        self.q_network_2 = QNetwork(action_size, config)
        self.q_target_network_1 = QNetwork(action_size, config)
        self.q_target_network_2 = QNetwork(action_size, config)
        self.q_target_network_1.load_state_dict(self.q_network_1.state_dict())
        self.q_target_network_2.load_state_dict(self.q_network_2.state_dict())
        self.q_target_network_1.eval()
        self.q_target_network_2.eval()

        self.value_network = ValueNetwork(config)
        self.actor_network = ActorNetwork(config, action_size)

        self.pure_vis_encoder = PureVisEncoder(config)

        
        transformer_width = config.state_emb_size
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in self.lang_goal_encoder.clip.state_dict() if k.startswith("transformer.resblocks")))
        
        self.transformerClip = TransformerSeq(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
        )
        scale = transformer_width ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(transformer_width, config.lang_emb_size))
        self.frame_position_embeddings = nn.Embedding(50, config.state_emb_size)
        self.ln_post = LayerNorm(transformer_width)
        self.logit_scale = nn.Parameter(torch.ones([]))

    def get_q_values(self, state, goal, target=False):
        lang_emb = self.lang_goal_encoder(goal)
        vis_emb = self.visual_encoder(state, lang_emb)
        if target:
            q_values_1 = self.q_target_network_1(vis_emb)
            q_values_2 = self.q_target_network_2(vis_emb)
        else:
            q_values_1 = self.q_network_1(vis_emb)
            q_values_2 = self.q_network_2(vis_emb)

        return q_values_1, q_values_2
    
    def get_state_values(self, state, goal):
        lang_emb = self.lang_goal_encoder(goal)
        vis_emb = self.visual_encoder(state, lang_emb)
        value = self.value_network(vis_emb)

        return value

    def get_policy(self, state, goal):
        lang_emb = self.lang_goal_encoder(goal)
        vis_emb = self.visual_encoder(state, lang_emb)
        action = self.actor_network(vis_emb)

        return action
    
    def forward(self, state, goal):
        lang_emb = self.lang_goal_encoder(goal)
        vis_emb = self.visual_encoder(state, lang_emb)
        q_values_1 = self.q_network_1(vis_emb)
        q_values_2 = self.q_network_2(vis_emb)
        value = self.value_network(vis_emb)
        action = self.actor_network(vis_emb)

        return (q_values_1, q_values_2), value, action
    
    def get_seq_emb(self, states_seq, actions_seq, attn_mask):
        bs = states_seq.shape[0]
        seq_len = states_seq.shape[1]
        
        states_seq = states_seq.view(bs*seq_len, *states_seq.shape[2:])
        # actions_seq = actions_seq.view(bs*seq_len, *actions_seq.shape[2:])
        
        # states_actions_seq, _, _ = self.state_action_encoder(states_seq, actions_seq)
        states_emb_seq = self.pure_vis_encoder(states_seq)
        
        states_emb_seq= states_emb_seq.view(bs, seq_len, -1)
        
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(bs, -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        visual_output = states_emb_seq + frame_position_embeddings
        
        extended_video_mask = (1.0 - attn_mask.unsqueeze(1)) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, attn_mask.size(1), -1)
        
        visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
        visual_output = self.transformerClip(visual_output, extended_video_mask)
        visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
        
        visual_output = self.ln_post(visual_output)
        visual_output = visual_output @ self.proj
        
        return visual_output

'''
----------------------------------- For History encoded Model -----------------------------------
'''

class HistoryStateModel(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.visual_encoder = PureVisEncoder(config)
        self.lstm = nn.LSTM(config.state_emb_size, 2048,
                            num_layers=2,
                            dropout=0.2,
                            batch_first=True)
        self.fc = nn.Linear(2048, config.state_emb_size)

    def forward(self, state, history=None):
        bs = state.shape[0]
        seq_len = state.shape[1]
        state = state.view(bs*seq_len, *state.shape[2:])
        x = self.visual_encoder(state)
        x = x.view(bs, seq_len, -1)
        if history is None:
            h_0 = torch.zeros(self.lstm.num_layers, bs, self.lstm.hidden_size).to(state.device)
            c_0 = torch.zeros(self.lstm.num_layers, bs, self.lstm.hidden_size).to(state.device)
            history = (h_0, c_0)
        x, h_n = self.lstm(x, history)
        x = self.fc(x)
        x = F.relu(x)

        return x, h_n
    
class FiLM1d(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.gamma_fc = nn.Linear(in_features, out_features)
        self.beta_fc = nn.Linear(in_features, out_features)

    def forward(self, x, condition):
        # 生成gamma和beta
        gamma = self.gamma_fc(condition)
        gamma = F.tanh(gamma)
        beta = self.beta_fc(condition)
        
        # 扩展gamma和beta以匹配输入x的形状
        gamma = gamma.expand_as(x)
        beta = beta.expand_as(x)
        
        # 应用FiLM操作
        return gamma * x + beta
    
class HistoryIQLNetwork(nn.Module):
    def __init__(self, device, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.device = device
        action_size = 7
        self.lang_goal_encoder = LangGoalEncoder(device, config.lang_emb_size)

        self.film1 = FiLM1d(config.lang_emb_size, config.state_emb_size)
        self.film2 = FiLM1d(config.lang_emb_size, config.state_emb_size)

        self.q_network_1 = QNetwork(action_size, config)
        self.q_network_2 = QNetwork(action_size, config)
        self.q_target_network_1 = QNetwork(action_size, config)
        self.q_target_network_2 = QNetwork(action_size, config)
        self.q_target_network_1.load_state_dict(self.q_network_1.state_dict())
        self.q_target_network_2.load_state_dict(self.q_network_2.state_dict())
        for param in self.q_target_network_1.parameters():
            param.requires_grad = False
        for param in self.q_target_network_2.parameters():
            param.requires_grad = False
        self.q_target_network_1.eval()
        self.q_target_network_2.eval()

        self.value_network = ValueNetwork(config)
        self.actor_network = ActorNetwork(config, action_size)

    def get_q_values(self, h_state, goal, target=False):
        lang_emb = self.lang_goal_encoder(goal)

        film_out = self.film1(h_state, lang_emb) + h_state
        state_emb = self.film2(film_out, lang_emb) + film_out
        if target:
            q_values_1 = self.q_target_network_1(state_emb)
            q_values_2 = self.q_target_network_2(state_emb)
        else:
            q_values_1 = self.q_network_1(state_emb)
            q_values_2 = self.q_network_2(state_emb)

        return q_values_1, q_values_2

    def get_lang_goal_emb(self, goal):
        return self.lang_goal_encoder(goal)
    
    def forward(self, h_state, goal):
        # with torch.profiler.record_function("Lang Goal encoding"):
        lang_emb = self.lang_goal_encoder(goal)
        # with torch.profiler.record_function("Other Q values caculateing"):
        film_out = self.film1(h_state, lang_emb) + h_state
        state_emb = self.film2(film_out, lang_emb) + film_out

        q_values_1 = self.q_network_1(state_emb)
        q_values_2 = self.q_network_2(state_emb)
        value = self.value_network(state_emb)
        action = self.actor_network(state_emb)

        return (q_values_1, q_values_2), value, action
    
class SimpleAlignModel(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.align_fc_0 = nn.Linear(config.state_emb_size, config.lang_emb_size)
        self.align_fc_1 = nn.Linear(config.lang_emb_size, config.lang_emb_size)
        self.logit_scale = nn.parameter.Parameter(torch.ones([]))

    def forward(self, state):
        x = self.align_fc_0(state)
        x = F.relu(x)
        x = self.align_fc_1(x)
        x = F.relu(x)
        return x, self.logit_scale
    
# ----------------------------------- For HIQL -----------------------------------

class GeneralizedValueModel(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.film1 = FiLM1d(config.lang_emb_size, config.state_emb_size)
        self.film2 = FiLM1d(config.lang_emb_size, config.state_emb_size)
        self.value_network = ValueNetwork(config)

    def forward(self, hidden_state, goal_emb):
        film_out = self.film1(hidden_state, goal_emb) + hidden_state
        state_emb = self.film2(film_out, goal_emb) + film_out

        value = self.value_network(state_emb)

        return value

class HighlevelPolicy(nn.Module):
    def __init__(self, config, hidden_size=512, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        self.film1 = FiLM1d(config.lang_emb_size, config.state_emb_size)
        self.film2 = FiLM1d(config.lang_emb_size, config.state_emb_size)
        self.fc_0 = nn.Linear(config.state_emb_size, hidden_size)
        self.fc_1 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, config.lang_emb_size)
        self.fc_log_std = nn.Linear(hidden_size, config.lang_emb_size)

        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, hidden_state, goal_emb):
        film_out = self.film1(hidden_state, goal_emb) + hidden_state
        state_emb = self.film2(film_out, goal_emb) + film_out

        x = self.fc_0(state_emb)
        x = F.relu(x)
        x = self.fc_1(x)
        x = F.relu(x)
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_stds = torch.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = torch.distributions.Normal(mean, torch.exp(log_std))

        return distribution