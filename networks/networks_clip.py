import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import LazyBatchNorm1d
from networks.resnet import ResNetEncoder, ResNet18

import networks.CLIP.clip.clip as clip
from networks.network_seq import Transformer as TransformerSeq

import gym

from networks.networks_base_babyai import FiLM, ImageConv, StateActionEncoder, LayerNorm, GoalEncoder, GoalEncoderLSTM

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class DecisionMaker(nn.Module):
    def __init__(self, feature_size=512, action_size=7, n_atoms=51) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.action_size = action_size
        self.ff1 = nn.Linear(feature_size, feature_size)
        self.norm1 = nn.LayerNorm(feature_size)
        self.relu1 = nn.ReLU()
        
        self.ff2 = nn.Linear(feature_size, feature_size)
        self.norm2 = nn.LayerNorm(feature_size)
        self.relu2 = nn.ReLU()
        
        self.ff3 = nn.Linear(feature_size, 1)
        
        self.film1 = FiLM(feature_size, feature_size)
        self.film2 = FiLM(feature_size, feature_size)
        
    def forward(self, state_action_emb, goal, h_c=None):        
        state_action_emb = state_action_emb.view(*(state_action_emb.shape[:2]), self.action_size, self.feature_size)
        goal = goal.unsqueeze(2).expand(*(goal.shape[:2]), self.action_size, goal.shape[-1])
        
        x = self.film1(state_action_emb, goal) + state_action_emb
        x = self.film2(x, goal) + x
            
        x = self.ff1(x)
        x = self.relu1(x)
        x = self.ff2(x)
        x = self.relu2(x)
        x = self.ff3(x)
                
        x = x.squeeze(-1)
        return x, h_c

class DecisionMakerLSTM(nn.Module):
    def __init__(self, feature_size=512, lang_feature_size=512, action_size=7, n_atoms=51) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.action_size = action_size
        self.ff1 = nn.Linear(2 * feature_size, feature_size)
        self.norm1 = nn.LayerNorm(feature_size)
        self.relu1 = nn.ReLU()
        
        self.ff2 = nn.Linear(feature_size, feature_size)
        self.norm2 = nn.LayerNorm(feature_size)
        self.relu2 = nn.ReLU()
        
        self.ff3 = nn.Linear(feature_size, 1)
        
        self.film1 = FiLM(lang_feature_size, feature_size)
        self.film2 = FiLM(lang_feature_size, feature_size)
        
    def forward(self, state_action_emb, goal, ht):
        state_action_emb = state_action_emb.view(*(state_action_emb.shape[:2]), self.action_size, self.feature_size)
        goal = goal.unsqueeze(2).expand(*(goal.shape[:2]), self.action_size, goal.shape[-1])
        
        x = self.film1(state_action_emb, goal) + state_action_emb
        x = self.film2(x, goal) + x
            
        x = self.ff1(torch.cat([ht.unsqueeze(2).expand(*(ht.shape[:2]), self.action_size, ht.shape[-1]), x], dim=-1))
        x = self.relu1(x)
        x = self.ff2(x)
        x = self.relu2(x)
        x = self.ff3(x)
                
        x = x.squeeze(-1)
        return x

class DecisionMakerC51(nn.Module):
    def __init__(self, feature_size=512, action_size=7) -> None:
        super().__init__()
        self.n_atoms = 51
        self.feature_size = feature_size
        self.action_size = action_size
        
        self.ff1 = nn.Linear(feature_size, int(feature_size/4))
        self.norm1 = nn.LayerNorm(feature_size)
        self.relu1 = nn.ReLU()
        
        self.ff2 = nn.Linear(int(feature_size/4), self.n_atoms)
        self.norm2 = nn.LayerNorm(feature_size)
        self.relu2 = nn.ReLU()
              
        self.film1 = FiLM(feature_size, feature_size)
        self.film2 = FiLM(feature_size, feature_size)
        
    def forward(self, state_action_emb, goal, h_c=None):
        state_action_emb = state_action_emb.view(*(state_action_emb.shape[:2]), self.action_size, self.feature_size)
        goal = goal.unsqueeze(2).expand(*(goal.shape[:2]), self.action_size, goal.shape[-1])
        
        x = self.film1(state_action_emb, goal) + state_action_emb
        x = self.film2(x, goal) + x
        
        x = self.ff1(x)
        x = self.relu1(x)
        x = self.ff2(x)
        
        out = F.softmax(x.view(*(state_action_emb.shape[:2]), self.action_size, self.n_atoms), dim=-1)
        return out, h_c

class DecisionMakerC51LSTM(nn.Module):
    def __init__(self, feature_size=512, lang_feature_size=512, action_size=7, n_atoms=51) -> None:
        super().__init__()
        self.n_atoms = n_atoms
        self.feature_size = feature_size
        self.action_size = action_size
        
        self.ff1 = nn.Linear(2 * feature_size, feature_size)
        self.norm1 = nn.LayerNorm(feature_size)
        self.relu1 = nn.ReLU()
        
        self.ff2 = nn.Linear(feature_size, int(feature_size/2))
        self.norm2 = nn.LayerNorm(feature_size)
        self.relu2 = nn.ReLU()
        
        self.ff3 = nn.Linear(int(feature_size/2), self.n_atoms)
        self.norm3 = nn.LayerNorm(feature_size)
        self.relu3 = nn.ReLU()
        
        self.film1 = FiLM(lang_feature_size, feature_size)
        self.film2 = FiLM(lang_feature_size, feature_size)
        
    def forward(self, state_action_emb, goal, ht):
        state_action_emb = state_action_emb.view(*(state_action_emb.shape[:2]), self.action_size, self.feature_size)
        goal = goal.unsqueeze(2).expand(*(goal.shape[:2]), self.action_size, goal.shape[-1])
        
        x = self.film1(state_action_emb, goal) + state_action_emb
        x = self.film2(x, goal) + x
        
        x = self.ff1(torch.cat([ht.unsqueeze(2).expand(*(ht.shape[:2]), self.action_size, ht.shape[-1]), x], dim=-1))
        x = self.relu1(x)
        x = self.ff2(x)
        x = self.relu2(x)
        x = self.ff3(x)
        
        x = F.softmax(x.view(*(state_action_emb.shape[:2]), self.action_size, self.n_atoms), dim=-1)
        return x


class Network(nn.Module):
    def __init__(self, env, atoms, action_size, config, hidden_size=64):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.hidden_size_lang = 256
        self.atoms = atoms
        self.action_size = action_size
        self.device = torch.device(config.device)
        
        if config.language_encoder == "LSTM":
            self.goal_encoder = GoalEncoderLSTM(hidden_size)
        else:
            self.goal_encoder = GoalEncoder(self.hidden_size_lang, config.device)
        self.visual_encoder = ImageConv(env.observation_space, history_frame=config.history_frame, features_dim=hidden_size)
        
        self.value_network = ValueNetwork(hidden_size)
        self.actor_network = ActorNetwork(hidden_size)
        
        if ('C51' in config.model_type or 'QR' in config.model_type) and config.LSTM:
            Q_Net = DecisionMakerC51LSTM
        elif 'C51' not in config.model_type and config.LSTM:
            Q_Net = DecisionMakerLSTM
        elif 'C51' in config.model_type and not config.LSTM:
            Q_Net = DecisionMakerC51
        else:
            Q_Net = DecisionMaker
        
        self.q_net_1 = Q_Net(hidden_size, self.hidden_size_lang, n_atoms=config.n_atoms)
        self.q_net_2 = Q_Net(hidden_size, self.hidden_size_lang, n_atoms=config.n_atoms)
        
        self.q_target_net_1 = Q_Net(hidden_size, self.hidden_size_lang, n_atoms=config.n_atoms)
        self.q_target_net_2 = Q_Net(hidden_size, self.hidden_size_lang, n_atoms=config.n_atoms)
        
        self.q_target_net_1.load_state_dict(self.q_net_1.state_dict())
        self.q_target_net_2.load_state_dict(self.q_net_2.state_dict())
        
        self.q_target_net_1.eval()
        self.q_target_net_2.eval()
            
        embed_dim = hidden_size
        transformer_width = hidden_size
        transformer_heads = transformer_width // 64
        transformer_layers = 12
        
        self.transformerClip = TransformerSeq(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
        )
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
        
        scale = transformer_width ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(transformer_width, embed_dim))
        self.frame_position_embeddings = nn.Embedding(300, hidden_size)
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.ln_post = LayerNorm(transformer_width)
        
        self.init_out = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    
    def update(self):
        self.q_target_net_1.load_state_dict(self.q_net_1.state_dict())
        self.q_target_net_2.load_state_dict(self.q_net_2.state_dict())
    
    def get_policy(self, states, goals):
        lang_emb = self.goal_encoder(goals)
        vis_emb = self.visual_encoder(states)
        action = self.actor_network(vis_emb, lang_emb)
        return action
    
    def get_init_out(self):
        return self.init_out
    
    def get_next_ht(self, states, action, ht):
        states_seq_ = states.view(*(states.shape[:2]), self.action_size, self.hidden_size)
        
        action = torch.tensor(action, device=self.device).unsqueeze(0).unsqueeze(0)
        action = action[..., None].expand(*(action.shape[:2]), 1, self.hidden_size)
        
        states_actions_seq = states_seq_.gather(dim=2, index=action).squeeze(2)
        
        out, ht = self.lstm(states_actions_seq, ht)
        return out, ht
    
    def get_q_values(self, states, goals_emb, ht, target=False):
        # batch_size x seq_len x frame_size x pic_dims (3x7x7)
        
        if target:
            q_values_1 = self.q_target_net_1(states, goals_emb, ht)
            q_values_2 = self.q_target_net_2(states, goals_emb, ht)
        else:
            q_values_1 = self.q_net_1(states, goals_emb, ht)
            q_values_2 = self.q_net_2(states, goals_emb, ht)
        
        if 'C51' in self.config.model_type:
            # probs_1 = F.softmax(q_values_1, dim=-1)
            # probs_2 = F.softmax(q_values_2, dim=-1)
            probs_1, probs_2= q_values_1, q_values_2
            q_values_1 = (self.atoms * probs_1).sum(dim=-1)
            q_values_2 = (self.atoms * probs_2).sum(dim=-1)
            
            return q_values_1, q_values_2, probs_1, probs_2
        elif 'QR' in self.config.model_type:
            return q_values_1, q_values_2
        else:  
            return q_values_1, q_values_2
    
    def get_state_values(self, states, goals):
        goal_emb = self.goal_encoder(goals)
        states_emb = self.visual_encoder(states)
        values = self.value_network(states_emb, goal_emb)
        
        return values
    
    def get_seq_emb(self, states_seq, actions_seq, attn_mask):
        bs = states_seq.shape[0]
        seq_len = states_seq.shape[1]
        
        actions_seq = torch.argmax(actions_seq, dim=2, keepdim=True)
        actions_seq = actions_seq[..., None].expand(*(actions_seq.shape[:2]), 1, self.hidden_size)
        
        states_seq_ = states_seq.view(*(states_seq.shape[:2]), self.action_size, self.hidden_size)
        states_actions_seq = states_seq_.gather(dim=2, index=actions_seq).squeeze(2)
        
        packed_input = pack_padded_sequence(states_actions_seq, attn_mask.sum(1).to('cpu'), batch_first=True, enforce_sorted=False)
        
        
        output, _ = self.lstm(packed_input)
        
        output, _ = pad_packed_sequence(output, batch_first=True)
        
        return output

class ActorNetwork(nn.Module):
    def __init__(self, feature_size=512, action_size=7) -> None:
        super().__init__()
        self.action_size = action_size
        self.ff1 = nn.Linear(feature_size, feature_size)
        self.norm1 = nn.LayerNorm(feature_size)
        self.relu1 = nn.ReLU()
        
        self.ff2 = nn.Linear(feature_size, feature_size)
        self.norm2 = nn.LayerNorm(feature_size)
        self.relu2 = nn.ReLU()
        
        self.ff3 = nn.Linear(feature_size, 7)
        
        self.film1 = FiLM(feature_size, feature_size)
        self.film2 = FiLM(feature_size, feature_size)
        
    def forward(self, state_action_emb, goal):
        x = self.film1(state_action_emb, goal) + state_action_emb
        x = self.film2(x, goal) + x
        
        x = self.ff1(x)
        x = self.relu1(x)
        x = self.ff2(x)
        x = self.relu2(x)
        x = self.ff3(x)
        
        return x
    
class ValueNetwork(nn.Module):
    def __init__(self, feature_size) -> None:
        super().__init__()
        self.ff1 = nn.Linear(feature_size, feature_size)
        self.norm1 = nn.LayerNorm(feature_size)
        self.relu1 = nn.ReLU()
        
        self.ff2 = nn.Linear(feature_size, int(feature_size/2))
        self.norm2 = nn.LayerNorm(feature_size)
        self.relu2 = nn.ReLU()
        
        self.ff3 = nn.Linear(int(feature_size/2), 1)
        
        self.film1 = FiLM(feature_size, feature_size)
        self.film2 = FiLM(feature_size, feature_size)

    def forward(self, state_emb, goal):
        x = self.film1(state_emb, goal) + state_emb
        x = self.film2(x, goal) + x
        
        x = self.ff1(x)
        x = self.relu1(x)
        x = self.ff2(x)
        x = self.relu2(x)
        x = self.ff3(x)
        
        return x
