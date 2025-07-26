import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import networks.CLIP.clip.clip as clip
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import os
import numpy as np

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class FiLM(nn.Module):
    def __init__(self, in_features, out_features):
        super(FiLM, self).__init__()
        self.gamma_fc = nn.Linear(in_features, out_features)
        self.beta_fc = nn.Linear(in_features, out_features)

    def forward(self, x, condition):
        gamma = self.gamma_fc(condition)
        gamma = F.tanh(gamma)
        beta = self.beta_fc(condition)
        
        return gamma * x + beta

def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
            
class ImageBOWEmbedding(nn.Module):
   def __init__(self, max_value, embedding_dim, channel=3):
       super().__init__()
       self.max_value = 11
       self.embedding_dim = embedding_dim
       self.channel = channel
       self.embedding = nn.Embedding(channel * self.max_value, embedding_dim)
       self.apply(initialize_parameters)

   def forward(self, inputs):
       offsets = [i * self.max_value for i in range(self.channel)]
       offsets = torch.Tensor(offsets).to(inputs.device)
       inputs = (inputs + offsets[None, :, None, None]).long()
       return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)

class ImageConv(nn.Module):
    def __init__(self, observation_space: gym.Space, history_frame = 4, features_dim: int = 512) -> None:
        super().__init__()
        
        endpool = False
        
        self.bow = ImageBOWEmbedding(observation_space, int(features_dim/2), 3*history_frame)
        
        self.cnn= nn.Sequential(*[
            nn.Conv2d(
                # in_channels= history_frame * 3, out_channels=int(features_dim/2),
                in_channels= int(features_dim/2), out_channels=int(features_dim/2),
                kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(features_dim/2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(features_dim/2), out_channels=features_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            nn.MaxPool2d(7, 2),
            nn.Flatten()
        ])
        
        self.linear = nn.Linear(features_dim, features_dim * 7)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = x.to(torch.float32)
        bs = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(bs * seq_len, x.shape[2]*x.shape[3], *(x.shape[4:]))
        x = self.bow(x)
        x = self.cnn(x)
        x = self.linear(x)
        x = x.view(bs, seq_len, -1)
        return x

class ImageConvHt(nn.Module):
    def __init__(self, observation_space: gym.Space, history_frame = 4, features_dim: int = 512) -> None:
        super().__init__()
        
        endpool = False
        
        self.bow = ImageBOWEmbedding(observation_space, int(features_dim/2), channel=3)
        
        self.cnn= nn.Sequential(*[
            nn.Conv2d(
                in_channels= int(features_dim/2), out_channels=int(features_dim/2),
                kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(int(features_dim/2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(features_dim/2), out_channels=features_dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(features_dim),
            nn.ReLU(),
            nn.MaxPool2d(7, 2),
            nn.Flatten()
        ])
        
        self.linear = nn.Linear(512, features_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = x.to(torch.float32)
        bs = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(bs * seq_len, x.shape[2], *(x.shape[3:]))
        x = x.permute(0, 3, 1, 2)
        x = self.cnn(x)
        x = self.linear(x)
        x = x.view(bs, seq_len, -1)
        return x

class StateActionEncoder(nn.Module):
    def __init__(self, env, history_frame, action_size, feature_size = 512) -> None:
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Embedding(action_size, feature_size),
        )
        
        self.ff1 = nn.Linear(feature_size, feature_size)
        self.norm1 = nn.LayerNorm(feature_size)
        self.relu1 = nn.ReLU()
        
        self.ff2 = nn.Linear(feature_size, feature_size)
        self.norm2 = nn.LayerNorm(feature_size)
        self.relu2 = nn.ReLU()
        
        self.ff3 = nn.Linear(feature_size, feature_size)
        self.norm3 = nn.LayerNorm(feature_size)
        self.relu3 = nn.ReLU()
        
        self.film1 = FiLM(feature_size, feature_size)
        self.film2 = FiLM(feature_size, feature_size)
        
    
    def forward(self, state_emb, action):
        action_emb = self.action_encoder(action.to(torch.int))[:, 0, :]
        
        out = self.film1(state_emb, action_emb) + state_emb
        out = self.film2(out, action_emb) + out
        out = self.relu1(out)
        
        return out

class GoalEncoder(nn.Module):
    def __init__(self, hidden_size, device) -> None:
        super().__init__()     
        self.clip, _ = clip.load("RN50", device=device)
        self.clip.visual = None
        
        for p in self.clip.parameters():
            p.requires_grad = False
        
        self.layer = nn.Sequential(
            nn.Linear(1024, hidden_size),
            nn.ReLU()
        )
        
        self.clip.text_projection.requires_grad = False
    
    def forward(self, goals):
        return self.layer(goals)

class SelfAttn(nn.Module):
    '''
    self-attention with learnable parameters
    '''

    def __init__(self, dhid):
        super().__init__()
        self.scorer = nn.Linear(dhid, 1)

    def forward(self, inp):
        scores = F.softmax(self.scorer(inp), dim=1)
        cont = scores.transpose(1, 2).bmm(inp).squeeze(1)
        return cont

class GoalEncoderLSTM(nn.Module):
    def __init__(self, hidden_size=None) -> None:
        super().__init__()
        feature_size = hidden_size
        
        self.emb_word = nn.Embedding(50000, feature_size)
        
        self.enc = nn.LSTM(feature_size, int(feature_size/2), bidirectional=True, batch_first=True)
        self.enc_att = SelfAttn(feature_size)
        self.pad = 0
    
    def forward(self, goals):
        seqs = [torch.tensor(vv, device=device) for vv in goals]
        pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
        seq_lengths = np.array(list(map(len, goals)))
        embed_seq = self.emb_word(pad_seq.to(torch.int))
        packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
        
        enc_lang_goal_instr, _ = self.enc(packed_input)
        enc_lang_goal_instr, lens = pad_packed_sequence(enc_lang_goal_instr, batch_first=True)
        cont_lang_goal_instr = self.enc_att(enc_lang_goal_instr)
        
        return cont_lang_goal_instr


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        # super().__init__()
        n_input_channels = observation_space['image'].shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(4 * n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            size = torch.zeros((12, 7, 7))
            n_flatten = self.cnn(torch.as_tensor(size, dtype=torch.float32).unsqueeze(0)).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = x.view(x.shape[0], x.shape[1]*x.shape[2], *(x.shape[3:]))
        return self.linear(self.cnn(x))

