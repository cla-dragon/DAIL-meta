import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.network_base import LangGoalEncoder, VisualEncoder
from networks.network_seq import Transformer as TransformerSeq
from networks.networks_llfp import LayerNorm

class PureVisEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        layers = [int(config.state_emb_size/2), config.state_emb_size]
        self.conv1 = nn.Conv2d(3, layers[0], kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(layers[0])
        self.conv2 = nn.Conv2d(layers[0], layers[1], kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(layers[1])

        self.pool = nn.MaxPool2d(7, 2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.pool(out)
        out = F.relu(out)
        out = self.flatten(out)
        
        return out

class StateActionEncoder(nn.Module):
    '''
    Need a state encoder in the main network as input
    '''
    def __init__(self, config, action_size, state_encoder) -> None:
        super().__init__()
        self.state_encoder = state_encoder
        feature_size = config.state_emb_size
        # self.state_encoder = ResNetEncoder(env.observation_space, features_dim=feature_size)
        self.action_encoder = nn.Sequential(
            nn.Embedding(action_size, feature_size),
        )
        
        self.ff1 = nn.Linear(feature_size, feature_size)
        self.relu1 = nn.ReLU()
        
        self.ff2 = nn.Linear(feature_size, feature_size)
        self.relu2 = nn.ReLU()
        
        self.ff3 = nn.Linear(feature_size, feature_size)
        self.relu3 = nn.ReLU()
        
    
    def forward(self, state, action):
        state_emb = self.state_encoder(state)
        action_emb = self.action_encoder(action.to(torch.int))[:, 0, :]
        
        out = self.ff1(state_emb * action_emb)
        out += state_emb  # Add the saved output of the first layer (jump connection)
        out = self.relu1(out)
        
        out = self.ff2(out * action_emb)
        out += state_emb  # Add the saved output of the first layer (jump connection)
        out = self.relu2(out)
        
        return out, action_emb


class VideoEncoder(nn.Module):
    def __init__(self, device, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.device = device

        self.vis_encoder = PureVisEncoder(config)
        self.state_action_encoder = StateActionEncoder(config, 7, self.vis_encoder)
        #TODO Add transformer for videoencoder

        transformer_width = config.state_emb_size
        transformer_heads = transformer_width // 64
        transformer_layers = 12
        
        self.transformerClip = TransformerSeq(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
        )
        scale = transformer_width ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(transformer_width, config.video_emb_size))
        self.frame_position_embeddings = nn.Embedding(350, config.state_emb_size)
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.ln_post = LayerNorm(config.video_emb_size)

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out
    
    def forward(self, states_seq, actions_seq, attn_mask):
        bs = states_seq.shape[0]
        seq_len = states_seq.shape[1]
        
        states_seq = states_seq.view(bs*seq_len, *states_seq.shape[2:])
        actions_seq = actions_seq.view(bs*seq_len, *actions_seq.shape[2:])
        
        states_actions_seq, _, = self.state_action_encoder(states_seq, actions_seq)
        
        states_actions_seq = states_actions_seq.view(bs, seq_len, -1)
        
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(bs, -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        visual_output = states_actions_seq + frame_position_embeddings
        
        extended_video_mask = (1.0 - attn_mask.unsqueeze(1)) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, attn_mask.size(1), -1)
        
        visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
        visual_output = self.transformerClip(visual_output, extended_video_mask)
        visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
        
        # visual_output = self.ln_post(visual_output)
        visual_output = visual_output @ self.proj
        visual_output = self.ln_post(visual_output)

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)        
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, attn_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        return visual_output
    
class DecisionMaker(nn.Module):
    def __init__(self, action_size, config, hidden_size=1024, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.state_emb_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc_1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc_2 = nn.Linear(int(hidden_size / 2), action_size)

    def forward(self, input, hidden_states=None, cell_states=None, single_step=False):
        if single_step:
            if (hidden_states is not None) and (cell_states is not None):
                x, (h_n, c_n) = self.lstm(input, (hidden_states, cell_states))
            else:
                x, (h_n, c_n) = self.lstm(input)
        else:
            x, (h_n, c_n) = self.lstm(input)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x, h_n, c_n
    
class BCZNetwork(nn.Module):
    def __init__(self, device, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.device = device
        action_size = 7
        self.lang_encoder = LangGoalEncoder(device, config.lang_emb_size)
        self.vis_encoder = VisualEncoder(config)
        self.decision_maker = DecisionMaker(action_size, config)

        # BC-Z Add video encoder & compute the loss between the video encoder output and the goal
        self.video_encoder = VideoEncoder(device, config)

    def forward(self, states_seq, goal, hidden_states=None, cell_states=None, single_step=False):
        lang_emb = self.lang_encoder(goal)

        bs = states_seq.shape[0]
        seq_len = states_seq.shape[1]

        states_seq = states_seq.view(bs*seq_len, *states_seq.shape[2:])
        lang_emb = lang_emb.unsqueeze(1).repeat(1, seq_len, 1).view(bs*seq_len, -1)
        states_emb = self.vis_encoder(states_seq, lang_emb)
        vis_emb = states_emb.view(bs, seq_len, -1)

        output, h_n, c_n = self.decision_maker(vis_emb, hidden_states, cell_states, single_step)
        return output, h_n, c_n
    
    def get_video_lang_emb(self, states_seq, actions_seq, attn_mask, goal):
        video_emb = self.video_encoder(states_seq, actions_seq, attn_mask)
        lang_emb = self.lang_encoder(goal)
        return video_emb, lang_emb