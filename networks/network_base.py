import torch
import torch.nn as nn
import torch.nn.functional as F

import networks.CLIP.clip.clip as clip

class LangGoalEncoder(nn.Module):
    def __init__(self, device, feature_size=512) -> None:
        super().__init__()     
        self.clip, _ = clip.load("RN50", device=device)
        # Close the visual part
        self.clip.visual = None
        for p in self.clip.parameters():
            p.requires_grad = False
        
        self.layer = nn.Sequential(
            nn.Linear(1024, feature_size),
            nn.ReLU(),
        )
        
        self.clip.text_projection.requires_grad = False
    
    def forward(self, goals):
        if len(goals.shape) == 3:
            goals = goals[:, 0, :]
        goals = self.clip.encode_text(goals.to(torch.int)).to(torch.float32)

        output = self.layer(goals)
        return output

class FiLM(nn.Module):
    def __init__(self, in_features, out_features):
        super(FiLM, self).__init__()
        self.gamma_fc = nn.Linear(in_features, out_features)
        self.beta_fc = nn.Linear(in_features, out_features)

    def forward(self, x, condition):
        gamma = self.gamma_fc(condition)
        gamma = F.tanh(gamma)
        beta = self.beta_fc(condition)
        
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        beta = beta.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        
        return gamma * x + beta
    
class VisualEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        layers = [int(config.state_emb_size/2), config.state_emb_size]
        self.conv1 = nn.Conv2d(3, layers[0], kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(layers[0])
        self.conv2 = nn.Conv2d(layers[0], layers[1], kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(layers[1])
        if config.use_film:
            self.film1 = FiLM(config.lang_emb_size, config.state_emb_size)
            self.film2 = FiLM(config.lang_emb_size, config.state_emb_size)
        else:
            self.conbinefc1 = nn.Linear(config.state_emb_size + config.lang_emb_size, config.state_emb_size)
            self.conbinefc2 = nn.Linear(config.state_emb_size + config.lang_emb_size, config.state_emb_size)

        self.pool = nn.MaxPool2d(7, 2)
        self.flatten = nn.Flatten()

    def forward(self, x, lang_emb):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        if self.config.use_film:
            film_out = self.film1(out, lang_emb) + out
            lang_out = self.film2(film_out, lang_emb) + film_out
            out = self.pool(lang_out)
            out = F.relu(out)
            out = self.flatten(out)
        else:
            out = self.pool(out)
            out = F.relu(out)
            out = self.flatten(out)
            lang_combine_input = torch.cat((out, lang_emb), dim=-1)
            lang_out = F.relu(self.conbinefc1(lang_combine_input)) + out
            out = F.relu(self.conbinefc2(lang_combine_input)) + lang_out
        
        return out
    
class DecisionMaker(nn.Module):
    def __init__(self, action_size, config, hidden_size=2048, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.state_emb_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc_1 = nn.Linear(hidden_size, int(hidden_size / 4))
        self.fc_2 = nn.Linear(int(hidden_size / 4), action_size)

    def forward(self, input, hidden_states=None, cell_states=None, single_step=False):
        x, (h_n, c_n) = self.lstm(input, (hidden_states, cell_states))

        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x, h_n.detach(), c_n.detach()
      
class BaseNetwork(nn.Module):
    def __init__(self, device, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.device = device
        action_size = 7
        self.lang_encoder = LangGoalEncoder(device, config.lang_emb_size)
        self.vis_encoder = VisualEncoder(config)
        self.decision_maker = DecisionMaker(action_size, config)

    def forward(self, states_seq, goal, hidden_states=None, cell_states=None, single_step=False):
        lang_emb = self.lang_encoder(goal)

        bs = states_seq.shape[0]
        seq_len = states_seq.shape[1]

        states_seq = states_seq.reshape(bs*seq_len, *states_seq.shape[2:])
        lang_emb = lang_emb.unsqueeze(1).repeat(1, seq_len, 1).view(bs*seq_len, -1)
        states_emb = self.vis_encoder(states_seq, lang_emb)
        vis_emb = states_emb.view(bs, seq_len, -1)

        output, h_n, c_n = self.decision_maker(vis_emb, hidden_states, cell_states, single_step)
        return output, h_n, c_n
    
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