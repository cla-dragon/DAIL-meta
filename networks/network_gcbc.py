import torch
import torch.nn as nn
import torch.nn.functional as F

import networks.CLIP.clip.clip as clip

def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class LangGoalEncoder(nn.Module):
    def __init__(self, device, feature_size=512) -> None:
        super().__init__()     
        self.clip, _ = clip.load("RN50", device=device)
        # self.clip = self.clip.to(torch.float32)
        
        # self.clip.train()
        # self.clip.initialize_parameters()
        # Close the visual part
        self.clip.visual = None
        for p in self.clip.parameters():
            p.requires_grad = False
        
        self.layer = nn.Sequential(
            nn.Linear(1024, feature_size),
            # nn.LayerNorm(feature_size),
            nn.ReLU(),
        )
        
        self.clip.text_projection.requires_grad = False
        # for p in self.clip.transformer.resblocks[-1].mlp.c_proj.parameters():
        #     p.requires_grad = True

    def forward(self, goals):
        if type(goals) == dict:
            goals_clip = goals['clip']
        else:
            if len(goals.shape) == 3:
                goals = goals[:, 0, :]
            goals_clip = self.clip.encode_text(goals.to(torch.int)).to(torch.float32)
        # goals_detached = goals.detach()
        output = self.layer(goals_clip)
        return output
        # return goals

class FiLM(nn.Module):
    def __init__(self, in_features, out_features):
        super(FiLM, self).__init__()
        self.gamma_fc = nn.Linear(in_features, out_features)
        self.beta_fc = nn.Linear(in_features, out_features)

    def forward(self, x, condition):
        # 生成gamma和beta
        gamma = self.gamma_fc(condition)
        gamma = F.tanh(gamma)
        beta = self.beta_fc(condition)
        
        # 扩展gamma和beta以匹配输入x的形状
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        beta = beta.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        
        # 应用FiLM操作
        return gamma * x + beta
    
class ImageBOWEmbedding(nn.Module):
   def __init__(self, max_value, embedding_dim):
       super().__init__()
       self.max_value = max_value
       self.embedding_dim = embedding_dim
       self.embedding = nn.Embedding(3 * max_value, embedding_dim)
       self.register_buffer("offsets", torch.tensor([0, self.max_value, 2 * self.max_value]))
    #    self.offsets = torch.Tensor([0, self.max_value, 2 * self.max_value])
       # self.apply(initialize_parameters)

   def forward(self, inputs):
    #    offsets = self.offsets.to(inputs.device)
       inputs = (inputs + self.offsets[None, :, None, None]).long()
       return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)
    
class VisualEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        layers = [int(config.state_emb_size/2), config.state_emb_size]
        self.bow = ImageBOWEmbedding(20, 128)
        self.conv1 = nn.Conv2d(128, layers[0], kernel_size=3, stride=1, padding=1)
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
        out = self.bow(x)
        out = self.conv1(out)
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
        # if (hidden_states is None) or (cell_states is None):
        #     hidden_states = torch.zeros(self.lstm.num_layers, input.size(0), self.lstm.hidden_size).to(input.device)
        #     cell_states = torch.zeros(self.lstm.num_layers, input.size(0), self.lstm.hidden_size).to(input.device)

        x, (h_n, c_n) = self.lstm(input, (hidden_states, cell_states))
        # else:
        #     x, (h_n, c_n) = self.lstm(input)

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
        self.bow = ImageBOWEmbedding(20, 128)
        self.conv1 = nn.Conv2d(128, layers[0], kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(layers[0])
        self.conv2 = nn.Conv2d(layers[0], layers[1], kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(layers[1])
        self.pool = nn.MaxPool2d(7, 2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.bow(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.pool(out)
        out = F.relu(out)
        out = self.flatten(out)
        
        return out