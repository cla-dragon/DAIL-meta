import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.network_base import LangGoalEncoder, PureVisEncoder
from networks.network_bcz import DecisionMaker
from utils_general.encoder import FiLM

class PriorEncoder(nn.Module):
    def __init__(self, config, hidden_size=1024, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.hidden_size = hidden_size
        self.fc_1 = nn.Linear(config.state_emb_size*2, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, hidden_size)
        self.fc_3 = nn.Linear(hidden_size, config.lang_emb_size)

    def forward(self, state_emb_init, state_emb_goal):
        state_emb = torch.cat((state_emb_init, state_emb_goal), dim=1)
        x = F.relu(self.fc_1(state_emb))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))

        return x

class DecisionMaker(nn.Module):
    def __init__(self, action_size, config, hidden_size=1024, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config = config
        self.film1 = FiLM(config.lang_emb_size, config.state_emb_size)
        self.film2 = FiLM(config.lang_emb_size, config.state_emb_size)
        self.lstm = nn.LSTM(
            input_size=config.state_emb_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc_1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc_2 = nn.Linear(int(hidden_size / 2), action_size)

    def forward(self, vis_emb, goal_emb, hidden_states=None, cell_states=None, single_step=False):
        bs = vis_emb.shape[0]
        seq_len = vis_emb.shape[1]
        vis_emb = vis_emb.view(bs*seq_len, *vis_emb.shape[2:])

        states_emb_film1 = self.film1(vis_emb, goal_emb) + vis_emb
        states_emb_film2 = self.film2(states_emb_film1, goal_emb) + states_emb_film1
        input = states_emb_film2.view(bs, seq_len, -1)
        
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

class GRIFNetwork(nn.Module):
    def __init__(self, device, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.device = device
        action_size = 7
        self.lang_encoder = LangGoalEncoder(device, config.lang_emb_size)
        self.vis_encoder = PureVisEncoder(config)
        self.prior_encoder = PriorEncoder(config)
        self.decision_maker = DecisionMaker(action_size, config)
        self.logit_scale = nn.Parameter(torch.ones([]))

    def forward(self, states_seq, episode_length, goal, hidden_states=None, cell_states=None, single_step=False):
        lang_emb = self.lang_encoder(goal)

        bs = states_seq.shape[0]
        seq_len = states_seq.shape[1]

        states_seq = states_seq.view(bs*seq_len, *states_seq.shape[2:])
        lang_emb = lang_emb.unsqueeze(1).repeat(1, seq_len, 1).view(bs*seq_len, -1)
        states_emb = self.vis_encoder(states_seq)
        vis_emb = states_emb.view(bs, seq_len, -1)

        # Compute the prior embedding
        episode_length = episode_length.to(torch.int64) - 1
        state_emb_init = vis_emb[:, 0, :]
        episode_length = episode_length.unsqueeze(-1).expand(-1, vis_emb.shape[-1]) # (bs, state_emb_size)
        state_emb_goal = torch.gather(vis_emb, 1, episode_length.unsqueeze(1)).squeeze(1)
        # state_goal = vis_emb[:, episode_length, :]
        goal_prior_emb = self.prior_encoder(state_emb_init, state_emb_goal)
        goal_prior_emb = goal_prior_emb.unsqueeze(1).repeat(1, seq_len, 1).view(bs*seq_len, -1)
        lang_policy_output = self.decision_maker(vis_emb, lang_emb, hidden_states, cell_states, single_step)
        prior_policy_output = self.decision_maker(vis_emb, goal_prior_emb, hidden_states, cell_states, single_step)
        
        return lang_policy_output, prior_policy_output
    
    def get_prior_lang_emb(self, states_seq, episode_length, goal):
        lang_emb = self.lang_encoder(goal)
        bs = states_seq.shape[0]
        seq_len = states_seq.shape[1]

        states_seq = states_seq.view(bs*seq_len, *states_seq.shape[2:])
        states_emb = self.vis_encoder(states_seq)
        vis_emb = states_emb.view(bs, seq_len, -1)

        # Compute the prior embedding
        episode_length = episode_length.to(torch.int64) - 1
        state_emb_init = vis_emb[:, 0, :]
        episode_length = episode_length.unsqueeze(-1).expand(-1, vis_emb.shape[-1]) # (bs, state_emb_size)
        state_emb_goal = torch.gather(vis_emb, 1, episode_length.unsqueeze(1)).squeeze(1)
        # state_emb_goal = vis_emb[:, episode_length, :]
        # print(f"state_emb_init: {state_emb_init.shape}, state_emb_goal: {state_emb_goal.shape}")
        # print(f"episode_length: {episode_length.shape}")
        goal_prior_emb = self.prior_encoder(state_emb_init, state_emb_goal)

        return goal_prior_emb, lang_emb, self.logit_scale
    
    def get_action(self, states_seq, goal, hidden_states=None, cell_states=None, single_step=True):
        lang_emb = self.lang_encoder(goal)

        bs = states_seq.shape[0]
        seq_len = states_seq.shape[1]

        states_seq = states_seq.view(bs*seq_len, *states_seq.shape[2:])
        lang_emb = lang_emb.unsqueeze(1).repeat(1, seq_len, 1).view(bs*seq_len, -1)
        states_emb = self.vis_encoder(states_seq)
        vis_emb = states_emb.view(bs, seq_len, -1)

        output, h_n, c_n = self.decision_maker(vis_emb, lang_emb, hidden_states, cell_states, single_step)
        return output, h_n, c_n