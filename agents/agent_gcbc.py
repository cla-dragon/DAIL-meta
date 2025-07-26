import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel as DDP
from networks.network_base import BaseNetwork


class BaseAgent():
    def __init__(self, config, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.device = device

        self.network = BaseNetwork(device, config).to(self.device)
        if config.distributed:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network).to(self.device)
            self.network = DDP(self.network, device_ids=[device], find_unused_parameters=False)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.max_time_steps = config.max_time_steps
    
    def get_action(self, state, goal, hidden_state=None, cell_state=None):
        state = torch.from_numpy(state['image']).to(self.device).to(torch.float32)
        goal = goal.to(self.device)
        if self.config.pixel_input:
            state = state / 255.0
            goal = goal / 255.0

        if len(state.shape) == 3:
            state = state.unsqueeze(dim=0)
        state = state.permute(0, 3, 1, 2)
        state = state.unsqueeze(dim=1) # (bs=1, seq_len=1, c, h, w)
        
        if len(goal.shape) == 1:
            goal = goal.unsqueeze(dim=0)

        with torch.no_grad():
            if (hidden_state is None) or (cell_state is None):
                if self.config.distributed:
                    hidden_state = torch.zeros(self.network.module.decision_maker.lstm.num_layers, state.size(0), 
                                                self.network.module.decision_maker.lstm.hidden_size).to(self.device)
                    cell_state = torch.zeros(self.network.module.decision_maker.lstm.num_layers, state.size(0), 
                                                self.network.module.decision_maker.lstm.hidden_size).to(self.device)
                else:
                    hidden_state = torch.zeros(self.network.decision_maker.lstm.num_layers, state.size(0), 
                                                self.network.decision_maker.lstm.hidden_size).to(self.device)
                    cell_state = torch.zeros(self.network.decision_maker.lstm.num_layers, state.size(0), 
                                                self.network.decision_maker.lstm.hidden_size).to(self.device)
                
            action_prob, hidden_state, cell_state = self.network(state, goal, 
                                                                   hidden_state, 
                                                                   cell_state, 
                                                                   single_step=True)
        action = np.argmax(action_prob.squeeze().cpu().data.numpy())

        return action, hidden_state, cell_state

    def learn(self, experiences, train=True):
        metrics = {}
        if train:
            self.network.train()
            self.optimizer.zero_grad()
        else:
            self.network.eval()
        states_seq, actions_seq, rewards_seq, next_states_seq, attn_masks, mcs, goals_lang, goals_lang_clip = experiences
        states_seq = states_seq.to(self.device).to(torch.float32)
        if self.config.pixel_input:
            states_seq = states_seq / 255.0
        actions_seq = actions_seq.to(self.device)
        attn_masks = attn_masks.to(self.device)
        # goals_lang = goals_lang.to(self.device)
        goals_lang_clip = goals_lang_clip.to(self.device)
        # -------------------Flitering the successful trajectories-------------------
        if self.config.use_fliter:
            success_ids = torch.sum(rewards_seq, dim=1) > 0
            if torch.sum(success_ids) == 0:
                return {'total_loss':0}
            states_seq = states_seq[success_ids]
            actions_seq = actions_seq[success_ids]
            attn_masks = attn_masks[success_ids]
            goals_lang = goals_lang[success_ids]
            goals_lang_clip = goals_lang_clip[success_ids]
        # -------------------BPTT-------------------
        if not self.config.distributed and self.config.use_bptt:
            h_0 = torch.zeros(self.network.decision_maker.lstm.num_layers, states_seq.size(0), 
                                        self.network.decision_maker.lstm.hidden_size).to(self.device)
            c_0 = torch.zeros(self.network.decision_maker.lstm.num_layers, states_seq.size(0), 
                                        self.network.decision_maker.lstm.hidden_size).to(self.device)
            total_loss = 0
            for start_idx in range(0, states_seq.size(1), self.max_time_steps):
                end_idx = min(start_idx + self.max_time_steps, states_seq.size(1))
                
                states_seq_sub = states_seq[:, start_idx:end_idx, ...]
                actions_seq_sub = actions_seq[:, start_idx:end_idx, ...]
                attn_masks_sub = attn_masks[:, start_idx:end_idx, ...]


                # print(states_seq_sub.shape, actions_seq_sub.shape, attn_masks_sub.shape, goals_lang_sub.shape)
                # exit()
                actions_pred, h_0, c_0 = self.network(states_seq_sub, goals_lang, h_0, c_0)
                # h_0 = h_0.detach()
                # c_0 = c_0.detach()
                actions_target = torch.argmax(actions_seq_sub, dim=-1)
                lengths = torch.sum(attn_masks_sub, dim=1)
                row_idx = torch.arange(actions_target.size(0)).unsqueeze(1)
                col_idx = torch.arange(actions_target.size(1)).unsqueeze(0).to(self.device)
                mask = col_idx >= lengths.unsqueeze(1)
                # For ignoring the unpadded actions
                actions_target.masked_fill_(mask, -1)
                loss = self.loss_func(actions_pred.reshape(-1, actions_pred.shape[2]),
                                                actions_target.view(-1))
                # if train:
                #     loss.backward()
                total_loss = total_loss + loss

        else:
            if self.config.distributed:
                h_0 = torch.zeros(self.network.module.decision_maker.lstm.num_layers, states_seq.size(0), 
                                            self.network.module.decision_maker.lstm.hidden_size).to(self.device)
                c_0 = torch.zeros(self.network.module.decision_maker.lstm.num_layers, states_seq.size(0), 
                                            self.network.module.decision_maker.lstm.hidden_size).to(self.device)
            else:
                h_0 = torch.zeros(self.network.decision_maker.lstm.num_layers, states_seq.size(0), 
                                            self.network.decision_maker.lstm.hidden_size).to(self.device)
                c_0 = torch.zeros(self.network.decision_maker.lstm.num_layers, states_seq.size(0), 
                                            self.network.decision_maker.lstm.hidden_size).to(self.device)
            goals_lang_clip = {'clip': goals_lang_clip}
            actions_pred, _, _ = self.network(states_seq, goals_lang_clip, h_0, c_0)
            actions_target = torch.argmax(actions_seq, dim=-1)
            lengths = torch.sum(attn_masks, dim=1)
            row_idx = torch.arange(actions_target.size(0)).unsqueeze(1)
            col_idx = torch.arange(actions_target.size(1)).unsqueeze(0).to(self.device)
            mask = col_idx >= lengths.unsqueeze(1)
            # For ignoring the unpadded actions
            actions_target.masked_fill_(mask, -1)

            total_loss = self.loss_func(actions_pred.reshape(-1, actions_pred.shape[2]),
                                            actions_target.view(-1))
            # if train:
            #     total_loss.backward()
        
        metrics['total_loss'] = total_loss.item()

        if train:
            total_loss.backward()
            self.optimizer.step()
            return metrics
        else:
            return metrics, actions_pred