import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist

from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP

from networks.network_iql import IQLNetwork
from utils_general.utils import asymmetric_l2_loss, mean_pooling_for_similarity_visual
from algorithms.agent_babyai import CrossEn

EXP_ADV_MAX = 100.

class IQLAgent():
    def __init__(self, config, device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.device = device

        self.network = IQLNetwork(device, config).to(self.device)
        if config.distributed:
            self.network = nn.SyncBatchNorm.convert_sync_batchnorm(self.network).to(self.device)
            self.network = DDP(self.network, device_ids=[device], find_unused_parameters=False)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        # self.optimizer_clip = torch.optim.Adam(params=self.network.parameters(), lr=config.clip_learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.episodes)
        self.gamma = 0.99
        self.tau = 1.0
        self.tau_l2 = 0.9
        self.beta = 2

        self._on_calls = 0
        self.target_update_interval = 1000

        self.mc_q_weight = 0.8
        self.align_loss_fct = CrossEn()

    def get_action(self, state, goal):
        state = torch.from_numpy(state['image']).to(self.device).to(torch.float32)
        goal = goal.to(self.device)
        if self.config.pixel_input:
            state = state / 255.0
            goal = goal / 255.0

        if len(state.shape) == 3:
            state = state.unsqueeze(dim=0)
        state = state.permute(0, 3, 1, 2)# (bs=1, c, h, w)
        
        if len(goal.shape) == 1:
            goal = goal.unsqueeze(dim=0)

        with torch.no_grad():
            _, _, action_prob = self.network(state, goal)

        action = np.argmax(action_prob.squeeze().cpu().data.numpy())

        return action

    def get_q_values(self, experiences):
        states_init, states, actions, rewards, next_states, dones, goals_lang, goals_state, mcs = experiences
        states = states.to(self.device).to(torch.float32)
        actions = actions.to(self.device).to(torch.float32)
        rewards = rewards.to(self.device).to(torch.float32).reshape(rewards.shape[0], -1)
        next_states = next_states.to(self.device).to(torch.float32)
        dones = dones.to(self.device).to(torch.float32).reshape(dones.shape[0], -1)
        goals_lang = goals_lang.squeeze().to(self.device).to(torch.float32)
        mcs = mcs.to(self.device).to(torch.float32)

        batch_size = states.shape[0]
        if self.config.pixel_input:
            states = states / 255.0
            next_states = next_states / 255.0

        with torch.no_grad():
            Q_values, _, _ = self.network(states, goals_lang)
            Q_values_1, Q_values_2 = Q_values
            actions = torch.argmax(actions, dim=1, keepdim=True)
            Q_values_1 = torch.gather(Q_values_1, dim=1, index=actions).squeeze(dim=1) # (bs,)
        
        return Q_values_1
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def learn(self, experiences, policy_extract=False, train=True):
        if train:
            self.network.train()
            # self.network.q_target_network_1.eval()
            # self.network.q_target_network_2.eval()
            if not policy_extract:
                self._on_calls += 1
        else:
            self.network.eval()
        
        metrics = {}
        states_init, states, actions, rewards, next_states, dones, goals_lang, goals_state, mcs = experiences
        states = states.to(self.device).to(torch.float32)
        actions = actions.to(self.device).to(torch.float32)
        rewards = rewards.to(self.device).to(torch.float32).reshape(rewards.shape[0], -1)
        next_states = next_states.to(self.device).to(torch.float32)
        dones = dones.to(self.device).to(torch.float32).reshape(dones.shape[0], -1)
        goals_lang = goals_lang.squeeze().to(self.device).to(torch.float32)
        mcs = mcs.to(self.device).to(torch.float32)

        batch_size = states.shape[0]
        if self.config.pixel_input:
            states = states / 255.0
            next_states = next_states / 255.0

        with torch.no_grad():
            if self.config.distributed:
                Q_targets_1, Q_targets_2 = self.network.module.get_q_values(states, goals_lang, target=True)
            else:
                Q_targets_1, Q_targets_2 = self.network.get_q_values(states, goals_lang, target=True)
            actions = torch.argmax(actions, dim=1, keepdim=True)
            Q_targets_1 = torch.gather(Q_targets_1, dim=1, index=actions).squeeze(dim=1) # (bs,)
            Q_targets_2 = torch.gather(Q_targets_2, dim=1, index=actions).squeeze(dim=1) # (bs,)

            Q_targets = torch.min(Q_targets_1, Q_targets_2)
            _, next_state_values, _ = self.network(next_states, goals_lang)
        
        # Update value function
        _, values, _ = self.network(states, goals_lang)
        # 修改了此处，试一试mc效果
        advantage = Q_targets - values
        if not policy_extract:  # Q and V learning
            v_loss = asymmetric_l2_loss(advantage, self.tau_l2)
            metrics['v_loss'] = v_loss.detach().item()
            
            # Update Q function
            targets_value = (rewards + (self.gamma * next_state_values * (1 - dones))).squeeze(dim=1)
            if self.config.use_mc_help:
                # targets_value = (1 - self.mc_q_weight) * targets_value + self.mc_q_weight * mcs
                targets_value = torch.max(targets_value, mcs)
            Q_values, _, _ = self.network(states, goals_lang) #(bs, n_actions)
            Q_a_s_1, Q_a_s_2 = Q_values
            Q_a_s_1 = torch.gather(Q_a_s_1, dim=1, index=actions).squeeze(dim=1) #(bs,)
            Q_a_s_2 = torch.gather(Q_a_s_2, dim=1, index=actions).squeeze(dim=1) #(bs,)

            q_loss_1 = F.mse_loss(Q_a_s_1, targets_value)
            q_loss_2 = F.mse_loss(Q_a_s_2, targets_value)
            q_loss = q_loss_1 + q_loss_2
            metrics['q_loss(two)'] = q_loss.detach().item()

            loss = q_loss + v_loss
            metrics['total_loss'] = loss.detach().item()

        else:
            # Policy extraction
            exp_advantage = torch.exp(self.beta * advantage.detach()).clamp(max=EXP_ADV_MAX)
            _, _, action_prob = self.network(states, goals_lang)
            action_prob = F.log_softmax(action_prob, dim=1) #(bs, n_actions)
            data_action_prob = torch.gather(action_prob, dim=1, index=actions).squeeze(dim=1) #(bs,)
            actor_loss = - (data_action_prob * exp_advantage).mean()
            loss = actor_loss
            metrics['actor_loss'] = loss.detach().item()
    
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.network.parameters(), 1.)
            self.optimizer.step()
            # Update target Q network
            if self._on_calls % self.target_update_interval == 0:
                if self.config.distributed and dist.get_rank() == 0:
                    self.soft_update(self.network.module.q_network_1, self.network.module.q_target_network_1)
                    self.soft_update(self.network.module.q_network_2, self.network.module.q_target_network_2)
                elif not self.config.distributed:
                    self.soft_update(self.network.q_network_1, self.network.q_target_network_1)
                    self.soft_update(self.network.q_network_2, self.network.q_target_network_2)

        return metrics

    # def clip_learn(self, seq_batch):
    #     states_seq, actions_seq, masks, goals = seq_batch
    #     states_seq = states_seq.to(self.device).to(torch.float32)
    #     attn_mask = masks.to(self.device).to(torch.float32)
    #     goals = goals.to(self.device).to(torch.float32)

    #     visual_output = self.network.get_seq_emb(states_seq, actions_seq, attn_mask)
    #     goals = self.network.lang_goal_encoder(goals)
        
    #     visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)        
    #     visual_output = mean_pooling_for_similarity_visual(visual_output, attn_mask)
    #     visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        
    #     # goals = goals.squeeze(1)
    #     goals = goals / goals.norm(dim=-1, keepdim=True)

    #     logit_scale = self.network.logit_scale.exp()
    #     retrieve_logits = logit_scale * torch.matmul(goals, visual_output.t())
        
    #     sim_loss1 = self.align_loss_fct(retrieve_logits)
    #     sim_loss2 = self.align_loss_fct(retrieve_logits.T)
    #     sim_loss = (sim_loss1 + sim_loss2) / 2
    #     loss = sim_loss
    #     metrics = {}
    #     metrics['sim_loss'] = loss.detach().item()
        
    #     self.optimizer_clip.zero_grad()
    #     loss.backward()
    #     clip_grad_norm_(self.network.parameters(), 1.)
    #     self.optimizer_clip.step()
        
    #     return metrics