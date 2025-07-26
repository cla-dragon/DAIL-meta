from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
# from torch._C import device
from torch.utils.data import Dataset, IterableDataset, Sampler
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
import torch
from transformers import BertConfig, BertModel, BertTokenizer

import pickle
import io

import networks.CLIP.clip.clip as clip

from collections import deque

import re
import sys

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import sys
sys.path.append('LLM2CLIP/llm2clip')

from transformers import AutoModel, AutoConfig, AutoTokenizer
from PIL import Image
import torch


class DatasetOri(Dataset):
    def __init__(
            self,
            path,
            max_words=16,
            max_frames=32,
            history_frame=4,
    ):
        print("Start to load dataset.")
        self.clip, _ = clip.load("RN50", device='cuda:1')
        with open(path, 'rb') as f:
            d = pickle.load(f)
        # d = CPU_Unpickler(open(path,"rb")).load()
        
        # d = torch.load(path, map_location=torch.device('cpu'))
        # with open(path, 'rb') as f:
        #     d = pickle.load(f)
        data= d['data']
        
        self.data = {"states":[], "next_states":[], "action":[], "reward": [], "done":[], "goal": [], "mc": []}
        
        frame_num = history_frame
        
        self.goal_tensor_record = {}
        
        # if not os.path.exists("data/trajs/Synth_mixed_Bot50000_IL50000_Random25000_merged_stack_frame.pk"):
        #     for idx, p in enumerate(data):
        #         print(f"{idx}", end='\r')
        #         state_buffer = deque([], maxlen=frame_num)
        #         for _ in range(frame_num):
        #             state_buffer.append(torch.zeros(3, 7, 7))
                
        #         traj = p['traj']
        #         state_buffer.append(torch.tensor(traj[0][0]).squeeze(dim=0))
                
        #         for t in traj:
        #             action_array = np.zeros(7)
        #             action_array[t[4]] = 1
        #             self.data['states'].append(torch.stack(list(state_buffer), dim=0).unsqueeze(0))
                    
        #             state_buffer.append(torch.tensor(t[1]).squeeze(dim=0))
        #             self.data['next_states'].append(torch.stack(list(state_buffer), dim=0).unsqueeze(0))
        #             self.data['reward'].append(t[2])
        #             self.data['done'].append(int(t[3]))
        #             self.data['action'].append(action_array)
        #             # self.data['goal'].append(t[5]) # multihot
        #             self.data['goal'].append(t[8]) # input ids
        #             self.data['mc'].append(t[7])
                    
        #             goal_ids = "_".join([str(int(i)) for i in t[8][0]])
        #             if goal_ids not in self.goal_tensor_record:
        #                 with torch.no_grad():
        #                     goal_tensor = self.clip.encode_text(t[8].to(torch.int).to(torch.device("cuda:1"))).to(torch.float32)
        #                 self.goal_tensor_record[goal_ids] = goal_tensor.detach()
                        
        #         self.data['done'][-1] = 1
            
        #     with open('data/trajs/Synth_mixed_Bot50000_IL50000_Random25000_merged_stack_frame.pk', 'wb') as f:
        #         pickle.dump({
        #             "goal_tensor":self.goal_tensor_record,
        #             "data": self.data
        #         }, f)
        
        # else:
        #     with open('data/trajs/Synth_mixed_Bot50000_IL50000_Random25000_merged_stack_frame.pk', 'rb') as f:
        #         d = pickle.load(f)
        #     self.data = d['data']
        #     self.goal_tensor_record = d['goal_tensor']

        with open('data/trajs/data_100000_state_SynthLoc_IN_clip_Alternate_merged.pkl', 'rb') as f:
            self.data = pickle.load(f)
            
        self.max_words = max_words
        self.max_frames = max_frames
        
        # self.size = 8
        self.size = 256
        print("Data loaded")

    def __len__(self):
        return len(self.data['states'])

    def __getitem__(self, idx):
        states = self.data['states'][idx]
        action = self.data['action'][idx]
        reward = self.data['reward'][idx]
        next_states = self.data['next_states'][idx]
        done = self.data['done'][idx]
        mc = self.data['mc'][idx]
        goal_tensor = self.data['goal'][idx]
        
        # goal = self.data['goal'][idx]
        # goal_tensor = self.goal_tensor_record["_".join([str(int(i)) for i in goal[0]])]
        
        # return (states[0], action, reward, next_states[0], done, goal, mc)
        return (states[0], action, reward, next_states[0], done, goal_tensor, mc)
    
class DatasetSeq(Dataset):
    def __init__(
            self,
            path,
            max_words=16,
            max_frames=32,
    ):
        self.clip, _ = clip.load("RN50", device='cuda:1')
        d = CPU_Unpickler(open(path,"rb")).load()
        # with open(path, 'rb') as f:
        #     d = pickle.load(f)
        data= d['data']
        
        self.goal_tensor_record = {}
        
        self.data = {"states":[], "next_states":[], "action":[], "reward": [], "done":[], "goal": []}
        for p in data:
            traj = p['traj']
            
            state_seq = []
            next_state_seq = []
            action_seq = []
            reward_seq = []
            done_seq = []
            goal = ""
            
            for t in traj:
                action_array = np.zeros(7)
                action_array[t[4]] = 1
                
                state_seq.append(t[0])
                next_state_seq.append(t[1])
                reward_seq.append(t[2])
                done_seq.append(int(t[3]))
                action_seq.append(action_array)
                goal = t[8] # input ids
            
            self.data['states'].append(state_seq)
            self.data['next_states'].append(next_state_seq)
            self.data['action'].append(action_seq)
            self.data['reward'].append(reward_seq)
            self.data['done'].append(done_seq)
            self.data['goal'].append(goal)
            
            goal_ids = "_".join([str(int(i)) for i in goal[0]])
            if goal_ids not in self.goal_tensor_record:
                with torch.no_grad():
                    goal_tensor = self.clip.encode_text(goal.to(torch.int).to(torch.device("cuda:1"))).to(torch.float32)
                self.goal_tensor_record[goal_ids] = goal_tensor.detach()
            
            # if len(self.data['states']) > 200:
            #     break
        
        self.max_words = max_words
        self.max_frames = max_frames
        
        # self.size = 8
        self.size = 256

    def __len__(self):
        return len(self.data['states'])

    def __getitem__(self, idx):
        states = self.data['states'][idx]
        action = self.data['action'][idx]
        reward = self.data['reward'][idx]
        next_states = self.data['next_states'][idx]
        done = self.data['done'][idx]
        goal = self.data['goal'][idx]
        goal_tensor = self.goal_tensor_record["_".join([str(int(i)) for i in goal[0]])]
        
        return (states, action, reward, next_states, done, goal_tensor)

class DatasetTransform(Dataset):
    def __init__(
            self,
            path,
            slice_size=1000,
            shuffle=False
    ):
        self.goals = []
        self.goal_tensor_record = {}
        self.clip, _ = clip.load("RN50", device='cuda:1')
        
        self.file_dir_path = path
        self.slice_size = slice_size
        self.shuffle = shuffle
        numbers = map(int, re.findall(r'\d+', self.file_dir_path))
        length = sum(numbers)
        if shuffle:
            self.sliceidx_mapping = torch.randperm(length // self.slice_size)
            self.idx_mapping = torch.randperm(self.slice_size)
        else:
            self.sliceidx_mapping = torch.arange(length // self.slice_size)
            self.idx_mapping = torch.arange(self.slice_size)
        self.curr_slice = 0
        self.transform()
    
    def get_goal_tesnor(self, goals):
        goal_ids = "_".join([str(int(i)) for i in goals[0]])
        if goal_ids not in self.goal_tensor_record:
            with torch.no_grad():
                goal_tensor = self.clip.encode_text(goals.to(torch.int).to(torch.device("cuda:1"))).to(torch.float32)
            self.goal_tensor_record[goal_ids] = goal_tensor[0].detach().to('cpu')
        return self.goal_tensor_record[goal_ids]
    
    def get_goal_tesnor_str(self, goal):
        if goal not in self.goal_tensor_record:
            with torch.no_grad():
                goal_tensor = self.clip.encode_text(goal.to(torch.int).to(torch.device("cuda:1"))).to(torch.float32)
            self.goal_tensor_record[goal] = goal_tensor[0].detach().to('cpu')
        return self.goal_tensor_record[goal]

    def transform(self):
        data_path = "data/trajs/data_100000_state_SynthLoc_IN_clip_Alternate.pkl"
        
        with open(data_path, 'rb') as f:
            d = pickle.load(f)
            data= d['data']
        
        self.data = {"states":[], "next_states":[], "action":[], "reward": [], "done":[], "goal": [], 'mask': []}
        for p in data:
            traj = p['traj']
            
            state_seq = []
            next_state_seq = []
            action_seq = []
            reward_seq = []
            done_seq = []
            
            for t in traj:
                action_array = np.zeros(7)
                action_array[t[4]] = 1
                
                state_seq.append(torch.from_numpy(t[0]))
                next_state_seq.append(torch.from_numpy(t[1]))
                reward_seq.append(t[2])
                done_seq.append(int(t[3]))
                action_seq.append(torch.from_numpy(action_array).view(1, 7))
                # goal = t[8] # input ids
                # goal = p['goal'] # string
            goal = p['goal_clip']
            
            state_seq = torch.cat(state_seq, dim=0)
            action_seq = torch.cat(action_seq, dim=0)
            next_state_seq = torch.cat(next_state_seq, dim=0)
            reward_seq = torch.tensor(reward_seq).unsqueeze(1)
            done_seq = torch.tensor(done_seq).unsqueeze(1)
            mask = torch.ones(state_seq.shape[0])
            
            self.data['states'].append(state_seq)
            self.data['next_states'].append(next_state_seq)
            self.data['action'].append(action_seq)
            self.data['reward'].append(reward_seq)
            self.data['done'].append(done_seq)
            self.data['goal'].append(goal)
            self.data['mask'].append(mask)
            
            # self.get_goal_tesnor_str(goal)
            # self.goals.append(goal)
        
        path = os.path.join('data/trajs/data_100000_state_SynthLoc_IN_clip_Alternate_merged.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
        
        # goal_tensor_generated(self.goals)
        # with open('data/trajs/goal_tensor_clip_alternate.pk', 'wb') as f:
        #     pickle.dump(self.goal_tensor_record, f)
        
    def __len__(self):
        numbers = map(int, re.findall(r'\d+', self.file_dir_path))
        length = sum(numbers)
        return length

    def __getitem__(self, idx):
        file_slice = idx // self.slice_size
        if file_slice != self.curr_slice:
            self.curr_slice = file_slice
            self._load_file_slice()
        idx = idx % self.slice_size
        real_idx = self.idx_mapping[idx].item()
        states = self.data['states'][real_idx]
        action = self.data['action'][real_idx]
        reward = self.data['reward'][real_idx]
        next_states = self.data['next_states'][real_idx]
        done = self.data['done'][real_idx]
        goal = self.data['goal'][real_idx]
        mask = self.data['mask'][real_idx]
        # goal_tensor = self.goal_tensor_record["_".join([str(int(i)) for i in goal[0]])]
        
        # return (states, action, reward, next_states, done, goal_tensor, mask)
        return (states, action, reward, next_states, done, None, mask)

class SlicedDatasetSeq(Dataset):
    def __init__(
            self,
            path,
            slice_size=1000,
            shuffle=False
    ):
        self.goals = []
        self.goal_tensor_record = {}
        self.clip, _ = clip.load("RN50", device='cuda:1')
        
        self.file_dir_path = path
        self.slice_size = slice_size
        self.shuffle = shuffle
        numbers = map(int, re.findall(r'\d+', self.file_dir_path))
        length = sum(numbers)
        if shuffle:
            self.sliceidx_mapping = torch.randperm(length // self.slice_size)
            self.idx_mapping = torch.randperm(self.slice_size)
        else:
            self.sliceidx_mapping = torch.arange(length // self.slice_size)
            self.idx_mapping = torch.arange(self.slice_size)
        self.curr_slice = 0
        self._load_file_slice()
    
    def get_goal_tesnor(self, goals):
        goal_ids = "_".join([str(int(i)) for i in goals[0]])
        if goal_ids not in self.goal_tensor_record:
            with torch.no_grad():
                goal_tensor = self.clip.encode_text(goals.to(torch.int).to(torch.device("cuda:1"))).to(torch.float32)
            self.goal_tensor_record[goal_ids] = goal_tensor[0].detach().to('cpu')
        return self.goal_tensor_record[goal_ids]
    
    def get_goal_tesnor_str(self, goal):
        if goal not in self.goal_tensor_record:
            with torch.no_grad():
                goal_tensor = self.clip.encode_text(goal.to(torch.int).to(torch.device("cuda:1"))).to(torch.float32)
            self.goal_tensor_record[goal] = goal_tensor[0].detach().to('cpu')
        return self.goal_tensor_record[goal]

    def _load_file_slice(self):
        print(self.curr_slice, end='\r')
        real_slice_idx = self.sliceidx_mapping[self.curr_slice].item()
        path = os.path.join(self.file_dir_path, f'slice_{real_slice_idx}.pk')
        
        path_preprocessed = "data/trajs/data_50000_state_SynthLoc_IN_clip_Alternate"
        
        with open(path, 'rb') as f:
            d = pickle.load(f)
            data= d['data']
        
        self.data = {"states":[], "next_states":[], "action":[], "reward": [], "done":[], "goal": [], 'mask': []}
        for p in data:
            traj = p['traj']
            
            state_seq = []
            next_state_seq = []
            action_seq = []
            reward_seq = []
            done_seq = []
            goal = ""
            
            for t in traj:
                action_array = np.zeros(7)
                action_array[t[4]] = 1
                
                state_seq.append(torch.from_numpy(t[0]))
                next_state_seq.append(torch.from_numpy(t[1]))
                reward_seq.append(t[2])
                done_seq.append(int(t[3]))
                action_seq.append(torch.from_numpy(action_array).view(1, 7))
                # goal = t[8] # input ids
                goal = p['goal'] # string
            
            state_seq = torch.cat(state_seq, dim=0)
            action_seq = torch.cat(action_seq, dim=0)
            next_state_seq = torch.cat(next_state_seq, dim=0)
            reward_seq = torch.tensor(reward_seq).unsqueeze(1)
            done_seq = torch.tensor(done_seq).unsqueeze(1)
            mask = torch.ones(state_seq.shape[0])
            
            self.data['states'].append(state_seq)
            self.data['next_states'].append(next_state_seq)
            self.data['action'].append(action_seq)
            self.data['reward'].append(reward_seq)
            self.data['done'].append(done_seq)
            self.data['goal'].append(goal)
            self.data['mask'].append(mask)
            
            self.get_goal_tesnor_str(goal)
            # self.goals.append(goal)
        
        path = os.path.join(path_preprocessed, f'slice_{real_slice_idx}.pk')
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
        
        # goal_tensor_generated(self.goals)
        with open('data/trajs/goal_tensor_clip_alternate.pk', 'wb') as f:
            pickle.dump(self.goal_tensor_record, f)
        
    def __len__(self):
        numbers = map(int, re.findall(r'\d+', self.file_dir_path))
        length = sum(numbers)
        return length

    def __getitem__(self, idx):
        file_slice = idx // self.slice_size
        if file_slice != self.curr_slice:
            self.curr_slice = file_slice
            self._load_file_slice()
        idx = idx % self.slice_size
        real_idx = self.idx_mapping[idx].item()
        states = self.data['states'][real_idx]
        action = self.data['action'][real_idx]
        reward = self.data['reward'][real_idx]
        next_states = self.data['next_states'][real_idx]
        done = self.data['done'][real_idx]
        goal = self.data['goal'][real_idx]
        mask = self.data['mask'][real_idx]
        # goal_tensor = self.goal_tensor_record["_".join([str(int(i)) for i in goal[0]])]
        
        # return (states, action, reward, next_states, done, goal_tensor, mask)
        return (states, action, reward, next_states, done, None, mask)
    
# class SlicedIterDataset(IterableDataset):
class SlicedIterDataset(Dataset):
    def __init__(
            self,
            path,
            slice_size=1000,
            history_frame=2,
    ):
        self.history_frame = history_frame
        self.goal_tensor_record = {}
        self.clip, _ = clip.load("RN50", device='cuda:1')
        
        self.file_dir_path = path
        self.slice_size = slice_size
        
        numbers = map(int, re.findall(r'\d+', self.file_dir_path))
        length = sum(numbers)
        
        self.sliceidx_mapping = torch.randperm(length // self.slice_size)
        self.idx_mapping = torch.randperm(self.slice_size)
        self.curr_slice = 0
        
        self._load_file_slice()
        self.curr_iter = 0
    
    def get_goal_tesnor(self, goals):
        goal_ids = "_".join([str(int(i)) for i in goals[0]])
        if goal_ids not in self.goal_tensor_record:
            with torch.no_grad():
                goal_tensor = self.clip.encode_text(goals.to(torch.int).to(torch.device("cuda:1"))).to(torch.float32)
            self.goal_tensor_record[goal_ids] = goal_tensor[0].detach()
        return self.goal_tensor_record[goal_ids]
    
    def __len__(self):
        numbers = map(int, re.findall(r'\d+', self.file_dir_path))
        length = sum(numbers)
        return length
    
    def _load_file_slice(self):
        real_slice_idx = self.sliceidx_mapping[self.curr_slice].item()
        path = os.path.join(self.file_dir_path, f'slice_{real_slice_idx}.pk')

        with open(path, 'rb') as f:
            d = pickle.load(f)
            data= d['data']
        
        self.data = {"states":[], "next_states":[], "action":[], "reward": [], "done":[], "goal": []}
        for p in data:
            state_buffer = deque([], maxlen=self.history_frame)
            for _ in range(self.history_frame):
                state_buffer.append(torch.zeros(3, 7, 7))
            
            traj = p['traj']
            state_buffer.append(torch.tensor(traj[0][0]).squeeze(dim=0))
            
            for t in traj:
                action_array = np.zeros(7)
                action_array[t[4]] = 1
                self.data['states'].append(torch.stack(list(state_buffer), dim=0))
                
                state_buffer.append(torch.tensor(t[1]).squeeze(dim=0))
                self.data['next_states'].append(torch.stack(list(state_buffer), dim=0))
                self.data['reward'].append(t[2])
                self.data['done'].append(int(t[3]))
                self.data['action'].append(action_array)
                self.data['goal'].append(t[8]) # input ids
                
                goal_ids = "_".join([str(int(i)) for i in t[8][0]])
                if goal_ids not in self.goal_tensor_record:
                    with torch.no_grad():
                        goal_tensor = self.clip.encode_text(t[8].to(torch.int).to(torch.device("cuda:1"))).to(torch.float32)
                    self.goal_tensor_record[goal_ids] = goal_tensor[0].detach()
                    
            self.data['done'][-1] = 1
        
    def _sample(self, idx):
        states = self.data['states'][idx]
        action = self.data['action'][idx]
        reward = self.data['reward'][idx]
        next_states = self.data['next_states'][idx]
        done = self.data['done'][idx]
        goal = self.data['goal'][idx]
        
        goal_tensor = self.goal_tensor_record["_".join([str(int(i)) for i in goal[0]])]
        return (states, action, reward, next_states, done, goal_tensor)

    def __getitem__(self, idx):
        file_slice = idx // self.slice_size
        if file_slice != self.curr_slice:
            self.curr_slice = file_slice
            self._load_file_slice()
        idx = idx % self.slice_size
        real_idx = self.idx_mapping[idx].item()
        states = self.data['states'][real_idx]
        action = self.data['action'][real_idx]
        reward = self.data['reward'][real_idx]
        next_states = self.data['next_states'][real_idx]
        done = self.data['done'][real_idx]
        goal = self.data['goal'][real_idx]
        
        goal_tensor = self.goal_tensor_record["_".join([str(int(i)) for i in goal[0]])]
        return (states, action, reward, next_states, done, goal_tensor)
        # while self.curr_slice < len(self.sliceidx_mapping):
        #     self.curr_iter = 0
        #     while self.curr_iter < self.slice_size:
        #         yield self._sample(self.idx_mapping[self.curr_iter])
        #         self.curr_iter += 1
        #     self.curr_slice += 1
        #     self._load_file_slice()
        # raise StopIteration
    
class MySampler(Sampler):
    def __init__(self, data_source, slice_size=1000, seed = 0):
        self.data_source = data_source
        self.slice_size = slice_size

        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()

        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_source)//self.slice_size, generator=g).tolist()
        indices = indices[self.rank:len(self.data_source)//self.slice_size:self.num_replicas]
        return iter(self.generate_list(indices))
    
    def __len__(self):
        return len(self.data_source)//self.num_replicas
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def generate_list(self, numbers):
        result = []
        length = self.slice_size
        for num in numbers:
            result.extend([num * length + i for i in range(length)])
        return result

def collate_fn(batch):
    lengths = [len(data[0]) for data in batch]
    
    max_length = max(lengths)
    
    padded_states = torch.zeros(len(batch), max_length, *(batch[0][0].shape[1:]), device=device)
    padded_action = torch.zeros(len(batch), max_length, batch[0][1].shape[1], device=device)
    padded_rewards = torch.zeros(len(batch), max_length, 1, device=device)
    padded_dones = torch.zeros(len(batch), max_length, 1, device=device)
    padded_mask = torch.zeros(len(batch), max_length, device=device)
    
    for i, data in enumerate(batch):
        length = data[4].shape[0]
        
        padded_states[i, :length, :] = data[0].to(device)
        padded_action[i, :length, :] = data[1].to(device)
        padded_rewards[i, :length, :] = data[2].to(device)
        padded_dones[i, :length, :] = data[4].to(device)
        padded_mask[i, :length] = data[6].to(device)
        
    return padded_states, padded_action, padded_rewards, padded_dones, None, padded_mask
