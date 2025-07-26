from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
# from torch._C import device
from torch.utils.data import Dataset, IterableDataset, Sampler
import numpy as np
import torch

import pickle
import io

import networks.CLIP.clip.clip as clip

from collections import deque

import re

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

class SlicedDatasetSeq(Dataset):
    def __init__(
            self,
            path,
            slice_size=1000,
            shuffle=False
    ):
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

    def _load_file_slice(self):
        real_slice_idx = self.sliceidx_mapping[self.curr_slice].item()
        path = os.path.join(self.file_dir_path, f'slice_{real_slice_idx}.pk')

        with open(path, 'rb') as f:
            d = pickle.load(f)
            data= d['data']
        
        self.data = {"states":[], "next_states":[], "action":[], "reward": [], "done":[], "goal": [], "mc": []}
        for p in data:
            traj = p['traj']
            
            state_seq = []
            next_state_seq = []
            action_seq = []
            reward_seq = []
            done_seq = []
            goal = ""
            mc_seq = []
            
            for t in traj:
                action_array = np.zeros(7)
                action_array[t[4]] = 1
                
                state_seq.append(t[0])
                next_state_seq.append(t[1])
                reward_seq.append(t[2])
                done_seq.append(int(t[3]))
                action_seq.append(action_array)
                goal = t[8] # input ids
                mc_seq.append(t[7])
            
            self.data['states'].append(state_seq)
            self.data['next_states'].append(next_state_seq)
            self.data['action'].append(action_seq)
            self.data['reward'].append(reward_seq)
            self.data['done'].append(done_seq)
            self.data['goal'].append(goal)
            self.data['mc'].append(mc_seq)

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
        mcs = self.data['mc'][real_idx]

        return (states, action, reward, next_states, done, goal, mcs)

class DatasetOri(Dataset):
    def __init__(
            self,
            config,
            path,
            max_words=16,
            max_frames=32,
            history_frame=4,
    ):
        print("Start to load dataset.")
        self.config = config
        self.clip, _ = clip.load("RN50", device=config.device)
        
        self.data = {"states":[], "next_states":[], "action":[], "reward": [], "done":[], "goal": [], "mc": []}
        
        frame_num = history_frame
        
        with open('data/trajs/Synth_mixed_Bot50000_IL50000_Random25000_preprocessed_merged.pk', 'rb') as f:
            self.data = pickle.load(f)
        
        with open('data/goal_tensor.pk', 'rb') as f:
            self.goal_tensor_record = pickle.load(f)
            
        self.max_words = max_words
        self.max_frames = max_frames
        
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
        goal = self.data['goal'][idx]
        mask = self.data['mask'][idx]
        
        goal = self.goal_tensor_record["_".join([str(int(i)) for i in goal[0]])]
        return (states, action, reward, next_states, done, goal, mask)
        
class DatasetMedium(Dataset):
    def __init__(
            self,
            config,
            path,
            max_words=16,
            max_frames=32,
            history_frame=4,
    ):
        print("Start to load dataset.")
        self.config = config
        self.clip, _ = clip.load("RN50", device=config.device)
        
        with open('data/goal_tensor.pk', 'rb') as f:
            self.goal_tensor_record = pickle.load(f)
        
        self.data = {"states":[], "next_states":[], "action":[], "reward": [], "done":[], "goal": [], "mc": []}
        
        frame_num = history_frame
        
        path_preprocessed = "data/trajs/data_Synth_mixed_Bot12500_IL25000_Random40000_clip_preprocessed.pkl"
        if not os.path.exists(path_preprocessed):
            self.preprocess(path, path_preprocessed)
        else:
            with open(path_preprocessed, 'rb') as f:
                self.data = pickle.load(f)

        self.max_words = max_words
        self.max_frames = max_frames
        
        self.size = 256
        print("Data loaded")
    
    def preprocess(self, path, path_preprocessed):
        with open(path, 'rb') as f:
            data = pickle.load(f)['data']
        
        self.data = {
            "states":[], 
            "next_states":[], 
            "action":[], 
            "reward": [], 
            "done":[], 
            "goal": [], 
            'mask': [],
            "clip_tensor": []
        }

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
                goal = p['goal']
            
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
            self.data['clip_tensor'].append(p['goal_clip'])
            
        with open(path_preprocessed, 'wb') as f:
            pickle.dump(self.data, f)
        
    def __len__(self):
        return len(self.data['states'])

    def __getitem__(self, idx):
        states = self.data['states'][idx]
        action = self.data['action'][idx]
        reward = self.data['reward'][idx]
        next_states = self.data['next_states'][idx]
        done = self.data['done'][idx]
        goal = self.data['goal'][idx]
        mask = self.data['mask'][idx]
        goal = self.data['clip_tensor'][idx]
        
        return (states, action, reward, next_states, done, goal, mask)
    
class DatasetSeq(Dataset):
    def __init__(
            self,
            path,
            max_words=16,
            max_frames=32,
    ):
        self.clip, _ = clip.load("RN50", device='cuda:1')
        d = CPU_Unpickler(open(path,"rb")).load()
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
            
        self.max_words = max_words
        self.max_frames = max_frames
        
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


class SlicedDatasetSeq(Dataset):
    def __init__(
            self,
            path,
            slice_size=1000,
            shuffle=False
    ):
        self.goal_tensor_record = {}
        with open('data/trajs/goal_tensor.pk', 'rb') as f:
            self.goal_tensor_record = pickle.load(f)
        
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
            self.goal_tensor_record[goal_ids] = goal_tensor[0].detach()
        return self.goal_tensor_record[goal_ids]

    def _load_file_slice(self):
        real_slice_idx = self.sliceidx_mapping[self.curr_slice].item()
        path = os.path.join(self.file_dir_path, f'slice_{real_slice_idx}.pk')
        
        path_preprocessed = "data/trajs/Synth_mixed_Bot50000_IL50000_Random25000_preprocessed"
        path = os.path.join(path_preprocessed, f'slice_{real_slice_idx}.pk')
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
    
        # with open(path, 'rb') as f:
        #     d = pickle.load(f)
        #     data= d['data']
        
        # self.data = {"states":[], "next_states":[], "action":[], "reward": [], "done":[], "goal": []}
        # for p in data:
        #     traj = p['traj']
            
        #     state_seq = []
        #     next_state_seq = []
        #     action_seq = []
        #     reward_seq = []
        #     done_seq = []
        #     goal = ""
            
        #     for t in traj:
        #         action_array = np.zeros(7)
        #         action_array[t[4]] = 1
                
        #         state_seq.append(t[0])
        #         next_state_seq.append(t[1])
        #         reward_seq.append(t[2])
        #         done_seq.append(int(t[3]))
        #         action_seq.append(action_array)
        #         goal = t[8] # input ids
            
        #     self.data['states'].append(state_seq)
        #     self.data['next_states'].append([state_seq[0]]+next_state_seq)
        #     self.data['action'].append(action_seq)
        #     self.data['reward'].append(reward_seq)
        #     self.data['done'].append(done_seq)
        #     self.data['goal'].append(goal)
            
        #     self.get_goal_tesnor(goal)

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
        return (states, action, reward, next_states, done, goal, mask)
    
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

class DatasetSeqClip(Dataset):
    def __init__(
            self,
            path,
            max_words=16,
            max_frames=32,
    ):
        with open(path, 'rb') as f:
            d = pickle.load(f)
            data= d['data']
        
        self.data = {"states":[], "next_states":[], "action":[], "reward": [], "done":[], "goal": [], "mc": [], "clip": []}
        for p in data:
            traj = p['traj']
            
            state_seq = []
            next_state_seq = []
            action_seq = []
            reward_seq = []
            done_seq = []
            mc_seq = []
            goal =traj[0][8].cpu() # input ids
            goal_clip = p['goal_clip']
            
            for t in traj:
                action_array = np.zeros(7)
                action_array[t[4]] = 1
                
                state_seq.append(t[0])
                next_state_seq.append(t[1])
                reward_seq.append(t[2])
                done_seq.append(int(t[3]))
                action_seq.append(action_array)
                mc_seq.append(t[7])
            
            self.data['states'].append(state_seq)
            self.data['next_states'].append(next_state_seq)
            self.data['action'].append(action_seq)
            self.data['reward'].append(reward_seq)
            self.data['done'].append(done_seq)
            self.data['goal'].append(goal)
            self.data['mc'].append(mc_seq)
            self.data['clip'].append(goal_clip)
        
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
        goal_clip = self.data['clip'][idx]
        mc = self.data['mc'][idx]

        return (states, action, reward, next_states, done, goal, mc, goal_clip)

def collate_fn_clip(batch):
    '''
    Padded actions are 0
    '''
    
    lengths = [len(data[0]) for data in batch]
    
    sorted_indices = sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True)
    
    sorted_batch = [batch[i] for i in sorted_indices]
    
    max_length = max(lengths)
    
    padded_states = torch.zeros(len(batch), max_length, *(batch[0][0][0].shape[1:]))
    padded_next_states = torch.zeros(len(batch), max_length, *(batch[0][3][0].shape[1:]))
    padded_action = torch.zeros(len(batch), max_length, batch[0][1][0].shape[0])
    padded_mask = torch.zeros(len(batch), max_length)
    goals = torch.zeros(len(batch), batch[0][5][0].shape[0])
    goals_clip = torch.zeros(len(batch), batch[0][7][0].shape[0])
    padded_rewards = torch.zeros(len(batch), max_length)
    padded_mcs = torch.zeros(len(batch), max_length)
    
    
    for i, data in enumerate(sorted_batch):
        length = len(data[4])
        
        states_tensor = [torch.from_numpy(arr) for arr in data[0]]
        action_tensor = [torch.from_numpy(arr).view(1, 7) for arr in data[1]]
        next_states_tensor = [torch.from_numpy(arr) for arr in data[3]]
        # reward_tensor = [torch]
        
        padded_states[i, :length, :] = torch.cat(states_tensor, dim=0)
        padded_action[i, :length, :] = torch.cat(action_tensor, dim=0)
        padded_next_states[i, :length, :] = torch.cat(next_states_tensor, dim=0)
        padded_mask[i, :length] = torch.ones(length)
        padded_rewards[i, :length] = torch.tensor(data[2])
        padded_mcs[i, :length] = torch.tensor(data[6])
        
        goals[i, :] = data[5]
        goals_clip[i, :] = data[7]
        
    return (padded_states, padded_action, padded_rewards, padded_next_states, padded_mask, padded_mcs, goals, goals_clip)