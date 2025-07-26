import gym
import numpy as np
from collections import deque
import torch
import argparse
import random
import os

from torch.utils.data import DataLoader
import sys

sys.path.append('../')

from agents.agent_dail import CQLAgentNaiveLSTM, CQLAgentC51LSTM
from agents.agent_meta import CQLAgentNaiveLSTMMeta

from babyai.new_missions import *
from utils.data_loader import DatasetOri

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import networks.CLIP.clip.clip as clip

import pickle

from tensorboardX import SummaryWriter
import wandb

# os.environ['WANDB_MODE'] = 'offline'

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL-DQN", help="Run name, default: CQL-DQN")
    parser.add_argument("--env", type=str, default="CartPole-v0", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes, default: 200")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--batch_size", type=int, default=64, help="default: 256")
    parser.add_argument("--batch_size_clip", type=int, default=128, help="default: 256")
    parser.add_argument("--min_eps", type=float, default=0.01, help="Minimal Epsilon, default: 4")
    parser.add_argument("--eps_frames", type=int, default=1e4, help="Number of steps for annealing the epsilon value to the min epsilon, default: 1e5")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--eval_every", type=int, default=1000, help="")
    parser.add_argument("--all_data", type=bool, default=True, help="")
    parser.add_argument("--ood_test", type=bool, default=True, help="")
    
    parser.add_argument("--train_goal_q", type=bool, default=True, help="")
    parser.add_argument("--train_goal_clip", type=bool, default=True, help="")
    parser.add_argument("--train_state_q", type=bool, default=True, help="")
    parser.add_argument("--train_state_clip", type=bool, default=True, help="")
    
    parser.add_argument("--alpha", type=float, default=2, help="")
    parser.add_argument("--gamma", type=float, default=0.99, help="")
    parser.add_argument("--q_learning_rate", type=float, default=3e-4, help="")
    parser.add_argument("--clip_learning_rate", type=float, default=3e-5, help="")
    
    parser.add_argument("--feature_size", type=int, default=256, help="")
    parser.add_argument("--feature_extract", type=str, default='resnet', help="")
    parser.add_argument("--goal_format", type=str, default='multi-hot', help="")
    parser.add_argument("--n_atoms", type=int, default=51, help="")
    parser.add_argument("--history_frame", type=int, default=1, help="")
    parser.add_argument("--target_gap", type=int, default=0.8, help="")
    parser.add_argument("--update_frequency", type=int, default=1000, help="")
    parser.add_argument("--device", type=str, default=device_str, help="LSTM or CLIP")
    
    parser.add_argument("--seed", type=int, default=234, help="Seed, default: 1")
    parser.add_argument("--if_clip", type=bool, default=False, help="")
    parser.add_argument("--model_type", type=str, default="CQL", help="")
    parser.add_argument("--load_batches", type=int, default=0, help="")
    parser.add_argument("--align_weight", type=float, default=0.2, help="")
    

    parser.add_argument("--with_lagrange", type=bool, default=False, help="")
    parser.add_argument("--LSTM", type=bool, default=True, help="")
    parser.add_argument("--language_encoder", type=str, default="CLIP", help="LSTM or CLIP")
    parser.add_argument("--expert", type=bool, default=False, help="")
    parser.add_argument("--use_ht", type=bool, default=False, help="use lstm to substitute the original")
    
    parser.add_argument("--meta_type", type=str, default="csro", help="meta RL type, default: focal")
    args = parser.parse_args()
    return args

def collate_fn(batch):
    removed_batch = []
    for i in range(len(batch)):
        if batch[i][2][-1][0] > 0:
            removed_batch.append(batch[i])
    batch = removed_batch
    
    lengths = [len(data[0]) for data in batch]
    
    max_length = max(lengths)
    
    padded_states = torch.zeros(len(batch), max_length, *(batch[0][0].shape[1:]), device=device)
    padded_action = torch.zeros(len(batch), max_length, batch[0][1].shape[1], device=device)
    padded_rewards = torch.zeros(len(batch), max_length, 1, device=device)
    padded_dones = torch.zeros(len(batch), max_length, 1, device=device)
    goals = torch.zeros(len(batch), batch[0][5].shape[0], device=device)
    padded_mask = torch.zeros(len(batch), max_length, device=device)
    
    for i, data in enumerate(batch):
        length = data[4].shape[0]
        
        padded_states[i, :length, :] = data[0].to(device)
        padded_action[i, :length, :] = data[1].to(device)
        padded_rewards[i, :length, :] = data[2].to(device)
        padded_dones[i, :length, :] = data[4].to(device)
        padded_mask[i, :length] = data[6].to(device)
        
        goals[i, :] = data[5].to(device)
    
    return padded_states, padded_action, padded_rewards, padded_dones, goals, padded_mask

def get_clip_tensor(text):
    with torch.no_grad():
        goals_ids = clip.tokenize([text]).to(device)
        goals_ids_ = "_".join([str(int(i)) for i in goals_ids[0]])
        if goals_ids_ not in goal_tensor_record:
            with torch.no_grad():
                goal_tensor = clip_encoder.encode_text(goals_ids.to(torch.int).to(device)).to(torch.float32)
            goal_tensor_record[goals_ids_] = goal_tensor.detach()
        return goal_tensor_record[goals_ids_]

def evaluate(config, policy, eval_runs=4, ood_test=True): 
    """
    Makes an evaluation run with the current policy
    """
    env = SynthLoc()
    with open('data/in_missions.pk', 'rb') as f:
        in_missions = pickle.load(f)

    rewards_task = {'GoTo':[], 'Pickup':[], 'Open':[], 'PutNext':[], 'SynthLoc':[]}
    win_ct = {'GoTo':0, 'Pickup':0, 'Open':0, 'PutNext':0, 'SynthLoc':0}
    test_ct = {'GoTo':0, 'Pickup':0, 'Open':0, 'PutNext':0, 'SynthLoc':0}
    
    eval_counts = 0
    while eval_counts < eval_runs:
        state, _ = env.reset()
        policy.reset()
        if (env.mission in in_missions) and (not ood_test):
            eval_counts += 1
        elif (env.mission not in in_missions) and ood_test:
            eval_counts += 1
        else:
            continue
        # eval_counts += 1
        
        with torch.no_grad():
            goals_ids = clip.tokenize([env.mission])

        rewards = 0
        if config.use_ht:
            h_0 = torch.zeros(2, 1, 7 * config.feature_size).to(device)
            c_0 = torch.zeros(2, 1, 7 * config.feature_size).to(device)
        else:
            h_0 = torch.zeros(2, 1, config.feature_size).to(device)
            c_0 = torch.zeros(2, 1, config.feature_size).to(device)
        
        ht = (h_0, c_0)
        out = None
        for j in range(300):
            action, out, ht = policy.get_action(state, goals_ids, ht, out)
            state, reward, done, _, _ = env.step(action)
            rewards += reward
            
            if done:
                break
        # ---------------- Log the results ----------------
        if 'go' in env.mission:
            key = 'GoTo'
        elif 'pick' in env.mission:
            key = 'Pickup'
        elif 'open' in env.mission:
            key = 'Open'
        elif 'put' in env.mission:
            key = 'PutNext'

        test_ct[key] += 1
        test_ct['SynthLoc'] += 1
        rewards_task[key].append(rewards)
        rewards_task['SynthLoc'].append(rewards)
        if rewards > 0:
            win_ct[key] += 1
            win_ct['SynthLoc'] += 1

    win_rate = {}
    for key in win_ct.keys():
        if test_ct[key] != 0:
            win_rate[key] = win_ct[key] / test_ct[key]
        else:
            win_rate[key] = -0.01
    
    return rewards_task, win_rate

def eval(config, agent, batches, writer):
    eval_metrics = {}
    for ood in [0, 1]:
        rewards_task, win_rate = evaluate(config, agent, eval_runs=100, ood_test=ood)
        
        posfixs = ['ID', 'OOD']
        for key in rewards_task.keys():
            eval_metrics[f"Eval {posfixs[ood]}/{key} rewards"] = np.mean(rewards_task[key])
            eval_metrics[f"Eval {posfixs[ood]}/{key} win rate"] = win_rate[key]
        
        # for key in eval_metrics.keys():
        #     writer.add_scalar(key, eval_metrics[key], batches)
        wandb.log(eval_metrics, step=batches)

def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    task_config = {}
    env = PickupDist(task_config)
    
    average10 = deque(maxlen=10)
    batches = 0
    batches_clip = 0
    
    model_code = f'babyai_{config.model_type}_{config.alpha}_{config.if_clip}_{config.seed}'
    
    if config.model_type == 'CQL' or config.model_type == 'CQL fixed':
        Agent = CQLAgentNaiveLSTM
    elif config.model_type == 'C51 CQL' or config.model_type == 'C51 CQL fixed' or config.model_type == 'C51 CQL Focal':
        Agent = CQLAgentC51LSTM
    elif 'Meta' in config.model_type:
        Agent = CQLAgentNaiveLSTMMeta
    
    agent = Agent(env=env,
        action_size=env.action_space.n,
        hidden_size=config.feature_size,
        device=device,
        config=config
    )
    
    if config.load_batches != 0:
        batches = agent.load_model(f'Put file to downloaded model file here')

    evaluate(config, agent, 10,  0)
    data_path = "data/trajs/Synth_mixed_Bot50000_IL50000_Random25000"
    env = SynthLoc()
    dataset = DatasetOri(config, data_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # writer = SummaryWriter(f'data/babyai/{model_code}')
    wandb.init(project="DAIL", name=model_code, config=config)
    
    if config.log_video:
        env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)

    for i in range(1, config.episodes+1):
                          
        batches_critic = batches
        for batch_idx, experience in enumerate(dataloader):
            states, actions, rewards, dones, goals, masks = experience
            loss_dict, q = agent.learn_step((states, actions, rewards, dones, goals, masks))
            if loss_dict == None:
                continue
            
            for key in loss_dict.keys():
                # writer.add_scalar(f"loss_dict/{key}", loss_dict[key], global_step=batches_critic)
                wandb.log({f"loss_dict/{key}": loss_dict[key]}, step=batches_critic)
                
            batches_critic += 1
            
            if batches_critic % config.eval_every == 0:
                # eval(config, agent, batches_critic, writer)
                eval(config, agent, batches_critic, None)
                
                agent.save_model(f'data/models/{model_code}.pt', batches_critic)
            
            batches = batches_critic

pre_set = {
    0: ['cuda:2', 'CQL', False, 2],
    1: ['cuda:0', 'CQL', True, 2],
    2: ['cuda:1', 'C51 CQL', False, 2],
    3: ['cuda:0', 'C51 CQL Focal', True, 2],
    4: ['cuda:1', 'Meta_Focal', False, 2],
    5: ['cuda:5', 'Meta_csro', False, 2],
    6: ['cuda:3', 'Meta_unicorn', False, 2],
}
# Remember to change the device and meta_type in config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    sets = pre_set[5]

    device_str = sets[0]
    config = get_config()
    config.device = sets[0]
    config.model_type = sets[1]
    config.if_clip = sets[2]
    config.alpha = sets[3]
    config.n_atoms = 51
    config.clip_type = 'CLIP'

    if 'fix' in config.model_type:
        config.fix = True
    else:
        config.fix = False

    config.seed = 24
    set_seed(config.seed)
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    
    goal_tensor_record = {}
    clip_encoder,_ = clip.load("RN50",device=device_str)

    train(config)