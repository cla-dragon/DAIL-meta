import gymnasium as gym
import numpy as np
import torch
import wandb
import glob
import json
import pickle
import argparse
import os
import csv
import torch.multiprocessing as mp
import torch.distributed as dist

from collections import deque
from torch.utils.data import DataLoader
from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir) 

from babyai.new_missions import *
from utils.data_loader import SlicedDatasetSeq, DatasetSeqClip, collate_fn_clip, MySampler
from utils.utils import find_max_run_number, init_process, cleanup
import networks.CLIP.clip.clip as clip
from agents.agent_bcz import BCZAgent

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="Synth-new", help="Run name description")
    parser.add_argument("--model_saved_path", type=str, default="saved_models", help="model saved path")
    parser.add_argument("--pixel_input", type=bool, default=False, help="Use pixel input")
    parser.add_argument("--data_path", type=str, default="BabyAI/data/data_Synth_medium_Bot12500_IL25000_Random40000_clip.pkl", help="Path to data")
    parser.add_argument("--load_model", type=bool, default=False, help="")
    parser.add_argument("--load_model_path", type=str, default="", help="Path to model")

    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=1, help="Seed")
    parser.add_argument("--device", type=int, default=0, help="GPU device num")
    parser.add_argument("--batch_size", type=int, default=32, help="default: 256")
    parser.add_argument("--log_video", type=bool, default=False, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--eval_every", type=int, default=1000, help="")
    parser.add_argument("--all_env", type=bool, default=False, help="")
    parser.add_argument("--all_data", type=bool, default=True, help="")
    parser.add_argument("--ood_test", type=bool, default=True, help="")
    
    parser.add_argument("--save_model", type=bool, default=True, help="")
    parser.add_argument("--save_every", type=int, default=50, help="Saves the network every x epochs, default: 25")
    
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="")

    parser.add_argument("--state_emb_size", type=int, default=256, help="")
    parser.add_argument("--lang_emb_size", type=int, default=256, help="")
    parser.add_argument("--video_emb_size", type=int, default=256, help="")


    parser.add_argument("--use_film", type=bool, default=True, help="")
    parser.add_argument("--distributed", type=bool, default=False, help="")
        
    args = parser.parse_args()
    return args

def prep_dataset(config, batch_size=256):
    ood_path = config.data_path
    if config.distributed:
        dataset = SlicedDatasetSeq(ood_path, shuffle=True)
    else:
        dataset = DatasetSeqClip(ood_path)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # task_config = {}
    env = SynthLoc()
    # env = RGBImgPartialObsWrapper(env, tile_size=8)
    
    return dataset, env

def evaluate(config, policy:BCZAgent, eval_runs=100, ood_test=True): 
    """
    Makes an evaluation run with the current policy
    """
    policy.network.eval()
    env = SynthLoc()

    with open('BabyAI/data/in_missions.pk', 'rb') as f:
        in_missions = pickle.load(f)

    rewards_task = {'GoTo':[], 'Pickup':[], 'Open':[], 'PutNext':[], 'SynthLoc':[]}
    win_ct = {'GoTo':0, 'Pickup':0, 'Open':0, 'PutNext':0, 'SynthLoc':0}
    test_ct = {'GoTo':0, 'Pickup':0, 'Open':0, 'PutNext':0, 'SynthLoc':0}
    
    eval_counts = 0
    while eval_counts < eval_runs:
            
        if config.pixel_input:
            env = RGBImgPartialObsWrapper(env, tile_size=8)

        state, _ = env.reset()
        if (env.mission in in_missions) and (not ood_test):
            eval_counts += 1
        elif (env.mission not in in_missions) and ood_test:
            eval_counts += 1
        else:
            continue
        
        with torch.no_grad():
            goals_ids = clip.tokenize([env.mission])

        rewards = 0
        hidden_state = None
        cell_state = None
        for j in range(300):
            action, hidden_state, cell_state = policy.get_action(state, goals_ids, 
                                                                hidden_state=hidden_state,
                                                                cell_state=cell_state)
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
            win_rate[key] = -0.1
    

    return rewards_task, win_rate

def train(local_rank, world_size, config, run_num, dataset, metric_queue=None):
    if config.distributed:
        init_process(local_rank, world_size)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.distributed:
        device = torch.device("cuda:{}".format(local_rank))
    else:
        device = torch.device(f"cuda:{config.device}")
    
    if not config.distributed:
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_clip)
    else:
        # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        sampler = MySampler(dataset, seed=config.seed)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, collate_fn=collate_fn_clip, 
                                shuffle=False, num_workers=0)

    # average10 = deque(maxlen=10)
    batches = 0
    medium_postfix = 'medium' if 'medium' in config.data_path else 'expert'
    if local_rank == 0:
        wandb.init(project="Babyai-BCZ",name=f'run-{run_num}-{config.run_name}-{medium_postfix}-seed{config.seed}' , config=config)
        # writer = SummaryWriter()
        
    agent = BCZAgent(config, device)
    if config.load_model:
        agent.network.load_state_dict(torch.load(config.load_model_path))

    # wandb.watch(agent.network, log="gradients", log_freq=10)

    if config.log_video:
        env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)
    
    eval_metrics_list = []
    for i in range(1, config.episodes+1):
        if config.distributed:
            sampler.set_epoch(i-1)
        for batch_idx, experience in enumerate(dataloader):
            metrics = agent.learn(experience, train=True)
            if config.distributed:
                dist.barrier()
            batches += 1
            metrics["Batches"] = batches
            metrics["Episode"]  = i

            if (batches % config.eval_every == 0) and (local_rank == 0 or local_rank == 1):
                eval_metrics = {}
                agent.network.eval()
                posfixs = ['ID', 'OOD']
                if config.distributed:
                    ood_test = (local_rank == 1)
                    rewards_task, win_rate= evaluate(config, agent, eval_runs=100, ood_test=ood_test)
                    
                    for key in rewards_task.keys():
                        eval_metrics[f"Eval-{posfixs[local_rank]}/{key} rewards"] = np.mean(rewards_task[key])
                        eval_metrics[f"Eval-{posfixs[local_rank]}/{key} win rate"] = win_rate[key]
                else:
                    for posfix in posfixs:
                        rewards_task, win_rate= evaluate(config, agent, eval_runs=100, ood_test=(posfix == 'OOD'))
                        
                        for key in rewards_task.keys():
                            eval_metrics[f"Eval-{posfix}/{key} rewards"] = np.mean(rewards_task[key])
                            eval_metrics[f"Eval-{posfix}/{key} win rate"] = win_rate[key]

                if local_rank == 1:
                    metric_queue.put(eval_metrics)
                elif local_rank == 0 and config.distributed:
                    ood_metrics = metric_queue.get()
                    for key, value in ood_metrics.items():
                        eval_metrics[key] = value
                if config.distributed:
                    print("Episode: {} | {}-Reward: {} | Win Rate: {} | Loss: {} | Batches: {}"
                            .format(i, posfixs[local_rank], eval_metrics[f'Eval-{posfixs[local_rank]}/SynthLoc rewards'], 
                            eval_metrics[f"Eval-{posfixs[local_rank]}/SynthLoc win rate"], metrics['total_loss'], batches))
                else:
                    for posfix in posfixs:
                        print("Episode: {} | {}-Reward: {} | Win Rate: {} | Loss: {} | Batches: {}"
                            .format(i, posfix,  eval_metrics[f'Eval-{posfix}/SynthLoc rewards'], 
                            eval_metrics[f"Eval-{posfix}/SynthLoc win rate"], metrics['total_loss'], batches))
                        
                eval_metrics_list.append(eval_metrics)
                
                with open(f"log_results/BCZ_new_{medium_postfix}_seed_{config.seed}.csv", mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=eval_metrics.keys())
                    writer.writeheader() 
                    writer.writerows(eval_metrics_list)  

                for key in eval_metrics.keys():
                        metrics[key] = eval_metrics[key]

            if local_rank == 0:
                wandb.log(metrics)
        
        if (i % config.save_every == 0) and config.save_model and local_rank == 0:
            torch.save(agent.network.state_dict(), f"{config.model_saved_path}/Fin-new-BCZ_{medium_postfix}_seed{config.seed}_ep_{i}.pth")
            config_name = f"{config.model_saved_path}/Fin-new-BCZ_config_{medium_postfix}_seed{config.seed}.json"
            with open(config_name, 'w') as f:
                json.dump(vars(config), f, indent=4)

    if local_rank == 0:
        wandb.finish()
        # writer.close()
    if config.distributed:
        cleanup()

if __name__ == "__main__":
    config = get_config()
    if not os.path.exists(config.model_saved_path):
        os.makedirs(config.model_saved_path)
    prefix = 'Fin-BCZ_run_'
    run_num = find_max_run_number(config.model_saved_path, prefix) + 1

    # Prepare the dataset
    dataset, env = prep_dataset(config, config.batch_size)
    print("Dataset prepared!")
    if not config.distributed:
        train(0, 1, config, run_num, dataset)

    else:
        mp.set_start_method("spawn", force=True)
        metric_queue = mp.Queue()
        gpu_count = torch.cuda.device_count()
        # gpu_count = 2
        try:
            # mp.set_sharing_strategy('file_system')
            mp.spawn(train, args=(gpu_count, config, run_num, dataset, metric_queue), nprocs=gpu_count)
        except Exception as e:
            print(e)

            wandb.finish()