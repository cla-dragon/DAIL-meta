import gymnasium as gym
import numpy as np
import torch
import wandb
import glob
import json
import argparse
import os
import pickle

from collections import deque
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper

from envs.babyai.new_missions import *
from utils_general.data_loader import SlicedDatasetSeq, DatasetOri, collate_fn, MySampler
from utils_general.utils import find_max_run_number, init_process, cleanup
import networks.CLIP.clip.clip as clip
from algorithms.agent_iql import IQLAgent

os.environ['TORCH_DISTRIBUTED_DEBUG']='DETAIL'
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4"

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="Synth-no-h-dis", help="Run name description")
    parser.add_argument("--model_saved_path", type=str, default="saved_models", help="model saved path")
    parser.add_argument("--pixel_input", type=bool, default=False, help="Use pixel input")
    parser.add_argument("--data_path", type=str, default="data/trajs/data_Synth_mixed_Bot50000_IL50000_Random25000.pkl", help="Path to data")
    # parser.add_argument("--valid_data_path", type=str, default="data/trajs/data_7000_state_PickupGoto_valid_without_redballgreenkey.pk", help="Path to data")
    parser.add_argument("--load_model", type=bool, default=False, help="")
    parser.add_argument("--load_model_path", type=str, default="", help="Path to model")

    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=1, help="Seed")
    parser.add_argument("--batch_size", type=int, default=512, help="default: 256")
    parser.add_argument("--log_video", type=bool, default=False, help="Log agent behaviour to wanbd when set to 1, default: 0")
    
    parser.add_argument("--save_model", type=bool, default=True, help="")
    parser.add_argument("--save_every", type=int, default=25, help="Saves the network every x epochs, default: 25")
    
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="")
    # parser.add_argument("--clip_learning_rate", type=float, default=5e-5, help="")

    parser.add_argument("--state_emb_size", type=int, default=512, help="")
    parser.add_argument("--lang_emb_size", type=int, default=512, help="")

    parser.add_argument("--use_film", type=bool, default=True, help="")
    parser.add_argument("--use_mc_help", type=bool, default=True, help="")

    parser.add_argument("--distributed", type=bool, default=False, help="")

    args = parser.parse_args()
    return args

def prep_dataset(config, batch_size=256):
    ood_path = config.data_path
    dataset = DatasetOri(ood_path)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # task_config = {}
    env = SynthLoc()
    # env = RGBImgPartialObsWrapper(env, tile_size=8)
    
    return dataset, env

def evaluate(config, policy:IQLAgent, eval_runs=100, ood_test=True): 
    """
    Makes an evaluation run with the current policy
    """
    policy.network.eval()
    env = SynthLoc()

    with open('data/in_missions.pk', 'rb') as f:
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
        # hidden_state = None
        # cell_state = None
        for j in range(300):
            # action, hidden_state, cell_state = policy.get_action(state, goals_ids, 
            #                                                     hidden_state=hidden_state,
            #                                                     cell_state=cell_state)
            action = policy.get_action(state, goals_ids)
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
        device = torch.device("cuda:7")
    
    if not config.distributed:
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    else:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        # sampler = MySampler(dataset, seed=config.seed)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, collate_fn=collate_fn, 
                                shuffle=False, num_workers=0)

    batches = 0
    batches_policy = 0

    if local_rank == 0:    
        wandb.init(project="Babyai-IQL",name=f'run-{run_num}-{config.run_name}' , config=config)
        
    agent = IQLAgent(config, device)
    if config.load_model:
        agent.network.load_state_dict(torch.load(config.load_model_path))

    # wandb.watch(agent.network, log="gradients", log_freq=10)

    # if config.log_video:
    #     env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)
    
    for i in range(1, config.episodes+1):
        # for batch_idx, experience in enumerate(seq_dataloader):
        #     metrics = agent.clip_learn(experience)
        #     batches_align += 1
        #     metrics["Batches Align"] = batches_align
        #     metrics["Episode"]  = i
        #     wandb.log(metrics)

        for batch_idx, experience in enumerate(dataloader):
            # Test Run
            if batch_idx == 5 and i == 1:
                break

            if (batch_idx == 0) and (i == 1):
                valid_batch = tuple(element[:30] for element in experience)

            metrics = agent.learn(experience, policy_extract=False, train=True)            
            batches += 1

            # --------------------------- Valid Q value tracking ---------------------------
            valid_q_values = agent.get_q_values(valid_batch)
            for j, q in enumerate(valid_q_values):
                metrics[f"Q-Value/{j}"] = q.item()

            metrics["Batches_QV"] = batches
            metrics["Episode"]  = i
            if local_rank == 0:
                wandb.log(metrics)

        for batch_idx, experience in enumerate(dataloader):
            # Test Run
            if batch_idx == 5 and i == 1:
                break

            metrics = agent.learn(experience, policy_extract=True, train=True)
            batches_policy += 1
            metrics["Batches_policy"] = batches_policy
            metrics["Episode"]  = i
            if local_rank == 0:
                wandb.log(metrics)
        
        agent.mc_q_weight = agent.mc_q_weight * 0.8
        agent.scheduler.step()

        # --------------------------- Validation ---------------------------
        # agent.network.eval()
        # for batch_idx, experience in enumerate(val_dataloader):
        #     metrics = agent.learn(experience, policy_extract=False, train=False)
        #     batches_valid += 1
        #     metrics["Batches_valid"] = batches_valid
        #     metrics["Episode"]  = i
        #     log_metrics = {}
        #     for key in metrics.keys():
        #         log_metrics[f"Valid/{key}"] = metrics[key]
            
        #     wandb.log(log_metrics)
        
        # for batch_idx, experience in enumerate(val_dataloader):
        #     metrics = agent.learn(experience, policy_extract=True, train=False)
        #     batches_valid_policy += 1
        #     metrics["Batches_policy"] = batches_valid_policy
        #     metrics["Episode"]  = i
        #     log_metrics = {}
        #     for key in metrics.keys():
        #         log_metrics[f"Valid/{key}"] = metrics[key]
            
        #     wandb.log(log_metrics)

        # --------------------------- Evaluation --------------------------- 
        if (local_rank == 0 or local_rank == 1):
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
                print("Episode: {} | {}-Reward: {} | Win Rate: {} | Actor Loss: {} | Batches: {}"
                        .format(i, posfixs[local_rank], eval_metrics[f'Eval-{posfixs[local_rank]}/SynthLoc rewards'], 
                        eval_metrics[f"Eval-{posfixs[local_rank]}/SynthLoc win rate"], metrics['actor_loss'], batches))
            else:
                for posfix in posfixs:
                    print("Episode: {} | {}-Reward: {} | Win Rate: {} | Actor Loss: {} | Batches: {}"
                        .format(i, posfix,  eval_metrics[f'Eval-{posfix}/SynthLoc rewards'], 
                        eval_metrics[f"Eval-{posfix}/SynthLoc win rate"], metrics['actor_loss'], batches))
        
            if local_rank == 0:
                wandb.log(eval_metrics)

        
        # if (i % 10 == 0) and config.log_video:
        #     mp4list = glob.glob('video/*.mp4')
        #     if len(mp4list) > 1:
        #         mp4 = mp4list[-2]
        #         wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=1, format="gif"), "Episode": i})

        if (i % config.save_every == 0) and config.save_model:
            torch.save(agent.network.state_dict(), f"{config.model_saved_path}/N-IQL_without_hidden_run_{run_num}_{config.run_name}_ep_{i}.pth")
            config_name = f"{config.model_saved_path}/N-IQL_without_hidden_config_run_{run_num}_{config.run_name}.json"
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
    prefix = 'N-IQL_without_hidden_run_'
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