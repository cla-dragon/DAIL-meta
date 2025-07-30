

from collections import deque
import torch
import argparse
import os

import torch.multiprocessing as mp

from agents.agent_dail import CQLAgentNaiveLSTM, CQLAgentC51LSTM
from agents.agent_meta import CQLAgentNaiveLSTMMeta

from babyai.new_missions import *

import networks.CLIP.clip.clip as clip

import pickle

from PIL import Image

import copy

os.environ['WANDB_MODE'] = 'offline'

device_str = "cuda:6"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL-DQN", help="Run name, default: CQL-DQN")
    parser.add_argument("--env", type=str, default="CartPole-v0", help="Gym environment name, default: CartPole-v0")
    parser.add_argument("--episodes", type=int, default=400, help="Number of episodes, default: 200")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--batch_size", type=int, default=8, help="default: 256")
    parser.add_argument("--batch_size_clip", type=int, default=128, help="default: 256")
    # parser.add_argument("--batch_size", type=int, default=32, help="default: 256")
    parser.add_argument("--min_eps", type=float, default=0.01, help="Minimal Epsilon, default: 4")
    parser.add_argument("--eps_frames", type=int, default=1e4, help="Number of steps for annealing the epsilon value to the min epsilon, default: 1e5")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--eval_every", type=int, default=20000, help="")
    parser.add_argument("--all_data", type=bool, default=True, help="")
    parser.add_argument("--ood_test", type=bool, default=True, help="")
    
    parser.add_argument("--train_goal_q", type=bool, default=True, help="")
    parser.add_argument("--train_goal_clip", type=bool, default=True, help="")
    parser.add_argument("--train_state_q", type=bool, default=True, help="")
    parser.add_argument("--train_state_clip", type=bool, default=True, help="")
    
    parser.add_argument("--alpha", type=float, default=0.2, help="")
    parser.add_argument("--gamma", type=float, default=0.99, help="")
    parser.add_argument("--q_learning_rate", type=float, default=3e-4, help="")
    # parser.add_argument("--q_learning_rate", type=float, default=5e-5, help="")
    parser.add_argument("--clip_learning_rate", type=float, default=3e-5, help="")
    
    parser.add_argument("--feature_size", type=int, default=256, help="")
    parser.add_argument("--feature_extract", type=str, default='resnet', help="")
    parser.add_argument("--goal_format", type=str, default='multi-hot', help="")
    parser.add_argument("--n_atoms", type=int, default=51, help="")
    parser.add_argument("--history_frame", type=int, default=1, help="")
    parser.add_argument("--target_gap", type=int, default=2, help="")
    # parser.add_argument("--update_frequency", type=int, default=1000, help="")
    parser.add_argument("--update_frequency", type=int, default=1000, help="")
    parser.add_argument("--device", type=str, default=device_str, help="LSTM or CLIP")
    
    parser.add_argument("--if_clip", type=bool, default=False, help="")
    # parser.add_argument("--if_clip", type=bool, default=True, help="")
    parser.add_argument("--model_type", type=str, default="C51 CQL", help="")
    # parser.add_argument("--model_type", type=str, default="CQL", help="")
    parser.add_argument("--LSTM", type=bool, default=True, help="")
    parser.add_argument("--language_encoder", type=str, default="CLIP", help="LSTM or CLIP")
    parser.add_argument("--expert", type=bool, default=False, help="")
    parser.add_argument("--with_lagrange", type=bool, default=False, help="")
    parser.add_argument("--use_ht", type=bool, default=False, help="use lstm to substitute the original")
    
    
    args = parser.parse_args()
    return args

config = get_config()
def collate_fn(batch):
    removed_batch = []
    list_in = [False] * len(batch)
    
    for i in range(len(batch)):
        if list_in[i] == True:
            continue
        removed_batch.append(batch[i])
        for j in range(i+1, len(batch)):
            goal_i = batch[i][5]
            goal_j = batch[j][5]
            if (goal_i - goal_j).sum() == 0:
                list_in[j] = True
    
    batch = removed_batch
    
    lengths = [len(data[0]) for data in batch]
    max_length = max(lengths)
    
    padded_states = torch.zeros(len(batch), max_length, *(batch[0][0].shape[1:]), device=device)
    padded_action = torch.zeros(len(batch), max_length, batch[0][1].shape[1], device=device)
    padded_rewards = torch.zeros(len(batch), max_length, 1, device=device)
    padded_dones = torch.zeros(len(batch), max_length, 1, device=device)
    goals = torch.zeros(len(batch), batch[0][5].shape[0], device=device)
    padded_mask = torch.zeros(len(batch), max_length, device=device)
    
    if config.language_encoder == "LSTM":
        goals = []
        for i in range(len(batch)):
            goal = []
            for j in batch[i][5][0]:
                if j != 0:
                    goal.append(int(j))
            goals.append(goal)
        
    for i, data in enumerate(batch):
        length = data[4].shape[0]
        
        padded_states[i, :length, :] = data[0].to(device)
        padded_action[i, :length, :] = data[1].to(device)
        padded_rewards[i, :length, :] = data[2].to(device)
        padded_dones[i, :length, :] = data[4].to(device)
        padded_mask[i, :length] = data[6].to(device)
        
        if config.language_encoder != "LSTM":
            goals[i, :] = data[5].to(device)
    
    return padded_states, padded_action, padded_rewards, padded_dones, goals, padded_mask

def clip_ori(config, agents, settings, in_missions, ood_test, gen_image, batches, lock, task_queue):
    while True:
        if task_queue.qsize() == 0:
            break
        
        record_type = {}
        env_ori = SynthLoc(render_mode='rgb_array')
        state, _ = env_ori.reset()
        
        if (env_ori.mission in in_missions) and (not ood_test):
            pass
        elif (env_ori.mission not in in_missions) and ood_test:
            pass
        else:
            continue
        
        eval_ct = task_queue.get()
        
        print(eval_ct)
        
        agent = agents[0]
        s = settings[0]
        record = {
            "traj": [],
            "type": None,
            "goal": None,
            "goal emb": None,
            "result": None,
        }
        env = copy.deepcopy(env_ori)
        config.model_type = s[0]
        
        agent.reset()
        
        # eval_counts += 1
        
        with torch.no_grad():
            goals_ids = clip.tokenize([env.mission])
        
        record['goal'] = env.mission
        record['goal emb'] = agent.clip.encode_text(goals_ids.to(torch.int).to(device)).detach().cpu()
        
        # ---------------- Log the results ----------------
        if 'go' in env.mission:
            key = 'GoTo'
        elif 'pick' in env.mission:
            key = 'Pickup'
        elif 'open' in env.mission:
            key = 'Open'
        elif 'put' in env.mission:
            key = 'PutNext'
        record['type'] = key
        
        posfixs = ['ID', 'OOD']
        
        record_type[f"{s[0]}_{s[2]}"] = record
    
        if len(record_type.keys()) != 0:
            folder = f"results/naive_clip/{posfixs[ood_test]}/{batches}/{eval_ct}_{env.mission}"
            if not os.path.exists(folder):
                os.makedirs(folder)
            with open(f"{folder}/clip_ori.pt", 'wb') as f:
                pickle.dump(record_type, f)

def evaluate_one(config, agents, settings, in_missions, ood_test, gen_image, batches, lock, task_queue, test_ct, success_ct):
    while True:
        if task_queue.qsize() == 0:
            break
        
        env_ori = SynthLoc()
        
        state_init, _ = env_ori.reset()
        
        if (env_ori.mission in in_missions) and (not ood_test):
            pass
        elif (env_ori.mission not in in_missions) and ood_test:
            pass
        else:
            continue
        
        eval_ct = task_queue.get()
        
        agent_id = 0
        for agent, s in zip(agents, settings):
            env = copy.deepcopy(env_ori)
            config.model_type = s[0]
            
            agent.reset()
            
            with torch.no_grad():
                goals_ids = clip.tokenize([env.mission])
            
            rewards = 0

            h_0 = torch.zeros(2, 1, config.feature_size).to(device)
            c_0 = torch.zeros(2, 1, config.feature_size).to(device)
            
            ht = (h_0, c_0)
            out = None
            
            
            images = []
            state = copy.deepcopy(state_init)
            for j in range(300):
                if gen_image:
                    image = env.render()
                    image = Image.fromarray(image)
                    images.append(image)
                    
                action, out, ht = agent.get_action(state, goals_ids, ht, out)
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
            
            posfixs = ['ID', 'OOD']
            
            success_flag = "Succeed" if reward > 0 else "Failed"
            if gen_image:
                folder = f"results/threads/{posfixs[ood_test]}/{batches}/{eval_ct}_{env.mission}/{s[0]}_{s[2]}_{success_flag}"
                if not os.path.exists(folder):
                    os.makedirs(folder)
            
                for idx, i in enumerate(images):
                    i.save(f"{folder}/{idx}.jpg")

            lock.acquire()
            test_ct_tp = test_ct[posfixs[ood_test]]
            test_ct_tp[key][agent_id] += 1
            test_ct_tp["All"][agent_id] += 1
            test_ct[posfixs[ood_test]] = test_ct_tp
            if rewards > 0:
                success_ct_tp = success_ct[posfixs[ood_test]]
                success_ct_tp[key][agent_id] += 1
                success_ct_tp["All"][agent_id] += 1
                success_ct[posfixs[ood_test]]=success_ct_tp
            
            # print(f"===== {posfixs[ood_test]} =====")
            # for task in success_ct[posfixs[ood_test]].keys():
            #     res = 0
            #     for i in range(3):
            #         if test_ct[posfixs[ood_test]][task][i] > 0:
            #             res += float(success_ct[posfixs[ood_test]][task][i])/test_ct[posfixs[ood_test]][task][i]
            #     res /= 3
            #     print(f"    Task {task}: {res}")
            lock.release()

            agent_id += 1
        
        # lock.acquire()
        # with lock:
            # if len(record_type.keys()) != 0:
            #     folder = f"results/threads/{posfixs[ood_test]}/{batches}/{eval_ct}_{env.mission}"
            #     with open(f"{folder}/records.pt", 'wb') as f:
            #         pickle.dump(record_type, f)
        # lock.release()

def evaluate(config, agents, settings, batches, manager, test_ct, success_ct, eval_runs=4, ood_test=True, gen_image=True): 
    """
    Makes an evaluation run with the current policy
    """
    
    with open('data/in_missions.pk', 'rb') as f:
        in_missions = pickle.load(f)

    task_queue = manager.Queue()
    for i in range(eval_runs):
        task_queue.put(i)
    
    threads = []
    lock = manager.Lock()
    
    for n in range(10):
        with torch.no_grad():
            thread = mp.Process(
                target=evaluate_one,
                args=(config, agents, settings, in_missions, ood_test, gen_image, batches, lock, task_queue, test_ct, success_ct)
            )
    
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()
    
def eval(config, agents, settings, batches, manager, gen_image):
    for a, s in zip(agents, settings):
        a.load_model(f'{s[-1]}')
    
    test_ct = manager.dict()
    test_ct["ID"] = {
            "Open": [0, 0, 0],
            "GoTo": [0, 0, 0],
            "Pickup": [0, 0, 0],
            "PutNext": [0, 0, 0],
            "All": [0, 0, 0]
        }
    test_ct["OOD"] = {
            "Open": [0, 0, 0],
            "GoTo": [0, 0, 0],
            "Pickup": [0, 0, 0],
            "PutNext": [0, 0, 0],
            "All": [0, 0, 0]
        }

    success_ct = manager.dict()
    success_ct["ID"] = {
            "Open": [0, 0, 0],
            "GoTo": [0, 0, 0],
            "Pickup": [0, 0, 0],
            "PutNext": [0, 0, 0],
            "All": [0, 0, 0]
        }
    success_ct["OOD"] = {
            "Open": [0, 0, 0],
            "GoTo": [0, 0, 0],
            "Pickup": [0, 0, 0],
            "PutNext": [0, 0, 0],
            "All": [0, 0, 0]
        }

    for ood in [0, 1]:
        evaluate(config, agents, settings, batches, manager, test_ct, success_ct, eval_runs=1000, ood_test=ood, gen_image=gen_image)
    
    for key in success_ct.keys():
        res_ave = []
        print(f"===== {key} =====")
        for task in success_ct[key].keys():
            res = 0
            for i in range(1):
                res += float(success_ct[key][task][i])/test_ct[key][task][i]
            res /= 1
            res_ave.append(res)
            print(f"    Task {task}: {res}")
        print("    Ave all", (res_ave[0]+res_ave[1]+res_ave[2]+res_ave[3])/4)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    
    config = get_config()
    config.clip_type = 'DAIL'
    # task_config = {}
    # env = PickupDist(task_config)
    env = SynthLoc()
    
    average10 = deque(maxlen=10)

    settings = [
        ['CQL', CQLAgentNaiveLSTM, False, 'none', "data/models/babyai_CQL_2_False_1.pt"],
    ]
      
    agents = []
    for idx, s in enumerate(settings):
        config.model_type = s[0]
        config.if_config = s[2]
        config.meta_type = s[3]
        agent = s[1](env=env,
            action_size=env.action_space.n,
            hidden_size=config.feature_size,
            device=device,
            config=copy.deepcopy(config)
        )
        agent.net.share_memory()
        agents.append(agent)
    
    with torch.no_grad():
        eval(config, agents, settings, None, manager, gen_image=False)
    
        
        
