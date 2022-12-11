import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from traj_from_ppo import gen_env_agent, feedback_from_trajectory, step_env
import sys
sys.path.insert(0, '../ICML2019-TREX-SparseFeedback/atari')
sys.path.insert(0, '../ICML2019-TREX-SparseFeedback/atari/baselines')
from baselines.common.trex_utils import preprocess
import pickle
import time
from matplotlib import pyplot as plt

class SHREXNet(nn.Module):
    def __init__(self, n_out=4):
        super().__init__()

        #ENCODER
        self.conv1 = nn.Conv2d(4, 16, 7, stride=3) # 7 3
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1) # 5 2
        self.pool2 = nn.MaxPool2d(3)
        # self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        # self.conv4 = nn.Conv2d(16, 16, 3, stride=1)

        #HEAD
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, n_out)

    def forward(self, state):
        '''compute cumulative return for each trajectory and return logits'''
        x = torch.Tensor(state).permute(0,3,1,2)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        # x = F.leaky_relu(self.conv3(x))
        # x = F.leaky_relu(self.conv4(x))
        x = x.reshape(-1, 64)
        x = F.tanh(self.fc1(x))
        r = F.tanh(self.fc2(x))

        return r

def checkpoints_folder(game="pong"):
    return f"{game}_ppo_checkpoints"

# calls functions in traj_from_ppo to generate demonstrations and collect feedback
def generate_data(num_per_checkpoint, game):
    actions = []
    trajectories = []
    feedbacks = []
    folder = checkpoints_folder(game)
    env, agent = gen_env_agent(name=game, game_type="atari")
    for i, filename in enumerate(os.listdir(folder)):
        if i%2!=0: # increase mod to speed up testing
            continue
        agent.load(os.path.join(folder, filename))
        for i in range(num_per_checkpoint):
            t, a, f, r = feedback_from_trajectory(env, agent, global_elapsed=10, framerate=20)
            actions.append(a)
            trajectories.append(t)
            feedbacks.append(f)
        
    env.reset()
    env.viewer=None
    env.render()
    env.close()

    try:
        del env
    except KeyError:
        pass

    return trajectories, feedbacks

# trains state-action pair network using feedback and states in demonstrations
# offline learning!
def train_model(game="pong", epochs_per_checkpoint=1, alpha=0.1):

    trajs, feeds = generate_data(epochs_per_checkpoint, game)

    # direct mapping of s --> h
    nn = SHREXNet(1)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(nn.parameters(), lr = 5e-6, weight_decay = 0.0)

    print("=== STARTING TRAINING ===")
    
    for q in range(200):
        print(f'== EPOCH {q} ==')
        for traj, feed in zip(trajs, feeds):
            # traj, feed = dropout(traj, feed, alpha)
            losses = []
            f=feed.dropout(0.6)
            for i, o in enumerate(traj):
                h = f[i]
                if h == 0: 
                    continue
                h_hat = nn(o)
                optimizer.zero_grad()
                loss = loss_function(h_hat, torch.Tensor(h))
                loss.backward()
                optimizer.step()
                losses.append(loss)
            if len(losses)>0 and q % 20 == 0:
                print(losses[-1])
    
    print("=== WRITE TO FILE ===")

    torch.save(nn.state_dict(), f'./models/atari_{game}_{alpha}_{epochs_per_checkpoint}_save')

# def test_model(game, filename):
#     env, agent = gen_env_agent(game, "atari")
#     done=False
#     trajectory=[env.reset()]
#     reward=[0]
#     while not done:
#         done = step_env(env, agent, )
#         time.sleep(0.0333)

def test():
    env, agent = gen_env_agent(game, "atari", model="./pong_ppo_checkpoints/01450")
    model = SHREXNet(1)
    model.load_state_dict(torch.load("./models/atari_pong_0.1_1_save"))
    done = False
    trajectory=[env.reset()]
    reward=[0]
    action=[]
    while not done:
        done = step_env(env, agent, trajectory, action, reward, mask=False)
        print(model(trajectory[-1]).detach().numpy()[0])
        time.sleep(1/3.0)

def run(mode="test", game="pong", epochs=1, f='./models/atari_pong_0.1_1_save'):
    if mode=="test":
        # test_model(game, f)
        test()
    else:
        train_model(game, epochs)

if __name__=="__main__":
    game="pong"
    epochs_per=1
    # train_model(game, 1)
    run("test", "pong", 1, './models/atari_pong_0.1_1_save')
    # print(SHREXNet(1))