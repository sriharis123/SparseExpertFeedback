import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from traj_from_ppo import gen_env_agent, feedback_from_trajectory
import pickle
import time

class SHREXNet(nn.Module):
    def __init__(self, n_out=4):
        super().__init__()

        #ENCODER
        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)

        #HEAD
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, n_out)

    def forward(self, state):
        '''compute cumulative return for each trajectory and return logits'''
        x = torch.Tensor(state).permute(0,3,1,2)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.reshape(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        r = self.fc2(x)

        return r

def checkpoints_folder(game="pong"):
    return f"{game}_ppo_checkpoints"

# calls functions in traj_from_ppo to generate demonstrations and collect feedback
def generate_data(num_per_checkpoint, game):
    trajectories = []
    feedbacks = []
    folder = checkpoints_folder(game)
    env, agent = gen_env_agent(name=game, game_type="atari")
    for i, filename in enumerate(os.listdir(folder)):
        if i%10!=0:
            continue
        agent.load(os.path.join(folder, filename))
        for i in range(num_per_checkpoint):
            t, f, r = feedback_from_trajectory(env, agent, global_elapsed=10, framerate=20)
            trajectories.append(t)
            feedbacks.append(f)

    env.close()
    time.sleep(3)

    try:
        del env
    except KeyError or ImportError:
        pass

    return trajectories, feedbacks

# keep only some state-action pairs
def dropout(trajectory, feedback, alpha):
    return None

# trains state-action pair network using feedback and states in demonstrations
# offline learning!
def train_model(game="pong", epochs_per_checkpoint=1, alpha=0.1):

    trajs, feeds = generate_data(epochs_per_checkpoint, game)

    # direct mapping of s --> h
    nn = SHREXNet(1)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(nn.parameters(), lr = 1e-1, weight_decay = 1e-8)

    print("=== STARTING TRAINING ===")

    for traj, feed in zip(trajs, feeds):
        # traj, feed = dropout(traj, feed, alpha)
        losses = []
        for i, o in enumerate(traj):
            h = feed[i]
            h_hat = nn(o)
            loss = loss_function(h_hat, torch.Tensor(h))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss)
        print("Loss:", losses[-1])
    
    torch.save(nn.state_dict(), f'./models/atari_{game}_{alpha}_{epochs_per_checkpoint}_save')



if __name__=="__main__":
    game="pong"
    epochs_per=1
    train_model(game, epochs_per)