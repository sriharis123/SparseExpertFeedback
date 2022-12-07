import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from traj_from_ppo import gen_env_agent, feedback_from_trajectory

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
    for filename in os.listdir(checkpoints_folder(game)):
        for i in range(num_per_checkpoint):
            env, agent = gen_env_agent(name=game, game_type="atari", model=os.path.join(folder, filename))
            feedback_from_trajectory(env, agent, global_elapsed=16)
            
    return trajectories, feedbacks


def dropout():


# trains state-action pair network using feedback and states in demonstrations
def train_model(epochs_per_checkpoint=1, alpha=0.1):
    return None


if __name__=="__main__":
    game_type="pong"
    print(SHREXNet(4))