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
from autoencoder import Encoder

class Projection(nn.Module):
    def __init__(self, n_out=6):
        super().__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(800, 64)
        self.fc2 = nn.Linear(64, n_out)

    def forward(self, state):
        '''compute cumulative return for each trajectory and return logits'''
        x = self.flat(state)
        x = torch.tanh(self.fc1(x))
        r = torch.tanh(self.fc2(x))

        return r


def checkpoints_folder(game="pong"):
    return f"{game}_ppo_checkpoints"

# calls functions in traj_from_ppo to generate demonstrations and collect feedback
def generate_data(num_per_checkpoint, game, time, rate):
    actions = []
    trajectories = []
    feedbacks = []
    processed = []
    folder = checkpoints_folder(game)
    env, agent = gen_env_agent(name=game, game_type="atari")
    for i, filename in enumerate(os.listdir(folder)):
        print(filename)
        if i%1!=0: # increase mod to speed up testing
            continue
        agent.load(os.path.join(folder, filename))
        for i in range(num_per_checkpoint):
            t, p, a, f, r = feedback_from_trajectory(env, agent, mode='uniform', env_name=game, global_elapsed=time, framerate=rate)
            actions.append(np.array(a))
            trajectories.append(np.array(t))
            processed.append(np.array(p))
            feedbacks.append(f)
        
    env.reset()
    env.viewer=None
    env.render()
    env.close()

    try:
        del env
    except KeyError:
        pass

    return trajectories, processed, feedbacks, actions

# trains state-action pair network using feedback and states in demonstrations
# offline learning!
def train_model(game="pong", epochs=5000, provide_trajs=True, rounds_per_checkpoint=1, batch_size=32, lr=5e-6, alpha=0.9, beta=0.975, time=16, rate=20, label="default"):

    trajs=None
    feeds=None
    procs=None
    acts=None

    if provide_trajs:
        trajs, procs, feeds, acts = generate_data(rounds_per_checkpoint, game, time, rate)
        with open(f'./saved_trajectories/{game}/{game}_{time}_{rate}_{label}_trajectories', "wb") as fp:
            pickle.dump(trajs, fp)
        with open(f'./saved_trajectories/{game}/{game}_{time}_{rate}_{label}_processed', "wb") as fp:
            pickle.dump(procs, fp)
        with open(f'./saved_trajectories/{game}/{game}_{time}_{rate}_{label}_feedback', "wb") as fp:
            pickle.dump(feeds, fp)
        with open(f'./saved_trajectories/{game}/{game}_{time}_{rate}_{label}_actions', "wb") as fp:
            pickle.dump(acts, fp)
    else:
        if f'{game}_{time}_{rate}_trajectories' not in os.listdir(f'./saved_trajectories/{game}'):
            print("provide appropriate traj and feedback dir")
            return
        with open(f'./saved_trajectories/{game}/{game}_{time}_{rate}_{label}_trajectories', "rb") as fp:
            trajs = pickle.load(fp)
        with open(f'./saved_trajectories/{game}/{game}_{time}_{rate}_{label}_processed', "rb") as fp:
            procs = pickle.load(fp)
        with open(f'./saved_trajectories/{game}/{game}_{time}_{rate}_{label}_feedback', "rb") as fp:
            feeds = pickle.load(fp)
        with open(f'./saved_trajectories/{game}/{game}_{time}_{rate}_{label}_actions', "rb") as fp:
            acts = pickle.load(fp)

    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'cuda available: {torch.cuda.is_available()}')

    # direct mapping of s --> h
    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load(f'./ae/{game}_encoder_mse'))
    shrex = Projection(1).to(device)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(shrex.parameters(), lr = lr, weight_decay = 0.0)

    print("=== STARTING TRAINING ===")

    temp = []

    for traj in trajs:
        temp.append(torch.Tensor(traj).to(device))
    
    trajs = temp
    
    for q in range(epochs):
        if q % 100 == 0:
            print(f'== EPOCH {q} ==')
        losses = []
        for trajectory, feedback in zip(trajs, feeds):

            feed,k=feedback.dropout(alpha, beta)
            h = torch.Tensor(feed).unsqueeze(1).to(device)
            traj=trajectory[k[:,0]]

            choice = np.random.choice(traj.shape[0], batch_size, replace=True)
            traj = traj[choice,:,:,:]
            h = h[choice,:]

            h_hat = shrex(encoder(traj))
            optimizer.zero_grad()
            loss = loss_function(h_hat, h)
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())

        if len(losses)>0 and q % 20 == 0:
            print(sum(losses)/len(losses))
    
    print("=== WRITE TO FILE ===")

    torch.save(shrex.state_dict(), f'./models/{game}/atari_{game}_{alpha}_{beta}_{batch_size}_{lr}_{epochs_per_checkpoint}_save')

def test():
    env, agent = gen_env_agent(game, "atari", model="./pong_ppo_checkpoints/01450")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'cuda available: {torch.cuda.is_available()}')

    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load(f'./ae/pong_encoder_mse'))
    shrex = Projection(1).to(device)
    shrex.load_state_dict(torch.load(f'./models/pong/atari_pong_0.9_0.975_32_5e-08_1_save'))
    done = False
    trajectory=[env.reset()]
    processed = []
    reward=[0]
    action=[]
    while not done:
        done = step_env(env, agent, trajectory, processed, action, reward, mask=False)
        print(shrex(encoder(torch.Tensor(np.array(trajectory[-1])).unsqueeze(0).to(device))).cpu().detach().numpy()[0])
        time.sleep(1/20.0)

def run(mode="test", game="pong"):
    if mode=="test":
        test()
    else:
        if game=='pong':
            train_model(game, epochs=1000, provide_trajs=True, rounds_per_checkpoint=3, lr=5e-8, alpha=0.9, beta=0.85, time=16, rate=20, label='testing')

if __name__=="__main__":
    game="pong"
    epochs_per=1
    # train_model(game, 1)
    run("train", "pong")
    # print(SHREXNet(1))