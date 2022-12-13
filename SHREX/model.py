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
        self.fc1 = nn.Linear(1600, 64)
        self.fc2 = nn.Linear(64, n_out)

    def forward(self, state):
        '''compute cumulative return for each trajectory and retur n logits'''
        x = self.flat(state)
        x = F.leaky_relu(self.fc1(x))
        r = self.fc2(x)

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
    env, agent = gen_env_agent(name=game, game_type="atari", model=f'./{game}_ppo_checkpoints/00050')
    for i, filename in enumerate(os.listdir(folder)):
        print(filename)
        if i%1!=0: # increase mod to speed up testing
            continue
        agent.load(os.path.join(folder, filename))
        for i in range(num_per_checkpoint):
            t, p, a, f, r = feedback_from_trajectory(env, agent, mode='uniform-dense', env_name=game, global_elapsed=time, framerate=rate)
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
        if f'{game}_{time}_{rate}_{label}_trajectories' not in os.listdir(f'./saved_trajectories/{game}'):
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
    encoder.load_state_dict(torch.load(f'./ae/{game}_encoder_mse_bigkernel'))

    n_out = 0
    if game=="pong":
        n_out = 6
    elif game=="breakout":
        n_out = 4

    shrex = Projection(n_out).to(device)
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(shrex.parameters(), lr = lr, weight_decay = 1e-8)

    print("=== STARTING TRAINING ===")

    temp = []

    for traj in procs:
        temp.append(torch.Tensor(traj).to(device))
    
    trajs = temp
    
    for p in range(epochs):
        i = np.random.choice(len(trajs))
        trajectory = trajs[i]
        actions = acts[i]
        feedback = feeds[i]

        # feed,k=feedback.dropout(alpha, beta)
        feed=feedback.get_feedback()
        # h = torch.Tensor(feed).unsqueeze(1).to(device)
        h = torch.Tensor(feed).to(device)
        # traj=trajectory[k[:,0]]

        loc = np.random.choice(trajectory.shape[0]-30)
        # we are interested in where feedback is mostly != 0

        # while (feed[loc:min(loc+25, h.shape[0]-1),:]==0).sum()/(min(loc+25, h.shape[0]-1)-loc)>0.1:
        #     loc = np.random.choice(trajectory.shape[0])

        # choice = np.random.choice(traj.shape[0], batch_size, replace=True)
        traj = trajectory[loc:loc+30,:,:,:] #30 frames = 1s
        h_subset = h[loc:loc+30,:]
        a_subset = actions[loc:loc+30]

        losses = []

        for q in range(30):

            action = a_subset[q]
            h_hat = shrex(encoder(traj[q].unsqueeze(0).detach()))
            hq = h_hat.clone().detach().to(device) * 0.75
            hq[0, action] = h_subset[q]

            optimizer.zero_grad()
            loss = loss_function(h_hat, hq)
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())

        if len(losses)>0 and p % 10 == 0:
            print(f'== EPOCH {p} ==')
            print(sum(losses)/len(losses))
        if p % 2500 == 0 and p != 0:
            torch.save(decoder.state_dict(), f'./models/{game}/atari_{game}_{alpha}_{beta}_{batch_size}_{lr}_{rounds_per_checkpoint}_save_{q}')
    
    # print(np.array(losses))

    print("=== WRITE TO FILE ===")

    with open(f'./models/{game}/losses/atari_{game}_{alpha}_{beta}_{batch_size}_{lr}_{rounds_per_checkpoint}_loss.txt', 'wb') as fp:
        pickle.dump(losses, fp)
    torch.save(shrex.state_dict(), f'./models/{game}/atari_{game}_{alpha}_{beta}_{batch_size}_{lr}_{rounds_per_checkpoint}_shrex')
    final_net = nn.Sequential(
        encoder,
        shrex
    )
    torch.save(final_net.state_dict(), f'./models/{game}/atari_{game}_{alpha}_{beta}_{batch_size}_{lr}_{rounds_per_checkpoint}_final')

def test(game='pong'):

    minmax = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
    if game=='breakout':
        minmax = [[0,0],[0,0],[0,0],[0,0]]

    env, agent = gen_env_agent(game, "atari", model=f"./{game}_ppo_checkpoints/01450")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'cuda available: {torch.cuda.is_available()}')

    model = nn.Sequential(
        Encoder(),
        Projection(6 if game=='pong' else 4)
    ).to(device)

    model.load_state_dict(torch.load(f'./models/{game}/atari_{game}_0.9_0.85_32_1e-20_1_final'))

    done = False
    trajectory=[env.reset()]
    processed = []
    reward=[0]
    action=[]
    while not done:
        done = step_env(env, agent, trajectory, processed, action, reward, mask=True)
        out = (model(torch.Tensor(np.array(processed[-1])).unsqueeze(0).to(device))).cpu().detach().numpy()[0]
        for i in range(len(minmax)):
            minmax[i][0] = min(minmax[i][0], out[i])
            minmax[i][1] = max(minmax[i][1], out[i])
            #out[i] = (out[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
        print(out)
        time.sleep(1/3.0)

def run(mode="test", game="pong"):
    if mode=="test":
        test(game)
    else:
        if game=='pong':
            train_model(game, epochs=400, provide_trajs=False, rounds_per_checkpoint=1, lr=1e-14, alpha=0.9, beta=0.85, time=16, rate=20, label='testing')
        if game=='breakout':
            train_model(game, epochs=1000, provide_trajs=True, rounds_per_checkpoint=1, lr=1e-20, alpha=0.9, beta=0.85, time=16, rate=20, label='uniform-dense')

if __name__=="__main__":
    game="pong"
    epochs_per=1
    # train_model(game, 1)
    run("train", "breakout")
    # print(SHREXNet(1))