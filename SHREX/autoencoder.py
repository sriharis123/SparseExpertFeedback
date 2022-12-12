import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import numpy as np
from traj_from_ppo import gen_env_agent, rollout, step_env
sys.path.insert(0, '../ICML2019-TREX-SparseFeedback/atari')
sys.path.insert(0, '../ICML2019-TREX-SparseFeedback/atari/baselines')
from baselines.common.trex_utils import preprocess
import pickle
from matplotlib import pyplot as plt
import time

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # in_channels, out_channels, kernel_size, stride=1
        self.conv1 = nn.Conv2d(4, 16, 7, padding=3)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(16, 16, 5, padding=2)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv3.weight)
        self.pool = nn.MaxPool2d(2, 2)


        # self.bn1 = nn.BatchNorm2d(16)
        # self.bn2 = nn.BatchNorm2d(8)
        # self.fc1 = nn.Linear(784, 64)
        # self.fc2 = nn.Linear(64, 1)

    def forward(self, state):
        '''compute cumulative return for each trajectory and return logits'''
        x = state.permute(0,3,1,2)
        # print(x.shape)
        x = self.pool(F.leaky_relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.leaky_relu(self.conv2(x)))
        # print(x.shape)
        x = self.pool(F.leaky_relu(self.conv3(x)))
        # print(x.shape)
        # x = self.fc1(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # in_channels, out_channels, kernel_size, stride=1
        self.deconv1 = nn.ConvTranspose2d(16, 16, 2, stride=2, output_padding=1)
        torch.nn.init.kaiming_uniform_(self.deconv1.weight)
        self.deconv2 = nn.ConvTranspose2d(16, 16, 2, stride=2)#, output_padding=2)
        torch.nn.init.kaiming_uniform_(self.deconv2.weight)
        self.deconv3 = nn.ConvTranspose2d(16, 4, 2, stride=2)#, output_padding=3)
        torch.nn.init.kaiming_uniform_(self.deconv3.weight)

        # self.up = nn.Upsample(scale_factor=2, mode='nearest')
        # self.bn1 = nn.BatchNorm2d(16)
        # self.bn2 = nn.BatchNorm2d(4)
        # self.fc1 = nn.Linear(64, 784)

    def forward(self, state):
        '''compute cumulative return for each trajectory and return logits'''
        # x = self.fc1(x)
        x = state.view(state.shape[0], 16, 10, 10)
        # F.relu(self.fc1...)
        # print(x.shape)
        x = F.leaky_relu(self.deconv1(x))
        # print(x.shape)
        x = F.leaky_relu(self.deconv2(x))
        # print(x.shape)
        x = F.leaky_relu(self.deconv3(x))
        # print(x.shape)
        x = x.permute(0,2,3,1)

        return x

def test_model(game):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'cuda available: {torch.cuda.is_available()}')

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    env, agent = gen_env_agent(game, "atari", model="./pong_ppo_checkpoints/01450", seed=807492863)
    encoder.load_state_dict(torch.load(f'./ae/{game}_encoder_mse_bigkernel'))
    decoder.load_state_dict(torch.load(f'./ae/{game}_decoder_mse_bigkernel'))

    done = False
    trajectory=[env.reset()]
    processed = []
    reward=[0]
    action=[]
    while not done:
        done = step_env(env, agent, trajectory, processed, action, reward, mask=True, render=False)
        
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(processed[-1])
        axarr[1].imshow(decoder(encoder(torch.Tensor(np.array(processed[-1])).unsqueeze(0).to(device))).cpu().detach().numpy()[0])
        # print(decoder(encoder(torch.Tensor(np.array(processed[-1])).unsqueeze(0).to(device))).cpu().detach().numpy()[0])
        plt.show()
        time.sleep(1/20.0)
        plt.close()

    # t, a, r, p = rollout(env, agent, game, 5)
    # t = np.array(p)

    # f, axarr = plt.subplots(2,4)
    # for i in range(4):
    #     a = np.random.choice(t.shape[0])
    #     axarr[0,i].imshow(t[a])
    #     axarr[1,i].imshow(decoder(encoder(torch.Tensor(t[a]).unsqueeze(0).to(device)))[0].cpu().detach().numpy())
    # plt.show()
    
    

def train_encoder(game='pong', epochs=1500, batch_size=32, lr=1e-4):

    torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'cuda available: {torch.cuda.is_available()}')

    env, agent = gen_env_agent(game, "atari", model="./pong_ppo_checkpoints/01450")
    encoder=Encoder().to(device)
    decoder=Decoder().to(device)

    # COMMENT OUT THE FOLLOWING LINES TO WARM START MODEL PARAMS
    # encoder.load_state_dict(torch.load(f'./ae/{game}_encoder_mse'))
    # decoder.load_state_dict(torch.load(f'./ae/{game}_decoder_mse'))

    print(encoder)
    print(decoder)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr = lr, weight_decay = 0)
    eps = 1e-16

    print("=== ROLLOUT ===")

    t, a, r, p = rollout(env, agent, game, 10)

    t = np.array(p)

    losses = []

    print("=== TRAINING ===")

    for e in range(epochs):
        batch = torch.Tensor(t[np.random.choice(t.shape[0], batch_size, replace=False),:,:,:]).to(device)

        encoded = encoder(batch)
        decoded = decoder(encoded)

        loss = loss_function(decoded, batch + eps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 100 == 0 and e != 0:
            losses.append(loss.cpu().detach().numpy())
            print(f'Epoch {e}, loss {loss}')
        if e % 10000 == 0 and e != 0:
            torch.save(encoder.state_dict(), f'./ae/{game}_encoder_mse_bigkernel_{e}')
            torch.save(decoder.state_dict(), f'./ae/{game}_decoder_mse_bigkernel_{e}')

    print(np.array(losses))

    print("=== WRITE TO FILE ===")

    torch.save(encoder.state_dict(), f'./ae/{game}_encoder_mse_bigkernel')
    torch.save(decoder.state_dict(), f'./ae/{game}_decoder_mse_bigkernel')



if __name__=="__main__":
    # train_encoder('pong', 50000, 16, 1e-4) # 8, 4e-6)
    test_model('pong')
    

