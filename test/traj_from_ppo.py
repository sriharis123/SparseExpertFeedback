import gym
import time
import sys
sys.path.insert(0, '../ICML2019-TREX-SparseFeedback/atari')
sys.path.insert(0, '../ICML2019-TREX-SparseFeedback/atari/baselines')
from run_test import *
import tensorflow as tf
    
tf.compat.v1.disable_eager_execution()

# Get the appropriate atari game
def get_env_id(env_name="pong"):
    env_id = None
    if env_name == "spaceinvaders":
        env_id = "SpaceInvadersNoFrameskip-v4"
    elif env_name == "mspacman":
        env_id = "MsPacmanNoFrameskip-v4"
    elif env_name == "videopinball":
        env_id = "VideoPinballNoFrameskip-v4"
    elif env_name == "beamrider":
        env_id = "BeamRiderNoFrameskip-v4"
    else:
        env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"
    return env_id

# Needed for appropriate dimensions
env = VecFrameStack(make_vec_env(get_env_id("pong"), 'atari', 1, 3141592653,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       }), nstack=4)

agent = PPO2Agent(env, "atari", True)
# choose an RL agent from the PPO checkpoints
agent.load("./pong_ppo_checkpoints/01050")
observation = env.reset()
reward = 0
t = 0
done = False
while True:
    # 30 FPS
	time.sleep(1.0 / 30)
	t += 1
	env.render(mode="human")
	action = agent.act(observation, reward, done)
	observation, reward, done, info = env.step(action)
	if done:
		print("Episode finished after {} timesteps".format(t+1))
		break
env.close()