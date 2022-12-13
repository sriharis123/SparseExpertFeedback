import gym
import time
import keyboard
import sys
sys.path.insert(0, '../ICML2019-TREX-SparseFeedback/atari')
sys.path.insert(0, '../ICML2019-TREX-SparseFeedback/atari/baselines')
from baselines.common.trex_utils import preprocess
from run_test import *
import tensorflow as tf
from credit_assignment import Credit

tf.compat.v1.disable_eager_execution() # this may slow down tf calculations so might want to address this later

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

# Generate the environment and agent for feedback assignment
def gen_env_agent(name="pong", game_type="atari", seed=3141592653, n=4, model="./pong_ppo_checkpoints/01050"):
    env = VecFrameStack(make_vec_env(get_env_id(name), game_type, 1, seed,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       }), nstack=n)
    agent = PPO2Agent(env, game_type, True)
    agent.load(model)
    return env, agent

# Takes wrapped atari env, PPO/other RL agent, prev observation. Returns whether finished and reward for step.
def step_env(environment, agent, trajectory, processed, actions, reward, mask=True, contrast=False, env_name='pong', render=True):
    if render:
        environment.render(mode="human") # uncomment for viz; look at source for replay support? issue - demonstration has 4 dims
    action = agent.act(trajectory[-1], reward[-1], False)
    # print(action)
    o, r, done, info = environment.step(action)
    if mask:
        p = preprocess(o, env_name, contrast=contrast) #TODO is preprocess necessary???
        processed.append(p[0])
    trajectory.append(o[0]) # append just HWC
    actions.append(action)
    reward.append(r)
    return done

def read_signal():
    if keyboard.is_pressed("1"):
        return -1.0
    elif keyboard.is_pressed("2"):
        return -0.75
    elif keyboard.is_pressed("3"):
        return -0.5
    elif keyboard.is_pressed("4"):
        return -0.25
    elif keyboard.is_pressed("7"):
        return 0.25
    elif keyboard.is_pressed("8"):
        return 0.5
    elif keyboard.is_pressed("9"):
        return 0.75
    elif keyboard.is_pressed("0"):
        return 1.0

    return 0.0

# rollout without render
def rollout(env, agent, env_name="pong", global_elapsed=20, contrast=False):
    import time

    trajectory = []
    processed = []
    actions = []
    reward = [] # ppo agent reward
    
    trajectory.append(env.reset()) # initial value, to be erased
    reward.append(0) # initial value, to be erased

    global_start = time.time() # used for time exit

    done = False
    while not done:
        step_env(env, agent, trajectory, processed, actions, reward, mask=True, contrast=contrast, env_name=env_name, render=False)
        done = (time.time()-global_start>global_elapsed)

    trajectory = trajectory[1:]
    reward = reward[1:]

    print(f'trajectory length: {len(trajectory)}')

    return trajectory, actions, reward, processed



def feedback_from_trajectory(env, agent, mode='uniform', env_name='pong', global_elapsed=8, framerate=30):
    import time

    render_correction = 1 # locks  delay to proper fps

    trajectory = []
    processed = []
    actions = []
    feedback = {} # maps timestep to feedback signal
    reward = [] # ppo agent reward

    delay = 1.0 / framerate
    cdelay = 1.0 / (framerate + render_correction)

    trajectory.append(env.reset()) # initial value, to be erased
    reward.append(0) # initial value, to be erased

    prev_time = 0
    prev_signal = 0
    t = 0
    global_start = time.time() # used for time exit

    done = False
    while not done:

        start = time.time()
        # correction if elapsed time exceeds FPS
        if start - prev_time > delay:
            render_correction += 1
            cdelay = 1.0 / (framerate + render_correction)

        prev_time = start
        step_env(env, agent, trajectory, processed, actions, reward, mask=True, env_name=env_name)

        # feedback assignment. MAKE SURE TO FOCUS THE PROMPT YOU ARE RUNNING IN
        signal = read_signal()
        if signal != prev_signal: # to deal with constant press. feedback is provided instantaneously.
            prev_signal = signal
            if signal != 0.0:
                print("Signal received:", signal)
                feedback[t] = signal

        # Need to wait between timesteps.
        t += 1
        wait_time = cdelay - (time.time() - start)
        if wait_time >= 0:
            # iteration took under "fps" seconds. proceed to next round after waiting necessary time.
            time.sleep(wait_time)
        
        done = time.time() - global_start > global_elapsed # for testing. comment out to continue until actual done condition
    
    time.sleep(1)
    
    trajectory = trajectory[1:]
    credit = Credit(len(reward)-1, mode)

    for t in feedback:
        credit.assign(feedback[t], t)

    return trajectory, processed, np.array(actions), credit, reward


if __name__=="__main__":

    env, agent = gen_env_agent(name='pong', model='./pong_ppo_checkpoints/01450') # default is pong

    traj, actions, credit, reward = feedback_from_trajectory(env, agent, mode='atari')

    env.close()

    credit.plot_feedback()

    print("Cumulative reward:", credit.get_feedback())


