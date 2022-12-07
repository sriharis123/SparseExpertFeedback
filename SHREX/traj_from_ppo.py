import gym
import time
import keyboard
import sys
sys.path.insert(0, '../ICML2019-TREX-SparseFeedback/atari')
sys.path.insert(0, '../ICML2019-TREX-SparseFeedback/atari/baselines')
from run_test import *
import tensorflow as tf
from credit_assignment import assign_credit, plot_reward

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
    agent = PPO2Agent(env, "atari", True)
    agent.load("./pong_ppo_checkpoints/01050")
    return env, agent

# Takes wrapped atari env, PPO/other RL agent, prev observation. Returns whether finished and reward for step.
def step_env(environment, agent, trajectory, reward):
    env.render(mode="human") # uncomment for viz; look at source for replay support? issue - demonstration has 4 dims
    action = agent.act(trajectory[-1], reward[-1], False)
    o, r, done, info = env.step(action)
    trajectory.append(o)
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

def feedback_from_trajectory(env, agent, global_elapsed=8, framerate=30):
    import time

    render_correction = 1 # locks delay to proper fps

    trajectory = []
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
        done = step_env(env, agent, trajectory, reward)

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
    
    env.close()
    time.sleep(1)
    
    trajectory = trajectory[1:]
    fb_np = np.zeros_like(reward[1:])#[:,np.newaxis]

    for time in feedback:
        fb_np = assign_credit(fb_np, time, feedback[time])

    return trajectory, fb_np, reward


if __name__=="__main__":

    env, agent = gen_env_agent() # default is pong

    traj, feedback, reward = feedback_from_trajectory(env, agent)

    plot_reward(feedback)

    print("Cumulative reward:", feedback)


