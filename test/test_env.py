import gym
import time
env = gym.make('PongDeterministic-v4', render_mode='human')
# Uncomment following line to save video of our Agent interacting in this environment
# This can be used for debugging and studying how our agent is performing
# env = gym.wrappers.Monitor(env, './video/', force = True)
env.reset()
t = 0
while True:
	time.sleep(0.1)
	t += 1
	#env.render()
	observation = env.reset()
	print(observation)
	action = env.action_space.sample()
	observation, reward, done, info = env.step(action)
	if done:
		print("Episode finished after {} timesteps".format(t+1))
		break
env.close()