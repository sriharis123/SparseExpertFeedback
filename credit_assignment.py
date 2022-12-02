from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np


gamma_a = 3
gamma_rv = gamma(gamma_a)

def assign_credit(rewards, t, new_reward):
    credit_frames = 60

    x = np.linspace(gamma.ppf(0.001, gamma_a),
                gamma.ppf(0.999, gamma_a), credit_frames)
    gamma_pdf = gamma_rv.pdf(x)[::-1]

    credit = np.zeros_like(rewards, dtype=float)
    t_start_credit = max(0, (t - credit_frames))
    credit[t_start_credit:t] = new_reward * gamma_pdf[:t]

    rewards = rewards + credit
    
    return rewards

def plot_reward(rewards):
    fig, ax = plt.subplots(1, 1)

    num_frames = len(rewards)
    x = np.arange(num_frames)

    ax.plot(x, rewards)
    plt.show()


rewards = [0] * 120
rew_after_credit = assign_credit(rewards, t=100, new_reward=8)
plot_reward(rew_after_credit)
