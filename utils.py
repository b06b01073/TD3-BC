import numpy as np
import matplotlib.pyplot as plt
import os

def plot_result(trials_BC_avg_returns, trials_avg_returns, behavior_avg_return, max_steps, eval_freq, env_name, result_dir, imperfect_demo):
    trials_BC_avg_returns = np.array(trials_BC_avg_returns)
    BC_std = np.std(trials_BC_avg_returns, axis=0)
    BC_mean = np.mean(trials_BC_avg_returns, axis=0)

    trials_avg_returns = np.array(trials_avg_returns)
    std = np.std(trials_avg_returns, axis=0)
    mean = np.mean(trials_avg_returns, axis=0)
    x = [i * eval_freq for i in range(max_steps // eval_freq + 1)]

    plt.title(env_name)
    plt.xlabel('Time steps')
    plt.ylabel('Average Return')


    plt.plot(x, BC_mean, color='#f7761c', label='TD3-BC')
    plt.fill_between(x, BC_mean + BC_std, BC_mean - BC_std, color='#f6bf6b', alpha=0.4)


    plt.plot(x, mean, color='#2277aa', label='TD3')
    plt.fill_between(x, mean + std, mean - std, color='#a8d1df', alpha=0.4)
    
    plt.axhline(behavior_avg_return, color='r', label='Behavior')

    plt.legend(loc='best')

    file_name = f'{env_name}_imperfect.jpg' if imperfect_demo else f'{env_name}.jpg' 


    plt.savefig(os.path.join(result_dir, file_name)) 