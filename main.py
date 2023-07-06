from argparse import ArgumentParser
import gymnasium as gym
import torch

from TD3_agent import PretrainedTD3, TD3Agent
import utils

import os

if __name__ == '__main__':

    # parse argument
    parser = ArgumentParser()
    parser.add_argument('--max_step', type=int, default=10000)
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    parser.add_argument('--smoother_mu', type=float, default=0)
    parser.add_argument('--smoother_sigma', type=float, default=0.2)
    parser.add_argument('--smoother_clip', type=float, default=0.5)
    parser.add_argument('--delay', '-d', type=int, default=2)
    parser.add_argument('--batch_size', '-b', type=int, default=256) 
    parser.add_argument('--gamma', '-g', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--tau', type=float, default=5e-3)
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_size', type=int, default=1000000) # dataset size is suggested to be greater than max len of an episode and batch size, otherwise errors may occur
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=2.5)
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--imperfect_demo', '-i', action='store_true')
    parser.add_argument('--exploration_prob', '-p', type=float, default=0.3)
    parser.add_argument('--exploration_std', '-s', type=float, default=0.3)
    parser.add_argument('--trials', '-t', type=int, default=5)
      
    parser.add_argument('--load_dir', type=str, default='./pretrain_model')
    parser.add_argument('--save_dir', type=str, default='./model_params')
    parser.add_argument('--transition_path', type=str, default='./transition_data')
    parser.add_argument('--result_dir', type=str, default='./result')

    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device}')

    # create necessary dirs
    data_path = os.path.join(args.transition_path, args.env_name) # create dir for transition data
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # generate dataset from pretrained agent
    env = gym.make(args.env_name)
    load_path = os.path.join(args.load_dir, f'{args.env_name}.pth')
    pretrained_agent = PretrainedTD3(env.observation_space, env.action_space, args, device, load_path) 
    with torch.no_grad():
        replay_buffer, behavrior_avg_return = pretrained_agent.generate_dataset(env, data_path, args.exploration_prob, args.exploration_std, args.imperfect_demo)
        mean, std = replay_buffer.normalize_states(eps=args.eps)

    # TD3-BC and TD3
    trials_BC_returns = []
    trials_avg_returns = []
    for _ in range(args.trials):
        TD3_BC_agent = TD3Agent(env.observation_space, env.action_space, args, device)
        TD3_agent = TD3Agent(env.observation_space, env.action_space, args, device)
        BC_avg_returns = [TD3_BC_agent.evaluate(mean, std, 0)]
        avg_returns = [TD3_agent.evaluate(mean, std, 0)]
        for i in range(args.max_step):
            transitions = replay_buffer.sample(args.batch_size, device) # always assume the size of dataset is larget than batch_size

            TD3_BC_agent.learn(transitions=transitions, batch_cloning=True)
            TD3_agent.learn(transitions=transitions, batch_cloning=False)
            if (i + 1) % args.eval_freq == 0:
                with torch.no_grad():
                    print('evaluating TD3-BC...')
                    BC_avg_return = TD3_BC_agent.evaluate(mean, std, i)        
                    BC_avg_returns.append(BC_avg_return)

                    print('evaluating TD3...')
                    avg_return = TD3_agent.evaluate(mean, std, i)        
                    avg_returns.append(avg_return)

        trials_BC_returns.append(BC_avg_returns)
        trials_avg_returns.append(avg_returns)

    utils.plot_result(trials_BC_returns, trials_avg_returns, behavrior_avg_return, args.max_step, args.eval_freq, args.env_name, args.result_dir, args.imperfect_demo)

