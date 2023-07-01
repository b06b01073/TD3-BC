from argparse import ArgumentParser
import gym
import torch

from TD3_agent import PretrainedTD3, TD3Agent

import os

if __name__ == '__main__':

    # parse argument
    parser = ArgumentParser()
    parser.add_argument('--max_step', type=int, default=500000)
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v4')
    parser.add_argument('--smoother_mu', type=float, default=0)
    parser.add_argument('--smoother_sigma', type=float, default=0.2)
    parser.add_argument('--smoother_clip', type=float, default=0.5)
    parser.add_argument('--delay', '-d', type=int, default=2)
    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--gamma', '-g', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--tau', '-t', type=float, default=5e-3)
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--dataset_size', type=int, default=1000000)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_cloning', action='store_false')
    parser.add_argument('--alpha', type=float, default=2.5)
    parser.add_argument('--eps', type=float, default=1e-3)
      
    parser.add_argument('--load_dir', type=str, default='./pretrain_model')
    parser.add_argument('--save_dir', type=str, default='./model_params')
    parser.add_argument('--transition_path', type=str, default='./transition_data')
    parser.add_argument('--write_transitions', '-w', action='store_true')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device}')

    # create necessary dirs


    # generate dataset from pretrained agent
    env = gym.make(args.env_name)
    load_path = os.path.join(args.load_dir, f'{args.env_name}.pth')
    pretrained_agent = PretrainedTD3(env.observation_space, env.action_space, args, device, load_path) 
    with torch.no_grad():
        dataset = pretrained_agent.generate_dataset(env)
        mean, std = dataset.normalize_states(eps=args.eps)

        avg_return = pretrained_agent.evaluate()
        print(f'The model {load_path} has an avg_return of {avg_return} (sample from {args.eval_episodes} episodes)')

    # TD3-BC
    TD3_agent = TD3Agent(env.observation_space, env.action_space, args, device)
    for i in range(args.max_step):
        transitions = dataset.sample(args.batch_size, device) # always assume the size of dataset is larget than batch_size
        losses = TD3_agent.learn(transitions)
        if (i + 1) % args.eval_freq == 0:
            avg_return = TD3_agent.evaluate(mean, std, i)        