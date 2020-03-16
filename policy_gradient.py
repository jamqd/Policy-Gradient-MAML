import random
import cherry as ch
import gym
import numpy as np
import torch
from cherry.algorithms import a2c
from cherry.models.robotics import LinearValue
from torch import optim
from tqdm import tqdm

import learn2learn as l2l
from learn2learn.algorithms.base_learner import BaseLearner
from policy_network import Policy


def compute_advantages(baseline, discount, rewards, dones, states, next_states):
    # Update baseline
    returns = ch.td.discount(discount, rewards, dones)
    baseline.fit(states, returns)
    values = baseline(states)
    next_values = baseline(next_states)
    bootstraps = values * (1.0 - dones) + next_values * dones
    next_value = torch.zeros(1, device=values.device)
    return ch.pg.generalized_advantage(tau=1.0,
                                       gamma=discount,
                                       rewards=rewards,
                                       dones=dones,
                                       values=bootstraps,
                                       next_value=next_value)


def pg_loss(train_episodes, learner, baseline, discount):
    # Update policy and baseline
    states = train_episodes.state()
    actions = train_episodes.action()
    rewards = train_episodes.reward()
    dones = train_episodes.done()
    next_states = train_episodes.next_state()
    log_probs = learner.log_prob(states, actions)
    weights = torch.ones_like(dones)
    weights[1:].add_(-1.0, dones[:-1])
    weights /= dones.sum()
    def weighted_cumulative_sum(values, weights):
        for i in range(values.size(0)):
            values[i] += values[i - 1] * weights[i]
        return values
    cum_log_probs = weighted_cumulative_sum(log_probs, weights)
    advantages = compute_advantages(baseline, discount, rewards,
                                    dones, states, next_states)
    return a2c.policy_loss(l2l.magic_box(cum_log_probs), advantages)

def maml_pg(
        env_name='AntDirection-v1',
        policy_hidden=[128],
        adapt_lr=0.001,
        meta_lr=0.001,
        adapt_steps=1,
        num_iterations=200,
        meta_batch_size=20,
        adapt_batch_size=20,
        discount=0.99,
        num_workers=4,
        seed=0,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def make_env():
        return gym.make(env_name)

    env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])
    env.seed(seed)
    env = ch.envs.Torch(env)
    policy = Policy(input_size=env.state_size, output_size=env.action_size, hidden_dims=policy_hidden)
    meta_learner = l2l.algorithms.MAML(policy, lr=adapt_lr)
    baseline = LinearValue(env.state_size, env.action_size)
    opt = optim.Adam(policy.parameters(), lr=meta_lr)
    all_rewards = []

    for iteration in range(num_iterations):
        iteration_loss = 0.0
        iteration_reward = 0.0
        for task_config in tqdm(env.sample_tasks(meta_batch_size), leave=False, desc='Data'):
            learner = meta_learner.clone()
            env.set_task(task_config)
            env.reset()
            task = ch.envs.Runner(env)

            # adptation
            for step in range(adapt_steps):
                train_episodes = task.run(learner, episodes=adapt_batch_size)
                loss = pg_loss(train_episodes, learner, baseline, discount)
                learner.adapt(loss)

            # validation
            valid_episodes = task.run(learner, episodes=adapt_batch_size)
            loss = pg_loss(valid_episodes, learner, baseline, discount)
            iteration_loss += loss
            iteration_reward += valid_episodes.reward().sum().item() / adapt_batch_size

        print('\nIteration', iteration)
        adaptation_reward = iteration_reward / meta_batch_size
        print('adaptation_reward', adaptation_reward)
        all_rewards.append(adaptation_reward)
        adaptation_loss = iteration_loss / meta_batch_size
        # print('adaptation_loss', adaptation_loss.item())

        opt.zero_grad()
        adaptation_loss.backward()
        opt.step()

    torch.save(learner.state_dict(), './models/' + env_name + "/" + "_".join([str(n) for n in policy_hidden]) + '/pg_maml' + '.pt')


def pretrain_pg(
    env_name='AntDirection-v1',
    policy_hidden=[128],
    adapt_lr=0.001,
    adapt_steps=1,
    num_iterations=200,
    meta_batch_size=20,
    adapt_batch_size=20,
    discount=0.99,
    num_workers=4,
    seed=0
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def make_env():
        return gym.make(env_name)

    env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])
    env.seed(seed)
    env = ch.envs.Torch(env)
    policy = Policy(input_size=env.state_size, output_size=env.action_size, hidden_dims=policy_hidden)
    learner = BaseLearner(policy)
    baseline = LinearValue(env.state_size, env.action_size)
    opt = optim.Adam(policy.parameters(), lr=adapt_lr)
    all_rewards = []

    for iteration in range(num_iterations):
        iteration_loss = 0.0
        iteration_reward = 0.0
        for task_config in tqdm(env.sample_tasks(meta_batch_size), leave=False, desc='Data'):
            env.set_task(task_config)
            env.reset()
            task = ch.envs.Runner(env)

            # update policy
            for step in range(adapt_steps):
                train_episodes = task.run(learner, episodes=adapt_batch_size)
                loss = pg_loss(train_episodes, learner, baseline, discount)
                opt.zero_grad()
                loss.backward()
                opt.step()

            # validation 
            valid_episodes = task.run(learner, episodes=adapt_batch_size)
            loss = pg_loss(valid_episodes, learner, baseline, discount)
            iteration_loss += loss
            iteration_reward += valid_episodes.reward().sum().item() / adapt_batch_size

        print('\nIteration', iteration)
        validation_reward = iteration_reward / meta_batch_size
        print('Validation Reward', validation_reward)
        all_rewards.append(validation_reward)
        validation_loss = iteration_loss / meta_batch_size
        # print('Validation loss', validation_loss.item())
    
    torch.save(learner.state_dict(), './models/' + env_name + "/" + "_".join([str(n) for n in policy_hidden]) + '/pg_pretrain' + '.pt')


def train_pg(
    env_name='AntDirection-v1',
    policy_hidden=[128],
    lr=0.001,
    num_iterations=5,
    batch_size=20,
    discount=0.99,
    num_workers=4,
    seed=0,
    filepath=None,
    mode_str="scratch"
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def make_env():
        return gym.make(env_name)

    env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])
    env.seed(seed)
    env = ch.envs.Torch(env)
    policy = Policy(input_size=env.state_size, output_size=env.action_size, hidden_dims=policy_hidden)
    learner = BaseLearner(policy)
    if filepath:
        print("Using weights from ", filepath)
        learner.load_state_dict(torch.load(filepath))
    baseline = LinearValue(env.state_size, env.action_size)
    opt = optim.Adam(policy.parameters(), lr=lr)

    train_rewards = []
    val_rewards = []

    task_config = env.sample_tasks(1)

    for iteration in range(num_iterations):
        task_config = env.sample_tasks(1)[0]
        env.set_task(task_config)
        env.reset()
        task = ch.envs.Runner(env)

        # update policy
        train_episodes = task.run(learner, episodes=batch_size)
        train_loss = pg_loss(train_episodes, learner, baseline, discount)
        train_reward = train_episodes.reward().sum().item() / batch_size
        train_rewards.append(train_reward)

        opt.zero_grad()
        train_loss.backward()
        opt.step()

        # validation 
        valid_episodes = task.run(learner, episodes=batch_size)
        validation_loss = pg_loss(valid_episodes, learner, baseline, discount)
        validation_reward = valid_episodes.reward().sum().item() / batch_size
        val_rewards.append(validation_reward)

        print('\nIteration', iteration)
        print('Validation Reward', validation_reward)
        # print('Validation loss', validation_loss.item())
    
    torch.save(learner.state_dict(), './models/' + env_name + "/" + "_".join([str(n) for n in policy_hidden]) + '/pg_' + 'train_' + mode_str + '.pt')
    np.save('./performance_data/' + env_name + "/" + "_".join([str(n) for n in policy_hidden]) + '/pg_' + 'train_' + mode_str + '_train_rewards.npy', train_rewards)
    np.save('./performance_data/' + env_name + "/" + "_".join([str(n) for n in policy_hidden]) + '/pg_' + 'train_' + mode_str +  '_val_rewards.npy', val_rewards)
    



if __name__ == '__main__':
    # for testing only
    envs = [
        'AntDirection-v1',
    ]
    for env in envs:
        print("\nUsing environment " + env)
        maml_pg(env_name=env, num_iterations=5)
        # pretrain_pg(env_name=env, num_iterations=5)
        # train_pg(env_name=env)