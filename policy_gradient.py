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
from policy_network import Policy


def weighted_cumsum(values, weights):
    for i in range(values.size(0)):
        values[i] += values[i - 1] * weights[i]
    return values


def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):
    # Update baseline
    returns = ch.td.discount(gamma, rewards, dones)
    baseline.fit(states, returns)
    values = baseline(states)
    next_values = baseline(next_states)
    bootstraps = values * (1.0 - dones) + next_values * dones
    next_value = torch.zeros(1, device=values.device)
    return ch.pg.generalized_advantage(tau=tau,
                                       gamma=gamma,
                                       rewards=rewards,
                                       dones=dones,
                                       values=bootstraps,
                                       next_value=next_value)


def pg_loss(train_episodes, learner, baseline, gamma, tau):
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
    cum_log_probs = weighted_cumsum(log_probs, weights)
    advantages = compute_advantages(baseline, tau, gamma, rewards,
                                    dones, states, next_states)
    return a2c.policy_loss(l2l.magic_box(cum_log_probs), advantages)


def train(env_name='Particles2D-v1', filepath=None):


    policy = Policy(input_size=env.state_size, output_size=env.action_size)
    if filepath:
        policy.load_state_dict(torch.load(filepath))

    return 


def maml_pg(
        env_name='AntDirection-v1',
        adapt_lr=0.001,
        meta_lr=0.001,
        adapt_steps=1,
        num_iterations=200,
        meta_batch_size=20,
        adapt_batch_size=20,
        tau=1.00,
        gamma=0.99,
        num_workers=2,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def make_env():
        return gym.make(env_name)

    env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])
    env.seed(seed)
    env = ch.envs.Torch(env)
    policy = Policy(input_size=env.state_size, output_size=env.action_size)
    meta_learner = l2l.algorithms.MAML(policy, lr=adapt_lr)
    baseline = LinearValue(env.state_size, env.action_size)
    opt = optim.Adam(policy.parameters(), lr=meta_lr)
    all_rewards = []

    for iteration in range(num_iterations):
        iteration_loss = 0.0
        iteration_reward = 0.0
        for task_config in tqdm(env.sample_tasks(meta_batch_size), leave=False, desc='Data'):  # Samples a new config
            learner = meta_learner.clone()
            env.set_task(task_config)
            env.reset()
            task = ch.envs.Runner(env)

            # Fast Adapt
            for step in range(adapt_steps):
                train_episodes = task.run(learner, episodes=adapt_batch_size)
                loss = pg_loss(train_episodes, learner, baseline, gamma, tau)
                learner.adapt(loss)

            # Compute Validation Loss
            valid_episodes = task.run(learner, episodes=adapt_batch_size)
            loss = pg_loss(valid_episodes, learner, baseline, gamma, tau)
            iteration_loss += loss
            iteration_reward += valid_episodes.reward().sum().item() / adapt_batch_size

        # Print statistics
        print('\nIteration', iteration)
        adaptation_reward = iteration_reward / meta_batch_size
        print('adaptation_reward', adaptation_reward)
        all_rewards.append(adaptation_reward)

        adaptation_loss = iteration_loss / meta_batch_size
        print('adaptation_loss', adaptation_loss.item())

        opt.zero_grad()
        adaptation_loss.backward()
        opt.step()

    torch.save(model.state_dict(), './models/pg_maml.pt')


def pretrain_pg(
    env_name='AntDirection-v1',
    adapt_lr=0.001,
    adapt_steps=1,
    num_iterations=200,
    meta_batch_size=20,
    adapt_batch_size=20,
    tau=1.00,
    gamma=0.99,
    num_workers=2,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def make_env():
        return gym.make(env_name)

    env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])
    env.seed(seed)
    env = ch.envs.Torch(env)
    policy = Policy(input_size=env.state_size, output_size=env.action_size)
    learner = l2l.algorithms.MAML(policy, lr=adapt_lr)
    baseline = LinearValue(env.state_size, env.action_size)
    opt = optim.Adam(policy.parameters(), lr=adapt_lr)
    all_rewards = []

    for iteration in range(num_iterations):
        iteration_loss = 0.0
        iteration_reward = 0.0
        for task_config in tqdm(env.sample_tasks(meta_batch_size), leave=False, desc='Data'):  # Samples a new config
            env.set_task(task_config)
            env.reset()
            task = ch.envs.Runner(env)

            # update policy
            for step in range(adapt_steps):
                train_episodes = task.run(learner, episodes=adapt_batch_size)
                loss = pg_loss(train_episodes, learner, baseline, gamma, tau)
                # learner.adapt(loss)
                opt.zero_grad()
                loss.backward()
                opt.step()

            # Compute Validation Loss
            valid_episodes = task.run(learner, episodes=adapt_batch_size)
            loss = pg_loss(valid_episodes, learner, baseline, gamma, tau)
            iteration_loss += loss
            iteration_reward += valid_episodes.reward().sum().item() / adapt_batch_size

        # Print statistics
        print('\nIteration', iteration)
        adaptation_reward = iteration_reward / meta_batch_size
        print('adaptation_reward', adaptation_reward)
        all_rewards.append(adaptation_reward)

        adaptation_loss = iteration_loss / meta_batch_size
        print('adaptation_loss', adaptation_loss.item())
        
    torch.save(model.state_dict(), './models/pg_pretrain.pt')


def train_pg(
    env_name='AntDirection-v1',
    lr=0.001,
    num_iterations=200,
    meta_batch_size=20,
    adapt_batch_size=20,
    tau=1.00,
    gamma=0.99,
    num_workers=2,
    seed=42
    filepath=None
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    def make_env():
        return gym.make(env_name)

    env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])
    env.seed(seed)
    env = ch.envs.Torch(env)
    policy = Policy(input_size=env.state_size, output_size=env.action_size)
    learner = l2l.algorithms.MAML(policy, lr=adapt_lr)
    baseline = LinearValue(env.state_size, env.action_size)
    opt = optim.Adam(policy.parameters(), lr=adapt_lr)
    all_rewards = []

    for iteration in range(num_iterations):
        iteration_loss = 0.0
        iteration_reward = 0.0
        for task_config in tqdm(env.sample_tasks(meta_batch_size), leave=False, desc='Data'):  # Samples a new config
            env.set_task(task_config)
            env.reset()
            task = ch.envs.Runner(env)

            # update policy
            for step in range(adapt_steps):
                train_episodes = task.run(learner, episodes=adapt_batch_size)
                loss = pg_loss(train_episodes, learner, baseline, gamma, tau)
                # learner.adapt(loss)
                opt.zero_grad()
                loss.backward()
                opt.step()

            # Compute Validation Loss
            valid_episodes = task.run(learner, episodes=adapt_batch_size)
            loss = pg_loss(valid_episodes, learner, baseline, gamma, tau)
            iteration_loss += loss
            iteration_reward += valid_episodes.reward().sum().item() / adapt_batch_size

        # Print statistics
        print('\nIteration', iteration)
        adaptation_reward = iteration_reward / meta_batch_size
        print('adaptation_reward', adaptation_reward)
        all_rewards.append(adaptation_reward)

        adaptation_loss = iteration_loss / meta_batch_size
        print('adaptation_loss', adaptation_loss.item())
        
    torch.save(model.state_dict(), './models/pg_pretrain.pt')



if __name__ == '__main__':
    # testing vpg
    # envs = [
    #     'HalfCheetahForwardBackward-v1', 
    #     'AntForwardBackward-v1', 
    #     'AntDirection-v1', 
    #     'HumanoidForwardBackward-v1', 
    #     'HumanoidDirection-v1',
    #     'Particles2D-v1'
    # ]
    envs = [
        'AntDirection-v1',
    ]
    for env in envs:
        print("\nUsing environment " + env)
        # maml_pg(env_name=env, num_iterations=5, num_workers=4)
        pretrain_pg(env_name=env, num_iterations=5, num_workers=4)