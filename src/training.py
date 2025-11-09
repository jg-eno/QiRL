"""Training functions for QRL implementation."""

import numpy as np
import torch
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import Adam


def epsilon_greedy_policy(state, model, n_outputs, epsilon=0):
    """
    Select action using epsilon-greedy policy.
    
    Args:
        state: Current state
        model: Q-value model
        n_outputs: Number of possible actions
        epsilon: Exploration probability
        
    Returns:
        Selected action index
    """
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        with torch.no_grad():
            Q_values = model(Tensor(state)).numpy()
        return np.argmax(Q_values[0])


def sample_experiences(replay_memory, batch_size):
    """
    Sample a batch of experiences from replay memory.
    
    Args:
        replay_memory: Deque containing past experiences
        batch_size: Number of experiences to sample
        
    Returns:
        Tuple of (states, actions, rewards, next_states, dones)
    """
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    
    states_list = []
    next_states_list = []
    for experience in batch:
        state_val = experience[0]
        if isinstance(state_val, tuple):
            state_val = state_val[0]
        state = np.asarray(state_val, dtype=np.float32).flatten()
        
        next_state_val = experience[3]
        if isinstance(next_state_val, tuple):
            next_state_val = next_state_val[0]
        next_state = np.asarray(next_state_val, dtype=np.float32).flatten()
        
        states_list.append(state)
        next_states_list.append(next_state)
    
    states = np.stack(states_list)
    actions = np.array([experience[1] for experience in batch])
    rewards = np.array([experience[2] for experience in batch], dtype=np.float32)
    next_states = np.stack(next_states_list)
    dones = np.array([experience[4] for experience in batch], dtype=np.bool_)
    
    return states, actions, rewards, next_states, dones


def play_one_step(env, state, model, n_outputs, replay_memory, epsilon):
    """
    Perform one action in the environment and store the experience.
    
    Args:
        env: Gymnasium environment
        state: Current state
        model: Q-value model
        n_outputs: Number of possible actions
        replay_memory: Deque to store experiences
        epsilon: Exploration probability
        
    Returns:
        Tuple of (next_state, reward, done, info)
    """
    action = epsilon_greedy_policy(state, model, n_outputs, epsilon)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info


def sequential_training_step(model, optimizer, replay_memory, batch_size, discount_rate, n_outputs):
    """
    Training step using sequential loss evaluation.
    
    This implementation evaluates individual losses sequentially instead of using batches.
    This is due to an issue in the TorchConnector, which yields vanishing gradients if it
    is called with a batch of data.
    
    Args:
        model: Q-value model
        optimizer: PyTorch optimizer
        replay_memory: Deque containing past experiences
        batch_size: Number of experiences to sample
        discount_rate: Discount factor for future rewards
        n_outputs: Number of possible actions
    """
    experiences = sample_experiences(replay_memory, batch_size)
    states, actions, rewards, next_states, dones = experiences
    
    with torch.no_grad():
        next_Q_values = model(Tensor(next_states)).numpy()
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards + (1 - dones) * discount_rate * max_next_Q_values)
    
    loss = 0.
    for j, state in enumerate(states):
        single_Q_value = model(Tensor(state))
        Q_value = single_Q_value[actions[j]]
        loss += (target_Q_values[j] - Q_value) ** 2
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def training_step(model, optimizer, loss_fn, replay_memory, batch_size, discount_rate, n_outputs):
    """
    Training step using batched loss evaluation.
    
    Can be used for classical models. For quantum models, use sequential_training_step instead.
    
    Args:
        model: Q-value model
        optimizer: PyTorch optimizer
        loss_fn: Loss function
        replay_memory: Deque containing past experiences
        batch_size: Number of experiences to sample
        discount_rate: Discount factor for future rewards
        n_outputs: Number of possible actions
    """
    experiences = sample_experiences(replay_memory, batch_size)
    states, actions, rewards, next_states, dones = experiences
    
    with torch.no_grad():
        next_Q_values = model(Tensor(next_states)).numpy()
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards + (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = torch.nn.functional.one_hot(Tensor(actions).long(), n_outputs)
    
    all_Q_values = model(Tensor(states))
    Q_values = torch.sum(all_Q_values * mask, dim=1, keepdims=True)
    loss = loss_fn(Tensor(target_Q_values), Q_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

