"""Main training script for QRL implementation."""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
from torch.optim import Adam
from torch.nn import MSELoss

import gymnasium as gym

from config import (
    NUM_QUBITS, REUPLOADING, REPS, ACTION_SPACE, INPUT_SHAPE,
    REPLAY_MEMORY_SIZE, BATCH_SIZE, DISCOUNT_RATE, LEARNING_RATE,
    MAX_EPISODES, MAX_STEPS_PER_EPISODE, TARGET_SCORE,
    TRAINING_START_EPISODE, EXPLORATION_DECAY_EPISODES, MIN_EPSILON,
    ENV_NAME
)
from models import create_model
from training import play_one_step, sequential_training_step
from utils import visualize_episode, EpisodeVisualizer


def train(visualize_every=1, visualize_final=True):
    """
    Main training loop.
    
    Args:
        visualize_every: Visualize cart pole every N episodes (1 = every episode, None to disable)
        visualize_final: Whether to visualize the final trained model
    """
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    
    model = create_model(
        num_qubits=NUM_QUBITS,
        reuploading=REUPLOADING,
        reps=REPS,
        action_space=ACTION_SPACE
    )
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    rewards = []
    best_score = 0
    
    visualizer = None
    if visualize_every:
        visualizer = EpisodeVisualizer()
    
    for episode in range(MAX_EPISODES):
        obs, info = env.reset()
        
        for step in range(MAX_STEPS_PER_EPISODE):
            epsilon = max(1 - episode / EXPLORATION_DECAY_EPISODES, MIN_EPSILON)
            obs, reward, done, info = play_one_step(
                env, obs, model, ACTION_SPACE, replay_memory, epsilon
            )
            if done:
                break
        
        rewards.append(step)
        
        if step >= best_score:
            best_score = step
        
        print(f"\rEpisode: {episode}, Steps: {step + 1}, eps: {epsilon:.3f}", end="")
        
        if episode > TRAINING_START_EPISODE:
            sequential_training_step(
                model, optimizer, replay_memory, BATCH_SIZE,
                DISCOUNT_RATE, ACTION_SPACE
            )
        
        if visualize_every and (episode + 1) % visualize_every == 0:
            print(f"\nVisualizing episode {episode + 1}...")
            test_env = gym.make(ENV_NAME, render_mode="rgb_array")
            visualize_episode(test_env, model, ACTION_SPACE, max_steps=MAX_STEPS_PER_EPISODE, visualizer=visualizer)
            test_env.close()
    
    env.close()
    
    if visualizer:
        visualizer.close()
    
    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Sum of rewards", fontsize=14)
    plt.title("Training Progress")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    if visualize_final:
        print("\nVisualizing final trained model...")
        final_visualizer = EpisodeVisualizer()
        test_env = gym.make(ENV_NAME, render_mode="rgb_array")
        visualize_episode(test_env, model, ACTION_SPACE, max_steps=MAX_STEPS_PER_EPISODE, seed=42, visualizer=final_visualizer)
        test_env.close()
        input("\nPress Enter to close the visualization...")
        final_visualizer.close()
    
    return model, rewards


if __name__ == "__main__":
    model, rewards = train(visualize_every=1, visualize_final=True)

