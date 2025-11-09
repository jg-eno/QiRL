"""Script to visualize a trained QRL model."""

import gymnasium as gym
import torch

from config import ENV_NAME, ACTION_SPACE, MAX_STEPS_PER_EPISODE, NUM_QUBITS, REUPLOADING, REPS
from models import create_model
from utils import visualize_episode


def visualize_model(model_path=None, num_episodes=1, seed=42):
    """
    Visualize a trained model running in the environment.
    
    Args:
        model_path: Path to saved model weights (None to use a new model)
        num_episodes: Number of episodes to visualize
        seed: Random seed for reproducibility
    """
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    
    model = create_model(
        num_qubits=NUM_QUBITS,
        reuploading=REUPLOADING,
        reps=REPS,
        action_space=ACTION_SPACE
    )
    
    if model_path:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print("Using untrained model (random weights)")
    
    model.eval()
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        visualize_episode(env, model, ACTION_SPACE, max_steps=MAX_STEPS_PER_EPISODE, seed=seed + episode)
    
    env.close()


if __name__ == "__main__":
    import sys
    
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    visualize_model(model_path=model_path, num_episodes=1, seed=42)

