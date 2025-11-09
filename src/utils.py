"""Utility functions for visualization and animation."""

import warnings
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
from torch import Tensor


class EpisodeVisualizer:
    """Manages a single persistent window for visualizing episodes."""
    
    def __init__(self):
        """Initialize the visualizer with a single figure window."""
        self.fig = None
        self.patch = None
        self.anim = None
        self.current_frames = []
    
    def update_frames(self, frames):
        """
        Update the visualization with new frames from an episode.
        
        Args:
            frames: List of frame images to display
        """
        if not frames:
            return
        
        self.current_frames = frames
        
        if self.fig is None:
            self.fig = plt.figure(figsize=(6, 4))
            self.patch = plt.imshow(frames[0])
            plt.axis('off')
            plt.title("CartPole Simulation", fontsize=12)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
        
        old_anim = None
        if self.anim is not None:
            old_anim = self.anim
            try:
                self.anim.event_source.stop()
                self.anim._stop()
            except:
                pass
        
        def update(frame_num):
            if frame_num < len(self.current_frames):
                self.patch.set_data(self.current_frames[frame_num])
            return self.patch,
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
            self.anim = animation.FuncAnimation(
                self.fig, update, frames=len(frames),
                interval=40, repeat=True, blit=False
            )
        
        self.fig.canvas.draw()
        plt.pause(0.1)
        
        if old_anim is not None:
            try:
                old_anim._stop()
            except:
                pass
            del old_anim
    
    def close(self):
        """Close the visualization window."""
        if self.anim is not None:
            self.anim.event_source.stop()
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.patch = None
            self.anim = None


def update_scene(num, frames, patch):
    """
    Update function for animation frames.
    
    Args:
        num: Current frame number
        frames: List of frame images
        patch: Image patch to update
        
    Returns:
        Updated patch
    """
    patch.set_data(frames[num])
    return patch,


def plot_animation(frames, repeat=False, interval=40):
    """
    Create and return an animation from a list of frames.
    
    Args:
        frames: List of frame images
        repeat: Whether to repeat the animation
        interval: Time interval between frames in milliseconds
        
    Returns:
        Animation object
    """
    fig = plt.figure(figsize=(6, 4))
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval
    )
    plt.show(block=False)
    return anim


def record_episode(env, model, n_outputs, max_steps=200, render_mode="rgb_array"):
    """
    Record an episode of the environment using the trained model.
    
    Args:
        env: Gymnasium environment
        model: Trained Q-value model
        n_outputs: Number of possible actions
        max_steps: Maximum number of steps in the episode
        render_mode: Rendering mode for the environment
        
    Returns:
        Tuple of (frames, total_reward, steps)
    """
    frames = []
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    
    for step in range(max_steps):
        if render_mode == "rgb_array":
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        with torch.no_grad():
            Q_values = model(Tensor(obs)).numpy()
        action = np.argmax(Q_values[0])
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            if render_mode == "rgb_array":
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            break
    
    env.close()
    return frames, total_reward, steps


def visualize_episode(env, model, n_outputs, max_steps=200, seed=None, visualizer=None):
    """
    Visualize a single episode using the trained model.
    
    Args:
        env: Gymnasium environment
        model: Trained Q-value model
        n_outputs: Number of possible actions
        max_steps: Maximum number of steps in the episode
        seed: Random seed for reproducibility
        visualizer: EpisodeVisualizer instance to update (creates new if None)
        
    Returns:
        Tuple of (frames, total_reward, steps)
    """
    frames = []
    obs, info = env.reset(seed=seed)
    total_reward = 0
    steps = 0
    
    for step in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        with torch.no_grad():
            Q_values = model(Tensor(obs)).numpy()
        action = np.argmax(Q_values[0])
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            break
    
    print(f"Episode completed: {steps} steps, Total reward: {total_reward}")
    
    if frames and visualizer is not None:
        visualizer.update_frames(frames)
    
    return frames, total_reward, steps

