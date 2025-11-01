# Quantum Reinforcement Learning (QiRL) - Deep Q-Network

A quantum-enhanced Deep Q-Network (DQN) implementation using Qiskit Machine Learning and PyTorch for reinforcement learning on the CartPole-v1 environment.

## Overview

This project implements a quantum-classical hybrid neural network for reinforcement learning, combining:
- **Quantum Neural Networks (QNN)** using Qiskit
- **Classical pre/post-processing layers** using PyTorch
- **Deep Q-Learning algorithm** for value-based reinforcement learning

The quantum circuit uses parameterized quantum gates to learn Q-values, which are then processed through classical layers to make action decisions in the CartPole-v1 environment.

## Features

- Quantum parameterized circuits with reuploading strategy
- Classical encoding layer for state preprocessing
- Quantum neural network with EstimatorQNN
- Classical expectation value layer for Q-value estimation
- Experience replay buffer
- Epsilon-greedy exploration strategy
- Sequential training to handle quantum gradient issues

## Requirements

- Python 3.13+
- Qiskit (1.x, < 2.0)
- Qiskit Machine Learning (0.8.2)
- PyTorch
- Gymnasium (for CartPole-v1 environment)
- NumPy, Matplotlib

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd QiRL
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies using `uv`:
```bash
uv pip install -r requirements.txt
```

Or using `pip`:
```bash
pip install -r requirements.txt
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook DQN.ipynb
```

2. Run the cells sequentially to:
   - Set up the quantum and classical layers
   - Define the encoding and parameterized quantum circuits
   - Create the hybrid quantum-classical model
   - Train the agent on CartPole-v1 environment
   - Analyze training results

## Project Structure

```
QiRL/
├── DQN.ipynb              # Main notebook with QNN-DQN implementation
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── venv/                 # Virtual environment (gitignored)
```

## Architecture

The model consists of three main components:

1. **Encoding Layer**: Classical preprocessing layer that transforms the state
2. **Quantum Neural Network**: Parameterized quantum circuit using EstimatorQNN
3. **Expectation Value Layer**: Classical post-processing to compute Q-values for each action

## Key Components

- **Parameterized Quantum Circuit**: Uses TwoLocal ansatz with rotation gates (Ry, Rz) and CZ entanglers
- **Replay Memory**: Experience replay buffer for training stability
- **Sequential Training**: Handles quantum gradient vanishing issues when batching

## Environment

- **CartPole-v1**: Classic control problem where the agent must balance a pole on a cart

## Training

The agent trains for 2000 episodes with:
- Batch size: 16
- Discount rate: 0.99
- Learning rate: 1e-2
- Epsilon decay: Linear from 1.0 to 0.01 over 1500 episodes

## Reference

This implementation is based on the work from:
- [LauraGentini/QRL - Deep Q-Learning](https://github.com/LauraGentini/QRL/blob/main/2-QNNDeepQLearning/Deep_Q_Learning.ipynb)

## License

This project is for educational purposes.

## Notes

- Compatible with Qiskit 1.x (< 2.0) due to API changes in Qiskit 2.0
- Uses `qiskit_aer` for local quantum simulation
- Sequential training is used instead of batched training to avoid quantum gradient vanishing issues

