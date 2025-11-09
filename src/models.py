"""PyTorch model components for QRL implementation."""

import warnings
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, Sequential

from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from circuits import parametrized_circuit


class EncodingLayer(Module):
    """Classical encoding layer that preprocesses inputs before quantum circuit."""
    
    def __init__(self, num_qubits=4):
        """
        Initialize the encoding layer.
        
        Args:
            num_qubits: Number of qubits (determines weight size)
        """
        super().__init__()
        weights = torch.Tensor(num_qubits)
        self.weights = torch.nn.Parameter(weights)
        torch.nn.init.uniform_(self.weights, -1, 1)
    
    def forward(self, x):
        """
        Forward pass: apply weights and arctangent transformation.
        
        Args:
            x: Input tensor
            
        Returns:
            Transformed tensor
        """
        if not isinstance(x, Tensor):
            x = Tensor(x)
        x = self.weights * x
        x = torch.atan(x)
        return x


class ExpValLayer(Module):
    """Post-processing layer that maps quantum outputs to Q-values."""
    
    def __init__(self, action_space=2):
        """
        Initialize the expectation value layer.
        
        Args:
            action_space: Number of possible actions
        """
        super().__init__()
        weights = torch.Tensor(action_space)
        self.weights = torch.nn.Parameter(weights)
        torch.nn.init.uniform_(self.weights, 35, 40)
        
        self.mask_ZZ_12 = torch.tensor(
            [1., -1., -1., 1., 1., -1., -1., 1., 1., -1., -1., 1., 1., -1., -1., 1.],
            requires_grad=False
        )
        self.mask_ZZ_34 = torch.tensor(
            [-1., -1., -1., -1., 1., 1., 1., 1., -1., -1., -1., -1., 1., 1., 1., 1.],
            requires_grad=False
        )
    
    def forward(self, x):
        """
        Forward pass: compute expectation values and map to Q-values.
        
        Args:
            x: Input tensor from quantum layer
            
        Returns:
            Q-values for each action
        """
        expval_ZZ_12 = self.mask_ZZ_12 * x
        expval_ZZ_34 = self.mask_ZZ_34 * x
        
        if len(x.shape) == 1:
            expval_ZZ_12 = torch.sum(expval_ZZ_12)
            expval_ZZ_34 = torch.sum(expval_ZZ_34)
            out = torch.cat((expval_ZZ_12.unsqueeze(0), expval_ZZ_34.unsqueeze(0)))
        else:
            expval_ZZ_12 = torch.sum(expval_ZZ_12, dim=1, keepdim=True)
            expval_ZZ_34 = torch.sum(expval_ZZ_34, dim=1, keepdim=True)
            out = torch.cat((expval_ZZ_12, expval_ZZ_34), 1)
        
        return self.weights * ((out + 1.) / 2.)


def create_model(num_qubits=4, reuploading=True, reps=6, action_space=2):
    """
    Create the complete QRL model combining classical and quantum layers.
    
    Args:
        num_qubits: Number of qubits in the quantum circuit
        reuploading: Whether to use reuploading strategy
        reps: Number of repetition layers in the quantum circuit
        action_space: Number of possible actions
        
    Returns:
        Sequential model combining encoding, quantum, and expectation value layers
    """
    qc = parametrized_circuit(num_qubits=num_qubits, reuploading=reuploading, reps=reps)
    X = list(qc.parameters)[:num_qubits]
    params = list(qc.parameters)[num_qubits:]
    
    estimator = AerEstimator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=X,
            weight_params=params,
            estimator=estimator
        )
    
    initial_weights = (2 * np.random.rand(qnn.num_weights) - 1)
    quantum_nn = TorchConnector(qnn, initial_weights)
    
    encoding = EncodingLayer(num_qubits=num_qubits)
    exp_val = ExpValLayer(action_space=action_space)
    
    model = Sequential(encoding, quantum_nn, exp_val)
    return model

