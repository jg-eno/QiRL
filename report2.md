### Members:
- J Glen Enosh
- Vibhaas Nirantar Srivastava
- Vipin Karthic

## 1. Abstract 
We explore the integration of quantum computing with reinforcement learning through a hybrid quantum classical Deep Q Network. The architecture takes advantage of the parameterized quantum circuits as function approximators inside the DQN  to learn the best strategies for CartPole v1 control task. The QNN is constructed using Qiskit and with PyTorch and are merged together using the TorchConnector interface. The quantum circuit uses a data reuploading method where the input is fed into the circuit multiple times. It is made of Rotation gates and controlled Z gates, In the classical layers, first the input is transformed, then the last layer converts the quantum outputs into Q values, The experiments done show that, value based RL can be done successfully

---
## 2. Introduction 
#### 2.1 Motivation
RL trains the AI agents to make continuous good decisions in a given environment, The DQN shows that merging Q Learning with NN can reach performances shown by Humans, The more complex the problems get the more the classical NNs struggle with capacity, training and scalability.

Quantum computing gives us a new path to process information using the properties such as superposition, entanglement and interference. This basically led us to QML, which improves the feature representation and optimization. These both when used together, QML and Classical RL, we have the QRL (Quantum Reinforcement Learning)

The quantum circuits can represent the complex patterns better and more efficiently, The training speed and convergence to the optimal solution is faster. The Quantum models can handle the tasks that are more complex tasks efficiently which classical ones struggle. Today's generation of the Quantum computers, (NISQ devices) allow the running os small scale QRL models

#### 2.2 Related Literature Work
In the past decade or so, QRL has advanced both theoretically and experimentally.
+ Variational Quantum Algorithms : VQE and QAOA allowed us to do a quantum classical optimization, where the quantum circuits are treated as trainable models
+ Quantum Circuit Learning : Mitarai and Farhi & Neven using the parameter shift rule, and using the gradient based training created frameworks for this
+ Data Reuploading : Perez Salinas proved that repeatedly encoding data between circuit, makes the model better overtime
+ Quantum Policy Gradients : Jerbi made the QRL better by using policy gradient methods
+ Hybrid Quantum Classical models : Chen and Sholik merged quantum circuits with classical networks to show that it can be done while balancing performance and practicality
+ Practical Implementations : Lockwood and Saggio showed the running of QRL on real devices, and also handled issues like noise and circuit depth
+ CartPole Experiments : Gemini showed that properly designed quantum circuits can learn the optimal CartPole control strategies

#### 2.3 Brief Report on Upcoming Sections
The following sections are as follows : 
+ The problem statement is formally defined, with respect to the task we are trying to perform
+ We analyze the current research gap in QRL which identifies the challenges in the quantum gradient computation, and lack of good benchmarking studies
+ We then detail the proposed architecture and the details for the system we are trying to implement, starting from the quantum circuit design to the training methods for the QNNs
+ @Vibhaas finish this part, idk what to write here (Just one sentence on what the section is doing)


## 3. Problem Statement
The goal of this report is to test whether the PQS can replace DQNs for reinforcement learning, We mainly focus on the CartPole v1 environment where the AI agent learns to balance a pole on a moving cart

The setup is as follows  : (It is modelled as a Markov Decision Process) :
- S : Continuous 4D state space
- A : Discrete action space : {push left, push right}
- P : Transition probabilities between states
- R : Reward function (+1 per step while balanced)
- γ : Discount factor for future rewards
+ The quantum network uses a parameterized circuit Uθ(x) that:
	+ Encodes the classical state  `x` into quantum form
	+ Applies tunable quantum gates with parameters `θ`
	+ Measures observables to output Q-values for each action
+ The goal is to learn the parameters `θ*` such that : 
$$
Q^*(s, a) \approx Q_{\theta^*}(s, a) = \langle \psi_{\theta^*}(s) | \hat{O}_a | \psi_{\theta^*}(s) \rangle
$$

where $|\psi_{\theta^*}(s)\rangle = U_{\theta^*}(x)|0\rangle$ with $x$ being the encoded classical state, and $\hat{O}_a$ are Pauli observables (specifically $Z_0 Z_1$ for action 0 and $Z_2 Z_3$ for action 1).

## 4. Current Research Gap
#### 4.1 Gradient Vanishing
This issue occurs when we train a quantum network with batches of data, in the quantum models, the gradients converge to near zero when inputs are processed together, So the training must be done one sample at a time, which ends up resulting in : 
+ Slow learning time
+ Scalability is an issue for tasks that are bigger (not complex)
We have not yet understood why this happens, so methods to combat this behavior is yet to be discovered

#### 4.2 Scalability to Complex Environments
CartPole task is a relatively simple experiment, with simpler states but when we scale this to continuous environments, we have the following issues :
+ State encoding : the current methods such as amplitude or angle encoding do not scale as the state variables increase and the circuit complexity grows exponentially
+ Action representation : The quantum models needs new ways to represent the actions that are in multiple ways or in continuous format, the current ways only work for discrete spaces


#### 4.3 Quantum Advantage Proofs
The speedups are all predicted theoretically, without proper and clear experimental proof of the quantum advantage in RL, so the challenges are :
+ Classical RL already are really good in standard tasks
+ Practical quantum devices are noisy which shadows the advantages we get
+ The studies are done in simulators or simple environments like CartPole or FrozenLake, which don't test the quantum advantages througoughly
The future work needs to be done on identifying problems where the quantum method is better than the classical methods by a margin and develop algorithms that can handle noise better

## 5. Proposed Model / Proposed Methodology

#### Overview
We implement a **hybrid quantum–classical Deep Q Network (Q-DQN)** where a parameterized quantum circuit (PQC) — a Quantum Neural Network (QNN) built in Qiskit — replaces the classical function approximator inside a standard DQN pipeline. The QNN receives a classical state encoding, applies trainable gates, and returns expectation-based outputs that a small classical head converts into Q-values for discrete actions.

Key components (as implemented in the notebook):
- **Encoding layer (classical)** — a small trainable preprocessing block (`encoding_layer`) that scales inputs and applies `atan` nonlinearity.  
- **Feature map + Ansatz (quantum)** — a reuploading-style parameterized circuit that alternates repeated encodings (`Rx` per input dimension) with variational layers (rotation gates + `cz` entanglers). Reuploading (multiple insertions of input data between variational layers) increases expressivity.  
- **EstimatorQNN + TorchConnector** — Qiskit’s `EstimatorQNN` primitive is used to evaluate expectation values and is wrapped as a PyTorch module using `TorchConnector`. This lets the QNN sit inside `torch.nn.Sequential` and be trained with classical optimizers.  
- **Postprocessing head (`exp_val_layer`)** — maps the QNN’s measured components (probabilities/expectations) to action Q-values via fixed masks (to compute observables such as ZZ correlators) and small learnable scaling factors.
- **RL algorithm** — standard DQN machinery: epsilon-greedy policy, replay buffer, target estimation via Bellman backup, and gradient-based updates. Because of gradient/compilation issues with Qiskit’s connector, training is performed **sample-wise** (sequential updates) rather than full batched updates.

#### Why this architecture
- PQCs can represent complex nonlinear functions with fewer parameters or different inductive biases than classical networks; reuploading improves expressivity for low-qubit circuits.  
- The hybrid pipeline allows classical optimizers to update quantum parameters via estimated gradients, enabling integration into RL loops without specialized quantum optimizers. `EstimatorQNN` provides a convenient estimator-based interface for expectation evaluation.

#### Mathematical formulation
Let the classical state be $x \in \mathbb{R}^4$. Define the PQC unitary $U_\theta(x)$ consisting of repeated blocks:
- encode$(x)$ via rotations $R_x(x_i)$,
- variational layer $V(\theta)$ (parameterized single-qubit rotations + entanglers).

The Q-value for action $a$ is computed as an expectation:

$$
Q_\theta(s,a) \approx \langle 0| U_\theta(x)^\dagger \, \hat{O}_a \, U_\theta(x) |0\rangle = \langle \psi_\theta(x) | \hat{O}_a | \psi_\theta(x) \rangle
$$

where $|\psi_\theta(x)\rangle = U_\theta(x)|0\rangle$, $x \in \mathbb{R}^4$ is the encoded classical state (after preprocessing through the encoding layer), and $\hat{O}_a$ are Pauli observables: $\hat{O}_0 = Z_0 \otimes Z_1$ and $\hat{O}_1 = Z_2 \otimes Z_3$. The expectation values are computed from measurement probabilities in the computational basis, and the classical head (`exp_val_layer`) linearly rescales these expectation values into final Q-values via learnable weights.

---

## 6. Experimental Details

### Dataset description
- **Environment**: OpenAI Gymnasium `CartPole-v1`.  
  - State space $S$: continuous, 4-dim (cart position, cart velocity, pole angle, pole angular velocity).  
  - Action space $A$: discrete $\{0,1\}$ (push left / push right).  
  - Reward: +1 for every timestep the pole remains upright (episode terminates when pole angle or cart position exceed thresholds or after 500 steps in Gym; notebook uses 200 steps as target).

#### Experimental setup
- **Quantum circuit parameters**
  - `num_qubits = 4`
  - `reuploading = True`
  - `reps = 6`
  - Entangling gates: controlled-Z (`cz`) in circular topology
  - Input encoding: `Rx` rotations per input dimension

- **Neural / RL hyperparameters**
  - Replay buffer: `deque(maxlen=2000)`
  - Batch size: `16` (used for classical batched training; quantum training uses sequential sample-wise updates)
  - Discount factor $\gamma = 0.99$
  - Optimizer: `Adam(model.parameters(), lr=1e-2)`
  - Loss: MSE between target Q and predicted Q
  - Episodes: 2000
  - Training begins after 20 episodes (to populate replay)

- **Quantum backend**
  - `AerEstimator` from Qiskit Aer is used to evaluate expectations and gradients.
  - `EstimatorQNN` wrapped by `TorchConnector` connects the circuit with PyTorch autograd.
  - Calls to the estimator are synchronous and can become a runtime bottleneck.

- **Practical constraints**
  - Because `TorchConnector` + `EstimatorQNN` triggers many estimator jobs per gradient call, the notebook experienced blocking/waiting and eventual `KeyboardInterrupt` during `.backward()`.  
  - This is expected when many gradient evaluations are done serially with large circuits.

#### Packages / tools / libraries used
- **Qiskit** (qiskit, qiskit-aer, qiskit-machine-learning) — circuit construction, EstimatorQNN, TorchConnector.  
- **PyTorch** — training loop, optimizer, autograd, model wrapping.  
- **Gymnasium** — `CartPole-v1` environment.  
- **NumPy, Matplotlib** — data handling and plotting.  
- **Python 3.13** — runtime environment for the notebook.

---

## 7. Results and Discussion

#### Observations
1. **Expressivity vs cost trade-off**  
   Using 4 qubits and 6 reuploading layers gives high expressivity but extreme computational cost due to many estimator evaluations. Training was extremely slow and often blocked waiting for Estimator job results.

2. **Gradient vanishing**  
   Batched training caused gradients to vanish; only sequential per-sample updates worked. This significantly slows training but avoids zero-gradient issues.

3. **Instability in postprocessing**  
   The `exp_val_layer` initialized with large weights (35–40) led to unstable Q-values. Reducing this to near 1 improved numerical stability.

4. **Learning evidence**  
   Partial reward improvement was observed in logged files, though training was interrupted due to performance bottlenecks. The setup demonstrates feasibility but not stable convergence.

#### Significance
- Confirms **integration feasibility** of a quantum model inside a DQN pipeline using Qiskit + PyTorch.  
- Highlights **practical limitations**: expensive gradient computation, synchronous job execution, and fragile gradient dynamics.  
- Suggests that meaningful quantum advantage in reinforcement learning remains an open question at current hardware/simulator scales.

---

## 8. Future Directions

1. **Smaller circuits for debugging**  
   - Reduce qubit count and reuploading reps to speed up early experimentation.  
   - Use statevector or analytic simulators for faster training.

2. **Improved gradient computation**  
   - Use parameter-shift or analytic gradients (if supported) instead of estimator-based numerical gradients.  
   - Try variance reduction and adaptive shot allocation.

3. **Hybrid baselines and comparison**  
   - Build a classical MLP baseline under the same training loop to measure quantum benefits clearly.

4. **Refined observables**  
   - Replace ad-hoc masking in the output layer with a more principled linear readout or learned observable mappings.

5. **Noise handling**  
   - Add error mitigation or run on real hardware with shallow circuits to analyze NISQ-era performance.

6. **Asynchronous estimator usage**  
   - Explore Qiskit Runtime or batched asynchronous primitives to handle large numbers of quantum evaluations.

7. **Expanded benchmarks**  
   - Move beyond CartPole to environments that better test representational capacity (e.g., continuous control tasks, discretized mountain car, or simple gridworlds).

---
## 9. References

1. A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, and J. I. Latorre, “Data re-uploading for a universal quantum classifier,” *Quantum*, vol. 4, p. 226, 2020.
2. K. Mitarai, M. Negoro, M. Kitagawa, and K. Fujii, “Quantum circuit learning,” *Phys. Rev. A*, vol. 98, no. 3, p. 032309, 2018.
3. A. Peruzzo et al., “A variational eigenvalue solver on a photonic quantum processor,” *Nature Communications*, vol. 5, p. 4213, 2014.
4. E. Farhi, J. Goldstone, and S. Gutmann, “A quantum approximate optimization algorithm,” *arXiv:1411.4028*, 2014.
5. Nico Meyer, Christian Ufrecht, Maniraman Periyasamy, Daniel D Scherer, Axel Plinge, Christopher Mutschler, “A survey on quantum reinforcement learning,” *arXiv:2211.03464*, 2022.
6. Qiskit Machine Learning Documentation — EstimatorQNN and TorchConnector, Qiskit Community Tutorials, 2024.