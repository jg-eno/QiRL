"""Quantum circuit definitions for QRL implementation."""

import qiskit as qk
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import TwoLocal


def encoding_circuit(inputs, num_qubits=4):
    """
    Create an encoding circuit that maps classical information onto a quantum register.
    
    Args:
        inputs: Input values to encode (can be ParameterVector or actual values)
        num_qubits: Number of qubits in the circuit
        
    Returns:
        QuantumCircuit: Circuit with RX gates encoding the inputs
    """
    qc = QuantumCircuit(num_qubits)
    for i in range(len(inputs)):
        qc.rx(inputs[i], i)
    return qc


def parametrized_circuit(num_qubits=4, reuploading=False, reps=2, insert_barriers=True, meas=False):
    """
    Create a parameterized quantum circuit for Q-value estimation.
    
    Args:
        num_qubits: Number of qubits in the circuit
        reuploading: Whether to use reuploading strategy (re-encode inputs at each layer)
        reps: Number of repetition layers
        insert_barriers: Whether to insert barriers between layers
        meas: Whether to add measurement gates
        
    Returns:
        QuantumCircuit: Parameterized quantum circuit
    """
    qr = qk.QuantumRegister(num_qubits, 'qr')
    qc = qk.QuantumCircuit(qr)
    
    if meas:
        qr = qk.QuantumRegister(num_qubits, 'qr')
        cr = qk.ClassicalRegister(num_qubits, 'cr')
        qc = qk.QuantumCircuit(qr, cr)
    
    if not reuploading:
        inputs = qk.circuit.ParameterVector('x', num_qubits)
        
        qc.compose(encoding_circuit(inputs, num_qubits=num_qubits), inplace=True)
        if insert_barriers:
            qc.barrier()
        
        qc.compose(TwoLocal(num_qubits, ['ry', 'rz'], 'cz', 'circular',
                           reps=reps, insert_barriers=insert_barriers,
                           skip_final_rotation_layer=True), inplace=True)
        if insert_barriers:
            qc.barrier()
        
        if meas:
            qc.measure(qr, cr)
    
    elif reuploading:
        inputs = qk.circuit.ParameterVector('x', num_qubits)
        θ = qk.circuit.ParameterVector('θ', 2 * num_qubits * reps)
        
        for rep in range(reps):
            qc.compose(encoding_circuit(inputs, num_qubits=num_qubits), inplace=True)
            if insert_barriers:
                qc.barrier()
            
            for qubit in range(num_qubits):
                qc.ry(θ[qubit + 2 * num_qubits * rep], qubit)
                qc.rz(θ[qubit + 2 * num_qubits * rep + num_qubits], qubit)
            if insert_barriers:
                qc.barrier()
            
            qc.cz(qr[-1], qr[0])
            for qubit in range(num_qubits - 1):
                qc.cz(qr[qubit], qr[qubit + 1])
            if insert_barriers:
                qc.barrier()
        
        if meas:
            qc.measure(qr, cr)
    
    return qc

