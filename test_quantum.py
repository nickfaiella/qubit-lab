# Tests that the simulator obeys real quantum physics. Run with:  pytest

import math

import numpy as np
import pytest

from quantum import QuantumCircuit, grover_search, grover_optimal_iterations


def close(a, b, tol=1e-9):
    return np.allclose(a, b, atol=tol)


def test_starts_in_all_zeros():
    qc = QuantumCircuit(3)
    assert close(qc.probabilities(), [1, 0, 0, 0, 0, 0, 0, 0])


def test_x_flips_the_right_qubit():
    qc = QuantumCircuit(2).gate("X", 0)
    assert qc.labels()[np.argmax(qc.probabilities())] == "10"  # qubit 0 is the left bit


def test_hadamard_makes_a_fair_coin():
    qc = QuantumCircuit(1).gate("H", 0)
    assert close(qc.probabilities(), [0.5, 0.5])


def test_hadamard_twice_undoes_itself():
    qc = QuantumCircuit(1).gate("H", 0).gate("H", 0)
    assert close(qc.state, [1, 0])  # interference cancels the |1⟩ path


def test_bell_state():
    qc = QuantumCircuit(2).gate("H", 0).gate("CNOT", 1, control=0)
    assert close(qc.state, [math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    assert qc.ket_string() == "0.707|00⟩ + 0.707|11⟩"


def test_ghz_state_three_qubits():
    qc = QuantumCircuit(3).gate("H", 0).gate("CNOT", 1, control=0).gate("CNOT", 2, control=1)
    probs = dict(zip(qc.labels(), qc.probabilities()))
    assert math.isclose(probs["000"], 0.5) and math.isclose(probs["111"], 0.5)


def test_cnot_with_control_below_target():
    qc = QuantumCircuit(3).gate("X", 2).gate("CNOT", 0, control=2)
    assert qc.labels()[np.argmax(qc.probabilities())] == "101"


def test_swap():
    qc = QuantumCircuit(2).gate("X", 0).gate("SWAP", 1, control=0)
    assert qc.labels()[np.argmax(qc.probabilities())] == "01"


def test_rotation_ry_pi_flips():
    qc = QuantumCircuit(1).gate("RY", 0, angle=math.pi)
    assert close(qc.probabilities(), [0, 1])


def test_state_stays_normalized():
    qc = QuantumCircuit(4)
    for name, target in [("H", 0), ("T", 0), ("RX", 2), ("Y", 3), ("S", 1)]:
        qc.gate(name, target, angle=0.7)
    qc.gate("CNOT", 3, control=0).gate("CZ", 1, control=2)
    assert math.isclose(qc.probabilities().sum(), 1.0)


def test_bloch_vectors():
    assert close(QuantumCircuit(1).bloch_vector(0), (0, 0, 1))                  # |0⟩ at the top
    assert close(QuantumCircuit(1).gate("H", 0).bloch_vector(0), (1, 0, 0))     # |+⟩ on the equator
    assert close(QuantumCircuit(1).gate("H", 0).gate("S", 0).bloch_vector(0), (0, 1, 0))
    bell = QuantumCircuit(2).gate("H", 0).gate("CNOT", 1, control=0)
    assert close(bell.bloch_vector(0), (0, 0, 0))  # entangled: no state of its own


def test_measurement_matches_probabilities():
    qc = QuantumCircuit(2).gate("H", 0).gate("CNOT", 1, control=0)
    counts = qc.measure(shots=10000, seed=1)
    assert counts["01"] == 0 and counts["10"] == 0
    assert 4700 < counts["00"] < 5300


def test_grover_finds_the_marked_item():
    for n in range(2, 7):
        rounds = grover_optimal_iterations(n)
        probs, history = grover_search(n, marked=3, iterations=rounds)
        assert np.argmax(probs) == 3
        assert history[-1] > 0.9
        assert math.isclose(probs.sum(), 1.0)


def test_bad_qubit_index():
    with pytest.raises(ValueError):
        QuantumCircuit(2).gate("H", 5)
