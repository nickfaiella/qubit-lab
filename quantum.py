# description: A small quantum computer simulator written from scratch with NumPy.
#              It tracks the full quantum state, applies gates, simulates measurement,
#              computes Bloch-sphere vectors, and runs Grover's search algorithm.
# author: Nick Faiella (built with help from Claude)
# date: October 14, 2026
#
# Big idea: n qubits are described by 2**n complex numbers called amplitudes.
# The chance of measuring a result is the amplitude's size squared.
# Gates are small matrices that rotate those amplitudes around.
#
# Qubit order: qubit 0 is the LEFTMOST bit of a label, so in "10", qubit 0 is 1.

import math

import numpy as np

SQRT_HALF = 1 / math.sqrt(2)

# ---------- single-qubit gates (2x2 matrices) ----------
GATES = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),          # bit flip (quantum NOT)
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),         # phase flip
    "H": np.array([[1, 1], [1, -1]], dtype=complex) * SQRT_HALF,  # makes superposition
    "S": np.array([[1, 0], [0, 1j]], dtype=complex),         # quarter-turn of phase
    "T": np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex),  # eighth-turn
}

GATE_INFO = {
    "H": "Hadamard: puts a qubit into an equal superposition of 0 and 1",
    "X": "Pauli-X: flips 0 and 1, like a classical NOT",
    "Y": "Pauli-Y: flips the bit and rotates its phase",
    "Z": "Pauli-Z: flips the phase of 1 without changing probabilities",
    "S": "S: rotates the phase of 1 by 90°",
    "T": "T: rotates the phase of 1 by 45°",
    "RX": "Rotation around the X axis by any angle",
    "RY": "Rotation around the Y axis by any angle",
    "RZ": "Rotation around the Z axis by any angle",
    "CNOT": "Controlled-NOT: flips the target only when the control is 1 (creates entanglement)",
    "CZ": "Controlled-Z: flips the phase only when both qubits are 1",
    "SWAP": "Swaps the states of two qubits",
}

TWO_QUBIT_GATES = ["CNOT", "CZ", "SWAP"]
ROTATION_GATES = ["RX", "RY", "RZ"]


def rotation(axis, theta):
    """Return the matrix that rotates a qubit by angle theta (radians) around an axis."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    if axis == "X":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if axis == "Y":
        return np.array([[c, -s], [s, c]], dtype=complex)
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)


class QuantumCircuit:
    def __init__(self, n_qubits):
        if not 1 <= n_qubits <= 10:
            raise ValueError("Use between 1 and 10 qubits.")
        self.n = n_qubits
        self.state = np.zeros(2 ** n_qubits, dtype=complex)
        self.state[0] = 1.0  # every qubit starts as 0
        self.steps = []      # a record of the gates applied, used to draw the circuit

    # ----- applying gates -----
    def _apply_matrix(self, matrix, target):
        """Apply a 2x2 matrix to one qubit.

        Trick: reshape the 2**n list into an n-dimensional grid (2 x 2 x ... x 2).
        Each axis of the grid is one qubit, so we only multiply along that axis.
        """
        grid = self.state.reshape([2] * self.n)
        grid = np.tensordot(matrix, grid, axes=([1], [target]))
        grid = np.moveaxis(grid, 0, target)
        self.state = grid.reshape(-1)

    def _apply_controlled(self, matrix, control, target):
        """Apply a matrix to the target only in the part of the state where control is 1."""
        if control == target:
            raise ValueError("Control and target must be different qubits.")
        grid = self.state.reshape([2] * self.n).copy()
        where_control_is_1 = [slice(None)] * self.n
        where_control_is_1[control] = 1
        piece = grid[tuple(where_control_is_1)]           # this slice has one fewer axis
        axis = target if target < control else target - 1  # so the target's axis shifts
        piece = np.tensordot(matrix, piece, axes=([1], [axis]))
        piece = np.moveaxis(piece, 0, axis)
        grid[tuple(where_control_is_1)] = piece
        self.state = grid.reshape(-1)

    def gate(self, name, target, control=None, angle=None):
        """Apply a gate by name, e.g. gate("H", 0) or gate("CNOT", 1, control=0)."""
        self._check(target)
        if name in GATES:
            self._apply_matrix(GATES[name], target)
        elif name in ROTATION_GATES:
            self._apply_matrix(rotation(name[1], angle), target)
        elif name == "CNOT":
            self._check(control)
            self._apply_controlled(GATES["X"], control, target)
        elif name == "CZ":
            self._check(control)
            self._apply_controlled(GATES["Z"], control, target)
        elif name == "SWAP":
            self._check(control)
            # a SWAP is three CNOTs in a row
            self._apply_controlled(GATES["X"], control, target)
            self._apply_controlled(GATES["X"], target, control)
            self._apply_controlled(GATES["X"], control, target)
        else:
            raise ValueError(f"Unknown gate: {name}")
        self.steps.append({"gate": name, "target": target, "control": control, "angle": angle})
        return self

    def _check(self, qubit):
        if qubit is None or not 0 <= qubit < self.n:
            raise ValueError(f"Qubit must be between 0 and {self.n - 1}.")

    # ----- reading results -----
    def labels(self):
        """Basis-state names like '00', '01', '10', '11'."""
        return [format(i, f"0{self.n}b") for i in range(2 ** self.n)]

    def probabilities(self):
        return np.abs(self.state) ** 2

    def measure(self, shots=1000, seed=None):
        """Simulate measuring the circuit many times. Returns {label: count}."""
        rng = np.random.default_rng(seed)
        probs = self.probabilities()
        probs = probs / probs.sum()  # guard against tiny rounding errors
        results = rng.choice(2 ** self.n, size=shots, p=probs)
        counts = np.bincount(results, minlength=2 ** self.n)
        return {label: int(count) for label, count in zip(self.labels(), counts)}

    def ket_string(self, tolerance=1e-9, digits=3):
        """Write the state in Dirac notation, like 0.707|00⟩ + 0.707|11⟩."""
        parts = []
        for label, amp in zip(self.labels(), self.state):
            if abs(amp) < tolerance:
                continue
            real, imag = round(amp.real, digits), round(amp.imag, digits)
            if imag == 0:
                text = f"{real:g}"
            elif real == 0:
                text = f"{imag:g}i"
            else:
                text = f"({real:g}{imag:+g}i)"
            parts.append(f"{text}|{label}⟩")
        return " + ".join(parts).replace("+ -", "− ")

    def bloch_vector(self, qubit):
        """Return (x, y, z) for one qubit's point on the Bloch sphere.

        For an entangled qubit, the point moves INSIDE the sphere: the qubit has
        no definite state of its own, only a shared state with its partner.
        """
        grid = np.moveaxis(self.state.reshape([2] * self.n), qubit, 0).reshape(2, -1)
        rho = grid @ grid.conj().T  # this qubit's "reduced density matrix"
        x = 2 * rho[0, 1].real
        y = -2 * rho[0, 1].imag
        z = (rho[0, 0] - rho[1, 1]).real
        return float(x), float(y), float(z)


# ---------- Grover's search ----------

def grover_optimal_iterations(n_qubits):
    """About (π/4)·√N rounds gives the best chance of finding the answer."""
    return max(1, math.floor(math.pi / 4 * math.sqrt(2 ** n_qubits)))


def grover_search(n_qubits, marked, iterations):
    """Run Grover's algorithm to find one marked item among 2**n.

    Returns (final probabilities, chance of the marked item after each round).

    Each round has two steps:
      1. Oracle: flip the sign of the marked item's amplitude.
      2. Diffusion: reflect every amplitude around the average. The marked item,
         now below the average, gets pushed up; everything else shrinks a little.
    """
    size = 2 ** n_qubits
    if not 0 <= marked < size:
        raise ValueError("Marked item is out of range.")
    state = np.full(size, 1 / math.sqrt(size))  # Hadamard on every qubit
    history = [float(state[marked] ** 2)]
    for _ in range(iterations):
        state[marked] *= -1                  # 1. oracle
        state = 2 * state.mean() - state     # 2. diffusion (inversion about the mean)
        history.append(float(state[marked] ** 2))
    return state ** 2, history
