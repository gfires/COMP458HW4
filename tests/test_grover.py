import numpy as np

import cirq

from grover_circuit import build_circuit, optimal_iterations
from sat_problem import EXPECTED_SOLUTIONS, VARIABLES, bits_to_set, brute_force_solutions


def test_brute_force_matches_expected_solutions():
    sols = brute_force_solutions()
    assert len(sols) == 4
    assert all(s in sols for s in EXPECTED_SOLUTIONS)


def test_optimal_iterations_is_three():
    assert optimal_iterations(6, 4) == 3


def test_exact_distribution_concentrates_on_solutions():
    circuit, _, _ = build_circuit(measure=False)
    data = [cirq.NamedQubit(v) for v in VARIABLES]
    ancillas = sorted(circuit.all_qubits() - set(data))
    state = cirq.Simulator().simulate(circuit, qubit_order=data + ancillas).final_state_vector
    # Data qubits are the high-order bits; ancillas are uncomputed back to |0>.
    probs = (np.abs(state) ** 2).reshape(2 ** len(data), -1).sum(axis=1)
    solution_prob = sum(
        p for i, p in enumerate(probs)
        if bits_to_set([(i >> (len(data) - 1 - k)) & 1 for k in range(len(data))]) in EXPECTED_SOLUTIONS
    )
    assert solution_prob > 0.95
