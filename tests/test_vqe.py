import numpy as np
import pytest

import cirq

from vqe_maxcut import (LAYERS, NUM_QUBITS, WEIGHTS, brute_force, cut_value, exact_ground_energy,
                        expectation_value, hamiltonian, num_params, qubits, run_vqe,
                        variational_circuit)


def test_brute_force_ground_state():
    e0, best_cut, sols = brute_force()
    assert best_cut == pytest.approx(5.7)
    assert e0 == pytest.approx(-4.4)
    assert set(sols) == {(0, 1, 0, 1), (1, 0, 1, 0)}


def test_exact_eigenvalue_matches_brute_force():
    assert exact_ground_energy() == pytest.approx(brute_force()[0])


def test_energy_is_total_weight_minus_twice_cut():
    for idx in range(2**NUM_QUBITS):
        bits = tuple((idx >> (NUM_QUBITS - 1 - k)) & 1 for k in range(NUM_QUBITS))
        state = np.zeros(2**NUM_QUBITS, dtype=np.complex64)
        state[idx] = 1
        e = hamiltonian.expectation_from_state_vector(state, {q: i for i, q in enumerate(qubits)}).real
        assert e == pytest.approx(sum(WEIGHTS.values()) - 2 * cut_value(bits), abs=1e-5)


def test_circuit_has_rotations_and_entangler():
    circuit = variational_circuit(np.zeros(num_params()))
    gates = [op.gate for op in circuit.all_operations()]
    assert any(isinstance(g, cirq.YPowGate) for g in gates)
    assert any(isinstance(g, cirq.ZPowGate) for g in gates)
    assert any(g == cirq.CNOT for g in gates)


def test_zero_params_give_all_zeros_energy():
    # |0000> has every Z_i Z_j = +1, so E = sum of weights.
    assert expectation_value(np.zeros(num_params())) == pytest.approx(sum(WEIGHTS.values()), abs=1e-5)


def test_expectation_value_returns_real_float():
    assert isinstance(expectation_value(np.random.default_rng(1).uniform(0, 6, num_params())), float)


def test_vqe_finds_max_cut():
    e0, best_cut, _ = brute_force()
    result = run_vqe(layers=LAYERS, restarts=5, seed=0)
    assert result["energy"] == pytest.approx(e0, abs=1e-2)
    assert result["cut"] == pytest.approx(best_cut)
    assert result["energy"] >= e0 - 1e-6  # variational bound
    assert len(result["history"]) > 1


def test_hamiltonian_has_at_least_six_terms_three_multi_qubit():
    terms = list(hamiltonian)
    assert len(terms) >= 6
    assert sum(len(t.qubits) >= 2 for t in terms) >= 3
    assert all(set(t.values()) <= {cirq.X, cirq.Y, cirq.Z} for t in terms)
