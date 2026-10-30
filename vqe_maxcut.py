"""VQE for H = sum w_ij Z_i Z_j; its ground state is the MaxCut optimum."""
import itertools
import os

import cirq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

NUM_QUBITS = 4
LAYERS = 1
WEIGHTS = {(0, 1): 1.0, (0, 2): 0.5, (0, 3): 2.0, (1, 2): 1.5, (1, 3): 0.8, (2, 3): 1.2}
qubits = cirq.LineQubit.range(NUM_QUBITS)

# Simulator
simulator = cirq.Simulator(dtype=np.complex128)

# Hamiltonian: H = sum w_ij Z_i Z_j
hamiltonian = cirq.PauliSum.from_pauli_strings(
    [w * cirq.Z(qubits[i]) * cirq.Z(qubits[j]) for (i, j), w in WEIGHTS.items()])


def num_params(layers=LAYERS):
    return (layers + 1) * NUM_QUBITS * 2


# Define parameterized quantum circuit
def variational_circuit(params, layers=LAYERS):
    """Ry/Rz on every qubit, then a CNOT chain, repeated; ends with a rotation layer."""
    params = np.asarray(params).reshape(layers + 1, NUM_QUBITS, 2)  # numbers or sympy symbols
    circuit = cirq.Circuit()
    for layer in range(layers + 1):
        circuit.append(cirq.ry(params[layer, q, 0]).on(qubits[q]) for q in range(NUM_QUBITS))
        circuit.append(cirq.rz(params[layer, q, 1]).on(qubits[q]) for q in range(NUM_QUBITS))
        if layer < layers:
            circuit.append(cirq.CNOT(qubits[q], qubits[q + 1]) for q in range(NUM_QUBITS - 1))
    return circuit


def expectation_value(params, layers=LAYERS):
    circuit = variational_circuit(params, layers)
    result = simulator.simulate(circuit, qubit_order=qubits)
    state_vector = result.final_state_vector

    # Compute expectation value using Cirq's expectation_from_state_vector
    energy = np.real(hamiltonian.expectation_from_state_vector(
        state_vector, qubit_map={q: i for i, q in enumerate(qubits)}))
    return float(energy)


def cut_value(bits):
    return sum(w for (i, j), w in WEIGHTS.items() if bits[i] != bits[j])


def brute_force():
    """Return (ground energy, optimal cut weight, optimal bitstrings)."""
    all_bits = list(itertools.product((0, 1), repeat=NUM_QUBITS))
    best = max(cut_value(b) for b in all_bits)
    sols = [b for b in all_bits if np.isclose(cut_value(b), best)]
    return sum(WEIGHTS.values()) - 2 * best, best, sols


def exact_ground_energy():
    """Smallest eigenvalue of the Hamiltonian matrix."""
    return float(np.linalg.eigvalsh(hamiltonian.matrix(qubits)).min())


def run_vqe(layers=LAYERS, restarts=5, seed=0, maxiter=2000):
    """Minimize the energy from several random starts; return the best result dict.

    `history` is the energy at every evaluation for the best restart.
    """
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        history = []

        def objective(p):
            history.append(expectation_value(p, layers))
            return history[-1]

        res = minimize(objective, rng.uniform(0, 2 * np.pi, num_params(layers)),
                       method="COBYLA", options={"maxiter": maxiter})
        if best is None or res.fun < best["energy"]:
            best = {"energy": float(res.fun), "params": res.x, "history": history}
    circuit = variational_circuit(best["params"], layers)
    probs = np.abs(simulator.simulate(circuit, qubit_order=qubits).final_state_vector) ** 2
    idx = int(np.argmax(probs))
    bits = tuple((idx >> (NUM_QUBITS - 1 - k)) & 1 for k in range(NUM_QUBITS))
    best.update(layers=layers, circuit=circuit, bitstring=bits, cut=cut_value(bits),
                bit_probability=float(probs[idx]))
    return best


def sweep_depth(max_layers=4, tol=1e-3, **kwargs):
    """Shallowest depth whose energy is within `tol` of the exact ground energy."""
    e0 = exact_ground_energy()
    results = []
    for layers in range(1, max_layers + 1):
        results.append(run_vqe(layers=layers, **kwargs))
        if results[-1]["energy"] <= e0 + tol:
            break
    return results


def plot_convergence(history, exact, path):
    best_so_far = np.minimum.accumulate(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(best_so_far, color="tab:orange", linewidth=3, label="best so far")
    ax.plot(history, color="tab:blue", linewidth=1, label="energy per evaluation")
    ax.axhline(exact, color="r", linestyle="--", label=f"exact ground energy ({exact:.4f})")
    ax.set_xlabel("function evaluation")
    ax.set_ylabel("<H>")
    ax.set_title("VQE energy convergence")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    exact = exact_ground_energy()
    _, best_cut, sols = brute_force()
    print(f"Exact ground energy {exact:.4f}, max cut {best_cut:.2f}, optimal bitstrings {sols}")
    for r in sweep_depth():
        print(f"layers={r['layers']}  E={r['energy']:.4f}  "
              f"bitstring={''.join(map(str, r['bitstring']))} (p={r['bit_probability']:.3f})  "
              f"cut={r['cut']:.2f}")
    print(r["circuit"])
    print(f"\nEvaluations (best restart): {len(r['history'])}")
    print(f"Final optimized energy: {r['energy']:.8f}")
    print(f"Exact eigenvalue:       {exact:.8f}")
    print(f"E_VQE - E_exact:        {r['energy'] - exact:+.2e}")
    print(f"Absolute error:         {abs(r['energy'] - exact):.2e}")
    plot_convergence(r["history"], exact, "outputs/vqe_convergence.png")
    print("Saved outputs/vqe_convergence.png")
