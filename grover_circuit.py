"""Grover search circuit for the SAT instance in sat_problem.py."""
import math

import cirq

from sat_problem import CLAUSES, VARIABLES, brute_force_solutions


def optimal_iterations(num_vars, num_solutions):
    """Grover iteration count maximizing success probability: round(π/(4θ) - 1/2)."""
    theta = math.asin(math.sqrt(num_solutions / 2**num_vars))
    return max(0, round(math.pi / (4 * theta) - 0.5))


def success_probability(num_vars, num_solutions, iterations):
    theta = math.asin(math.sqrt(num_solutions / 2**num_vars))
    return math.sin((2 * iterations + 1) * theta) ** 2


def _clause_compute(clause, var_qubits, anc):
    """Set `anc` (initially |0>) to 1 iff the clause is satisfied.

    A clause is false iff every literal is false. Flip the positive-literal
    qubits so "literal false" == |1>, MCX onto anc, then X so anc == "satisfied".
    """
    pos = [var_qubits[v] for v, positive in clause if positive]
    controls = [var_qubits[v] for v, _ in clause]
    yield cirq.X.on_each(*pos)
    yield cirq.X(anc).controlled_by(*controls)
    yield cirq.X.on_each(*pos)
    yield cirq.X(anc)


def phase_oracle(var_qubits, clause_ancillas):
    """Flip the phase of basis states that satisfy every clause."""
    compute = [op for clause, anc in zip(CLAUSES, clause_ancillas)
               for op in _clause_compute(clause, var_qubits, anc)]
    yield compute
    yield cirq.Z(clause_ancillas[-1]).controlled_by(*clause_ancillas[:-1])
    yield cirq.inverse(compute)


def diffuser(var_qubits):
    """Inversion about the mean on the variable register."""
    yield cirq.H.on_each(*var_qubits)
    yield cirq.X.on_each(*var_qubits)
    yield cirq.Z(var_qubits[-1]).controlled_by(*var_qubits[:-1])
    yield cirq.X.on_each(*var_qubits)
    yield cirq.H.on_each(*var_qubits)


def build_circuit(iterations=None, measure=True):
    """Returns (circuit, var_qubits, iterations)."""
    var_qubits = {v: cirq.NamedQubit(v) for v in VARIABLES}
    clause_ancillas = [cirq.NamedQubit(f"c{i}") for i in range(len(CLAUSES))]
    data = [var_qubits[v] for v in VARIABLES]

    if iterations is None:
        iterations = optimal_iterations(len(VARIABLES), len(brute_force_solutions()))

    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(*data))
    for _ in range(iterations):
        circuit.append(phase_oracle(var_qubits, clause_ancillas))
        circuit.append(diffuser(data))
    if measure:
        circuit.append(cirq.measure(*data, key="result"))
    return circuit, var_qubits, iterations
