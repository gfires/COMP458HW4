"""Standalone circuit-diagram utility for the Grover and VQE circuits (does not simulate)."""
import argparse
from pathlib import Path

import cirq
import numpy as np
import sympy

from grover_circuit import build_circuit
from vqe_maxcut import LAYERS, NUM_QUBITS, variational_circuit


def build_vqe_circuit(layers):
    """Ansatz with symbolic angles: theta<layer>_<qubit> for Ry, phi<layer>_<qubit> for Rz."""
    symbols = np.empty((layers + 1, NUM_QUBITS, 2), dtype=object)
    for layer in range(layers + 1):
        for q in range(NUM_QUBITS):
            symbols[layer, q] = [sympy.Symbol(f"θ{layer}_{q}"), sympy.Symbol(f"φ{layer}_{q}")]
    return variational_circuit(symbols, layers)


def text_sections(circuit, sections):
    """ASCII diagram split by moments into stacked, equally sized sections."""
    qubits = sorted(circuit.all_qubits())
    n = len(circuit)
    size = -(-n // sections)  # ceil
    parts = []
    for i in range(0, n, size):
        sub = cirq.Circuit(circuit[i:i + size])
        parts.append(f"=== section {len(parts) + 1}/{sections} (moments {i}-{min(i + size, n) - 1}) ===\n"
                     + sub.to_text_diagram(qubit_order=qubits))
    return "\n\n".join(parts)


def main():
    p = argparse.ArgumentParser(description="Draw the Grover SAT or VQE MaxCut circuit.")
    p.add_argument("--circuit", choices=["grover", "vqe"], default="grover",
                   help="which circuit to draw (default: grover)")
    p.add_argument("--layers", type=int, default=LAYERS, help="VQE ansatz layers (default: %d)" % LAYERS)
    p.add_argument("--iterations", type=int, default=None, help="Grover iterations (default: optimal)")
    p.add_argument("--out", type=Path, default=None,
                   help="output file; .svg, or .txt for ASCII (default: outputs/circuit.svg, "
                        "or outputs/vqe_circuit.svg with --circuit vqe)")
    p.add_argument("--single-iteration", action="store_true",
                   help="draw one iteration only (much more readable)")
    p.add_argument("--sections", type=int, default=None,
                   help="for .txt output, split the diagram into this many stacked sections "
                        "(default 3 for grover, 1 for vqe)")
    args = p.parse_args()

    if args.out is None:
        args.out = Path("outputs/vqe_circuit.svg" if args.circuit == "vqe" else "outputs/circuit.svg")
    if args.circuit == "vqe":
        circuit = build_vqe_circuit(args.layers)
        label = f"{args.layers} layer(s)"
    else:
        iterations = 1 if args.single_iteration else args.iterations
        circuit, _, iterations = build_circuit(iterations)
        label = f"{iterations} iteration(s)"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.out.suffix == ".svg":
        from cirq.contrib.svg import SVGCircuit
        args.out.write_text(SVGCircuit(circuit)._repr_svg_())
    else:
        args.out.write_text(text_sections(circuit, args.sections or (1 if args.circuit == "vqe" else 3)))
    print(f"wrote {args.out} ({label}, {len(circuit.all_qubits())} qubits)")


if __name__ == "__main__":
    main()
