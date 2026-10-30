"""Run Grover on the SAT instance; save a histogram + raw counts per run."""
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import cirq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grover_circuit import build_circuit, success_probability
from sat_problem import VARIABLES, brute_force_solutions, bits_to_set, set_label


def run_once(simulator, circuit, repetitions):
    result = simulator.run(circuit, repetitions=repetitions)
    counts = Counter(tuple(int(b) for b in row) for row in result.measurements["result"])
    return counts


def save_histogram(counts, repetitions, solutions, iterations, path):
    import itertools
    states = list(itertools.product([0, 1], repeat=len(VARIABLES)))
    sol_labels = {set_label(s) for s in solutions}
    labels = [set_label(bits_to_set(s)) for s in states]
    values = [counts.get(s, 0) for s in states]
    colors = ["tab:green" if l in sol_labels else "tab:gray" for l in labels]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(states)), values, color=colors)
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("counts")
    ax.set_title(f"Grover SAT: {iterations} iterations, {repetitions} shots "
                 f"(green = satisfying assignment)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Run Grover search on the SAT instance.")
    p.add_argument("--runs", type=int, default=1, help="independent runs, each saving its own histogram (default 1)")
    p.add_argument("--repetitions", type=int, default=1000, help="shots per run (default 1000)")
    p.add_argument("--iterations", type=int, default=None, help="Grover iterations (default: optimal)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = p.parse_args()

    solutions = brute_force_solutions()
    circuit, _, iterations = build_circuit(args.iterations)
    expected = success_probability(len(VARIABLES), len(solutions), iterations)
    print(f"{len(solutions)} solutions / 64 states; {iterations} iterations; "
          f"predicted success probability {expected:.3f}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(args.runs):
        seed = None if args.seed is None else args.seed + i
        counts = run_once(cirq.Simulator(seed=seed), circuit, args.repetitions)
        hits = sum(c for s, c in counts.items() if bits_to_set(s) in solutions)
        stem = args.outdir / f"grover_{stamp}_run{i + 1}"
        save_histogram(counts, args.repetitions, solutions, iterations, stem.with_suffix(".png"))
        stem.with_suffix(".json").write_text(json.dumps({
            "iterations": iterations, "repetitions": args.repetitions, "seed": seed,
            "counts": {set_label(bits_to_set(s)): c for s, c in sorted(counts.items())},
        }, indent=2))
        print(f"run {i + 1}: {hits}/{args.repetitions} satisfying ({hits / args.repetitions:.3f}) -> {stem}.png")


if __name__ == "__main__":
    main()
