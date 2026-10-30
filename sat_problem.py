"""The SAT instance: variables, CNF clauses, and a brute-force reference solver."""
from itertools import product

VARIABLES = ["a", "d", "f", "m", "n", "s"]

# Each clause is a list of (variable, is_positive) literals.
CLAUSES = [
    [("s", False), ("d", False)],               # (¬s ∨ ¬d)
    [("m", False), ("a", True)],                # (¬m ∨ a)
    [("f", False), ("a", True), ("n", True)],   # (¬f ∨ a ∨ n)
    [("a", False), ("n", False)],               # (¬a ∨ ¬n)
    [("m", False), ("f", False)],               # (¬m ∨ ¬f)
    [("a", False), ("f", False), ("s", False)], # (¬a ∨ ¬f ∨ ¬s)
    [("m", False), ("d", False)],               # (¬m ∨ ¬d)
    [("m", True), ("f", True)],                 # (m ∨ f)
    [("s", True), ("d", True)],                 # (s ∨ d)
]

EXPECTED_SOLUTIONS = [{"n", "f", "d"}, {"n", "f", "s"}, {"a", "f", "d"}, {"a", "m", "s"}]


def satisfies(assignment):
    """assignment: dict var -> bool."""
    return all(any(assignment[v] == pos for v, pos in clause) for clause in CLAUSES)


def brute_force_solutions():
    """All satisfying assignments, as sets of the variables that are True."""
    sols = []
    for bits in product([False, True], repeat=len(VARIABLES)):
        assignment = dict(zip(VARIABLES, bits))
        if satisfies(assignment):
            sols.append({v for v, b in assignment.items() if b})
    return sols


def bits_to_set(bits):
    """Measurement bits (ordered as VARIABLES) -> set of True variables."""
    return {v for v, b in zip(VARIABLES, bits) if b}


def set_label(s):
    return "{" + ",".join(v for v in VARIABLES if v in s) + "}"
