"""Independent implementation of relative H_1(K, K[S]) for directed flag
complexes that shares no chain code with the package.

Checks, on random DAGs:
  1. repo GF2RelativeHomologyEvaluator.beta1 == independent GF(2) beta1
  2. GF(2) beta1 vs rational beta1 (field sensitivity of the *values*)
  3. g = q - m + h - t and the bounds -m <= g <= q, 0<=h<=m, 0<=t<=q
"""
from __future__ import annotations

import itertools
import random
import sys
from fractions import Fraction

import networkx as nx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from relhom.topology import GF2RelativeHomologyEvaluator  # noqa: E402


def random_dag(n: int, p: float, rng: random.Random) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    order = list(range(n))
    rng.shuffle(order)
    for i, j in itertools.combinations(range(n), 2):
        if rng.random() < p:
            g.add_edge(order[i], order[j])
    return g


def simplices(g: nx.DiGraph):
    v = sorted(g.nodes)
    e = sorted(g.edges)
    t = []
    for a, b in e:
        for c in g.successors(b):
            if g.has_edge(a, c):
                t.append((a, b, c))
    return v, e, sorted(t)


def rank_gf2(rows: list[list[int]]) -> int:
    m = [r[:] for r in rows if any(r)]
    rank, ncol = 0, (len(m[0]) if m else 0)
    for col in range(ncol):
        piv = next((i for i in range(rank, len(m)) if m[i][col] % 2), None)
        if piv is None:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        for i in range(len(m)):
            if i != rank and m[i][col] % 2:
                m[i] = [(x + y) % 2 for x, y in zip(m[i], m[rank])]
        rank += 1
    return rank


def rank_q(rows: list[list[int]]) -> int:
    m = [[Fraction(x) for x in r] for r in rows if any(r)]
    rank, ncol = 0, (len(m[0]) if m else 0)
    for col in range(ncol):
        piv = next((i for i in range(rank, len(m)) if m[i][col] != 0), None)
        if piv is None:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        for i in range(len(m)):
            if i != rank and m[i][col] != 0:
                f = m[i][col] / m[rank][col]
                m[i] = [x - f * y for x, y in zip(m[i], m[rank])]
        rank += 1
    return rank


def relative_betti(g, S, rank):
    v, e, t = simplices(g)
    S = set(S)
    rv = [x for x in v if x not in S]
    re_ = [x for x in e if not (x[0] in S and x[1] in S)]
    rt = [x for x in t if not set(x) <= S]
    vi = {x: i for i, x in enumerate(rv)}
    ei = {x: i for i, x in enumerate(re_)}
    d1 = []
    for a, b in re_:
        row = [0] * len(rv)
        if b in vi:
            row[vi[b]] += 1
        if a in vi:
            row[vi[a]] -= 1
        d1.append(row)
    d2 = []
    for a, b, c in rt:
        row = [0] * len(re_)
        for edge, sign in (((b, c), 1), ((a, c), -1), ((a, b), 1)):
            if edge in ei:
                row[ei[edge]] += sign
        d2.append(row)
    r1 = rank(d1) if d1 and rv else 0
    r2 = rank(d2) if d2 and re_ else 0
    return len(rv) - r1, len(re_) - r1 - r2, r1, r2, len(rv), len(re_)


def main():
    rng = random.Random(20261011)
    checks = field_diff = bound_viol = decomp_viol = 0
    for trial in range(400):
        n = rng.randint(4, 11)
        p = rng.choice([0.15, 0.3, 0.5, 0.7])
        g = random_dag(n, p, rng)
        if g.number_of_edges() == 0:
            continue
        ev = GF2RelativeHomologyEvaluator(g)
        nodes = list(g.nodes)
        for _ in range(6):
            S = set(rng.sample(nodes, rng.randint(0, n - 1)))
            rest = [x for x in nodes if x not in S]
            C = set(rng.sample(rest, rng.randint(1, len(rest))))
            b0A, b1A, r1A, r2A, vA, eA = relative_betti(g, S, rank_gf2)
            b0B, b1B, r1B, r2B, vB, eB = relative_betti(g, S | C, rank_gf2)
            assert ev.compute_relative_betti(S) == b1A, (trial, "repo vs indep A")
            assert ev.compute_relative_betti(S | C) == b1B, (trial, "repo vs indep B")
            qA = relative_betti(g, S, rank_q)[1]
            qB = relative_betti(g, S | C, rank_q)[1]
            if qA != b1A or qB != b1B:
                field_diff += 1
            gg, m, q = b1A - b1B, vA - vB, eA - eB
            h, t = b0A - b0B, r2A - r2B
            if gg != q - m + h - t:
                decomp_viol += 1
            if not (-m <= gg <= q and 0 <= h <= m and 0 <= t <= q):
                bound_viol += 1
            checks += 1
    print(f"states checked: {checks}")
    print("repo evaluator vs independent GF(2): all equal (asserted)")
    print(f"states where GF(2) beta1 != rational beta1: {field_diff}")
    print(f"decomposition violations: {decomp_viol}")
    print(f"bound violations: {bound_viol}")


if __name__ == "__main__":
    main()
