"""A from-scratch implementation of the Apriori algorithm and rule generation.

Apriori (Agrawal & Srikant, 1994) finds every itemset whose support is at
least `min_support`. It relies on the downward-closure property: every
subset of a frequent itemset must also be frequent. So candidates of size k
are built only by joining frequent itemsets of size k-1, and any candidate
with an infrequent (k-1)-subset is pruned before the data is scanned again.

Rules X -> Y are then generated from each frequent itemset with:
    support(X -> Y)    = support(X u Y)
    confidence(X -> Y) = support(X u Y) / support(X)
    lift(X -> Y)       = confidence(X -> Y) / support(Y)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, Iterable, List

Itemset = FrozenSet[str]


@dataclass(frozen=True)
class Rule:
    antecedent: Itemset
    consequent: Itemset
    support: float
    confidence: float
    lift: float

    def __str__(self) -> str:
        left = ", ".join(sorted(self.antecedent))
        right = ", ".join(sorted(self.consequent))
        return (f"{{{left}}} -> {{{right}}} "
                f"(support={self.support:.3f}, confidence={self.confidence:.2f}, lift={self.lift:.2f})")


def _generate_candidates(previous: Iterable[Itemset], k: int) -> set:
    """Apriori-gen: join frequent (k-1)-itemsets, then prune by downward closure."""
    prev_sorted = sorted(tuple(sorted(s)) for s in previous)
    prev_set = {frozenset(s) for s in prev_sorted}
    candidates = set()
    for i, a in enumerate(prev_sorted):
        for b in prev_sorted[i + 1:]:
            # Itemsets are sorted, so once the (k-2)-prefix differs no later one matches.
            if a[: k - 2] != b[: k - 2]:
                break
            candidate = frozenset(a) | frozenset(b)
            if len(candidate) != k:
                continue
            if all(frozenset(sub) in prev_set for sub in combinations(candidate, k - 1)):
                candidates.add(candidate)
    return candidates


def apriori(transactions: Iterable[Iterable[str]], min_support: float,
            max_len: int | None = None) -> Dict[Itemset, float]:
    """Return {itemset: support} for every itemset with support >= min_support."""
    baskets = [frozenset(t) for t in transactions]
    n = len(baskets)
    if n == 0:
        return {}

    counts: Dict[Itemset, int] = defaultdict(int)
    for basket in baskets:
        for item in basket:
            counts[frozenset([item])] += 1
    current = {s: c / n for s, c in counts.items() if c / n >= min_support}
    frequent = dict(current)

    k = 2
    while current and (max_len is None or k <= max_len):
        candidates = _generate_candidates(current.keys(), k)
        if not candidates:
            break
        counts = defaultdict(int)
        for basket in baskets:
            if len(basket) < k:
                continue
            for candidate in candidates:
                if candidate <= basket:
                    counts[candidate] += 1
        current = {s: c / n for s, c in counts.items() if c / n >= min_support}
        frequent.update(current)
        k += 1
    return frequent


def generate_rules(frequent: Dict[Itemset, float], min_confidence: float = 0.0,
                   min_lift: float = 0.0) -> List[Rule]:
    """Generate every rule X -> Y from the frequent itemsets that meets the thresholds."""
    rules: List[Rule] = []
    for itemset, support in frequent.items():
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for left in combinations(sorted(itemset), size):
                antecedent = frozenset(left)
                consequent = itemset - antecedent
                confidence = support / frequent[antecedent]
                if confidence < min_confidence:
                    continue
                lift = confidence / frequent[consequent]
                if lift < min_lift:
                    continue
                rules.append(Rule(antecedent, consequent, support, confidence, lift))
    return rules
