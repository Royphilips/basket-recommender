"""BasketRecommender: turns association rules into explained, controllable recommendations.

Extensions to the mlxtend baseline (which only prints a table of rules):
  * scoring   - ranks candidate items by the best rule that fires for the basket
  * explanation - every suggestion carries the rule that produced it ("because you have ...")
  * user control - shoppers can exclude items or whole categories (e.g. "no dairy")
  * diversity - an optional re-ranking step that stops one category taking every slot
  * cold start - falls back to popular items when no rule matches the basket
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .apriori import Rule, apriori, generate_rules

SCORING_METHODS = ("confidence", "lift", "conf_lift")


@dataclass
class Recommendation:
    item: str
    score: float
    category: str
    reason: str
    rule: Optional[Rule] = None

    def explain(self) -> str:
        return f"{self.item} ({self.category}): {self.reason}"


class BasketRecommender:
    def __init__(self, min_support: float = 0.01, min_confidence: float = 0.2,
                 min_lift: float = 1.0, scoring: str = "conf_lift", max_len: int = 3,
                 min_improvement: float = 0.1):
        if scoring not in SCORING_METHODS:
            raise ValueError(f"scoring must be one of {SCORING_METHODS}")
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.min_improvement = min_improvement
        self.scoring = scoring
        self.max_len = max_len
        self.rules: List[Rule] = []
        self.popularity: List[tuple] = []
        self.categories: Dict[str, str] = {}
        self.items: List[str] = []

    # ------------------------------------------------------------------ training
    def fit(self, transactions: Sequence[Iterable[str]],
            categories: Optional[Dict[str, str]] = None) -> "BasketRecommender":
        baskets = [set(t) for t in transactions]
        frequent = apriori(baskets, self.min_support, max_len=self.max_len)
        all_rules = generate_rules(frequent, self.min_confidence, self.min_lift)
        # A shopper is offered one item at a time, so keep single-item consequents.
        single = [r for r in all_rules if len(r.consequent) == 1]
        self.rules = self._prune_redundant(single, self.min_improvement)
        counts = Counter(item for basket in baskets for item in basket)
        n = len(baskets)
        self.popularity = [(item, c / n) for item, c in counts.most_common()]
        self.items = sorted(counts)
        self.categories = dict(categories or {})
        return self

    @staticmethod
    def _prune_redundant(rules: List[Rule], min_improvement: float) -> List[Rule]:
        """Drop a rule when a simpler rule (a subset of its antecedent) predicts the
        same item almost as well. Without this, coincidental extras such as
        {Paper Towels, Pasta} -> {Pasta Sauce} crowd out the real reason, {Pasta}."""
        if min_improvement <= 0:
            return rules
        by_consequent: Dict[frozenset, List[Rule]] = {}
        for rule in rules:
            by_consequent.setdefault(rule.consequent, []).append(rule)
        kept = []
        for rule in rules:
            simpler = [r.confidence for r in by_consequent[rule.consequent]
                       if r.antecedent < rule.antecedent]
            if not simpler or rule.confidence >= max(simpler) + min_improvement:
                kept.append(rule)
        return kept

    def category_of(self, item: str) -> str:
        return self.categories.get(item, "Other")

    def _score(self, rule: Rule) -> float:
        if self.scoring == "confidence":
            return rule.confidence
        if self.scoring == "lift":
            return rule.lift
        return rule.confidence * rule.lift

    # ------------------------------------------------------------- recommending
    def recommend(self, basket: Iterable[str], k: int = 5,
                  exclude_items: Iterable[str] = (), exclude_categories: Iterable[str] = (),
                  diversity: float = 0.0, fill_with_popular: bool = True) -> List[Recommendation]:
        basket = set(basket)
        banned_items = set(exclude_items)
        banned_cats = {c.lower() for c in exclude_categories}

        def allowed(item: str) -> bool:
            return (item not in basket and item not in banned_items
                    and self.category_of(item).lower() not in banned_cats)

        # 1. Best firing rule per candidate item.
        best: Dict[str, Recommendation] = {}
        for rule in self.rules:
            if not rule.antecedent <= basket:
                continue
            (item,) = tuple(rule.consequent)
            if not allowed(item):
                continue
            score = self._score(rule)
            current = best.get(item)
            if (current is None or score > current.score
                    or (score == current.score and len(rule.antecedent) > len(current.rule.antecedent))):
                because = " and ".join(sorted(rule.antecedent))
                reason = (f"{rule.confidence:.0%} of baskets with {because} also had it "
                          f"(lift {rule.lift:.1f})")
                best[item] = Recommendation(item, score, self.category_of(item), reason, rule)

        ranked = sorted(best.values(), key=lambda r: (-r.score, r.item))
        picks = self._diversify(ranked, k, diversity)

        # 2. Cold start / short list: fill remaining slots with popular items.
        if fill_with_popular and len(picks) < k:
            chosen = {r.item for r in picks}
            for item, share in self.popularity:
                if len(picks) >= k:
                    break
                if item in chosen or not allowed(item):
                    continue
                picks.append(Recommendation(item, 0.0, self.category_of(item),
                                            f"popular: in {share:.0%} of all baskets"))
        return picks

    @staticmethod
    def _diversify(ranked: List[Recommendation], k: int, diversity: float) -> List[Recommendation]:
        """Greedy re-ranking: each repeat of a category multiplies the score by (1 - diversity)."""
        if diversity <= 0:
            return ranked[:k]
        penalty = 1.0 - min(diversity, 0.99)
        remaining = list(ranked)
        picks: List[Recommendation] = []
        used: Counter = Counter()
        while remaining and len(picks) < k:
            choice = max(remaining, key=lambda r: r.score * penalty ** used[r.category])
            picks.append(choice)
            used[choice.category] += 1
            remaining.remove(choice)
        return picks

    # ------------------------------------------------------------------ helpers
    def rules_for(self, item: str, limit: int = 3) -> List[Rule]:
        """The strongest rules that recommend `item` (used by the 'why' command)."""
        matching = [r for r in self.rules if item in r.consequent]
        return sorted(matching, key=self._score, reverse=True)[:limit]
