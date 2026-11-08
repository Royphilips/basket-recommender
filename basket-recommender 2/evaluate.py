"""Offline evaluation of the recommender.

Method (hold-out, "hide one item"):
  1. Shuffle the baskets with a fixed seed; train on 80%, test on 20%.
  2. For every test basket with at least two items, hide one random item.
  3. Ask each system for 5 suggestions given the rest of the basket.
  4. Hit rate@5 = share of baskets where the hidden item was suggested.

Also reported: catalog coverage (share of products ever suggested) and the
average number of distinct categories in a list of suggestions.

Run:  python evaluate.py      (prints tables and writes results/evaluation.md)
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori as mlx_apriori
from mlxtend.preprocessing import TransactionEncoder

from recommender import BasketRecommender, apriori, load_catalog, load_transactions

SEED = 7
K = 5
RESULTS = Path(__file__).resolve().parent / "results"


def split(transactions, test_share=0.2):
    baskets = list(transactions)
    random.Random(SEED).shuffle(baskets)
    cut = int(len(baskets) * (1 - test_share))
    return baskets[:cut], baskets[cut:]


def make_queries(test):
    rng = random.Random(SEED)
    queries = []
    for basket in test:
        if len(basket) < 2:
            continue
        hidden = rng.choice(basket)
        queries.append(([i for i in basket if i != hidden], hidden))
    return queries


def evaluate(recommend, queries, catalog, n_items):
    hits, shown, cat_counts = 0, set(), []
    for query, hidden in queries:
        recs = recommend(query)
        names = [r.item for r in recs]
        hits += hidden in names
        shown.update(names)
        cat_counts.append(len({catalog.get(n, "Other") for n in names}))
    return {"hit_rate@5": hits / len(queries), "coverage": len(shown) / n_items,
            "avg_categories": sum(cat_counts) / len(cat_counts)}


def main() -> None:
    transactions, catalog = load_transactions(), load_catalog()
    train, test = split(transactions)
    queries = make_queries(test)
    n_items = len(catalog)
    rows = []

    popular = BasketRecommender(min_support=1.0).fit(train, catalog)  # no rules -> popularity only
    rows.append(("Popularity baseline", evaluate(lambda q: popular.recommend(q, K), queries, catalog, n_items)))
    for scoring in ("confidence", "lift", "conf_lift"):
        model = BasketRecommender(scoring=scoring).fit(train, catalog)
        rows.append((f"Rules, score = {scoring}",
                     evaluate(lambda q, m=model: m.recommend(q, K), queries, catalog, n_items)))
    unpruned = BasketRecommender(scoring="conf_lift", min_improvement=0).fit(train, catalog)
    rows.append(("Rules, conf_lift, no pruning",
                 evaluate(lambda q: unpruned.recommend(q, K), queries, catalog, n_items)))
    model = BasketRecommender(scoring="conf_lift").fit(train, catalog)
    rows.append(("Rules, conf_lift + diversity 0.5",
                 evaluate(lambda q: model.recommend(q, K, diversity=0.5), queries, catalog, n_items)))
    main_table = pd.DataFrame([{"System": name, **m} for name, m in rows])
    rule_counts = f"Rules kept: {len(model.rules)} with pruning, {len(unpruned.rules)} without."

    sens = []
    for min_support in (0.005, 0.01, 0.02, 0.05):
        start = time.perf_counter()
        m = BasketRecommender(min_support=min_support).fit(train, catalog)
        fit_s = time.perf_counter() - start
        res = evaluate(lambda q: m.recommend(q, K), queries, catalog, n_items)
        sens.append({"min_support": min_support, "rules": len(m.rules),
                     "fit_seconds": round(fit_s, 2), "hit_rate@5": res["hit_rate@5"]})
    sens_table = pd.DataFrame(sens)

    start = time.perf_counter()
    ours = apriori(train, 0.01, max_len=3)
    ours_s = time.perf_counter() - start
    enc = TransactionEncoder()
    df = pd.DataFrame(enc.fit(train).transform(train), columns=enc.columns_)
    start = time.perf_counter()
    theirs = mlx_apriori(df, min_support=0.01, use_colnames=True, max_len=3)
    theirs_s = time.perf_counter() - start
    speed = (f"From-scratch Apriori: {len(ours)} itemsets in {ours_s:.2f}s; "
             f"mlxtend: {len(theirs)} itemsets in {theirs_s:.2f}s (same itemsets: "
             f"{set(ours) == set(map(frozenset, theirs.itemsets))})")

    fmt = {"hit_rate@5": "{:.3f}".format, "coverage": "{:.2f}".format, "avg_categories": "{:.2f}".format}
    text = (f"# Evaluation results\n\nTrain baskets: {len(train)}; test queries: {len(queries)}; k = {K}\n\n"
            f"## Systems compared\n\n{main_table.to_markdown(index=False, floatfmt='.3f')}\n\n"
            f"{rule_counts}\n\n"
            f"## Sensitivity to minimum support (score = conf_lift)\n\n"
            f"{sens_table.to_markdown(index=False)}\n\n## Apriori implementation check\n\n{speed}\n")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "evaluation.md").write_text(text, encoding="utf-8")
    main_table.to_csv(RESULTS / "systems.csv", index=False)
    sens_table.to_csv(RESULTS / "sensitivity.csv", index=False)
    print(main_table.to_string(index=False, formatters=fmt))
    print(rule_counts)
    print()
    print(sens_table.to_string(index=False))
    print()
    print(speed)


if __name__ == "__main__":
    main()
