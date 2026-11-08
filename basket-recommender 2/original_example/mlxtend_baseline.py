"""Baseline: the starting point for this project.

This script reproduces "Example 1 -- Generating Association Rules from
Frequent Itemsets" from the mlxtend user guide by Sebastian Raschka:
https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/

mlxtend is distributed under the BSD 3-Clause license. The five-transaction
dataset and the thresholds (min_support=0.6, confidence >= 0.7) are taken
from that example so the baseline behaves exactly as documented.

The example stops at printing a table of rules. Everything in the
`recommender/` package, the apps, the tests and the evaluation are the
extensions built on top of it for this project.

Run:  python original_example/mlxtend_baseline.py
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

DATASET = [
    ["Milk", "Onion", "Nutmeg", "Kidney Beans", "Eggs", "Yogurt"],
    ["Dill", "Onion", "Nutmeg", "Kidney Beans", "Eggs", "Yogurt"],
    ["Milk", "Apple", "Kidney Beans", "Eggs"],
    ["Milk", "Unicorn", "Corn", "Kidney Beans", "Yogurt"],
    ["Corn", "Onion", "Onion", "Kidney Beans", "Ice cream", "Eggs"],
]


def run_baseline(dataset=DATASET, min_support=0.6, min_confidence=0.7):
    """Return (frequent_itemsets, rules) exactly as the mlxtend example does."""
    encoder = TransactionEncoder()
    onehot = encoder.fit(dataset).transform(dataset)
    df = pd.DataFrame(onehot, columns=encoder.columns_)
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
        num_itemsets=len(df.index),
    )
    return frequent_itemsets, rules


if __name__ == "__main__":
    itemsets, rules = run_baseline()
    pd.set_option("display.width", 120)
    print("Frequent itemsets (support >= 0.6):")
    print(itemsets.to_string(index=False))
    print("\nAssociation rules (confidence >= 0.7):")
    print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].to_string(index=False))
