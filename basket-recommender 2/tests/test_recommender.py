"""Tests: the from-scratch Apriori must agree with mlxtend, and the
recommender extensions must behave as documented.

Run:  python -m pytest -q
"""

import random
import sys
from pathlib import Path

import pandas as pd
import pytest
from mlxtend.frequent_patterns import apriori as mlx_apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app_cli import ShoppingAssistant  # noqa: E402
from original_example.mlxtend_baseline import DATASET  # noqa: E402
from recommender import BasketRecommender, apriori, generate_rules, load_catalog, load_transactions  # noqa: E402


def mlxtend_itemsets(dataset, min_support):
    enc = TransactionEncoder()
    df = pd.DataFrame(enc.fit(dataset).transform(dataset), columns=enc.columns_)
    fi = mlx_apriori(df, min_support=min_support, use_colnames=True)
    return df, fi


@pytest.mark.parametrize("min_support", [0.2, 0.4, 0.6])
def test_itemsets_match_mlxtend_on_original_example(min_support):
    _, fi = mlxtend_itemsets(DATASET, min_support)
    expected = {frozenset(s): round(v, 9) for s, v in zip(fi.itemsets, fi.support)}
    ours = {s: round(v, 9) for s, v in apriori(DATASET, min_support).items()}
    assert ours == expected


def test_rules_match_mlxtend_on_original_example():
    df, fi = mlxtend_itemsets(DATASET, 0.6)
    mlx = association_rules(fi, metric="confidence", min_threshold=0.7, num_itemsets=len(df.index))
    expected = {(frozenset(a), frozenset(c)): (round(conf, 6), round(lift, 6))
                for a, c, conf, lift in zip(mlx.antecedents, mlx.consequents, mlx.confidence, mlx.lift)}
    ours = {(r.antecedent, r.consequent): (round(r.confidence, 6), round(r.lift, 6))
            for r in generate_rules(apriori(DATASET, 0.6), min_confidence=0.7)}
    assert ours == expected


def test_itemsets_match_mlxtend_on_project_data():
    sample = random.Random(0).sample(load_transactions(), 800)
    _, fi = mlxtend_itemsets(sample, 0.02)
    expected = {frozenset(s): round(v, 9) for s, v in zip(fi.itemsets, fi.support)}
    ours = {s: round(v, 9) for s, v in apriori(sample, 0.02).items()}
    assert ours == expected


@pytest.fixture(scope="module")
def model():
    return BasketRecommender().fit(load_transactions(), load_catalog())


def test_recommends_planted_pattern_with_explanation(model):
    recs = model.recommend(["Pasta"], k=3)
    assert recs[0].item == "Pasta Sauce"
    assert "Pasta" in recs[0].reason and recs[0].rule is not None


def test_never_recommends_items_already_in_basket(model):
    basket = ["Pasta", "Pasta Sauce", "Parmesan"]
    assert not {r.item for r in model.recommend(basket, k=10)} & set(basket)


def test_category_and_item_exclusions(model):
    recs = model.recommend(["Pasta"], k=10, exclude_items=["Pasta Sauce"],
                           exclude_categories=["Dairy & Eggs"])
    assert "Pasta Sauce" not in {r.item for r in recs}
    assert all(r.category != "Dairy & Eggs" for r in recs)


def test_cold_start_falls_back_to_popular_items(model):
    recs = model.recommend([], k=3)
    assert len(recs) == 3 and all(r.reason.startswith("popular") for r in recs)


def test_diversity_limits_repeated_categories(model):
    basket = ["Tortillas", "Ground Beef"]
    plain = model.recommend(basket, k=4, fill_with_popular=False)
    diverse = model.recommend(basket, k=4, diversity=0.9, fill_with_popular=False)
    count = lambda recs: max(sum(r.category == c for r in recs) for c in {r.category for r in recs})
    assert count(diverse) <= count(plain)


def test_redundant_rules_are_pruned(model):
    # The coincidental rule {Paper Towels, Pasta} -> {Pasta Sauce} must not survive,
    # because {Pasta} -> {Pasta Sauce} explains it almost as well.
    for rule in model.rules_for("Pasta Sauce", limit=50):
        assert "Paper Towels" not in rule.antecedent


def test_removed_items_are_not_suggested_again(model):
    bot = ShoppingAssistant(model)
    bot.handle("add pasta, parmesan, ground beef")
    reply = bot.handle("remove pasta")
    assert "1. Pasta (" not in reply and "Pasta" in bot.dismissed
    bot.handle("allow pasta")
    assert "Pasta" not in bot.dismissed


def test_cli_understands_free_text_and_longest_match(model):
    bot = ShoppingAssistant(model)
    bot.handle("I've got pasta sauce and some parmesan")
    assert bot.basket == {"Pasta Sauce", "Parmesan"}
    bot.handle("add tortilas")  # typo is fuzzy-matched
    assert "Tortillas" in bot.basket
    reply = bot.handle("avoid dairy")
    assert "Dairy & Eggs" in bot.avoid_categories and "Cheddar Cheese" not in reply
