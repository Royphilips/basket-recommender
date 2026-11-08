"""Generate a reproducible synthetic grocery dataset.

The mlxtend example has only five baskets, which is too small to test a
recommender. No real retail data could be used, so this script simulates
shoppers. Each basket is built from one to three shopping "missions"
(for example, pasta night or a baby-supplies run), plus a few random
everyday items. The planted missions give the data realistic co-purchase
patterns; the random items add noise.

Because the patterns are planted, results on this data show that the
system works as designed. They do not prove how it would perform on real
shoppers.

Run:  python data/generate_transactions.py   (writes transactions.csv and catalog.csv)
"""

import csv
import random
from pathlib import Path

SEED = 42
N_BASKETS = 3000
OUT = Path(__file__).resolve().parent

CATALOG = {
    "Produce": ["Bananas", "Apples", "Onions", "Tomatoes", "Lettuce", "Avocado", "Lemons"],
    "Dairy & Eggs": ["Milk", "Butter", "Cheddar Cheese", "Parmesan", "Yogurt", "Eggs"],
    "Bakery": ["Bread", "Bagels", "Burger Buns", "Tortillas"],
    "Pantry": ["Pasta", "Pasta Sauce", "Rice", "Cereal", "Peanut Butter", "Jam",
               "Coffee", "Tea", "Sugar", "Honey", "Salsa"],
    "Meat": ["Ground Beef", "Chicken Breast", "Bacon"],
    "Snacks": ["Potato Chips", "Cookies"],
    "Beverages": ["Soda", "Orange Juice", "Beer"],
    "Household": ["Diapers", "Baby Wipes", "Paper Towels"],
}

# (weight, {item: probability the item is bought on this mission})
MISSIONS = [
    (10, {"Pasta": .95, "Pasta Sauce": .80, "Parmesan": .55, "Ground Beef": .35, "Onions": .30}),
    (10, {"Eggs": .70, "Bacon": .50, "Bread": .50, "Orange Juice": .40, "Butter": .35}),
    (9, {"Cereal": .95, "Milk": .85, "Bananas": .50}),
    (8, {"Bread": .90, "Peanut Butter": .60, "Jam": .55}),
    (7, {"Burger Buns": .90, "Ground Beef": .85, "Cheddar Cheese": .50, "Lettuce": .40, "Tomatoes": .45}),
    (7, {"Tortillas": .90, "Ground Beef": .60, "Salsa": .60, "Cheddar Cheese": .50, "Avocado": .45, "Lettuce": .30}),
    (8, {"Potato Chips": .90, "Salsa": .55, "Soda": .70, "Beer": .45}),
    (8, {"Coffee": .95, "Milk": .60, "Sugar": .45}),
    (5, {"Tea": .95, "Honey": .55, "Lemons": .40}),
    (4, {"Diapers": .95, "Baby Wipes": .75, "Beer": .20}),
    (8, {"Yogurt": .70, "Bananas": .60, "Apples": .60, "Chicken Breast": .50, "Rice": .40}),
]

# Everyday "noise" items and how likely each is to be picked at random.
STAPLE_WEIGHTS = {"Milk": 6, "Bread": 5, "Eggs": 5, "Bananas": 4, "Paper Towels": 4,
                  "Cookies": 3, "Soda": 3, "Apples": 3, "Butter": 2, "Bagels": 2, "Rice": 2}


def make_basket(rng: random.Random) -> list:
    basket = set()
    n_missions = rng.choices([1, 2, 3], weights=[55, 35, 10])[0]
    weights = [w for w, _ in MISSIONS]
    for _ in range(n_missions):
        _, mission = rng.choices(MISSIONS, weights=weights)[0]
        basket.update(item for item, p in mission.items() if rng.random() < p)
    all_items = [i for items in CATALOG.values() for i in items]
    noise_weights = [STAPLE_WEIGHTS.get(i, 1) for i in all_items]
    n_noise = min(rng.choices([0, 1, 2, 3], weights=[25, 40, 25, 10])[0], 3)
    basket.update(rng.choices(all_items, weights=noise_weights, k=n_noise))
    if not basket:
        basket.add(rng.choice(all_items))
    return sorted(basket)


def main() -> None:
    rng = random.Random(SEED)
    with open(OUT / "transactions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["basket_id", "items"])
        for i in range(N_BASKETS):
            writer.writerow([i + 1, ";".join(make_basket(rng))])
    with open(OUT / "catalog.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "category"])
        for category, items in CATALOG.items():
            for item in items:
                writer.writerow([item, category])
    print(f"Wrote {N_BASKETS} baskets to {OUT / 'transactions.csv'}")


if __name__ == "__main__":
    main()
