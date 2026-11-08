# Basket Recommender

A shopping-basket recommendation system built on association-rule mining
(the Apriori algorithm). Tell it what is in your basket, by chat or in a
web GUI, and it suggests what else you are likely to need. Every
suggestion explains itself, and you can control what it suggests.

## Credit: the foundation project

This project extends **Example 1, "Generating Association Rules from
Frequent Itemsets," in the mlxtend user guide** by Sebastian Raschka:
<https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/>.
mlxtend is distributed under the BSD 3-Clause license.

`original_example/mlxtend_baseline.py` reproduces that example unchanged:
five transactions, `min_support=0.6`, rules with confidence ≥ 0.7. The
example ends by printing a table of rules. Everything else in this
repository was added on top of it.

The Apriori algorithm itself comes from Agrawal and Srikant (1994),
*Fast algorithms for mining association rules*, VLDB '94.

## What was changed and added

| Area | Baseline (mlxtend example) | This project |
|---|---|---|
| Algorithm | Calls `mlxtend.apriori` | Own Apriori (join, prune, count) in `recommender/apriori.py`, tested to give identical results to mlxtend |
| Data | 5 hand-written baskets | 3,000 reproducible synthetic baskets, 41 products in 8 categories (`data/`) |
| Output | Table of rules | Ranked recommendations for a shopper's basket |
| Ranking | — | Choice of confidence, lift, or confidence × lift |
| Rule quality | All rules kept | Redundant rules pruned (a rule must beat its simpler versions by 0.10 confidence) |
| Explanations | — | Each suggestion says which rule produced it; `why <item>` shows the evidence |
| User control | — | Exclude items or whole categories; removed items are not suggested again |
| Diversity | — | Optional re-ranking so one category doesn't take every slot |
| Cold start | — | Falls back to popular items when no rule applies |
| Interface | Python script | Conversational text assistant (`app_cli.py`) and Streamlit GUI (`app_gui.py`) |
| Quality | — | 13 automated tests and an offline evaluation (`evaluate.py`) |

## Running it

Requires Python 3.9 or later.

```bash
pip install -r requirements.txt

python original_example/mlxtend_baseline.py   # the original example
python app_cli.py                             # chat-style assistant
streamlit run app_gui.py                      # graphical interface in the browser
python -m pytest -q                           # tests
python evaluate.py                            # evaluation -> results/evaluation.md
python data/generate_transactions.py          # regenerate the dataset (same seed, same data)
```

### Running it in Google Colab

`colab_demo.ipynb` loads the project, installs what it needs, runs the
original example, the tests and the evaluation, and lets you try the
chat assistant and a point-and-click interface, all in the browser.

1. Go to <https://colab.research.google.com> and choose **File ▸ Upload
   notebook**, then pick `colab_demo.ipynb`. Once this repository is on
   GitHub, you can also open it directly at
   `https://colab.research.google.com/github/<your-username>/basket-recommender/blob/main/colab_demo.ipynb`.
2. In the first code cell, paste the repository URL into `REPO_URL`, or
   leave it empty and upload `basket-recommender.zip` when asked.
3. Run the cells in order. The tests should report **13 passed**, and the
   evaluation should report a hit rate of **0.728**.

### Example conversation

```
> I have pasta and parmesan
Added Parmesan, Pasta.
You might also want:
  1. Pasta Sauce (Pantry): 76% of baskets with Pasta also had it (lift 4.8)
  2. Onions (Produce): 29% of baskets with Pasta also had it (lift 4.1)
  ...
> avoid dairy
> why pasta sauce
> remove pasta
```

Commands: `add`, `remove`, `clear`, `show`, `avoid`, `allow`, `why`,
`more`, `less`, `diversity <0-0.9>`, `help`, `quit`. Plain sentences such
as "I've got tortillas and ground beef" also work, and small typos are
corrected.

## Results

Hold-out test: 20% of baskets, one item hidden per basket, 5 suggestions
(full output in `results/evaluation.md`).

| System | Hit rate@5 | Catalog coverage |
|---|---|---|
| Popularity baseline | 0.323 | 0.28 |
| Rules, confidence × lift | 0.728 | 0.92 |
| Rules + diversity 0.5 | 0.715 | 0.92 |

The dataset is synthetic with planted shopping patterns, so these results
show the system works as designed, not how it would perform with real
shoppers.

## Project structure

```
original_example/   mlxtend example used as the foundation
recommender/        apriori.py (algorithm), engine.py (recommender), data.py (loading)
data/               dataset generator, transactions.csv, catalog.csv
app_cli.py          conversational text interface
app_gui.py          Streamlit graphical interface
evaluate.py         offline evaluation
colab_demo.ipynb    runs and tests everything in Google Colab
tests/              automated tests
results/            evaluation output and a sample chat session
```

## License

MIT for this project's code (see `LICENSE`). The baseline example follows
mlxtend's BSD 3-Clause license.
