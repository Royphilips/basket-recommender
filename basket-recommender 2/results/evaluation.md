# Evaluation results

Train baskets: 2400; test queries: 585; k = 5

## Systems compared

| System                           |   hit_rate@5 |   coverage |   avg_categories |
|:---------------------------------|-------------:|-----------:|-----------------:|
| Popularity baseline              |        0.323 |      0.282 |            4.068 |
| Rules, score = confidence        |        0.728 |      0.923 |            3.926 |
| Rules, score = lift              |        0.677 |      0.923 |            3.848 |
| Rules, score = conf_lift         |        0.728 |      0.923 |            3.906 |
| Rules, conf_lift, no pruning     |        0.735 |      0.923 |            3.911 |
| Rules, conf_lift + diversity 0.5 |        0.715 |      0.923 |            4.238 |

Rules kept: 214 with pruning, 632 without.

## Sensitivity to minimum support (score = conf_lift)

|   min_support |   rules |   fit_seconds |   hit_rate@5 |
|--------------:|--------:|--------------:|-------------:|
|         0.005 |     345 |          0.93 |     0.724786 |
|         0.01  |     214 |          0.26 |     0.728205 |
|         0.02  |     180 |          0.12 |     0.724786 |
|         0.05  |      79 |          0.08 |     0.675214 |

## Apriori implementation check

From-scratch Apriori: 594 itemsets in 0.26s; mlxtend: 594 itemsets in 0.02s (same itemsets: True)
