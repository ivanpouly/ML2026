# CS-433 Project 1 – Predicting coronary heart disease (MICHD) from BRFSS survey data

Team: Author One, Author Two, Author Three

We predict whether a respondent of the 2015 Behavioral Risk Factor Surveillance
System (BRFSS) reports coronary heart disease or a heart attack, from 321 survey
answers. The final model is an **L2-regularized logistic regression trained with
Newton's method** on cleaned and encoded features plus their interactions with age,
with a decision threshold tuned for the F1 score.

| | F1 |
|---|---|
| 5-fold cross-validation | 0.438 ± 0.005 |
| AIcrowd test set (submission produced by `run.py`) | 0.448 |

The report (2 pages) is in `report/report.pdf`, with its LaTeX source in `report/`.

## Setup

- Python 3 with NumPy only (the course `project1-grading` conda environment works).
  `ridge_search.py` also uses matplotlib, only to draw a figure.
- Put `x_train.csv`, `y_train.csv` and `x_test.csv` in `data/` (not tracked by git).
  The first run caches them in `data/cache.npz` (about 1 GB), which makes later runs
  much faster.

## Reproduce the submission

```bash
python run.py
```

It writes `submission.csv` in a few minutes (about 3.5 GB of memory). Expected output:

```
tx_train: (328135, 490) | tx_test: (109379, 490)
train F1: 0.4415
predicted positive rate: train 13.4% | test 13.4%
```

## Repository structure

| File | Role |
|---|---|
| `run.py` | Trains the final model on the whole training set and writes `submission.csv`. |
| `implementations.py` | The six required methods (GD, SGD, least squares, ridge, logistic, regularized logistic). |
| `helpers.py` | Course helpers: CSV loading and submission writing. |
| `utils.py` | `load_data`: loads the CSV files once, then from the binary cache. |
| `specifications.py` | `SPECS`: the 76 variables kept and how to recode each of them. |
| `preprocess.py` | Recoding, imputation, one-hot encoding, missing-value indicators, standardization (`fit_transform` / `transform`). |
| `features.py` | Interactions between age and every other feature. |
| `model.py` | L2-regularized logistic regression solved with Newton's method. |
| `cross_validation.py` | Fold split, metrics (F1, RMSE), F1-optimal threshold, per-fold preprocessing. |
| `model_search.py` | Selects λ and the decision threshold by 5-fold cross-validation. |
| `ridge_search.py` | Ridge regression baseline: λ selection by cross-validation, RMSE figure. |
| `ablations.py` | Reproduces every number of Tables I and II of the report. |
| `report/` | Report source (`report.tex`, `literature.bib`, figure) and PDF. |

## Pipeline

1. **Variable selection** (`specifications.py`). From the CDC codebook we keep 73 raw
   variables and build 3 derived ones (76 in total). We drop record identifiers and
   interview logistics, phone-screening questions, survey weights, raw answers
   duplicated by a cleaner CDC calculated variable, and the 109 optional-module
   questions (asked only in some states).
2. **Recoding** (`preprocess.py`, rules in `SPECS`). "Don't know" / "refused" codes
   (7/9, 77/99, …) become missing; "none" (88) becomes 0; values outside the documented
   range (e.g. sentinels 900 or 99900) become missing; heavy-tailed quantities get
   `log(1 + x)`.
3. **Derived variables**. `IS_PARTIAL` (interview stopped early), `HH_ADULTS` (number of
   adults, merging the landline and cell-phone questions) and `HAREHAB1_PRESENT`.
   **Target leak:** `HAREHAB1` (cardiac rehabilitation after a heart attack) is only asked
   to respondents who had a heart attack, so its presence reveals the label for 773
   training respondents. The column is part of the provided test data; the report gives
   scores with and without it.
4. **Encoding**, with statistics fitted on the training data only. Numeric variables:
   median imputation. Categorical variables: one-hot encoding without the most frequent
   level. Missing-value indicator when at least 0.5% of the values are missing.
   Variables missing in more than 95% of the rows would be dropped (none are).
   Standardization of every column. Result: 245 features.
5. **Age interactions** (`features.py`): product of the standardized age with every
   other feature, standardized again: 489 features, plus an intercept.
6. **Model** (`model.py`): logistic regression with L2 penalty λ = 1e-2 on every weight
   except the intercept, solved with Newton's method (about 10 iterations).
7. **Decision**: predict 1 when the probability is at least 0.1962, the threshold that
   maximizes F1 on the out-of-fold predictions.

## Cross-validation

- 5 folds, one fixed random split (seed 1), shared by all experiments, so that two
  models can be compared fold by fold (paired differences).
- The preprocessing is refitted inside each training fold; the decision threshold is
  always chosen on training data, never on the evaluated fold.
- `python model_search.py` (about 10 minutes) compares λ ∈ {1e-2, 1e-3, 1e-4}, with and
  without age interactions, and prints the `LAMBDA` and `THRESHOLD` used in `run.py`.
- `python ridge_search.py` selects the ridge regression baseline (λ = 3e-3) on the RMSE
  and saves `figures/ridge_lambda.png`.
- `python ablations.py` (about 20 minutes) reproduces Tables I and II of the report.
  For speed, the ablations of Table II fit the label-free preprocessing once on all
  training rows; this gives the same score as per-fold preprocessing for the
  reference model.

## Required methods

`implementations.py` contains `mean_squared_error_gd`, `mean_squared_error_sgd`
(mini-batch size 1), `least_squares`, `ridge_regression`, `logistic_regression` and
`reg_logistic_regression`. Each returns `(w, loss)`, the last weight vector and its
loss; the loss of the regularized methods does not include the penalty term. They pass
the course's public grading tests.
