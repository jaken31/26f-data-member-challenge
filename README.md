# Predicting Developer Salaries

Predicting `annual_salary_usd` from a 5,000-response subsample of the Stack Overflow Annual
Developer Survey 2025.

Everything is in [`notebook.ipynb`](notebook.ipynb), which runs top to bottom and has its outputs
stored, so the numbers below can be checked against the cells that produced them.

## Results

Five-fold cross-validation, out-of-fold predictions, all four models scored on identical folds:

| model | MAE | MedAE | RMSE | R²_log |
|---|---|---|---|---|
| global median | $47,332 | $36,289 | $65,812 | −0.009 |
| country median (out-of-fold) | $35,614 | $24,181 | $52,326 | 0.375 |
| Ridge (one-hot) | $29,761 | $18,831 | $45,851 | 0.570 |
| **HistGradientBoosting** | **$27,808** | **$17,409** | **$43,297** | **0.598** |

R²_log is computed in log space, where the model is fitted; the dollar metrics are back-transformed
so they can be interpreted. Exported to [`outputs/metrics.json`](outputs/metrics.json), with per-row
predictions in [`outputs/predictions.csv`](outputs/predictions.csv).

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Restart & Run All takes a couple of minutes. Built on Python 3.13. Only pandas, numpy,
scikit-learn and matplotlib are used.

## What I found in the data

**The target had been converted from local currency at one fixed rate.** 66 respondents report
exactly $69,609. The most repeated values sit in exact simple ratios to each other (116,015 / 69,609
= 5/3), which is what happens when one unit is rescaled rather than when independent numbers
coincide. Dividing by an implied 1.16014 USD/EUR turns every repeated value into a round number of
euros — 50,000, 60,000, 65,000, 70,000. This is why `Currency` is worth keeping as a feature and why
the target's precision is illusory.

**501 rows sit outside a plausible range.** The bottom of the distribution is $1 to $10 for
full-time employed developers. These are identifiable as wrong but not repairable: `CompFreq` is
absent from this subsample, so there is no period to multiply by. The top is a genuine mixture — a
$6.89M figure for a freelancer paid in hryvnia is a conversion escape, but nine of the fifteen
largest are US employees between $900k and $2M, which is rare rather than impossible. Section 3
measures the trim instead of guessing it, holding the model fixed and varying only the filter.

**1,133 of 5,000 rows separate the currency code from its name with a tab** rather than a space,
so the raw string is an unreliable key. `currency_code` is split out and the raw column dropped.

**One exact duplicate row**, visible only when `ResponseId` is excluded from the comparison.

Cleaning removes 586 rows (11.7%): 84 with no target, 501 outside the $10k–$400k trim, 1 duplicate.
Every column's verdict — keep, clean, engineer, drop — is tabulated in section 1 with the reasoning.

## Modelling choices

- **`log1p` target.** Takes skew from +1.36 to −0.52, so neither tail dominates the squared error,
  and error becomes proportional — which is how a salary miss is actually judged.
- **Two baselines before the model.** The global median exists to confirm R²_log ≈ 0 for a
  no-information predictor. The country median is the real opponent. It is computed **out-of-fold**:
  medians built from training rows only, using the same splitter as the model. Computing them on all
  rows instead inflates that baseline by 0.047, because each test row then helps set the bar it is
  judged against.
- **HistGradientBoosting** takes categoricals and NaN natively, so there is no encoder, imputer or
  scaler to fit incorrectly inside a fold.
- **Ridge for comparison.** The same features through the full encoding stack reach 0.570 against
  HGB's 0.598. The boosting wins, but by 0.028 — which is itself informative, and covered below.
- **19 assertions** guard the pipeline: row counts, unique columns, that the ordinal maps match the
  survey's exact strings, that no prediction slot went unfilled.

## Known weaknesses

**`Country` is most of the model.** Permuting it costs 0.426 of R²_log; permuting each of the other
83 features and summing the damage comes to 0.411. One column outweighs everything else combined.
The out-of-fold country median reaching 0.375 on its own says the same thing from the other
direction, and it explains why Ridge lands so close: if the signal is largely a per-country offset,
then in log space that is exactly what a linear model represents natively.

**The trim behaves as an income filter.** A flat $10,000 floor is not neutral across countries. It
removes 73.3% of Nigeria's respondents, 69.6% of Bangladesh's and 30.2% of India's, against 4.7% of
the United States'. Eleven countries with 15+ respondents lose more than a fifth of their rows.
Section 8 re-runs the identical pipeline under a country-relative rule (±3 SD on the log ratio to
each country's own median), which cuts that from eleven countries to one. It scores worse — MAE
$34,831, R²_log 0.556 — but on a different and harder population of 4,775 rows, so the two numbers
are not directly comparable. The finding is that part of the headline 0.598 is bought by deleting
the respondents the model would have been worst at.

**The error is large, and unevenly distributed.** MAE is 33.7% of the median salary. Only 45% of
predictions land within 20% of the truth. Grouped by how many respondents share a country, MAE as a
share of that group's median runs 77% for countries with under 10 respondents against 29% for those
with 200+ — the model serves well-surveyed rich markets best and everyone else worst, which is
backwards from where pay information is scarcest.

**The sample is self-selected twice over** — people who took the survey, reached the compensation
question, and chose to answer it. Someone who feels badly underpaid has an obvious reason to skip
it. I would not use this model to set pay; it learned the existing gaps between countries.

## What I would do next

1. **Move to the country-relative trim, and add shrinkage to `Country` in the same change.** They
   are coupled: a fairer trim keeps more thin-country rows, which is exactly where an unshrunk
   high-cardinality category starts to hurt.
2. **Recover the pre-conversion local-currency amounts** and model those, so the model stops fitting
   exchange-rate arithmetic and the ceiling means the same thing everywhere.
3. **Add sub-national location.** "United States" spans San Francisco and rural Alabama as one
   level, and `Country` already carries half the model.
4. **Promote error-by-country-sample-size into the evaluation table**, rather than leaving it as a
   limitations note.

## Repository

```
notebook.ipynb           analysis, start to finish
data/survey.csv          5,000 responses, 16 columns
outputs/metrics.json     the results table above
outputs/predictions.csv  per-row actual, predicted and absolute error (4,414 rows)
requirements.txt
```

Data: Stack Overflow Annual Developer Survey 2025, published by Stack Exchange under the
[Open Database License](https://opendatacommons.org/licenses/odbl/1-0/).
