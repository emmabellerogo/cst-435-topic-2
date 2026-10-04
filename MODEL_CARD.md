# Model Card: Income Insight (`gelu`, Supabase run 3)

All numbers below come from the committed result files in `models/gelu/` and
`models/experiments/`. The fairness rates come from the live `GET /audit` endpoint,
read on 2026-10-04.

## Model details

| | |
|---|---|
| Task | Binary classification: is annual income `>50K` (positive class) or `<=50K`? |
| Type | PyTorch multi-layer perceptron: `[Linear → GELU → Dropout(0.1)] × 2 → Linear(1)` |
| Shape | 84 encoded inputs → 64 → 32 → 1 logit (7,553 trainable parameters) |
| Training | `BCEWithLogitsLoss`, AdamW (lr 1e-3, weight decay 1e-4), batch 256, seed 42, ≤ 30 epochs with early stopping (patience 5) on validation loss. The kept checkpoint is from **epoch 18** |
| Calibration | Temperature scaling, **T = 1.0238**, fitted on the validation split after training, with the weights frozen |
| Output | Calibrated probability p = σ(z / T); label `>50K` when p ≥ 0.5 (fixed threshold, never tuned) |
| Artifacts | `models/gelu/model.pt`, `preprocessor.joblib`, `calibrator.json`; served read-only by the FastAPI service on Render |
| Owners | Emma Rogoveanu (data, training, evaluation, API, final UI fixes) and Komal Khan (initial Streamlit UI). CST-435 Topic 2 team project; built with AI assistance (see `ai-documentation/`) |

## Intended use

- **Primary:** a teaching demonstration of the full tabular-ML loop. That covers data
  in Postgres, train-only preprocessing, controlled experiments, validation-based
  selection, calibration, a one-time test evaluation, serving, prediction logging and
  a SQL fairness audit.
- **Users:** students, instructors and reviewers of the course project.
- **Out of scope:** any real decision about a person, such as hiring, pay, lending,
  insurance, housing, benefits or eligibility. The data is from 1994, and the model
  makes group-dependent errors (see *Fairness*).

## Data

- **Source:** UCI *Adult* dataset (1994 U.S. Census extract, Kohavi & Becker). The
  `adult.data` and `adult.test` files are combined into **48,842 rows** and stored in
  the Supabase `adult_income` table. Raw `?` values become NULL.
- **Split:** one fixed split (seed 42), stratified on income × sex and stored in
  `adult_income.split`:

  | Split | Rows | Used for |
  |-------|-----:|----------|
  | train | 34,188 | fitting the preprocessor and the network weights |
  | validation | 7,327 | early stopping, checkpoint choice, model selection, fitting T |
  | test | 7,327 | one final evaluation of the selected model, plus the fairness audit |

- **Inputs (10):**
  - numeric: `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`
  - categorical: `workclass`, `marital_status`, `occupation`, `relationship`, `native_country`
- **Preprocessing:**
  - numeric columns: median imputation, then standardization
  - categorical columns: missing values become `"Unknown"`, then one-hot encoding
    (unknown categories are ignored)
  - fitted on train rows only, giving 84 features
- **Deliberately excluded:** `sex` and `race` are protected attributes, kept only for
  the audit. `education` duplicates `education_num`, and `fnlwgt` is a sampling weight.

## Model selection

Three configurations were trained under identical controls: same split,
preprocessor, seed, learning rate, weight decay, batch size, epoch budget, patience
and dropout. Each one differs from `baseline` in exactly one way. The selection rule
was fixed in advance: **lowest validation BCE**, with ties broken by the higher
validation ROC-AUC.

| Config | Hidden | Activation | Best epoch | Val loss | Val acc | Val ROC-AUC |
|--------|--------|------------|-----------:|---------:|--------:|------------:|
| **gelu** (selected) | [64, 32] | GELU | 18 | **0.3156** | 0.8534 | 0.9070 |
| deep | [128, 64, 32] | ReLU | 9 | 0.3158 | 0.8557 | 0.9074 |
| baseline | [64, 32] | ReLU | 18 | 0.3163 | 0.8567 | 0.9074 |

The differences are tiny (0.0007 in validation loss, one seed), so this is weak
evidence that GELU is better in general. Only the selected run was evaluated on test.

## Performance (test split, 7,327 rows)

| Metric | Value |
|--------|------:|
| Accuracy | 0.8540 |
| ROC-AUC | 0.9060 |
| Precision (>50K) | 0.7322 |
| Recall (>50K) | 0.6144 |
| F1 (>50K) | 0.6681 |
| BCE, uncalibrated → calibrated | 0.3190 → 0.3189 |

- **Confusion matrix** `[[TN, FP], [FN, TP]] = [[5180, 394], [676, 1077]]`.
  - `<=50K`: precision 0.8846, recall 0.9293, F1 0.9064 (5,574 rows)
  - `>50K`: precision 0.7322, recall 0.6144, F1 0.6681 (1,753 rows)
- The minority class `>50K` is the harder one. The model misses 676 of the 1,753 people
  who really earn more than \$50K.
- **Train / validation / test** (same checkpoint, threshold 0.5): accuracy is 0.8614 /
  0.8534 / 0.8540, BCE 0.2983 / 0.3156 / 0.3190, and ROC-AUC 0.9182 / 0.9070 / 0.9060.
  These show a small generalization gap and no strong overfitting.

## Calibration

- T was fitted by minimizing BCE on the **validation** split. The test split was not
  used.
- Because T > 0, labels, the confusion matrix and ROC-AUC are unchanged; only the
  confidence of the probabilities moves.
- On test, ECE (10 equal-width bins) went from 0.0100 to 0.0099 and the Brier score
  from 0.1014 to 0.1014. The network was already well calibrated, and calibration
  changed it very little.

## Feature importance

Permutation importance on the test split shuffles each raw feature (10 repeats,
seed 42) and measures the mean drop in ROC-AUC. This shows what the model relies on,
not what causes income.

| Rank | Feature | ROC-AUC drop |
|-----:|---------|-------------:|
| 1 | marital_status | 0.0549 |
| 2 | capital_gain | 0.0370 |
| 3 | age | 0.0335 |
| 4 | education_num | 0.0327 |
| 5 | occupation | 0.0175 |
| 6 | hours_per_week | 0.0138 |
| 7 | relationship | 0.0108 |
| 8 | capital_loss | 0.0051 |
| 9 | workclass | 0.0039 |
| 10 | native_country | 0.0016 |

## Fairness findings (implemented)

**How it is measured.** The SQL view `v_fairness_audit` joins `predictions` to
`adult_income` and counts only predictions linked to a **labeled test row**. All
7,327 test predictions of run 3 were logged for this by `db/log_test_predictions.py`.
The view computes FPR and FNR by sex in SQL, and the API's `/audit` endpoint passes
its rows through unchanged. The UI can also read the same view directly with the
read-only anon key. Predictions made in the app have no true label and are never
used for FPR/FNR.

| Group | Test rows | Base rate (>50K) | FPR | FNR |
|-------|----------:|-----------------:|----:|----:|
| Female | 2,429 | 10.9% | 0.0268 | **0.4151** |
| Male | 4,898 | 30.4% | **0.0985** | 0.3804 |

- Women who really earn >50K are missed more often than men (FNR 41.5% vs 38.0%).
- Men who earn ≤50K are wrongly predicted >50K about 3.7× as often as women (FPR
  9.9% vs 2.7%).
- `sex` is not an input, but `relationship` (Husband/Wife) and `marital_status`
  (the most important feature) are strong proxies for it. The groups' base rates
  also differ substantially.
- Scope: one attribute (sex), one test split, one threshold (0.5), no confidence
  intervals. This describes the model's behavior and does not certify it as fair.
  `race` is stored but has not been audited.

## Proposed mitigations (not implemented)

None of the following has been applied. The served model is the one described above.

- **Group-aware thresholds (post-processing):** choose per-group cut-offs on
  validation to equalize FNR (equal opportunity) or both FPR and FNR (equalized odds).
  This needs `sex` at decision time, which raises its own legal and ethical issues.
- **Reweighting (pre-processing):** give each (sex, income) combination equal total
  weight in the training loss.
- **Fairness penalty (in-processing):** add a loss term that grows with the FPR/FNR
  gap between groups.
- **Proxy analysis:** retrain without `relationship` (or with it merged), then
  measure the change in both accuracy and the gaps.
- **Better evidence:** add confidence intervals, multiple seeds and splits, and an
  audit by `race`.

Any of these would have to be tuned on validation, evaluated once on test, and would
likely cost some overall accuracy.

## Limitations

- The data is from 1994 and reflects that period's U.S. labor market and gender
  roles. It is not representative of incomes today.
- \$50K is a fixed nominal cut-off that has not been adjusted for inflation.
- Recall for `>50K` is only 0.61.
- The experiment comparison and the evaluation use a single seed and a single split.
- The API accepts only numeric values inside the observed ranges (e.g. age 17–90)
  and only categories seen in training. Anything else is rejected, so the model cannot
  describe people outside those ranges.
- Calibration was checked overall, not per group.

## Logging and privacy (implemented)

- **What is logged:** each `/predict` and `/predict_batch` row writes one
  `predictions` record. The record holds a SHA-256 `request_hash` of the 10 feature
  values, the predicted label, the calibrated probability, the serving run id and a
  timestamp.
- **What is not logged:** raw feature values, uploaded files, protected attributes and
  any user identity. The hash is for traceability only. It is not anonymization,
  because the input space is small enough that a given profile's hash can be
  recomputed.
- **Access:** RLS is enabled on every table. Only the API, holding the server-side
  service-role key, can write. The public anon key can only `SELECT` from `runs` and
  from the aggregate `v_fairness_audit` view.
- **Data:** `adult_income` holds the public UCI records, including `sex` and `race`,
  which are used only for the audit.

## Ethical considerations

Users must be told plainly that this is a demonstration trained on 1994 census data.
Its probabilities are estimates and not facts about a person, and its errors differ
between women and men. It must not be used, or presented, as a basis for decisions
about real individuals.
