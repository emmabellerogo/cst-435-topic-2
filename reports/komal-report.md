# Komal Khan — Engineering Report
**Income Insight · CST-435 Topic 2**

## 1. About the App

- **What it does:** predicts if a person earns more than $50K a year
- **Data:** UCI Adult Income, 48,842 records from the 1994 US census
- **Three clouds:**
  - Streamlit = the website people use
  - FastAPI on Render = runs the model
  - Supabase = stores the data and logs every prediction
- **Purpose:** a teaching tool, not a tool for real hiring or loan decisions

## 2. My Role: The Frontend

I built the first full version of the Streamlit app, using the API handoff from Emma (commit `f34683e`).

**What I made:**
- **API client** that sends requests to FastAPI, plus settings helpers
- **Concepts tab** — forward and backward propagation, with a hands-on XOR demo
- **Score a Row tab** — a form that builds itself from the `/schema` endpoint
- **Score CSV tab** — upload a spreadsheet, get back a scored file to download
- **Model Performance tab** — metrics, confusion matrix, calibration
- **Bias Audit tab** — error rates by sex from `/audit`
- **Model Card tab** — one-page summary of the model
- **First frontend tests** — `test_ui_app.py`, `test_ui_api_client.py`, `test_ui_data.py`

**Design choices I followed:**
- The website is a **thin client**: it never loads the model
- Every prediction goes through the API, so everyone uses the same model
- The tabs show weak spots (confusion matrix, bias numbers), not just good scores

**Not my work:** the backend, model training, and the later UI fixes (learning curves, direct Supabase audit display) were done by Emma.

## 3. Decision Justifications

### (a) Why GELU was picked

The team tested three setups with the same data split, seed and settings:

| Setup | Layers | Activation | Validation loss |
|---|---|---|---:|
| GELU ✅ | [64, 32] | GELU | **0.3156** |
| Deep | [128, 64, 32] | ReLU | 0.3158 |
| Baseline | [64, 32] | ReLU | 0.3163 |

- GELU had the **lowest validation loss**, which was our rule set before training
- ReLU cuts every negative value to zero; GELU is smoother and lets small negatives through a little
- **Honest note:** the gap is tiny and from one run, and the baseline had slightly higher accuracy. GELU won this test, but it is not proof that GELU is always better.

### (b) Which class is harder

Test results: **85.4% accuracy**, ROC-AUC **0.906**

| | Predicted ≤50K | Predicted >50K |
|---|---:|---:|
| **Actual ≤50K** | 5,180 | 394 |
| **Actual >50K** | 676 | 1,077 |

- The **>50K class is harder**
- The model missed **676 of 1,753** real high earners (recall 0.61)
- For ≤50K, recall is much better (0.93)
- Why: only about 1 in 4 people in the data earn over $50K
- High accuracy can hide this, so the app shows per-class results

### (c) Which features drive predictions

Permutation importance (drop in ROC-AUC when a feature is shuffled):

1. Marital status — 0.055
2. Capital gain — 0.037
3. Age — 0.034
4. Education level — 0.033

- These are what the model **relies on**
- They do **not** prove what causes someone's income

## 4. Fairness Check

From our Supabase SQL audit on the labeled test rows:

| Group | False Positive Rate | False Negative Rate |
|---|---:|---:|
| Female | 0.027 | **0.415** |
| Male | **0.099** | 0.380 |

**What this means:**
- Women who earn over $50K are **missed more often** (41.5% vs 38.0%)
- Men who earn less are **wrongly marked high more often** (9.9% vs 2.7%)

**Why it happens:**
- The model never sees sex, but "relationship" (Husband/Wife) and marital status act as stand-ins
- Base rates differ: 10.9% of women vs 30.4% of men earn over $50K

**What a responsible deployment would do:**
- Not use this model for real decisions yet
- Try reweighting the training data
- Check and limit proxy features
- Try fairness-aware training
- Keep logging and re-checking every prediction
- *None of these fixes are built yet.*

## 5. Honesty and Integrity: AI and Data

**AI use**
- I used Claude to help write the frontend code
- I directed the work and reviewed what was built
- The full session transcript is saved in `ai-documentation/` so anyone can check it
- I only claim the work I did; Emma's parts are named as hers

**Being honest about testing**
- I wrote the frontend tests in my AI session
- The transcript does not show them being run, so I am not claiming they passed on my side
- The team's GitHub Actions run (set up by Emma) is the record that the suite passes

**Honest data and results**
- Every number in this report comes from the saved project results, not guesses
- The test set was used **only once**, after the model was picked, so the scores are fair
- We show the bad numbers (missed high earners, bias gap), not just the good ones

**Limits we admit**
- The data is from 1994, and $50K meant something very different then
- The data carries old patterns of inequality
- The prediction log stores a hash of inputs, but that is **not** true anonymization
- The website uses only the read-only anon key; the secret key stays on the API

## 6. Faith and Fairness

**The duty:** Deuteronomy 1:17 says, *"You shall not be partial in judgment."* Our model is partial, even if no one meant it to be.

**Who our model treats worse:** women who earn over $50K. It misses 41.5% of them, so their real success often goes unseen.

**Three verses that shape what I think we owe them:**

- **Proverbs 11:1 — honest scales.** God hates a false balance. A model is a kind of scale. If it weighs women and men differently, it is a false balance, and we must fix it before we use it.
- **Proverbs 31:8–9 — speak up for others.** We are told to speak for people who cannot speak for themselves. The women in this data cannot see the model or argue with it. As builders, it is our job to point out the gap for them, which is why the Bias Audit tab shows it openly.
- **Micah 6:8 — do justice, love kindness, walk humbly.**
  - *Justice:* close the error gap before deployment
  - *Kindness:* never let a wrong prediction cost someone a job, loan or home
  - *Humility:* admit the model's limits and let people question its results

**My personal takeaway:** as the person who built the screens people see, I am responsible for what the app shows and what it hides. Showing the unfair numbers clearly is one small way to act with integrity.

## 7. Key Takeaways

- How you **show** results matters as much as the model
- One accuracy number can make a model look fair when it is not
- Putting the confusion matrix and bias audit inside the app makes problems easy to see
- Being honest about AI help and data limits builds trust in the work
