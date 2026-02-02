# Fraud Detection (HackML 2026) — Multi-Class Urgency Prediction

Built during my first HackML hackathon hosted by the :contentReference[oaicite:0]{index=0} at :contentReference[oaicite:1]{index=1}.

We worked with real-world–style transaction data to detect **suspicious / potentially fraudulent transactions** by predicting an **investigation urgency level** (multi-class classification: 0–3).

**Key result:** improved **Macro F1 from 0.82 → 0.88 in ~9 hours**.

---

## Problem Statement

Financial institutions process huge volumes of transactions, but only a tiny fraction are fraudulent. Instead of a simple “fraud / not fraud” label, this competition focuses on **how urgently a transaction should be investigated**:

| Label | Urgency Level | Meaning |
|------:|---------------|---------|
| 0 | No Action | Transaction appears legitimate |
| 1 | Monitor | Low-risk suspicious activity |
| 2 | Review | Likely fraud requiring analyst review |
| 3 | Immediate Action | High-risk fraud requiring urgent response |

This is a **supervised multi-class classification** task on anonymized, transaction-level data with **extreme class imbalance** (≈99.8% non-suspicious).

---

## Evaluation Metric

**Primary metric: Macro F1-score** — computes F1 per class and averages across classes.

Why it matters:
- Treats all urgency levels equally (rare classes still matter)
- Prevents models from “winning” by predicting only the majority class

---

## Approach (What we did)

### 1) Feature + data handling for imbalance
The biggest challenge was the extreme imbalance, so we focused on making the most of the signal we had:

- **Feature selection / pruning**
  - Dropped highly correlated or unhelpful columns (e.g., `id`, `newbalanceOrig`, `newbalanceDest`).  
- **Transformations**
  - Applied `log1p` to skewed numeric features like `amount` and balance fields.  
- **High-leak / low-signal cleanup**
  - Dropped anonymized name/id-like columns (`nameOrig`, `nameDest`).  
- **Categorical encoding**
  - One-hot encoded transaction `type` (e.g., CASH_IN, CASH_OUT, TRANSFER, etc.).

Implementation details are in:
- `config.py` (selected columns, transforms, weights)
- `helpers.py` (preprocessing helpers + training utilities)

### 2) Modeling + iteration under time pressure
We trained and compared multiple models quickly and used **Macro F1** as the selection metric.

What helped us move fast:
- **Iterative parameter tuning** and logging results
- **Early stopping** to avoid wasting time on over-training
- **Subsetting** for faster experiments when needed

Our strongest baseline/solution used **LightGBM multi-class classification** with early stopping and class-balancing options.

---

## Repo Structure

```text
.
├── main.ipynb        # end-to-end training + submission workflow
├── eda.ipynb         # exploration + feature ideas
├── helpers.py        # preprocessing + LightGBM training utilities
├── config.py         # feature lists, transforms, class weights
└── (data/)           # place train/test csv files here (not committed)
