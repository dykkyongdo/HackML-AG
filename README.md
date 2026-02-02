# Fraud Detection — HackML 2026 (Multi-Class Urgency Prediction)

This project was built during my first **HackML hackathon** organized by the **SFU Data Science Student Society**.

We worked with real-world–style transactions data and applied machine learning to detect suspicious and potentially fraudulent transactions by predicting an **investigation urgency level** (multi-class classification: **0–3**).

**Key result:** improved **Macro F1 from 0.82 → 0.88 in ~9 hours**.

---

## Table of Contents
- [Problem](#problem)
- [Dataset](#dataset)
- [Evaluation Metric](#evaluation-metric)
- [Approach](#approach)
- [Results](#results)
- [Repo Structure](#repo-structure)
- [How to Run](#how-to-run)
- [Submission Format](#submission-format)
- [Team](#team)
- [Acknowledgements](#acknowledgements)
- [Resume Bullets](#resume-bullets)
- [Citation](#citation)

---

## Problem

Financial institutions process millions of transactions daily, but only a small fraction are fraudulent. In practice, fraud teams don’t only ask **“Is this fraud?”** — they decide **how urgently** a transaction should be investigated given limited analyst resources.

This competition frames fraud detection as a **multi-class classification** task to predict the **urgency level** of investigation:

| Label | Urgency Level        | Business Context |
|------:|----------------------|------------------|
| 0     | No Action            | Transaction appears legitimate |
| 1     | Monitor              | Low-risk suspicious activity |
| 2     | Review               | Likely fraud requiring analyst review |
| 3     | Immediate Action     | High-risk fraud requiring urgent response |

---

## Dataset
Dataset (Kaggle): https://www.kaggle.com/competitions/fraud-hack-ml-2026/data

- Each row represents a **single transaction**
- Features include:
  - Transaction details (amount, time, frequency)
  - User behavior indicators
  - Merchant characteristics
  - Device/channel information
- Target: **`urgency_level`** (integer class label **0–3**)
- The dataset is intentionally **extremely imbalanced** (≈ **99.8%** non-suspicious)

### Columns
- `step`: unit of time in the simulation (1 step = 1 hour since the start)
- `type`: transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
- `amount`: transaction amount
- `oldbalanceOrg`, `newbalanceOrig`: origin account balances before/after
- `oldbalanceDest`, `newbalanceDest`: destination balances before/after (not applicable for merchants)
- `nameOrig`, `nameDest`: anonymized account identifiers
- `urgency_level`: target variable (0–3)

---

## Evaluation Metric

**Primary metric: Macro F1-score**

Macro F1 computes the F1-score for each class independently, then averages them:

- Treats all urgency levels equally
- Prevents models from ignoring rare but critical fraud cases
- Reflects real-world fraud prioritization needs

---

## Approach

### Main Challenge: Extreme Class Imbalance
~99.8% of transactions weren’t suspicious, so optimizing for accuracy would lead to a model that ignores minority classes. We focused on techniques that improve minority-class performance under time constraints.

### What We Did
- **Carefully selected variables** and removed low-signal / identifier-like features
- **Derived and transformed features**, including log transforms for skewed numeric values
- **Used Macro F1** for model selection and performance evaluation
- **Iteratively tuned parameters** across multiple models and logged results
- **Early stopping** to train efficiently and avoid overfitting
- **Subsetting** for faster experimentation under a tight hackathon timeline

### Model
Our strongest solution used a **LightGBM multi-class classifier** with:
- Early stopping
- Class balancing strategies
- Fast hyperparameter iteration

---

## Results

- Started around **Macro F1 ≈ 0.82**
- Improved to **Macro F1 ≈ 0.88**
- Achieved within **~9 hours** during the hackathon

---

## Repo Structure

```text
.
├── main.ipynb        # end-to-end training + validation + submission workflow
├── eda.ipynb         # exploration + feature ideas
├── helpers.py        # preprocessing + model training utilities
├── config.py         # feature lists, transforms, optional class weights
└── data/             # put train/test CSVs here (not committed)

```

## How to Run

### 1) Create an environment & install dependencies

```bash
pip install -r requirements.txt
```
If you don’t have requirements.txt, a minimal setup is:
```bash
pip install numpy pandas scikit-learn lightgbm jupyter
```
### 2) Add the dataset
Create a folder like:
```text
data/
  train.csv
  test.csv
```
### 3) Run notebooks
```bash
jupyter notebook
```
Then run:
``` text
eda.ipynb
```
for exploration and feature hypotheses
```text
main.ipynb
```
for training, validation, and generating the submission file

#### Submission Format
The competition expects a CSV like:
``` text
id,urgency_level
1,0
2,0
3,0
...
```
Your pipeline should produce predictions for every id in the test set.

## Team
- Kirill Markin

- Dyk Kyong Do

- Nazar Lytvinchuk

## Acknowledgements
Thanks to SFU Data Science Student Society for organizing HackML and providing an awesome real-world ML experience.
