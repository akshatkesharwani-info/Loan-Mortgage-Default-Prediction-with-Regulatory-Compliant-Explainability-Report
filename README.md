# Loan/Mortgage Default Prediction with Regulatory-Compliant Explainability Report

 ML Engineer Track |

100% free and open-source tools only — no paid APIs, no credit card, runs on Google Colab's free tier.

---

## 📌 Overview

Regulators (under ECOA/Reg B) require lenders to give applicants specific reasons for a credit denial. Build a mortgage default model plus an automated adverse-action reason generator that turns SHAP values into the plain-English 'reason codes' a compliant denial letter must include -- entirely with free, open-source tools.

**Domain:** Consumer Lending / Regulatory Compliance

---

## 📊 Dataset

HMDA mortgage data or any loan-level default dataset (auto-generates sample data if file is missing)

**Source:** [https://ffiec.cfpb.gov/data-publication/](https://ffiec.cfpb.gov/data-publication/)

> This script also includes a **safe fallback**: if the real dataset file isn't found next to the
> notebook/script, it automatically generates a small realistic sample dataset with the same column
> names, so the whole pipeline still runs end-to-end even before you've downloaded the real data.

---

## 🛠️ Tech Stack

Python 3 | scikit-learn | XGBoost | SHAP | reportlab

**Skills demonstrated:** Python, scikit-learn, SHAP, ReportLab, Adverse Action Reasoning

---

## 🎯 What This Project Builds

- A logistic regression + XGBoost comparison for mortgage default prediction
- SHAP-based feature attribution for every individual loan decision
- An adverse-action reason mapper that converts top negative SHAP features into standard reason codes
- A compliant decision report per applicant: approve/deny + top reason codes
- A model-level fairness check across a protected-class proxy variable (illustrative, not legal advice)
- A PDF explainability report ready for a compliance file

---

## 🧭 Step-by-Step Approach

### Step 1: Load Data (with a Safe Fallback)

**What:** Load mortgage_data.csv, or auto-build a realistic sample dataset if the file isn't found

**Why:** A missing dataset file shouldn't stop the compliance pipeline from being testable end-to-end

**How:** if os.path.exists(path): pd.read_csv(path) else: generate synthetic mortgage rows with numpy


### Step 2: Train Two Comparable Models

**What:** Fit both logistic regression (interpretable baseline) and XGBoost (higher accuracy)

**Why:** Regulators often want to see a simpler model's coefficients alongside any complex model's explanations

**How:** LogisticRegression().fit(...) and XGBClassifier().fit(...) on the same feature set


### Step 3: SHAP Attribution per Applicant

**What:** Compute SHAP values for every test-set applicant's XGBoost prediction

**Why:** SHAP gives a mathematically consistent per-applicant reason for the model's decision

**How:** shap.TreeExplainer(model).shap_values(X_test)


### Step 4: Map to Adverse-Action Reason Codes & Report

**What:** Rank each applicant's top negative SHAP features, map to reason codes, and export a PDF

**Why:** Reg B requires denial letters to state specific principal reasons, not just 'model score too low'

**How:** reason_map = {'dti':'Debt-to-income ratio too high', ...}; reportlab platypus for the PDF


---

## 📈 Dashboard / Reporting Ideas

- Table: all applicants with decision, default probability, and top reason codes
- Bar chart: frequency of each adverse-action reason code across all denials this period
- KPI cards: approval rate, average default probability of approved vs denied
- Comparison chart: logistic regression coefficients vs XGBoost SHAP importances for the same features
- Card: model AUC and a note on last recalibration date, for a model-governance log

---

## 💡 Key Insights

- SHAP values map naturally onto adverse-action reason codes because both rank features by contribution to the decision
- Keeping a simpler logistic regression alongside XGBoost helps satisfy regulators who want an interpretable benchmark
- Reason codes should always be the top 2-4 factors, not every feature, to match real Reg B denial letter practice
- This is an educational implementation of the explainability *pattern* used in real lending compliance -- always pair with actual legal/compliance review before production use
- Free, open-source SHAP is now the de facto standard explainability tool across the credit risk industry

---

## 🚀 How to Run

1. Open the `.py` file in **Google Colab** (free tier — no GPU or paid compute needed) or run it locally with Python 3.
2. Install dependencies with the `pip install ...` line at the top of the script (all free, open-source packages).
3. (Optional) Download the real dataset from the Kaggle link above and place it in the same folder — the filename the script expects is noted in the code's data-loading step. If you skip this, the script auto-generates sample data so you can still see it run.
4. Run the script top to bottom. Outputs (charts, CSVs, model files) are saved in the working directory.

```bash
pip install -r requirements.txt   # or the pip install line at the top of the script
python MLEng_05_Loan_Mortgage_Default_Explainability_Report.py
```

---

## 📂 Repo Structure

```
loan-mortgage-default-prediction-regulatory-compliant-explainability/
├── MLEng_05_Loan_Mortgage_Default_Explainability_Report.py       # complete, runnable, free-only solution code
├── README.md              # this file
└── outputs/                # charts, CSVs, and model files generated on run
```

---

## ⚠️ Disclaimer

This project is built for educational and portfolio purposes to demonstrate applied ML/quant-risk
skills. It is not financial, credit, or investment advice, and should not be used for real lending,
trading, or compliance decisions without proper review by a licensed professional.

---

*Part of a 20-project AI Engineer + ML Engineer portfolio focused on finance and consulting use cases —
built entirely with free, open-source tools.*
