# Machine Learning-Based UAV Intrusion Detection:
# Feature Selection versus Feature Engineering

---

# 1. Project Background

This project studies the impact of Feature Selection (FS) and Feature Engineering / Feature Extraction (FE) techniques on machine learning classifiers for UAV intrusion detection.

The study is inspired by the paper:

"Machine Learning-Based UAV Intrusion Detection:
Feature Selection versus Feature Extraction"

However, the objective of this project is different from simply reproducing the original work.

The project aims to perform a comprehensive comparative study between multiple FS and FE methods across different categories of machine learning classifiers.

---

# 2. Research Motivation

Many UAV intrusion detection studies focus on achieving the highest detection performance.

However, different classifiers may react differently to feature reduction techniques.

Examples:

- Tree-based models may benefit from FS.
- Distance-based models may benefit from FE.
- Neural networks may respond differently from ensemble methods.

Therefore, a systematic comparison is needed.

---

# 3. Research Question (RQ)

RQ:

For the UAV-NIDD dataset, which Feature Selection (FS) or Feature Engineering (FE) technique is most suitable for each machine learning classifier?

Classifiers include:

- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- KNN
- MLP

---

# 4. Optimization Objective

The study does not optimize only F1-score.

Evaluation criteria include:

Detection Performance:
- Precision
- Recall
- F1-score
- PR-AUC

Computational Cost:
- Training Time
- Inference Time

The best method is determined by the trade-off between:

Detection effectiveness
and
Computational efficiency.

Example:

Method A:
F1 = 97.2
Training Time = 15s

Method B:
F1 = 97.3
Training Time = 90s

Although Method B has slightly higher F1-score, Method A may be more practical.

---

# 5. Expected Contribution

Contribution 1

Provide a systematic comparison between FS and FE methods for UAV intrusion detection.

Contribution 2

Identify the most suitable feature reduction technique for each classifier family.

Contribution 3

Analyze the trade-off between detection performance and computational cost.

Contribution 4

Provide practical guidelines for selecting feature reduction methods in UAV IDS systems.

---

# 6. Dataset

Dataset:

UAV-NIDD

Approximate size:

~803,000 samples

~44 original features

Multi-class intrusion detection dataset.

---

# 7. Classifier Categories

The selected classifiers represent different learning paradigms.

## Decision Tree

Representative of:

Single Tree Learning

---

## Random Forest

Representative of:

Bagging Ensemble Learning

---

## XGBoost

Representative of:

Boosting Ensemble Learning

---

## LightGBM

Representative of:

Gradient Boosting Learning

---

## KNN

Representative of:

Distance-Based Learning

---

## MLP

Representative of:

Neural Network-Based Learning

---

# 8. Feature Selection Methods

## Correlation-Based Selection

Category:

Filter Method

---

## Chi-Square

Category:

Filter Method

---

## XGBoost Feature Importance

Category:

Embedded Method

---

## SHAP Feature Importance

Category:

Explainable Embedded Method

---

## Recursive Feature Elimination (RFE)

Category:

Wrapper Method

---

## Consensus Feature Selection

Category:

Hybrid Method

Combines rankings from multiple FS methods.

---

# 9. Feature Engineering / Feature Extraction Methods

## Raw Features

Baseline

No feature reduction

---

## PCA

Principal Component Analysis

Category:

Unsupervised Linear Feature Extraction

---

## LDA

Linear Discriminant Analysis

Category:

Supervised Linear Feature Extraction

---

## Kernel PCA

Category:

Nonlinear Manifold Learning

---

## AutoEncoder

Category:

Deep Nonlinear Feature Extraction

---

## Statistical Feature Generation

Category:

Domain Knowledge-Based Feature Engineering

Examples:

Mean
Variance
Entropy
Packet statistics

---

# 10. Feature Dimension Settings

The study follows the original paper's philosophy.

Number of retained/generated features:

K = 4
K = 8
K = 16

These dimensions are used for both FS and FE methods whenever applicable.

---

# 11. Experimental Protocol

## Dataset Split

Train = 70%

Validation = 15%

Test = 15%

Method:

Stratified Split

Reason:

Maintain class distribution.

---

## Random State

random_state = 42

---

## Validation Usage

Validation Set is used for:

- Hyperparameter tuning
- Model selection

---

## Test Usage

The test set is used only once for final evaluation.

No model selection is allowed on the test set.

---

# 12. Hyperparameter Optimization Strategy

The project does not tune every FS/FE combination independently.

Instead:

Step 1

Optimize each classifier on raw features only.

Raw Features
→ Hyperparameter Search
→ Validation Set

Result:

DT_BEST

RF_BEST

XGB_BEST

LGBM_BEST

KNN_BEST

MLP_BEST

---

Step 2

Fix these parameters.

Reuse them in every FS and FE experiment.

Reason:

Ensure fair comparison.

Performance differences should come from FS/FE methods rather than hyperparameter changes.

---

# 13. Experimental Pipeline

Phase 1

Dataset Preparation

↓

Preprocessing

↓

Train / Validation / Test Split

---

Phase 2

Hyperparameter Optimization

(Raw Features Only)

↓

DT_BEST

RF_BEST

XGB_BEST

LGBM_BEST

KNN_BEST

MLP_BEST

---

Phase 3

Feature Reduction

↓

FS Branch

or

FE Branch

↓

K = 4 / 8 / 16

---

Phase 4

Classification

↓

DT_BEST

RF_BEST

XGB_BEST

LGBM_BEST

KNN_BEST

MLP_BEST

---

Phase 5

Repeated Runs

5 Seeds

Example:

42
52
62
72
82

---

Phase 6

Evaluation

Accuracy

Precision

Recall

F1-score

PR-AUC

Training Time

Inference Time

---

Phase 7

Comparative Analysis

Determine:

Best FS for each classifier.

Best FE for each classifier.

Best overall feature reduction strategy.

---

# 14. Baseline Study

Before running FS and FE experiments:

Run all classifiers using raw features only.

Purpose:

Verify:

- Data preprocessing
- Data splitting
- Evaluation metrics
- Model training pipeline

Baseline Table:

Model

DT

RF

XGB

LGBM

KNN

MLP

Metrics:

Accuracy

Precision

Recall

F1-score

Training Time

Inference Time

---

# 15. Reviewer Requirements

The project must address:

✓ Related Work with comparison table

✓ Additional SOTA models if possible

✓ Correct use of LDA

✓ No model selection using test set

✓ Complete preprocessing documentation

✓ Repeated runs

✓ Mean ± Std

✓ Statistical significance tests

✓ Runtime study

✓ Reproducible code and configurations

✓ Per-class metrics

✓ FPR/FNR

✓ PR-AUC

✓ Public repository

---

# 16. Final Goal

The final objective is not simply to obtain the highest F1-score.

The final objective is to answer:

"Which feature reduction technique is most suitable for each classifier family in UAV intrusion detection considering both detection effectiveness and computational efficiency?"