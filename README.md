# Integrating Blood-Based Transcriptomics and Explainable Machine Learning to Predict Parkinson's Disease Motor Progression

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Ensemble-green.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-orange.svg)](https://github.com/slundberg/shap)
[![Status](https://img.shields.io/badge/Status-Research-yellow.svg)]()

> A machine learning framework for predicting 12-month motor progression in Parkinson's disease using baseline blood-based RNA-seq and clinical data, with SHAP-based explainability.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Performance Metrics](#performance-metrics)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [SHAP Analysis](#shap-analysis)
- [Citation](#citation)
- [Contact](#contact)

---

## 🔬 Overview

Parkinson's disease (PD) exhibits significant heterogeneity in progression rates across patients. This project presents an **explainable machine learning framework** that integrates:

- **Blood-based transcriptomics** (RNA-seq from PPMI dataset)
- **Clinical features** (baseline UPDRS Part III, age, gender)
- **PD risk genes** (SNCA, LRRK2, GBA, PRKN, PINK1, PARK7, VPS35)
- **Biological pathways** (mitochondrial dysfunction, neuroinflammation, autophagy)
- **Gene-clinical interactions** (e.g., UPDRS_BL × PINK1)

The model predicts **12-month motor outcomes (UPDRS Part III scores)** with **R²=0.551** on an independent clinical test set, providing clinically actionable insights through SHAP analysis.

### Key Findings

1. **UPDRS_BL × PINK1 interaction** is the most important predictor (SHAP=0.283)
2. **Mitochondrial dysfunction** pathway shows highest contribution among biological processes (SHAP=0.008)
3. **VPS35** is the most important individual PD risk gene (SHAP=0.010)
4. Model achieves **MAE=6.01 UPDRS points**, enabling risk stratification for clinical decision support

---

## ✨ Key Features

### Machine Learning
- **Stacking Ensemble** with 3 gradient boosting models (XGBoost, LightGBM, CatBoost)
- **Huber Regressor** meta-learner for robustness to outliers
- **Bayesian hyperparameter optimization** using Optuna (30 trials)
- **7-fold cross-validation** for robust performance estimation
- **Stratified train-test split** (80/20) based on progression status

### Explainability
- **SHAP (SHapley Additive exPlanations)** for feature importance
- **Three-panel SHAP analysis**: Clinical features, PD risk genes, Pathway scores
- **Interaction effects** quantified and visualized
- **Biological interpretation** of predictions

### Clinical Application
- **Interactive prediction tool** for individual patients
- **Risk categorization**: Stable, Mild, Moderate, Rapid progression
- **95% confidence intervals** for predictions
- **Minimal input requirements**: Only baseline UPDRS, age, and gender

---

## 📊 Performance Metrics

### Independent Clinical Test Set (n=78)

| Metric | Value |
|--------|-------|
| **R² Score** | **0.551** |
| **MAE** | **6.01 UPDRS points** |
| **RMSE** | 7.45 UPDRS points |
| **Pearson r** | 0.74 |

### 7-Fold Cross-Validation (n=312)

| Metric | Value |
|--------|-------|
| **R² Score** | 0.513 ± 0.052 |
| **MAE** | 6.15 ± 0.25 UPDRS points |
| **RMSE** | 7.82 ± 0.31 UPDRS points |

### Clinical Interpretation

- **MAE=6.01 points** is clinically meaningful (UPDRS Part III range: 0-108)
- **Minimal clinically important difference (MCID)** for UPDRS Part III: ~5 points
- Model predictions enable **risk stratification** for treatment planning

---

## 📁 Repository Structure

```
.
├── 0.513_Parkinson_Optimized_Model_Package.zip
│   ├── parkinson_optimized_model_package/
│   │   ├── codes/
│   │   │   ├── lightweight_optimization.py          # Main training script
│   │   │   ├── generate_cv_predictions.py           # Cross-validation predictions
│   │   │   └── predict_new_patient.py               # Prediction for new patients
│   │   ├── data/
│   │   │   └── example_data.csv                     # Example patient data
│   │   ├── model/
│   │   │   └── lightweight_optimized_model.pkl      # Trained model (582 KB)
│   │   └── visualization/
│   │       ├── plot_shap_three_panel_clinical.py    # SHAP analysis plots
│   │       ├── plot_scatter_simple.py               # Prediction scatter plots
│   │       ├── plot_residual_0551.py                # Residual analysis
│   │       └── plot_feature_importance.py           # Feature importance plots
│   └── Parkinson_Clinical_Decision_Support/
│       ├── scripts/
│       │   └── predict_patient.py                   # Clinical prediction tool
│       ├── docs/
│       │   ├── USER_GUIDE.md                        # Comprehensive user guide
│       │   └── TECHNICAL_DETAILS.md                 # Technical documentation
│       ├── examples/
│       │   └── example_patients.csv                 # Example patient data
│       └── README.md                                # Quick start guide
│
├── Parkinson_Model_Pseudocode.md.pdf                # Detailed pseudo code (70+ pages)
│
└── Copyright_license.docx                           # Copyright information
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11** or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 4 GB (8 GB recommended)
- **Disk Space**: ~1 GB

### Installation

1. **Extract the package**

```bash
unzip 0.513_Parkinson_Optimized_Model_Package.zip
cd parkinson_optimized_model_package
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

Required packages:
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `scikit-learn >= 1.3.0`
- `xgboost >= 2.0.0`
- `lightgbm >= 4.0.0`
- `catboost >= 1.2.0`
- `shap >= 0.42.0`
- `optuna >= 3.3.0`
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`

---

## 💻 Usage

### Option 1: Interactive Prediction (Recommended for Single Patients)

```bash
cd Parkinson_Clinical_Decision_Support/scripts
python predict_patient.py --interactive
```

**Example interaction:**

```
INTERACTIVE PATIENT PREDICTION MODE
====================================================================

Please enter patient baseline data:

Patient ID (optional): PATIENT_001
Baseline UPDRS Part III score (required): 20
Age in years (required): 68
Gender (0=Female, 1=Male, required): 1

🔮 Making prediction...

PREDICTION RESULTS
====================================================================

Patient ID: PATIENT_001
Baseline UPDRS Part III: 20.0

Predicted UPDRS at 12 months: 26.5
   95% Confidence Interval: [20.5, 32.5]

Predicted Change: +6.5 points
Progression Risk: Moderate Progression

Clinical Interpretation:
   Moderate progression expected (6.5 points). 
   Consider treatment adjustment.
====================================================================

Save results to file? (y/n): y
✅ Results saved to: results/prediction_PATIENT_001.csv
```

### Option 2: Batch Prediction from CSV

1. **Prepare patient data** (`patients.csv`):

```csv
PATNO,UPDRS_BL,AGE,GENDER
PATIENT_001,20.0,68.0,1.0
PATIENT_002,25.0,72.0,0.0
PATIENT_003,10.0,55.0,1.0
```

2. **Run batch prediction:**

```bash
python predict_patient.py --input patients.csv --output predictions.csv
```

3. **View results** in `results/predictions.csv`:

```csv
PATNO,UPDRS_BL,AGE,GENDER,PREDICTED_UPDRS_12M,PREDICTED_CHANGE,LOWER_BOUND,UPPER_BOUND,PROGRESSION_RISK
PATIENT_001,20.0,68.0,1.0,26.5,6.5,20.5,32.5,Moderate Progression
PATIENT_002,25.0,72.0,0.0,27.0,2.0,21.0,33.0,Stable
PATIENT_003,10.0,55.0,1.0,17.1,7.1,11.1,23.1,Moderate Progression
```

### Option 3: Python API

```python
import joblib
import pandas as pd
import numpy as np

# Load model package
model_package = joblib.load('model/lightweight_optimized_model.pkl')

# Prepare patient data
patient_data = {
    'PATNO': 'PATIENT_001',
    'UPDRS_BL': 20.0,
    'AGE': 68.0,
    'GENDER': 1.0  # 0=Female, 1=Male
}

# Extract model artifacts
model = model_package['ensemble_model']
scaler = model_package['scaler']
target_transformer = model_package['target_transformer']
feature_names = model_package['feature_names']

# Create feature vector (impute missing features with 0)
patient_df = pd.DataFrame([patient_data])
for feature in feature_names:
    if feature not in patient_df.columns:
        patient_df[feature] = 0.0

patient_df = patient_df[feature_names]

# Scale and predict
X_patient = patient_df.values
X_patient_scaled = scaler.transform(X_patient)
y_pred_trans = model.predict(X_patient_scaled)

# Inverse transform to UPDRS scale
y_pred_updrs = target_transformer.inverse_transform(
    y_pred_trans.reshape(-1, 1)
).flatten()[0]

print(f"Predicted UPDRS at 12 months: {y_pred_updrs:.1f}")
print(f"Predicted change: {y_pred_updrs - patient_data['UPDRS_BL']:.1f} points")
```

---

## 🏗️ Model Architecture

### Stacking Ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                    STACKING REGRESSOR                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: 116 Features                                            │
│    ├── Clinical (3): UPDRS_BL, AGE, GENDER                     │
│    ├── Top Genes (100): Selected by correlation with ΔUPDRS    │
│    ├── PD Risk Genes (7): SNCA, LRRK2, GBA, PRKN, PINK1,       │
│    │                       PARK7, VPS35                         │
│    ├── Pathways (3): Inflammation, Mitochondrial, Autophagy    │
│    └── Interactions (3): PINK1×PARK7, AGE×PINK1, UPDRS×PINK1   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              LEVEL 0: BASE MODELS                       │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │   │
│  │  │   XGBoost    │  │   LightGBM   │  │   CatBoost   │ │   │
│  │  │              │  │              │  │              │ │   │
│  │  │ • Pseudo-    │  │ • Huber      │  │ • RMSE       │ │   │
│  │  │   Huber loss │  │   objective  │  │   loss       │ │   │
│  │  │ • ~200 trees │  │ • ~200 trees │  │ • ~200 iters │ │   │
│  │  │ • L1+L2 reg  │  │ • L1+L2 reg  │  │ • L2 reg     │ │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │   │
│  │         │                 │                 │         │   │
│  │         └─────────────────┼─────────────────┘         │   │
│  │                           │                           │   │
│  └───────────────────────────┼───────────────────────────┘   │
│                              │                               │
│  ┌───────────────────────────▼───────────────────────────┐   │
│  │         LEVEL 1: META-LEARNER (Huber Regressor)       │   │
│  ├───────────────────────────────────────────────────────┤   │
│  │                                                       │   │
│  │  • Combines base model predictions                   │   │
│  │  • Robust to outliers (Huber loss)                   │   │
│  │  • L2 regularization                                  │   │
│  │                                                       │   │
│  └───────────────────────────┬───────────────────────────┘   │
│                              │                               │
│  OUTPUT: Predicted UPDRS Part III at 12 months               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Training Pipeline

1. **Data Preprocessing**
   - Merge clinical and RNA-seq data (n=390 patients)
   - Remove outliers using IQR method (2 patients removed)
   - Impute missing values with median

2. **Feature Selection**
   - Select top 100 genes by correlation with ΔUPDRS
   - Include 7 PD risk genes
   - Compute 3 pathway scores
   - Create 3 interaction features

3. **Data Splitting**
   - Stratified split: 80% train+val (n=312), 20% test (n=78)
   - Stratification based on progression status (ΔUPDRS ≥ 5)

4. **Hyperparameter Optimization**
   - Bayesian optimization using Optuna (30 trials)
   - 7-fold cross-validation for each trial
   - Optimize for R² score

5. **Final Training**
   - Train on full training set with best hyperparameters
   - Validate on independent test set

6. **SHAP Analysis**
   - Compute SHAP values for test set
   - Analyze feature importance by category

---

## 🔍 SHAP Analysis

### Top Features by SHAP Importance

| Rank | Feature | SHAP Value | Category |
|------|---------|------------|----------|
| 1 | UPDRS_BL × PINK1 | 0.283 | Interaction |
| 2 | UPDRS_BL | 0.258 | Clinical |
| 3 | ENSG00000243053 | 0.025 | Top Gene |
| 4 | ENSG00000176422 | 0.022 | Top Gene |
| 5 | ENSG00000255872 | 0.020 | Top Gene |
| ... | ... | ... | ... |

### Feature Categories

```
┌─────────────────────────────────────────────────────────────────┐
│              SHAP IMPORTANCE BY CATEGORY                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Clinical Features:                                             │
│    ├── UPDRS_BL × PINK1: 0.283 ████████████████████████████    │
│    ├── UPDRS_BL: 0.258         ██████████████████████████      │
│    ├── AGE × PINK1: 0.003      █                               │
│    ├── AGE: 0.001              ▌                               │
│    └── GENDER: 0.001           ▌                               │
│                                                                 │
│  PD Risk Genes:                                                 │
│    ├── VPS35: 0.010            ████                            │
│    ├── GBA: 0.005              ██                              │
│    ├── LRRK2: 0.005            ██                              │
│    ├── PRKN: 0.004             █▌                              │
│    ├── PARK7: 0.003            █                               │
│    ├── PINK1: 0.003            █                               │
│    └── SNCA: 0.002             ▌                               │
│                                                                 │
│  Pathway Scores:                                                │
│    ├── Mitochondrial: 0.008    ███                             │
│    ├── Inflammation: 0.005     ██                              │
│    └── Autophagy: 0.003        █                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Insights

1. **Gene-Clinical Interactions Dominate**
   - UPDRS_BL × PINK1 is the most important feature
   - Suggests baseline motor severity modulates genetic effects

2. **Mitochondrial Dysfunction is Central**
   - Mitochondrial pathway has highest SHAP among pathways
   - VPS35 (endosomal trafficking) is top PD risk gene
   - Supports mitochondrial quality control hypothesis

3. **Demographics Have Minimal Impact**
   - Age and gender contribute <0.002
   - Disease-specific features dominate

---

## 📖 Documentation

### Main Files

1. **`0.513_Parkinson_Optimized_Model_Package.zip`**
   - Complete model package with training code, trained model, and prediction tools
   - Size: ~1 MB (compressed)
   - Contains:
     - Training scripts (`lightweight_optimization.py`)
     - Trained model (`lightweight_optimized_model.pkl`, 582 KB)
     - Prediction tools (`predict_new_patient.py`, `predict_patient.py`)
     - Visualization scripts (SHAP, scatter plots, residuals)
     - Example data and documentation

2. **`Parkinson_Model_Pseudocode.md.pdf`**
   - Comprehensive pseudo code documentation 
   - Includes:
     - 9 detailed algorithms (training, preprocessing, SHAP, prediction)
     - System architecture diagrams
     - Computational complexity analysis
     - Performance metrics summary
   - Format: Academic-style pseudo code suitable for publication

3. **`Copyright_license.docx`**
   - Copyright and licensing information

### Additional Resources

- **User Guide** (`docs/USER_GUIDE.md`): Step-by-step instructions for clinical use
- **Technical Details** (`docs/TECHNICAL_DETAILS.md`): In-depth technical documentation
- **Quick Start** (`QUICK_START.md`): 5-minute getting started guide

---

## 📚 Citation

If you use this code or model in your research, please cite:

```bibtex
@article{parkinson_ml_2025,
  title={Integrating Blood-Based Transcriptomics and Explainable Machine Learning to Predict Parkinson's Disease Motor Progression},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2025},
  volume={[Volume]},
  pages={[Pages]},
  doi={[DOI]}
}
```

---

## 🔬 Research Context

### Dataset

- **Source**: Parkinson's Progression Markers Initiative (PPMI)
- **Patients**: 390 (after outlier removal)
- **Features**: 116 (clinical + genomic + pathways + interactions)
- **Outcome**: UPDRS Part III at 12 months
- **Data Type**: Baseline blood RNA-seq + clinical assessments

### Clinical Significance

- **Prediction Horizon**: 12 months
- **Clinical Utility**: Risk stratification for treatment planning
- **Minimal Input**: Only baseline UPDRS, age, and gender required for prediction
- **Explainability**: SHAP analysis reveals biological mechanisms

### Limitations

- **Cross-sectional prediction**: Does not use longitudinal data
- **Single outcome**: UPDRS Part III only (motor symptoms)
- **Dataset**: PPMI cohort may not generalize to all PD populations
- **RNA-seq**: Model trained on blood RNA-seq; clinical prediction uses imputation

---

## Contact

For questions, issues, or collaboration inquiries:

- **Email**: [yelda.firat@mudanya.edu.tr]

---

## 🙏 Acknowledgments

- **PPMI**: Parkinson's Progression Markers Initiative for providing the dataset
- **Open Source Community**: scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, Optuna
- ** Thanks to Meral Seferoğlu from the Department of Neurology, University of Health Sciences, Bursa Yüksek Ihtisas Training and Research Hospital, Bursa, Turkey.

---

## ⚠️ Disclaimer

This tool is intended for **research purposes only** and should not be used as the sole basis for clinical decision-making. Always consult with qualified healthcare professionals for medical advice and treatment decisions.

---

**Status**: Research Code
