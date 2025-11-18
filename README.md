# Explainable Ensemble Learning for Parkinson's Disease Motor Progression Prediction

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Ensemble-green.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-orange.svg)](https://github.com/slundberg/shap)
[![Dataset](https://img.shields.io/badge/Data-PPMI-red.svg)](https://www.ppmi-info.org/)

> A SHAP-based explainable machine learning framework for predicting 12-month motor progression in Parkinson's disease using baseline blood transcriptomics and clinical data.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Performance Metrics](#performance-metrics)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Model Architecture](#model-architecture)
- [SHAP Analysis](#shap-analysis)
- [Citation](#citation)
- [Contact](#contact)

---

## 🔬 Overview

Parkinson's disease (PD) exhibits significant heterogeneity in motor progression rates across patients. This project presents an **explainable stacking ensemble framework** that integrates:

**Data Sources:**
- **Blood-based transcriptomics** (RNA-seq from PPMI dataset)
- **Clinical features** (baseline UPDRS Part III, age, gender)
- **PD risk genes** (SNCA, LRRK2, GBA, PRKN, PINK1, PARK7, VPS35)
- **Biological pathways** (mitochondrial dysfunction, neuroinflammation, autophagy)
- **Gene-clinical interactions** (e.g., UPDRS_BL × PINK1)

**Model Output:**
- Predicts **12-month motor outcomes (UPDRS Part III scores)**
- Achieves **R²=0.551** and **MAE=6.01** on independent clinical test set
- Provides **SHAP-based explanations** for individual predictions

### Key Findings

1. **UPDRS_BL × PINK1 interaction** is the strongest predictor (mean |SHAP|=0.283)
2. **Mitochondrial dysfunction** pathway shows highest contribution among biological processes (mean |SHAP|=0.008)
3. **VPS35** is the most important individual PD risk gene (mean |SHAP|=0.010)
4. **Baseline UPDRS** alone has mean |SHAP|=0.258, but interaction with PINK1 increases predictive power

---

## ✨ Key Features

### Machine Learning Architecture
- **Stacking Ensemble** with 3 gradient boosting base models:
  - XGBoost (extreme gradient boosting)
  - LightGBM (light gradient boosting machine)
  - CatBoost (categorical boosting)
- **Huber Regressor** meta-learner for robustness to outliers
- **Bayesian hyperparameter optimization** using Optuna (30 trials)
- **7-fold cross-validation** for robust performance estimation
- **Stratified train-test split** (80/20) based on progression status

### Explainability & Interpretability
- **SHAP (SHapley Additive exPlanations)** for feature importance quantification
- **Three-panel SHAP analysis**:
  - Clinical features & interactions
  - PD risk genes
  - Biological pathway scores
- **Gene-clinical interaction effects** explicitly modeled and explained
- **Biological interpretation** linking predictions to known PD mechanisms

### Clinical Decision Support
- **Interactive prediction tool** for individual patients
- **Risk categorization**: Stable, Mild, Moderate, Rapid progression
- **95% confidence intervals** for predictions
- **Minimal input requirements**: Only baseline UPDRS, age, and gender needed
- **Batch processing** capability for multiple patients

---

## 📊 Performance Metrics

### Independent Clinical Test Set (n=78)

| Metric | Value | Clinical Interpretation |
|--------|-------|------------------------|
| **R² Score** | **0.551** | Explains 55.1% of variance in 12-month motor outcomes |
| **MAE** | **6.01** | Average prediction error of 6 UPDRS points |
| **RMSE** | **7.21** | Root mean squared error |
| **Pearson r** | **0.74** | Strong correlation between predicted and actual scores |

### 7-Fold Cross-Validation (n=312)

| Metric | Value |
|--------|-------|
| **R² Score** | 0.513 ± 0.052 |
| **MAE** | 6.15 ± 0.25 |
| **RMSE** | 7.86 ± 0.57 |

### Clinical Significance

- **MAE=6.01 points** is clinically meaningful (UPDRS Part III range: 0-108)
- **Minimal clinically important difference (MCID)** for UPDRS Part III: ~5 points
- Model predictions enable **personalized risk stratification** for treatment planning
- **Baseline-only approach** allows immediate prognostic assessment at diagnosis

---

## 📁 Repository Structure

```
.
├── parkinson_optimized_model_package/
│   ├── codes/
│   │   ├── lightweight_optimization.py          # Main training script
│   │   ├── generate_cv_predictions.py           # Cross-validation predictions
│   │   └── predict_new_patient.py               # Prediction for new patients
│   ├── data/
│   │   ├── example_data.csv                     # Example patient data
│   │   ├── shap_values_real.csv                 # SHAP values for all features
│   │   └── feature_importance_top20.csv         # Top 20 feature importance
│   ├── model/
│   │   └── lightweight_optimized_model.pkl      # Trained ensemble model (582 KB)
│   ├── results/
│   │   ├── lightweight_optimization_results.csv # Model performance metrics
│   │   └── lightweight_cv_detailed.csv          # Detailed CV results
│   ├── visualization/
│   │   ├── create_figure1_6panels.py            # Main figure generation
│   │   ├── create_shap_500dpi.py                # SHAP visualization
│   │   ├── plot_shap_three_panel_clinical.py    # 3-panel SHAP analysis
│   │   └── plot_feature_importance.py           # Feature importance plots
│   └── documentation/
│       └── model_documentation.md               # Detailed model documentation
│
└── Parkinson_Clinical_Decision_Support/
    ├── scripts/
    │   └── predict_patient.py                   # Clinical prediction tool
    ├── examples/
    │   └── example_patients.csv                 # Example patient data
    ├── results/
    │   └── test_predictions.csv                 # Example predictions
    └── docs/
        ├── USER_GUIDE.md                        # Comprehensive user guide
        └── TECHNICAL_DETAILS.md                 # Technical documentation
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.11** or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 4 GB (8 GB recommended for training)
- **Disk Space**: ~1 GB

### Step 1: Clone Repository

```bash
git clone https://github.com/yeldafrt/PD-Blood-Transcriptomics-SHAP.git
cd PD-Blood-Transcriptomics-SHAP
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
```
pandas >= 2.0.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
xgboost >= 2.0.0
lightgbm >= 4.0.0
catboost >= 1.2.0
shap >= 0.42.0
optuna >= 3.3.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
joblib >= 1.3.0
```

---

## 🏃 Quick Start

### Option 1: Interactive Prediction (Single Patient)

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
   Consider treatment adjustment and closer monitoring.
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

---

## 💻 Usage Examples

### Python API for Custom Integration

```python
import joblib
import pandas as pd
import numpy as np

# Load trained model package
model_package = joblib.load('parkinson_optimized_model_package/model/lightweight_optimized_model.pkl')

# Prepare patient data
patient_data = {
    'PATNO': 'PATIENT_001',
    'UPDRS_BL': 20.0,
    'AGE': 68.0,
    'GENDER': 1.0  # 0=Female, 1=Male
}

# Extract model components
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

# Scale features and make prediction
X_patient = patient_df.values
X_patient_scaled = scaler.transform(X_patient)
y_pred_trans = model.predict(X_patient_scaled)

# Inverse transform to UPDRS scale
y_pred_updrs = target_transformer.inverse_transform(
    y_pred_trans.reshape(-1, 1)
).flatten()[0]

# Calculate predicted change
predicted_change = y_pred_updrs - patient_data['UPDRS_BL']

print(f"Predicted UPDRS at 12 months: {y_pred_updrs:.1f}")
print(f"Predicted change: {predicted_change:+.1f} points")

# Categorize progression risk
if predicted_change < 3:
    risk = "Stable"
elif predicted_change < 6:
    risk = "Mild Progression"
elif predicted_change < 10:
    risk = "Moderate Progression"
else:
    risk = "Rapid Progression"

print(f"Progression risk: {risk}")
```

### SHAP Explanation for Individual Prediction

```python
import shap

# Load SHAP explainer (if available in model package)
explainer = model_package.get('shap_explainer')

if explainer is None:
    # Create new explainer
    explainer = shap.Explainer(model, X_patient_scaled)

# Calculate SHAP values
shap_values = explainer(X_patient_scaled)

# Visualize feature contributions
shap.waterfall_plot(shap_values[0])
```

---

## 🏗️ Model Architecture

### Stacking Ensemble Framework

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
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │   XGBoost    │  │  LightGBM    │  │  CatBoost    │  │   │
│  │  │              │  │              │  │              │  │   │
│  │  │ Prediction 1 │  │ Prediction 2 │  │ Prediction 3 │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           LEVEL 1: META-LEARNER                         │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │                                                         │   │
│  │              Huber Regressor                            │   │
│  │         (Robust to outliers)                            │   │
│  │                                                         │   │
│  │  Optimal linear weighting of base model predictions    │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│                                                                 │
│  OUTPUT: Predicted UPDRS Part III at 12 months                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Training Pipeline

1. **Data Preprocessing**
   - Outlier removal (IQR method)
   - Feature scaling (StandardScaler)
   - Target transformation (Yeo-Johnson)

2. **Feature Selection**
   - Top 100 genes by correlation with ΔUPDRS
   - 7 PD risk genes (literature-based)
   - 3 pathway scores (biological knowledge)
   - 3 gene-clinical interactions

3. **Hyperparameter Optimization**
   - Bayesian optimization (Optuna)
   - 30 trials per base model
   - 7-fold cross-validation
   - Objective: Minimize MAE

4. **Model Training**
   - Stratified 80/20 train-test split
   - Ensemble stacking with 3 base models
   - Huber meta-learner for final prediction

5. **Validation**
   - Independent clinical test set (n=78)
   - Cross-validation (7 folds, n=312)
   - Performance metrics: R², MAE, RMSE

---

## 🔍 SHAP Analysis

### Feature Importance Hierarchy

**Top 10 Features (by mean |SHAP| value):**

| Rank | Feature | Mean \|SHAP\| | Type | Interpretation |
|------|---------|--------------|------|----------------|
| 1 | UPDRS_BL × PINK1 | 0.283 | Interaction | Initial severity modulated by PINK1 expression |
| 2 | UPDRS_BL | 0.258 | Clinical | Baseline motor severity |
| 3 | VPS35 | 0.010 | PD Gene | Vesicle trafficking, autophagy dysfunction |
| 4 | Mitochondrial | 0.008 | Pathway | Mitochondrial dysfunction score |
| 5 | GBA | 0.005 | PD Gene | Lysosomal function, α-synuclein clearance |
| 6 | LRRK2 | 0.005 | PD Gene | Kinase activity, autophagy regulation |
| 7 | Neuroinflammation | 0.005 | Pathway | Inflammatory response score |
| 8 | AGE | 0.004 | Clinical | Patient age at baseline |
| 9 | Autophagy | 0.003 | Pathway | Autophagy-lysosome pathway score |
| 10 | GENDER | 0.002 | Clinical | Biological sex |

### Biological Insights

1. **Gene-Clinical Synergy**: The UPDRS_BL × PINK1 interaction (SHAP=0.283) exceeds individual contributions, suggesting that initial disease severity is critically modulated by PINK1-mediated mitochondrial quality control.

2. **Mitochondrial Dysfunction**: The mitochondrial pathway score (SHAP=0.008) is the strongest biological process predictor, consistent with established PD pathophysiology.

3. **VPS35 Dominance**: Among PD risk genes, VPS35 (SHAP=0.010) shows the highest importance, highlighting the role of vesicle trafficking and autophagy in progression.

4. **Pathway Hierarchy**: Mitochondrial dysfunction > Neuroinflammation > Autophagy, reflecting the cascade of pathological events in PD progression.

### SHAP Visualization

The repository includes scripts to generate:
- **Summary plots**: Overall feature importance
- **Dependence plots**: Feature-outcome relationships
- **Waterfall plots**: Individual prediction explanations
- **Force plots**: Contribution breakdown for specific patients

---

## 📖 Citation

If you use this code or model in your research, please cite:

```bibtex
@article{firat2025parkinson,
  title={An explainable ensemble machine learning model using baseline blood transcriptomics to predict Parkinson's Disease motor progression},
  author={Fırat, Yelda and Kılıçaslan, Yılmaz},
  journal={Parkinsonism and Related Disorders},
  year={2025},
  note={Manuscript submitted for publication}
}
```

**Dataset:**
```bibtex
@misc{ppmi2024,
  title={Parkinson's Progression Markers Initiative Database},
  author={PPMI},
  year={2024},
  howpublished={\url{https://www.ppmi-info.org}},
  note={Accessed: 2024}
}
```

---

## 📧 Contact

**Yelda Fırat**  
Department of Computer Engineering  
Mudanya University, Bursa, Turkey  
Email: yelda.firat@mudanya.edu.tr

**Yılmaz Kılıçaslan**  
Department of Computer Engineering  
Mudanya University, Bursa, Turkey  
Email: yilmaz.kilicaslan@mudanya.edu.tr

---

## 🙏 Acknowledgments

This study utilized de-identified data from the Parkinson's Progression Markers Initiative (PPMI) database (www.ppmi-info.org), accessed through the Image and Data Archive (IDA) at LONI (https://ida.loni.usc.edu/). PPMI is sponsored and partially funded by The Michael J. Fox Foundation for Parkinson's Research.

---

## 📝 Notes

- This model is intended for **research purposes only** and should not be used for clinical decision-making without appropriate validation.
- The model requires **baseline blood RNA-seq data** for full feature utilization. For clinical use with only UPDRS, age, and gender, prediction accuracy may be reduced.
- **External validation** in independent cohorts is recommended before clinical deployment.
- The code and model are provided "as-is" without warranty of any kind.

---

**Last Updated:** November 2025  
**Version:** 1.0.0  
**Status:** Research - Manuscript Under Review
