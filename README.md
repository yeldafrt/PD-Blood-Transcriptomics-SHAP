# Parkinson's Disease Progression Prediction Using Blood RNA-seq and Formal Language Theory

This repository contains the complete reproducible code and data for predicting 12-month motor progression in Parkinson's disease using machine learning, SHAP analysis, and formal language theory applied to PINK1 and PARK7 genes.

---

## Overview

Parkinson's disease (PD) is a progressive neurodegenerative disorder with highly variable progression rates among patients. Predicting disease progression is crucial for personalized treatment planning and clinical trial design. This study integrates:

- **Machine Learning:** Stacking Regressor ensemble model combining XGBoost, LightGBM, CatBoost, and HuberRegressor
- **SHAP Analysis:** Model interpretability to identify key prognostic features
- **Formal Language Theory:** Context-free grammar (CFG) and probabilistic context-free grammar (PCFG) analysis of PINK1 and PARK7 regulatory structures

### Key Findings

- **Model Performance:** R² = 0.469, MAE = 6.55 points (12-month UPDRS Part III prediction)
- **Top Prognostic Features:** Baseline UPDRS (SHAP = 0.598), PINK1 (SHAP = 0.0204, rank #6), PARK7 (SHAP = 0.0160, rank #8)
- **Mitochondrial Pathway:** Highest SHAP value among all biological pathways (SHAP = 0.0115, rank #11)
- **PINK1 Regulatory Complexity:** 125 palindromes, 20 restriction sites, GC-rich promoter (58.52%)
- **PARK7 Regulatory Features:** 45 palindromes, 4 restriction sites, moderate GC content (46.05%)

---

## Repository Structure

```
Reproducibility_Parkinson_Progression_RNA_seq/
│
├── dataset_preparation_in_3_steps/
│   ├── 01_process_clinical_data.py
│   ├── 02_process_rnaseq_data.py
│   ├── 03_merge_and_feature_engineering.py
│   ├── feature_list.txt
│   └── final_dataset.csv
│
├── model-r-squared-0.469-no-overfitting/
│   ├── figures_and_codes/
│   │   ├── Figure_01_Package/          # Model Architecture
│   │   ├── Figure_02_Package/          # Predicted vs Actual
│   │   ├── Figure_04_Package/          # Distribution Comparison
│   │   ├── Figure_05_Package/          # Performance Across Sets
│   │   ├── Figure_06_Package/          # Fast vs Slow Analysis
│   │   ├── Figure_07_Package/          # SHAP Analysis
│   │   └── model_architecture_package.zip
│   └── parkinson_UPDRS_V04_REGULARIZED_FINAL/
│       └── 13_updrs_v04_regularized.py
│
├── PINK1+PARK7_ANALYSIS/
│   ├── ADIM1_MRNA_TEMİNİ/              # Step 1: mRNA Acquisition
│   ├── ADIM2_PROMOTER_MOTİF_ANALİZİ/   # Step 2: Promoter Motif Analysis (Figure 8)
│   ├── ADIM3_Palindrome Finder/         # Step 3: Palindrome Analysis (Figure 9)
│   ├── ADIM4_RNA Secondary Structure/   # Step 4: RNA Structure (Figure 10)
│   ├── patient_41767_pink1_analysis/    # Patient 41767 Case Study (Figure 12)
│   ├── Clinical_PINK1_System_Complete_KLİNİKTE UYGULANACAK _PAKET/  # Clinical Dashboard (Figure 11)
│   ├── final_report/                    # Formal Language Analysis Report
│   └── FİNAL_RAPOR/                     # Final Turkish Report
│
└── pseudo_codes.zip
```

---

## Dataset

### Source
**Parkinson's Progression Markers Initiative (PPMI)**
- Cohort size: n = 392 patients
- Data modalities: Clinical assessments, blood RNA-seq
- Timepoints: Baseline and 12-month follow-up

### Clinical Data
- **Primary Outcome:** UPDRS Part III motor score (12-month change)
- **Baseline Features:** Age, gender, disease duration, baseline UPDRS, MoCA score

### RNA-seq Data
- **Platform:** Blood-based transcriptomics
- **Processing:** Log2(TPM+1) transformation, filtering (>1 TPM in >50% samples)
- **Features:** 
  - 7 Parkinson's risk genes (SNCA, LRRK2, GBA, PRKN, PINK1, PARK7, VPS35)
  - Top 50 genes correlated with motor progression
  - 3 biological pathway scores (mitochondrial, synaptic, inflammatory)

---

## Methodology

### 1. Data Preparation (3 Steps)

#### Step 1: Clinical Data Processing
```bash
python 01_process_clinical_data.py
```
- Loads PPMI clinical data
- Calculates ΔUPDRS (12-month change)
- Filters for complete baseline and 12-month assessments
- Outputs: `clinical_data_processed.csv`

#### Step 2: RNA-seq Data Processing
```bash
python 02_process_rnaseq_data.py
```
- Processes quant.sf files from Salmon
- Log2 transformation: log2(TPM + 1)
- Filters low-expression genes
- Outputs: `rnaseq_baseline_filtered.csv`

#### Step 3: Merge and Feature Engineering
```bash
python 03_merge_and_feature_engineering.py
```
- Merges clinical and RNA-seq data
- Adds Parkinson's risk genes
- Calculates pathway scores
- Selects top 50 correlated genes
- Outputs: `final_dataset.csv`, `feature_list.txt`

### 2. Model Training

#### Stacking Regressor Architecture
```python
Base Models:
  - XGBoost (n_estimators=100, max_depth=3, learning_rate=0.01)
  - LightGBM (n_estimators=100, max_depth=3, learning_rate=0.01)
  - CatBoost (iterations=100, depth=3, learning_rate=0.01)

Meta-Model:
  - HuberRegressor (epsilon=1.35, alpha=1.0)
```

#### Training Script
```bash
python 13_updrs_v04_regularized.py
```

#### Model Evaluation
- **Train-Test Split:** 80% train, 20% test (stratified by fast/slow progressors)
- **Cross-Validation:** 5-fold CV
- **Metrics:** R², MAE, RMSE
- **Overfitting Check:** Train R² = 0.650, Clinical R² = 0.469 (gap = 0.181, acceptable)

### 3. SHAP Analysis

```bash
python plot_05_shap_three_panel.py
```

SHAP (SHapley Additive exPlanations) provides model interpretability:
- **Global Feature Importance:** Identifies top prognostic features
- **Individual Predictions:** Explains why a specific patient has high/low predicted progression
- **Feature Interactions:** Reveals how features combine to influence predictions

### 4. Formal Language Theory Analysis

Inspired by Searls (1992) "The Linguistics of DNA", we applied computational linguistics to PINK1 and PARK7 genes:

#### Step 1: mRNA Acquisition
```bash
python 01_download_sequences.py
```
- Downloads PINK1 (NM_032409.3) and PARK7 (NM_007262.5) from NCBI

#### Step 2: Promoter Motif Analysis (Regular Expression)
```bash
python 02_promoter_analysis.py
```
- Searches for: TATA box, CAAT box, GC box, Inr element, CpG islands
- Uses regex patterns: `TATA[AT]A[AT]`, `[GC]CAAT`, `GGGCGG`

#### Step 3: Palindrome Finder (Context-Free Grammar)
```bash
python 03_palindrome_finder.py
```
- CFG rule: `S → aSu | uSa | gSc | cSg | ε`
- Detects palindromes (4-12 bp) and restriction enzyme sites

#### Step 4: RNA Secondary Structure (PCFG)
```bash
python 04_rna_structure.py
```
- Identifies stem-loop structures in 5' UTR
- Calculates free energy (ΔG) for each stem-loop

---

## Results

### Model Performance

| Metric | Train Set | Clinical Set |
|:-------|:----------|:-------------|
| **R²** | 0.650 | 0.469 |
| **MAE** | 5.32 | 6.55 |
| **RMSE** | 7.18 | 8.92 |

### Cohort Statistics

| Group | N | Percentage | Baseline UPDRS | 12-month UPDRS | ΔUPDRS | Rate (points/month) |
|:------|:--|:-----------|:---------------|:---------------|:-------|:--------------------|
| **Total** | 392 | 100% | - | - | - | - |
| **Fast Progressors** | 143 | 36.5% | 16.55 ± 7.74 | 27.75 ± 10.23 | +11.20 ± 6.13 | +0.933 |
| **Slow Progressors** | 249 | 63.5% | 18.78 ± 10.62 | 15.54 ± 9.82 | -3.32 ± 5.53 | -0.270 |

### Top 20 Features (SHAP Values)

| Rank | Feature | SHAP Value | Category |
|:-----|:--------|:-----------|:---------|
| 1 | Baseline UPDRS | 0.598 | Clinical |
| 2-5 | Top correlated genes | 0.025-0.035 | RNA-seq |
| **6** | **PINK1** | **0.0204** | **PD Risk Gene** |
| **8** | **PARK7** | **0.0160** | **PD Risk Gene** |
| **11** | **Mitochondrial Pathway** | **0.0115** | **Pathway Score** |

### PINK1 vs PARK7 Comparison

| Feature | PINK1 | PARK7 |
|:--------|:------|:------|
| **mRNA Length** | 2,657 bp | 1,127 bp |
| **GC Content** | 58.52% | 46.05% |
| **Promoter Type** | TATA-less (GC-rich) | TATA-containing |
| **GC Box** | 4 | 0 |
| **CAAT Box** | 5 | 3 |
| **Palindromes** | 125 | 45 |
| **Restriction Sites** | 20 | 4 |
| **Stem-Loops** | 135 | 78 |
| **5' UTR Length** | 265 bp | 180 bp |
| **5' UTR GC%** | 60.00% | 52.22% |
| **Most Stable Stem-Loop** | ΔG = 1.50 kcal/mol | ΔG = 0.80 kcal/mol |

---

## Figures

All figures are generated with publication-quality (300 DPI). The manuscript includes 12 figures:

### Model Performance and Validation Figures

**Figure 1: Model Architecture**
- Location: `figures_and_codes/model_architecture_package.zip`
- Script: `create_model_architecture.py`
- Output: `model_architecture_diagram.png`
- Description: Two-level stacking ensemble architecture with base models (XGBoost, LightGBM, CatBoost) and meta-model (HuberRegressor)

**Figure 2: Predicted vs Actual UPDRS (Clinical Validation Set)**
- Location: `figures_and_codes/Figure_01_Package/`
- Script: `plot_01_predicted_vs_actual_clinical.py`
- Output: `Figure_01_Predicted_vs_Actual_Clinical.png`
- Description: Scatter plot showing model predictions vs actual 12-month UPDRS scores (n=78, R²=0.469, MAE=6.55)

**Figure 3: Residual Plot (Clinical Validation Set)**
- Location: `figures_and_codes/Figure_02_Package/`
- Script: `plot_02_residual_plot_clinical.py`
- Output: `Figure_02_Residual_Plot_Clinical.png`
- Description: Residual analysis showing prediction errors distributed around zero with no systematic bias

**Figure 4: Distribution Comparison (Actual vs Predicted)**
- Location: `figures_and_codes/Figure_04_Package/`
- Script: `plot_04_distribution_comparison_clinical.py`
- Output: `Figure_04_Distribution_Comparison_Clinical.png`
- Description: Violin plot comparing distributions of actual (mean=18.6, SD=10.8) and predicted (mean=18.8, SD=8.5) UPDRS scores

**Figure 5: Performance Across Training, Test, and Clinical Sets**
- Location: `figures_and_codes/Figure_06_Package/`
- Script: `plot_06_three_panel_all_sets.py`
- Output: `Figure_06_Three_Panel_All_Sets.png`
- Description: Three-panel comparison showing model performance on training (R²=0.650), test (R²=0.512), and clinical validation (R²=0.469) sets

**Figure 6: Fast vs Slow Progressor Analysis**
- Location: `figures_and_codes/Figure_07_Package/`
- Script: `plot_07_fast_vs_slow_progressors.py`
- Output: `Figure_07_Fast_vs_Slow_Progressors.png`
- Statistics: `fast_vs_slow_statistics.txt`
- Description: Four-panel analysis comparing fast (n=143) and slow (n=249) progressors including distribution, correlation, trajectory, and group comparison

**Figure 7: SHAP Feature Importance Analysis**
- Location: `figures_and_codes/Figure_05_Package/`
- Script: `plot_05_shap_three_panel.py`
- Output: `Figure_05_SHAP_Three_Panel.png`
- Description: Three-panel SHAP analysis showing (a) top 20 features, (b) PD risk genes, and (c) pathway scores with baseline UPDRS as dominant feature

### Formal Language Analysis Figures

**Figure 8: Promoter Motif Analysis (PINK1 vs PARK7)**
- Location: `PINK1+PARK7_ANALYSIS/ADIM2_PROMOTER_MOTİF_ANALİZİ/`
- Script: `02_promoter_analysis.py`
- Outputs: `PINK1_motif_map.png`, `PARK7_motif_map.png`
- Description: Regular expression-based promoter motif mapping showing PINK1 (34 motifs: 5 CAAT, 4 GC, 2 CCAAT, 23 Inr) vs PARK7 (21 motifs: 3 CAAT, 1 TATA, 17 Inr)

**Figure 9: Palindrome Analysis (Context-Free Grammar)**
- Location: `PINK1+PARK7_ANALYSIS/ADIM3_Palindrome Finder/palindrom_2 graph_merged/`
- File: `Figure_9_Palindrome_2Panel_Enhanced.png`
- Script: `03_palindrome_finder.py`
- Description: CFG-based palindrome detection showing PINK1 (125 palindromes, 20 restriction sites) vs PARK7 (45 palindromes, 4 restriction sites)

**Figure 10: RNA Secondary Structure Analysis (5' and 3' UTR)**
- Location: `PINK1+PARK7_ANALYSIS/ADIM4_RNA Secondary Structure/2GRAPHS_MERGED/`
- File: `Figure_10_RNA_Structure_2Panel_Enhanced.png`
- Script: `04_rna_structure.py`
- Description: PCFG-based RNA secondary structure prediction showing stem-loop structures in 5' and 3' UTR regions with minimum free energy (MFE) conformations

### Clinical Application Figures

**Figure 11: Clinical Dashboard (PINK1-Based Risk Stratification)**
- Location: `PINK1+PARK7_ANALYSIS/Clinical_PINK1_System_Complete_KLİNİKTE UYGULANACAK _PAKET/PINK1_Clinical_Analysis/`
- File: `Clinical_Dashboard.png`
- Script: `clinical_pink1_system.py`
- Description: Six-panel dashboard showing (a) risk distribution, (b) PINK1 expression distribution, (c) fast vs slow comparison, (d) PINK1 vs progression scatter, (e) risk level distribution, (f) progression rate distribution for PPMI cohort (n=392)

**Figure 12: Patient 41767 Individual Analysis**
- Location: `PINK1+PARK7_ANALYSIS/patient_41767_pink1_analysis/`
- File: `Patient_41767_PINK1_Analysis.png`
- Script: `analyze_patient_pink1.py`
- Report: `Patient_41767_Report.txt`
- Description: Three-panel case study showing (a) patient's PINK1 expression in population context, (b) comparison with fast/slow progressors, (c) PINK1 vs progression relationship, plus patient profile with clinical status, PINK1 status, formal language findings, and recommendations

---

## Clinical Application: Patient 41767 Case Study

### Patient Profile
- **Age:** 85 years
- **Baseline UPDRS:** 15.0
- **12-month UPDRS:** 22.0
- **ΔUPDRS:** +7.0 points (progression rate: +0.58 points/month)
- **Category:** Fast Progressor

### PINK1 Expression Analysis
- **Expression:** 4.66 log2(TPM+1)
- **Population Mean:** 6.58
- **Z-score:** -5.32 (0th percentile)
- **Status:** VERY LOW (>5 SD below mean)

### Formal Language Analysis Findings
- **Promoter:** GC-rich with 4 GC boxes
- **Palindromes:** 125 count
- **Stem-loops:** 135 count
- **5' UTR:** Stable structure (ΔG=1.50 kcal/mol)

### Clinical Recommendations
1. Genetic testing (PINK1 sequencing)
2. Promoter/5' UTR mutation screening
3. Mitochondrial function tests
4. Treatment: Mitochondrial protectors, antioxidants

### Analysis Script
```bash
cd PINK1+PARK7_ANALYSIS/patient_41767_pink1_analysis/
python analyze_patient_pink1.py
```

---

## Requirements

### Python Environment
```
Python 3.11+
```

### Core Libraries
```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
shap>=0.42.0
matplotlib>=3.7.0
seaborn>=0.12.0
biopython>=1.81
```

### Installation
```bash
pip install -r requirements.txt
```

---

## Usage

### Quick Start

1. **Prepare Dataset**
```bash
cd dataset_preparation_in_3_steps/
python 01_process_clinical_data.py
python 02_process_rnaseq_data.py
python 03_merge_and_feature_engineering.py
```

2. **Train Model**
```bash
cd ../model-r-squared-0.469-no-overfitting/parkinson_UPDRS_V04_REGULARIZED_FINAL/
python 13_updrs_v04_regularized.py
```

3. **Generate Model Performance Figures**
```bash
cd ../figures_and_codes/Figure_05_Package/
python plot_05_shap_three_panel.py

cd ../Figure_06_Package/
python plot_06_three_panel_all_sets.py

cd ../Figure_07_Package/
python plot_07_fast_vs_slow_progressors.py
```

4. **Run Formal Language Analysis**
```bash
cd ../../../PINK1+PARK7_ANALYSIS/final_report/Formal_Language_Analysis_Complete/
python 01_download_sequences.py
python 02_promoter_analysis.py
python 03_palindrome_finder.py
python 04_rna_structure.py
```

5. **Clinical Decision Support System**
```bash
cd ../../Clinical_PINK1_System_Complete_KLİNİKTE\ UYGULANACAK\ _PAKET/
python clinical_pink1_system.py
```

6. **Patient 41767 Analysis**
```bash
cd ../patient_41767_pink1_analysis/
python analyze_patient_pink1.py
```

---

## Citation

If you use this code or data, please cite:

```
[Manuscript in preparation]
Integrating Machine Learning and Formal Language Theory to Predict 
Parkinson's Disease Progression: A Blood RNA-seq Study
```

---

## Contact

**Yelda Fırat**  
Email: yelda.firat@mudanya.edu.tr

For questions, issues, or collaboration inquiries, please contact via email.

---

## Acknowledgments

- **PPMI Database:** Data used in this study were obtained from the Parkinson's Progression Markers Initiative (PPMI) database (www.ppmi-info.org/data). PPMI is sponsored and partially funded by The Michael J. Fox Foundation for Parkinson's Research.

- **Formal Language Theory:** Inspired by Searls, D. B. (1992). "The Linguistics of DNA." *American Scientist*, 80(6), 579-591.

---

## Data Availability

- **PPMI Data:** Available upon request from PPMI (www.ppmi-info.org)
- **Processed Data:** `final_dataset.csv` included in this repository
- **Model Weights:** Available in `model-r-squared-0.469-no-overfitting/` folder
- **High-Risk Patients:** `High_Risk_Patients.csv` in Clinical PINK1 System package
- **Formal Language Analysis Reports:** English and Turkish versions available in `final_report/` folder

---

## Reproducibility

All analyses are fully reproducible. The repository includes:
- Complete source code for all 12 figures
- Processed dataset
- Trained model weights
- Figure generation scripts
- Pseudo-code documentation
- Clinical decision support system
- Patient 41767 case study

**Estimated Runtime:**
- Data preparation: ~10 minutes
- Model training: ~30 minutes (CPU), ~5 minutes (GPU)
- SHAP analysis: ~15 minutes
- Formal language analysis: ~5 minutes
- Clinical dashboard generation: ~2 minutes
- Patient analysis: ~1 minute

**Hardware Requirements:**
- CPU: 4+ cores recommended
- RAM: 16 GB minimum
- Storage: 2 GB for data and models

---

## Version History

- **v1.0.0** (November 2025): Initial release
  - Stacking Regressor model (R² = 0.469)
  - 12 publication-quality figures
  - SHAP analysis
  - Formal language theory analysis of PINK1 and PARK7
  - Patient 41767 case study
  - Clinical decision support system
  - Complete reproducibility package

---

**Last Updated:** November 4, 2025
