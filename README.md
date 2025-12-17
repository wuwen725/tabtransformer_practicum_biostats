# TabTransformer-Based Multi-Dataset Pipeline for Breast Cancer Risk Stratification

This repository contains the full workflow for my biostatistics practicum project,
which evaluates the performance and robustness of **TabTransformer** models for
breast cancer risk stratification using three GEO RNA-seq datasets:

- **GSE164641**
- **GSE95640**
- **GSE240671**

Across datasets, gene expression features are combined with available clinical
variables, followed by feature processing, differential expression analysis,
model training/testing with cross-validation, and comparison against baseline
models such as logistic regression and random forests.

The entire project is designed to be **fully transparent, modular, and reproducible**.

---

## Project Objectives

1. **Evaluate TabTransformer performance** on multiple heterogeneous RNA-seq datasets.  
2. **Assess robustness** of the model across different populations and clinical variables.  
3. **Integrate gene expression + clinical covariates** into a unified classification pipeline.  
4. **Perform DESeq2-based feature selection**, including volcano plots and PCA visualization.  
5. **Compare TabTransformer with classical ML models**:  
   - Logistic Regression  
   - Random Forest  
   - XGBoost / others (depending on dataset)  
6. **Generate reproducible modeling workflows** for each dataset.

---

## Project Structure
'''
tabtransformer_practicum_biostats/
├── data/
│ ├── GSE164641/
│ │ ├── raw/ # optional raw GEO files
│ │ └── processed/ # master_data.csv, metadata.csv, DESeq outputs
│ ├── GSE95640/
│ └── GSE240671/
│
├── notebooks/
│ ├── GSE164641/
│ │ ├── 01_preprocessing_and_eda_GSE164641.ipynb
│ │ └── 02_model_training_and_evaluation_GSE164641.ipynb
│ ├── GSE95640/
│ └── GSE240671/
│
├── output/
│ ├── GSE164641/
│ │ ├── figures/ # PCA, volcano plots, ROC curves, etc.
│ │ └── tables/ # AUC summary, metric tables, DEG tables
│ ├── GSE95640/
│ └── GSE240671/
│
└── src/
├── data_utils.py # load & preprocess datasets
├── feature_utils.py # DESeq2 filtering, clinical feature handling
├── model_utils.py # TabTransformer model definition
├── train_utils.py # training loops for CV
├── evaluation_utils.py # metrics, ROC, AUC, CIs
└── plot_utils.py # visualization utilities
'''
---

## Datasets Used

All datasets come from the NCBI Gene Expression Omnibus (GEO):

| Dataset | Samples | Platform | Label Definition |
|--------|---------|-----------|------------------|
| **GSE164641** | 187 | RNA-seq | High-risk vs Average-risk |
| **GSE95640** | 382 | RNA-seq | dataset-specific labels |
| **GSE240671** | 69 | RNA-seq | dataset-specific labels |

Clinical variables differ across datasets (age, BMI, gender, pregnancy history, family cancer history, etc.).  
These differences allow for evaluating **model generalizability** under feature heterogeneity.

---

## Methodology

### 1. **Data Preprocessing**
- Gene filtering based on variance and DESeq2  
- Removal of low-count genes  
- Log2 transformation  
- Scaling and normalization  
- Clinical feature harmonization  
- Train/test splitting or k-fold cross-validation group assignment  

### 2. **Differential Expression Analysis**
We apply **DESeq2 (via PyDESeq2)** for:

- Volcano plot generation  
- PCA visualization of sample separation  
- Log2 fold-change filtering  
- padj thresholding (Benjamini–Hochberg FDR)

### 3. **Modeling**
The primary model is:

### 🔹 **TabTransformer (PyTorch implementation)**  
- 50 training epochs  
- Embedding dimension: *dataset-specific*  
- Multi-head attention encoder  
- Categorical + continuous hybrid feature integration  

### 🔹 Baseline Models  
Used for comparison across datasets:

- Logistic Regression  
- Random Forest  
- XGBoost (optional)  

### 4. **Evaluation Metrics**
Each dataset is evaluated under **stratified k-fold CV**, using:

- AUC (ROC)  
- Macro-F1  
- Weighted-F1  
- Sensitivity / Specificity  
- Confidence intervals for AUC (bootstrapping)

All result tables and plots appear in `output/`.

---

## Key Outputs

Each dataset includes:

### Figures
- PCA plot  
- Volcano plot  
- ROC curves (per fold + mean curve)  
- Feature importance charts (if applicable)

### Tables
- Mean AUC summary  
- Fold-by-fold performance  
- DEG tables  
- Clinical variable summaries  

These files are automatically saved in:

output/DATASET_NAME/figures/
output/DATASET_NAME/tables/


---

## Reproducibility

### Install dependencies
```bash
pip install -r requirements.txt

(or using conda)

Run preprocessing notebook

notebooks/GSE164641/01_preprocessing_and_eda_GSE164641.ipynb

Run model training notebook

notebooks/GSE164641/02_model_training_and_evaluation_GSE164641.ipynb

To extend to another dataset, simply change the dataset folder name.

Project Status

✔ GSE164641: preprocessing, EDA, modeling, full results

✔ GSE95640: uploaded & processed

✔ GSE240671: uploaded & processed

✔ All notebooks added

✔ src module completed

✔ outputs uploaded

References

Huang et al., TabTransformer: Tabular Data Modeling Using Contextual Embeddings.

Love et al., DESeq2: Moderated estimation of fold change and dispersion for RNA-seq data.

Pedregosa et al., Scikit-learn: Machine Learning in Python.