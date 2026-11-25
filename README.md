# Superhero Intelligence System
> Can six ability scores tell a hero from a villain?

An end-to-end Machine Learning project investigating supervised classification and unsupervised clustering on superhero intelligence metrics using the Hugging Face Superheroes dataset.

---

## 📌 Project Overview
* **Objective:** Evaluate whether 6 physical ability scores (`intelligence_score`, `strength_score`, `speed_score`, `durability_score`, `power_score`, `combat_score`) can predict moral alignment (`Good / Hero` vs. `Bad / Villain`) and uncover natural character archetypes.
* **Core Tasks:**
  1. **Data Preparation (Section 2):** Inspection, deduplication, missing value marking (zeros/blanks to `NaN`), weak row filtering, and pipeline-based median imputation. Creation of Clean Set A (Clustering) and Clean Set B (Supervised).
  2. **Task 1 — Classification (Section 3):** Baseline Dummy, Decision Tree, and K-Nearest Neighbors (KNN) optimized via 5-fold Stratified Cross-Validation on Macro-F1.
  3. **Task 2 — Clustering (Section 4):** K-Means and Agglomerative Hierarchical Clustering (Ward linkage) evaluated via Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score, and Adjusted Rand Index (ARI).
  4. **Task 3 — Feed-Forward Neural Networks (Section 5):** Shallow (Network A, 65 params) and Deep (Network B, 257 params) Multi-Layer Perceptrons trained with Adam, binary cross-entropy, and early stopping.
  5. **Final Evaluation (Section 6):** Comparative model ranking across all 5 models on the unseen final test set (`test.csv`).

---

## 🏆 Final Model Ranking on `test.csv`

| Rank | Model | Test Accuracy | Test Macro-F1 | Villain Recall | Beats Baseline? |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **KNN** | **0.5956** | **0.5009** | **0.2045** | **yes** |
| **2** | **Network A** | **0.6178** | **0.4549** | 0.0909 | **yes** |
| **3** | **Network B** | 0.6000 | 0.4214 | 0.0568 | **yes** |
| **4** | **Decision Tree** | 0.5600 | 0.4080 | 0.0682 | **yes** |
| **5** | **Baseline** | 0.6089 | 0.3785 | 0.0000 | — |

---

## 📂 Repository Structure
```text
├── superhero_project.ipynb   # Complete, executed Jupyter Notebook with all outputs & G1-G16 plots
├── REPORT.md                 # Detailed academic report with Tables T1-T11 and Q1-Q12 answers
├── README.md                 # Project documentation and summary
├── data/
│   ├── train.csv             # Training dataset
│   └── test.csv              # Unseen test dataset
└── scripts/
    ├── build_clean_notebook.py
    └── execute_nb.py
```

---

## 🚀 Getting Started

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt # (pandas, scikit-learn, scipy, matplotlib, seaborn, tensorflow)
```

### 2. Run Notebook
Launch Jupyter Lab or Jupyter Notebook:
```bash
jupyter notebook superhero_project.ipynb
```
