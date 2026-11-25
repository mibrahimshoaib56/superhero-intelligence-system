# Superhero Intelligence System: Project Report
**Can Six Ability Scores Tell a Hero from a Villain?**

---

## 1. Executive Summary
This project investigates whether six standardized physical ability attributes (**intelligence, strength, speed, durability, power, combat**) can predict moral alignment (**Good / Hero vs. Bad / Villain**) and reveal distinct unsupervised character archetypes.

Across supervised models evaluated on the unseen final test set, **K-Nearest Neighbours (KNN)** ranked #1 with a test Macro-F1 of **0.5009**, outperforming the Dummy Baseline (**0.3785**), Decision Tree (**0.4080**), Network A (**0.4549**), and Network B (**0.4214**). Unsupervised clustering showed that characters group primarily by **power scale** (Grounded vs. Cosmic Titans) rather than moral alignment, with heroes and villains co-existing in nearly identical proportions across power tiers (~55% Good to ~31% Bad).

---

## 2. Section 2: Data Preparation (Tables T1 & T2)

### Table T1: Count of Zeros and Blanks Transformed to NaN
In superhero databases, a zero score indicates missing or unrecorded metrics rather than the total absence of physical capability. Blanks and zeros were therefore mapped to `NaN`.

| Feature | Train Blanks | Train Zeros | Train Total $\rightarrow$ NaN | Test Blanks | Test Zeros | Test Total $\rightarrow$ NaN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `intelligence_score` | 0 | 85 | **85** | 0 | 20 | **20** |
| `strength_score` | 0 | 91 | **91** | 0 | 22 | **22** |
| `speed_score` | 0 | 86 | **86** | 0 | 20 | **20** |
| `durability_score` | 0 | 87 | **87** | 0 | 20 | **20** |
| `power_score` | 0 | 57 | **57** | 0 | 14 | **14** |
| `combat_score` | 0 | 85 | **85** | 0 | 20 | **20** |

### Table T2: Character Row Counts After Each Cleaning Step
Characters with $\ge 3$ missing scores were pruned as weak rows. Remaining missing values ($< 3$ missing) were imputed using the training median inside scikit-learn pipelines.

| Step | `train.csv` | `test.csv` |
| :--- | :---: | :---: |
| **Loaded** | $n = 1158$ | $n = 290$ |
| **After step 2 (duplicates)** | $n = 1158$ | $n = 290$ |
| **After step 4 (weak rows)** | $n = 1072$ | $n = 270$ |
| **Clean Set A (Clustering)** | $n = 1072$ | *not used* |
| **Clean Set B (Classification)** | $n = 916$ | $n = 225$ |

*Note on Step 4 to Clean Set B filtering:*
- In `train.csv`, **156 rows** were removed (93 Neutral and 63 missing/blank alignments).
- In `test.csv`, **45 rows** were removed (31 Neutral and 14 missing/blank alignments).

---

## 3. Section 3: Task 1 Supervised Classification

### Table T3: Class Distribution in Clean Set B
| Split | Hero (0) | Hero % | Villain (1) | Villain % | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `train.csv` | 588 | 64.19% | 328 | 35.81% | 916 |
| `test.csv` | 137 | 60.89% | 88 | 39.11% | 225 |

### Table T4: Optimal Hyperparameters from 5-Fold Stratified Cross-Validation
| Model | Best Hyperparameters | Best CV Macro-F1 |
| :--- | :--- | :---: |
| **Decision Tree** | `max_depth=5, min_samples_leaf=10` | **0.5405** |
| **KNN** | `n_neighbors=9, weights='uniform'` | **0.5145** |

### Table T5: Task 1 Classification Performance Summary
| Model | Train Macro-F1 | Test Accuracy | Precision Hero | Recall Hero | F1 Hero | Precision Villain | Recall Villain | F1 Villain | Test Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 0.3910 | 0.6089 | 0.6089 | 1.0000 | 0.7569 | 0.0000 | 0.0000 | 0.0000 | **0.3785** |
| **Decision Tree** | 0.5628 | 0.5600 | 0.5941 | 0.8759 | 0.7080 | 0.2609 | 0.0682 | 0.1081 | **0.4080** |
| **KNN** | 0.6011 | 0.5956 | 0.6237 | 0.8467 | 0.7183 | 0.4615 | 0.2045 | 0.2835 | **0.5009** |

### Task 1 Questions & Answers:
* **Q1: Which model wins by the rules? Quote the numbers.**
  - **KNN** is the winner. The primary ranking metric is test **Macro-F1**. KNN achieves **0.5009**, outperforming the Decision Tree (**0.4080**) and the Baseline (**0.3785**).
* **Q2: Did both models beat the Baseline? By how much?**
  - **Yes**, both beat the baseline:
    - Decision Tree improves upon Baseline by **+0.0295** (+2.95%).
    - KNN improves upon Baseline by **+0.1224** (+12.24%).
* **Q3: Which class is harder to predict? Use the confusion matrices.**
  - The **Villain (1)** class is substantially harder to predict. The Decision Tree caught only **6 of 88 Villains** (Recall = 6.82%), and KNN caught only **18 of 88 Villains** (Recall = 20.45%), whereas Hero recall exceeded 84% for both models.
* **Q4: Does either model overfit? Use the train–test gap.**
  - Under the criteria ($\text{Train Macro-F1} - \text{Test Macro-F1} > 0.10$):
    - **Decision Tree:** $0.5628 - 0.4080 = \mathbf{0.1548} > 0.10 \implies$ **Overfits significantly**.
    - **KNN:** $0.6011 - 0.5009 = \mathbf{0.1002} \approx 0.10 \implies$ Borderline, demonstrating much stronger generalization.

---

## 4. Section 4: Task 2 Unsupervised Clustering

Optimal cluster count selected by maximum Silhouette score: **$k = 2$** (Silhouette = **0.3070**).

### Table T6: Clustering Evaluation Metrics and ARI
| Algorithm | $k$ | Silhouette $\uparrow$ | Davies–Bouldin $\downarrow$ | Calinski–Harabasz $\uparrow$ | Wins (of 3) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Means** | 2 | **0.3070** | **1.2740** | **586.82** | **3 (Winner)** |
| **Agglomerative** | 2 | 0.2779 | 1.2898 | 460.91 | 0 |
| **ARI between the two** | 2 | **0.4834** | — | — | — |

### Table T7: K-Means Cluster Profiles (Original 0–100 Scale)
Clusters are renumbered by average ability score (Lowest average = Cluster 1).

| Cluster | Size | INT | STR | SPD | DUR | POW | COM | Assigned Archetype Name |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 682 | 53.6 | 32.8 | 34.6 | 49.3 | 52.8 | 54.0 | **Moderate / Grounded Characters** |
| **2** | 390 | 79.5 | 78.4 | 69.8 | 83.2 | 87.2 | 77.8 | **Cosmic / High-Power Titans** |

### Table T8: Alignment Breakdown by Cluster for K-Means (%)
| Cluster | Size | Good (%) | Bad (%) | Neutral (%) | Blank (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 682 | 54.5% | 30.5% | 9.0% | 6.0% |
| **2** | 390 | 56.5% | 30.9% | 7.9% | 4.7% |

### Task 2 Questions & Answers:
* **Q5: Which k did you choose? What does the Elbow say?**
  - **$k = 2$** was selected because it maximizes the Silhouette Score (**0.3070**). The inertia elbow curve (G5) bends visibly between $k=2$ and $k=4$ before leveling off into a gradual descent.
* **Q6: Did the two algorithms find similar groups? Use ARI and profiles.**
  - **Yes.** The Adjusted Rand Index is **0.4834**, indicating moderate-to-strong structural agreement (band 0.40–0.69). Both algorithms isolate grounded heroes/villains (~50 average score) from cosmic powerhouse entities (~80+ average score).
* **Q7: Which grouping is better separated, and which is easier to interpret?**
  - **Better Separated:** **K-Means** wins all 3 separation metrics (higher Silhouette, lower Davies-Bouldin, higher Calinski-Harabasz).
  - **Easier to Interpret:** Both offer equivalent clear interpretations representing power tiers.
* **Q8: Describe each cluster in plain words. Is any cluster mostly villains?**
  - **No cluster is mostly villains.** Both Cluster 1 (Moderate) and Cluster 2 (Titans) have almost identical moral compositions (~55% Good vs. ~31% Bad). Moral alignment does not dictate power levels.

---

## 5. Section 5: Task 3 Neural Networks

* **Network A:** Input (6) $\rightarrow$ Dense(8, ReLU) $\rightarrow$ Dense(1, Sigmoid) — **65 parameters**.
* **Network B:** Input (6) $\rightarrow$ Dense(16, ReLU) $\rightarrow$ Dense(8, ReLU) $\rightarrow$ Dense(1, Sigmoid) — **257 parameters**.

### Table T9: Network Architectures & Early Stopping Summary
| Network | Trainable Parameters | Optimizer | Batch Size | Best Epoch | Min Val Loss |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Network A** | 65 | Adam (lr=0.001) | 32 | **151** | **0.6208** |
| **Network B** | 257 | Adam (lr=0.001) | 32 | **57** | **0.6332** |

### Table T10: Neural Network Performance Across Splits
| Network | Split | Accuracy | Macro-F1 | Log Loss |
| :--- | :--- | :---: | :---: | :---: |
| **Network A** | train (80%) | 0.6530 | 0.4944 | — |
| **Network A** | validation (20%) | 0.6359 | 0.4600 | 0.6208 |
| **Network A** | test (`test.csv`) | **0.6178** | **0.4549** | 0.6816 |
| **Network B** | train (80%) | 0.6612 | 0.4852 | — |
| **Network B** | validation (20%) | 0.6250 | 0.4223 | 0.6332 |
| **Network B** | test (`test.csv`) | **0.6000** | **0.4214** | 0.6747 |

*Per-class metrics on `test.csv`:*
- **Network A:** Hero (P=0.6209, R=0.9562, F1=0.7529) | Villain (P=0.5714, R=0.0909, F1=0.1569)
- **Network B:** Hero (P=0.6103, R=0.9489, F1=0.7429) | Villain (P=0.4167, R=0.0568, F1=0.1000)

### Task 3 Questions & Answers:
* **Q9: Did Network B beat A by at least 0.01 Macro-F1? Why or why not?**
  - **No.** Network B did not beat Network A. Network A achieved **0.4549** while Network B achieved **0.4214** (Network B was **0.0335 lower**). With only 6 tabular features and 732 training samples, the additional capacity of Network B led to slight overparameterization rather than enhanced pattern extraction.
* **Q10: Do the loss curves show over- or underfitting? Which epoch was best?**
  - **Network A:** Best epoch is **151**. Demonstrates good fit with training and validation losses staying synchronized.
  - **Network B:** Best epoch is **57**. Demonstrates onset of overfitting after epoch 57, where validation loss turns upward while training loss continues falling.

---

## 6. Section 6: Final Ranking & Global Synthesis

### Table T11: Final Performance Ranking of All 5 Models on `test.csv`
| Rank | Model | Test Accuracy | Test Macro-F1 | Villain Recall | Beats Baseline? |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **KNN** | 0.5956 | **0.5009** | **0.2045** | **yes** |
| **2** | **Network A** | 0.6178 | **0.4549** | 0.0909 | **yes** |
| **3** | **Network B** | 0.6000 | **0.4214** | 0.0568 | **yes** |
| **4** | **Decision Tree** | 0.5600 | **0.4080** | 0.0682 | **yes** |
| **5** | **Baseline** | 0.6089 | **0.3785** | 0.0000 | — |

### Final Questions & Answers:
* **Q11: Did either network beat the Decision Tree and KNN? Give numbers.**
  - **Against Decision Tree (0.4080):** **Yes.** Network A (+0.0469) and Network B (+0.0134) both beat the Decision Tree.
  - **Against KNN (0.5009):** **No.** KNN outperformed Network A by **+0.0460** and Network B by **+0.0795**. KNN benefited from training on 100% of `train.csv` (via 5-fold CV) and natural distance-based local voting in continuous physical space.
* **Q12: Name two limitations of this dataset or experiment.**
  1. **Moral Alignment Independence from Physical Stats:** Capability scores (*intelligence, strength, speed, durability, power, combat*) reflect power scale rather than ethical stance. Characters like Superman and General Zod share virtually indistinguishable attribute values.
  2. **Class Imbalance & Restriction on Rebalancing:** The ~64% to ~36% imbalance favoring Heroes caused all models to bias heavily toward predicting the majority class. Prohibiting class weights or resampling techniques (like SMOTE) capped Villain recall across all models.
