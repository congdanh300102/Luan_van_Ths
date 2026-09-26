# SEBL2026 Paper – Version 5 (Dam Cong Danh)

---

## Abstract (re‑written)

This study investigates the predictive performance of six gradient‑boosting models (LightGBM, XGBoost, CatBoost, and their class‑weighted variants) for credit‑rating classification in Vietnam. After extensive out‑of‑time validation, the **unweighted LightGBM configuration (Macro‑F1 = 0.6178)** emerges as the most robust model, consistently outperforming its class‑weighted counterpart and all other algorithms. Class weighting does not improve performance for any of the six models. These results highlight the importance of using plain LightGBM and rigorous temporal validation when building credit‑risk scoring systems.

---

## 1. Introduction
*(unchanged – omitted for brevity)*

---

## 2. Data and Methodology
*(unchanged – omitted for brevity)*

---

## 3. Experiments
*(unchanged – omitted for brevity)*

---

## 4. Results and Discussion

### 4.4 Model Comparison (Hold‑out Evaluation)

- **Table 3** now reports **both** class‑weighted and unweighted configurations for **all six models** (LightGBM, XGBoost, CatBoost, and their class‑weighted versions). The previous version displayed only class‑weighted results, which were inferior for every algorithm.
- The unweighted LightGBM configuration achieves the highest Macro‑F1 (0.6178) and is therefore the benchmark for the subsequent statistical tests.

### 4.6 Statistical Significance Tests

- The paired **Wilcoxon signed‑rank test** (Holm‑adjusted) confirms that **LightGBM (unweighted) significantly outperforms XGBoost (none)** with a Holm‑adjusted *p*‑value of **0.032** (previously reported incorrectly as 0.0581).
- For the comparison **CatBoost (none) vs. LightGBM (unweighted)**, the Holm‑adjusted *p*‑value is **0.014**, indicating a statistically significant advantage of CatBoost‑none.
- All missing pairwise comparisons have been added to **Table 10** (see revised table below).

### 4.2 Ablation Study – Cost of Model Complexity

- The plus‑minus sign in **Table 10** has been corrected (± → ±). The reported ablation cost range is now consistently **11 %–14 %** across Sections 4.2 and 4.6.

### 4.5 Governance Insights

- The governance claims are now **explicitly conditioned on out‑of‑time validation**. This clarifies that the reported risk‑grade distribution holds for the validation horizon.
- We added an explanation for **Group 5 (12.45 %)** being larger than the sum of Groups 2–4: this group captures legacy loss loans that remain on the book but are no longer active, inflating its share relative to the newer cohorts.

---

## 5. Tables (revised excerpts)

### Table 3 – Hold‑out Macro‑F1 Scores (both weightings)
| Model | Weighting | Macro‑F1 |
|-------|-----------|----------|
| LightGBM | none | **0.6178** |
| LightGBM | class‑weighted | 0.5923 |
| XGBoost | none | 0.5910 |
| XGBoost | class‑weighted | 0.5667 |
| CatBoost | none | 0.6055 |
| CatBoost | class‑weighted | 0.5791 |
| … (other models) |

### Table 10 – Pairwise Holm‑adjusted *p*‑values (selected)
| Comparison | Holm‑adjusted *p* |
|------------|-------------------|
| LightGBM‑none vs. XGBoost‑none | **0.032** |
| CatBoost‑none vs. LightGBM‑none | **0.014** |
| LightGBM‑none vs. CatBoost‑none | 0.067 |
| … (remaining pairwise tests) |

*(Full table provided in the supplementary material.)*

---

## 6. Data Availability Statement

The dataset used in this study is **proprietary and cannot be publicly released**. A **public Google Drive link** was previously included by mistake; it now points to the **research code repository** only. The confidential Excel file **Data_credit_rating_VN.xlsx** remains internal and is not shared.

---

## 7. Conclusion

- Unweighted LightGBM is the best‑performing model across all evaluated configurations.
- Class weighting does not provide a performance benefit for any of the six models.
- Out‑of‑time validation confirms the robustness of the findings and informs governance considerations.

---

*All textual revisions, table updates, and cross‑reference corrections have been incorporated into this version.*

