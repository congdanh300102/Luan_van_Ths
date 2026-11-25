# Response to Reviewers — ICEIR 2026, Paper 282

**Reproducing or Predicting? Regulation-Grounded Evaluation of Debt-Group Classification and Early Warning in a Vietnamese Commercial Bank**

We thank the three reviewers for their careful reading and constructive comments. All comments have been addressed. Two of them (information parity of the regulatory baseline and customer-level leakage) led to new experiments that changed some numbers and sharpened conclusion (iii); we report these changes in full below. Section and table numbers refer to the revised manuscript. All new analyses use the same two real monthly snapshots (30 April and 31 May 2026); no new data were added.

## Summary of main changes

1. **Customer-grouped evaluation is now the main protocol for Task 2** (Reviewer 2, comment 4; Reviewer 3). Contracts of the same customer always share the debt group in our data, and 119 of the 265 migrations occur in customers with two or more migrating contracts. Contract-level cross-validation therefore overstated Task-2 performance substantially; all Task-2 results are now reported with customer-grouped, stratified 5-fold CV repeated 10 times, with contract-level figures kept only to show the inflation.
2. **Information parity for the regulatory baseline** (Reviewer 2, comment 1; Reviewer 3). The current debt group $G_T$ is now given to the models. The main Task-2 model is XGBoost with the 20 fields, the overdue fields and $G_T$; conclusion (iii) has been revised accordingly.
3. **IV claims narrowed to decile binning** (Reviewer 2, comment 2; Reviewer 3), with the definition of "bad" stated and the IV under coarse classing with a separate zero bin reported.
4. **Robustness checks repeated on the conservative configuration and bootstrap intervals added** (Reviewer 2, comment 3; Reviewer 3).
5. Minor corrections: attribution of the Macro-F1 inflation to both interest fields, the 18-field count, and the position of Appendix A.

---

## Reviewer 1

> *The evaluation is subject to notable limitations as the dataset spans only a single bank and one 31-day observation window with 265 migration events, of which only 45 are non-mechanical, leaving long-term generalizability to 3, 6, or 12-month horizons unverified.*

We agree. The limitation is stated in Section 7, which now also notes that the customer-grouped evaluation reduces the effective sample to 181 migrating customers. We could not add a second real window for this revision: only two real monthly extracts are available to us. Validation on later monthly windows and on 3-, 6- and 12-month horizons is the first item of future work, and we recommend in Section 6 that a model fitted on one window be validated on the next before use.

> *The study relies on unconfirmed operational assumptions regarding the non-retroactive admissibility of interest and overdue fields at prediction time, while failing to incorporate reason codes for contract exits.*

We agree that both are limitations, and both remain stated in Section 7. Two analyses bound their effect:

- **Admissibility.** Removing off-balance-sheet interest, or both interest fields, from the main Task-2 model changes AP from 0.292 to 0.295 and 0.293 (corrected $p \ge$ 0.71). The early-warning conclusions therefore do not depend on whether the interest fields are admissible. The overdue fields are the regulation's own classification inputs at $T$; they could be revised retroactively only if the bank restated past delinquency, which remains to be confirmed with the bank.
- **Exit reason codes.** The bank could not provide reason codes. As a sensitivity analysis, counting the 45 exits from groups 2–5 as migration events lowers the AP of the main model from 0.292 to 0.234; the conclusions do not change.

---

## Reviewer 2

> **1.** *The distance-to-threshold rule uses both the current group G_T and DPD, while "XGB, 20 features" has neither, and "XGB + overdue fields" has DPD but not G_T. … Please report XGB-20 + G_T and XGB + overdue fields + G_T (or use the rule score as a feature), and revise conclusion (3) if the results change.*

Thank you; this was an important omission. We added $G_T$ and, separately, the rule score as features (new rows in Table 5). The results (mean AP over the 50 folds) are:

| Model | Contract-level CV (as submitted) | Customer-grouped CV (main) |
|---|---|---|
| Rule: distance to next threshold | 0.203 | 0.206 |
| XGB, 20 fields | 0.179 | 0.026 |
| XGB, 20 fields + $G_T$ | 0.207 | 0.044 |
| XGB + overdue fields | 0.393 | 0.203 |
| XGB + overdue fields + $G_T$ (new main model) | 0.483 | 0.292 |
| XGB + overdue fields + $G_T$ + rule score | 0.576 | 0.417 |

With the same information as the rule (20 fields + $G_T$), XGBoost is still statistically indistinguishable from the rule under contract-level CV (AP 0.207 vs 0.203, corrected $p = 0.91$). Under customer-grouped CV, however, models without DPD fall far below the rule (AP 0.044 with $G_T$ and 0.026 without it, against 0.206; corrected $p<0.001$), so contract-level evaluation understated the strength of the rule.

Giving $G_T$ to the model with overdue fields raises AP from 0.393 to 0.483 (contract-level, $p<0.001$). Under the customer-grouped protocol, XGBoost with overdue fields but without $G_T$ only ties the rule (0.203 vs 0.206, $p =$ 0.91). With $G_T$ it beats the rule (0.292, $p =$ 0.004). Conclusion (iii) has been revised to: *most 31-day migrations follow the regulation's DPD thresholds; a model given the same information as the rule (the current group and DPD) outperforms it, mainly at the top of the review queue, whereas a model given less information does not.* Using the rule score itself as a feature gives a further gain (0.417). We report this gain as evidence that the regulation supplies a useful feature, not as the main model, because the score is a hand-built transformation of the fields the model already sees.

> **2.** *The near-zero IV comes from decile binning. With a separate zero bin … off-balance interest would get an IV of about 5 … The result therefore shows that decile-binned IV fails, not IV screening in general. Please reword the abstract, G1, C1 and Sec. 6 accordingly, and state how "bad" was defined for IV.*

The reviewer's estimate is correct. With "bad" defined as groups 2–5 and "good" as group 1 (11,742 vs 88,875 April loans, now stated in Section 4.2), the IV of off-balance-sheet interest is 0.000 under decile binning and 5.09 with a separate bin for zero. We have reworded the Abstract, Related Work, G1, C1, Section 4.2 and Section 6. They now state that decile-binned IV misses the field, and that a well-binned IV flags it. The point we keep is narrower: the well-binned screen also flags loan age (IV 1.12), which reflects seasoning, and rates on-balance-sheet interest at 0.85. IV measures association with the label, not provenance, so it cannot separate a field that accounting rules record *from* the debt group from a strong legitimate predictor. The regulation can.

> **3.** *Task 2 is evaluated within a single April-to-May window, and the Task-1 robustness checks were run only on the leaky full configuration. … bootstrap intervals for Table 4 and repeating the robustness checks on the conservative configuration would strengthen the conclusions.*

**Second window.** Only two real monthly extracts are available, so a second real window could not be added; see our reply to Reviewer 1.

**Bootstrap intervals.** Table 4 now reports 95% intervals from 1,000 bootstrap resamples of customers (cluster bootstrap, so that contracts of one customer are resampled together). For the main conservative configuration (LightGBM), Macro-F1 is 0.429 [0.410, 0.447] on the internal test split and 0.404 [0.389, 0.418] on unseen May contracts. Under the leaky full configuration the corresponding values are 0.693 [0.675, 0.713] and 0.635 [0.618, 0.652]. The intervals do not overlap between the two configurations, nor between seen and unseen contracts.

**Robustness on the conservative configuration.** All Task-1 checks were repeated on the conservative configuration (LightGBM, internal test split; base Macro-F1 0.429):

| Check | Macro-F1 |
|---|---|
| No resampling | 0.429 |
| Class weights | 0.377 |
| SMOTE + class weights | 0.388 |
| SMOTE-NC (LightGBM / XGBoost) | 0.442 / 0.425 |
| Branch code as categorical (LightGBM / XGBoost) | 0.430 / 0.420 |
| Top-8 features by Gain | 0.433 |
| Gain vs SHAP ranking of feature groups | Spearman ρ = 0.83, p = 0.042 |

Moderate SMOTE and no resampling perform equally well; class weighting is worse. The conclusions are unchanged. Without the interest fields, group 2 is almost never recognised (recall 0.03), and 3.0% of predictions are off by two or more groups, against 0.4% in the full configuration. Section 5.1 now reports these figures.

> **4.** *All debts of one customer at a bank are placed in the same (highest-risk) group, yet both tasks split at contract level … Please report contracts per customer and results with customer-grouped splits.*

We thank the reviewer for this comment, which had the largest effect on the paper. Section 4.1 now reports the customer structure. The April snapshot has 73,494 customers (1.37 contracts per customer, maximum 313). 16.9% of customers hold more than one contract, and these customers account for 39.3% of contracts. No multi-contract customer has contracts in different groups. Under the contract-level split, 35.7% of internal-test contracts belong to a customer with a contract in the training split.

- **Task 1.** With a customer-grouped 80/20 split, results change little. In the conservative configuration, LightGBM reaches Macro-F1 0.419 internally (0.429 with the contract split) and 0.413 on May contracts of customers absent from training (0.404 for unseen contracts in the contract split). The full configuration gives 0.665 and 0.620. These results are added to Table 4, and the leakage and overlap conclusions are unchanged.
- **Task 2.** The effect is large. The 265 migrations occur in 181 customers, and 119 of them belong to customers with two or more migrating contracts, so contract-level CV lets the model recognise a customer's other contracts. Under customer-grouped CV, the AP of XGBoost on the 20 fields falls from 0.179 to 0.026, and that of XGBoost with overdue fields from 0.393 to 0.203. The untrained rule is unaffected (0.206). Customer-grouped, stratified 5-fold CV repeated 10 times is therefore now the main Task-2 protocol: Table 5, Figure 1, the learner comparison, the ablations, the exit sensitivity, the review-budget analysis and SHAP all use it. Contract-level figures are retained in one column of Table 5 to show the inflation. We also added customer-level leakage to Section 6 as a further check that OOT and CV evaluations of debt-group models should include.

### Minor comments

> *The abstract and C1 attribute the 0.43 → 0.69 increase to off-balance interest alone.*

Corrected in the Abstract, C1 and Section 5.1. Off-balance-sheet interest accounts for most of the increase (Macro-F1 0.479 without it, i.e. 0.21 of 0.26). On-balance-sheet interest, which is recorded under the same article of Decree 135/2025/ND-CP, accounts for the rest.

> *Table 2 says Task 1 uses 20 fields, but the main (conservative) configuration uses 18.*

Corrected in Table 2, Section 4.3 and the caption of Table 4: the main configuration uses 18 fields and the full configuration 20. The pipeline diagram (former Figure 1), which also stated 20 fields, was removed to keep the revised paper within the page limit.

> *Appendix A (Table 6) appears in the middle of the reference list.*

Fixed: the table is now anchored in Appendix A.

---

## Reviewer 3

> *The regulatory baseline comparison is not fully controlled. The distance-to-threshold rule uses both current debt group and DPD, while the XGBoost configurations lack equivalent inputs.*

Addressed by new experiments; see our reply to Reviewer 2, comment 1. The models are now compared with the rule under identical information (20 fields + $G_T$; 20 fields + overdue fields + $G_T$). Conclusion (iii) has been revised.

> *The Information Value claim is also too broad … This supports a limitation of the implemented screening procedure rather than IV screening generally.*

We agree, and the claim has been narrowed throughout; see our reply to Reviewer 2, comment 2. The definition of "bad" (groups 2–5) is now stated, and the IV under a separate zero bin (5.09) is reported.

> *The abstract also attributes the full Macro-F1 increase to off-balance interest alone, although the reported ablation indicates contributions from both interest fields.*

Corrected; see the minor comments of Reviewer 2.

> *Contract-level splitting leaves possible customer overlap between training and test data unresolved.*

Resolved; see our reply to Reviewer 2, comment 4. All Task-2 results now use customer-grouped CV, and Task-1 results are reported for both splits.

> *Validation is further limited to one bank and one 31-day window, with only 45 non-mechanical migration events. Operational field admissibility and reasons for contract exits remain unconfirmed.*

We agree, and these remain stated as limitations in Section 7. Their effect is bounded by the ablation and exit-sensitivity analyses described in our reply to Reviewer 1.

> *Robustness checks cover only the leaky configuration.*

Addressed; see our reply to Reviewer 2, comment 3. All checks were repeated on the conservative configuration.
