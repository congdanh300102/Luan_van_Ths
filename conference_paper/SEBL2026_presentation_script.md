# SEBL 2026 — Kịch bản thuyết trình

**Bài báo:** Machine Learning for Credit Classification: Case Study in a Vietnamese Commercial Bank
**Người trình bày:** Đàm Công Danh (HUST)
**Hội nghị:** 3rd Global Conference on Sustainability in Economics, Business and Law (UEL SEBL 2026) — TP.HCM, 06/10/2026
**Slot:** 15 phút trình bày + 5 phút hỏi đáp
**Deck:** `SEBL2026_presentation_Dam_Cong_Danh.pptx` — **15 trang** (12 slide trình bày + 3 slide dự phòng B1–B3)

> **Cách dùng file này.** Phần **lời nói** viết bằng tiếng Anh — đây là thứ nói ra miệng, không cần thuộc lòng từng chữ nhưng nên giữ đúng các con số và các câu in đậm. Phần *ghi chú nghiêng* bằng tiếng Việt là chỉ dẫn diễn đạt, không đọc lên. Toàn bộ lời nói này cũng đã nhúng vào Speaker Notes của từng slide, kèm mốc thời gian — bật Presenter View là thấy ngay.
>
> Bản dài 24 trang (17 slide trình bày + 6 backup, cho slot 20–25 phút) được giữ ở `SEBL2026_presentation_Dam_Cong_Danh_FULL_24p.pptx`.

---

## Phân bổ thời gian — tổng 15:00

| # | Slide | Thời lượng | Mốc tích luỹ |
|---|---|---|---|
| 1 | Tiêu đề | 0:25 | 0:25 |
| 2 | Động lực, câu hỏi nghiên cứu & khoảng trống | 1:20 | 1:45 |
| 3 | Dữ liệu & phương pháp | 1:35 | 3:20 |
| 4 | **RQ1 — So sánh mô hình** ⭐ | 1:55 | 5:15 |
| 5 | RQ1 — Đánh đổi Precision–Recall | 1:25 | 6:40 |
| 6 | RQ2 — Hiệu quả đặc trưng | 0:55 | 7:35 |
| 7 | **Kiểm chứng biến chủ đạo (leakage)** ⭐ | 1:55 | 9:30 |
| 8 | RQ3 — Gain vs. SHAP | 1:10 | 10:40 |
| 9 | **Kiểm định thống kê** ⭐ | 1:35 | 12:15 |
| 10 | **Khi thứ hạng bị đảo** ⭐ | 1:10 | 13:25 |
| 11 | Kết luận & hàm ý quản trị | 0:55 | 14:20 |
| 12 | Hạn chế + Cảm ơn | 0:40 | 15:00 |

**Bốn slide ⭐ (4, 7, 9, 10) là xương sống — chiếm 6:35, gần một nửa thời lượng. Nếu bị nhắc giờ, cắt slide 6 và rút ngắn slide 2, 3; tuyệt đối không động vào bốn slide này.**

---

## Slide 1 — Tiêu đề  ⏱ 0:25

> Good morning. My name is Dam Cong Danh, from Hanoi University of Science and Technology, presenting joint work with Le Thi Thanh An of the University of Economics and Law, and Le Kim Thu.
>
> Our paper asks what it actually takes to govern a machine-learning credit-classification model responsibly in a Vietnamese bank — and the answer turned out to be less about which algorithm you pick than we expected.

*Đứng yên, chưa bấm slide. Câu thứ hai là "hook" — nói chậm ở vế cuối.*

---

## Slide 2 — Động lực, câu hỏi nghiên cứu & khoảng trống  ⏱ 1:20

> In Vietnam, debt-group classification is a regulatory obligation, not an internal modelling choice. Every loan goes into one of five groups, Groups 3 to 5 are non-performing, and that classification drives provisioning — which in turn drives how much credit the bank can extend to private borrowers.
>
> So a model that is accurate but opaque, or that has quietly learned the labelling rule rather than genuine risk signal, is unusable in a regulated process. That is the governance problem.
>
> Three questions on the left. Notice how RQ1 is phrased: not "which algorithm wins", but **how far the ranking depends on the imbalance treatment applied.** That phrasing turned out to matter a great deal.
>
> On the right, the three gaps that motivated our design. The 81.84 percent figure is from the study closest in subject to ours — an early-warning system for debt-group migration. It is an Accuracy figure, on data with this kind of imbalance.
>
> And the box at the bottom is the real contribution: *every non-trivial claim is cross-checked against an independent test — so that governance conclusions rest on tested, not asserted, evidence.*

*Slide này gộp 2 slide cũ nên dày. Đừng đọc hết — chỉ nói 3 RQ và lướt qua 3 gap, chỉ tay lên màn hình. Đọc nguyên văn câu trong hộp xanh. KHÔNG nói giọng chê bai về literature: khung đúng là "đây là các kiểm định chúng tôi thấy còn thiếu" — tác giả các bài đó có thể đang ngồi trong phòng.*

---

## Slide 3 — Dữ liệu & phương pháp  ⏱ 1:35

> 27,001 loan contracts, 21 raw variables, one Vietnamese commercial bank. Stratified 80/20 split.
>
> Two things on the left are worth pausing on. NHOMNO correlates 0.98 with the target — it is an alternate debt-group field produced by the same credit-monitoring process. We treat removing it as leakage control, not feature selection, because retaining it would invalidate every comparison downstream. And IV screening flagged five variables above the conventional 1.0 threshold — **we did not auto-exclude them.** We reviewed each against a stated criterion and documented the decision. I will come back to that on slide seven.
>
> On the right, the key word is symmetric: six algorithms, both imbalance treatments, all twelve configurations, on the same hold-out and the same twenty-five cross-validation folds. Hyperparameters were fixed in advance — a real limitation I will own at the end, but it means no evaluation partition was ever consumed by model selection.
>
> Now the red box, and I want to raise it here rather than let someone find it in the results. **Group 5 alone, at 12.45 percent, is larger than Groups 2, 3 and 4 combined.** The median Group-5 loan is roughly six years past its original maturity, and 92 percent are past maturity against 44 percent in Group 2. This is a legacy loss stock, not a flow of newly deteriorating credits.
>
> So when you see an F1 of 0.93 on Group 5 later in this talk, **that is not early warning.** That is a settled end state which is easy to separate. The groups that are operationally demanding are 2 to 4 — and those are exactly where the model is weakest.

*Slide dày nhất của deck. Ưu tiên: leakage control → thiết kế đối xứng → hộp đỏ Group 5. Hộp đỏ là phần xây dựng uy tín, nói chậm.*

---

## Slide 4 — RQ1: So sánh mô hình  ⏱ 1:55 ⭐

> This is the core comparison. Twelve rows: six algorithms, each under both treatments.
>
> Three readings.
>
> First, and this is our headline: **the imbalance treatment matters more than the choice of algorithm.** Every tree-based model does better with class weighting turned OFF. That is counter-intuitive — class weighting is the standard reflex for imbalanced data — and here it costs Macro-F1 in five cases out of five, by between 0.02 and 0.13 points.
>
> Second, Logistic Regression is the sole exception. Weighting helps it, 0.3946 against 0.3628. But it trails every tree configuration either way, so the honest statement is that weighting is dominated *for the tree-based models*, not for all six without qualification.
>
> Third — look at the top of the Macro-F1 column. LightGBM without weighting: 0.6248. Decision Tree without weighting: 0.6245. **Three ten-thousandths apart, on a single split.** I am not going to call that a win, and slide nine shows why.
>
> One last thing. Compare the Accuracy column against the Macro-F1 column for CatBoost: 0.9093 against 0.6074. That divergence is precisely the metric gap from slide two, visible on our own data.

*Dùng con trỏ chỉ vào cột Macro-F1 khi nói ý thứ ba. Nếu cháy giờ, ý thứ hai rút còn một câu.*

---

## Slide 5 — RQ1: Đánh đổi Precision–Recall  ⏱ 1:25

> This is the most directly useful slide for any practitioner in the room.
>
> Two configurations of the *same* algorithm. Macro-F1 0.625 against 0.604 — close. But look at how they get there.
>
> Unweighted: macro Precision 0.748, macro Recall 0.572. When it flags a loan, it is usually right. But it misses most of what it should catch — Group-2 Recall is 0.183. **Roughly four out of five special-mention loans get classified as standard.**
>
> Weighted: the mirror image. Group-2 Recall jumps from 0.183 to 0.471. But Group-2 Precision collapses from 0.603 to 0.245 — so about three out of four flags are now false.
>
> Which is better? Neither, in the abstract. It depends entirely on whether a flag triggers costly manual review — in which case false positives burn scarce analyst time — or whether missing a deteriorating loan is the more expensive error.
>
> **That is a credit-risk committee decision, not a modelling decision.** And a single Macro-F1 number hides it completely.

*Nhấn "Neither" rồi dừng một nhịp. Slide này dễ tạo câu hỏi tốt từ người làm ngân hàng.*

---

## Slide 6 — RQ2: Hiệu quả đặc trưng  ⏱ 0:55

> Does adding more features always help? No.
>
> Thirteen features gives XGBoost 0.6194. Nine features gives 0.6139 — that is 98 and a half percent of it. Three features still recovers 95 to 97 percent.
>
> Two honesty caveats, in the blue box. The curve is not monotone — k equals 7 beats k equals 9 — because each k refits a different model rather than pruning one, so those wiggles are noise. **Please do not read an optimum off this curve; there isn't one.**
>
> And the in-fold number matters: if you select the top nine features using a ranking computed over the whole dataset, you have peeked at the test partition. Recomputing inside each training fold costs under one percent.
>
> Why a bank cares: each feature dropped is one fewer data feed to source, document, monitor for drift, and explain to a supervisor.

*Slide ngắn nhất, dễ cắt nhất nếu cháy giờ. Điểm bán hàng là dòng cuối — chi phí quản trị, không phải độ chính xác.*

---

## Slide 7 — Kiểm chứng biến chủ đạo  ⏱ 1:55 ⭐

> This slide exists for one reason: it is the single most likely question in this room.
>
> DAYS_TO_MATURITY dominates every importance ranking we compute, and its Information Value is 4.554 — more than four times the conventional leakage flag. So the fair question is: **is it just a restatement of the label, the way NHOMNO was?**
>
> Four tests. And each one could have gone the other way.
>
> One. Build the dumbest possible classifier: call a loan non-performing if it is past maturity. If DAYS_TO_MATURITY encoded the labelling rule, that would reconstruct the label almost perfectly. **It gets 67.3 percent accuracy, F1 of 0.508.** That is not a label copy. Compare NHOMNO, which we excluded at r equals 0.98.
>
> Two. Look at the boundaries. If the variable *were* the overdue criterion, essentially no Group-1 loan would be past maturity. In fact 35 percent are — plausibly rolled-over working capital, a common practice in Vietnamese commercial lending. And 8 percent of Group-5 loans are *not* past maturity, so the label clearly reflects restructuring history and qualitative criteria too.
>
> Three. Ablation. Take the variable out entirely: Macro-F1 falls by 16 to 23 percent. That is a genuine, non-redundant strong predictor. If it were the label, the model would collapse, not degrade.
>
> Four. We chose the reference date arbitrarily. Shifting d-zero adds one constant to every record, so tree partitions are unchanged. Macro-F1 is identical to four decimal places across five candidate dates.
>
> Now the red box, and please let me deliver this rather than skip it. **None of this proves there is no leakage.** The raw file has no days-past-due column of any kind, so we cannot exclude a correlated unobserved field from the data alone. We say so in the paper and I say so here. What we claim is narrower: these are four tests that could have failed and did not — which is a materially stronger basis than a plausible narrative on its own.

*Slide dài nhất và quan trọng nhất về phương pháp. Giữ nhịp, đừng vội. Hộp đỏ nói chậm, nhìn thẳng khán giả.*

---

## Slide 8 — RQ3: Gain vs. SHAP  ⏱ 1:10

> Two tables, same four model-and-treatment combinations, two different importance measures. We set LightGBM's importance_type explicitly to "gain", so both libraries report the same quantity.
>
> On the left, everyone agrees that tenure and maturity leads. Below that, disagreement. **Follow the "Product and loan purpose" row: 24.7, then 7.1, then 21.7, then 5.7.** A factor of four — enough to move it from second place to fifth depending only on which library computed it.
>
> On the right, SHAP. Tenure and maturity: 40.8, 41.2, 42.0, 42.8. **The ordering is identical in all four combinations.**
>
> The small print under the tables matters, and I will say it out loud rather than hide it. The two tables sit on slightly different feature sets. So please do not compare a Gain cell against a SHAP cell — the contrast to read is *across the four columns, within each table.*
>
> The implication is concrete: if you are deciding which data to invest in collecting, and you base that on one model's Gain ranking, you may be buying the wrong data.

*Chủ động đọc to đoạn chú thích nhỏ — nếu không, người hiểu chuyện sẽ hỏi ngay và mình mất thế chủ động.*

---

## Slide 9 — Kiểm định thống kê  ⏱ 1:35 ⭐

> Now the proper test. Twelve configurations, twenty-five identical train/test partitions, so every pairwise difference is properly paired.
>
> Why the correction: folds in repeated k-fold cross-validation overlap — they share data. A naive paired t-test treats them as independent, which understates the variance and overstates significance. **Nadeau–Bengio replaces a variance factor of 0.04 with 0.29** — a factor of seven. Then Holm on top, for the family of eleven comparisons.
>
> The result. Highest point estimate: Decision Tree without weighting, 0.6271. **A single depth-ten pruned tree, ahead of three gradient-boosting libraries.**
>
> But read the Holm column. XGBoost-none: 0.789. Random Forest-none: 0.789. LightGBM-none: 0.663. Nowhere near rejection.
>
> So the correct statement is not "Decision Tree wins". It is that these four configurations **cannot be separated on this evidence.** We therefore name no winning algorithm — and choosing among them has to rest on other grounds: training cost, interpretability, or the precision-recall profile from slide five. Stating which ground you used is itself part of the governance record.
>
> What does survive correction: the bottom eight rows. Every class-weighted configuration rejects. The treatment effect is roughly an order of magnitude larger than any algorithm effect — which is exactly what Brown and Mues report: the best algorithm changes with the degree of imbalance.

*Việc từ chối chọn người thắng là KẾT QUẢ, không phải né tránh — nói dứt khoát.*

---

## Slide 10 — Khi thứ hạng bị đảo  ⏱ 1:10 ⭐

> I want to show you that same discipline applied to ourselves, because this is the part of the paper I am most confident about.
>
> On the hold-out split, LightGBM-none was first. That is the natural headline: "LightGBM is best for Vietnamese debt-group classification." It would have been a perfectly publishable sentence.
>
> Under repeated cross-validation it drops to fourth, behind XGBoost. And the corrected paired test on that specific pair gives **p equals 0.3844** — not close to rejecting in either direction. The apparent ranking was an artefact of one random split.
>
> The third bullet is an erratum, and I am stating it deliberately. Every row of that test is measured against the same reference configuration — so you cannot read a comparison between two *non-reference* rows off the table. An earlier version of our paper did exactly that. We went back and tested it directly: the direction holds, but after Holm adjustment it is p equals 0.063 — not significant. We corrected the claim rather than leaving it standing.
>
> The takeaway generalises well beyond this dataset. **A ranking that flips between two honest evaluation designs is a signal to report both, not to pick the flattering one.**

*Slide gây ấn tượng mạnh nhất với hội đồng khoa học. Nói tự tin, không có vẻ xin lỗi — chủ động công bố erratum là dấu hiệu nghiên cứu nghiêm túc.*

---

## Slide 11 — Kết luận & hàm ý quản trị  ⏱ 0:55

> Three findings on the left, four governance implications on the right.
>
> RQ1: the imbalance treatment is better identified than the algorithm. And let me say this plainly: **we deliberately name no winning algorithm, and that refusal is a result, not a failure to reach one.**
>
> RQ2: nine of thirteen features recover 98 to 99 percent of peak, surviving honest in-fold selection. RQ3: SHAP is stable, Gain is not — triangulate before you spend money on data.
>
> On the right, point four is the one I would most want you to take away. An automated screen flagged five of our variables. The wrong response is to auto-drop them. But the wrong response is *also* to write a paragraph explaining why they are fine. **The right response is a check that could fail.**
>
> And the box at the bottom: these are conditional recommendations. One institution, one snapshot, random resampling. What transfers is the ordering of the checks, not the coefficients.

---

## Slide 12 — Hạn chế + Cảm ơn  ⏱ 0:40

> Four limitations, stated plainly. I would much rather raise them than be asked.
>
> 121 Group-3 loans in the test set, so the minority-group numbers are noisy. One bank, one snapshot, no recorded classification date — so all our resampling is random, not temporal. And I will go further than the paper does: given the Group-5 legacy stock, **I expect out-of-time numbers to be worse than these**, not better. Label encoding imposes an arbitrary ordering on branch codes. And no hyperparameter tuning — a genuine trade-off, since tuning would have consumed an evaluation partition.
>
> Thank you. I have three backup slides — the leakage diagnostics and ablation, both confusion matrices, and the exact hyperparameters — if any of those would be useful.

*Câu "I expect out-of-time numbers to be worse" là câu mạnh nhất về liêm chính học thuật. Đừng bỏ. Slide này gộp cả lời cảm ơn nên không có slide riêng — dừng lại ở đây cho phần Q&A.*

---

# Chuẩn bị Q&A

Xếp theo xác suất giảm dần. **Cách bật slide dự phòng khi đang trình chiếu: gõ số trang + Enter.** B1 = trang **13**, B2 = trang **14**, B3 = trang **15**.

### Q1. "Nhóm nợ được phân loại chủ yếu theo SỐ NGÀY QUÁ HẠN. Vậy sao biến ngày quá hạn trong bài lại không quyết định được nhãn? Nghiên cứu này có thực tiễn không?" → **Slide 7**

*Đây là câu hỏi khó nhất và nhiều khả năng nhất từ người làm ngân hàng. Người hỏi đúng — đừng phòng thủ, hãy công nhận ngay rồi tách bạch hai khái niệm.*

**Trả lời cốt lõi (~60 giây):**

> You are right about the rule, and that distinction is central to this paper — let me take it in two parts.
>
> First, a definitional point. **The variable in our model is not days past due.** It is DAYS_TO_MATURITY, computed from NGAYDENHAN — the loan's *original contractual maturity date* — not from a payment due date. Those are different quantities. An amortising borrower can be ninety days past due on an instalment while the contract still has three years to run: days past due is 90, days to maturity is positive. Conversely, a bullet loan past its maturity date but quietly rolled over has days to maturity negative and days past due zero.
>
> And critically: **the raw twenty-one-column file contains no days-past-due field of any kind.** We could not have used it even if we had wanted to.
>
> Second, that is exactly what the numbers on this slide measure. Thirty-five percent of Group-1 loans are already past original maturity yet remain standard. Eight percent of Group-5 loans are not past maturity. If the two concepts were the same, those figures would be zero and one hundred. The 67.3 percent is precisely a measurement of how good a proxy contractual maturity is for the real criterion — informative, but far from deterministic.
>
> Now, is it practical? I would separate two different tasks.
>
> **If the bank has the days-past-due field — and of course it does, in its core banking system — then classification is a deterministic rule lookup, not a prediction problem.** Nobody needs a machine-learning model to decide which bucket a ninety-five-day-overdue loan belongs in. We are not proposing to replace the regulatory rule, and the paper does not claim to.
>
> The useful problem is the one where days past due has not moved yet: early warning. And there you *must not* use days past due as a feature, because it is the label. The feature set we use — maturity, tenor, balance, utilization, product, purpose, interest rate, branch — is exactly what you would have available for that task.
>
> So read our Macro-F1 of 0.62 as the finding itself: **that is how much of the five-group structure is recoverable from contract-level attributes alone.** The gap between 0.62 and 1.0 is the information content of days past due, plus restructuring history and the CIC cross-institution pull-up.

**Nếu bị hỏi tiếp, hoặc muốn chốt mạnh — bằng chứng nội tại (~30 giây):**

> And here is the evidence that this reading is right rather than a convenient excuse. Compare our per-class performance against how well contractual maturity proxies the overdue state in each group:
>
> Group 2 — ten to ninety days overdue — is 44 percent past contractual maturity, and our F1 there is 0.28. Group 3 is 60 percent, F1 0.42. Group 4 is 68 percent, F1 0.54. Group 5 — over 360 days overdue — is 92 percent past maturity, and our F1 is 0.93.
>
> **Perfectly monotone. The model is good exactly where contractual maturity is a good proxy for days past due, and bad exactly where it is not.** That is not a coincidence — it is the signature of a model working from a proxy rather than from the rule itself. And it is a far more honest account of what this model can and cannot do than any single headline number.

| Nhóm | Tiêu chí quá hạn (TT 11/2021) | % quá hạn đáo hạn HĐ | F1 của mô hình |
|---|---|---|---|
| 2 | 10 – 90 ngày | 44.2% | 0.281 |
| 3 | 91 – 180 ngày | 60.3% | 0.415 |
| 4 | 181 – 360 ngày | 68.1% | 0.543 |
| 5 | > 360 ngày | 92.1% | 0.931 |

*Spearman ρ = 1.00 trên nhóm 2–5. Hai con số cần nhớ nếu không kịp nhìn bảng: **nhóm 2 → 44% / F1 0.28**, **nhóm 5 → 92% / F1 0.93**.*

**Ba điều TUYỆT ĐỐI không được nói:**
- ❌ "Mô hình có thể thay thế quy trình phân loại nợ" — sai về bản chất, và người hỏi sẽ bắt bài ngay
- ❌ "Ngày quá hạn không quan trọng" — ngược lại, nó quan trọng đến mức chính là nhãn
- ❌ Lảng tránh bằng cách nói về độ chính xác — người hỏi đang hỏi về *tính hợp lệ của bài toán*, không phải về điểm số

### Q2. "DAYS_TO_MATURITY có IV = 4.554. Đó chắc chắn là leakage — sao vẫn giữ?" → **B1 (trang 13)**

> This is the right question, and slide seven is our answer. Four independent tests. The univariate rule reconstructs the label at only 67.3 percent. 35 percent of standard loans are already past maturity without being downgraded. Ablation degrades the model by 16 to 23 percent rather than collapsing it. And the result is invariant to the reference date.
>
> But I want to be precise about what we claim. We do not claim the variable is proven free of leakage. The raw file contains no days-past-due field at all, so a correlated unobserved field cannot be excluded from the data alone. What we claim is that four tests that could have failed did not — and that this is a stronger basis than a written justification. Before operational use, a bank would have to confirm the variable against its own classification methodology.

### Q3. "Tại sao không tune siêu tham số? Kết quả có thể khác hẳn." → **B3 (trang 15)**

> A fair challenge, and it is our fourth stated limitation. The reason is that we wanted no configuration to receive a search budget the others did not — and tuning would have consumed an evaluation partition, so the hold-out and the folds would no longer be clean.
>
> The consequence is that we compare algorithm families at sensible defaults, not at their optima. I accept that a tuning budget could plausibly reorder the four configurations that the corrected test already cannot separate — that would not change our conclusion, since we already decline to rank them. It is much less likely to overturn the treatment effect, which is roughly an order of magnitude larger and consistent in sign across all five tree-based models.

### Q4. "Macro-F1 chỉ 0.62 — mô hình như vậy có dùng được không?" → **B2 (trang 14)**

> Not as an autonomous classifier, no — and we do not propose it as one. Two things put that number in context.
>
> First, Macro-F1 weights all five groups equally, including groups with 121 test observations. Weighted-F1 is 0.895 and macro ROC-AUC 0.925. We report Macro-F1 as primary precisely because it refuses to let the 77.65 percent majority flatter the result.
>
> Second, and more honestly, the number reflects genuine difficulty on the transitional groups, 2 to 4. That is the useful finding for a bank: the model is a triage aid on those groups, not a decision-maker, and the per-class table tells you exactly how far to trust each flag.

### Q5. "Sao không dùng SMOTE hay các kỹ thuật resampling khác?" → **B3 (trang 15)**

> Deliberate choice. Five of the eleven features are categorical, including 31-category product sub-type, 79-category loan purpose, and nominal branch identifiers. SMOTE interpolates between neighbours in feature space — and a synthetic point halfway between product code 12 and product code 27 has no contractual meaning. It would manufacture loans that cannot exist.
>
> SMOTE-NC would be the principled alternative and is a reasonable extension. What we did instead was treat the imbalance question as a comparison — weighting versus no weighting, applied symmetrically to all six algorithms — which is what let us identify the treatment effect at all.

### Q6. "Decision Tree thắng cả XGBoost? Nghe không hợp lý." → **B3 (trang 15)**

> It has the highest point estimate — 0.6271 — but I would resist the word "wins". After the Nadeau–Bengio correction and Holm adjustment, it cannot be separated from XGBoost-none, Random Forest-none or LightGBM-none. Those p-values are 0.79, 0.79 and 0.66.
>
> Two things make it less surprising than it sounds. It is a pruned tree — max depth 10, minimum 10 samples per leaf — not an unconstrained one. And with eleven features and strong signal concentrated in one or two of them, the variance reduction that ensembles buy you has less to work with. This is also exactly what Brown and Mues report.

### Q7. "Dữ liệu thuộc Thông tư nào? 31/2024 hay 11/2021?"

> The file records no classification date, so we cannot establish this from the data alone, and we say so in Section 2.3. The indirect evidence — OPEN_DATE concentrated in 2016 to 2019, nothing after November 2020 — points to Circular 11/2021 as the applicable framework for this snapshot. Circular 31/2024, as amended by 37/2025, is cited only as the framework currently in force.
>
> What we do claim, narrowly, is that the target operationalises Vietnam's five-group regulatory structure, which has been stable across every version since Decision 493 in 2005. The exact instrument and the bank's internal assignment rules remain unverified.

### Q8. "Chỉ một ngân hàng, một thời điểm — kết quả có tổng quát được không?"

> No, and we are explicit about that. Every governance claim in the paper is stated conditionally. What we argue transfers is the *ordering of the checks*, not the coefficients — report per-class metrics, cross-check importance measures, test rather than assert admissibility, and use a corrected paired test.
>
> I would add one specific caution. Slide three showed a legacy loss stock in Group 5. An out-of-time evaluation would very likely produce worse numbers than these, not better. So I would not want anyone to take 0.62 as a benchmark for Vietnamese debt-group classification.

### Q9. "SHAP ổn định hơn, hay chỉ vì bảng SHAP bỏ SEX và LOAIKH?" → **Slide 8**

> A sharp question, and it is why we put that footnote on the slide. The two tables are on different feature sets by construction: Gain on the 13-variable pool, SHAP on the 11-feature main configuration.
>
> But the stability claim is not a comparison *between* the tables. It is read *within* each table, across the four model-and-treatment columns. Within Gain, product and purpose ranges from 5.7 to 24.7 percent. Within SHAP, tenure and maturity stays between 40.8 and 42.8 percent across all four. Removing two near-zero-IV variables — SEX at 0.008, LOAIKH at 0.000 — cannot explain a four-fold swing in a different feature group.

### Q10. "Label encoding cho biến danh mục — chẳng phải đó là sai sót?" → **B3 (trang 15)**

> It is a limitation, and it is stated as one. Label encoding imposes an arbitrary ordering on nominal codes such as branch identifiers, which tree models can partially work around through repeated splits but cannot fully undo.
>
> We verified native categorical handling for CatBoost only. Worth noting that CatBoost, with native handling, does not outperform the label-encoded models here — it sits fifth on repeated CV. So on this dataset the encoding does not appear to be what separates the configurations. A one-hot or target-encoded comparison would be a reasonable extension.

### Q11. "Đóng góp thực tiễn cho ngân hàng là gì?"

> Three concrete things, and none of them is the model itself.
>
> One: a reporting standard. Per-class precision and recall alongside Macro-F1, because slide five showed that two configurations with similar Macro-F1 behave in opposite ways — and choosing between them is a credit-risk committee decision that must be recorded as one.
>
> Two: a cost reduction. Nine features instead of thirteen, for under one percent of performance. Each dropped feature is one fewer feed to source, document, monitor and explain to a supervisor.
>
> Three, and most portable: a verification protocol. When an automated screen flags a variable, the response should be a documented check that could fail — not a written justification. That is the practice we think should be a governance baseline before any model of this kind informs credit decisions.

### Câu hỏi không trả lời được

Nếu bị hỏi điều không có trong bài (ví dụ: so sánh với ngân hàng khác, dữ liệu out-of-time, chi phí triển khai thực tế):

> That is outside what this dataset can tell us, and I would rather say so than speculate. What I can say is [nêu điều gần nhất mình có bằng chứng]. It is a good direction for the out-of-time work.

---

# Checklist trước buổi trình bày

- [ ] Mở deck ở **Presenter View** để thấy speaker notes (Slide Show → Use Presenter View). Mỗi ghi chú đều bắt đầu bằng mốc thời gian, ví dụ `[1:55]`
- [ ] Kiểm tra font **Calibri** có trên máy hội nghị; nếu không, nhúng font (File → Options → Save → Embed fonts in the file)
- [ ] Mang bản **PDF dự phòng** trong USB — phòng khi máy chiếu không đọc được .pptx
- [ ] Tập riêng slide **4, 7, 9, 10** — bốn slide này chiếm 6:35 trong 15 phút và toàn bộ sức nặng lập luận
- [ ] Nhớ 6 con số cốt lõi, để không phải nhìn slide:
      **0.6248** (LightGBM hold-out) · **0.6271** (Decision Tree repeated-CV) · **0.3844** (p, LGBM vs XGB) ·
      **67.3%** (kiểm định univariate DTM) · **16–23%** (chi phí ablation) · **77.65%** (tỷ trọng Group 1)
- [ ] Nhớ số trang backup: **B1 = 13** (leakage & ablation), **B2 = 14** (confusion matrices), **B3 = 15** (hyperparameters). Gõ số + Enter khi đang trình chiếu
- [ ] Chạy thử một lần có bấm giờ. Nếu quá 15:00, cắt slide 6 trước — rồi rút slide 2 và 3
