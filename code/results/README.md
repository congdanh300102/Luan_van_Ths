# Giải thích kết quả trong `code/results/`

Thư mục này chứa các file được **sinh tự động** từ những lần chạy huấn luyện /
phân tích trong app Streamlit (chủ yếu từ trang **7 — Phân tích & Lựa chọn Đặc
trưng**, trang **2 — Huấn luyện**, và module `src/iv_analysis.py`), cộng thêm
các file `credit_info_*.csv` sinh bởi `src/generate_credit_info_report.py`.
Tất cả đều phục vụ 2 bộ dữ liệu song song trong luận văn:

| Ký hiệu | Tên file gốc | Số cột thô | Biến mục tiêu | Ghi chú |
|---|---|---|---|---|
| **Bộ A** | `Data_credit_rating_VN.xlsx` | 21 | `NHOMNOMOI` (Nhóm nợ 1–5) | Dữ liệu nhỏ, dùng ở Trang 2 |
| **Bộ B** | `Thông tin tín dụng 20260430.xlsx` (train) / `20260531.xlsx` (test độc lập) | 41 / 33 (33 cột chung) | `Nhóm nợ tự phân loại` (1–5) | Thay thế fct_l.xlsx, dùng ở Trang 5 |

> **Lưu ý:** hai file rời `grid_search_results.csv` và `grid_search_best.csv`
> vẫn được sinh từ bộ B **cũ** (fct_l.xlsx, đã bị loại khỏi luận văn) — các
> hàng "Bộ A" trong chúng vẫn hợp lệ, nhưng hàng "Bộ B" đã lỗi thời.
>
> Riêng `model_runs_summary.xlsx` **đã được dựng lại**: toàn bộ sheet `B_*`,
> `IV_BoB_*` và phần "Bộ B" của các sheet `GridSearch_*` hiện lấy số liệu từ
> các file `credit_info_*.csv` (bộ B hiện hành — train 20260430 / kiểm tra kỳ
> sau 20260531). Xem sheet `Nguon_DuLieu` trong workbook để biết mỗi sheet
> đến từ file CSV nào; script dựng lại là
> `src/build_model_runs_summary.py` (bản workbook trước khi thay được lưu ở
> `model_runs_summary.xlsx.bak`).

Cả hai đều là bài toán **phân loại đa lớp mất cân bằng mạnh** — Nhóm 1 (nợ đủ
tiêu chuẩn) chiếm phần lớn, Nhóm 3/4 (nợ xấu) rất hiếm (tỉ lệ mất cân bằng ghi
nhận ở Trang 2: ~34.7 lần giữa nhóm nhiều nhất và ít nhất). Điều này giải
thích vì sao **Macro-F1** luôn thấp hơn nhiều so với **Accuracy** trong mọi
bảng dưới đây — Accuracy bị nhóm đa số (Nhóm 1) chi phối, còn Macro-F1 lấy
trung bình đều các nhóm nên phản ánh đúng hơn khả năng phát hiện nợ xấu.

---

## 1. `iv_table_A.csv` / `credit_info_iv_table.csv` — Information Value (IV)

Sinh bởi `src/iv_analysis.py` (hàm `compute_iv_table`). IV/WoE là kỹ thuật
chuẩn Basel II/III để đo **mức độ một biến tách được nợ "tốt" khỏi nợ "xấu"**
trước khi đưa vào mô hình.

Cột | Ý nghĩa
---|---
`feature` | Tên biến đầu vào
`iv` | Information Value — càng cao càng tách lớp tốt
`level` | Xếp hạng theo ngưỡng Siddiqi (2006)

Ngưỡng phân loại `level`:

| IV | Mức | Diễn giải |
|---|---|---|
| < 0.02 | Useless | Bỏ — gần như không có thông tin phân biệt |
| 0.02 – 0.1 | Weak | Yếu |
| 0.1 – 0.3 | Medium | Trung bình — nên giữ |
| 0.3 – 0.5 | Strong | Mạnh |
| 0.5 – 1.0 | Very strong | Rất mạnh |
| > 1.0 | ⚠️ Suspicious (leakage?) | **Nghi ngờ rò rỉ dữ liệu (data leakage)** — biến này thường được tính toán *sau khi* đã biết kết quả xếp nhóm nợ (VD số ngày quá hạn, dư nợ quá hạn tại thời điểm chấm), nên IV cao bất thường không phản ánh khả năng dự báo thực sự |

**Cách dùng trong luận văn:** các biến bị gắn cờ ⚠️ (VD `DAYS_TO_MATURITY`,
`UTIL_RATE` ở Bộ A; `ENG_LOAN_AGE_DAYS` với IV = 1.1170 ở Bộ B) cần được xem
xét loại khỏi tập huấn luyện chính thức vì chúng "nhìn thấy tương lai".
IV_table là **căn cứ định lượng** để chọn ra bộ đặc trưng "đã lọc qua IV"
(11 biến ở Bộ A, tập con ở Bộ B) dùng làm cấu hình chính trong Trang 2/5.

Bảng IV của Bộ B hiện hành (`credit_info_iv_table.csv`, cũng là sheet
`IV_BoB_20bien`) chỉ có **20 dòng** — 33 cột chung giữa 2 kỳ 20260430/20260531
sau khi loại ID/ngày/leakage còn lại 20 đặc trưng. Bảng IV 98 dòng của bộ B cũ
(fct_l.xlsx) đã bị loại khỏi luận văn; file `iv_table_B.csv` tương ứng không
còn trong thư mục này.

---

## 2. `grid_search_results.csv` — Toàn bộ 130 cấu hình đã thử

Kết quả **grid search thủ công** (không phải GridSearchCV của sklearn) chạy
lần lượt từng tổ hợp hyperparameter cho 6 mô hình, trên cả 2 bộ dữ liệu, rồi
đánh giá trên cùng 1 tập test (holdout, `train_test_split` với `stratify`).

Cột | Ý nghĩa
---|---
`Bộ dữ liệu` | A hoặc B
`Mô hình` / `model_key` | Tên hiển thị / khoá nội bộ của thuật toán (`decision_tree`, `random_forest`, `xgboost`, `lightgbm`, `catboost`, `logistic`)
`config_id` | Số thứ tự cấu hình hyperparameter trong grid của mô hình đó
`params` | Bộ hyperparameter cụ thể đã thử (dict dạng chuỗi Python)
`Accuracy` | Tỉ lệ dự đoán đúng trên tổng số (bị lệch bởi nhóm đa số)
`Macro-F1` | Trung bình F1 không trọng số qua 5 nhóm nợ — **chỉ số chính** để so sánh vì dữ liệu mất cân bằng
`Weighted-F1` | Trung bình F1 có trọng số theo support từng nhóm — gần giống Accuracy hơn
`ROC-AUC macro` | AUC trung bình one-vs-rest qua 5 nhóm — đo khả năng phân tách xác suất
`Trạng thái` | `OK` nếu chạy thành công, ngược lại ghi lỗi ở cột `Lỗi`

**130 dòng = 65 dòng Bộ A + 65 dòng Bộ B (cũ)** (số cấu hình mỗi mô hình không
đều nhau vì mỗi thuật toán có grid hyperparameter riêng, ví dụ Logistic
Regression chỉ thử 4 giá trị `C`, Random Forest thử 18 tổ hợp
`n_estimators × max_depth × min_samples_leaf`).

Đây là **dữ liệu thô** dùng để tổng hợp ra `grid_search_best.csv`. Trong file
Excel, chỉ 65 dòng Bộ A của file này còn được dùng; 60 dòng Bộ B lấy từ
`credit_info_grid_search_results.csv`.

---

## 3. `grid_search_best.csv` — Cấu hình tốt nhất mỗi mô hình

Với mỗi tổ hợp (Bộ dữ liệu × Mô hình) — 2 bộ × 6 mô hình = **12 dòng** — chọn
ra dòng có **Macro-F1 cao nhất** trong `grid_search_results.csv`. Cấu trúc cột
giống hệt file trên. Đây chính là bảng "mô hình vô địch" dùng để:

- So sánh 6 thuật toán với nhau công bằng (mỗi thuật toán đã được tinh chỉnh
  tốt nhất có thể trong không gian tìm kiếm đã định).
- Lấy `params` làm cấu hình mặc định khi huấn luyện mô hình chính thức ở
  Trang 2 / Trang 5.

Quan sát nhanh: ở cả 2 bộ, **LightGBM** và **XGBoost** cho Macro-F1/ROC-AUC
cao nhất và ổn định; **CatBoost** cho Accuracy thấp bất thường ở Bộ A
(~0.81 so với ~0.90 của các mô hình khác) dù ROC-AUC vẫn cao — dấu hiệu
CatBoost với cấu hình mặc định dự đoán lệch nhãn nhiều dù xếp hạng xác suất
tốt; **Decision Tree** đơn lẻ và **Logistic Regression** yếu nhất ở nhóm hiếm
(Macro-F1 thấp) vì không xử lý tốt phi tuyến/tương tác giữa biến.

---

## 4. `model_runs_summary.xlsx` — File tổng hợp đầy đủ (16 sheet)

Đây là bản **tổng hợp toàn diện nhất**, gộp mọi kết quả phân tích của Trang 2,
5, 7 và IV vào một workbook, dùng để dán trực tiếp vào luận văn (bảng/biểu).
Phần Bộ A giữ nguyên từ app Streamlit; phần Bộ B lấy từ các file
`credit_info_*.csv` (bộ hiện hành). Dựng lại bằng:

```bash
cd code && python3 -m src.build_model_runs_summary
```

| Sheet | Bộ | Nội dung |
|---|---|---|
| `Nguon_DuLieu` | — | Bảng tra: mỗi sheet lấy số liệu từ nguồn nào, kèm ghi chú về khác biệt giữa 2 bộ |
| `A_SoSanhMoHinh` | A | So sánh 6 mô hình trên Bộ A với cấu hình chính thức (1 dòng/mô hình): Accuracy, Macro-F1, Weighted-F1, ROC-AUC |
| `A_ChiTietTheoNhom` | A | Precision/Recall/F1/Support **theo từng nhóm nợ (1–5)** cho từng mô hình ở Bộ A — cho thấy mô hình yếu ở nhóm nào (thường là Nhóm 2, 3, 4 — ít dữ liệu) |
| `A_TopK_XGBoost` | A | Đường cong hiệu năng XGBoost khi chỉ dùng Top-k đặc trưng quan trọng nhất (k = 3, 5, 7, 9, 11, 13) — trả lời câu hỏi "nhiều đặc trưng hơn có tốt hơn không?" |
| `A_GroupImportance` | A | Gộp importance theo **nhóm nghiệp vụ** (Kỳ hạn khoản vay, Sản phẩm & Mục đích vay, Dư nợ & Hạn mức, Đơn vị/Chi nhánh, Nhân khẩu học, Lãi suất) — trả lời "nhóm đặc trưng nào đóng góp nhiều nhất?" |
| `B_SoSanhMoHinh` | B | 6 mô hình × **2 tầng đánh giá**: kiểm định (20% kỳ 20260430) và kiểm tra kỳ sau (20260531). Mỗi tầng có Accuracy, Macro-F1, Weighted-F1, ROC-AUC macro |
| `B_ChiTietTheoNhom` | B | Precision/Recall/F1/Support theo nhóm nợ 1–5 của **LightGBM** (mô hình dẫn đầu tập kiểm định), trên cả 2 tầng đánh giá. Script chỉ lưu chi tiết cho mô hình tốt nhất, nên khác với `A_ChiTietTheoNhom` (đủ 6 mô hình) |
| `B_TopK_LightGBM` | B | Đường cong hiệu năng theo Top-k đặc trưng (k = 3, 5, 8, 11, 14, 17, 20) — dùng LightGBM vì đây là mô hình tốt nhất của Bộ B (Bộ A dùng XGBoost). Thứ tự đặc trưng lấy theo **gain** (xem ghi chú về `importance_type` bên dưới) |
| `B_GroupImportance` | B | Importance (**Gain**) gộp theo **6 nhóm nghiệp vụ** của Bộ B hiện hành: Dư nợ & Thanh toán 69.3%, Đặc trưng kỹ thuật (engineered) 20.9%, Chi nhánh 4.4%, Lãi suất 4.0%, Hình thức & Mục đích vay 0.7%, Kỳ hạn & Cơ cấu lại 0.7% |
| `B_UuTienDauTu` | B | Kết hợp % đóng góp importance với % dữ liệu thiếu của từng nhóm để tính **"Điểm ưu tiên"** đầu tư thu thập/hoàn thiện dữ liệu — nhóm nào vừa quan trọng vừa đang thiếu nhiều dữ liệu thì ưu tiên cao |
| `IV_BoA_13bien` | A | Bảng IV đầy đủ 13 biến của Bộ A (giống `iv_table_A.csv` nhưng chỉ pool 13 biến khả dụng sau khi loại leakage/trùng lặp/hằng số) |
| `IV_BoB_20bien` | B | Bảng IV 20 biến của Bộ B (= `credit_info_iv_table.csv`). Bộ B cũ có 98 biến; bộ hiện hành chỉ giữ 20 đặc trưng hợp lệ từ 33 cột chung giữa 2 kỳ |
| `GridSearch_ChiTiet125` | A + B | Toàn bộ cấu hình đã thử: **65 dòng Bộ A** (từ `grid_search_results.csv`) + **60 dòng Bộ B** (từ `credit_info_grid_search_results.csv`). Cả 2 bộ đều có đủ `Accuracy` |
| `GridSearch_TotNhat` | A + B | Cấu hình Macro-F1 cao nhất của mỗi mô hình (12 dòng = 6 mô hình × 2 bộ) |
| `GridSearch_DoNhay` | A + B | **Độ nhạy (sensitivity)** của từng mô hình với hyperparameter: Macro-F1 nhỏ nhất/lớn nhất/trung bình/độ lệch chuẩn qua tất cả cấu hình đã thử. Range (max−min) lớn ⇒ mô hình đó nhạy cảm với việc chỉnh tham số (VD Logistic Regression ở Bộ B dao động 0.112 Macro-F1); range nhỏ ⇒ mô hình ổn định, không cần tune nhiều (VD Random Forest ở Bộ A, 0.017) |
| `GridSearch_MacDinh_vs_ToiUu` | A + B | So sánh Macro-F1 giữa **cấu hình mặc định** (`src/models.py::build_pipeline`) và **cấu hình tốt nhất từ grid search**, kèm mức cải thiện tuyệt đối và %. Ở bộ B hiện hành mức cải thiện đều nhỏ (≤ 3%): CatBoost +3.02%, Decision Tree +0.95%, Random Forest +0.89%, LightGBM +0.44%, XGBoost +0.41%, Logistic Regression +0.00% (lưới không tìm được cấu hình nào tốt hơn `C=1.0`) |

### Cách đọc nhanh các sheet `*_ChiTietTheoNhom`

`Support` là số quan sát thực tế của nhóm đó trong tập đánh giá — càng nhỏ thì
Precision/Recall/F1 càng biến động mạnh (dễ thấy giá trị 0.000000 khi mô hình
không dự đoán đúng quan sát nào của nhóm hiếm). Đây là lý do luận văn dùng
thêm **Repeated Stratified K-Fold** (xem
`src/evaluation.py::repeated_stratified_recall`, chạy ở Trang 2) để ước lượng
ổn định hơn cho các nhóm hiếm thay vì tin vào 1 lần chia train/test.

Ở Bộ A mỗi mô hình có 5 dòng (Nhóm 1 → Nhóm 5). Ở Bộ B chỉ có LightGBM nhưng
được lặp 2 lần — một khối cho tập kiểm định, một khối cho kỳ sau 20260531 —
nên đọc kèm cột `Tầng đánh giá`. So sánh 2 khối này chính là bằng chứng về
suy giảm hiệu năng theo thời gian (Recall Nhóm 5 rơi từ 0.9059 xuống 0.2583).

### Cách đọc `*_TopK_*` và mối liên hệ với `GroupImportance`

Hai bảng này cùng trả lời 3 trong 4 câu hỏi nghiên cứu ở Trang 7:
1. *Nhiều đặc trưng hơn có luôn tốt hơn?* → xem đường Macro-F1 theo k có đi
   ngang/giảm sau một điểm nào đó không.
2. *Nhóm nghiệp vụ nào đóng góp lớn nhất?* → xem `%` cao nhất trong
   `*_GroupImportance`.
3. *Giảm còn bao nhiêu đặc trưng mà vẫn giữ hiệu năng?* → k nhỏ nhất mà
   Macro-F1 đã gần đạt đỉnh trong `A_TopK_XGBoost` / `B_TopK_LightGBM`. Ở Bộ B
   hiện hành, k = 8 (trên 20 biến) đạt 0.6961 so với 0.6930 của tập đầy đủ, và
   đỉnh nằm ở k = 17 với 0.7034.
4. *Nên ưu tiên thu thập nhóm nào?* → `B_UuTienDauTu` (chỉ có ở Bộ B vì Bộ A
   không được chấm điểm ưu tiên hoá trong app).

---

## 5. Đối chiếu lần chạy lại 13/09/2026 (Bộ B)

Script `generate_credit_info_report.py` được chạy lại để bổ sung Accuracy.
Đối chiếu 3 chiều (lần chạy mới / bản backup 18/08 / số đang in trong
`luanvan_latex/sections/Chuong4.tex`) cho 4 kết luận:

**(1) Dữ liệu nguồn không đổi.** `credit_info_missing_pct.csv`,
`credit_info_describe_numerical.csv`, `credit_info_outliers.csv` giống hệt bản
cũ về giá trị. Cùng 20 đặc trưng, cùng số dòng.

**(2) Macro-F1 / Weighted-F1 / ROC-AUC tái lập tốt.** Decision Tree và CatBoost
khớp **chính xác đến 4 chữ số** ở cả 2 tầng đánh giá. Bốn mô hình dùng đa
luồng lệch nhẹ:

| Mô hình | Macro-F1 val (mới) | (Chương 4) | Δ |
|---|---|---|---|
| Decision Tree | 0.6590 | 0.6590 | 0.0000 |
| CatBoost | 0.6046 | 0.6046 | 0.0000 |
| LightGBM | 0.6930 | 0.6954 | −0.0024 |
| Random Forest | 0.6587 | 0.6568 | +0.0019 |
| XGBoost | 0.6846 | 0.6941 | −0.0095 |
| Logistic Regression | 0.3959 | 0.4186 | −0.0227 |

Hai mô hình đơn luồng/xác định khớp tuyệt đối, bốn mô hình đa luồng lệch —
dấu hiệu của thứ tự cộng dồn số thực khi huấn luyện song song. **Con số 0.6930
của LightGBM đã được tái lập 2 lần độc lập** (script `generate_imbalance_
comparison.py` ngày 06/09 và lần chạy này), nên chênh lệch 0.0024 so với
0.6954 nêu ở Mục về chiến lược mất cân bằng **là do môi trường/phiên bản thư
viện**, không phải do khác chiến lược: `STRATEGIES_B["SMOTE"]` chính là
`imbalance_strategy="custom"` + `CREDIT_INFO_SMOTE_STRATEGY`, y hệt cấu hình
của `generate_credit_info_report.py`.

**(3) Accuracy của Logistic Regression và CatBoost trong Chương 4 không khớp
pipeline này.** Đây là sai lệch lớn nhất phát hiện được:

| Mô hình | Tầng | Accuracy mới | Chương 4 | Δ |
|---|---|---|---|---|
| LightGBM | kiểm định | 0.9367 | 0.9367 | 0.0000 |
| Random Forest | kiểm định | 0.9337 | 0.9333 | +0.0004 |
| XGBoost | kiểm định | 0.9363 | 0.9350 | +0.0013 |
| Decision Tree | kiểm định | 0.9328 | 0.9350 | −0.0022 |
| **Logistic Regression** | kiểm định | **0.9056** | **0.8151** | **+0.0905** |
| **CatBoost** | kiểm định | **0.6913** | **0.8236** | **−0.1323** |
| **LightGBM** | kỳ sau | **0.9071** | **0.9143** | **−0.0072** |
| **Logistic Regression** | kỳ sau | **0.8915** | **0.7583** | **+0.1332** |
| **CatBoost** | kỳ sau | **0.7066** | **0.7930** | **−0.0864** |

Với Logistic Regression và CatBoost, Macro-F1 / Weighted-F1 / ROC-AUC khớp gần
như tuyệt đối nhưng Accuracy lệch 0.09–0.13 — điều **không thể xảy ra nếu cùng
một tập dự báo**, vì cả 4 chỉ số đều tính từ cùng `y_pred`. Kết luận: 2 giá trị
Accuracy đó trong Chương 4 đến từ một nguồn khác, không phải pipeline này.
Riêng LightGBM kỳ sau, cả bản cũ (0.9068) và bản mới (0.9071) đều khác 0.9143.
**Accuracy trong `B_SoSanhMoHinh` hiện tại là bộ số nhất quán nội bộ** — được
tính từ đúng tập dự báo sinh ra các chỉ số còn lại.

**(4) `group_importance` và `topk` đổi do sửa `importance_type`.** `models.py`
nay đặt `importance_type="gain"` cho LightGBM (trước đó LightGBM mặc định trả
**số lần split**). Hệ quả:

- `B_GroupImportance` đảo thứ hạng: "Dư nợ & Thanh toán" từ 30.8% lên **69.3%**,
  "Đặc trưng kỹ thuật" từ 37.2% xuống **20.9%**.
- `B_TopK_LightGBM` đổi mạnh ở k nhỏ vì thứ tự đặc trưng đổi: k=3 từ 0.2989 lên
  **0.6606**, k=5 từ 0.4346 lên **0.6767**. Nhận định "Macro-F1 giảm rõ rệt tại
  k=3 và k=5" **không còn đúng** với importance theo gain.
- Đỉnh vẫn ở k=17 (0.7034) và k=8 vẫn là ứng viên rút gọn hợp lệ (0.6961 so với
  0.6930 của tập đầy đủ) — kết luận cốt lõi không đổi.
- SHAP gần như không đổi (lệch ≤ 0.019), nên `credit_info_shap_*` vẫn dùng được.

**Thứ hạng mô hình — kết luận của luận văn vẫn đứng vững:** LightGBM vẫn tốt
nhất trên tập kiểm định, Random Forest (0.5065) vẫn tốt nhất trên kỳ sau, và
Macro-F1 của cả 6 mô hình đều giảm khi sang kỳ mới. Chỉ có XGBoost và LightGBM
đổi chỗ với nhau ở kỳ sau (LightGBM 0.4995 > XGBoost 0.4911, trước là ngược
lại).

---

## Lưu ý khi trích số liệu vào luận văn

- **Luôn ưu tiên Macro-F1** làm chỉ số chính khi kết luận "mô hình nào tốt
  hơn" — Accuracy dễ gây hiểu nhầm do mất cân bằng lớp cực đoan (Nhóm 1 chiếm
  77–96% tuỳ bộ dữ liệu).
- Các biến bị gắn nhãn **"⚠️ Suspicious (leakage?)"** trong bảng IV nên được
  giải thích rõ trong luận văn là *đã bị loại khỏi tập huấn luyện chính thức*
  (nếu đúng vậy) hoặc *cần thu thập lại tại thời điểm chấm điểm* — nếu vẫn
  giữ chúng trong mô hình, con số hiệu năng sẽ bị thổi phồng, không phản ánh
  khả năng dự báo thực tế khi triển khai.
- Các sheet `GridSearch_*` trong Excel **không còn** là bản sao của
  `grid_search_*.csv`: phần Bộ A vẫn lấy từ đó, nhưng phần Bộ B đã chuyển sang
  `credit_info_grid_search_*.csv`. Khi cần số liệu Bộ B, trích từ file Excel
  hoặc từ `credit_info_*.csv`, **không** từ `grid_search_*.csv`.
- **Số liệu Bộ B đã được chạy lại ngày 13/09/2026** để bổ sung Accuracy cho cả
  6 mô hình và cả 60 cấu hình grid search (trước đó script chỉ ghi Macro-F1 /
  Weighted-F1 / ROC-AUC). Bản trước khi chạy lại nằm ở
  `_backup_credit_info_pre_accuracy/`, log lần chạy ở
  `credit_info_rerun_log.txt`. Xem mục "Đối chiếu lần chạy lại" bên dưới —
  **một số con số trong Chương 4 cần cập nhật theo lần chạy này**.
