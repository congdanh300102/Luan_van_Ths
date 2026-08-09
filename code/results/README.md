# Giải thích kết quả trong `code/results/`

Thư mục này chứa các file được **sinh tự động** từ những lần chạy huấn luyện /
phân tích trong app Streamlit (chủ yếu từ trang **7 — Phân tích & Lựa chọn Đặc
trưng**, trang **2 — Huấn luyện**, và module `src/iv_analysis.py`). Tất cả đều
phục vụ 2 bộ dữ liệu song song trong luận văn:

| Ký hiệu | Tên file gốc | Số cột thô | Biến mục tiêu | Ghi chú |
|---|---|---|---|---|
| **Bộ A** | `Data_credit_rating_VN.xlsx` | 21 | `NHOMNOMOI` (Nhóm nợ 1–5) | Dữ liệu nhỏ, dùng ở Trang 2 |
| **Bộ B** | `fct_l.xlsx` | 178 | `CLASSIFICATION` (Nhóm nợ 1–5) | Dữ liệu lớn, dùng ở Trang 5 |

Cả hai đều là bài toán **phân loại đa lớp mất cân bằng mạnh** — Nhóm 1 (nợ đủ
tiêu chuẩn) chiếm phần lớn, Nhóm 3/4 (nợ xấu) rất hiếm (tỉ lệ mất cân bằng ghi
nhận ở Trang 2: ~34.7 lần giữa nhóm nhiều nhất và ít nhất). Điều này giải
thích vì sao **Macro-F1** luôn thấp hơn nhiều so với **Accuracy** trong mọi
bảng dưới đây — Accuracy bị nhóm đa số (Nhóm 1) chi phối, còn Macro-F1 lấy
trung bình đều các nhóm nên phản ánh đúng hơn khả năng phát hiện nợ xấu.

---

## 1. `iv_table_A.csv` / `iv_table_B.csv` — Information Value (IV)

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
`UTIL_RATE` ở Bộ A; `SEAB_POLICY`, `ACC_PD_BAL_RATE` ở Bộ B) cần được xem xét
loại khỏi tập huấn luyện chính thức vì chúng "nhìn thấy tương lai". IV_table
là **căn cứ định lượng** để chọn ra bộ đặc trưng "đã lọc qua IV" (11 biến ở Bộ
A, tập con ở Bộ B) dùng làm cấu hình chính trong Trang 2/5.

`iv_table_B.csv` có tới 98 dòng vì Bộ B nhiều đặc trưng thô hơn — phần lớn rơi
vào "Medium/Useless", cho thấy dữ liệu Bộ B tuy nhiều cột nhưng nhiều cột
trùng lặp/gần như hằng số (IV = 0.1155 lặp lại ở rất nhiều biến gợi ý các
biến đó có cùng phân phối/tương quan cao với nhau).

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

**130 dòng = 67 dòng Bộ A + 63 dòng Bộ B** (số cấu hình mỗi mô hình không đều
nhau vì mỗi thuật toán có grid hyperparameter riêng, ví dụ Logistic Regression
chỉ thử 4 giá trị `C`, Random Forest thử 18 tổ hợp `n_estimators × max_depth ×
min_samples_leaf`).

Đây là **dữ liệu thô** dùng để tổng hợp ra 2 file/ sheet tiếp theo
(`grid_search_best.csv` và các sheet `GridSearch_*` trong file Excel).

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

## 4. `model_runs_summary.xlsx` — File tổng hợp đầy đủ (15 sheet)

Đây là bản **tổng hợp toàn diện nhất**, gộp mọi kết quả phân tích của Trang 2,
5, 7 và IV vào một workbook, dùng để dán trực tiếp vào luận văn (bảng/biểu).

| Sheet | Nội dung |
|---|---|
| `A_SoSanhMoHinh` | So sánh 6 mô hình trên Bộ A với cấu hình chính thức (1 dòng/mô hình): Accuracy, Macro-F1, Weighted-F1, ROC-AUC |
| `A_ChiTietTheoNhom` | Precision/Recall/F1/Support **theo từng nhóm nợ (1–5)** cho từng mô hình ở Bộ A — cho thấy mô hình yếu ở nhóm nào (thường là Nhóm 2, 3, 4 — ít dữ liệu) |
| `A_TopK_XGBoost` | Đường cong hiệu năng XGBoost khi chỉ dùng Top-k đặc trưng quan trọng nhất (k = 3, 5, 7, 9, 11, 13) — trả lời câu hỏi "nhiều đặc trưng hơn có tốt hơn không?" |
| `A_GroupImportance` | Gộp importance theo **nhóm nghiệp vụ** (Kỳ hạn khoản vay, Sản phẩm & Mục đích vay, Dư nợ & Hạn mức, Đơn vị/Chi nhánh, Nhân khẩu học, Lãi suất) — trả lời "nhóm đặc trưng nào đóng góp nhiều nhất?" |
| `B_SoSanhMoHinh` | Giống `A_SoSanhMoHinh` nhưng cho Bộ B |
| `B_ChiTietTheoNhom` | Giống `A_ChiTietTheoNhom` nhưng cho Bộ B |
| `B_TopK_XGBoost` | Đường cong hiệu năng theo Top-k đặc trưng cho Bộ B (k = 5…98) |
| `B_GroupImportance` | Importance gộp theo 9 nhóm nghiệp vụ của Bộ B (Dư nợ & Giải ngân, Lãi suất, Sản phẩm/Mục đích/Kênh, Trả nợ & Thu hồi, …) |
| `B_UuTienDauTu` | Kết hợp % đóng góp importance với % dữ liệu thiếu của từng nhóm để tính **"Điểm ưu tiên"** đầu tư thu thập/hoàn thiện dữ liệu — nhóm nào vừa quan trọng vừa đang thiếu nhiều dữ liệu thì ưu tiên cao |
| `IV_BoA_13bien` | Bảng IV đầy đủ 13 biến của Bộ A (giống `iv_table_A.csv` nhưng chỉ pool 13 biến khả dụng sau khi loại leakage/trùng lặp/hằng số) |
| `IV_BoB_98bien` | Bảng IV 98 biến của Bộ B (giống `iv_table_B.csv`) |
| `GridSearch_ChiTiet130` | = `grid_search_results.csv` (130 dòng thô) |
| `GridSearch_TotNhat` | = `grid_search_best.csv` (12 dòng best-per-model) |
| `GridSearch_DoNhay` | **Độ nhạy (sensitivity)** của từng mô hình với hyperparameter: Macro-F1 nhỏ nhất/lớn nhất/trung bình/độ lệch chuẩn qua tất cả cấu hình đã thử. Range (max−min) lớn ⇒ mô hình đó nhạy cảm với việc chỉnh tham số (VD Decision Tree ở Bộ B dao động 0.12 Macro-F1); range nhỏ ⇒ mô hình ổn định, không cần tune nhiều (VD Random Forest) |
| `GridSearch_MacDinh_vs_ToiUu` | So sánh Macro-F1 giữa **cấu hình mặc định** (do người dùng/paper hay chọn sẵn) và **cấu hình tốt nhất từ grid search**, kèm mức cải thiện tuyệt đối và % — cho thấy việc tune hyperparameter có đáng công không (VD Decision Tree ở Bộ B cải thiện tới +35.7%, nhưng XGBoost/LightGBM ở Bộ A gần như không đổi vì cấu hình mặc định đã trùng với cấu hình tốt nhất) |

### Cách đọc nhanh các sheet `*_ChiTietTheoNhom`

Mỗi mô hình có 5 dòng (Nhóm 1 → Nhóm 5). `Support` là số quan sát thực tế của
nhóm đó trong tập test — càng nhỏ thì Precision/Recall/F1 càng biến động
mạnh (dễ thấy giá trị 0.000000 khi mô hình không dự đoán đúng quan sát nào
của nhóm hiếm). Đây là lý do luận văn dùng thêm **Repeated Stratified K-Fold**
(xem `src/evaluation.py::repeated_stratified_recall`, chạy ở Trang 2) để ước
lượng ổn định hơn cho các nhóm hiếm thay vì tin vào 1 lần chia train/test.

### Cách đọc `*_TopK_XGBoost` và mối liên hệ với `GroupImportance`

Hai bảng này cùng trả lời 3 trong 4 câu hỏi nghiên cứu ở Trang 7:
1. *Nhiều đặc trưng hơn có luôn tốt hơn?* → xem đường Macro-F1 theo k có đi
   ngang/giảm sau một điểm nào đó không.
2. *Nhóm nghiệp vụ nào đóng góp lớn nhất?* → xem `%` cao nhất trong
   `*_GroupImportance`.
3. *Giảm còn bao nhiêu đặc trưng mà vẫn giữ hiệu năng?* → k nhỏ nhất mà
   Macro-F1 đã gần đạt đỉnh trong `*_TopK_XGBoost`.
4. *Nên ưu tiên thu thập nhóm nào?* → `B_UuTienDauTu` (chỉ có ở Bộ B vì Bộ A
   chỉ có 6 nhóm nghiệp vụ, ít cần ưu tiên hoá).

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
- File `grid_search_*.csv` và các sheet `GridSearch_*` trong Excel là **cùng
  một nguồn dữ liệu**, chỉ khác cách tổng hợp — không cần chạy lại nếu chỉ
  cần một trong hai định dạng.
