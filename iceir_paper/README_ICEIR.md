# Bài báo ICEIR 2026: ghi chú

- `ICEIR2026_paper.tex`: bản nộp gốc, lớp `llncs` (Springer LNCS). Biên dịch trên Overleaf bằng pdfLaTeX; tải lên kèm thư mục `figures/`. Tệp `llncs.cls` có sẵn trên Overleaf, hoặc lấy từ gói LaTeX của Springer.
- `make_figures.py`: tạo hình của bản nộp gốc từ `code/results/iceir/*.csv`.

## Dữ liệu

Bài chỉ dùng hai kỳ dữ liệu thật 30/4/2026 và 31/5/2026: bài toán 5 nhóm (kể cả kiểm tra ngoài thời gian trên tháng 5) và bài toán chuyển nhóm 31 ngày (cặp 30/4 → 31/5).

## Cập nhật 15/10/2026 (sau góp ý của cô)

- Tệp `Thông tin tín dụng 20260531.xlsx` đã chuẩn hóa 5 cột ngày về `yyyymmdd` (bản gốc ở `code/data/raw/_backup_original/`); `_parse_yyyymmdd` nay báo lỗi khi gặp định dạng khác.
- Kết quả chính: `code/src/rerun_iceir.py` → `code/results/iceir/t1_*`, `t2_*`; phân tích theo góp ý: `code/src/review_iceir.py` → `m0_offbalance_by_group.csv`, `m1_*`, `m2_*`, `m6_*`.
- Bài đổi tên thành "Reproducing or Predicting? ...".
- `ICEIR2026_paper_review.*` và `ICEIR2026_giai_thich_bai_bao.docx` được tạo từ phiên bản trước, chưa cập nhật theo bản này.
- Abstract: câu seen/unseen chuyển sang cấu hình bảo thủ (0,49/0,40), cho khớp góp ý M3.
- Pháp lý đã đối chiếu: NĐ 135/2025/NĐ-CP ngày 12/6/2025 (đã được NĐ 316/2026/NĐ-CP ngày 11/8/2026 sửa đổi, sau hai kỳ dữ liệu); TT 37/2025/TT-NHNN ngày 31/10/2025, hiệu lực 15/12/2025. Ngày của 27/VBHN-NHNN (bài ghi 21/11/2025) và nguyên văn Điều 13(2)(a) vẫn cần kiểm tra tay.
- Sửa chú thích hình SHAP (màu của mã chi nhánh), ghi rõ đơn vị VND, câu "no missing values" và nhận định về ca không cơ học.
- Máy hiện thiếu `libomp`. Khi chạy, cần đặt `DYLD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/sklearn/.dylibs`.
- M6: ngân hàng không cung cấp được mã lý do tất toán (đã ghi trong phần hạn chế). Các tác giả chưa có ORCID.

## Cập nhật 15/10/2026: sửa câu chữ theo rà soát góp ý
- Mục 5.1 mở đầu bằng cấu hình bảo thủ; ghi rõ các kiểm tra độ bền chỉ chạy trên cấu hình đầy đủ; thêm tỷ lệ lãi ngoại bảng > 0 của nhóm 2–5 tháng 5; "−8%" ghi rõ là so với 0,693.
- Bỏ cách nói "48 times the random level"; đơn vị chi phí – lợi ích là triệu VND; thêm sd của R@5% và P@5% vào chú thích Bảng 4.
- Ghi rõ depth 6/8 "respectively"; pairwise LTR dùng mỗi fold huấn luyện là một query (`qid` = 0); dẫn thêm Engelmann et al. (2003) cho CAP; nhắc NĐ 316/2026 sửa đổi NĐ 135 sau hai kỳ dữ liệu.
- `p1_cost_benefit.csv` và `figures/fig3_shap_beeswarm.*` của bản nộp gốc được tạo bởi một script cũ, còn trong lịch sử git (commit 4a42670).

## Cập nhật 25/11/2026: bản sửa theo phản biện (Paper 282)
- Bản mới: `ICEIR2026_paper_rev1.tex` + `references_rev1.bib`; thư trả lời: `ICEIR2026_response_to_reviewers_rev1.md` (và `.docx`). Bản nộp cũ `ICEIR2026_paper.tex` giữ nguyên.
- Phân tích mới: `code/src/review_iceir_rev1.py` (IV, T2G, GRP, ROB, BOOT, MAIN, LRN, EXTRA) → `code/results/iceir/rev1/`.
- Bài toán 2 nay đánh giá chính bằng CV 5x10 chia theo khách hàng (StratifiedGroupKFold); mô hình chính = XGB + 20 biến + quá hạn + G_T.
- Hình mới: `figures/fig2_recall_at_k_rev1.*` (make_figures_rev1.py), `figures/fig3_shap_beeswarm_rev1.*` (EXTRA).
- Cần tải lên Overleaf: tex rev1, references_rev1.bib, thư mục figures (có hai hình _rev1).
- Biên dịch bằng Tectonic 0.17 (XeTeX) → `ICEIR2026_paper_rev1.pdf`, 15 trang. Để vừa 15 trang: bỏ sơ đồ quy trình (Hình 1 cũ), rút gọn vài đoạn. Hình Recall@k nay là Hình 1, SHAP là Hình 2.
