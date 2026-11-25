# Bài báo ICEIR 2026: ghi chú

- `ICEIR2026_paper.tex`: bản thảo hoàn chỉnh, lớp `llncs` (Springer LNNS). Biên dịch trên Overleaf bằng pdfLaTeX; tải lên kèm thư mục `figures/`. Tệp `llncs.cls` có sẵn trên Overleaf, hoặc lấy từ gói LaTeX của Springer.
- `make_figures.py`: tạo Hình 2 và Hình 3 từ `code/results/*.csv`. Hình 1 là sơ đồ TikZ nằm ngay trong `.tex`.

## Dữ liệu được dùng và không được dùng

Bài chỉ dùng hai kỳ thật là 20260430 và 20260531: bài toán 5 nhóm (kể cả kiểm tra ngoài thời gian) và cặp chuyển nhóm 1.

Cặp 2 (20260531 → 20260630) **không** được dùng. Kết quả "cặp 2" trong luận văn
(`transition_pair2_*`, `fixed_transition_cv.csv` dòng cap2, `diagnose_*`) trùng khớp đúng
với kỳ **mô phỏng** do `src/generate_synthetic_period.py` sinh ra:

- `transition_multiperiod_demo.csv` (531_to_630_sim): 98.620 khoản vay, 258 sự kiện, 0,2616%.
- "cặp 2 thật": 98.620 khoản vay khớp, 258 sự kiện, 0,2616%.
- 100% khế ước của kỳ 20260630 đều có ở kỳ 20260531, tức không có khoản giải ngân mới nào trong tháng 6. Trong khi đó kỳ 20260531 có khoảng 1.306 khế ước mới so với tháng 4.
- Ở kỳ mô phỏng, nhãn chỉ phụ thuộc nhóm nợ hiện tại, nên không đặc trưng nào mang tín hiệu (AUC đơn biến ≈ 0,5). Đây đúng là hiện tượng "tín hiệu biến mất" được mô tả trong luận văn.

**Đã xác nhận ngày 12/10/2026** trên tệp `Thông tin tín dụng 20260630.xlsx` vừa được cập nhật vào `code/data/raw`:
cả 30 cột dữ liệu (dư nợ, lãi nội bảng, lãi ngoại bảng, dự phòng cụ thể...) của 98.620 khế ước
**trùng 100%** với kỳ 20260531; chỉ cột `Nhóm nợ tự phân loại` và `ETL_DATE` là khác.
Để so sánh, giữa hai kỳ thật 20260430 và 20260531, dư nợ chỉ trùng 90,5% và lãi nội bảng chỉ trùng 61,7%.
Ngoài ra, 258 khoản vay xấu đi ở tháng 6 vẫn giữ nguyên dự phòng cụ thể, điều không thể xảy ra theo
Nghị định 86/2024. Vậy tệp này là kỳ mô phỏng, và cần sửa Mục 3.5 (Chương 3), Chương 4 và Phụ lục của luận văn.

## Cập nhật 15/10/2026 (sau góp ý của cô)

- Tệp `Thông tin tín dụng 20260531.xlsx` đã chuẩn hóa 5 cột ngày về `yyyymmdd` (bản gốc ở `code/data/raw/_backup_original/`); `_parse_yyyymmdd` nay báo lỗi khi gặp định dạng khác.
- Kết quả chính: `code/src/rerun_iceir.py` → `code/results/iceir/t1_*`, `t2_*`; phân tích theo góp ý: `code/src/review_iceir.py` → `m0_*`, `m1_*`, `m2_*`, `m6_*`.
- Bài đổi tên thành "Reproducing or Predicting? ..." (12 trang). Kiểm tra M0 xác nhận tệp tháng 6 là kỳ mô phỏng (xem `ICEIR2026_giai_trinh_gop_y.docx`).
- `ICEIR2026_paper_review.*` và `ICEIR2026_giai_thich_bai_bao.docx` được tạo từ phiên bản trước, chưa cập nhật theo bản này.

## Cập nhật 15/10/2026 (lần 3): đính chính về tệp tháng 6
- Mục "lần 2" trước đây ghi `Thông tin tín dụng 20260630.xlsx` là sổ cho vay tháng 6 thật. Ghi nhận đó **sai**. Tệp (tạo bằng openpyxl lúc 10:24:04Z ngày 15/10/2026) là kỳ mô phỏng được sinh lại: so với `_old.xlsx` (sinh bởi generate_synthetic_period_v2.py), tệp có cùng 99.916 dòng theo cùng thứ tự, các trường tĩnh trùng 100%, còn các trường biến động chỉ trùng một phần. Số hợp đồng rời danh mục (1.660) và số hợp đồng mới (1.302) khớp đúng tham số của script.
- Bài báo quay về chỉ dùng hai kỳ thật (W1: 4→5). Bản có W2 lưu ở `.ICEIR2026_paper_W2_synthetic_june.tex.bak` (kèm `.pdf.bak`) và **không được nộp**.
- Từ `review_iceir2.py`, bài chỉ dùng các kết quả của cặp thật: `p1_cost_benefit.csv` và `figures/fig3_shap_beeswarm.*`. Các tệp `p2_*`, `tr_transfer.csv`, `t1_june_subsets.csv` và cột tháng 6 của `off_by_group_3months.csv` đều dựa trên dữ liệu mô phỏng.
- Đã viết lại `ICEIR2026_giai_trinh_gop_y.docx`; bản cũ lưu ở `.ICEIR2026_giai_trinh_gop_y_W2_synthetic_june.docx.bak`.

## Cập nhật 15/10/2026 (lần 4): tệp 20260630 lưu bằng Excel lúc 11:52Z
- Kiểm tra lại tệp `Thông tin tín dụng 20260630.xlsx` mới (Excel for Mac, tạo 11:52:07Z). Kết luận: vẫn là kỳ mô phỏng, **không dùng**.
- 99.916 dòng, 1.660 hợp đồng rời đi, 1.302 hợp đồng mới: khớp đúng tham số của `generate_synthetic_period_v2.py`.
- Hợp đồng chung giữ nguyên thứ tự dòng của tháng 5 (100%). 13 trường tĩnh (chi nhánh, tên, tiền tệ, ngày kết thúc...) trùng 100,00%; giữa hai kỳ thật 4→5 các trường này vẫn có thay đổi nhỏ (99,27–99,99%).
- 100% hợp đồng "mới" tháng 6 thuộc khách hàng đã có ở tháng 5 (thật 4→5: 54,8%). Không có mã khế ước tiền tố `0000` (thật: 357/1.306); thay vào đó là các số 12 chữ số ngẫu nhiên.
- Tỷ lệ khoản quá hạn được chữa khỏi trong tháng: 3,3% (thật 4→5: 16,8%).
- Abstract: câu seen/unseen chuyển sang cấu hình bảo thủ (0,49/0,40), cho khớp góp ý M3.

## Cập nhật 15/10/2026 (lần 5): tệp 20260630 thay lúc 20:03
- Tệp mới (tạo 12:34Z, trường tác giả để trống) có **cùng 99.916 số khế ước** với tệp ở lần 4. 28/32 trường dữ liệu trùng 100%, ma trận chuyển nhóm giống hệt. Chỉ khác ở bốn trường định danh (chi nhánh, mã khách hàng, tên khách hàng, số hợp đồng), bị thay bằng `CN_xxxxxx`/`KH_xxxxxx`, và thứ tự dòng bị xáo trộn. Đây vẫn là kỳ mô phỏng, **không dùng**.
- Pháp lý đã đối chiếu: NĐ 135/2025/NĐ-CP ngày 12/6/2025 (đã được NĐ 316/2026/NĐ-CP ngày 11/8/2026 sửa đổi, sau hai kỳ dữ liệu); TT 37/2025/TT-NHNN ngày 31/10/2025, hiệu lực 15/12/2025. Ngày của 27/VBHN-NHNN (bài ghi 21/11/2025) và nguyên văn Điều 13(2)(a) vẫn cần kiểm tra tay.
- Bài: sửa chú thích Hình 3 (màu của mã chi nhánh), ghi rõ đơn vị VND, câu "no missing values" và nhận định về ca không cơ học.

## Cập nhật 15/10/2026 (lần 6): đưa cặp 5→6 vào bài dưới dạng cửa sổ MÔ PHỎNG
- Theo yêu cầu của tác giả, cặp 31/5 → 30/6 được đưa vào bài thành **W2-sim** (Mục 5.3, Bảng `tab:sim`). Bài ghi rõ đây là dữ liệu mô phỏng, không phải dữ liệu ngân hàng, và chỉ dùng để kiểm thử quy trình đánh giá.
- Chạy lại `python3 -m src.review_iceir2 P2 TR CB` trên tệp hiện tại. Kết quả: `p2_*`, `tr_transfer.csv`, `log_review3_sim.txt`. Kết quả cũ (tính trên phiên bản tháng 6 trước, 268 sự kiện) lưu ở `_p2_old_june_version/`.
- Tệp hiện tại: 98.614 khoản vay, 257 sự kiện, 190 ca cơ học. Đặc trưng lấy từ tháng 5, tháng 6 chỉ đóng góp nhóm nợ, nên việc định danh tháng 6 bị thay bằng mã giả không ảnh hưởng.
- Không dùng tệp tháng 6 cho Bài toán 1 (`t1_june_subsets.csv`): mã chi nhánh tháng 6 đã bị thay bằng mã giả.
- Máy hiện thiếu `libomp`. Khi chạy, cần đặt `DYLD_LIBRARY_PATH=/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/sklearn/.dylibs`.
- M6: ngân hàng không cung cấp được mã lý do tất toán (đã ghi trong phần hạn chế). Các tác giả chưa có ORCID.

## Cập nhật 15/10/2026 (lần 7): tệp 20260630 lưu lúc 21:11
- Tệp lúc 20:56 bị cắt cụt (XML sheet1 thiếu thẻ đóng ở dòng 99.917) nên không đọc được. Tệp lúc 21:11 đọc được bình thường.
- Tệp 21:11 có cùng 99.916 số khế ước với tệp lúc 20:03 và ma trận chuyển nhóm 5→6 giống hệt; trường định danh lại là tên/mã dạng thật.
- Đã chạy lại `review_iceir2 P2 TR CB` (log: `results/iceir/log_review4.txt`). Mọi số trong `p2_*` và `tr_transfer.csv` trùng tuyệt đối (sai khác 0) với lần chạy trên tệp 20:03, nên Mục 5.3 và Bảng `tab:sim` không cần sửa.

## Cập nhật 15/10/2026 (lần 8): sửa câu chữ theo rà soát góp ý
- Bản trước khi sửa lưu ở `.ICEIR2026_paper_before_review_fixes.tex.bak`.
- Mục 5.1 mở đầu bằng cấu hình bảo thủ; ghi rõ các kiểm tra độ bền chỉ chạy trên cấu hình đầy đủ; thêm tỷ lệ lãi ngoại bảng > 0 của nhóm 2–5 tháng 5; "−8%" ghi rõ là so với 0,693.
- Bỏ cách nói "48 times the random level"; đơn vị chi phí – lợi ích là triệu VND; thêm sd của R@5% và P@5% vào chú thích Bảng 4; thêm R@5% cho cột CV của Bảng `tab:sim`.
- Ghi rõ depth 6/8 "respectively"; pairwise LTR dùng mỗi fold huấn luyện là một query (`qid` = 0); dẫn thêm Engelmann et al. (2003) cho CAP; nhắc NĐ 316/2026 sửa đổi NĐ 135 sau hai kỳ dữ liệu.

## Cập nhật 15/10/2026 (lần 9): bỏ hẳn W2-sim
- Theo quyết định của tác giả, bài chỉ dùng hai kỳ thật (30/4 và 31/5/2026). Đã xóa Mục 5.3, Bảng `tab:sim`, dòng tháng 6 ở Bảng 1, các nhắc đến W2-sim trong Abstract, C3, Mục 3.2, 4.1, 6 và 7. Bảng 1 thêm dòng tỷ lệ % của tháng 5.
- Bản có W2-sim (ghi là mô phỏng) lưu ở `.ICEIR2026_paper_with_W2sim_labelled.tex.bak`. Lưu ý: `.ICEIR2026_paper_W2_synthetic_june.tex.bak` cũng là bản CÓ W2, không phải bản chỉ có cặp thật.
- Các tệp `p2_*`, `tr_transfer.csv` không còn được bài sử dụng.

## Cập nhật 25/11/2026: bản sửa theo phản biện (Paper 282)
- Bản mới: `ICEIR2026_paper_rev1.tex` + `references_rev1.bib`; thư trả lời: `ICEIR2026_response_to_reviewers_rev1.md`. Bản nộp cũ `ICEIR2026_paper.tex` giữ nguyên.
- Phân tích mới: `code/src/review_iceir_rev1.py` (IV, T2G, GRP, ROB, BOOT, MAIN, LRN, EXTRA) → `code/results/iceir/rev1/`.
- Bài toán 2 nay đánh giá chính bằng CV 5x10 chia theo khách hàng (StratifiedGroupKFold); mô hình chính = XGB + 20 biến + quá hạn + G_T.
- Hình mới: `figures/fig2_recall_at_k_rev1.*` (make_figures_rev1.py), `figures/fig3_shap_beeswarm_rev1.*` (EXTRA).
- Cần tải lên Overleaf: tex rev1, references_rev1.bib, thư mục figures (có hai hình _rev1).
- 25/11: biên dịch bằng Tectonic 0.17 (XeTeX, tải tạm vào scratchpad) → `ICEIR2026_paper_rev1.pdf`, 15 trang. Để vừa 15 trang: bỏ sơ đồ quy trình (Hình 1 cũ), rút gọn vài đoạn; bản trước khi rút ở `.ICEIR2026_paper_rev1_before_trim.tex.bak`. Hình Recall@k nay là Hình 1, SHAP là Hình 2.
