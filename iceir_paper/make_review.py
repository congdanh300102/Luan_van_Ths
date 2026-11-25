"""Build the colour-coded review copy of the ICEIR 2026 paper.

Each paragraph of the compiled PDF is highlighted green (understood and verified
against the data, code or sources) or yellow (not fully verifiable, or needs the
authors' confirmation), with a pop-up note; summary pages in Vietnamese are
prepended.

Run:  python3 iceir_paper/make_review.py <compiled.pdf> <output.pdf>   (needs pymupdf)
"""
import re
import sys

import pymupdf

GREEN, YELLOW = (0.55, 0.88, 0.55), (1.0, 0.86, 0.25)

# (id, section, start words, end words, colour, note)
PARAS = [
    (1, "Tiêu đề, tác giả", "Early Warning of Debt-Group Migration", "danh.dc251317M@sis.hust.edu.vn", "G",
     "ĐÃ ĐỔI TÊN BÀI: 17 từ (trước là 22), nêu rõ bài toán (cảnh báo sớm chuyển nhóm nợ), cách tiếp cận (retrieval trong ngân sách rà soát, khớp chủ đề Efficient Information Retrieval của hội thảo) và đóng góp về rò rỉ dữ liệu. Tác giả và đơn vị khớp bài SEBL 2026; running head: T. T. A. Le, K. T. Le, and C. D. Dam."),
    (2, "Abstract", "Machine-learning models for bank", "56.6% of migrating loans.", "G",
     "Mọi con số (0.699; 0.474; 0.702; 13-32%; 0.27%; 0.168-0.181; 67 lần; 56.6%) đã đối chiếu với các tệp CSV trong code/results và dữ liệu gốc. Tóm tắt dài 221 từ, trong giới hạn 150-250 từ của LNNS."),
    (3, "1. Introduction, đoạn 1", "Vietnamese commercial banks are required", "on tabular credit data", "G",
     "ĐÃ CẬP NHẬT. Căn cứ hạch toán lãi đổi từ Thông tư 16/2018/TT-BTC (bị Thông tư 10/2026/TT-BTC bãi bỏ, hiệu lực 21/3/2026) sang Nghị định 135/2025/NĐ-CP (ngày 12/6/2025, hiệu lực 01/8/2025, thay thế Nghị định 93/2017), là văn bản đang có hiệu lực tại thời điểm dữ liệu. Thông tư 31/2024 (sửa đổi bởi Thông tư 37/2025) và Nghị định 86/2024 vẫn còn hiệu lực. Đã bỏ Nguyen & Ngo (2025) khỏi danh sách 'benchmark'."),
    (4, "1. Introduction, đoạn 2", "Studies of this kind", "relates these questions to the literature.", "G",
     "Đã sửa ở vòng trước: bỏ khẳng định sai về thiết kế của 3 nghiên cứu Việt Nam. Vòng này: danh sách 3 câu hỏi mở được gộp thành 1 câu dẫn sang Mục 2.4 Research Gap, vì hai phần trùng ý."),
    (6, "1. Đóng góp C1-C3", "We examine these questions", "on a label-stratified sample.", "G",
     "Ba đóng góp khớp với phần kết quả và tương ứng với G1-G3. C1 nay dẫn chiếu Nghị định 135/2025."),
    (7, "1. Quan hệ với bài SEBL", "Relation to our prior work.", "is reused.", "G",
     "Đã đối chiếu với SEBL2026_paper.tex. Đoạn được rút gọn còn 2 câu."),
    (9, "1. Bố cục bài", "The remainder of the paper", "Section 7 concludes.", "G", "Đã thêm nhắc đến phần Research Gap."),
    (10, "2. Related work: credit scoring", "Large benchmarks", "migration into overdue debt.", "G",
     "ĐÃ SỬA. Mô tả các nghiên cứu Việt Nam chỉ còn những điểm kiểm chứng được: Nguyen và cộng sự (2024) so sánh 4 mô hình, accuracy 81,84%; bài Truong & Nguyen Vuong chỉ mô tả theo tiêu đề. Câu về khoảng trống được chuyển sang phần Research Gap."),
    (11, "2. Related work: leakage", "Kaufman et al.", "usual safeguard in scorecard development", "G",
     "Ngưỡng IV ghi đúng theo Siddiqi (0,3 trở lên là mạnh); mức 0,5 hoặc 1,0 ghi là thông lệ thực hành."),
    (12, "2. Related work: early warning", "Khandani et al.", "used in Section 4.", "G",
     "ĐÃ SỬA. Lando & Skødeberg mô tả theo đúng tiêu đề và nội dung chính (ước lượng xác suất chuyển hạng). Khandani dùng dữ liệu giao dịch và CIC."),
    (13, "2. Research Gap (MỚI)", "The literature reviewed above leaves three gaps", "to debt-group migration in Vietnam.", "Y",
     "PHẦN MỚI. G1-G3 tương ứng C1-C3. G3 dựa trên Nguyen và cộng sự (2024), đã kiểm chứng: bài này báo cáo accuracy, recall, precision. Các câu 'to our knowledge' / 'we are not aware' là khẳng định về tính mới, không thể kiểm chứng tuyệt đối; tác giả nên tìm thêm tài liệu (Google Scholar, các tạp chí ngân hàng Việt Nam) trước khi nộp."),
    (13.5, "Bảng 1 (nội dung)", "G1 G2 G3 G4 G5 Total", "31/05/2026 88,527 6,339 991 1,421 2,996 100,274", "G",
     "Phân bố nhóm nợ đã tính lại trực tiếp từ hai tệp Excel gốc."),
    (14, "3.1 Dữ liệu", "The data consist of two monthly", "also appear in the May snapshot.", "G",
     "Đã kiểm tra trực tiếp trên hai tệp Excel gốc: số dòng, 33 cột, ETL_DATE duy nhất, khóa không trùng, phân bố nhóm, 98.968 khế ước chung."),
    (15, "3.2 Bài toán 1", "Given the feature vector", "generalisation to new customers.", "G",
     "Số mẫu 80.493/20.124 khớp. Kỳ tháng 5 không dùng để chọn mô hình."),
    (16, "3.3 Bài toán 2", "denote the debt group of loan", "cannot be ruled out.", "G",
     "Đã tính lại từ dữ liệu gốc: 265 sự kiện (214/10/10/31), 166 cải thiện, 99,56%, 99,73%, 1.649 khoản bị loại, tỷ lệ loại 1,80% so với 0,20-0,46%."),
    (17, "4. Phương pháp: mở đầu và Hình 1", "Figure 1 summarises the pipeline.", "a fixed random seed", "G",
     "Khớp mã nguồn (tiền xử lý chỉ fit trên tập huấn luyện, seed 42)."),
    (18, "4.1 Lớp 1", "Layer 1: label restatements.", "directly encode the label.", "G",
     "Danh sách 7 trường khớp ID_LEAKAGE_COLS trong code. 'Ngày chậm trả' là trường ngày tháng. Dự phòng tính theo tỷ lệ của Nghị định 86/2024 (còn hiệu lực)."),
    (19, "4.1 Lớp 2", "Layer 2: identifiers", "zero for every record.", "G",
     "6 trường gồm 4 định danh, 'Thời điểm truy đòi' và 'Số tiền đã thanh toán'. Khớp code: 33 − 1 − 7 − 6 − 3 + 4 = 20."),
    (20, "4.1 Lớp 3", "Layer 3: rule-grounded review", "seasoning (recently disbursed", "G",
     "ĐÃ CẬP NHẬT. Căn cứ đổi sang Điều 13 khoản 2 điểm a Nghị định 135/2025/NĐ-CP: lãi phải thu chỉ hạch toán vào doanh thu với nợ đủ tiêu chuẩn, lãi của các khoản khác theo dõi ngoại bảng. Đã kiểm chứng qua bản tiếng Anh trên LuatVietnam; nội dung giống Điều 5 khoản 2 điểm a Thông tư 16/2018 cũ. Các tỷ lệ và IV đã tính lại từ dữ liệu gốc. Diễn giải 'seasoning' đã viết thận trọng ('more likely')."),
    (21, "4.1 Giữ hai biến lãi", "Both interest fields are", "the change up to", "Y",
     "Giả định quan trọng nhất của bài: hai biến lãi có sẵn tại T cho bài toán 2. Chưa có xác nhận từ ngân hàng về thời điểm cập nhật (đã nêu ở hạn chế thứ hai)."),
    (22, "4.2 Đặc trưng", "After the audit, 20 features remain", "with the training mode.", "G",
     "Khớp code: 12 biến số (8 gốc + 4 tạo thêm), 8 phân loại; biến cơ cấu được điền 0."),
    (23, "4.2 Mô hình", "Six learners are compared", "as a sensitivity check.", "G",
     "Siêu tham số khớp src/models.py; chiến lược SMOTE khớp config.py."),
    (24, "4.3 Đánh giá", "Task 1. Macro-F1 is the primary", "and 447 randomly selected non-events.", "G",
     "Công thức đúng; hệ số hiệu chỉnh Nadeau-Bengio 1/J + n_test/n_train với J = 50."),
    (25, "5.1 Kết quả bài toán 1", "Table 2 reports both evaluation layers.", "rather than weak ranking ability.", "G",
     "Khớp rev2_b_models.csv và rev2_b_grid.csv. Chạy lại với XGBoost 3.2/LightGBM 4.6 chênh ±0,005-0,015, do khác phiên bản thư viện."),
    (26, "Bảng 2 (nội dung)", "Model Acc Macro-F1 AUC MAE", "−3.3%", "G", "Đối chiếu từng ô với CSV."),
    (27, "5.1 Theo từng nhóm", "Per-group performance.", "rather than missed risk.", "G",
     "Khớp rev2_b_by_class.csv và rev2_b_confusion.csv."),
    (28, "5.1 Đóng góp biến rò rỉ", "Contribution of the leaking field.", "anticipates risk.", "G",
     "Khớp rev2_b_leak.csv và rev2_b_gain_shap_feature.csv (Gain 54,8%, SHAP 31,2%)."),
    (29, "Bảng 3 (nội dung)", "Configuration Internal test", "0.474 / 0.167", "G", "Khớp rev2_b_leak.csv."),
    (31, "5.1 Robustness (MỚI)", "Robustness.", "rather than on point estimates.", "G",
     "Đoạn tóm tắt mới thay cho Mục 5.3 cũ (đã chuyển xuống Phụ lục B) và gộp ý của đoạn thảo luận 'Reporting statistical ties' (đã bỏ vì trùng)."),
    (30, "5.2 Lỗi định dạng ngày", "In an earlier out-of-time run", "misinterpreted as dataset shift", "G",
     "Đã xác nhận trên tệp gốc: tháng 4 lưu 20220817, tháng 5 lưu 45965 (= 04/11/2025); tuổi khoản vay trung vị 395 → 426 ngày; mức giảm 12,7-32,1% lấy từ bản trước khi sửa lỗi."),
    (32, "5.3 Bài toán 2", "Table 4 reports, for each learner", "despite a ROC-AUC of 0.62.", "G",
     "Khớp transition_cv_summary.csv và revision_transition_tests.csv. Chạy lại XGBoost: AP 0,179 ± 0,046 so với 0,181 ± 0,049."),
    (33, "Bảng 4 (nội dung)", "Model Strategy AP ROC-AUC", "0.000 0.000 0.001", "G",
     "Khớp transition_cv_summary.csv và revision_transition_tests.csv."),
    (34, "5.3 Ảnh hưởng của xử lý mất cân bằng", "Effect of imbalance treatment.", "class weighting offers no benefit.", "G",
     "Đã sửa ở vòng trước: tác động của resampling phụ thuộc mô hình (LightGBM: SMOTE 0,176 so với không xử lý 0,163)."),
    (35, "5.3 Góc nhìn ngân sách rà soát", "Review-budget perspective.", "ROC-AUC 0.98", "G",
     "Khớp fixed_recall_at_k_xgb.csv và diagnose_d7 (cặp 1). Câu '6/53' đã bỏ ở vòng trước."),
    (36, "Bảng 5 (nội dung)", "Budget k Loans reviewed", "35 66.0% 1.8% 6.6", "G", "Khớp fixed_recall_at_k_xgb.csv; Lift = Recall@k / k."),
    (37, "5.3 Giải thích SHAP", "Explanation of the ranking.", "(maximum 0.44).", "G",
     "ĐÃ SỬA. 'Gain = 0' (không có trong tệp lưu) đổi thành 'SHAP = 0' (khớp revision_transition_shap_stratified.csv). Thứ hạng 8 biến đầu trùng transition_shap_top.csv."),
    (38, "6. Thảo luận: phân loại so với cảnh báo sớm", "Current-state classification versus early warning.", "rather than by its agreement with the current label.", "G",
     "Nhận định có điều kiện về thời điểm cập nhật biến lãi."),
    (39, "6. Thảo luận: nguồn gốc theo quy định", "Regulatory provenance as a leakage criterion.", "before any model is fitted.", "G",
     "ĐÃ SỬA: lập luận phản thực tế 'would have identified' đổi thành 'could flag'."),
    (40, "6. Thảo luận: kiểm tra toàn vẹn", "Integrity checks for out-of-time evaluation.", "in either direction.", "G", "Đã rút gọn, khớp Mục 5.2."),
    (42, "7. Kết luận", "Using two consecutive monthly snapshots", "grounded in accounting regulation.", "G", "Tóm tắt nhất quán với kết quả."),
    (43, "7. Hạn chế", "Two limitations are most important.", "of missed deterioration.", "G", "Hai hạn chế phản ánh đúng các điểm yếu chính."),
    (44, "Data availability", "Data availability.", "upon reasonable request.", "Y",
     "Tác giả cần xác nhận: có sẵn sàng chia sẻ mã nguồn theo yêu cầu không, và điều khoản thỏa thuận với ngân hàng."),
    (44.5, "Disclosure of interests (MỚI)", "Disclosure of interests.", "the content of this article.", "Y",
     "Mục mới theo thông lệ kỷ yếu Springer. Tác giả cần xác nhận không có xung đột lợi ích (ví dụ: tác giả có làm việc tại ngân hàng cung cấp dữ liệu hay không)."),
    (45, "Acknowledgements", "Acknowledgements.", "used for lending decisions.", "Y",
     "Tác giả cần xác nhận ngân hàng đồng ý được cảm ơn (ẩn danh) và nhận định 'không mô hình nào được dùng để cho vay'."),
    (8, "Phụ lục A, Bảng 6", "Aspect SEBL 2026 paper", "SHAP for rare events", "G",
     "ĐÃ CHUYỂN xuống Phụ lục A để phần chính gọn hơn; phần Mở đầu vẫn giữ đoạn nêu điểm mới so với bài SEBL. Cột căn trái, không còn ngắt chữ 'con-trol'."),
    (31, "Phụ lục B: Độ nhạy", "Imbalance strategy (LightGBM", "versus 38.8%.", "G",
     "ĐÃ CHUYỂN xuống Phụ lục B. Khớp rev2_b_imbalance, rev2_s1, rev2_s2, rev2_b_topk; Spearman 0,886/0,964 khớp rev2_m_log.txt."),
    (46, "Tài liệu tham khảo", "Baesens, B., Van Gestel", "https://doi.org/10.17654/0972361725006", "G",
     "Đã đối chiếu DOI với Crossref và NeurIPS. Mục Thông tư 16/2018 được thay bằng Nghị định 135/2025/NĐ-CP. Mục [11] (bài SEBL, 'to appear') cần cập nhật khi có thông tin kỷ yếu."),
]

FIXES = [
    "Thêm phần Research Gap (cuối Mục 2): G1 sàng lọc rò rỉ chưa theo nguồn gốc quy định; G2 kết quả ngoài thời gian chưa được kiểm tra toàn vẹn; G3 cảnh báo sớm chưa được đánh giá như bài toán tìm kiếm trong ngân sách rà soát. G1-G3 tương ứng với C1-C3.",
    "Cập nhật căn cứ pháp lý: Thông tư 16/2018/TT-BTC đã bị bãi bỏ bởi Thông tư 10/2026/TT-BTC (hiệu lực 21/3/2026). Bài nay trích Điều 13 khoản 2 điểm a Nghị định 135/2025/NĐ-CP (ngày 12/6/2025, hiệu lực 01/8/2025), văn bản đang có hiệu lực tại thời điểm dữ liệu.",
    "Kiểm tra Thông tư 31/2024/TT-NHNN (sửa đổi bởi 37/2025) và Nghị định 86/2024/NĐ-CP: vẫn còn hiệu lực. Dự thảo sửa đổi Nghị định 86 (tháng 8/2026) chưa ban hành.",
    "Bỏ Nguyen & Ngo (2025) khỏi danh sách 'benchmark'; mô tả bài Truong & Nguyen Vuong chỉ theo tiêu đề; viết lại mô tả Lando & Skødeberg.",
    "Mục 5.4: 'ba biến cơ cấu có Gain = 0' (không có trong tệp lưu) đổi thành 'SHAP = 0' (đã kiểm chứng).",
    "Viết thận trọng hơn: diễn giải 'seasoning' và lập luận phản thực tế ở Mục 6.",
    "Đổi tên bài thành 'Early Warning of Debt-Group Migration as Budget-Constrained Retrieval: Regulation-Grounded Leakage Control in a Vietnamese Commercial Bank'; cập nhật running title và từ khóa (thêm Early warning, Rare-event retrieval).",
    "Chuyển Bảng so sánh với bài SEBL xuống Phụ lục A và các phân tích độ nhạy (Mục 5.3 cũ) xuống Phụ lục B; thêm đoạn tóm tắt 'Robustness' ở Mục 5.1; gộp danh sách 3 câu hỏi ở Mục 1 vào Mục 2.4; bỏ đoạn thảo luận trùng 'Reporting statistical ties'. Lưu ý: phụ lục vẫn tính vào giới hạn 15 trang.",
    "Trình bày theo mẫu Springer: Mục 2 chia thành 4 tiểu mục đánh số 2.1-2.4 (trước là tiêu đề đoạn và một tiểu mục không đánh số); thêm mục 'Disclosure of interests'. Đã kiểm tra: lớp llncs, không gói thay đổi bố cục, chú thích bảng ở trên và hình ở dưới, tóm tắt 226 từ, 6 từ khóa, running head, 15 trang.",
    "Sửa diễn đạt số liệu: '100.617 hợp đồng trong hai kỳ' đổi thành 100.617 và 100.274 hợp đồng; 'khoảng 67 lần' đổi thành 'tối đa 67 lần' (0,168 chỉ gấp 63 lần); ghi rõ 56,6% là kết quả trên tập chia 80/20.",
    "Rút gọn: danh sách 3 câu hỏi ở Mục 1, đoạn quan hệ với bài SEBL, đoạn thảo luận về kiểm tra toàn vẹn. Bài vẫn 15 trang, trong giới hạn 12-15 trang.",
]

FIXES_PREV = [
    "Mục 1, đoạn 2: bỏ khẳng định sai về thiết kế của 3 nghiên cứu Việt Nam.",
    "Mục 2: trích dẫn Moreno-Torres cho 'out-of-time validation' sai đối tượng; ngưỡng IV 1,0 không phải của Siddiqi.",
    "Mục 4.1: 'Ngày chậm trả' là trường ngày tháng; dự phòng không phải 'hàm xác định' của nhãn.",
    "Mục 5.3: chiến lược kết hợp tốt nhất về Recall nhóm 4 và balanced accuracy.",
    "Mục 5.4: kết luận 'resampling hơi có hại' mâu thuẫn với LightGBM; bỏ câu '6/53'.",
    "Mục 6: 'unable' thành 'largely fails'.",
]


LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl", "−": "-"}


def norm(t):
    for k, v in LIGATURES.items():
        t = t.replace(k, v)
    return re.sub(r"[^0-9a-z@.%,/+\-]", "", t.lower()).strip(".,;:")


def tokens(doc):
    out = []  # (norm_text, page_no, rect)
    heads = ("T. T. A. Le, K. T. Le", "Leakage-Audited Ranking of Debt-Group Migration")
    for pno, page in enumerate(doc):
        words = page.get_text("words", sort=True)
        # drop the running head (first line of every page after the title page)
        if pno and words:
            top = min(w[1] for w in words)
            head = [w for w in words if abs(w[1] - top) < 3]
            if any(h.split()[0] in " ".join(w[4] for w in head) for h in heads):
                words = [w for w in words if abs(w[1] - top) >= 3]
        i = 0
        while i < len(words):
            w = words[i]
            txt, rects = w[4], [pymupdf.Rect(w[:4])]
            # re-join words hyphenated across a line break
            while txt.endswith("-") and i + 1 < len(words) and words[i + 1][1] > w[3] - 2:
                i += 1
                txt = txt[:-1] + words[i][4]
                rects.append(pymupdf.Rect(words[i][:4]))
            n = norm(txt)
            if n:
                for r in rects:
                    out.append((n, pno, r))
                # mark continuation rects so a phrase match spans them once
            i += 1
    return out


def find(seq, phrase, start=0):
    want = [norm(x) for x in phrase.split() if norm(x)]
    flat = [t[0] for t in seq]
    for i in range(start, len(flat) - len(want) + 1):
        if all(flat[i + j] == want[j] for j in range(len(want))):
            return i, i + len(want) - 1
    return None


def main(src, dst):
    global PARAS
    PARAS = [(i + 1,) + p[1:] for i, p in enumerate(PARAS)]  # number paragraphs consecutively
    doc = pymupdf.open(src)
    seq = tokens(doc)
    missing, pos = [], 0
    for pid, sec, a, b, col, note in PARAS:
        s = find(seq, a, pos) or find(seq, a, 0)
        if not s:
            missing.append((pid, "start", a))
            continue
        e = find(seq, b, s[0])
        if not e:
            missing.append((pid, "end", b))
            continue
        pos = s[0]
        lines = {}
        for n, pno, r in seq[s[0]:e[1] + 1]:
            key = (pno, round(r.y0 / 2))
            lines[key] = lines.get(key, pymupdf.Rect(r)) | r
        colour = GREEN if col == "G" else YELLOW
        first = None
        pages = {}
        for (pno, _), r in sorted(lines.items()):
            page = pages.setdefault(pno, doc[pno])
            ann = page.add_highlight_annot(r)
            ann.set_colors(stroke=colour)
            ann.set_info(title=f"Đoạn {pid}", content=note)
            ann.update()
            first = first or (pno, r)
        pno, r = first
        page = pages[pno]
        icon = page.add_text_annot(pymupdf.Point(max(r.x0 - 22, 4), r.y0), note, icon="Comment")
        icon.set_colors(stroke=colour)
        icon.set_info(title=f"Đoạn {pid} – {sec}")
        icon.update()
    if missing:
        print("NOT FOUND:", *missing, sep="\n  ")
        sys.exit(1)

    n_g = sum(p[4] == "G" for p in PARAS)
    n_y = len(PARAS) - n_g
    rows = "".join(
        f"<tr><td>{p[0]}</td><td>{p[1]}</td>"
        f"<td><b style='color:{'#1e7b1e' if p[4]=='G' else '#b07800'}'>{'Xanh' if p[4]=='G' else 'Vàng'}</b></td>"
        f"<td>{p[5]}</td></tr>" for p in PARAS)
    fixes = "".join(f"<li>{f}</li>" for f in FIXES)
    fixes_prev = "".join(f"<li>{f}</li>" for f in FIXES_PREV)
    yellow = "".join(f"<li><b>Đoạn {p[0]} ({p[1]})</b>: {p[5]}</li>" for p in PARAS if p[4] == "Y")
    html = f"""
<h1 style='font-size:17px'>BẢN NHẬN XÉT BÀI BÁO ICEIR 2026</h1>
<p style='font-size:10px'><i>Early Warning of Debt-Group Migration as Budget-Constrained Retrieval: Regulation-Grounded Leakage Control in a Vietnamese Commercial Bank</i></p>
<p style='font-size:10px'>Cách đọc: mỗi đoạn trong bài được tô <span style='background:#bfe8bf'>&nbsp;xanh&nbsp;</span> nếu đã hiểu rõ và kiểm chứng đúng với dữ liệu gốc, mã nguồn hoặc nguồn trích dẫn; tô <span style='background:#ffe08a'>&nbsp;vàng&nbsp;</span> nếu chưa kiểm chứng được đầy đủ hoặc cần tác giả xác nhận. Biểu tượng ghi chú cạnh mỗi đoạn mở ra lời nhận xét chi tiết. Tổng cộng {len(PARAS)} đoạn: {n_g} xanh, {n_y} vàng.</p>
<h2 style='font-size:13px'>1. Các chỉnh sửa trong vòng này</h2>
<ol style='font-size:10px'>{fixes}</ol>
<p style='font-size:10px'><b>Đã sửa ở vòng rà soát trước:</b></p>
<ul style='font-size:9.5px'>{fixes_prev}</ul>
<h2 style='font-size:13px'>2. Các đoạn tô vàng: việc tác giả cần làm</h2>
<ul style='font-size:10px'>{yellow}</ul>
<h2 style='font-size:13px'>3. Ghi chú về dữ liệu</h2>
<p style='font-size:10px'>Bài chỉ dùng hai kỳ thật 30/04 và 31/05/2026. Kỳ 20260630 KHÔNG được dùng: cả 30 cột dữ liệu của 98.620 khế ước trùng 100% với kỳ tháng 5, chỉ khác cột nhóm nợ, tức là kỳ mô phỏng. Toàn bộ số liệu thống kê mô tả đã được tính lại trực tiếp từ hai tệp Excel gốc và khớp với bài.</p>
<h2 style='font-size:13px'>4. Bảng nhận xét chi tiết theo từng đoạn</h2>
<table style='font-size:8.5px;border-collapse:collapse' border='1' cellpadding='3'>
<tr><th>#</th><th>Vị trí</th><th>Màu</th><th>Nhận xét</th></tr>{rows}</table>
"""
    import io
    story = pymupdf.Story(html=f"<body style='font-family:sans-serif'>{html}</body>")
    mediabox = doc[0].rect
    where = mediabox + (40, 40, -40, -40)
    buf = io.BytesIO()
    writer = pymupdf.DocumentWriter(buf)
    more = True
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    cover = pymupdf.open("pdf", buf.getvalue())
    doc.insert_pdf(cover, start_at=0)
    doc.save(dst, garbage=3, deflate=True)
    print(f"saved {dst}: {len(cover)} summary pages + paper; {n_g} green, {n_y} yellow")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
