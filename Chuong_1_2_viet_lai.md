# LỜI MỞ ĐẦU

Trong hoạt động ngân hàng, phân loại nợ không chỉ là yêu cầu ghi nhận chất lượng tài sản tại một thời điểm mà còn là cơ sở để trích lập dự phòng, quản lý danh mục và tổ chức các biện pháp xử lý tín dụng. Tuy nhiên, nếu ngân hàng chỉ nhận diện rủi ro sau khi khoản vay đã chuyển sang nhóm nợ có mức độ rủi ro cao hơn thì khả năng can thiệp thường bị thu hẹp và chi phí xử lý có thể gia tăng. Vì vậy, vấn đề có ý nghĩa quản trị không chỉ là xác định khách hàng đang thuộc nhóm nợ nào, mà còn là dự báo khách hàng có khả năng chuyển nhóm nợ trong một khoảng thời gian sắp tới.

Sự phát triển của dữ liệu và học máy tạo điều kiện để ngân hàng khai thác đồng thời nhiều nguồn thông tin về khách hàng, khoản vay, tài sản bảo đảm, lịch sử tín dụng, hành vi thanh toán và quan hệ tín dụng. Bộ dữ liệu sử dụng trong nghiên cứu ban đầu có 178 đặc trưng. Quy mô đặc trưng lớn mang lại khả năng mô tả khách hàng chi tiết hơn, nhưng không đồng nghĩa rằng toàn bộ đặc trưng đều tạo thêm giá trị dự báo. Những biến trùng lặp, ít biến động, có mức độ thiếu cao hoặc cùng phản ánh một tín hiệu có thể làm mô hình phức tạp hơn, tăng chi phí dữ liệu và nguy cơ quá khớp mà không cải thiện đáng kể hiệu năng ngoài mẫu.

Từ vấn đề trên, luận văn được định hướng theo đề tài **“Ứng dụng trí tuệ nhân tạo trong xếp hạng tín dụng: nghiên cứu từ một ngân hàng thương mại ở Việt Nam”**. Nghiên cứu không chỉ tìm mô hình có khả năng dự báo phù hợp, mà còn xem xét giá trị gia tăng của từng đặc trưng và từng nhóm thông tin nghiệp vụ. Trọng tâm là xác định liệu nhiều đặc trưng hơn có luôn làm mô hình tốt hơn; nhóm đặc trưng nào đóng góp nhiều nhất; có thể rút gọn 178 đặc trưng xuống mức nào mà vẫn duy trì gần như toàn bộ hiệu năng; và nếu ngân hàng tiếp tục đầu tư dữ liệu thì nên ưu tiên nhóm thông tin nào.

Để trả lời các vấn đề này, nghiên cứu xây dựng các tập đặc trưng lồng nhau theo thứ tự đóng góp, huấn luyện và đánh giá mô hình trên cùng một sơ đồ kiểm định. Hiệu năng của tập đầy đủ 178 đặc trưng được dùng làm mốc tham chiếu. Một tập rút gọn chỉ được chấp nhận khi mức suy giảm hiệu năng nằm trong ngưỡng xác định trước và kết quả ổn định trên dữ liệu ngoài mẫu. Đồng thời, SHAP, permutation importance và thí nghiệm loại bỏ/bổ sung từng nhóm đặc trưng được sử dụng để lượng hóa đóng góp ở cả cấp độ biến và cấp độ nhóm nghiệp vụ.

Kết quả kỳ vọng của nghiên cứu gồm hai phần có quan hệ chặt chẽ. Thứ nhất là một mô hình dự báo khả năng chuyển nhóm nợ, cung cấp tín hiệu hỗ trợ cán bộ quản trị rà soát các khoản vay có nguy cơ suy giảm chất lượng trước khi trạng thái xấu được ghi nhận. Thứ hai là bằng chứng định lượng phục vụ chiến lược dữ liệu: ngân hàng biết nhóm thông tin nào thực sự tạo giá trị, mức độ rút gọn hợp lý và phần dữ liệu nào cần được ưu tiên chuẩn hóa hoặc mở rộng. Theo cách tiếp cận này, luận văn hướng tới một giải pháp có hiệu năng dự báo, có khả năng giải thích và có tính khả thi khi triển khai, thay vì chỉ lựa chọn thuật toán đạt điểm số cao nhất.

# CHƯƠNG 1. GIỚI THIỆU NGHIÊN CỨU

## 1.1. Bối cảnh nghiên cứu

Hoạt động cấp tín dụng tạo ra nguồn thu quan trọng cho ngân hàng thương mại, đồng thời cũng làm phát sinh một trong những loại rủi ro trọng yếu nhất của ngân hàng: rủi ro tín dụng. Theo Basel Committee on Banking Supervision (2000), rủi ro tín dụng có thể được hiểu là khả năng bên vay hoặc đối tác không thực hiện đầy đủ nghĩa vụ đã cam kết. Khi rủi ro hiện thực hóa, tổn thất không chỉ giới hạn ở phần gốc và lãi không thu hồi được mà còn làm tăng chi phí dự phòng, giảm lợi nhuận, suy giảm khả năng cung ứng tín dụng và có thể ảnh hưởng tới an toàn của toàn hệ thống.

Trong thực tiễn quản trị ngân hàng, nhận diện sớm khả năng khách hàng chuyển thành nợ xấu có ý nghĩa khác với việc chỉ ghi nhận nợ xấu sau khi khoản vay đã quá hạn. Hệ thống cảnh báo sớm cho phép ngân hàng chủ động rà soát hồ sơ, điều chỉnh hạn mức, tăng cường giám sát, làm việc sớm với khách hàng hoặc áp dụng biện pháp thu hồi phù hợp. Vì vậy, bài toán dự báo rủi ro tín dụng cần được xem là một bài toán hỗ trợ quyết định trước tổn thất, thay vì chỉ là một bài toán phân loại mang tính kỹ thuật.

Tại Việt Nam, khuôn khổ phân loại tài sản có đối với ngân hàng thương mại, tổ chức tín dụng phi ngân hàng và chi nhánh ngân hàng nước ngoài hiện được quy định tại Thông tư số 31/2024/TT-NHNN, được sửa đổi, bổ sung bởi Thông tư số 37/2025/TT-NHNN và thể hiện trong Văn bản hợp nhất số 27/VBHN-NHNN ngày 21/11/2025. Theo khuôn khổ này, nợ được phân thành năm nhóm theo mức độ rủi ro tăng dần; trong đó nhóm 3, nhóm 4 và nhóm 5 là nợ xấu. Cách phân loại pháp lý là căn cứ bắt buộc để xác định biến mục tiêu của nghiên cứu. Điều này có nghĩa là nhãn “nợ xấu” trong dữ liệu không nên được xây dựng tùy ý, mà cần được ánh xạ rõ với trạng thái khoản vay và quy định có hiệu lực tại thời điểm quan sát.

Bên cạnh dữ liệu nội bộ, hoạt động thông tin tín dụng do Trung tâm Thông tin tín dụng Quốc gia Việt Nam (CIC) làm đầu mối theo Thông tư số 15/2023/TT-NHNN tạo ra nguồn thông tin quan trọng về lịch sử quan hệ tín dụng và nghĩa vụ của khách hàng. Tuy nhiên, dữ liệu nhiều nguồn cũng kéo theo các vấn đề về độ đầy đủ, tính nhất quán, sai lệch thời điểm, giá trị thiếu và rò rỉ thông tin. Do đó, chất lượng dữ liệu và cách xác định thời điểm dự báo là điều kiện nền tảng để một mô hình có giá trị sử dụng thực tế.

Các phương pháp chấm điểm tín dụng truyền thống chủ yếu dựa trên phân tích chuyên gia, phân tích biệt thức và hồi quy logistic. Công trình gốc của Altman (1968) sử dụng mẫu ghép cặp 66 doanh nghiệp sản xuất, gồm 33 doanh nghiệp phá sản và 33 doanh nghiệp không phá sản. Từ 22 tỷ số tài chính ban đầu, tác giả lựa chọn 5 tỷ số để xây dựng hàm phân biệt Z-score. Trên mẫu ước lượng và tại thời điểm một năm trước phá sản, mô hình phân loại đúng khoảng 95% số doanh nghiệp. Đây là một kết quả có ảnh hưởng lớn, nhưng cần nhấn mạnh rằng nó được tính trên mẫu nhỏ, cân bằng và thuộc doanh nghiệp sản xuất Hoa Kỳ; con số 95% không thể được xem là mức chính xác kỳ vọng khi áp dụng trực tiếp cho khách hàng ngân hàng Việt Nam. Ohlson (1980) khắc phục một phần hạn chế của thiết kế ghép cặp khi ước lượng mô hình logit trên 105 doanh nghiệp phá sản và 2.058 doanh nghiệp không phá sản, qua đó chuyển trọng tâm từ một điểm phân biệt sang xác suất phá sản. Trong lĩnh vực tín dụng tiêu dùng, Hand và Henley (1997) và Thomas (2000) hệ thống hóa nền tảng thống kê của credit scoring và behavioural scoring. Ưu điểm của hồi quy logistic là cấu trúc rõ ràng, dễ kiểm tra dấu của hệ số và thuận lợi khi giải thích quyết định. Tuy nhiên, quan hệ giữa đặc điểm khách hàng với xác suất vỡ nợ có thể phi tuyến, có tương tác và thay đổi theo thời gian; đây là những khía cạnh mà mô hình tuyến tính khó nắm bắt nếu không thiết kế biến thủ công.

Sự phát triển của học máy mở rộng đáng kể tập phương pháp dùng cho chấm điểm tín dụng. Random Forest kết hợp lấy mẫu bootstrap và lựa chọn ngẫu nhiên biến tại từng nút để giảm phương sai của cây quyết định (Breiman, 2001). Gradient Boosting xây dựng tuần tự các mô hình yếu nhằm hiệu chỉnh sai số của mô hình trước (Friedman, 2001). XGBoost bổ sung cơ chế chính quy hóa và triển khai có khả năng mở rộng; bài báo gốc cho thấy kiến trúc này có thể xử lý quy mô hàng tỷ quan sát trong các bài toán benchmark, nhưng đây là minh chứng về khả năng tính toán chứ không phải bằng chứng riêng về hiệu năng tín dụng (Chen & Guestrin, 2016). LightGBM sử dụng các kỹ thuật lấy mẫu và gom đặc trưng để tăng hiệu quả tính toán trên dữ liệu lớn; trong thử nghiệm của Ke et al. (2017), thời gian huấn luyện nhanh hơn GBDT truyền thống tới trên 20 lần trong khi độ chính xác gần tương đương. Kết quả tốc độ này phụ thuộc phần cứng, dữ liệu và cấu hình thử nghiệm, nên không được hiểu là LightGBM luôn nhanh hơn XGBoost đúng 20 lần trong mọi ứng dụng.

Việc ứng dụng học máy trong tín dụng cũng đặt ra ba thách thức. Thứ nhất, dữ liệu nợ xấu thường mất cân bằng: số khách hàng tốt lớn hơn nhiều số khách hàng xấu. Nếu chỉ tối đa hóa độ chính xác, mô hình có thể dự đoán tốt lớp đa số nhưng bỏ sót nhiều trường hợp rủi ro. Thứ hai, chi phí của hai loại sai lầm không giống nhau. Dự báo một khách hàng xấu thành tốt có thể gây tổn thất tín dụng, còn dự báo một khách hàng tốt thành xấu làm mất cơ hội kinh doanh và ảnh hưởng trải nghiệm khách hàng. Thứ ba, mô hình có hiệu năng cao nhưng không giải thích được sẽ khó kiểm định, khó giám sát và khó sử dụng trong quy trình ra quyết định có trách nhiệm. SHAP, dựa trên giá trị Shapley, là một hướng tiếp cận nhằm giải thích đóng góp của từng biến cho dự báo ở cấp độ toàn cục và từng hồ sơ (Lundberg & Lee, 2017).

Từ các vấn đề trên, nghiên cứu lựa chọn cách tiếp cận so sánh mô hình thống kê với các mô hình học máy trên cùng một quy trình dữ liệu, đồng thời xem xét mất cân bằng lớp, lựa chọn thước đo phù hợp và khả năng giải thích. Điểm khác biệt là hiệu năng được phân tích theo quy mô và nhóm đặc trưng, thay vì chỉ so sánh tên thuật toán. Tập đầy đủ 178 đặc trưng được dùng làm mốc tham chiếu; các tập rút gọn được đánh giá để tìm số lượng đặc trưng nhỏ nhất vẫn duy trì gần như toàn bộ hiệu năng ngoài mẫu.

## 1.2. Lý do lựa chọn đề tài

Đề tài được lựa chọn vì bốn lý do chính.

Thứ nhất, nợ xấu có tác động trực tiếp đến chất lượng tài sản, chi phí dự phòng và hiệu quả kinh doanh của ngân hàng. Một cải thiện nhỏ trong khả năng phát hiện sớm khách hàng rủi ro có thể mang lại giá trị quản trị đáng kể, đặc biệt khi được áp dụng trên danh mục tín dụng lớn.

Thứ hai, dữ liệu tín dụng ngày càng có quy mô và mức độ đa dạng cao. Ngoài thông tin hồ sơ tại thời điểm cấp tín dụng, ngân hàng có thể khai thác lịch sử thanh toán, dư nợ, tần suất quá hạn, biến động dòng tiền và thông tin quan hệ tín dụng. Cấu trúc dữ liệu này tạo điều kiện cho học máy phát hiện quan hệ phi tuyến và tương tác biến mà phương pháp truyền thống có thể bỏ sót.

Thứ ba, bằng chứng trong tài liệu chưa cho phép khẳng định một mô hình duy nhất luôn tối ưu. Lessmann et al. (2015) thực hiện một trong những đối sánh quy mô lớn trong credit scoring: 41 bộ phân loại được đánh giá trên 8 bộ dữ liệu tín dụng bán lẻ thực bằng 6 thước đo. Các tác giả tìm thấy nhiều mô hình dự báo chính xác hơn hồi quy logistic và nhóm ensemble không đồng nhất có kết quả nổi bật, nhưng thứ hạng thay đổi theo dữ liệu và thước đo. Brown và Mues (2012) sử dụng 5 bộ dữ liệu tín dụng thực, chia hai phần ba cho huấn luyện và một phần ba cho kiểm định, sau đó giảm dần tỷ lệ khách hàng xấu từ cấu hình ban đầu 70/30 tới mức cực đoan 99/1. Ở mức 99% tốt–1% xấu, Random Forest đạt thứ hạng AUC trung bình tốt nhất; C4.5, phân tích biệt thức bậc hai và k-láng giềng suy giảm đáng kể so với nhóm tốt nhất. Những kết quả này là bằng chứng rõ về ảnh hưởng của mất cân bằng, nhưng không thay thế việc kiểm định trên dữ liệu Việt Nam.

Thứ tư, hiệu năng dự báo chỉ là một điều kiện của mô hình ứng dụng trong ngân hàng. Mô hình còn phải có khả năng kiểm tra, giải thích, tái lập và giám sát. Việc kết hợp mô hình học máy với SHAP cho phép nghiên cứu xem xét đồng thời hai khía cạnh: khả năng phân biệt rủi ro và khả năng lý giải dự báo.

## 1.3. Vấn đề nghiên cứu

Vấn đề trung tâm của nghiên cứu là xây dựng và đánh giá quy trình dự báo khách hàng hoặc khoản vay có khả năng chuyển từ nhóm nợ hiện tại sang nhóm nợ có mức độ rủi ro cao hơn trong một cửa sổ dự báo xác định. Mô hình hướng tới hỗ trợ phân loại nợ và tạo tín hiệu cảnh báo sớm, không chỉ nhận diện nợ xấu đã phát sinh.

Bên cạnh hiệu năng dự báo, nghiên cứu giải quyết bài toán hiệu quả đặc trưng. Bộ dữ liệu có 178 đặc trưng thuộc nhiều nhóm nghiệp vụ, nhưng chi phí thu thập, làm sạch, liên kết và duy trì dữ liệu không giống nhau. Nếu một tập nhỏ hơn giữ gần như toàn bộ hiệu năng, mô hình rút gọn có thể giảm độ phức tạp, hạn chế thiếu dữ liệu và thuận lợi hơn cho triển khai. Ngược lại, nếu một số nhóm đặc trưng tạo mức tăng hiệu năng rõ rệt, đó là căn cứ để ngân hàng ưu tiên đầu tư dữ liệu.

Nghiên cứu không đồng nhất “mô hình tốt” với mô hình có Accuracy cao nhất. Một mô hình được xem là phù hợp khi đáp ứng đồng thời các yêu cầu:

1. Có khả năng phân biệt khách hàng tốt và khách hàng xấu trên dữ liệu ngoài mẫu;
2. Nhận diện đủ tốt lớp nợ xấu, là lớp thiểu số nhưng có ý nghĩa quản trị lớn;
3. Duy trì độ ổn định qua các tập kiểm định hoặc qua thời gian;
4. Có thể giải thích các biến thúc đẩy rủi ro ở cấp danh mục và cấp khách hàng;
5. Không sử dụng thông tin phát sinh sau thời điểm dự báo, qua đó tránh rò rỉ dữ liệu;
6. Có khả năng tích hợp vào quy trình nghiệp vụ và giám sát mô hình;
7. Xác định được tập đặc trưng rút gọn có hiệu quả và đóng góp của từng nhóm đặc trưng nghiệp vụ.

## 1.4. Mục tiêu nghiên cứu

### 1.4.1. Mục tiêu tổng quát

Mục tiêu tổng quát của nghiên cứu là đề xuất mô hình dự báo khả năng chuyển nhóm nợ nhằm hỗ trợ công tác phân loại nợ và cảnh báo sớm rủi ro tín dụng; đồng thời xác định tập đặc trưng và nhóm thông tin nghiệp vụ tạo ra giá trị dự báo lớn nhất để hỗ trợ ngân hàng tối ưu hóa mô hình và định hướng đầu tư dữ liệu.

### 1.4.2. Mục tiêu cụ thể

Nghiên cứu hướng tới các mục tiêu cụ thể sau:

1. Hệ thống hóa cơ sở lý thuyết về rủi ro tín dụng, nợ xấu, chấm điểm tín dụng và dự báo vỡ nợ;
2. Làm rõ căn cứ pháp lý dùng để xác định nhãn nợ xấu và nguyên tắc sử dụng thông tin tín dụng;
3. Xây dựng quy trình chuẩn bị dữ liệu, bao gồm kiểm tra chất lượng, xử lý giá trị thiếu, mã hóa biến, kiểm soát ngoại lệ và ngăn ngừa rò rỉ dữ liệu;
4. Xây dựng mô hình hồi quy logistic làm mô hình cơ sở và so sánh với các mô hình cây/ensemble như Decision Tree, Random Forest, XGBoost và LightGBM;
5. Đánh giá liệu việc sử dụng đầy đủ 178 đặc trưng có tạo ra hiệu năng tốt hơn một cách ổn định so với các tập đặc trưng rút gọn;
6. Xác định nhóm đặc trưng nghiệp vụ đóng góp lớn nhất cho dự báo chuyển nhóm nợ bằng SHAP, permutation importance và thí nghiệm loại bỏ/bổ sung theo nhóm;
7. Xác định số lượng đặc trưng tối thiểu có thể giữ gần như toàn bộ hiệu năng của mô hình 178 đặc trưng theo một ngưỡng chấp nhận được xác định trước;
8. Đánh giá tác động của biện pháp xử lý mất cân bằng, trong đó SMOTE chỉ được áp dụng trên tập huấn luyện;
9. So sánh mô hình bằng ROC-AUC, PR-AUC, Recall, Precision, F1-score và ma trận nhầm lẫn; đồng thời xem xét hiệu chỉnh xác suất chuyển nhóm;
10. Đề xuất thứ tự ưu tiên thu thập, chuẩn hóa và mở rộng các nhóm thông tin nghiệp vụ dựa trên mức đóng góp dự báo và chi phí dữ liệu;
11. Đề xuất cách triển khai, giải thích và giám sát mô hình trong hoạt động quản trị rủi ro.

## 1.5. Câu hỏi nghiên cứu

Nghiên cứu trả lời các câu hỏi sau:

**Câu hỏi 1:** Nhiều đặc trưng hơn có luôn giúp mô hình dự báo khả năng chuyển nhóm nợ tốt hơn không?

**Câu hỏi 2:** Nhóm đặc trưng nghiệp vụ nào đóng góp lớn nhất cho việc phân loại và dự báo chuyển nhóm nợ?

**Câu hỏi 3:** Có thể giảm từ 178 xuống bao nhiêu đặc trưng mà vẫn giữ gần như toàn bộ hiệu năng của mô hình?

**Câu hỏi 4:** Nếu ngân hàng muốn đầu tư mở rộng dữ liệu, nên ưu tiên thu thập và chuẩn hóa những nhóm thông tin nào để đạt hiệu quả dự báo cao nhất?

Để hỗ trợ trả lời bốn câu hỏi chính, nghiên cứu đồng thời xem xét hai câu hỏi phương pháp bổ trợ:

**Câu hỏi 5:** Trên cùng dữ liệu và sơ đồ kiểm định, mô hình học máy có cải thiện khả năng dự báo chuyển nhóm nợ so với hồi quy logistic hay không?

**Câu hỏi 6:** Việc xử lý mất cân bằng ảnh hưởng như thế nào đến khả năng phát hiện trường hợp chuyển nhóm và sự đánh đổi giữa Recall với Precision?

## 1.6. Đối tượng và phạm vi nghiên cứu

### 1.6.1. Đối tượng nghiên cứu

Đối tượng nghiên cứu là mối quan hệ giữa các đặc điểm của khách hàng/khoản vay với khả năng phát sinh nợ xấu, cùng các phương pháp thống kê và học máy dùng để ước lượng mối quan hệ đó.

Đơn vị quan sát cần được xác định thống nhất theo dữ liệu thực tế, chẳng hạn “một khách hàng tại một thời điểm chốt dữ liệu” hoặc “một khoản vay tại thời điểm quan sát”. Nếu một khách hàng có nhiều khoản vay, nghiên cứu phải quy định rõ cách tổng hợp hoặc cách xử lý phụ thuộc giữa các quan sát.

### 1.6.2. Phạm vi nội dung

Nghiên cứu tập trung vào dự báo sự dịch chuyển trạng thái nợ trong một cửa sổ tương lai. Ở cấu hình nhị phân, biến mục tiêu nhận giá trị 1 nếu khoản vay chuyển từ nhóm hiện tại sang nhóm có mức độ rủi ro cao hơn trong cửa sổ dự báo và nhận giá trị 0 nếu không chuyển nhóm. Tùy khả năng của dữ liệu, nghiên cứu có thể báo cáo bổ sung bài toán đa lớp dự báo nhóm nợ đích, nhưng bài toán chuyển/không chuyển là cấu hình chính để phục vụ cảnh báo sớm.

Để bảo đảm ý nghĩa nghiệp vụ, luận văn cần xác định rõ nhóm nợ tại ngày quan sát, nhóm nợ trong cửa sổ tương lai và cách xử lý trường hợp khoản vay tất toán hoặc quay về nhóm tốt hơn. Việc gộp nhóm 3, 4 và 5 thành nợ xấu chỉ là một phân tích bổ sung; mục tiêu chính không giới hạn ở thời điểm đã phát sinh nợ xấu.

Nghiên cứu không thay thế quy trình phê duyệt tín dụng hoặc quyết định của cán bộ có thẩm quyền. Kết quả mô hình là thông tin hỗ trợ và phải được sử dụng cùng các quy định nghiệp vụ, kiểm soát tuân thủ và đánh giá chuyên gia.

### 1.6.3. Phạm vi dữ liệu và thời gian

Phạm vi ngân hàng, phân khúc khách hàng, giai đoạn dữ liệu, số lượng quan sát và danh sách biến sẽ được mô tả chính xác theo bộ dữ liệu được cấp. Việc chia dữ liệu ưu tiên theo thời gian nếu dữ liệu có nhiều kỳ, bởi chia ngẫu nhiên có thể làm kết quả lạc quan hơn so với tình huống triển khai trên khách hàng tương lai.

Thông tin sau ngày quan sát, chẳng hạn số ngày quá hạn được ghi nhận sau khi khách hàng đã chuyển nợ xấu, kết quả xử lý nợ hoặc biến được tạo trực tiếp từ nhãn, không được dùng làm biến đầu vào.

## 1.7. Phương pháp nghiên cứu

Nghiên cứu kết hợp tổng quan tài liệu, phân tích định lượng và thực nghiệm mô hình.

Ở bước tổng quan, nghiên cứu tổng hợp nền tảng lý thuyết, bằng chứng thực nghiệm và phương pháp từ các công trình có nguồn gốc rõ ràng. Các nguồn nền tảng gồm Altman (1968), Ohlson (1980), Hand và Henley (1997), Thomas (2000), Baesens et al. (2003), Lessmann et al. (2015), Breiman (2001), Chen và Guestrin (2016), Ke et al. (2017), Chawla et al. (2002) và Lundberg và Lee (2017).

Ở bước định lượng, dữ liệu được mô tả và kiểm tra chất lượng trước khi mô hình hóa. Toàn bộ phép biến đổi học từ dữ liệu—như điền khuyết, chuẩn hóa, mã hóa, chọn biến và SMOTE—phải được ước lượng trong tập huấn luyện hoặc trong từng fold của cross-validation. Nguyên tắc này ngăn thông tin của tập kiểm định đi ngược vào quá trình học.

Các mô hình được huấn luyện gồm hồi quy logistic, cây quyết định, Random Forest, XGBoost và LightGBM. Siêu tham số được lựa chọn bằng cross-validation trên tập huấn luyện. Tập kiểm định cuối cùng chỉ được sử dụng để báo cáo hiệu năng ngoài mẫu.

Để trả lời nhóm câu hỏi về đặc trưng, nghiên cứu tiến hành bốn lớp thực nghiệm. Thứ nhất, mô hình được huấn luyện với toàn bộ 178 đặc trưng để tạo mốc hiệu năng. Thứ hai, các đặc trưng được xếp hạng bằng phương pháp chỉ sử dụng dữ liệu huấn luyện. Thứ ba, các tập Top-k lồng nhau, chẳng hạn Top-10, 20, 30, 50, 75, 100, 125, 150 và 178, được đánh giá trên cùng sơ đồ kiểm định. Thứ tư, nghiên cứu thực hiện thí nghiệm theo nhóm nghiệp vụ bằng cách chỉ sử dụng từng nhóm, bổ sung tuần tự từng nhóm và loại bỏ từng nhóm khỏi tập đầy đủ. Số lượng đặc trưng tối ưu là giá trị nhỏ nhất thỏa ngưỡng “gần như toàn bộ hiệu năng”, dự kiến không thấp hơn 99% hiệu năng của mô hình đầy đủ hoặc mức suy giảm tuyệt đối không quá 0,005 ROC-AUC/PR-AUC. Ngưỡng cuối cùng phải được công bố trước khi xem kết quả tập kiểm định.

Nghiên cứu sử dụng ma trận nhầm lẫn để xác định đúng/sai theo từng lớp; ROC-AUC để đo khả năng xếp hạng tổng quát; PR-AUC, Recall và F1-score để nhấn mạnh lớp nợ xấu. Ngưỡng phân loại không mặc định là 0,5 mà được xác định theo mục tiêu nghiệp vụ hoặc chi phí sai lầm. Nếu mô hình được sử dụng như một ước lượng xác suất, cần đánh giá thêm calibration bằng Brier score và biểu đồ hiệu chỉnh.

Cuối cùng, SHAP được dùng để phân tích mức độ đóng góp của biến. Giải thích SHAP được hiểu là giải thích dự báo của mô hình, không được diễn giải tự động thành quan hệ nhân quả.

## 1.8. Đóng góp dự kiến của nghiên cứu

Về học thuật, nghiên cứu cung cấp một đối sánh nhất quán giữa mô hình thống kê và học máy trong bài toán dự báo chuyển nhóm nợ. Giá trị của đối sánh nằm ở việc sử dụng cùng tập dữ liệu, cùng sơ đồ kiểm định và cùng bộ thước đo, qua đó giảm nguy cơ kết luận do khác biệt quy trình.

Về phương pháp, nghiên cứu tích hợp năm thành phần thường bị tách rời: kiểm soát rò rỉ dữ liệu, xử lý mất cân bằng, đánh giá đa chỉ tiêu, giải thích mô hình và phân tích hiệu năng theo số lượng/nhóm đặc trưng. Thiết kế Top-k và loại bỏ nhóm cho phép phân biệt “đặc trưng quan trọng đối với một dự báo” với “nhóm dữ liệu tạo ra giá trị gia tăng ngoài mẫu”.

Về thực tiễn, nghiên cứu đề xuất mô hình cảnh báo khả năng chuyển nhóm nợ và cách lựa chọn ngưỡng theo mục tiêu quản trị, thay vì chỉ dựa trên Accuracy. Kết quả còn cung cấp đường cong hiệu năng–số lượng đặc trưng, tập đặc trưng rút gọn và thứ tự ưu tiên đầu tư dữ liệu. SHAP hỗ trợ nhận diện yếu tố rủi ro ở từng hồ sơ, còn kết quả loại bỏ nhóm hỗ trợ quyết định nên duy trì hoặc mở rộng nguồn dữ liệu nào.

## 1.9. Kết cấu luận văn

Ngoài phần mở đầu, kết luận, tài liệu tham khảo và phụ lục, luận văn dự kiến gồm:

- Chương 1 trình bày bối cảnh, vấn đề, mục tiêu, câu hỏi, phạm vi, phương pháp và đóng góp của nghiên cứu.
- Chương 2 trình bày cơ sở lý thuyết, cơ sở pháp lý, tổng quan phương pháp và khoảng trống nghiên cứu.
- Chương 3 trình bày dữ liệu, thiết kế nghiên cứu, quy trình tiền xử lý, mô hình và tiêu chí đánh giá.
- Chương 4 trình bày kết quả thực nghiệm, so sánh mô hình, giải thích SHAP và thảo luận.
- Chương 5 kết luận, đề xuất hàm ý quản trị, nêu hạn chế và hướng nghiên cứu tiếp theo.

## 1.10. Tóm tắt nội dung Chương 1

Chương 1 xác lập định hướng của luận văn là đề xuất mô hình dự báo khả năng chuyển nhóm nợ để hỗ trợ phân loại nợ và cảnh báo sớm rủi ro tín dụng. Khác với cách tiếp cận chỉ phân loại nợ tốt–nợ xấu tại thời điểm hiện tại, nghiên cứu đặt biến mục tiêu trong một cửa sổ tương lai và xem xét liệu khoản vay có chuyển sang nhóm rủi ro cao hơn hay không.

Trên cơ sở bộ dữ liệu gồm 178 đặc trưng, Chương 1 đặt ra bốn câu hỏi trọng tâm: nhiều đặc trưng hơn có luôn tốt hơn; nhóm đặc trưng nghiệp vụ nào đóng góp lớn nhất; có thể rút gọn còn bao nhiêu đặc trưng mà vẫn giữ gần như toàn bộ hiệu năng; và ngân hàng nên ưu tiên đầu tư nguồn dữ liệu nào. Để trả lời, nghiên cứu kết hợp so sánh thuật toán với thí nghiệm Top-k, bổ sung/loại bỏ nhóm đặc trưng và giải thích SHAP trên một quy trình kiểm định thống nhất.

Đóng góp dự kiến không chỉ là một mô hình dự báo, mà còn là bằng chứng phục vụ quản trị dữ liệu. Kết quả phải chỉ ra mức hiệu năng ngoài mẫu, tập đặc trưng tối thiểu, đóng góp của từng nhóm nghiệp vụ và cách chuyển dự báo thành tín hiệu cảnh báo có thể sử dụng trong quy trình ngân hàng.

# CHƯƠNG 2. CƠ SỞ LÝ THUYẾT VÀ TỔNG QUAN NGHIÊN CỨU

## 2.1. Các khái niệm nền tảng

### 2.1.1. Rủi ro tín dụng

Rủi ro tín dụng là khả năng người vay hoặc đối tác không thực hiện đầy đủ nghĩa vụ theo thỏa thuận, gây tổn thất cho tổ chức cấp tín dụng (Basel Committee on Banking Supervision, 2000). Khái niệm này rộng hơn nợ xấu. Rủi ro tồn tại từ khi ngân hàng phát sinh một khoản phải thu, trong khi nợ xấu là trạng thái đã được nhận diện theo tiêu chí pháp lý hoặc nghiệp vụ.

Tổn thất tín dụng thường được biểu diễn qua ba thành phần:

\[
EL = PD \times LGD \times EAD
\]

trong đó \(EL\) là tổn thất kỳ vọng; \(PD\) là xác suất vỡ nợ; \(LGD\) là tỷ lệ tổn thất khi vỡ nợ; và \(EAD\) là dư nợ tại thời điểm vỡ nợ. Luận văn tập trung chủ yếu vào thành phần \(PD\), tức khả năng khách hàng chuyển sang trạng thái nợ xấu trong một khoảng dự báo xác định. Kết quả phân loại không trực tiếp tương đương với tổn thất tiền tệ nếu chưa kết hợp LGD và EAD.

### 2.1.2. Nợ xấu và phân loại nợ

Theo khuôn khổ pháp lý Việt Nam áp dụng cho ngân hàng thương mại, tổ chức tín dụng phi ngân hàng và chi nhánh ngân hàng nước ngoài, nợ được phân loại thành năm nhóm theo mức độ rủi ro: nợ đủ tiêu chuẩn, nợ cần chú ý, nợ dưới tiêu chuẩn, nợ nghi ngờ và nợ có khả năng mất vốn. Nợ xấu gồm nợ thuộc nhóm 3, nhóm 4 và nhóm 5.

Việc phân loại không chỉ dựa trên số ngày quá hạn mà còn phải tuân theo các tiêu chí định lượng, định tính, kết quả phân loại của CIC và quy tắc điều chỉnh theo văn bản hiện hành. Do vậy, khi xây dựng dữ liệu mô hình, nghiên cứu cần lưu cả nguồn tạo nhãn, ngày chốt nhãn và quy tắc ánh xạ. Một nhãn không có dấu thời gian rõ ràng có thể gây rò rỉ dữ liệu và làm mất khả năng tái lập.

Thông tư số 31/2024/TT-NHNN có hiệu lực từ ngày 01/07/2024 nhưng đã được sửa đổi bởi Thông tư số 37/2025/TT-NHNN, có hiệu lực từ ngày 15/12/2025. Vì thế, nghiên cứu sử dụng Văn bản hợp nhất số 27/VBHN-NHNN ngày 21/11/2025 làm căn cứ trình bày hiện hành; Thông tư số 11/2021/TT-NHNN chỉ phù hợp khi mô tả giai đoạn lịch sử trước khi bị thay thế.

### 2.1.3. Chấm điểm tín dụng và chấm điểm hành vi

Chấm điểm tín dụng là quá trình sử dụng đặc điểm của người đề nghị vay hoặc khoản vay để ước lượng mức độ rủi ro, thường tại thời điểm cấp tín dụng. Chấm điểm hành vi sử dụng dữ liệu phát sinh sau khi quan hệ tín dụng đã tồn tại, như lịch sử thanh toán, sử dụng hạn mức, biến động dư nợ hoặc tần suất quá hạn, nhằm cập nhật đánh giá rủi ro (Thomas, 2000).

Hai bài toán có thể dùng thuật toán tương tự nhưng khác nhau về thời điểm dự báo và tập biến. Nếu luận văn sử dụng dữ liệu hành vi, tên gọi và kết luận phải phản ánh đúng bản chất; không nên gọi là mô hình phê duyệt hồ sơ mới nếu các biến chỉ xuất hiện sau giải ngân.

### 2.1.4. Hệ thống cảnh báo sớm

Hệ thống cảnh báo sớm nhằm phát hiện dấu hiệu suy giảm chất lượng tín dụng trước khi tổn thất trở nên rõ ràng. Một hệ thống hoàn chỉnh gồm dữ liệu, mô hình, ngưỡng cảnh báo, quy trình xử lý, cơ chế phản hồi và giám sát. Mô hình dự báo chỉ là một cấu phần. Nếu cảnh báo không gắn với hành động nghiệp vụ, hiệu năng thống kê cao chưa chắc chuyển thành hiệu quả quản trị.

## 2.2. Cơ sở pháp lý và quản trị dữ liệu tại Việt Nam

### 2.2.1. Phân loại tài sản có

Thông tư số 31/2024/TT-NHNN quy định việc phân loại tài sản có trong hoạt động của ngân hàng thương mại, tổ chức tín dụng phi ngân hàng và chi nhánh ngân hàng nước ngoài. Thông tư số 37/2025/TT-NHNN sửa đổi một số nội dung, và hai văn bản được hợp nhất tại Văn bản số 27/VBHN-NHNN. Đối với luận văn, các văn bản này có ba ý nghĩa:

1. Cung cấp định nghĩa vận hành cho biến mục tiêu;
2. Xác định các trạng thái nợ cần phân biệt;
3. Tạo căn cứ để đánh giá sự phù hợp của mô hình với quy trình quản trị.

Nghiên cứu cần ghi rõ giai đoạn dữ liệu chịu điều chỉnh bởi văn bản nào. Nếu dữ liệu trải qua nhiều chế độ pháp lý, thay đổi quy tắc phân loại có thể làm phát sinh “concept drift” do nhãn thay đổi ngay cả khi hành vi kinh tế không đổi.

### 2.2.2. Hoạt động thông tin tín dụng

Thông tư số 15/2023/TT-NHNN quy định hoạt động thông tin tín dụng của Ngân hàng Nhà nước do CIC làm đầu mối tổ chức, thực hiện. Dữ liệu CIC có thể bổ sung lịch sử quan hệ tín dụng ngoài phạm vi một ngân hàng, nhưng việc sử dụng phải đúng mục đích, đúng thẩm quyền và tuân thủ yêu cầu bảo mật.

Về phương pháp, dữ liệu thu thập từ nhiều hệ thống cần được kiểm tra khóa định danh, thời điểm cập nhật, bản ghi trùng, độ trễ và xung đột giá trị. Không nên mặc định dữ liệu được hợp nhất là hoàn toàn chính xác. Hồ sơ xử lý dữ liệu cần ghi nhận nguồn, quy tắc biến đổi và phiên bản.

## 2.3. Nền tảng lý thuyết của mô hình dự báo

### 2.3.1. Phân tích biệt thức và Z-score

Altman (1968) sử dụng phân tích biệt thức đa biến trên 66 doanh nghiệp sản xuất được ghép thành 33 cặp phá sản–không phá sản. Từ 22 tỷ số ứng viên, tác giả giữ lại 5 tỷ số trong Z-score và báo cáo tỷ lệ phân loại đúng khoảng 95% ở thời điểm một năm trước phá sản trên mẫu ban đầu. Công trình có ý nghĩa nền tảng vì chuyển đánh giá tài chính từ phân tích từng tỷ số riêng lẻ sang một chỉ số tổng hợp có căn cứ thống kê. Tuy nhiên, thiết kế mẫu cân bằng 50/50 khác xa tỷ lệ vỡ nợ tự nhiên trong danh mục tín dụng; do đó Accuracy của nghiên cứu gốc không nên được dùng làm chuẩn hiệu năng trực tiếp cho luận văn.

Tuy nhiên, Z-score được xây dựng cho một bối cảnh, mẫu và thời kỳ cụ thể. Khi áp dụng sang quốc gia, ngành, quy mô doanh nghiệp hoặc giai đoạn khác, hệ số và điểm cắt có thể không còn phù hợp. Do đó, Z-score nên được xem là nền tảng lịch sử và mô hình tham chiếu, không phải bằng chứng rằng một công thức cố định có giá trị phổ quát.

### 2.3.2. Hồi quy logistic

Ohlson (1980) áp dụng mô hình logit cho dự báo phá sản. Với bài toán nhị phân, xác suất khách hàng thuộc lớp nợ xấu có thể được biểu diễn:

\[
P(Y=1|X)=\frac{1}{1+\exp[-(\beta_0+\beta_1X_1+\cdots+\beta_pX_p)]}
\]

trong đó \(Y=1\) biểu thị nợ xấu; \(X_j\) là biến giải thích; và \(\beta_j\) là hệ số cần ước lượng. Log-odds là hàm tuyến tính của các biến:

\[
\log\left(\frac{P(Y=1|X)}{1-P(Y=1|X)}\right)
=\beta_0+\sum_{j=1}^{p}\beta_jX_j
\]

Ưu điểm của logistic là đơn giản, tốc độ huấn luyện nhanh, đầu ra xác suất và khả năng giải thích hệ số. Mô hình phù hợp làm baseline. Hạn chế chính là giả định tuyến tính trên log-odds, nhạy với đa cộng tuyến và cần thiết kế thủ công nếu quan hệ có tương tác hoặc phi tuyến (Hosmer, Lemeshow, & Sturdivant, 2013).

### 2.3.3. Cây quyết định

Cây quyết định phân chia không gian đặc trưng thành các vùng dựa trên quy tắc tại từng nút (Quinlan, 1986). Mô hình dễ trực quan hóa và có thể nắm bắt quan hệ phi tuyến cũng như tương tác biến. Tuy nhiên, một cây đơn lẻ thường không ổn định: thay đổi nhỏ trong dữ liệu có thể làm thay đổi cấu trúc cây; cây sâu cũng dễ quá khớp.

### 2.3.4. Random Forest

Random Forest huấn luyện nhiều cây trên các mẫu bootstrap và chỉ xem xét một tập con ngẫu nhiên của biến tại mỗi lần phân chia (Breiman, 2001). Dự báo cuối cùng được tổng hợp từ các cây. Việc giảm tương quan giữa cây giúp giảm phương sai so với cây đơn lẻ.

Random Forest có khả năng xử lý phi tuyến và tương tác mà không cần chỉ định trước. Đổi lại, mô hình khó trình bày bằng một bộ quy tắc ngắn và xác suất đầu ra có thể cần hiệu chỉnh nếu dùng như PD.

### 2.3.5. Gradient Boosting, XGBoost và LightGBM

Gradient Boosting xây dựng các mô hình yếu theo trình tự, mỗi mô hình mới tập trung giảm phần sai số còn lại (Friedman, 2001). XGBoost phát triển khung boosting có chính quy hóa, xử lý thưa và tối ưu tính toán; hệ thống trong bài gốc được thiết kế để mở rộng vượt quy mô hàng tỷ quan sát (Chen & Guestrin, 2016). LightGBM sử dụng Gradient-based One-Side Sampling và Exclusive Feature Bundling để giảm chi phí tính toán trên dữ liệu lớn, nhiều chiều. Trên các bộ dữ liệu công khai được Ke et al. (2017) sử dụng, LightGBM tăng tốc quá trình huấn luyện so với GBDT thông thường tới trên 20 lần trong khi duy trì độ chính xác gần tương đương. Con số này chứng minh hiệu quả kiến trúc trong điều kiện benchmark của tác giả, không phải một hệ số tốc độ cố định cho mọi bộ dữ liệu tín dụng.

Các thuật toán boosting thường đạt hiệu năng tốt trên dữ liệu bảng, nhưng kết quả phụ thuộc vào siêu tham số và quy trình kiểm định. Cây quá sâu, learning rate không phù hợp hoặc tối ưu quá nhiều lần trên một tập kiểm định có thể dẫn đến quá khớp. Bởi vậy, tên thuật toán không phải là bằng chứng đủ cho chất lượng mô hình.

## 2.4. Dữ liệu và thiết kế biến

### 2.4.1. Nhóm biến đầu vào

Dựa trên lý thuyết và tổng hợp tài liệu, biến đầu vào có thể được chia thành:

- **Đặc điểm khách hàng:** tuổi, nghề nghiệp, thời gian công tác, thu nhập và tình trạng cư trú, nếu việc sử dụng được pháp luật và chính sách nội bộ cho phép;
- **Đặc điểm khoản vay:** số tiền, kỳ hạn, lãi suất, mục đích vay, loại tài sản bảo đảm và tỷ lệ khoản vay trên giá trị tài sản;
- **Năng lực tài chính:** thu nhập, dòng tiền, nghĩa vụ nợ, tỷ lệ nợ trên thu nhập; với doanh nghiệp có thể gồm thanh khoản, đòn bẩy, sinh lời và dòng tiền;
- **Lịch sử tín dụng:** số khoản vay, dư nợ, lịch sử quá hạn, số lần cơ cấu, truy vấn tín dụng;
- **Hành vi sau giải ngân:** mức sử dụng hạn mức, biến động số dư, tần suất và mức độ chậm trả;
- **Biến bối cảnh:** thời gian, ngành, vùng hoặc biến kinh tế vĩ mô nếu dữ liệu và thiết kế nghiên cứu cho phép.

Việc đưa biến vào mô hình cần dựa trên tính sẵn có tại thời điểm dự báo. Một biến có tương quan rất cao với nợ xấu nhưng chỉ xuất hiện sau khi nợ xấu xảy ra không có giá trị dự báo hợp lệ.

### 2.4.2. Số lượng đặc trưng và giá trị thông tin

Việc tăng số lượng đặc trưng có thể giúp mô hình tiếp cận thêm tín hiệu, nhưng không bảo đảm hiệu năng ngoài mẫu luôn tăng. Khi số đặc trưng lớn, mô hình có thể gặp biến nhiễu, đa cộng tuyến, dữ liệu thiếu, đặc trưng trùng lặp và nguy cơ học những quan hệ không ổn định. Chi phí vận hành cũng tăng vì mỗi đặc trưng cần có nguồn dữ liệu, định nghĩa, kiểm soát chất lượng và cơ chế cập nhật.

Với 178 đặc trưng ban đầu, nghiên cứu không lựa chọn biến chỉ từ một bảng xếp hạng importance duy nhất. SHAP hoặc gain importance cho biết mô hình đang sử dụng biến nào, nhưng chưa đủ để chứng minh biến đó tạo ra giá trị tăng thêm ngoài mẫu. Hai biến tương quan có thể thay thế nhau; khi một biến bị loại, biến còn lại có thể đảm nhận tín hiệu tương tự. Vì vậy, luận văn kết hợp:

1. **Xếp hạng đặc trưng:** dùng SHAP, permutation importance hoặc phương pháp phù hợp trong từng fold huấn luyện;
2. **Đường cong Top-k:** huấn luyện lại mô hình với các tập đặc trưng lồng nhau để quan sát quan hệ giữa số biến và hiệu năng;
3. **Đánh giá từng nhóm:** xây mô hình chỉ với một nhóm đặc trưng để đo năng lực độc lập;
4. **Bổ sung theo nhóm:** thêm từng nhóm vào mô hình cơ sở để đo giá trị gia tăng;
5. **Loại bỏ theo nhóm:** loại một nhóm khỏi tập đầy đủ để đo mức suy giảm hiệu năng;
6. **Kiểm tra ổn định:** so sánh thứ hạng biến và kết quả Top-k giữa các fold hoặc các giai đoạn thời gian.

Một nhóm đặc trưng được xem là có đóng góp lớn khi việc bổ sung nhóm đó làm tăng hiệu năng ngoài mẫu một cách ổn định và việc loại bỏ làm giảm hiệu năng đáng kể. Quy mô đặc trưng tối ưu không được xác định bằng cảm tính. Nghiên cứu chọn giá trị \(k^*\) nhỏ nhất thỏa:

\[
M(k^*) \geq \tau \times M(178)
\]

trong đó \(M(k)\) là thước đo hiệu năng ngoài mẫu của mô hình dùng \(k\) đặc trưng; \(M(178)\) là hiệu năng với tập đầy đủ; và \(\tau\) là tỷ lệ duy trì hiệu năng, dự kiến bằng 0,99. Có thể sử dụng tiêu chí bổ sung là chênh lệch tuyệt đối không vượt 0,005 đối với ROC-AUC hoặc PR-AUC. Kết luận cuối cùng cần kèm khoảng tin cậy hoặc độ phân tán qua các fold; không nên khẳng định tập rút gọn tương đương nếu chênh lệch nằm trong nhiễu đánh giá.

### 2.4.3. Chất lượng dữ liệu

Giá trị thiếu có thể phản ánh lỗi nhập liệu, khác biệt quy trình hoặc đặc điểm của khách hàng. Vì vậy, trước khi điền khuyết cần phân tích cơ chế thiếu và tỷ lệ thiếu theo thời gian, phân khúc và nhãn. Ngoại lệ cũng không nên bị loại tự động, vì một giá trị cực đoan có thể chính là tín hiệu rủi ro.

Các bước tiền xử lý phải đặt trong pipeline. Thống kê dùng để điền khuyết, chuẩn hóa hoặc chọn biến chỉ được học từ dữ liệu huấn luyện. Nếu tính trên toàn bộ dữ liệu trước khi chia tập, kết quả kiểm định sẽ bị lạc quan.

### 2.4.4. Mất cân bằng lớp

Trong dữ liệu tín dụng, lớp nợ xấu thường chiếm tỷ lệ nhỏ. Accuracy vì vậy có thể gây hiểu lầm. Ví dụ, nếu nợ xấu chiếm 5%, mô hình dự đoán tất cả là nợ tốt vẫn đạt Accuracy 95% nhưng Recall của nợ xấu bằng 0.

SMOTE tạo quan sát tổng hợp của lớp thiểu số dựa trên các điểm lân cận (Chawla et al., 2002). Brown và Mues (2012) lượng hóa ảnh hưởng mất cân bằng bằng 5 bộ dữ liệu tín dụng thực và nhiều cấu hình tỷ lệ lớp, từ 70/30 tới 99/1. AUC được dùng làm tiêu chí chính; khác biệt thứ hạng được kiểm định bằng thống kê Friedman và hậu kiểm Nemenyi. Gradient Boosting đứng đầu về thứ hạng trung bình ở 2 trong 5 cấu hình tỷ lệ lớp, còn Random Forest đứng đầu ở cấu hình 10% khách hàng xấu và cấu hình cực đoan chỉ 1% khách hàng xấu. Kết quả cho thấy thuật toán ensemble chống chịu mất cân bằng tương đối tốt, nhưng không chứng minh SMOTE luôn làm tăng hiệu năng. SMOTE có thể làm chồng lấn lớp hoặc khuếch đại nhiễu. Vì vậy, nghiên cứu so sánh ít nhất ba cấu hình: dữ liệu gốc, trọng số lớp và SMOTE. SMOTE chỉ được thực hiện trong tập huấn luyện/từng fold, tuyệt đối không thực hiện trước khi chia tập.

## 2.5. Đánh giá mô hình

### 2.5.1. Ma trận nhầm lẫn

Với nợ xấu là lớp dương:

- True Positive (TP): dự báo đúng khách hàng nợ xấu;
- False Positive (FP): dự báo nợ xấu nhưng thực tế tốt;
- True Negative (TN): dự báo đúng khách hàng tốt;
- False Negative (FN): dự báo tốt nhưng thực tế nợ xấu.

Trong quản trị rủi ro, FN thường gây tổn thất trực tiếp hơn FP, nhưng FP cũng tạo chi phí cơ hội. Do đó, ngưỡng quyết định cần phản ánh sự đánh đổi này.

### 2.5.2. Các thước đo

\[
Precision=\frac{TP}{TP+FP}
\]

\[
Recall=\frac{TP}{TP+FN}
\]

\[
F1=2\times\frac{Precision\times Recall}{Precision+Recall}
\]

Recall đo tỷ lệ nợ xấu được nhận diện; Precision đo tỷ lệ đúng trong số cảnh báo; F1 cân bằng hai đại lượng. ROC-AUC đo khả năng xếp hạng một quan sát xấu cao hơn một quan sát tốt qua các ngưỡng (Fawcett, 2006). Khi lớp dương hiếm, PR-AUC cung cấp góc nhìn trực tiếp hơn về Precision–Recall.

Không nên chọn mô hình chỉ bằng một chỉ tiêu. Luận văn cần báo cáo ít nhất ROC-AUC, PR-AUC, Recall, Precision, F1 và ma trận nhầm lẫn tại ngưỡng được chọn.

### 2.5.3. Khả năng hiệu chỉnh xác suất

Discrimination và calibration là hai thuộc tính khác nhau. Mô hình có AUC cao có thể xếp hạng tốt nhưng xác suất dự báo không khớp với tần suất vỡ nợ thực tế. Nếu đầu ra được sử dụng như PD, cần kiểm tra biểu đồ calibration, Brier score và cân nhắc hiệu chỉnh xác suất trên tập xác thực riêng.

### 2.5.4. Thiết kế kiểm định

K-fold cross-validation phù hợp khi dữ liệu không có cấu trúc thời gian rõ rệt. Với dữ liệu nhiều kỳ, kiểm định theo thời gian phù hợp hơn với mục tiêu dự báo tương lai. Nếu cùng một khách hàng xuất hiện nhiều dòng, cần chia theo nhóm khách hàng để tránh hồ sơ của cùng người xuất hiện ở cả tập huấn luyện và kiểm định.

### 2.5.5. Đánh giá hiệu quả rút gọn đặc trưng

Đường cong hiệu năng–số lượng đặc trưng là công cụ chính để trả lời liệu nhiều đặc trưng hơn có luôn tốt hơn. Trục hoành thể hiện số đặc trưng \(k\); trục tung thể hiện ROC-AUC, PR-AUC hoặc Recall tại một mức Precision/chi phí cảnh báo xác định. Điểm “gối” của đường cong cho biết vùng mà việc thêm biến chỉ tạo cải thiện rất nhỏ.

Ngoài chênh lệch thước đo, nghiên cứu báo cáo mức giảm số biến, thời gian huấn luyện, thời gian suy luận, tỷ lệ hồ sơ có đủ dữ liệu và độ ổn định theo thời gian. Nhờ đó, quyết định rút gọn không chỉ dựa trên hiệu năng thống kê mà còn phản ánh tính khả thi vận hành.

## 2.6. Khả năng giải thích và SHAP

Lundberg và Lee (2017) đề xuất SHAP như một khung thống nhất để gán đóng góp của từng biến vào chênh lệch giữa dự báo của một quan sát và giá trị nền. SHAP có thể hỗ trợ:

1. Giải thích toàn cục thông qua phân bố độ lớn giá trị SHAP;
2. Nhận diện chiều tác động mà mô hình đã học;
3. Giải thích cục bộ cho từng hồ sơ;
4. Phát hiện dấu hiệu bất hợp lý, rò rỉ dữ liệu hoặc phụ thuộc quá mức vào một biến.
5. Tổng hợp giá trị tuyệt đối theo nhóm nghiệp vụ để mô tả nhóm tín hiệu mà mô hình sử dụng.

Tuy nhiên, SHAP giải thích hành vi của mô hình chứ không chứng minh quan hệ nhân quả. Tổng SHAP lớn cũng không tự động chứng minh rằng ngân hàng nên đầu tư thêm vào nhóm dữ liệu đó. Giá trị đầu tư phải được xác nhận bằng thí nghiệm bổ sung/loại bỏ nhóm trên dữ liệu ngoài mẫu và cân đối với chi phí thu thập. Giá trị SHAP của các biến tương quan cũng có thể được phân bổ theo cách khó diễn giải. Vì vậy, kết quả cần được đối chiếu với nghiệp vụ, thống kê mô tả và kiểm tra độ ổn định.

## 2.7. Tổng quan bằng chứng thực nghiệm

Các nghiên cứu ban đầu đặt nền móng cho mô hình định lượng. Altman (1968) xây dựng Z-score từ mẫu 66 doanh nghiệp, với 33 doanh nghiệp phá sản và 33 doanh nghiệp đối chứng, và đạt khoảng 95% phân loại đúng trên mẫu tại thời điểm một năm trước phá sản. Ohlson (1980) sử dụng mẫu lớn và tự nhiên hơn gồm 105 doanh nghiệp phá sản cùng 2.058 doanh nghiệp không phá sản để ước lượng xác suất bằng logit. Hai nghiên cứu minh họa tiến trình từ hàm điểm phân biệt sang mô hình xác suất, đồng thời cho thấy kết quả phụ thuộc mạnh vào thiết kế mẫu. Hand và Henley (1997) tổng quan các phương pháp phân loại thống kê trong chấm điểm tín dụng, còn Thomas (2000) mở rộng thảo luận sang chấm điểm hành vi.

Baesens et al. (2003) so sánh nhiều thuật toán chấm điểm tín dụng, cho thấy giá trị của đánh giá đối sánh có hệ thống. Lessmann et al. (2015) mở rộng đối sánh lên 41 bộ phân loại, 8 bộ dữ liệu tín dụng bán lẻ thực và 6 thước đo hiệu năng. Các bộ dữ liệu bao gồm Australian Credit, German Credit và dữ liệu từ các tổ chức tài chính tại Benelux và Vương quốc Anh. Kết quả cho thấy một số mô hình có độ chính xác cao hơn đáng kể so với chuẩn ngành là hồi quy logistic, đặc biệt là ensemble không đồng nhất. Tuy nhiên, việc tác giả dùng đồng thời 6 thước đo cũng cho thấy một mô hình có thể thay đổi thứ hạng khi tiêu chí đánh giá thay đổi.

Brown và Mues (2012) tập trung vào 5 bộ dữ liệu chấm điểm thực và chủ động thay đổi tỷ lệ lớp tới 99% khách hàng tốt–1% khách hàng xấu. Kết quả AUC cho thấy Random Forest và Gradient Boosting chống chịu tương đối tốt khi lớp xấu ngày càng hiếm, trong khi C4.5, phân tích biệt thức bậc hai và k-láng giềng suy giảm mạnh hơn. Xia et al. (2017) kết hợp boosted decision tree với tối ưu siêu tham số Bayes, minh họa vai trò của tuning có kiểm soát.

Barboza, Kimura và Altman (2017) dùng dữ liệu doanh nghiệp Bắc Mỹ giai đoạn 1985–2013, với hơn 10.000 quan sát doanh nghiệp–năm trong tập kiểm định, để dự báo phá sản trước một năm. Nghiên cứu so sánh SVM, bagging, boosting và Random Forest với phân tích biệt thức, hồi quy logistic và mạng nơ-ron; bagging, boosting và Random Forest là nhóm cho kết quả tốt hơn trong thiết kế của tác giả. Quy mô và khoảng thời gian dài làm tăng giá trị bằng chứng, nhưng đối tượng là doanh nghiệp niêm yết Bắc Mỹ nên không thể suy rộng trực tiếp cho tín dụng cá nhân Việt Nam.

Dastile et al. (2020) tổng hợp có hệ thống 74 nghiên cứu chính về credit scoring. Tổng quan ghi nhận hồi quy logistic vẫn được sử dụng phổ biến nhờ tính đơn giản và minh bạch, trong khi các mô hình học máy phức tạp được nghiên cứu để cải thiện hiệu năng. Con số 74 cho thấy cơ sở bằng chứng tương đối rộng, song tổng quan cũng hàm ý rằng khác biệt về bộ dữ liệu, xử lý dữ liệu và thước đo khiến các kết quả riêng lẻ khó so sánh trực tiếp.

### 2.7.1. Bảng tổng hợp các dẫn chứng định lượng

| Nghiên cứu | Quy mô/phạm vi | Thiết kế hoặc thước đo | Kết quả có thể sử dụng làm dẫn chứng | Giới hạn khi vận dụng |
|---|---:|---|---|---|
| Altman (1968) | 66 doanh nghiệp: 33 phá sản, 33 không phá sản; 22 tỷ số ứng viên, giữ 5 | Phân tích biệt thức; dự báo trước 1 năm | Khoảng 95% phân loại đúng trên mẫu ban đầu | Mẫu nhỏ, cân bằng và chỉ gồm doanh nghiệp sản xuất Hoa Kỳ |
| Ohlson (1980) | 105 doanh nghiệp phá sản và 2.058 doanh nghiệp không phá sản | Mô hình logit | Chứng minh khả năng ước lượng xác suất phá sản trên mẫu không ghép cặp lớn hơn | Dự báo phá sản doanh nghiệp, không phải nợ xấu bán lẻ |
| Brown & Mues (2012) | 5 bộ dữ liệu tín dụng thực; tỷ lệ lớp từ 70/30 tới 99/1 | AUC; kiểm định Friedman và Nemenyi | Random Forest đứng đầu thứ hạng trung bình ở cấu hình 99/1; ensemble chịu mất cân bằng tốt hơn tương đối | Kết quả phụ thuộc dữ liệu và cách giảm mẫu lớp xấu |
| Lessmann et al. (2015) | 41 bộ phân loại, 8 bộ dữ liệu tín dụng thực | 6 thước đo hiệu năng | Nhiều mô hình vượt logistic; ensemble không đồng nhất nổi bật | Không có mô hình tốt nhất tuyệt đối trên mọi dữ liệu/thước đo |
| Barboza et al. (2017) | Dữ liệu 1985–2013; hơn 10.000 quan sát doanh nghiệp–năm ở tập kiểm định | Dự báo phá sản trước 1 năm | Bagging, boosting và Random Forest tốt hơn nhóm phương pháp truyền thống trong thiết kế nghiên cứu | Doanh nghiệp Bắc Mỹ; khác bối cảnh tín dụng Việt Nam |
| Dastile et al. (2020) | 74 nghiên cứu chính | Tổng quan tài liệu có hệ thống | Logistic phổ biến vì minh bạch; học máy được dùng để cải thiện hiệu năng | Nghiên cứu nguồn không đồng nhất |
| Ke et al. (2017) | Nhiều bộ dữ liệu công khai | Benchmark hiệu quả tính toán | LightGBM nhanh hơn GBDT truyền thống tới trên 20 lần, độ chính xác gần tương đương | Benchmark thuật toán, không phải kết quả riêng của credit scoring |

Nhìn chung, tài liệu ủng hộ việc thử nghiệm ensemble nhưng không ủng hộ kết luận rằng ensemble luôn vượt trội. Hiệu năng được báo cáo ở các nghiên cứu không thể so sánh trực tiếp nếu khác bộ dữ liệu, tỷ lệ nợ xấu, cửa sổ dự báo, cách chia tập và thước đo.

## 2.8. Khoảng trống nghiên cứu

Từ tổng quan có thể xác định bảy khoảng trống mà luận văn hướng tới.

Thứ nhất, nhiều nghiên cứu nhấn mạnh hiệu năng nhưng chưa trình bày đồng thời calibration, chi phí sai lầm và lựa chọn ngưỡng. Điều này hạn chế khả năng chuyển kết quả kỹ thuật thành quyết định nghiệp vụ.

Thứ hai, so sánh mô hình có thể thiếu công bằng khi mỗi thuật toán dùng cách tiền xử lý hoặc tập dữ liệu khác nhau. Luận văn khắc phục bằng pipeline và sơ đồ kiểm định thống nhất.

Thứ ba, xử lý mất cân bằng đôi khi được thực hiện trước khi chia dữ liệu, gây rò rỉ. Nghiên cứu đặt resampling bên trong từng fold huấn luyện và đánh giá trên phân phối gốc.

Thứ tư, mô hình có hiệu năng cao thường khó giải thích. Luận văn kết hợp mô hình ensemble với SHAP nhưng giữ hồi quy logistic làm chuẩn minh bạch.

Thứ năm, bằng chứng trên dữ liệu quốc tế không thể tự động khái quát cho Việt Nam do khác biệt về hành vi khách hàng, chính sách tín dụng, dữ liệu CIC và khuôn khổ phân loại nợ. Vì vậy, nghiên cứu thực nghiệm trong bối cảnh Việt Nam có giá trị bổ sung.

Thứ sáu, nhiều nghiên cứu tập trung tìm thuật toán tốt nhất nhưng ít lượng hóa quan hệ giữa số lượng đặc trưng và hiệu năng. Vì vậy, chưa có câu trả lời thực nghiệm cho việc tập đầy đủ 178 đặc trưng có thực sự cần thiết hay một tập nhỏ hơn đã đủ.

Thứ bảy, feature importance thường được báo cáo ở cấp biến riêng lẻ mà chưa chuyển thành bằng chứng phục vụ chiến lược dữ liệu. Luận văn khắc phục bằng cách kết hợp SHAP theo nhóm với thí nghiệm chỉ dùng nhóm, bổ sung nhóm và loại bỏ nhóm, từ đó xác định nhóm thông tin có giá trị gia tăng thực sự.

## 2.9. Khung nghiên cứu đề xuất

Khung nghiên cứu gồm chuỗi bước:

\[
\text{Nguồn dữ liệu}
\rightarrow \text{Chốt thời điểm quan sát và nhãn}
\rightarrow \text{Kiểm tra chất lượng}
\rightarrow \text{Chia dữ liệu}
\rightarrow \text{Tiền xử lý trong pipeline}
\rightarrow \text{Mô hình với 178 đặc trưng}
\rightarrow \text{Xếp hạng và tạo các tập Top-k}
\rightarrow \text{Thí nghiệm theo nhóm nghiệp vụ}
\rightarrow \text{Đánh giá ngoài mẫu}
\rightarrow \text{SHAP}
\rightarrow \text{Tập đặc trưng tối ưu và ưu tiên đầu tư dữ liệu}
\]

Biến phụ thuộc là trạng thái chuyển nhóm nợ trong cửa sổ dự báo. Biến độc lập gồm 178 đặc trưng ban đầu, được phân thành các nhóm như đặc điểm khách hàng, khoản vay, năng lực tài chính, tài sản bảo đảm, lịch sử tín dụng, quan hệ tín dụng và hành vi, tùy theo cấu trúc dữ liệu thực tế tại ngày quan sát.

Hồi quy logistic là baseline. Decision Tree giúp tạo chuẩn diễn giải bằng quy tắc. Random Forest đại diện cho bagging; XGBoost và LightGBM đại diện cho boosting. Mỗi mô hình được thử trên dữ liệu gốc và cấu hình xử lý mất cân bằng phù hợp. Sau khi chọn họ mô hình phù hợp, nghiên cứu giữ cố định quy trình tuning để so sánh các tập Top-k và nhóm nghiệp vụ. Mô hình cuối cùng được lựa chọn dựa trên hiệu năng ngoài mẫu, độ ổn định, calibration, khả năng giải thích, số lượng đặc trưng và chi phí triển khai.

## 2.10. Kết luận chương

Chương 2 đã trình bày nền tảng về rủi ro tín dụng, phân loại và chuyển nhóm nợ, chấm điểm tín dụng và cảnh báo sớm; làm rõ căn cứ pháp lý hiện hành; phân tích các mô hình từ logistic đến ensemble; và thảo luận các vấn đề về dữ liệu mất cân bằng, đánh giá và giải thích.

Trọng tâm được bổ sung của Chương 2 là cơ sở phương pháp cho bốn câu hỏi về đặc trưng. Nhiều đặc trưng hơn không mặc nhiên tạo hiệu năng tốt hơn; đóng góp của một nhóm dữ liệu phải được xác nhận bằng kết quả ngoài mẫu; số lượng đặc trưng rút gọn phải được xác định theo một ngưỡng duy trì hiệu năng công bố trước; và khuyến nghị đầu tư dữ liệu phải cân đối giữa giá trị gia tăng dự báo với chi phí thu thập, chất lượng và khả năng vận hành.

Từ đó, Chương 2 đề xuất khung thực nghiệm gồm mô hình đầy đủ 178 đặc trưng, các tập Top-k lồng nhau, thí nghiệm chỉ dùng/bổ sung/loại bỏ nhóm và giải thích SHAP. Khung này là cơ sở để Chương 3 quy định cụ thể cửa sổ chuyển nhóm, cách phân nhóm 178 đặc trưng, pipeline chống rò rỉ và tiêu chí lựa chọn tập đặc trưng tối ưu.

# TÀI LIỆU THAM KHẢO SỬ DỤNG TRONG CHƯƠNG 1–2

1. Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance, 23*(4), 589–609.
2. Baesens, B., Van Gestel, T., Viaene, S., Stepanova, M., Suykens, J., & Vanthienen, J. (2003). Benchmarking state-of-the-art classification algorithms for credit scoring. *Journal of the Operational Research Society, 54*(6), 627–635.
3. Basel Committee on Banking Supervision. (2000). *Principles for the management of credit risk*. Bank for International Settlements.
4. Basel Committee on Banking Supervision. (2024). *Basel Framework*. Bank for International Settlements.
5. Barboza, F., Kimura, H., & Altman, E. I. (2017). Machine learning models and bankruptcy prediction. *Expert Systems with Applications, 83*, 405–417. https://doi.org/10.1016/j.eswa.2017.04.006
6. Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5–32.
7. Brown, I., & Mues, C. (2012). An experimental comparison of classification algorithms for imbalanced credit scoring data sets. *Expert Systems with Applications, 39*(3), 3446–3453. https://doi.org/10.1016/j.eswa.2011.09.033
8. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357.
9. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794).
10. Dastile, X., Celik, T., & Potsane, M. (2020). Statistical and machine learning models in credit scoring: A systematic literature survey. *Applied Soft Computing, 91*, 106263. https://doi.org/10.1016/j.asoc.2020.106263
11. Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874.
12. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232.
13. Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: A review. *Journal of the Royal Statistical Society: Series A, 160*(3), 523–541.
14. Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied logistic regression* (3rd ed.). Wiley.
15. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems*.
16. Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research, 247*(1), 124–136.
17. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems*.
18. Ngân hàng Nhà nước Việt Nam. (2023). *Thông tư số 15/2023/TT-NHNN quy định về hoạt động thông tin tín dụng của Ngân hàng Nhà nước Việt Nam*.
19. Ngân hàng Nhà nước Việt Nam. (2024). *Thông tư số 31/2024/TT-NHNN quy định về phân loại tài sản có trong hoạt động của ngân hàng thương mại, tổ chức tín dụng phi ngân hàng, chi nhánh ngân hàng nước ngoài*.
20. Ngân hàng Nhà nước Việt Nam. (2025). *Thông tư số 37/2025/TT-NHNN sửa đổi, bổ sung một số điều của Thông tư số 31/2024/TT-NHNN*.
21. Ngân hàng Nhà nước Việt Nam. (2025). *Văn bản hợp nhất số 27/VBHN-NHNN ngày 21/11/2025*.
22. Ohlson, J. A. (1980). Financial ratios and the probabilistic prediction of bankruptcy. *Journal of Accounting Research, 18*(1), 109–131. https://doi.org/10.2307/2490395
23. Quinlan, J. R. (1986). Induction of decision trees. *Machine Learning, 1*(1), 81–106.
24. Thomas, L. C. (2000). A survey of credit and behavioural scoring: Forecasting financial risk of lending to consumers. *International Journal of Forecasting, 16*(2), 149–172.
25. Xia, Y., Liu, C., Da, B., & Xie, F. (2017). A boosted decision tree approach using Bayesian hyper-parameter optimization for credit scoring. *Expert Systems with Applications, 78*, 225–241. https://doi.org/10.1016/j.eswa.2017.02.017

# GHI CHÚ HIỆU CHỈNH TRƯỚC KHI NỘP

1. Thay các mô tả chung ở mục 1.6.3 bằng tên ngân hàng/đơn vị, phân khúc, thời gian, số quan sát và số biến thực tế.
2. Chọn một cửa sổ dự báo cụ thể (3, 6 hoặc 12 tháng) và dùng nhất quán ở toàn luận văn.
3. Nếu đề tài là khách hàng doanh nghiệp, giữ phần Altman/Ohlson và mở rộng tỷ số tài chính; nếu là tín dụng cá nhân, rút gọn phần phá sản doanh nghiệp và nhấn mạnh behavioural scoring.
4. Không sử dụng các con số như “tăng Accuracy 60%”, “AUC > 0,73”, “hơn 70% ngân hàng” trong file Excel nếu chưa mở được bài gốc và kiểm tra đúng bảng/kết quả.
5. Workbook hiện có dấu hiệu lệch cột giữa tiêu đề, tác giả/tạp chí và phần tóm tắt ở nhiều dòng. Vì vậy, bản viết lại chỉ dùng các nguồn lõi có thể đối chiếu trong `references.bib` và văn bản pháp luật chính thức.
