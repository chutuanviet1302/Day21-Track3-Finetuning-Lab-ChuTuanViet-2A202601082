# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Hiện tượng đảo ngược thứ tự giữa Training Loss và Task Accuracy ở run `attn_only`: khi nâng rank lên cực đại $r=283$ để khớp số tham số, `attn_only` ép loss huấn luyện xuống thấp hơn hẳn bản `correct` ($0.5374$ vs $0.6264$), nhưng khi chấm điểm trên tác vụ target thật thì lại thua ($0.9380$ vs $0.9400$). Điều này chứng minh rank lớn ở $q,v$ chỉ giúp mô hình overfit dữ liệu huấn luyện chứ không mang lại biểu diễn tổng quát bằng việc rải rank nhỏ ($r=16$) trên toàn bộ các tầng `text-linear`.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Mất nhiều thời gian nhất ở khâu sinh văn bản đánh giá (Greedy Decode) ở NB2/NB5 và chạy 4 run đối chứng song song ở NB4 (~80 phút trên Colab T4). Ban đầu tôi dự đoán bước nạp mô hình và cấu hình LoRA sẽ phức tạp và tốn thời gian nhất, nhưng thực tế phần tính toán suy luận (inference) để sinh JSON và đo 3 baseline trước/sau train mới là phần ngốn thời gian dài nhất.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Trước lab này, tôi từng tin hai điều sai lầm:
1. Nghĩ rằng chỉ cần gắn adapter vào các ma trận Attention ($q, v$) rồi tăng rank $r$ thật cao là mô hình sẽ mạnh nhất. Giờ tôi tin vào nguyên lý *LoRA Without Regret*: gắn toàn bộ linear với rank nhỏ ($r=16$) vượt trội hơn hẳn.
2. Nghĩ rằng Training Loss giảm càng sâu thì mô hình càng tốt. Giờ tôi hiểu Training Loss chỉ là chỉ số thay thế (proxy metric) và hoàn toàn có thể đánh lừa người làm kỹ thuật nếu không đánh giá trực tiếp trên tác vụ (task accuracy).

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

- **Việc dùng:** Dùng AI assistant để hỗ trợ thiết lập môi trường local (xử lý lỗi UTF-8 trên Windows PowerShell), tự động hóa kiểm tra tính nhất quán dữ liệu giữa các file JSON với `verify.py`, và hỗ trợ rà soát cấu trúc báo cáo.
- **Chỗ AI sai:** Khi Colab gặp sự cố, AI ban đầu đề xuất chỉ chạy lại `nb2 nb5` mà không kiểm tra xem máy ảo Colab đã bị ngắt kết nối và xoá mất thư mục `adapters/correct` hay chưa, dẫn đến lỗi `ValueError: Can't find adapter_config.json`.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Đóng băng một tập test đánh giá chuẩn (Golden Evaluation Set) và đo lường kỹ lưỡng năng lực của mô hình nền với Prompt Engineering tối ưu (Baseline b). Nếu prompt engineering đã giải quyết được 80-90% yêu cầu về format và độ chính xác thì chưa vội fine-tune. Nếu bắt buộc fine-tune, bước kỹ thuật đầu tiên luôn là **chạy giải mã ngược để kiểm tra Loss Mask (Mask Proof)** nhằm đảm bảo 100% rằng mô hình chỉ học trên câu trả lời mong muốn chứ không tính loss trên prompt hay câu hỏi của khách hàng.
