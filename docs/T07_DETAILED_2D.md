# Bộ dò 2D nhiều chi tiết — auto-native-2

Trong luồng hiện tại `auto-cascade-1`, bộ dò này là **nhánh 2D dự phòng**.
Ảnh vẽ phẳng/elip/mờ phù hợp vẫn dùng `math2d-dev-1`; xem [kế hoạch cascade](T07_CASCADE_PLAN.md).

Lỗi được tái hiện trên ảnh hoạt hình đường phố mới do người dùng upload:
router chọn đúng 2D nhưng bộ dò cũ chỉ tìm 1 vùng mặt sai và 20 vùng biển,
gồm nhiều đèn/ô sáng. Rule cũ giả định mặt gần ellipse đồng màu và biển nằm
trong thân xe một màu; giả định này không hợp ảnh hoạt hình nhiều chi tiết.

Người dùng chọn giữ nhánh 2D chỉ dùng phép toán. Bản này không dùng YuNet/YOLO
ở tuyến 2D, không đọc nhãn/tên ảnh/tọa độ đã duyệt và không hiệu chỉnh bbox học.

| Phần | Phép toán đang dùng |
|---|---|
| Vùng mặt | Lọc màu trong HSV/RGB; lượng tử hóa màu, lấy tối đa128 màu đủ số pixel; tách component gần màu tham chiếu theo khoảng cách màu cục bộ |
| Đường bao | Xấp xỉ đa giác của contour; bỏ hình chỉ có vài cạnh thẳng; lọc kích thước và tỷ lệ |
| Nét mặt | Black-hat theo kích thước component; bỏ đường ngang/dọc dài; kiểm tra lại độ tối trên ảnh gốc, không chỉ trên ảnh biến đổi |
| Cấu trúc | Hai nét mắt cùng hàng hoặc cặp mắt–nét dưới cho mặt nghiêng; đối chiếu vùng tóc tối ngay phía trên; mặt chỉ lộ mắt vẫn được xét |
| Nền biển | Mask ít bão hòa màu, ba mức độ sáng145/190/220; closing3×3; lấy contour panel |
| Chữ biển | Component nét tối tương phản nền; bỏ viền panel và nét màu; tối thiểu2 nét, biển rất nhỏ cho1nét; tỷ lệ mực3–50% |
| Bổ sung rule cũ | Giữ các đề xuất toán học trước; panel cũ phải qua kiểm tra nét chữ. Mặt quá ít pixel được đưa vào chẩn đoán nghi ngờ thay vì tự che |
| Xuất | Gộp theo lớp; thanh trượt nới đúng một lần; blur trên ảnh gốc làm việc |

Source: `src/t07_drawing_details.py`, nối tại `src/t07_native_2d.py`.
Version tại `src/t07_auto.py`; web và CLI dùng chung dispatcher.

Trên chính ảnh dùng để phân tích lỗi, bản ghép đầy đủ đã tìm được **4 vùng mặt,
6 vùng biển** ở padding0%, bao gồm biển xe tải, biển xe trắng bị che một phần
và biển hai dòng xe máy. Đã xem overlay và ảnh che thật. Đây là chẩn đoán trên
mẫu phát triển, không phải độ chính xác độc lập, điểm dev/eval hay tỷ lệ tổng quát hóa.
Mặt và biển thật nhỏ/che khuất có thể cần người dùng nới thêm hoặc sửa.

Giới hạn: màu và độ sáng có thể đổi do phong cách vẽ/ánh sáng; phép kiểm tóc tối
có thể bỏ mặt tóc sáng; các ngưỡng kích thước và nét tối chưa được xác thực trên
nhiều bộ ảnh hoạt hình độc lập. Có thể nhầm chữ trên vật khác thành biển hoặc
bỏ vùng quá mờ. Confidence của rule là điểm cố định, không phải xác suất đúng.
Không tuyên bố đạt95% hoặc che thừa<10% từ một mẫu.

Kiểm thử bổ sung có ảnh tự dựng mới: biển có chữ nhưng không có thân xe màu,
ô sáng/đèn/icon màu không phải biển, mặt dịch sang vị trí khác và đầu không nét mặt.
Không dùng dữ liệu eval để chỉnh trong vòng này.

Ảnh upload và bản thử giữ local dưới `out/`; không thêm ảnh nguồn vào gói công khai,
không đổi phiên cũ hay tự tạo quyết định review.
