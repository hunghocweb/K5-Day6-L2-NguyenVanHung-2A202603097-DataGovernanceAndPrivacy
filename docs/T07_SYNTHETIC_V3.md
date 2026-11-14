# Nhánh synthetic v3 — đầu có nét mặt và chữ biển số tràn nền

Đây là thí nghiệm riêng theo yêu cầu người dùng ngày15/11/2026, không thay output baseline/PoC đã freeze. Không có lookup theo image_id, không đọc file review hoặc GT khi inference.

1. **Tìm đầu:** kết hợp vùng màu gần da, MSER, đường biên Canny và Hough circle. Các vùng tròn/elip phải khớp hình học với mask đầu; vùng đầu dính nền cùng màu được kiểm thêm bằng độ tương đồng màu cục bộ.
2. **Xác định có mặt:** tìm hai nét mắt cân đối và một nét thấp hơn ở trung tâm (miệng/mũi) bằng tương phản và black-hat. Hình vẽ không có mũi vẫn có thể hợp lệ nếu có mắt và miệng. Không chấp nhận chỉ vì vùng có hình elip; đầu không đủ nét mặt được để lại cho review, không khẳng định chắc chắn là an toàn.
3. **Tìm biển số:** vùng nền trắng/gần trung tính, hình chữ nhật ngang, có nét chữ đen bên trong. Loại vùng quá rỗng, không đủ độ phủ chữ nhật hoặc chỉ có viền kính. Các ngưỡng sáng90/130/170/210 nhằm tìm cùng một nền ở nhiều mức sáng.
4. **Gom chữ ngoài nền:** black-hat và connected-components theo dải chữ, mở rộng cả trái và phải. Bỏ đường viền xe dài; giữ nét đứng mảnh của chữ số. Khi vùng người cao/hẹp cắt ngang biển số, cho phép nối qua khoảng ngắt tương ứng vật cản thay vì dừng ngay ở mép nền trắng.
5. **Xuất:** gộp vùng trùng, clip trong ảnh, padding nhỏ rồi blur. Luôn xem lại PNG; số bbox hoặc test pass không chứng minh đã che hết chữ/mặt.

Ảnh thật dùng nhánh **general** với YuNet cho mặt và YOLOv9-t cho biển số, weights local đã kiểm hash. Hiện chưa có model YOLO person/head riêng; không gọi YuNet là YOLO và không tự che toàn người chỉ vì có class person.

## Chạy

```powershell
.\.venv\Scripts\python.exe -m src.main --config configs/lab_v3_dev.yaml --output out/poc_v3_dev --offline --seed 42
.\.venv\Scripts\python.exe -m pytest tests/test_t07_rules_v3.py -q -o addopts=
.\.venv\Scripts\python.exe -m tools.smoke_t07_walkthrough --mode synthetic_v3
```

Trên trang `/walkthrough`, chọn **Synthetic v3 · elip, nét mặt và chữ biển số**. V2 và detector general vẫn có lựa chọn riêng.

## Giới hạn

- Nét mặt bị mờ/khuất có thể mất tín hiệu; chữ đen trên thân xe tối có thể không đủ tương phản. Phần mở rộng chữ cũng có thể nhận nhầm viền/đèn hoặc vùng đen cùng dòng.
- Hình tròn/elip không đủ để kết luận là mặt; mặc dù có lọc nét mặt, false positive và bỏ sót vẫn cần đo bằng nhãn độc lập.
- Những ảnh fixture đã được xem trong hội thoại là ví dụ phát triển/kiểm tra mở rộng, không được gọi là test mới chưa xem. `configs/lab_v3_dev.yaml` chỉ chạy dev; điểm chính thức vẫn cần hidden eval mới.
- Hash phiên bản thí nghiệm ở `reports/t07_v3_freeze.json`. Những đề xuất bbox cho hai reject là artifact riêng, không phải model đã học hay hành động reviewer người thật.
