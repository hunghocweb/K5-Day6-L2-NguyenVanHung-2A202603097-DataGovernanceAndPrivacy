# Cải thiện trên ảnh thật và synthetic

Mục tiêu người dùng xác nhận ngày 2026-11-15: xử lý cả ảnh đường phố thật và ảnh synthetic mới. Khớp lại ảnh đã review là kiểm tra phát triển; không chứng minh khả năng tổng quát.

## Trạng thái thực tế

- Weights hiện có vẫn là YuNet 2023mar và YOLOv9-t plate ONNX. Chưa fine-tune hoặc tạo weights mới.
- `src/t07_rules_v2.py` là thí nghiệm chuyên cho synthetic. Không tự kích hoạt nó cho ảnh thật. Tham số được thử trên dev đã review; kết quả là training agreement, không phải điểm độc lập.
- `src/t07_multiscale.py` là nhánh inference không phụ thuộc màu da/biển số synthetic: giữ tỷ lệ ảnh bằng letterbox, dò toàn ảnh và crop 320 px chồng lấn 25%, đưa bbox về hệ tọa độ gốc, bỏ hộp cắt tại mép crop và NMS theo lớp. Cần đo recall/che thừa và latency trước khi thay nhánh chính.
- Baseline và các quyết định review hiện có được giữ nguyên. Chưa có tập ảnh thật có nhãn trong workspace; chưa phát hiện thư viện train torch/ultralytics hoặc công cụ nvidia-smi khi kiểm tra.

## Protocol phát triển model

1. Tập train gồm ảnh thật có quyền sử dụng và synthetic đa dạng. Mỗi ảnh phải có toàn bộ nhãn mặt/biển số; có cả negative: đầu quay lưng, kính xe, bảng hiệu. Không lấy CSV vùng che làm nhãn huấn luyện mà chưa kiểm tra: padding không phải bbox vật thể, dữ liệu hiện có có vùng kính và vùng 1×1 cần sửa.
2. Tách train/validation/test theo nguồn, camera hoặc video; các frame lân cận và biến thể augmentation của cùng ảnh luôn cùng một split. Kiểm hash và near-duplicate giữa split. Giữ eval chính thức của lab riêng, không đưa vào train/validation. Các ảnh eval đã xem trước đây cần công khai; muốn bằng chứng tổng quát mạnh cần tập test mới chưa xem.
3. Fine-tune checkpoint có train graph và license rõ, không coi ONNX inference là checkpoint huấn luyện. Dùng pretrained initialization, learning rate nhỏ, freeze backbone ban đầu rồi unfreeze có kiểm soát; early stopping chọn theo validation, không chọn theo test.
4. Augmentation chỉ trên train: scale/crop có cập nhật bbox, brightness/contrast, blur, JPEG và occlusion vừa phải. Giới hạn biến đổi để nhãn còn đúng. Trộn dữ liệu thật và synthetic có cân bằng; báo metric từng miền để tránh điểm tổng che giấu miền bị giảm.
5. Chọn confidence/padding trên validation theo recall dưới ràng buộc che thừa; không dùng confidence riêng làm bằng chứng an toàn. Bắt buộc báo mặt/biển số, nhỏ/vừa/lớn, tối/mờ/khuất và từng nguồn camera.
6. Đóng băng checkpoint/config rồi chạy test mới một lượt; báo baseline, candidate, runtime và chênh lệch. Chỉ promote khi cải thiện trên dữ liệu chưa dùng phát triển và guardrail đạt. Không cam kết hết overfitting chỉ vì dùng augmentation.

## Lệnh có thể chạy hiện tại

```powershell
.\.venv\Scripts\python.exe -m src.main --config configs/lab_multiscale_dev.yaml --output out/poc_multiscale_dev --offline --seed 42
.\.venv\Scripts\python.exe -m pytest tests/test_t07_multiscale.py tests/test_t07_web.py -q -o addopts=
```

Nhánh này giữ nguyên trọng số. Nó sửa cách đưa ảnh vào model, chưa phải model đã được huấn luyện lại và chưa có bằng chứng chất lượng trên ảnh thật.

Nguồn kỹ thuật: [OpenCV YuNet demo](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/demo.py) sử dụng kích thước ảnh thực; [YuNet model/training reference](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet); [open-image-models](https://github.com/ankandrew/open-image-models).
