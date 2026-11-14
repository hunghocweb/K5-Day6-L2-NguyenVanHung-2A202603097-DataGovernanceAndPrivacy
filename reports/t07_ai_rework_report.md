# T07 AI correction and visual re-review

> Lịch sử vòng sửa ngày 14/11/2026. Trạng thái human review hiện tại xem [báo cáo chính](group_report.md), mục 07; số audit bên dưới thuộc thời điểm ghi báo cáo này.

Reviewer: Codex (AI visual re-review)
Ghi nhận: 2026-11-14T20:14:17+07:00

Đã sửa riêng 15 ảnh theo review ban đầu và xem lại đủ 15 ảnh xuất.
Các lỗi đã ghi nhận được xử lý theo quan sát thị giác. Đây là AI review của chính bản sửa, không phải đánh giá độc lập hoặc điểm GT.

| Ảnh | Kết quả xem lại |
|---|---|
| pii_57 | Các ký tự biển số từng lộ ở mép phải đã nằm trong vùng làm mờ; các mặt đã che được giữ nguyên. |
| pii_10 | Các đuôi ký tự của ba biển số không còn nhìn thấy ở mép phải; hai mặt nổi vẫn được che. |
| pii_24 | Các mặt đã ghi nhận và biển số xe đỏ phía sau đã có vùng che; đầu quay lưng vẫn giữ nguyên. |
| pii_15 | Mặt lớn phía trên và mặt nhỏ cạnh xe xanh được che; ba biển số đã có vùng che đúng vị trí. |
| pii_48 | Các ký tự 6 và 1 từng lộ ngoài vùng che đã được phủ; vùng mặt trước đó vẫn được che. |
| pii_40 | Bốn mặt và hai biển số đã được che; kính xe xanh được khôi phục từ ảnh nguồn. |
| pii_52 | Ký tự 2 từng lộ ở mép phải biển số không còn hiện; ba vùng mặt được giữ nguyên. |
| pii_33 | Các đuôi chữ/số bên phải ba biển số không còn hiện; hai vùng mặt vẫn được che. |
| pii_08 | Mặt tối phía trên bên trái đã được che; biển số trước và hai biển số phía xa có vùng che rộng đủ phần chữ nhìn thấy. |
| pii_56 | Các đuôi ký tự bên phải ba biển số không còn hiện; các vùng mặt và đầu quay lưng giữ đúng hướng xử lý. |
| pii_38 | Ba mặt và các biển số được che; kính xe cam và xe nâu không còn bị làm mờ nhầm. |
| pii_59 | Hai mặt đã được che, biển số xe xanh phía sau được bổ sung và vùng biển số trước đã mở rộng. |
| pii_12 | Ba vùng mặt rất nhỏ đã được che; đuôi 27 và phần chữ biển số xe phía sau không còn lộ. |
| pii_01 | Các đuôi 14 và 58 từng lộ đã được phủ; hai vùng mặt giữ nguyên. |
| pii_19 | Ba vùng mặt đã được bổ sung; biển số xe hồng phía sau và xe tím phía trước có vùng che. |

## Dấu vết kiểm tra

- 15 ảnh có sửa; 45 ảnh còn lại có hash PNG giống bản PoC gốc.
- 15 sự kiện audit, actor Codex AI; chạy lại cùng đề xuất không tạo sự kiện trùng.
- Không sửa detector, confidence, padding hoặc split; không ghi đè out/poc.
- Bản xuất: out/poc_ai_corrected/redactions.csv và redacted/; contact_sheet.html để xem trước/sau.
- rework_verification.json và correction_manifest.json ghi kết quả kiểm bằng script.
- Không suy ra PII Miss Rate, over-redaction hoặc khả năng tổng quát từ vòng sửa theo ảnh này.
