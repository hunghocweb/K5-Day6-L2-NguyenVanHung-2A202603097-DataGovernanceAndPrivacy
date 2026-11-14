# T07 G07 AI visual review

> Lịch sử vòng kiểm tra ban đầu. Trạng thái human review và hồ sơ hiện tại xem [báo cáo chính](group_report.md), mục 07.

Reviewer: Codex (AI visual review)
Ghi nhận: 2026-11-14T20:13:33+07:00

Đã xem trực tiếp 15 cặp ảnh nguồn và ảnh xuất trong queue ưu tiên. Kết quả: 15 reject; đề xuất REWORK.

Quan sát từ ảnh hiển thị; không dùng GT ẩn và không suy ra PII Miss Rate hoặc over-redaction từ số quyết định.

## Kết quả theo ảnh

| Rank | Ảnh | Split | Quyết định | Lý do |
|---|---|---|---|---|
| 1 | pii_57 | eval | reject | Các vùng mặt nổi đã được che, đầu quay lưng giữ nguyên. Tuy nhiên vẫn thấy ký tự ở mép phải ngoài vùng che của biển số các xe, rõ nhất xe phía trước bên trái. Cần kiểm tra và che đủ chuỗi ký tự nhìn thấy. |
| 2 | pii_10 | dev | reject | Hai mặt nổi đã được che. Cả ba xe vẫn còn ký tự biển số lộ ngoài mép phải vùng che: xe tím bên trái, xe xanh phía sau và xe xanh vàng phía trước; còn nhìn thấy các đuôi như 7, 30, 45. Cần sửa phạm vi vùng che biển số. |
| 3 | pii_24 | dev | reject | Ảnh mờ nhưng vẫn thấy các mặt không được che: mặt lớn phía trên bên phải, hai mặt chồng gần xe bên trái và mặt nhỏ gần xe xanh bên phải. Biển số xe đỏ phía sau cũng chưa thấy vùng che rõ ràng. Cần bổ sung các vùng bị sót, không coi ảnh đã mờ là đã redaction. |
| 4 | pii_15 | dev | reject | Mặt lớn phía trên bên trái vẫn thấy mắt và miệng, chưa được che. Các biển số xe xanh lớn phía trước, xe đỏ và xe xanh phía sau vẫn nhìn thấy vùng chữ/biển số; chỉ có một vùng gần nóc xe xanh được che. Cần bổ sung mặt và biển số bị sót, kiểm tra lại vị trí vùng đang che. |
| 5 | pii_48 | eval | reject | Các mặt nổi bên trái đã được che và đầu quay lưng không bị che nhầm. Tuy nhiên xe trước bên trái còn ký tự 6 và xe trước bên phải còn ký tự 1 ở ngoài mép phải vùng che biển số. Cần che đủ phần ký tự lộ ra. |
| 6 | pii_40 | dev | reject | Bốn mặt nổi vẫn chưa được che: phía trên trái, phía trên phải, giữa bên phải và góc dưới bên phải. Biển số hai xe vẫn nhìn thấy. Vùng làm mờ đang nằm ở kính xe xanh phía trước thay vì biển số. Cần bỏ vùng che nhầm kính và bổ sung các vùng PII bị sót. |
| 7 | pii_52 | eval | reject | Ba mặt nổi đã có vùng che. Biển số xe xanh phía trước vẫn còn ký tự 2 lộ ngoài mép phải vùng che; cần kiểm tra toàn bộ chuỗi và nới/chỉnh bbox đúng đối tượng. |
| 8 | pii_33 | eval | reject | Hai mặt nổi đã được che. Cả ba xe còn ký tự biển số ở mép phải ngoài vùng che; các đuôi như 32, 40 và 39 vẫn hiện. Cần sửa phạm vi che biển số, giữ nguyên các đầu quay lưng không có mặt. |
| 9 | pii_08 | dev | reject | Mặt tối ở phía trên bên trái, gần tòa nhà đỏ, vẫn thấy mắt và miệng nhưng chưa được che. Biển số xe xanh lớn phía trước còn đuôi 37 ngoài vùng che; các xe nhỏ phía xa cũng còn ký tự ở mép phải. Cần bổ sung mặt bị sót và kiểm tra đủ chuỗi biển số. |
| 10 | pii_56 | eval | reject | Các mặt nổi đã được che, đầu quay lưng giữ nguyên. Ba biển số vẫn còn phần ký tự bên phải ngoài vùng che, quan sát được các đuôi như 04, 03 và 85. Cần chỉnh bbox che đủ chữ thay vì chỉ che nền trắng của biển. |
| 11 | pii_38 | dev | reject | Ba mặt nổi bên trái, cạnh người phía trước bên trái và phía trên bên phải chưa được che. Biển số các xe vẫn còn, trong khi các vùng mờ nằm trên kính xe cam và kính xe nâu. Cần loại vùng che nhầm kính, thêm mặt và biển số đúng vị trí. |
| 12 | pii_59 | dev | reject | Mặt nhỏ sát mép trái gần biển báo và mặt lớn phía trên bên phải chưa được che. Biển số xe xanh phía sau vẫn còn; vùng che biển số xe hồng phía trước cần kiểm tra phần ký tự ở mép phải còn lộ. Cần sửa các vùng này, không che các đầu quay lưng. |
| 13 | pii_12 | dev | reject | Các mặt rất nhỏ ở gần đường chân trời phía phải, vùng giữa ảnh phía sau người đi xe và mép phải ảnh vẫn nhìn thấy nhưng chưa có vùng che. Biển số xe trước còn đuôi 27 bên ngoài vùng che. Cần bổ sung mặt nhỏ và kiểm tra đủ chuỗi biển số. |
| 14 | pii_01 | eval | reject | Mặt lớn bên trái và mặt nhỏ phía xa đã được che. Biển số xe đỏ phía trước còn đuôi 14, biển số xe hồng phía sau còn đuôi 58 ngoài mép phải vùng làm mờ. Cần sửa phạm vi vùng che biển số. |
| 15 | pii_19 | dev | reject | Mặt lớn phía trên giữa, mặt nhỏ sát người bên trái và mặt rất nhỏ phía dưới gần giữa ảnh vẫn chưa được che. Biển số xe hồng phía sau chưa được che; xe tím phía trước đã có vùng mờ. Cần bổ sung các mặt và biển số xe hồng. |

## Nhóm lỗi cần làm lại

- Biển số còn ký tự ngoài vùng che: 10 ảnh trong queue. Một ảnh có thể có nhiều nhóm lỗi.
- Mặt bị bỏ sót: 8 ảnh trong queue. Một ảnh có thể có nhiều nhóm lỗi.
- Biển số chưa được che hoặc chưa có vùng che rõ ràng: 6 ảnh trong queue. Một ảnh có thể có nhiều nhóm lỗi.
- Vùng che nằm trên kính xe: 2 ảnh trong queue. Một ảnh có thể có nhiều nhóm lỗi.

## Hành động tiếp theo

- Giữ nguyên baseline, split và detector đã freeze. Không dùng ghi chú của ảnh eval để tune tham số chung.
- Với từng ảnh reject, lập đề xuất sửa vùng bị sót/che thiếu/che nhầm, lưu phiên bản trước và sau cùng tác giả thực hiện.
- Ưu tiên mặt còn thấy trong ảnh mờ/tối và phần chữ biển số lộ ngoài bbox; đối chiếu định nghĩa vùng GT khi đánh giá metric.

## File kết quả

- CSV quyết định: `out/ai_review/review_queue_reviewed.csv`.
- Tóm tắt và hash ảnh đã xem: `out/ai_review/review_summary.json`.
- Ghi chú nguồn: `reports/t07_ai_review_observations.json`.
- Các quyết định ghi dưới danh tính AI reviewer, tách riêng khỏi file reviewer người thật của pipeline.
