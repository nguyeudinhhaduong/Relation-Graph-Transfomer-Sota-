# Giải thích chi tiết CrossModalGatedFusion trong RelGT++

## 1. Bối cảnh trong report

Trong RelGT gốc, mỗi token của đồ thị quan hệ được tạo từ năm thành phần embedding:

- feature embedding: biểu diễn thuộc tính bảng, số, categorical, text, timestamp của node;
- type embedding: biểu diễn loại node hoặc loại bảng trong cơ sở dữ liệu quan hệ;
- hop embedding: biểu diễn khoảng cách của node lân cận so với seed node;
- time embedding: biểu diễn thông tin thời gian tương đối;
- positional encoding, hoặc PE: bổ sung vị trí/ngữ cảnh cấu trúc cho token.

RelGT gốc hợp nhất năm embedding này bằng cách nối vector lại với nhau, sau đó đưa qua một phép chiếu tuyến tính. Cách này đơn giản và dễ cài đặt, nhưng nó có một giả định ngầm: các nguồn thông tin đều được đưa vào cùng một không gian theo một công thức cố định. Mô hình có thể học trọng số trong lớp linear, nhưng bản thân bước fusion không trực tiếp hỏi: "ở token này, modality nào đang đáng tin hơn?".

RelGT++ thay bước nối-vector-rồi-chiếu-linear bằng CrossModalGatedFusion. Mục tiêu của module này là cho phép mô hình điều tiết động mức độ đóng góp của từng modality dựa trên chính modality đó và ngữ cảnh của các modality còn lại.

## 2. Ý tưởng trực giác

CrossModalGatedFusion có thể hiểu như một cơ chế "bộ lọc theo ngữ cảnh" cho năm nguồn embedding. Thay vì coi feature, type, hop, time và PE như nhau trong mọi trường hợp, module này tạo một gate riêng cho từng embedding.

Gate có giá trị nằm trong khoảng 0 đến 1, do hàm sigmoid sinh ra. Nếu gate gần 1, thông tin của modality đó được giữ lại mạnh hơn. Nếu gate gần 0, thông tin đó bị giảm ảnh hưởng. Vì gate là vector, việc mở hoặc đóng không diễn ra cho cả modality một cách thô, mà có thể diễn ra theo từng chiều ẩn của embedding.

Ví dụ:

- Trong bài toán churn, hành vi gần đây của người dùng thường rất quan trọng, nên time embedding có thể được gate cao hơn.
- Trong bài toán dự đoán sản phẩm, feature tabular và quan hệ giữa người dùng với sản phẩm có thể nổi bật hơn.
- Với các node ở xa seed node, hop embedding có thể giúp mô hình nhận biết độ tin cậy của thông tin lân cận.
- Với cơ sở dữ liệu có nhiều loại bảng, type embedding giúp tránh trộn lẫn ngữ nghĩa giữa customer, product, transaction, review, v.v.

## 3. Ký hiệu

Giả sử có \(M\) modality embedding. Trong report, \(M = 5\), tương ứng với feature, type, hop, time và PE.

Với modality thứ \(i\):

- \(e_i\): embedding đầu vào của modality thứ \(i\);
- \(\text{ctx}_i\): ngữ cảnh của modality \(i\), được tính từ trung bình các embedding còn lại;
- \(g_i\): gate của modality \(i\);
- \(W_\text{self}^{(i)}\): ma trận học được, biến đổi thông tin nội tại của \(e_i\);
- \(W_\text{ctx}^{(i)}\): ma trận học được, biến đổi ngữ cảnh của các modality khác;
- \(W_\text{val}^{(i)}\): ma trận học được, tạo value vector của modality \(i\);
- \(W_\text{out}\): phép chiếu đầu ra sau khi tổng hợp;
- \(\text{LN}\): Layer Normalization;
- \(\odot\): nhân từng phần tử.

## 4. Tính ngữ cảnh của từng modality

Công thức trong report:

```tex
\text{ctx}_i=\frac{1}{M-1}\sum_{j\ne i}e_j.
```

Nghĩa là, với modality \(i\), ta lấy trung bình tất cả các modality khác để tạo ra \(\text{ctx}_i\). Ngữ cảnh này trả lời câu hỏi: "các nguồn thông tin còn lại đang nói gì?".

Đây là điểm quan trọng của chữ "CrossModal". Gate của feature embedding không chỉ phụ thuộc vào feature embedding, mà còn phụ thuộc vào type, hop, time và PE. Tương tự, gate của time embedding cũng được điều chỉnh dựa trên feature, type, hop và PE.

Ví dụ, một timestamp có thể có ý nghĩa khác nhau nếu node là transaction so với nếu node là customer. Do đó, time embedding nên được đọc cùng type embedding. CrossModalGatedFusion tạo điều kiện cho sự phụ thuộc chéo này.

## 5. Tính gate cho từng modality

Công thức:

```tex
g_i=\sigma(W_\text{self}^{(i)}e_i+
W_\text{ctx}^{(i)}\text{ctx}_i).
```

Công thức này gồm hai thành phần:

- \(W_\text{self}^{(i)}e_i\): đánh giá thông tin nội tại của modality \(i\);
- \(W_\text{ctx}^{(i)}\text{ctx}_i\): đánh giá modality \(i\) trong tương quan với các modality khác.

Tổng của hai thành phần được đưa qua sigmoid:

```tex
\sigma(x)=\frac{1}{1+e^{-x}}.
```

Vì sigmoid đưa giá trị về khoảng 0 đến 1, \(g_i\) đóng vai trò như một vector điều khiển. Nó quyết định chiều nào của modality \(i\) nên được nhấn mạnh và chiều nào nên được làm nhẹ.

## 6. Hợp nhất các modality

Công thức hợp nhất:

```tex
h_\text{fused}=
\text{LN}\left(W_\text{out}
\sum_{i=1}^{M}g_i\odot W_\text{val}^{(i)}e_i\right).
```

Quá trình này có thể tách thành bốn bước:

1. Mỗi embedding \(e_i\) được đưa qua \(W_\text{val}^{(i)}\) để tạo value vector.
2. Value vector được nhân từng phần tử với gate \(g_i\).
3. Tất cả modality sau gating được cộng lại.
4. Tổng này được chiếu qua \(W_\text{out}\), sau đó chuẩn hóa bằng LayerNorm.

LayerNorm giúp ổn định phân phối của vector hợp nhất, đặc biệt khi các gate thay đổi linh hoạt theo từng token và từng batch. Nếu không có bước chuẩn hóa, biên độ của \(h_\text{fused}\) có thể dao động mạnh, làm quá trình huấn luyện khó ổn định hơn.

## 7. So sánh với concat + linear projection

Trong RelGT gốc:

```tex
h = W[e_\text{feat}; e_\text{type}; e_\text{hop}; e_\text{time}; e_\text{PE}]
```

Fusion được thực hiện bằng cách nối năm embedding thành một vector lớn rồi học một phép chiếu. Cách này có ưu điểm là gọn, nhanh và ít module phụ.

Trong RelGT++:

```tex
h_\text{fused} =
\text{LN}\left(W_\text{out}
\sum_i g_i \odot W_\text{val}^{(i)}e_i\right)
```

Fusion được thực hiện bằng cách học gate riêng cho từng modality. Điểm khác biệt chính là RelGT++ không chỉ học cách trộn các vector, mà học khi nào nên tin nguồn thông tin nào.

## 8. Lợi ích đối với dữ liệu quan hệ

Dữ liệu quan hệ thường không đồng nhất. Các bảng khác nhau có schema khác nhau, các cạnh có ngữ nghĩa khác nhau, và mức độ quan trọng của thời gian thay đổi theo task. Vì vậy, một cơ chế fusion cố định có thể chưa đủ linh hoạt.

CrossModalGatedFusion phù hợp với relational learning vì:

- Nó xử lý được tính đa nguồn của token: mỗi token gồm nhiều thành phần ngữ nghĩa khác nhau.
- Nó cho phép mỗi task tự học cách ưu tiên modality.
- Nó giảm rủi ro modality yếu hoặc nhiễu làm hỏng biểu diễn chung.
- Nó giữ được thông tin cross-modal, vì gate của từng modality được tính dựa trên các modality còn lại.
- Nó cải thiện khả năng giải thích tương đối: có thể quan sát gate để xem mô hình đang ưu tiên nguồn tin nào.

## 9. Vì sao CrossModalGatedFusion hiệu quả?

### 9.1. Fusion tuyến tính có trọng số cố định

Trong RelGT gốc, nếu bỏ qua bias, phép concat rồi linear projection có thể viết lại như:

```tex
h
= W[e_1;e_2;\cdots;e_M]
= \sum_{i=1}^{M} W_i e_i.
```

Ở đây, \(W_i\) là phần ma trận ứng với modality thứ \(i\). Công thức này cho thấy RelGT gốc thật ra đang cộng đóng góp của từng modality sau một phép biến đổi tuyến tính. Tuy nhiên, \(W_i\) là tham số cố định sau khi huấn luyện. Với mọi token, mọi mẫu dữ liệu và mọi ngữ cảnh, cùng một \(W_i\) được dùng để trộn modality \(i\).

Điều này có thể chưa tối ưu vì dữ liệu quan hệ rất không đồng nhất. Một node transaction, một node customer và một node product có thể cần cách đọc feature, type, hop, time khác nhau.

### 9.2. Gate tạo trọng số động theo từng token

CrossModalGatedFusion thay trọng số cố định bằng trọng số động:

```tex
h_\text{fused}
=
\text{LN}\left(
W_\text{out}
\sum_{i=1}^{M}
g_i(x)\odot W_\text{val}^{(i)}e_i
\right).
```

Trong đó gate phụ thuộc vào input:

```tex
g_i(x)
=
\sigma\left(
W_\text{self}^{(i)}e_i
+
W_\text{ctx}^{(i)}\text{ctx}_i
\right).
```

Vì \(g_i(x)\) thay đổi theo từng token, cùng một modality có thể được dùng mạnh ở token này nhưng dùng nhẹ ở token khác. Đây là điểm làm module hiệu quả hơn concat tuyến tính: fusion không còn là một công thức tĩnh, mà trở thành một hàm điều kiện theo dữ liệu.

Có thể xem \(g_i(x)\) như hệ số tin cậy của modality \(i\):

```tex
\text{contribution}_i
=
g_i(x)\odot W_\text{val}^{(i)}e_i.
```

Nếu modality \(i\) phù hợp với ngữ cảnh hiện tại, \(g_i(x)\) tăng và đóng góp của nó lớn hơn. Nếu modality đó nhiễu hoặc ít liên quan, \(g_i(x)\) giảm và ảnh hưởng của nó được làm nhẹ.

### 9.3. Gate dùng cả thông tin nội tại và thông tin chéo

Gate không chỉ nhìn vào \(e_i\), mà còn nhìn vào \(\text{ctx}_i\):

```tex
\text{ctx}_i
=
\frac{1}{M-1}
\sum_{j\ne i}e_j.
```

Do đó:

```tex
g_i
=
\sigma(
\underbrace{W_\text{self}^{(i)}e_i}_{\text{self signal}}
+
\underbrace{W_\text{ctx}^{(i)}\text{ctx}_i}_{\text{cross-modal signal}}
).
```

Thành phần self signal trả lời: "bản thân modality này có thông tin mạnh không?".

Thành phần cross-modal signal trả lời: "khi đặt modality này cạnh các modality khác, nó có còn đáng tin không?".

Ví dụ, time embedding có thể quan trọng khi feature và type cho thấy node là transaction gần thời điểm dự đoán. Nhưng cùng một dạng time embedding có thể ít quan trọng hơn nếu node là một thực thể tĩnh như product category. Nhờ \(\text{ctx}_i\), gate có khả năng phân biệt hai trường hợp này.

### 9.4. Giảm nhiễu bằng cơ chế nhân từng chiều

Vì \(g_i\) là vector, phép nhân

```tex
g_i\odot W_\text{val}^{(i)}e_i
```

không chỉ chọn hoặc bỏ toàn bộ modality. Nó có thể giữ một số chiều ẩn và làm yếu các chiều khác. Đây là cơ chế lọc mềm theo từng chiều đặc trưng.

Nếu một modality chứa cả tín hiệu hữu ích và nhiễu, gate có thể học:

```tex
g_{i,k}\approx 1
\quad \text{với chiều hữu ích,}
\qquad
g_{i,k}\approx 0
\quad \text{với chiều nhiễu.}
```

Nhờ vậy, CrossModalGatedFusion không cần loại bỏ toàn bộ feature, time hay PE. Nó chỉ giảm những phần không phù hợp trong biểu diễn ẩn.

### 9.5. Tăng khả năng biểu diễn phi tuyến

Concat + linear projection về cơ bản là một phép biến đổi tuyến tính trên vector đã nối:

```tex
h = \sum_i W_i e_i.
```

CrossModalGatedFusion có thêm sigmoid và phép nhân giữa gate với value:

```tex
h_\text{fused}
\propto
\sum_i
\sigma(W_\text{self}^{(i)}e_i + W_\text{ctx}^{(i)}\text{ctx}_i)
\odot
W_\text{val}^{(i)}e_i.
```

Do có \(\sigma(\cdot)\) và phép nhân \(\odot\), module này biểu diễn được tương tác phi tuyến giữa các modality. Nói cách khác, đóng góp của \(e_i\) không chỉ phụ thuộc vào chính \(e_i\), mà phụ thuộc vào quan hệ giữa \(e_i\) và các \(e_j\) khác.

Đây là lợi thế quan trọng trong dữ liệu quan hệ, nơi ý nghĩa của một tín hiệu thường phụ thuộc vào schema, loại node, khoảng cách hop và thời điểm xuất hiện.

### 9.6. Ổn định huấn luyện bằng chuẩn hóa

Sau khi cộng các modality đã gate, RelGT++ dùng LayerNorm:

```tex
h_\text{fused}
=
\text{LN}(z),
\qquad
z =
W_\text{out}
\sum_i g_i\odot W_\text{val}^{(i)}e_i.
```

LayerNorm giúp đưa biểu diễn về phân phối ổn định hơn:

```tex
\text{LN}(z)
=
\gamma
\frac{z-\mu(z)}{\sqrt{\sigma^2(z)+\epsilon}}
+\beta.
```

Điều này quan trọng vì gate thay đổi theo từng token. Nếu không chuẩn hóa, độ lớn của vector hợp nhất có thể dao động mạnh khi nhiều gate cùng cao hoặc cùng thấp. LayerNorm giúp phần sau của mô hình, như Local Transformer hoặc Global Module, nhận đầu vào ổn định hơn.

### 9.7. Tóm tắt bằng một câu

CrossModalGatedFusion hiệu quả vì nó biến bước hợp nhất embedding từ một phép trộn tuyến tính cố định thành một phép trộn động, có điều kiện theo ngữ cảnh, có khả năng lọc nhiễu theo từng chiều và biểu diễn được tương tác phi tuyến giữa các modality.

## 10. Ví dụ minh họa

Xét một seed node là customer trong bài toán dự đoán churn. Các lân cận có thể gồm transaction, product, session hoặc support ticket.

Nếu một customer có nhiều hành vi bất thường gần ngày dự đoán, time embedding nên có trọng số cao. Nếu các hành vi cũ hơn và ít liên quan hơn, gate của time có thể giảm. Nếu customer thuộc một nhóm có đặc trưng nhân khẩu học rất mạnh, feature embedding có thể được ưu tiên. Nếu thông tin đến từ node ở xa seed node, hop embedding giúp mô hình cân nhắc độ gần cấu trúc.

Như vậy, CrossModalGatedFusion không tạo một công thức hợp nhất cố định cho mọi token. Nó tạo một công thức hợp nhất phụ thuộc vào nội dung token và bối cảnh cross-modal của token đó.

## 11. Hạn chế và điểm cần lưu ý

CrossModalGatedFusion mạnh hơn concat + linear projection, nhưng đổi lại có thêm tham số và phép tính. Với mỗi modality, module cần các ma trận \(W_\text{self}^{(i)}\), \(W_\text{ctx}^{(i)}\), \(W_\text{val}^{(i)}\). Nếu số modality lớn, chi phí có thể tăng đáng kể.

Một điểm cần lưu ý khác là gate không nên bị bão hòa quá sớm. Nếu sigmoid sinh ra giá trị quá gần 0 hoặc quá gần 1 ở đầu huấn luyện, gradient có thể yếu. Trong cài đặt thực tế, có thể cần khởi tạo cẩn thận, LayerNorm, dropout hoặc regularization để giữ quá trình học ổn định.

Ngoài ra, gate cao không luôn đồng nghĩa với "modality quan trọng nhất" theo nghĩa nhân quả. Nó chỉ cho thấy mô hình đang sử dụng modality đó mạnh hơn trong biểu diễn ẩn. Khi dùng để giải thích, cần kết hợp với ablation hoặc phân tích theo task.

## 12. Đoạn diễn giải có thể đưa vào report

CrossModalGatedFusion được dùng để thay thế phép nối embedding trong RelGT gốc. Thay vì trộn feature, type, hop, time và PE bằng một phép chiếu tuyến tính cố định, RelGT++ học một gate riêng cho từng modality. Gate này được tính từ hai nguồn: thông tin nội tại của modality đang xét và ngữ cảnh trung bình của các modality còn lại. Nhờ đó, mô hình có thể điều chỉnh động mức độ đóng góp của từng nguồn thông tin theo từng token và từng tác vụ.

Sau khi có gate, mỗi modality được biến đổi thành value vector, nhân với gate rồi cộng lại. Vector tổng hợp được đưa qua phép chiếu đầu ra và LayerNorm để tạo \(h_\text{fused}\). Cơ chế này phù hợp với dữ liệu quan hệ vì mỗi tác vụ có thể phụ thuộc vào các nguồn tin khác nhau: churn thường nhạy với thời gian, dự đoán sản phẩm có thể phụ thuộc nhiều vào feature và quan hệ, còn các bài toán liên quan schema cần type embedding rõ ràng. Vì vậy, CrossModalGatedFusion giúp RelGT++ linh hoạt hơn và có khả năng biểu diễn tốt hơn so với fusion bằng concat đơn giản.
