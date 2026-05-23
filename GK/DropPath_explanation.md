# Giải thích vì sao dùng DropPath trong RelGT++

## 1. DropPath là gì?

DropPath, còn gọi là Stochastic Depth, là kỹ thuật regularization dùng trong các mạng sâu có residual connection.

Trong Transformer block, một nhánh residual thường có dạng:

```text
output = x + F(x)
```

Trong đó:

- `x`: input đi qua đường tắt residual.
- `F(x)`: nhánh xử lý chính, ví dụ attention hoặc FFN.
- `output`: kết quả sau khi cộng residual.

DropPath làm cho một số nhánh `F(x)` bị bỏ qua ngẫu nhiên trong lúc huấn luyện:

```text
output = x + DropPath(F(x))
```

Nếu nhánh được giữ:

```text
DropPath(F(x)) = F(x) / keep_prob
```

Nếu nhánh bị bỏ:

```text
DropPath(F(x)) = 0
```

Trong đó:

```text
keep_prob = 1 - drop_prob
```

Ví dụ nếu `drop_prob = 0.1`, thì `keep_prob = 0.9`. Nghĩa là trong lúc huấn luyện, khoảng 10% nhánh residual có thể bị bỏ ngẫu nhiên.

## 2. Vì sao không dùng Dropout thường?

Dropout thường bỏ ngẫu nhiên từng phần tử trong vector:

```text
x = [x1, x2, x3, x4]
dropout(x) có thể thành [x1, 0, x3, 0]
```

DropPath bỏ cả một nhánh residual:

```text
x + F(x) có thể thành x + 0
```

Nói đơn giản:

- Dropout làm nhiễu ở mức feature.
- DropPath làm nhiễu ở mức layer hoặc block.

Với Transformer sâu, DropPath thường phù hợp hơn vì nó ép mô hình không phụ thuộc quá nhiều vào một block cụ thể.

## 3. Vì sao dùng DropPath trong RelGT++?

RelGT++ có nhiều module mạnh hơn RelGT gốc, ví dụ:

- CrossModalGatedFusion.
- Local Transformer.
- SwiGLU FFN.
- Global module.
- Cross-attention bridge.

Khi mô hình mạnh hơn, số tham số và khả năng ghi nhớ dữ liệu cũng tăng. Điều này có thể dẫn đến overfitting, đặc biệt khi dữ liệu huấn luyện không đủ lớn hoặc các quan hệ trong graph có nhiễu.

DropPath giúp giảm overfitting bằng cách làm cho mô hình không được phụ thuộc cố định vào mọi nhánh xử lý trong mọi lần huấn luyện.

## 4. Trực giác chính

Không có DropPath:

```text
block_1 luôn hoạt động
block_2 luôn hoạt động
block_3 luôn hoạt động
...
block_N luôn hoạt động
```

Mô hình có thể học cách dựa quá mạnh vào một vài block nhất định.

Có DropPath:

```text
lần train 1: dùng block_1, bỏ block_2, dùng block_3
lần train 2: bỏ block_1, dùng block_2, dùng block_3
lần train 3: dùng block_1, dùng block_2, bỏ block_3
```

Mỗi batch giống như huấn luyện một phiên bản mạng hơi khác nhau. Vì vậy, mô hình học biểu diễn bền hơn thay vì phụ thuộc vào một đường xử lý cố định.

## 5. Vì sao DropPath hiệu quả?

### 5.1. Giảm overfitting

Trong mạng sâu, các layer có thể phối hợp quá chặt với nhau để ghi nhớ dữ liệu train. DropPath phá vỡ sự phụ thuộc cố định này.

Thay vì luôn có:

```text
output = x + F(x)
```

trong một số lần huấn luyện mô hình chỉ thấy:

```text
output = x
```

Điều này buộc các block khác cũng phải học biểu diễn hữu ích, không để toàn bộ hiệu quả phụ thuộc vào một nhánh duy nhất.

### 5.2. Tạo hiệu ứng ensemble ngầm

Mỗi lần DropPath bỏ một số nhánh khác nhau, mô hình đang huấn luyện một kiến trúc con khác nhau.

Có thể hình dung:

```text
model_full
model_without_block_2
model_without_block_5
model_without_block_2_and_5
...
```

Tất cả các kiến trúc con này chia sẻ cùng tham số. Khi inference, DropPath tắt, toàn bộ mô hình được sử dụng. Kết quả giống như lấy lợi ích của nhiều mô hình con, nhưng không cần huấn luyện nhiều mô hình riêng.

### 5.3. Giúp huấn luyện mạng sâu ổn định hơn

RelGT++ có thể sâu hơn và nhiều module hơn RelGT gốc. Khi mạng sâu, gradient phải đi qua nhiều block. Nếu mọi nhánh residual luôn hoạt động, mô hình có thể trở nên nhạy với một số nhánh mạnh.

DropPath giúp mạng học cách hoạt động cả khi một số nhánh bị thiếu. Điều này làm biểu diễn ổn định hơn và giảm rủi ro một nhánh gây nhiễu quá mạnh.

### 5.4. Tăng độ bền với graph nhiễu

Dữ liệu quan hệ thường có nhiễu:

- cạnh không thật sự liên quan đến task;
- timestamp thiếu chính xác;
- node lân cận mang thông tin yếu;
- feature tabular bị thiếu hoặc không đồng nhất.

Nếu một block học quá mạnh từ tín hiệu nhiễu, mô hình dễ overfit. DropPath làm giảm khả năng đó bằng cách ngẫu nhiên bỏ nhánh xử lý, buộc mô hình học tín hiệu phân tán và ổn định hơn.

## 6. Công thức dễ hiểu

Giả sử có một nhánh residual:

```text
y = x + F(x)
```

Khi dùng DropPath trong training:

```text
mask = 1 với xác suất keep_prob
mask = 0 với xác suất drop_prob

y = x + mask * F(x) / keep_prob
```

Nếu `mask = 1`:

```text
y = x + F(x) / keep_prob
```

Nếu `mask = 0`:

```text
y = x
```

Chia cho `keep_prob` để giữ kỳ vọng đầu ra không đổi giữa training và inference.

Ví dụ:

```text
drop_prob = 0.2
keep_prob = 0.8
```

Khi nhánh được giữ:

```text
F(x) được nhân với 1 / 0.8 = 1.25
```

Khi lấy trung bình qua nhiều lần huấn luyện, độ lớn kỳ vọng của nhánh vẫn gần giống lúc không dùng DropPath.

## 7. Training và inference khác nhau thế nào?

Trong training:

```text
output = x + DropPath(F(x))
```

Một số nhánh residual có thể bị bỏ ngẫu nhiên.

Trong inference:

```text
output = x + F(x)
```

DropPath được tắt. Toàn bộ mô hình được dùng để dự đoán.

Điều này quan trọng: DropPath chỉ dùng để regularize trong lúc huấn luyện, không làm mất thông tin khi dự đoán.

## 8. Ý nghĩa trong RelGT++

Trong RelGT++, DropPath thường được đặt ở các residual branch của Transformer block, ví dụ:

```text
x = x + DropPath(Attention(LayerNorm(x)))
x = x + DropPath(SwiGLU(LayerNorm(x)))
```

Ý nghĩa:

- Nhánh attention không phải lúc nào cũng được dùng trọn vẹn trong training.
- Nhánh FFN/SwiGLU cũng không phải lúc nào cũng được dùng trọn vẹn.
- Token representation phải đủ tốt ngay cả khi một số nhánh bị bỏ.

Vì vậy, DropPath giúp RelGT++ huấn luyện ổn định hơn, giảm overfitting và tăng khả năng tổng quát hóa.

## 9. So sánh nhanh

| Kỹ thuật | Bỏ cái gì? | Mục tiêu |
|---|---|---|
| Dropout | Một số chiều trong vector | Giảm phụ thuộc vào feature riêng lẻ |
| DropPath | Cả nhánh residual hoặc block | Giảm phụ thuộc vào layer/block riêng lẻ |
| Weight decay | Phạt trọng số lớn | Làm mô hình đơn giản hơn |
| Early stopping | Dừng train sớm | Tránh học quá mức trên train set |

## 10. Đoạn diễn giải có thể đưa vào report

DropPath được sử dụng trong RelGT++ như một kỹ thuật regularization cho các residual branch của Transformer block. Thay vì luôn dùng đầy đủ mọi nhánh attention và FFN trong quá trình huấn luyện, DropPath ngẫu nhiên bỏ qua một số nhánh với xác suất nhỏ. Khi một nhánh bị bỏ, output chỉ đi qua đường residual `x`; khi nhánh được giữ, output là `x + F(x)`.

Cơ chế này giúp giảm overfitting vì mô hình không thể phụ thuộc quá mạnh vào một block cụ thể. Mỗi batch huấn luyện tương ứng với một kiến trúc con hơi khác nhau, tạo hiệu ứng ensemble ngầm. Khi inference, DropPath được tắt và toàn bộ mô hình được sử dụng. Do đó, trong RelGT++, DropPath giúp huấn luyện các module sâu như Local Transformer và SwiGLU ổn định hơn, đồng thời tăng khả năng tổng quát hóa trên dữ liệu quan hệ có nhiễu.
