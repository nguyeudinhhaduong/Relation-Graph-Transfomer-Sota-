# Giải thích sâu các module 4.5, 4.6, 4.7 trong RelGT++

File này giải thích ba module trong RelGT++:

- `Multi-View Readout`
- `GumbelSoftCodebook`
- `CrossAttentionBridge`

Mục tiêu chung của ba module là làm biểu diễn của RelGT++ ổn định hơn, giàu thông tin hơn và ít bị phụ thuộc vào một cách tổng hợp duy nhất.

---

# 4.5. Multi-View Readout

## 1. Bài toán của readout

Sau khi Local Transformer xử lý các token trong subgraph, ta có nhiều vector node/token:

```text
h1, h2, h3, ..., hK
```

Trong đó:

- `hi`: biểu diễn của seed node hoặc token chính.
- `hj`: biểu diễn của node lân cận thứ `j`.
- `K`: số token trong subgraph.

Mục tiêu của readout là gom nhiều vector lân cận thành một vector ngữ cảnh duy nhất để cập nhật cho seed node.

Nếu chỉ dùng một kiểu pooling, mô hình dễ bị lệch:

- Chỉ dùng attention pooling: có thể bỏ qua bức tranh tổng thể.
- Chỉ dùng mean pooling: có thể làm loãng tín hiệu hiếm nhưng quan trọng.
- Chỉ dùng max pooling: có thể quá nhạy với outlier.

Vì vậy, RelGT++ dùng Multi-View Readout, tức là tổng hợp ba góc nhìn cùng lúc.

## 2. Ba góc nhìn của Multi-View Readout

### 2.1. Attention pooling

```text
v_attn = sum_j alpha_j * h_j
```

Trong đó:

- `alpha_j`: trọng số attention của node lân cận `j`.
- `h_j`: biểu diễn của node lân cận `j`.
- `v_attn`: vector tổng hợp theo attention.

Ý nghĩa:

```text
Node nào quan trọng hơn thì được trọng số lớn hơn.
```

Attention pooling giúp mô hình chọn lọc thông tin. Ví dụ trong bài toán churn, không phải mọi transaction đều quan trọng. Một vài transaction gần đây hoặc bất thường có thể đáng chú ý hơn.

Điểm mạnh:

- Bắt được quan hệ có chọn lọc.
- Phù hợp khi chỉ vài node lân cận thật sự liên quan đến task.
- Giúp mô hình tập trung vào tín hiệu quan trọng.

Điểm yếu nếu dùng một mình:

- Nếu attention học sai, thông tin tổng hợp có thể lệch.
- Có thể bỏ qua phân phối chung của toàn bộ lân cận.

### 2.2. Mean pooling

```text
v_mean = (1 / (K - 1)) * sum_j h_j
```

Trong đó:

- `K - 1`: số node lân cận, không tính seed node.
- `v_mean`: vector trung bình của các lân cận.

Ý nghĩa:

```text
Lấy bức tranh trung bình của toàn bộ vùng lân cận.
```

Mean pooling không cố chọn node nào quan trọng nhất. Nó giữ thông tin nền, phản ánh xu hướng chung của neighborhood.

Ví dụ:

- Một customer có nhiều transaction nhỏ đều đặn.
- Một product được nhiều user tương tác ở mức vừa phải.
- Một account có hành vi trung bình khác biệt với nhóm còn lại.

Những tín hiệu này không nhất thiết nằm ở một node nổi bật, mà nằm trong phân phối chung của nhiều node.

Điểm mạnh:

- Ổn định.
- Ít nhạy với nhiễu đơn lẻ.
- Giữ thông tin tổng quát của neighborhood.

Điểm yếu nếu dùng một mình:

- Có thể làm loãng tín hiệu hiếm.
- Nếu chỉ một vài node thật sự quan trọng, mean có thể che mất chúng.

### 2.3. Max pooling

```text
v_max = max_j h_j
```

Ở đây `max` được lấy theo từng chiều vector.

Ví dụ:

```text
h1 = [0.1, 2.0, 0.3]
h2 = [1.5, 0.4, 0.8]
h3 = [0.7, 1.1, 3.2]

v_max = [1.5, 2.0, 3.2]
```

Ý nghĩa:

```text
Giữ lại tín hiệu mạnh nhất xuất hiện trong neighborhood.
```

Max pooling hữu ích khi tín hiệu quan trọng rất hiếm nhưng có giá trị dự báo cao.

Ví dụ:

- Một giao dịch bất thường duy nhất có thể báo hiệu churn hoặc fraud.
- Một lần mua sản phẩm đặc biệt có thể gợi ý sở thích mạnh.
- Một tương tác gần đây có cường độ cao có thể quan trọng hơn nhiều tương tác bình thường.

Điểm mạnh:

- Giữ tín hiệu nổi bật hiếm.
- Không làm loãng extreme signal.
- Phù hợp với các task có sự kiện bất thường.

Điểm yếu nếu dùng một mình:

- Nhạy với outlier.
- Có thể lấy nhầm nhiễu làm tín hiệu mạnh.

## 3. Ghép ba góc nhìn

RelGT++ nối ba vector:

```text
readout = concat(v_attn, v_mean, v_max)
```

Sau đó đưa qua MLP:

```text
update = MLP(readout)
```

Cuối cùng cập nhật seed node:

```text
h_local = LayerNorm( h_i + update )
```

Trong đó:

- `h_i`: biểu diễn ban đầu của seed node.
- `update`: thông tin tổng hợp từ neighborhood.
- `LayerNorm`: chuẩn hóa để ổn định biểu diễn.

Residual connection `h_i + update` giúp mô hình giữ lại thông tin gốc của seed node, đồng thời bổ sung ngữ cảnh lân cận.

## 4. Tại sao Multi-View Readout thành công?

Multi-View Readout thành công vì nó không ép mô hình chọn một kiểu tổng hợp duy nhất.

Ba pooling tương ứng với ba loại tín hiệu khác nhau:

| Thành phần | Bắt loại tín hiệu nào? | Khi nào hữu ích? |
|---|---|---|
| `v_attn` | Tín hiệu có chọn lọc | Khi vài node quan trọng hơn phần còn lại |
| `v_mean` | Tín hiệu trung bình | Khi xu hướng chung của neighborhood quan trọng |
| `v_max` | Tín hiệu nổi bật hiếm | Khi một sự kiện mạnh có giá trị dự báo cao |

Trong dữ liệu quan hệ, cả ba loại tín hiệu này đều có thể xuất hiện. Vì vậy, dùng cả ba giúp mô hình bền hơn.

Nói ngắn gọn:

```text
attention = biết nhìn vào ai
mean      = biết nhìn toàn cảnh
max       = biết giữ tín hiệu bất thường
```

Đây là lý do Multi-View Readout thường hiệu quả hơn attention pooling đơn lẻ.

---

# 4.6. GumbelSoftCodebook

## 1. Codebook là gì?

Codebook là một tập các vector đại diện, còn gọi là centroid hoặc prototype.

Giả sử có `B` centroid:

```text
e1, e2, e3, ..., eB
```

Với một vector query `qi`, mô hình tìm xem `qi` gần centroid nào nhất.

Ý tưởng:

```text
qi gần centroid nào thì centroid đó đại diện cho qi.
```

Trong RelGT++, codebook giúp mô hình gom các biểu diễn global thành các mẫu đại diện. Điều này có thể làm biểu diễn global gọn hơn, có cấu trúc hơn và dễ tổng quát hơn.

## 2. Vấn đề codebook collapse

Codebook collapse xảy ra khi chỉ một vài centroid được dùng nhiều, còn các centroid khác gần như không được dùng.

Ví dụ có 8 centroid:

```text
centroid usage = [90%, 8%, 2%, 0%, 0%, 0%, 0%, 0%]
```

Khi đó, codebook gần như bị lãng phí. Dù có 8 centroid, mô hình thực tế chỉ dùng 2 hoặc 3 centroid.

Hậu quả:

- Giảm khả năng biểu diễn đa dạng.
- Các prototype không học được vai trò riêng.
- Global representation dễ bị nghèo thông tin.
- Mô hình có thể overfit vào vài centroid phổ biến.

## 3. Tính điểm gần centroid

RelGT++ tính logit giữa query `qi` và centroid `eb` bằng khoảng cách âm:

```text
l_i_b = - squared_norm(q_i - e_b)
```

Ý nghĩa:

- Nếu `qi` gần `eb`, khoảng cách nhỏ, nên `l_i_b` ít âm hơn, điểm cao hơn.
- Nếu `qi` xa `eb`, khoảng cách lớn, nên `l_i_b` âm mạnh hơn, điểm thấp hơn.

Ví dụ:

```text
distance(qi, e1) = 0.2  -> l_i_1 = -0.2
distance(qi, e2) = 3.0  -> l_i_2 = -3.0
```

Centroid `e1` có điểm cao hơn vì gần `qi` hơn.

## 4. Soft assignment

Thay vì chọn cứng một centroid duy nhất, RelGT++ dùng soft assignment:

```text
s_i_b = exp((l_i_b + g_i_b) / tau)
        / sum_over_b_prime exp((l_i_b_prime + g_i_b_prime) / tau)
```

Trong đó:

- `s_i_b`: xác suất hoặc trọng số gán query `i` vào centroid `b`.
- `l_i_b`: điểm gần centroid.
- `g_i_b`: Gumbel noise.
- `tau`: nhiệt độ.

Soft assignment có nghĩa là một query có thể phân bổ trọng số cho nhiều centroid:

```text
s_i = [0.70, 0.20, 0.08, 0.02]
```

Thay vì hard assignment:

```text
s_i = [1, 0, 0, 0]
```

Soft assignment giúp gradient truyền được đến nhiều centroid hơn, làm quá trình học mềm và ổn định hơn.

## 5. Gumbel noise là gì?

Gumbel noise được sinh như sau:

```text
U_i_b ~ Uniform(0, 1)
g_i_b = -log(-log(U_i_b))
```

Nó được cộng vào logit trước softmax:

```text
logit_noisy = l_i_b + g_i_b
```

Ý nghĩa:

```text
Thêm nhiễu có kiểm soát để mô hình khám phá nhiều centroid hơn.
```

Nếu không có Gumbel noise, mô hình có thể nhanh chóng chọn một vài centroid dễ dùng nhất. Khi một centroid được dùng nhiều, nó nhận nhiều gradient hơn, càng trở nên tốt hơn, rồi lại được dùng nhiều hơn. Vòng lặp này dẫn đến collapse.

Gumbel noise phá vòng lặp đó bằng cách thỉnh thoảng cho các centroid khác cơ hội được chọn.

## 6. Vai trò của nhiệt độ tau

Soft assignment dùng nhiệt độ `tau`:

```text
s_i_b = softmax((l_i_b + g_i_b) / tau)
```

Nếu `tau` lớn:

```text
assignment mềm hơn
nhiều centroid cùng nhận trọng số
khám phá nhiều hơn
```

Nếu `tau` nhỏ:

```text
assignment sắc hơn
một vài centroid nhận trọng số lớn
gần giống hard assignment
```

RelGT++ anneal nhiệt độ:

```text
tau_next = max(tau_current * (1 - r), tau_min)
```

Ý nghĩa:

- Đầu training: `tau` lớn, mô hình khám phá nhiều centroid.
- Cuối training: `tau` nhỏ dần, assignment rõ ràng hơn.
- `tau_min` tránh việc assignment quá cứng quá sớm.

Đây là chiến lược rất hợp lý:

```text
đầu tiên khám phá
sau đó chuyên môn hóa
```

## 7. Entropy để chống collapse

RelGT++ đo mức sử dụng codebook bằng entropy chuẩn hóa.

Trước hết tính tần suất dùng centroid:

```text
p_b = mức sử dụng trung bình của centroid b
```

Entropy:

```text
entropy = - sum_b p_b * log(p_b)
```

Entropy lớn nhất khi mọi centroid được dùng đều:

```text
p = [1/B, 1/B, ..., 1/B]
```

Entropy chuẩn hóa:

```text
normalized_entropy = entropy / log(B)
```

Loss chống collapse:

```text
L_entropy = 1 - normalized_entropy
```

Nếu centroid được dùng đều:

```text
normalized_entropy = 1
L_entropy = 0
```

Nếu collapse vào một centroid:

```text
normalized_entropy gần 0
L_entropy gần 1
```

Vì vậy, minimize `L_entropy` sẽ khuyến khích mô hình dùng codebook đều hơn.

## 8. Tại sao GumbelSoftCodebook thành công?

GumbelSoftCodebook thành công vì nó giải quyết ba vấn đề cùng lúc:

### 8.1. Tránh chọn centroid quá cứng

Hard assignment làm gradient chỉ đi vào centroid được chọn. Các centroid khác ít được cập nhật, dễ chết.

Soft assignment giúp nhiều centroid nhận gradient:

```text
s_i = [0.60, 0.25, 0.10, 0.05]
```

Nhờ vậy codebook học ổn định hơn.

### 8.2. Khuyến khích khám phá

Gumbel noise làm assignment không bị khóa quá sớm vào vài centroid đầu tiên.

```text
logit_noisy = distance_score + random_gumbel_noise
```

Nhiễu này giúp centroid ít được dùng vẫn có cơ hội nhận gradient và học.

### 8.3. Cân bằng giữa khám phá và khai thác

Annealing nhiệt độ giúp quá trình học đi theo hướng:

```text
early training: explore nhiều centroid
late training: exploit centroid phù hợp nhất
```

Nhờ đó, mô hình vừa tránh collapse lúc đầu, vừa có assignment rõ ràng lúc sau.

### 8.4. Có loss đo trực tiếp mức collapse

Entropy loss tạo áp lực rõ ràng để codebook được dùng đều hơn:

```text
L_entropy thấp  -> codebook dùng đa dạng
L_entropy cao   -> codebook có nguy cơ collapse
```

Điều này biến "dùng codebook đa dạng" thành một mục tiêu huấn luyện trực tiếp.

---

# 4.7. CrossAttentionBridge

## 1. Bài toán cần giải quyết

RelGT có hai loại biểu diễn quan trọng:

- `h_local`: biểu diễn từ local neighborhood.
- `h_global`: biểu diễn từ global/codebook context.

RelGT gốc thường nối hai vector:

```text
h = concat(h_local, h_global)
```

Sau đó đưa qua một lớp linear hoặc MLP.

Cách này đơn giản, nhưng có hạn chế:

```text
local và global bị trộn một lần, thiếu cơ chế hỏi-đáp giữa hai nguồn.
```

Nói cách khác, concat chỉ đặt hai vector cạnh nhau. Nó không trực tiếp học:

- local nên lấy bao nhiêu thông tin từ global?
- global nên lấy bao nhiêu thông tin từ local?
- khi nào local đáng tin hơn global?
- khi nào global nên điều chỉnh local?

CrossAttentionBridge giải quyết vấn đề này bằng bridge hai chiều.

## 2. Local nhận thông tin từ global

Đầu tiên, RelGT++ tính mức ảnh hưởng của global lên local:

```text
score_local_to_global =
dot(Wl_q * h_l, Wg_k * h_g) / sqrt(d)
```

Sau đó đưa qua sigmoid:

```text
alpha = sigmoid(score_local_to_global)
```

Rồi cập nhật local:

```text
h_l_tilde = h_l + alpha * Wg_v * h_g
```

Ý nghĩa:

- `Wl_q * h_l`: local đặt câu hỏi.
- `Wg_k * h_g`: global cung cấp key để so khớp.
- `alpha`: mức độ local nên nhận thông tin từ global.
- `Wg_v * h_g`: nội dung global truyền sang local.

Nếu `alpha` lớn:

```text
local cần nhiều thông tin global
```

Nếu `alpha` nhỏ:

```text
local giữ chủ yếu thông tin của chính nó
```

## 3. Global nhận thông tin từ local

Bridge chiều ngược lại:

```text
score_global_to_local =
dot(Wg_q * h_g, Wl_k * h_l) / sqrt(d)
```

```text
beta = sigmoid(score_global_to_local)
```

```text
h_g_tilde = h_g + beta * Wl_v * h_l
```

Ý nghĩa:

- Global cũng được phép nhìn lại local.
- Nếu local chứa tín hiệu cụ thể quan trọng, global representation được điều chỉnh theo local.
- Đây không phải concat một chiều, mà là trao đổi thông tin hai chiều.

## 4. Gate để trộn local và global sau bridge

Sau khi có hai vector đã được cập nhật:

```text
h_l_tilde
h_g_tilde
```

RelGT++ tính gate:

```text
gamma = sigmoid(W_gate * concat(h_l_tilde, h_g_tilde))
```

Sau đó trộn hai nguồn:

```text
mixed = gamma elementwise_mul h_l_tilde
        + (1 - gamma) elementwise_mul h_g_tilde
```

Cuối cùng:

```text
h_bridge = LayerNorm(W_out * mixed + h_l)
```

Ý nghĩa:

- Nếu `gamma` gần 1, mô hình ưu tiên local.
- Nếu `gamma` gần 0, mô hình ưu tiên global.
- Nếu `gamma` nằm giữa, mô hình trộn cả hai.

Vì `gamma` là vector, việc ưu tiên local/global diễn ra theo từng chiều ẩn, không phải một quyết định thô cho cả vector.

## 5. Vì sao không chỉ concat?

Concat:

```text
h = MLP(concat(h_local, h_global))
```

Cách này có thể học trộn local/global, nhưng thiếu ba điểm:

### 5.1. Không có trao đổi hai chiều rõ ràng

Concat chỉ ghép hai vector. CrossAttentionBridge cho phép:

```text
local hỏi global
global hỏi local
```

Điều này làm fusion có cấu trúc hơn.

### 5.2. Không có độ tin cậy động rõ ràng

CrossAttentionBridge có `alpha`, `beta`, `gamma`.

```text
alpha = local cần global bao nhiêu
beta  = global cần local bao nhiêu
gamma = cuối cùng tin local hay global hơn ở từng chiều
```

Các gate này thay đổi theo input, nên mô hình linh hoạt hơn concat cố định.

### 5.3. Giữ residual local

Kết quả cuối có:

```text
h_bridge = LayerNorm(W_out * mixed + h_l)
```

Nghĩa là biểu diễn local gốc vẫn được giữ qua residual. Điều này quan trọng vì local context thường là tín hiệu gần seed node nhất và trực tiếp nhất.

## 6. Tại sao CrossAttentionBridge thành công?

CrossAttentionBridge thành công vì nó giải quyết tốt xung đột giữa local và global.

Trong relational graph:

- Local context chứa tín hiệu cụ thể, gần seed node.
- Global context chứa mẫu tổng quát, prototype hoặc thông tin toàn cục.

Hai nguồn này bổ sung nhau nhưng không phải lúc nào cũng đáng tin như nhau.

Ví dụ:

- Với một customer có hành vi gần đây rất rõ, local nên được ưu tiên.
- Với một customer ít dữ liệu, global prototype có thể giúp bổ sung thông tin.
- Với dữ liệu nhiễu, global có thể làm trơn local.
- Với trường hợp đặc biệt, local có thể điều chỉnh global.

CrossAttentionBridge cho phép mô hình quyết định động:

```text
khi nào nghe local
khi nào nghe global
khi nào cần cả hai
```

Đây là điểm khiến bridge tốt hơn concat đơn giản.

## 7. Tóm tắt ba module hoạt động cùng nhau

Ba module 4.5, 4.6, 4.7 bổ sung cho nhau:

| Module | Làm gì? | Vì sao hiệu quả? |
|---|---|---|
| Multi-View Readout | Tổng hợp attention, mean, max | Bắt cả tín hiệu chọn lọc, trung bình và nổi bật hiếm |
| GumbelSoftCodebook | Học global prototype mềm có Gumbel noise | Tránh codebook collapse, tăng đa dạng centroid |
| CrossAttentionBridge | Trộn local và global bằng bridge hai chiều | Cho local/global trao đổi và chọn nguồn đáng tin theo input |

Luồng trực giác:

```text
Multi-View Readout
-> tạo local representation giàu thông tin

GumbelSoftCodebook
-> tạo global representation đa dạng, ít collapse

CrossAttentionBridge
-> kết hợp local và global một cách động, có chọn lọc
```

## 8. Đoạn diễn giải có thể đưa vào report

RelGT++ mở rộng phần tổng hợp biểu diễn bằng ba module bổ trợ. Multi-View Readout thay attention pooling đơn lẻ bằng ba góc nhìn gồm attention, mean và max. Attention giúp chọn lọc các lân cận quan trọng, mean giữ thông tin trung bình của toàn bộ neighborhood, còn max giữ các tín hiệu nổi bật hiếm. Nhờ đó, biểu diễn local không bị phụ thuộc vào một cơ chế pooling duy nhất.

GumbelSoftCodebook được dùng để cải thiện biểu diễn global và tránh codebook collapse. Thay vì gán cứng query vào một centroid, module này dùng soft assignment với Gumbel noise. Gumbel noise khuyến khích khám phá nhiều centroid hơn, còn annealing nhiệt độ giúp quá trình học chuyển dần từ khám phá sang chuyên môn hóa. Entropy loss tiếp tục khuyến khích các centroid được sử dụng đều, làm codebook đa dạng và ổn định hơn.

Cuối cùng, CrossAttentionBridge thay phép nối local-global đơn giản bằng một cơ chế trao đổi hai chiều. Local có thể nhận thông tin từ global thông qua hệ số `alpha`, global có thể nhận lại thông tin từ local thông qua hệ số `beta`, sau đó gate `gamma` quyết định từng chiều ẩn nên ưu tiên local hay global. Cơ chế này thành công vì nó cho phép mô hình kết hợp tín hiệu cụ thể từ neighborhood và tín hiệu tổng quát từ global context một cách động theo từng input.
