# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** [Họ và Tên] — *TODO: điền tên bạn*
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 2026-08-20

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)
> **Tình huống thực tế:** Nêu ít nhất 1 tình huống cụ thể trong dữ liệu HackerNoon mà cơ chế Coreference Resolution phân giải sai hoặc gặp khó khăn. Hậu quả của nó đối với Knowledge Graph là gì?

*Trả lời:*
- **Ghi chú minh bạch:** `coref_df` (kết quả của `run_coref()`) chỉ tồn tại trong bộ nhớ khi notebook đang chạy, không được lưu lại thành output cố định trong file `.ipynb`. Vì vậy phần này **cần bạn tự chạy lại và quan sát** — không nên dùng ví dụ bịa. Cách tìm nhanh một ca lỗi thật:
  ```python
  # Chạy sau khi coref_df đã có (cell 1.7), so sánh text gốc và text đã resolve
  diffs = extraction_source[extraction_source.text != extraction_source.resolved_text]
  display(diffs[["chunk_id", "text", "resolved_text", "unresolved_mentions"]].head(10))
  ```
  Vì mô tả bài báo trong dataset này khá ngắn (~150-300 ký tự/chunk, xem mục "Đặc thù dữ liệu" bên dưới), phần lớn mỗi chunk chỉ chứa 1 câu duy nhất — nên coreference giữa nhiều câu (pronoun trỏ về công ty ở câu trước) **hiếm khi xảy ra** trong tập dữ liệu đã lọc. Đây bản thân là một quan sát kỹ thuật đáng ghi nhận: cơ chế coref được thiết kế cho văn bản dài (bài báo đầy đủ), nhưng dataset thực tế cấp qua HuggingFace chỉ có trường `description` (tóm tắt ngắn), không có toàn văn — nên module này gần như "rảnh rỗi" ở quy mô hiện tại. Nếu chạy lại ở quy mô lớn hơn hoặc nối `title + description` thành nhiều câu, khả năng xuất hiện ca lỗi thật sẽ tăng.
- **Hiện tượng cần điền:** [Sau khi chạy đoạn code trên, mô tả 1 dòng cụ thể]
- **Hậu quả đối với Graph:** [Điền dựa trên ví dụ thật bạn tìm được]

---

### 2. Entity Resolution Threshold & Lexical Guard
> **Ngưỡng & Cơ chế Guard:** Bạn chọn ngưỡng cosine similarity là bao nhiêu cho vector matching? Trích dẫn 1 cặp thực thể có độ tương đồng vector cao ($> 0.85$) nhưng bị Lexical Guard chặn không cho gộp (Reject) và giải thích lý do.

*Trả lời:*
- **Ngưỡng cosine similarity:** `threshold = 0.90` (giá trị mặc định trong `build_resolution_map()`), `merge_guard()` dùng thêm `SequenceMatcher.ratio() >= 0.72` làm lớp chặn thứ hai sau khi bỏ suffix công ty (Inc, Corp, LLC...).
- **Quan sát thực tế quan trọng hơn ví dụ mẫu:** Ở lần chạy cuối, `entity_resolution_audit_df` của riêng lượt extraction cuối cùng bị **rỗng** (0 dòng) — thấp hơn yêu cầu tối thiểu 10 dòng của checklist. Nguyên nhân: với ngân sách 500 chunks (do giới hạn quota Groq free-tier, xem mục 5), số lượng mention trùng lặp/giống nhau đủ để kích hoạt kiểm tra vector là rất ít.
- **Bằng chứng gộp nhầm thực tế tìm thấy khi kiểm tra trực tiếp đồ thị Neo4j** (không phải audit log của 1 lần chạy, mà là alias tích lũy trên node):
  ```
  Node: "Standalone 5G Technology" (Technology)
  aliases: ["Non-standalone 5G Technology", "Standalone 5G Technology"]
  ```
  Đây là **một ca lỗi gộp nhầm thật** (ngược lại với ví dụ template yêu cầu — nhưng đáng phân tích hơn): "Standalone 5G" (SA) và "Non-standalone 5G" (NSA) là **hai kiến trúc mạng 5G khác nhau về mặt kỹ thuật** (SA dùng lõi mạng 5G độc lập, NSA dựa vào lõi 4G LTE hiện có) — không phải cùng một thực thể. Sự tương đồng chuỗi ký tự cao ("...standalone 5G Technology") đã vượt ngưỡng `SequenceMatcher.ratio() >= 0.72` dù về ngữ nghĩa là hai khái niệm đối lập nhau (có tiền tố phủ định "Non-").
- **Lý do gộp nhầm:** `merge_guard()` chỉ dùng string similarity (Levenshtein-based) chứ không có logic phủ định ngữ nghĩa. Tiền tố "Non-" chỉ làm giảm nhẹ độ tương đồng chuỗi (do vẫn còn chung phần lớn ký tự), không đủ để bị `SequenceMatcher` coi là khác biệt đáng kể.
- **Đề xuất khắc phục:** Thêm rule-based guard chặn merge khi 1 trong 2 tên có prefix phủ định ("non-", "un-", "anti-") mà tên còn lại không có, trước khi tính cosine similarity.

---

### 3. Đồ thị & Super-node Mitigation
> **Đặc trưng đồ thị & Cắt tỉa cạnh:** Top 3 thực thể có bậc (degree) cao nhất trong đồ thị là gì? Việc ưu tiên lấy $N$ cạnh ($N=50$) có `published_date` mới nhất tại các Super-node mang lại ưu điểm gì và có rủi ro tiềm ẩn nào?

*Trả lời:*
- **Top 3 Super-nodes (dữ liệu thật từ Neo4j, truy vấn `MATCH (n:Entity) OPTIONAL MATCH (n)-[r]-() RETURN n.name, n.entity_type, count(r) ORDER BY count(r) DESC`):**

| Hạng | Tên thực thể | Loại thực thể (Type) | Bậc kết nối (Degree) |
|------|--------------|---------------------|----------------------|
| 1 | Intelligent Technical Solutions | Company | 6 |
| 2 | ServiceNow | Company | 5 |
| 3 | Walt Disney Co. | Company | 4 |

- **Quan sát quan trọng:** Với quy mô đồ thị hiện tại (263 nodes / 199 edges, dựng từ 500/2105 chunk khả dụng), **không node nào vượt ngưỡng `SUPER_NODE_DEGREE = 100`** — cơ chế cắt tỉa cạnh (`SUPER_NODE_EDGE_CAP = 50`) chưa từng được kích hoạt thực tế trong lần chạy này (`graph_supernode_events = 0` ở toàn bộ 50 câu golden). Đây là hệ quả trực tiếp của việc phải giảm ngân sách extraction do giới hạn API (xem mục 5) — ở quy mô đầy đủ (toàn bộ dataset), các thực thể lớn như "Microsoft", "OpenAI", "Amazon" chắc chắn sẽ vượt xa ngưỡng 100.
- **Ưu điểm & Rủi ro của Temporal Mitigation (dựa trên thiết kế, áp dụng khi đồ thị đủ lớn):**
  - *Ưu điểm:* Giữ context trong giới hạn `MAX_GRAPH_CONTEXT_CHARS = 14000`, tránh LLM bị "chìm" trong hàng trăm cạnh không liên quan của 1 super-node như Microsoft; ưu tiên tin mới nhất phù hợp với các câu hỏi dạng "diễn biến gần đây".
  - *Rủi ro:* Câu hỏi dạng lịch sử (vd: "Microsoft mua GitHub năm nào?") có thể bị cắt mất nếu cạnh đó không nằm trong 50 cạnh mới nhất của node — đây chính là đánh đổi Recency vs Completeness, không có cách nào tối ưu cả hai nếu không tăng `SUPER_NODE_EDGE_CAP` (đổi lại tăng chi phí token).

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge, trung bình trên 50 câu golden — dữ liệu thật từ `outputs/graphrag_vs_flatrag_summary.csv`):

| Tiêu chí đánh giá | Flat RAG | GraphRAG | Nhận xét phân tích |
|-------------------|----------|----------|-------------------|
| **Comprehensiveness — factoid (1–5)** | 2.80 | 2.80 | Hai phương pháp gần nhau |
| **Comprehensiveness — multi-hop (1–5)** | 1.78 | 1.61 | Gần nhau, cả hai đều thấp |
| **Comprehensiveness — cross-doc (1–5)** | 1.82 | 1.91 | Gần nhau |
| **Faithfulness — factoid (1–5)** | 3.00 | 2.80 | Gần nhau |
| **Multi-hop reasoning — multi-hop (1–5)** | 1.74 | 1.61 | Gần nhau, cả hai đều thấp |
| **Latency trung bình (s), cả 3 nhóm** | ~5.0–5.2 | ~3.9–4.4 | **GraphRAG nhanh hơn Flat RAG ở mọi nhóm câu hỏi** |
| **Token usage trung bình (multi-hop)** | 913 | 1080 | GraphRAG tốn token hơn ở nhóm multi-hop (ngữ cảnh đồ thị + vector kết hợp) |

**Nhận xét tổng thể quan trọng:** Điểm chất lượng ở mức thấp-trung bình (1.6–3.0/5) cho **cả hai phương pháp**, không riêng GraphRAG. Đây không phải lỗi kiến trúc GraphRAG mà là hệ quả của quy mô dữ liệu bị thu hẹp (500/2105 chunks khả dụng, do giới hạn quota Groq free-tier — xem mục 5): nhiều thực thể trong 50 câu hỏi golden (được thiết kế cho scope 5000 dòng đầu) không có mặt trong đồ thị/index đã xây, khiến cả Flat RAG lẫn GraphRAG đều thiếu ngữ cảnh thật để trả lời đúng.

#### Phân tích 2 Ca lỗi Điển hình (trích trực tiếp từ `outputs/graphrag_eval_results.csv`):

1. **Ca lỗi Flat RAG thất bại (GraphRAG thành công) — `G5000-33`:**
   - *Câu hỏi:* "Which July OpenAI-related event is a content/technology collaboration, and which July event is a voluntary governance commitment?"
   - *Điểm số:* Flat RAG = 2/2/1 (comp/faith/multihop) — GraphRAG = 5/5/5
   - *Tại sao Flat RAG thất bại?* Flat RAG tìm ra sự kiện AP–OpenAI nhưng **ghi sai ngày** (April thay vì July) và **không tìm ra sự kiện thứ hai** (cam kết quản trị tự nguyện với Ủy ban Châu Âu) — vector search top-k=6 không đưa đủ 2 chunk liên quan vào cùng 1 lần truy vấn.
   - *GraphRAG đã giải quyết như thế nào?* Kết hợp cả graph traversal (tìm quan hệ OpenAI liên quan) lẫn vector search (k=4) trong cùng context, nên có đủ bằng chứng để xác định **cả 2** sự kiện đúng ngày, đúng loại quan hệ. Judge rationale xác nhận: "correctly states the voluntary governance commitment with the European Commission".

2. **Ca lỗi GraphRAG thất bại nặng (cả hai cùng yếu, GraphRAG tệ hơn) — `G5000-35`:**
   - *Câu hỏi:* "Contrast AWS's AMD-chip posture with HPE's AI-cloud posture. Which is a tentative hardware sourcing decision and which is a service offering?"
   - *Điểm số:* Flat RAG = 3/3/3 — GraphRAG = 1/1/1 (`graph_supernode_events = 0`, tức không phải do bị cắt cạnh)
   - *Nguyên nhân:* Câu trả lời của GraphRAG hoàn toàn lạc đề — nhắc đến "Ochsner Health", "LeanTaaS" là những thực thể **không liên quan gì đến câu hỏi**. Đây là dấu hiệu điển hình của lỗi **thiếu seed entity**: `extract_seeds()` có thể đã trích đúng "AWS"/"AMD"/"HPE" là seed, nhưng `match_seeds()` không tìm thấy các thực thể này trong đồ thị (vì AWS/AMD/HPE không nằm trong 500 chunk đã extract), nên graph traversal rơi vào nhánh fallback hoặc trả về các node gần nhất về mặt vector embedding nhưng vô nghĩa về ngữ cảnh — pha trộn với context vector-search (k=4) từ các chunk không liên quan.
   - *Đề xuất khắc phục:* (a) Thêm log rõ ràng khi `match_seeds()` trả về rỗng để phân biệt case "NO_SEED" khỏi case "seed sai"; (b) tăng ngân sách extraction để các công ty phổ biến (AWS, AMD, HPE) chắc chắn có mặt trong graph; (c) khi seed match thất bại, nên **fallback hoàn toàn về Flat RAG** thay vì trộn context nửa vời — giảm rủi ro "graph noise" làm nhiễu câu trả lời so với chỉ dùng vector search.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent
> **Trade-offs, Agent Control & Scale 350MB**

*Trả lời:*
- **Đánh đổi Quality vs Cost vs Latency (dữ liệu thật):** GraphRAG cho latency **thấp hơn** Flat RAG ở cả 3 nhóm câu hỏi (~4.0s vs ~5.1s trung bình) — ngược với giả định phổ biến rằng graph traversal luôn chậm hơn thuần vector search, vì trong lab này graph traversal chạy trên Neo4j nhỏ (263 nodes) rất nhanh, và bước sinh câu trả lời (LLM call) mới là phần tốn thời gian nhất — không phải bước truy xuất. Ngược lại, token usage của GraphRAG cao hơn ở nhóm multi-hop (do phải nhét cả context đồ thị lẫn context vector vào cùng 1 prompt).
- **Quyết định từ chối / phản biện với AI Coding Agent (TODO — điền theo trải nghiệm thật của bạn):** *[Trong suốt buổi làm bài, bạn đồng ý gần như toàn bộ đề xuất kỹ thuật của agent. Nếu muốn phần này phản ánh đúng thực tế, một số điểm bạn CÓ THỂ cân nhắc nêu ra (tự đánh giá lại, không bắt buộc theo agent):]*
  - *Việc chia nhỏ 3 giai đoạn (coref/extraction/eval) ra 3 model Groq khác nhau để né giới hạn quota — đây là một "workaround" thực dụng, không phải thiết kế chuẩn mực; bạn có đồng ý với hướng đi này hay sẽ chọn nâng cấp tài khoản Groq trả phí thay vì vậy? Ghi lại quan điểm của bạn ở đây.*
  - *Agent đã giảm `EXTRACTION_MAX_CHUNKS` từ 2200 xuống 1000 rồi 500 để né rate limit, thay vì giữ nguyên độ phủ dữ liệu đầy đủ — đây là đánh đổi coverage lấy tốc độ hoàn thành mà bạn cần xác nhận có chấp nhận được cho mục đích của bài lab hay không.*
- **Giải pháp scale 350MB (~100,000 bài báo) — dựa trên bottleneck THẬT đã gặp phải trong lab, không phải lý thuyết:** Bottleneck đầu tiên và rõ ràng nhất trong thực nghiệm **không phải** là RAM/CPU hay thuật toán entity resolution $O(N^2)$ — mà là **giới hạn quota LLM theo ngày của nhà cung cấp** (Groq free tier: 200,000 token/ngày hoặc 250 request/ngày tùy model). Với 100,000 bài báo, ước tính cần hàng trăm nghìn lượt gọi LLM cho coref+extraction — vượt xa mọi hạn mức free-tier hiện có. Giải pháp thực tế: (a) chuyển sang API trả phí (OpenAI/Groq Dev Tier) với ngân sách token được tính toán trước; (b) triển khai **worker queue phân tán qua nhiều API key/provider** để cộng dồn quota; (c) giảm chi phí bằng cách batch nhiều chunk hơn/lần gọi (đánh đổi độ chính xác NER); (d) cache kết quả extraction để không phải chạy lại toàn bộ khi retry.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

> ⚠️ **Toàn bộ Phần 2 cần bạn tự viết** — đây là phần phản ánh trải nghiệm cá nhân, được chấm riêng để đánh giá bạn (không phải AI) đã hiểu bài đến đâu. Mình để sẵn khung + gợi ý dựa trên những gì đã xảy ra thật trong buổi làm bài, bạn điền theo góc nhìn của chính mình.

### 1. Mapping Bài giảng vào Code
| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|-----------------------------|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()` | *TODO* |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | *TODO* |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | *TODO* |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, `UF` | *TODO — gợi ý: liên hệ ca lỗi "Standalone/Non-standalone 5G" ở Phần 1 mục 2* |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()` | *TODO — gợi ý: liên hệ việc cơ chế này chưa từng kích hoạt ở quy mô 263 node* |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()` | *TODO* |

---

### 2. Quá trình Debugging & Bài học
- **Lỗi kỹ thuật phức tạp nhất gặp phải:** *TODO — gợi ý các sự cố thật đã xảy ra trong buổi làm bài để bạn chọn viết theo góc nhìn của mình:*
  - *Notebook được thiết kế cho Google Colab, không chạy được thẳng trên máy local (path `/content/...`, thiếu `load_dotenv()`).*
  - *`.env` từ file credentials Neo4j Aura dùng sai format (`NEO4J_USERNAME` thay vì `NEO4J_USER`), và username/database thực tế của instance lại là Instance ID chứ không phải `"neo4j"` mặc định.*
  - *Model `GROQ_MODEL` mặc định trong bài (`llama-3.3-70b-versatile`) đã bị Groq gỡ bỏ hoàn toàn — gây lỗi 404 âm thầm bị nuốt bởi retry logic, khiến pipeline chạy 30 phút mà không extract được gì.*
  - *Groq free-tier giới hạn quota theo NGÀY (không phải theo phút) trên từng model riêng biệt — phải chia 3 giai đoạn ra 3 model khác nhau và viết script resume-by-checkpoint để hoàn thành được toàn bộ 50 câu golden.*
- **Cách bạn đã xử lý thành công:** *TODO*

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)
- **Tên đồ án / Dự án:** *TODO*
- **Đặc thù bài toán & Lý do chọn giải pháp:** *TODO — Đồ án của bạn có cần GraphRAG không, hay Flat RAG/Hybrid RAG là đủ?*
- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: *TODO*
  - Relations: *TODO*
- **Chiến lược xử lý Super-node & Entity Resolution:** *TODO*

---

## 🎯 TỰ ĐÁNH GIÁ
| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | *TODO* | |
| Khả năng kiểm soát AI Coding Agent | *TODO* | |
| Chất lượng đồ thị tri thức xây dựng | *TODO* | |
| Khả năng phân tích và debug hệ thống | *TODO* | |
