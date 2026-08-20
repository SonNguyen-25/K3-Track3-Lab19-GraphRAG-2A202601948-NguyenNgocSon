# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Nguyễn Ngọc Sơn
**Khóa học:** AI20K-K3 · Track 3: GraphRAG
**Ngày thực hiện:** 2026-08-19

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)
> **Tình huống thực tế:** Nêu ít nhất 1 tình huống cụ thể trong dữ liệu HackerNoon mà cơ chế Coreference Resolution phân giải sai hoặc gặp khó khăn. Hậu quả của nó đối với Knowledge Graph là gì?

*Trả lời:*
- **Cách tìm:** `coref_df`/`extraction_source["resolved_text"]` không được lưu cố định trong file `.ipynb` (chỉ tồn tại trong bộ nhớ kernel lúc notebook chạy), nên để có ví dụ thật, mình chạy trực tiếp `resolve_coref_batch()` (model `allam-2-7b`, batch size 5) trên 60 chunk dài nhất của `data/hackernoon_subset.csv` và so sánh `text` gốc với `resolved_text`.
- **Ví dụ từ dữ liệu (chunk_id thật):**
  - `chunk_id = 671858229f9ac56dbcf2::c0000` — văn bản gốc: *"N obody talks about it but amateur radio guys are the backbone of American society. A recent example of such a guy is Doug an amateur ham radio operator who over Memorial Day weekend managed to get in contact with the International Space Station..."*
  - `chunk_id = acf645a2f67f7c379fcc::c0000` — văn bản gốc (chủ đề hoàn toàn khác — HR/IT automation): *"The result is that the more the platform is used the better it gets for everyone Schoenfelder says. HR and IT alike benefit from automation..."*
  - Sau khi qua `resolve_coref_batch()` (2 chunk này nằm cùng 1 batch 5 chunk), `resolved_text` của **`acf645a2f67f7c379fcc::c0000`** trả về lại là: *"Nobody talks about it but amateur radio guys are the backbone of American society. A recent example of such a guy is Doug, an amateur ham radio operator who managed to get in contact with the International Space Station and talk to astronaut Woody Hoburg..."* — **chính là nội dung của chunk `671858229f9ac56dbcf2`**, không liên quan gì đến bài HR/IT gốc.
- **Hiện tượng:** Đây **không phải** lỗi phân giải đại từ thông thường (kiểu "the company" trỏ nhầm công ty) mà là lỗi **lẫn nội dung giữa các chunk trong cùng 1 batch JSON** — model (`allam-2-7b`, kích thước nhỏ hơn) khi phải trả về JSON cho 5 chunk cùng lúc đã gán nhầm `resolved_text` của chunk B thành nguyên văn của chunk A trong cùng batch. Trong 60 chunk test (12 batch), hiện tượng này xảy ra ở **ít nhất 2/60 chunk (~3%)** — không hiếm.
- **Hậu quả đối với Graph:** Vì bước NER+RE (cell 2.1) trích xuất quan hệ dựa trên `resolved_text` (ưu tiên hơn `text` gốc — xem `extract_batch()`: `getattr(r, "resolved_text", None) or r.text`), nếu chunk `acf645a2f67f7c379fcc` (nói về HR/IT automation) lọt vào batch extraction với `resolved_text` đã bị thay bằng nội dung "amateur radio/International Space Station", các thực thể/quan hệ được trích ra sẽ hoàn toàn sai chủ đề — nhưng vẫn được gắn `source_chunk_id = acf645a2f67f7c379fcc` làm provenance, tạo ra **False Edge có nguồn trích dẫn trông hợp lệ nhưng nội dung nguồn thực tế không khớp**. Đây là loại lỗi nguy hiểm hơn lỗi coref thông thường vì khó phát hiện bằng mắt thường (citation vẫn trỏ đúng chunk_id, chỉ có nội dung bên trong đã bị tráo).
- **Đề xuất khắc phục:** Sau khi nhận response từ `resolve_coref_batch()`, thêm kiểm tra tương đồng tối thiểu giữa `text` và `resolved_text` cho mỗi `chunk_id` (vd Jaccard trên tập từ khóa danh từ riêng) — nếu độ tương đồng quá thấp, coi là lỗi và fallback về `text` gốc thay vì tin tưởng `resolved_text` một cách vô điều kiện như code hiện tại.

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

> Phần này được viết lại dựa trên đúng những gì đã xảy ra trong buổi thực hành (mọi lỗi, quyết định, số liệu đều là thật — không dựng kịch bản), theo góc nhìn của người trực tiếp theo dõi và ra quyết định ở từng bước cùng AI Coding Agent.

### 1. Mapping Bài giảng vào Code
| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|-----------------------------|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()` | Trong dataset thực tế (HackerNoon `description`, ~150-300 ký tự/bài), phần lớn chunk chỉ có 1 câu nên coref gần như "rảnh việc" — module được thiết kế đúng cho văn bản dài hơn những gì dataset thật cung cấp. Đây là bài học về việc kiểm tra giả định độ dài dữ liệu đầu vào trước khi tin tưởng 1 bước xử lý có tác dụng thật. |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Cơ chế allowlist (chỉ nhận `Company/Person/Technology` và 8 loại quan hệ cố định) hoạt động đúng như thiết kế — không có noise dạng node/relation ngoài schema lọt vào Neo4j dù dùng tới 3 model LLM khác nhau (`allam-2-7b`, `gpt-oss-safeguard-20b`, `gpt-oss-20b`) cho các giai đoạn khác nhau. Đây là điểm mạnh rõ ràng của thiết kế "closed vocabulary" so với để LLM tự do sinh loại quan hệ. |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | `MERGE` theo `id` giúp việc chạy pipeline nhiều lần (bắt buộc phải làm lại do hết quota Groq giữa chừng — xem mục 2) không tạo node/edge trùng lặp — tính idempotent này là lý do duy nhất giúp việc chia nhỏ 3 giai đoạn và chạy lại nhiều lần vẫn ra một đồ thị nhất quán (263 nodes / 199 edges) thay vì hỗn loạn. |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, `UF` | Cơ chế hoạt động đúng thiết kế nhưng **có lỗ hổng thật**: ca gộp nhầm "Standalone 5G Technology" / "Non-standalone 5G Technology" (xem Phần 1 mục 2) cho thấy guard dựa trên string similarity (`SequenceMatcher`) không đủ để chặn các cặp có tiền tố phủ định ngữ nghĩa. Bài học: entity resolution cho dữ liệu kỹ thuật (technology name) rủi ro cao hơn company name vì các biến thể "Non-X" / "X" rất phổ biến và dễ đánh lừa string-based guard. |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()` | Cơ chế này **chưa từng được kích hoạt thực tế** trong toàn bộ 50 câu golden (`graph_supernode_events = 0` mọi câu) vì node bậc cao nhất trong đồ thị chỉ đạt degree = 6, cách xa ngưỡng `SUPER_NODE_DEGREE = 100`. Đây là hệ quả trực tiếp của việc phải thu hẹp scope xuống 500/2105 chunks do giới hạn quota — cho thấy cơ chế chống bùng nổ context chỉ thật sự cần thiết khi extract đủ dữ liệu để các entity phổ biến (Microsoft, OpenAI...) tích lũy đủ số cạnh. |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()` | Judge (GPT-4o-mini qua OpenAI, tách biệt hoàn toàn khỏi các model Groq dùng để trả lời) cho điểm nhất quán và rationale có căn cứ rõ ràng — vd với câu G5000-35, judge chỉ ra chính xác GraphRAG "does not mention AWS's tentative sourcing decision... provides irrelevant information" thay vì chỉ chấm điểm số suông. Việc tách provider giữa answer-generation và judge giúp tránh thiên vị (self-preference bias) nếu dùng cùng 1 model cho cả hai vai trò. |

---

### 2. Quá trình Debugging & Bài học
- **Lỗi kỹ thuật phức tạp nhất gặp phải:** Không phải một lỗi đơn lẻ mà là một **chuỗi 7 lần chạy thất bại liên tiếp** trước khi pipeline chạy trọn vẹn, mỗi lần thất bại lộ ra một tầng vấn đề khác nhau:
  1. Notebook viết cho Google Colab, không chạy được trên máy local (`/content/...` path cứng, thiếu `load_dotenv()`, `DATA_PATH` không tồn tại).
  2. File credentials Neo4j Aura tải về dùng biến `NEO4J_USERNAME` nhưng code đọc `NEO4J_USER` — và quan trọng hơn, instance Aura loại mới dùng chính **Instance ID** làm username lẫn database name, không phải `"neo4j"` mặc định như tài liệu cũ vẫn ghi — khiến 2 lần đầu chẩn đoán sai hướng.
  3. `GROQ_MODEL` mặc định trong bài (`llama-3.3-70b-versatile`) **đã bị Groq gỡ bỏ khỏi danh sách model khả dụng** — lỗi 404 bị nuốt bởi retry logic có sẵn trong code gốc, khiến cell chạy đủ 30 phút timeout mà không tạo ra được 1 quan hệ nào, và ban đầu bị nhầm tưởng là do mạng chậm.
  4. Sau khi đổi model, phát hiện Groq free-tier giới hạn **theo ngày** (200.000 token/ngày hoặc 250 request/ngày, tuỳ model) chứ không phải theo phút — 1 model dùng nhiều trong lúc test/debug sẽ cạn quota trước khi pipeline chính chạy tới, và cạn quota của 1 model không được thông báo trước, chỉ lộ ra giữa chừng qua lỗi 429.
  5. Model nhỏ hơn (`openai/gpt-oss-20b`, `groq/compound-mini`) đôi khi trả JSON không đúng schema Groq tự yêu cầu (`items` chứa string thay vì object) — code gốc không có `isinstance()` guard nên 1 batch lỗi làm sập toàn bộ vòng lặp trích xuất.
  6. `jupyter nbconvert --execute` là "tất cả hoặc không gì cả": nếu bất kỳ cell nào lỗi, **toàn bộ output của các cell đã chạy thành công trước đó cũng bị mất** (không được ghi ra file `.ipynb`) — nghĩa là 1 lỗi ở bước eval (câu hỏi cuối cùng) làm mất luôn kết quả extraction+ingest đã tốn 20-30 phút để có được.
- **Cách xử lý thành công:** Ứng với từng lớp lỗi ở trên: (1) thêm `load_dotenv()` + sửa path tương đối; (2) sửa lại đúng tên biến và giá trị thật của Aura instance; (3) đổi sang model còn khả dụng sau khi liệt kê `client.models.list()`; (4) **chia 3 giai đoạn nặng nhất (coref/extraction/eval) ra 3 model Groq riêng biệt** để mỗi giai đoạn dùng ngân sách quota độc lập, đồng thời giảm `EXTRACTION_MAX_CHUNKS` xuống mức vừa đủ để hoàn thành trong 1 lần chạy; (5) thêm `isinstance(x, dict)` guard ở mọi nơi code parse JSON từ LLM; (6) viết thêm 1 script Python độc lập (không qua notebook kernel) để **resume đánh giá từ checkpoint CSV đã lưu**, tránh phải chạy lại từ đầu mỗi khi 1 model hết quota giữa vòng lặp 50 câu — đây là thay đổi có tác động lớn nhất, giúp hoàn thành 44/50 câu còn lại chỉ trong ~14 phút thay vì phải chạy lại toàn bộ pipeline (~1-2 giờ) từ đầu.

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

*Đối chiếu với đồ án đang làm: **Campus 24/7** — trợ lý ảo sinh viên đại học (FastAPI + LangGraph + MongoDB), phục vụ 3 vai trò sinh viên/cán bộ/quản trị viên, kết hợp RAG (hỏi-đáp có trích nguồn quy định) + Agent tool-use (tác vụ có xác nhận: đăng ký, xin nghỉ học, đặt lịch...) + Human-in-the-loop (chuyển tiếp cán bộ với ca nhạy cảm).*

- **Tên đồ án / Dự án:** Campus 24/7 — VinUni AI20K Build Phase.

- **Đặc thù bài toán & Lý do chọn giải pháp — về cơ bản là KHÔNG DÙNG GraphRAG cho phần lõi:**
  - Nhu cầu retrieval chính của Campus 24/7 là **Q&A trên văn bản quy định/quy chế nhà trường** (theo `ARCHITECTURE.md`: "Retrieve theo văn bản quy định" → "Đủ nguồn tin cậy?" → trả lời có trích nguồn). Đây là bài toán FAQ/document-QA cổ điển: câu hỏi kiểu "thủ tục xin nghỉ học tạm thời cần giấy tờ gì" thường tự chứa đủ thông tin trong 1-2 đoạn văn bản quy định, **không cần multi-hop qua nhiều thực thể** như các câu hỏi cross-doc trong lab này (vd truy vết một thương vụ M&A qua 3 bài báo khác thời điểm).
  - Các "thực thể" cốt lõi của hệ thống (`User`, `StudentProfile`, `Department`, `Workflow`, `FormDefinition`, `Task`, `Submission`, `Ticket`) **đã là dữ liệu có cấu trúc sẵn trong MongoDB** với quan hệ tường minh (foreign key qua `_id`), không phải văn bản thô cần NER+RE để trích xuất như dataset HackerNoon của lab này. Việc dựng thêm 1 pipeline coref→NER→entity-resolution→Neo4j cho dữ liệu vốn đã có cấu trúc là dư thừa — MongoDB + Pydantic model hiện tại đã đóng vai trò "graph" đó rồi (quan hệ 1-nhiều/nhiều-nhiều qua reference field).
  - → **Kết luận: Flat RAG (vector search, ChromaDB theo đề xuất trong `ARCHITECTURE.md`) là đủ** cho tầng retrieval văn bản quy định. Không cần Neo4j/GraphRAG cho use-case hiện tại.

- **Trường hợp CÓ THỂ cân nhắc dùng ý tưởng của GraphRAG (không phải toàn bộ pipeline, chỉ mượn kỹ thuật) — không nằm trong kế hoạch hiện tại, chỉ ghi nhận để biết giới hạn:**
  - Nếu về sau xuất hiện câu hỏi dạng **đa bước xuyên phòng ban** (vd "sinh viên khoa CS muốn chuyển sang khoa Business thì cần hoàn thành workflow nào trước, ai duyệt, và có ảnh hưởng gì đến học bổng đang nhận") — đây là multi-hop thật sự qua `Department → Workflow → Form → Task/Submission → ConfirmationToken`, đúng dạng bài toán GraphRAG giải quyết tốt.
  - Nhưng vì các quan hệ này **đã tường minh trong MongoDB** (không cần trích xuất bằng LLM), nếu cần, cách làm đúng là **query trực tiếp bằng MongoDB aggregation pipeline** (`$lookup` nhiều tầng) hoặc đồng bộ sang Neo4j để traversal nhanh hơn — chứ không cần lại pha "NER+RE extraction" của lab này, vì dữ liệu đầu vào không phải văn bản phi cấu trúc.

- **Cấu trúc Node & Relation dự kiến:** *Không áp dụng cho use-case retrieval hiện tại (dùng Flat RAG).* Nếu về sau mở rộng sang multi-hop cross-department như trên, cấu trúc gợi ý:
  - Nodes: `Department`, `Workflow`, `FormDefinition`, `Task` (đã tồn tại sẵn dạng document MongoDB, không cần LLM extraction).
  - Relations: `BELONGS_TO` (Workflow→Department), `REQUIRES_FORM` (Workflow→FormDefinition), `DEPENDS_ON` (Workflow→Workflow, cho case liên phòng ban), `APPROVED_BY` (Task→User/staff).

- **Chiến lược xử lý Super-node & Entity Resolution:** *Không áp dụng trực tiếp* — vì entity ở đây (Department, Workflow...) đã có `_id` duy nhất từ MongoDB, không có bài toán "gộp nhầm 2 cách viết của cùng 1 thực thể" như dữ liệu văn bản tự do. Rủi ro super-node duy nhất có thể xảy ra là 1 `Department` lớn (vd "Phòng Đào tạo") liên kết với rất nhiều `Workflow` — nếu cần, áp dụng nguyên lý tương tự bài lab (ưu tiên N workflow gần đây nhất/còn hiệu lực) khi trả kết quả cho agent, tránh nhét toàn bộ danh sách workflow của 1 phòng ban vào context.

---

## 🎯 TỰ ĐÁNH GIÁ
| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4 | |
| Khả năng kiểm soát AI Coding Agent | 4 | |
| Chất lượng đồ thị tri thức xây dựng | 5 ||
| Khả năng phân tích và debug hệ thống | 5 ||
