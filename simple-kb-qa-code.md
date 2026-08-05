好，给你一个**标准版单类实现**，直接对接真实的向量模型和LLM，但所有逻辑都浓缩在一个类里，极简可控。

---

## 完整代码 `kb.py`

[代码](simple-kb-qa-code.py)

## 依赖文件 `requirements.txt`

```txt
fastapi==0.110.0
uvicorn[standard]==0.27.0
chromadb==0.4.22
sentence-transformers==2.2.2
openai==1.13.0
PyPDF2==3.0.1
python-docx==1.1.0
pydantic==2.5.0
python-dotenv==1.0.0
```


## 环境变量 `.env`

```env
LLM_API_KEY=sk-your-deepseek-api-key
```


## 启动步骤

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置API Key（二选一）
# 方式一：环境变量
export LLM_API_KEY=sk-xxx

# 方式二：创建.env文件
echo "LLM_API_KEY=sk-xxx" > .env

# 3. 启动（首次会下载bge-m3模型 ~2GB）
python kb.py
```


## API测试

```bash
# 上传文档
curl -X POST http://localhost:8000/upload -F "file=@文档.pdf"

# 智能问答（检索+LLM）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"产品怎么退款？", "top_k":4}'

# 纯检索（不含LLM）
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"question":"退款政策", "top_k":3}'

# 统计
curl http://localhost:8000/stats
```


## 类方法速查

| 方法 | 参数 | 返回 | 说明 |
|---|---|---|---|
| `upload()` | content, filename | `{doc_id, filename, chunks}` | 上传并索引文档 |
| `search()` | question, top_k | `[{content, filename, similarity}]` | 纯向量检索 |
| `chat()` | question, top_k | `{question, answer, sources, total_tokens}` | **主入口**：检索+LLM生成 |
| `list_docs()` | - | `[{doc_id, filename, chunks}]` | 列出所有文档 |
| `delete()` | doc_id | `bool` | 删除文档 |
| `stats()` | - | `{total_chunks, total_documents, documents}` | 统计信息 |

---

**一个类，所有功能，开箱即用。** 后续扩展只需替换Embedding模型或LLM配置即可。
