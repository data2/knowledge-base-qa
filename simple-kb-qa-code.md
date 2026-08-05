好，给你一个**标准版单类实现**，直接对接真实的向量模型和LLM，但所有逻辑都浓缩在一个类里，极简可控。

---

## 完整代码 `kb.py`

```python
"""
标准知识库 - 单类完整实现
对接真实向量模型(BAAI/bge-m3) + 大语言模型(DeepSeek API)
启动: pip install -r requirements.txt && python kb.py
"""

import os
import uuid
import json
from typing import List, Dict, Optional
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import PyPDF2
from docx import Document


class KnowledgeBase:
    """
    标准知识库 - 单类实现
    功能: 上传文档 → 向量化 → 检索 → LLM生成答案
    """
    
    def __init__(
        self,
        # Embedding配置
        embedding_model: str = "BAAI/bge-m3",
        embedding_device: str = "auto",
        # LLM配置（OpenAI兼容接口）
        llm_api_base: str = "https://api.deepseek.com/v1",
        llm_api_key: str = "",  # 从环境变量读取
        llm_model: str = "deepseek-chat",
        # RAG配置
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        top_k: int = 4,
        similarity_threshold: float = 0.5,
        # 存储路径
        persist_dir: str = "./chroma_data"
    ):
        # ---------- 1. 配置 ----------
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # ---------- 2. 向量模型(本地) ----------
        print(f"📥 加载向量模型: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model, device=embedding_device)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        print(f"   向量维度: {self.embedding_dim}")
        
        # ---------- 3. 向量数据库(本地) ----------
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="kb",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ 向量库已加载，共 {self.collection.count()} 个分块")
        
        # ---------- 4. 大语言模型 ----------
        self.llm_client = OpenAI(
            base_url=llm_api_base,
            api_key=llm_api_key or os.getenv("LLM_API_KEY", "")
        )
        self.llm_model = llm_model
        print(f"✅ LLM已配置: {llm_model}")
        
        # ---------- 5. 元数据 ----------
        self.meta_file = Path(persist_dir) / "meta.json"
        self.meta = self._load_meta()
        
        print("=" * 50)
        print("📚 知识库初始化完成")
        print(f"   文档数: {len(self.meta)}")
        print(f"   分块数: {self.collection.count()}")
        print("=" * 50)
    
    # ======================== 文档解析 ========================
    
    def _parse_pdf(self, content: bytes) -> str:
        reader = PyPDF2.PdfReader(BytesIO(content))
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    
    def _parse_docx(self, content: bytes) -> str:
        doc = Document(BytesIO(content))
        return "\n".join([p.text for p in doc.paragraphs])
    
    def _parse_txt(self, content: bytes) -> str:
        return content.decode("utf-8", errors="ignore")
    
    def _parse_file(self, content: bytes, filename: str) -> str:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return self._parse_pdf(content)
        elif ext == ".docx":
            return self._parse_docx(content)
        elif ext == ".txt":
            return self._parse_txt(content)
        else:
            raise ValueError(f"不支持的文件格式: {filename}")
    
    # ======================== 文本分块 ========================
    
    def _chunk_text(self, text: str) -> List[str]:
        """按固定长度分块，带重叠"""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks
    
    # ======================== 元数据管理 ========================
    
    def _load_meta(self) -> Dict:
        if self.meta_file.exists():
            with open(self.meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def _save_meta(self):
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
    
    # ======================== 核心方法 ========================
    
    def upload(self, content: bytes, filename: str) -> Dict:
        """上传并索引文档"""
        # 1. 解析
        text = self._parse_file(content, filename)
        if not text.strip():
            raise ValueError("文档内容为空或无法解析")
        
        # 2. 分块
        chunks = self._chunk_text(text)
        if not chunks:
            raise ValueError("分块结果为空")
        
        # 3. 生成ID
        doc_id = str(uuid.uuid4())[:8]
        
        # 4. 向量化
        print(f"🔄 向量化中: {filename} ({len(chunks)} 个分块)")
        embeddings = self.encoder.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=True
        ).tolist()
        
        # 5. 入库
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"doc_id": doc_id, "filename": filename, "chunk_index": i}
            for i in range(len(chunks))
        ]
        
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        # 6. 保存元数据
        self.meta[doc_id] = {
            "filename": filename,
            "chunks": len(chunks),
            "status": "completed"
        }
        self._save_meta()
        
        print(f"✅ 上传成功: {filename} -> {doc_id}")
        return {"doc_id": doc_id, "filename": filename, "chunks": len(chunks)}
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """检索相关文档片段（纯检索，不含LLM）"""
        if self.collection.count() == 0:
            return []
        
        top_k = top_k or self.top_k
        
        # 向量化查询
        q_vec = self.encoder.encode(query, normalize_embeddings=True).tolist()
        
        # 检索
        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化
        items = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                sim = 1 - results["distances"][0][i]
                if sim >= self.similarity_threshold:
                    items.append({
                        "content": doc,
                        "filename": results["metadatas"][0][i].get("filename", "unknown"),
                        "similarity": round(sim, 4)
                    })
        return items
    
    def chat(self, question: str, top_k: int = None) -> Dict:
        """
        智能问答：检索 + LLM生成
        这是主入口方法
        """
        # 1. 检索
        sources = self.search(question, top_k)
        
        if not sources:
            return {
                "question": question,
                "answer": "知识库中暂无相关信息，请上传相关文档后重试。",
                "sources": [],
                "total_tokens": 0
            }
        
        # 2. 构建上下文（截断过长内容）
        context = "\n\n---\n\n".join([s["content"][:2000] for s in sources])
        
        # 3. 构建Prompt
        prompt = f"""请根据以下参考资料回答问题。

参考资料：
{context}

用户问题：{question}

要求：
1. 基于参考资料给出准确答案
2. 如果资料中没有相关信息，明确说"资料中未提及"
3. 答案简洁明了，不要编造
"""
        
        # 4. 调用LLM
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "你是智能客服助手，基于参考资料回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4096
            )
            
            return {
                "question": question,
                "answer": response.choices[0].message.content,
                "sources": [
                    {"content": s["content"][:300] + "..." if len(s["content"]) > 300 else s["content"],
                     "filename": s["filename"],
                     "similarity": s["similarity"]}
                    for s in sources
                ],
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
        except Exception as e:
            return {
                "question": question,
                "answer": f"LLM调用失败: {str(e)}",
                "sources": sources,
                "total_tokens": 0
            }
    
    def delete(self, doc_id: str) -> bool:
        """删除文档"""
        ids = self.collection.get(where={"doc_id": doc_id})["ids"]
        if ids:
            self.collection.delete(ids=ids)
            if doc_id in self.meta:
                del self.meta[doc_id]
                self._save_meta()
            return True
        return False
    
    def list_docs(self) -> List[Dict]:
        """列出所有文档"""
        return [{"doc_id": k, **v} for k, v in self.meta.items()]
    
    def stats(self) -> Dict:
        """统计信息"""
        return {
            "total_chunks": self.collection.count(),
            "total_documents": len(self.meta),
            "documents": self.list_docs()
        }


# ============================================================
# FastAPI 接口
# ============================================================

app = FastAPI(title="标准知识库系统", version="1.0.0")

# 初始化知识库（从环境变量读取API Key）
kb = KnowledgeBase(
    llm_api_key=os.getenv("LLM_API_KEY", "")
)


class QueryReq(BaseModel):
    question: str
    top_k: int = 4


@app.post("/upload", summary="上传文档")
async def upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        return kb.upload(content, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/chat", summary="智能问答")
async def chat(req: QueryReq):
    return kb.chat(req.question, req.top_k)


@app.post("/search", summary="纯检索（不含LLM）")
async def search(req: QueryReq):
    return {"question": req.question, "results": kb.search(req.question, req.top_k)}


@app.get("/documents", summary="文档列表")
async def list_docs():
    return kb.list_docs()


@app.delete("/documents/{doc_id}", summary="删除文档")
async def delete_doc(doc_id: str):
    return {"doc_id": doc_id, "deleted": kb.delete(doc_id)}


@app.get("/stats", summary="统计信息")
async def stats():
    return kb.stats()


@app.get("/health", summary="健康检查")
async def health():
    return {"status": "healthy", "chunks": kb.collection.count(), "docs": len(kb.meta)}


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    # 检查API Key
    if not os.getenv("LLM_API_KEY"):
        print("\n⚠️  警告: 未设置 LLM_API_KEY 环境变量")
        print("   请执行: export LLM_API_KEY=sk-xxx")
        print("   或创建 .env 文件\n")
    
    print("\n" + "=" * 50)
    print("🚀 标准知识库系统")
    print("🌐 API文档: http://localhost:8000/docs")
    print("📊 统计: http://localhost:8000/stats")
    print("=" * 50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
```


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
