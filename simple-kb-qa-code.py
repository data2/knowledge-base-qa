"""
知识库问答系统 - 使用 transformers 替代 sentence-transformers
兼容 Python 3.12 / 3.13
"""
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import uuid
import json
from typing import List, Dict, Optional
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import chromadb
from openai import OpenAI
import PyPDF2
from docx import Document
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


class EmbeddingService:
    """向量化服务 - 基于 transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.device = device
        print(f"📥 加载向量模型: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        self.dim = self.model.config.hidden_size
        print(f"   向量维度: {self.dim}")
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用 CLS token 的向量
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        return embeddings.tolist()


class KnowledgeBase:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_device: str = "cpu",
        llm_api_base: str = "https://api.deepseek.com/v1",
        llm_api_key: str = "",
        llm_model: str = "deepseek-chat",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        top_k: int = 4,
        similarity_threshold: float = 0.5,
        persist_dir: str = "./chroma_data"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # 向量模型（使用 transformers）
        self.encoder = EmbeddingService(embedding_model, embedding_device)
        self.embedding_dim = self.encoder.dim
        
        # 向量数据库
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ 向量库已加载，共 {self.collection.count()} 个分块")
        
        # LLM
        self.llm_client = OpenAI(
            base_url=llm_api_base,
            api_key=llm_api_key or os.getenv("LLM_API_KEY", "")
        )
        self.llm_model = llm_model
        print(f"✅ LLM已配置: {llm_model}")
        
        self.meta_file = Path(persist_dir) / "meta.json"
        self.meta = self._load_meta()
        
        print("=" * 50)
        print("📚 知识库初始化完成")
        print(f"   文档数: {len(self.meta)}")
        print(f"   分块数: {self.collection.count()}")
        print("=" * 50)
    
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
        elif ext in [".doc", ".docx"]:
            return self._parse_docx(content)
        elif ext == ".txt":
            return self._parse_txt(content)
        else:
            raise ValueError(f"不支持的文件格式: {filename}")
    
    def _chunk_text(self, text: str) -> List[str]:
        if not text.strip():
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks
    
    def _load_meta(self) -> Dict:
        if self.meta_file.exists():
            with open(self.meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def _save_meta(self):
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
    
    def upload(self, content: bytes, filename: str) -> Dict:
        text = self._parse_file(content, filename)
        if not text.strip():
            raise ValueError("文档内容为空或无法解析")
        
        chunks = self._chunk_text(text)
        if not chunks:
            raise ValueError("分块结果为空")
        
        doc_id = str(uuid.uuid4())[:8]
        
        print(f"🔄 向量化中: {filename} ({len(chunks)} 个分块)")
        embeddings = self.encoder.encode(chunks)
        
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
        
        self.meta[doc_id] = {
            "filename": filename,
            "chunks": len(chunks),
            "status": "completed"
        }
        self._save_meta()
        
        print(f"✅ 上传成功: {filename} -> {doc_id}")
        return {"doc_id": doc_id, "filename": filename, "chunks": len(chunks)}
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        if self.collection.count() == 0:
            return []
        
        top_k = top_k or self.top_k
        q_vec = self.encoder.encode([query])[0]
        
        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
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
        sources = self.search(question, top_k)
        
        if not sources:
            return {
                "question": question,
                "answer": "知识库中暂无相关信息，请上传相关文档后重试。",
                "sources": [],
                "total_tokens": 0
            }
        
        context = "\n\n---\n\n".join([s["content"][:2000] for s in sources])
        
        prompt = f"""请根据以下参考资料回答问题。

参考资料：
{context}

用户问题：{question}

要求：
1. 基于参考资料给出准确答案
2. 如果资料中没有相关信息，明确说"资料中未提及"
3. 答案简洁明了，不要编造
"""
        
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
        ids = self.collection.get(where={"doc_id": doc_id})["ids"]
        if ids:
            self.collection.delete(ids=ids)
            if doc_id in self.meta:
                del self.meta[doc_id]
                self._save_meta()
            return True
        return False
    
    def list_docs(self) -> List[Dict]:
        return [{"doc_id": k, **v} for k, v in self.meta.items()]
    
    def stats(self) -> Dict:
        return {
            "total_chunks": self.collection.count(),
            "total_documents": len(self.meta),
            "documents": self.list_docs()
        }


app = FastAPI(title="知识库系统", version="1.0.0")

kb = KnowledgeBase(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_device="cpu",
    llm_api_key=os.getenv("LLM_API_KEY", "")
)


class QueryReq(BaseModel):
    question: str
    top_k: int = 4


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        return kb.upload(content, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/chat")
async def chat(req: QueryReq):
    return kb.chat(req.question, req.top_k)


@app.post("/search")
async def search(req: QueryReq):
    return {"question": req.question, "results": kb.search(req.question, req.top_k)}


@app.get("/documents")
async def list_docs():
    return kb.list_docs()


@app.delete("/documents/{doc_id}")
async def delete_doc(doc_id: str):
    return {"doc_id": doc_id, "deleted": kb.delete(doc_id)}


@app.get("/stats")
async def stats():
    return kb.stats()


@app.get("/health")
async def health():
    return {"status": "healthy", "chunks": kb.collection.count(), "docs": len(kb.meta)}


if __name__ == "__main__":
    import uvicorn
    
    if not os.getenv("LLM_API_KEY"):
        print("\n⚠️  警告: 未设置 LLM_API_KEY 环境变量")
        print("   请执行: set LLM_API_KEY=sk-xxx\n")
        print("   或创建 .env 文件:\n")
        print("   LLM_API_KEY=sk-xxx\n")
    
    print("\n" + "=" * 50)
    print("🚀 知识库系统启动")
    print("🌐 API文档: http://localhost:8000/docs")
    print("=" * 50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)