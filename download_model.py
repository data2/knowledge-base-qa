# download_model.py - 专门下载 bge-m3 模型
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download
import shutil

print("📥 正在从 HuggingFace 镜像下载 bge-m3 模型...")
print("⚠️ 模型大小约 2.2GB，下载需要 10-30 分钟，请耐心等待...")

# 下载到指定目录
model_dir = "./models/bge-m3"

# 如果目录已存在，先删除
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

# 下载模型
snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir=model_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=4,
    ignore_patterns=["*.safetensors", "*.h5", "*.msgpack"]  # 只下载 pytorch 版本
)

print(f"✅ 模型下载完成！保存在: {model_dir}")
print(f"   文件列表:")
for f in os.listdir(model_dir):
    size = os.path.getsize(os.path.join(model_dir, f)) / (1024*1024)
    print(f"   - {f} ({size:.1f} MB)")
