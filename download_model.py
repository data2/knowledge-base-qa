# download_model.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir="./models/bge-m3",  # 下载到当前目录的 models 文件夹
    local_dir_use_symlinks=False
)
print("✅ 模型下载完成！")
