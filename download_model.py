# download_model_modelscope.py
import os
import shutil

print("📥 正在从 ModelScope 下载 bge-m3 模型...")
print("⚠️ 模型大小约 2.2GB，下载需要 10-30 分钟，请耐心等待...")

# 删除旧目录
if os.path.exists("./models/bge-m3"):
    shutil.rmtree("./models/bge-m3")

try:
    from modelscope import snapshot_download
    
    # 使用 ModelScope 下载
    model_dir = snapshot_download(
        'Xorbits/bge-m3',
        cache_dir='./models',
        revision='master'
    )
    
    print(f"✅ 模型下载完成！")
    print(f"   保存在: {model_dir}")
    
    # 列出文件
    print("   文件列表:")
    for f in os.listdir(model_dir):
        size = os.path.getsize(os.path.join(model_dir, f)) / (1024*1024)
        print(f"   - {f} ({size:.1f} MB)")
        
except ImportError:
    print("❌ modelscope 未安装，请执行: pip install modelscope")
except Exception as e:
    print(f"❌ 下载失败: {e}")
    print("\n尝试备选方案...")
    
    # 备选：使用 ModelScope 的 Hub 下载
    try:
        from modelscope.hub.api import HubApi
        api = HubApi()
        api.login()
        
        # 使用 API 下载
        from modelscope.hub.file_download import model_file_download
        print("使用 API 方式下载...")
        
    except Exception as e2:
        print(f"❌ 备选方案也失败: {e2}")
        print("\n💡 建议:")
        print("   1. 确保网络连接正常")
        print("   2. 或者手动下载: https://modelscope.cn/models/Xorbits/bge-m3")