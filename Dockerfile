FROM python:3.9-slim

WORKDIR /app

# 配置国内源
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装系统依赖 (AI处理通常需要 libgl1 和 libgomp1)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 升级 pip
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 1. 优先安装 numpy 以防冲突
RUN pip install --no-cache-dir "numpy==1.23.5" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 单独安装 PyTorch (CPU版，为了减小体积)
# 如果你的服务器有显卡，可以去掉 --extra-index-url 使用默认源安装 GPU 版
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# 3. 安装 simple-lama-inpainting 和其他依赖
# 注意：这里可能会稍微慢一点，因为它会下载依赖
RUN pip install --no-cache-dir simple-lama-inpainting fastapi uvicorn python-multipart Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 3001

# 设置环境变量，防止模型下载时缓存占满 /tmp
ENV TORCH_HOME=/app/models

CMD ["python", "main.py"]
