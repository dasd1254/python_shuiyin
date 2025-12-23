FROM python:3.9-slim

WORKDIR /app

# 1. 基础环境 (换源 + 安装系统依赖 + 立即清理垃圾)
# 使用 && 连接命令，确保同一层内清理缓存，不占镜像体积
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. 升级 pip
RUN pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 【核心步骤】优先安装 CPU 版 PyTorch (节省 2GB 空间)
# --no-cache-dir: 不缓存安装包
# --index-url: 指定去下载 CPU 专用版本
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. 安装其他依赖
COPY requirements.txt .

# 关键：安装 requirements 时，告诉 pip 如果需要找 torch 依赖，优先去 cpu 源找
# 这样 simple-lama-inpainting 就不会因为找不到依赖而去下载巨大的 CUDA 版 torch
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 5. 植入模型
# 确保 big-lama.pt 文件在项目根目录下
RUN mkdir -p /root/.cache/simple_lama_inpainting/
COPY big-lama.pt /root/.cache/simple_lama_inpainting/big-lama.pt

# 6. 复制代码
COPY . .

EXPOSE 3001

CMD ["python", "main.py"]