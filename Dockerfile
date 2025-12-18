FROM python:3.9-slim

WORKDIR /app

# 1. 基础环境
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libgomp1 build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. 依赖安装
COPY requirements.txt .
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建约束文件
RUN echo "numpy==1.23.5" > constraints.txt && \
    echo "scipy==1.10.1" >> constraints.txt && \
    echo "scikit-learn==1.3.2" >> constraints.txt && \
    echo "scikit-image==0.19.3" >> constraints.txt && \
    echo "imageio==2.31.1" >> constraints.txt

# 分步安装
RUN pip install --no-cache-dir -c constraints.txt "numpy==1.23.5" -i https://pypi.tuna.tsinghua.edu.cn/simple
# 强制 CPU 版 torch
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -c constraints.txt -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 【关键】植入模型 (只需这一处)
# 手动创建目录
RUN mkdir -p /root/.cache/simple_lama_inpainting/

# 复制 big-lama.pt 到指定目录
# ⚠️ 注意：执行 docker build 前，请确保 big-lama.pt 就在 Dockerfile 旁边
COPY big-lama.pt /root/.cache/simple_lama_inpainting/big-lama.pt

# 4. 复制代码
COPY . .

EXPOSE 3001

CMD ["python", "main.py"]