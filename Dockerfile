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
    echo "scikit-image==0.19.3" >> constraints.txt

# 分步安装
RUN pip install --no-cache-dir -c constraints.txt "numpy==1.23.5" -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -c constraints.txt -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# =========================================================
# 3. 【关键】强制植入模型到默认路径
# =========================================================
# 这是 SimpleLama 默认寻找的位置
RUN mkdir -p /root/.cache/torch/hub/checkpoints/

# 复制当前目录下的 big-lama.pt 到系统默认缓存目录
COPY big-lama.pt /root/.cache/torch/hub/checkpoints/big-lama.pt

# =========================================================

COPY . .
EXPOSE 3001
CMD ["python", "main.py"]