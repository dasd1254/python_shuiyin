# 使用 python:3.9-slim 作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# =========================================================
# 1. 配置系统环境
# =========================================================
# 替换为国内源，加速系统包安装
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装必要的系统库
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置 PyTorch 模型下载目录的环境变量
# 这一步非常重要，告诉程序去哪里找模型
ENV TORCH_HOME=/app/models

# =========================================================
# 2. 准备依赖
# =========================================================
COPY requirements.txt .

# 升级 pip
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建版本约束文件 (防止依赖地狱)
RUN echo "numpy==1.23.5" > constraints.txt && \
    echo "scipy==1.10.1" >> constraints.txt && \
    echo "scikit-learn==1.3.2" >> constraints.txt && \
    echo "scikit-image==0.19.3" >> constraints.txt

# =========================================================
# 3. 分步安装依赖
# =========================================================
# 第一步：安装 PyTorch CPU 版 (减小体积)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 第二步：安装其他依赖 (带约束)
RUN pip install --no-cache-dir -c constraints.txt \
    -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# =========================================================
# 4. 【关键修复】植入 AI 模型
# =========================================================
# 创建模型目录结构
RUN mkdir -p /app/models/hub/checkpoints/

# 复制本地的 big-lama.pt 到镜像中
# 请确保您已经把 big-lama.pt 上传到了 Dockerfile 同级目录！！！
COPY big-lama.pt /app/models/hub/checkpoints/big-lama.pt

# =========================================================
# 5. 启动
# =========================================================
COPY . .

EXPOSE 3001

CMD ["python", "main.py"]