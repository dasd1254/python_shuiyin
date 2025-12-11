FROM python:3.9-slim

WORKDIR /app

# 1. 安装系统依赖 (这次去掉替换源的步骤)
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装 Python 库
COPY requirements.txt .
# 依然使用清华源加速 Python 包安装
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 3001
CMD ["python", "main.py"]
