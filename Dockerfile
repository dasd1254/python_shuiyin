# 使用官方 Python 3.9 轻量版
FROM python:3.9-slim

WORKDIR /app

# 1. 安装系统依赖 (OpenCV 和 Paddle 需要的库)
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. 复制依赖清单
COPY requirements.txt .

# 3. 安装 Python 依赖 (清华源加速)
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 复制代码
COPY . .

# 5. 暴露端口
EXPOSE 3001

# 6. 启动服务
CMD ["python", "main.py"]