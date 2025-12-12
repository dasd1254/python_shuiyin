FROM python:3.9-slim

WORKDIR /app

# 1. 安装系统依赖
# 【关键修改1111】将 libgl1-mesa-glx 改为 libgl1 (适配新版 Debian)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 复制代码
COPY . .

EXPOSE 3001
CMD ["python", "main.py"]