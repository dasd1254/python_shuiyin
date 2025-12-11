FROM python:3.9-slim

WORKDIR /app

# 安装系统库
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 安装 Python 库
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 3001
CMD ["python", "main.py"]