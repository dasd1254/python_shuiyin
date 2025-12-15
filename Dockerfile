# 使用 python:3.9-slim 作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# =========================================================
# 【关键修改】配置国内 APT 源 (解决 deb.debian.org 连接超时问题)
# 自动替换默认源为阿里云源，适配新旧版 Debian (sources.list 或 debian.sources)
# =========================================================
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# =========================================================
# 1. 安装系统依赖 (OpenCV/PaddleOCR 运行所需库)
# =========================================================
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装 Python 依赖
COPY requirements.txt .

# 使用清华源加速 pip 安装
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


# 3. 复制代码并启动
COPY . .

# 暴露端口
EXPOSE 3001

# 启动命令
CMD ["python", "main.py"]
