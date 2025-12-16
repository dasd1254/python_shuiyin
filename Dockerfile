# 使用 python:3.9-slim 作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# =========================================================
# 配置国内 APT 源
# =========================================================
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# =========================================================
# 1. 安装系统依赖 & 编译工具
# =========================================================
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# 2. 准备依赖文件
# =========================================================
COPY requirements.txt .

# 升级 pip
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# =========================================================
# 3. 【核心修复】创建版本约束文件
# =========================================================
# 这里手动创建一个 constraints.txt 文件
# 这告诉 pip：无论发生什么，都必须使用这几个版本，严禁去下载旧版本编译！
RUN echo "numpy==1.23.5" > constraints.txt && \
    echo "scipy==1.10.1" >> constraints.txt && \
    echo "scikit-learn==1.3.2" >> constraints.txt

# =========================================================
# 4. 安装依赖 (带上 -c 参数)
# =========================================================
# 注意：这里加了 -c constraints.txt，这就像给 pip 戴上了紧箍咒
RUN pip install --no-cache-dir -r requirements.txt \
    -c constraints.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# =========================================================
# 5. 复制代码并启动
# =========================================================
COPY . .

EXPOSE 3001

CMD ["python", "main.py"]
