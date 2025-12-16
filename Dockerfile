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
# 3. 【核心修复】创建更严格的版本约束文件
# =========================================================
# 我们在这里把可能会导致回溯的包全部锁死在“已知的稳定二进制版本”
# scikit-image==0.19.3 是最后一个广泛支持旧依赖且不需要编译的稳定版
# imageio==2.31.1 配合 scikit-image 使用
RUN echo "numpy==1.23.5" > constraints.txt && \
    echo "scipy==1.10.1" >> constraints.txt && \
    echo "scikit-learn==1.3.2" >> constraints.txt && \
    echo "scikit-image==0.19.3" >> constraints.txt && \
    echo "imageio==2.31.1" >> constraints.txt

# =========================================================
# 4. 安装依赖 (预装 + 约束)
# =========================================================
# 先把这几个最难搞的包通过约束文件装好
RUN pip install --no-cache-dir \
    -c constraints.txt \
    "numpy==1.23.5" \
    "scikit-image==0.19.3" \
    "scikit-learn==1.3.2" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 然后安装 requirements.txt，依然带着紧箍咒 (-c)
RUN pip install --no-cache-dir -r requirements.txt \
    -c constraints.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# =========================================================
# 5. 复制代码并启动
# =========================================================
COPY . .

EXPOSE 3001

CMD ["python", "main.py"]
