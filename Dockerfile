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
# 2. 安装 Python 依赖
# =========================================================
COPY requirements.txt .

# [步骤1] 升级 pip
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# [步骤2] 关键修复！！！精准锁定科学计算库版本
# 1. numpy==1.23.5: 严格匹配你的 requirements.txt，防止安装 numpy 2.0
# 2. scikit-learn==1.3.2: 锁定一个支持 python 3.9 的现代二进制版本，防止回溯到 0.x 版本
# 3. scipy==1.10.1: 配合 numpy 1.23 的稳定版本
RUN pip install --no-cache-dir \
    "numpy==1.23.5" \
    "scikit-learn==1.3.2" \
    "scipy==1.10.1" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# [步骤3] 安装剩余依赖
# 此时 numpy 已经被锁定在 1.23.5，pip 不会再因为版本冲突去重新编译旧包
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# =========================================================
# 3. 复制代码并启动
# =========================================================
COPY . .

EXPOSE 3001

CMD ["python", "main.py"]
