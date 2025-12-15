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
# 2. 安装 Python 依赖 (修复报错的关键步骤)
# =========================================================
COPY requirements.txt .

# [关键步骤1] 升级 pip
# 旧版 pip (23.0) 在处理复杂依赖回溯时很容易出错，升级到最新版 (25.x)
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# [关键步骤2] 预先安装科学计算库
# 强制安装较新的 numpy 和 scipy 二进制包，防止 pip 去下载古董版本进行源码编译
RUN pip install --no-cache-dir \
    "numpy>=1.23.5" \
    "scipy>=1.10.0" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# [关键步骤3] 安装剩余依赖
# 即使 requirements.txt 里有冲突，因为上面已经装好了 numpy/scipy，pip 通常会复用已安装的版本
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# =========================================================
# 3. 复制代码并启动
# =========================================================
COPY . .

EXPOSE 3001

CMD ["python", "main.py"]
