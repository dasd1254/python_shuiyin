# 使用 python:3.9-slim 作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# =========================================================
# 1. 安装系统依赖 (保持不变)
# =========================================================
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# 2. 准备环境
# =========================================================
# 升级 pip
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建版本约束文件 (这是防止回溯的“定海神针”)
RUN echo "numpy==1.23.5" > constraints.txt && \
    echo "scipy==1.10.1" >> constraints.txt && \
    echo "scikit-learn==1.3.2" >> constraints.txt && \
    echo "scikit-image==0.19.3" >> constraints.txt

# =========================================================
# 3. 分步安装依赖 (解决 resolution-too-deep 的关键！！！)
# =========================================================

# 第一步：先安装地基 (Numpy & PaddlePaddle)
# 这一步最关键，先把 numpy 锁死，不让后面的包乱改版本
RUN pip install --no-cache-dir -c constraints.txt \
    "numpy==1.23.5" \
    paddlepaddle \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 第二步：安装 PaddleOCR (最重的包)
# 单独安装它，Pip 只需要处理它的依赖，压力小很多
RUN pip install --no-cache-dir -c constraints.txt \
    "paddleocr>=2.7.0" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 第三步：安装 Albumentations (冲突源)
# 因为前面已经装好了 numpy/scikit-image，这一步会直接复用，不会报错
RUN pip install --no-cache-dir -c constraints.txt \
    "albumentations==1.3.1" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 第四步：安装剩余的 Web 依赖 (FastAPI 等)
# 复制 requirements.txt，安装剩下的东西
COPY requirements.txt .
RUN pip install --no-cache-dir -c constraints.txt \
    -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# =========================================================
# 4. 复制代码并启动
# =========================================================
COPY . .

EXPOSE 3001

CMD ["python", "main.py"]
