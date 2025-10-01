# 第一階段：Poetry 安裝環境
FROM python:3.11-slim AS poetry-base

# 安裝 Poetry
ENV POETRY_VERSION=2.1.3
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install --no-cache-dir -U pip setuptools \
    && $POETRY_VENV/bin/pip install --no-cache-dir poetry==${POETRY_VERSION}

# 第二階段：依賴安裝環境
FROM python:3.11-slim AS dependencies

# 從第一階段複製 Poetry
COPY --from=poetry-base /opt/poetry-venv /opt/poetry-venv
ENV PATH="/opt/poetry-venv/bin:$PATH"

# 設定 Poetry 不要創建虛擬環境
ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# 只複製依賴檔案
COPY pyproject.toml poetry.lock ./

# 安裝專案依賴（只安裝 production 依賴）
RUN poetry install --no-interaction --no-ansi --no-root --only main

# 第三階段：最終運行環境
FROM python:3.11-slim AS runtime

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    unrar-free \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 從依賴階段複製 Python 套件
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# 複製應用程式碼
COPY . .

# 建立 data 目錄（如果不存在）
RUN mkdir -p data

# 設定環境變數
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 暴露 8000 port
EXPOSE 8000

# 啟動應用程式
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]