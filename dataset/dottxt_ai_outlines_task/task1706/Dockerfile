FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (modern Python package manager)
RUN pip install uv

# Clone the repository and checkout the specific commit
WORKDIR /workspace
RUN git clone https://github.com/dottxt-ai/outlines.git repo && \
    cd repo && \
    git checkout 947c904608821ef6e33493181f87653e25b08862

# Set up Python environment and pre-install dependencies
WORKDIR /workspace/repo
ENV TRANSFORMERS_NO_TF=1
ENV TRANSFORMERS_NO_FLAX=1
ENV PANDAS_NO_USE_PYARROW=1
ENV LLAMA_CPP_FORCE_CPU=1
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
RUN uv pip install --system -e . && \
    uv pip install --system pytest pytest-xdist pytest_mock pytest-asyncio pytest-benchmark pytest-cov && \
    uv pip install --system torch "transformers<5" "tokenizers<0.21" sentencepiece xgrammar llama-cpp-python==0.3.16 psutil

# Copy the runner script
COPY runner.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/runner.sh

ENTRYPOINT ["/usr/local/bin/runner.sh"]

