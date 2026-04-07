FROM python:3.12-slim

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
RUN git clone https://github.com/huggingface/datasets.git repo && \
    cd repo && \
    git checkout b60ebb83cd668decaa21df66148beb44bce57739

# Set up Python environment and pre-install dependencies
WORKDIR /workspace/repo
RUN uv pip install --system -e .
RUN uv pip uninstall --system pyarrow && uv pip install --system "pyarrow>=17.0.0,<20.0.0"
# Install tensorflow, explicitly excluding tensorflow-macos (macOS-only package)
RUN uv pip install --system "tensorflow>=2.16.0,<2.17.0" || uv pip install --system "tensorflow==2.16.2"
RUN uv pip install --system torch jax
# Pin numpy to < 2.0 to avoid compatibility issues with np.array(copy=False)
RUN uv pip install --system "numpy<2.0"
RUN uv pip install --system pytest pytest-xdist pytest-datadir Pillow

# Copy the runner script
COPY runner.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/runner.sh

ENTRYPOINT ["/usr/local/bin/runner.sh"]

