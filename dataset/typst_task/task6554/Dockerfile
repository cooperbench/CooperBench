FROM rust:1.80-slim

# Install system dependencies (including Python for Modal compatibility)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

# Clone the repository and checkout the specific commit
WORKDIR /workspace
RUN git clone https://github.com/typst/typst.git repo && \
    cd repo && \
    git checkout b8034a343831e8609aec2ec81eb7eeda57aa5d81

# Pre-build everything at base commit (cache layer)
# This compiles all dependencies so runtime only recompiles changed files
WORKDIR /workspace/repo
RUN cargo build --package typst --package typst-cli && \
    cargo build --package typst-tests --tests

# Copy the runner script
COPY runner.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/runner.sh

ENTRYPOINT ["/usr/local/bin/runner.sh"]
