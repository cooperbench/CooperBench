FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository and checkout the specific commit
WORKDIR /workspace
RUN git clone https://github.com/stanfordnlp/dspy.git repo && \
    cd repo && \
    git checkout 8f773dc75e34da21b89480e8bc71df503d5ab5fc

# Install Python dependencies
WORKDIR /workspace/repo
RUN pip install --upgrade pip && \
    pip install -e ".[dev]"

# dspy declares `litellm>=1.64.0` with no upper bound, so the image floats to whatever is
# current at build time. In newer litellm `ModelResponse` no longer carries `.usage`, and
# `base_lm.py` does `dict(response.usage)` unconditionally — which turns four tests in
# tests/adapters/test_chat_adapter.py red on a PRISTINE checkout. Because the runner
# executes that file in full, every feature of the task then fails regardless of what the
# agent wrote. Pinned to the newest version where the base suite is green (20 passed).
RUN pip install "litellm==1.77.1"

# Copy the runner script
COPY runner.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/runner.sh

ENTRYPOINT ["/usr/local/bin/runner.sh"]
