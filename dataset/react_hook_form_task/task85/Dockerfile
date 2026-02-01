FROM node:22-slim

# Install git, Python (for Modal compatibility), and other required tools
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

# Install pnpm globally
RUN npm install -g pnpm

# Clone the repository and checkout the specific commit
WORKDIR /workspace
RUN git clone https://github.com/react-hook-form/react-hook-form.git repo && \
    cd repo && \
    git checkout b5863b46346416972c025f4b621cb624ffc4a955

# Install dependencies
WORKDIR /workspace/repo
RUN pnpm install && \
    cp -r node_modules /workspace/node_modules_cache

# Copy the runner script
COPY runner.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/runner.sh

ENTRYPOINT ["/usr/local/bin/runner.sh"]
