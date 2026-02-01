FROM golang:1.21-alpine

# Install system dependencies (including Python for Modal compatibility)
RUN apk add --no-cache \
    git \
    curl \
    wget \
    bash \
    python3 \
    py3-pip && \
    rm -f /usr/lib/python*/EXTERNALLY-MANAGED

# Clone the repository and checkout the specific commit
WORKDIR /workspace
RUN git clone https://github.com/go-chi/chi.git repo && \
    cd repo && \
    git checkout 9436cc85db0d35be06e66b91b4f74bbc0295b07b

# Set up Go environment
WORKDIR /workspace/repo
RUN go mod download

# Copy the runner script
COPY runner.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/runner.sh

ENTRYPOINT ["/usr/local/bin/runner.sh"]
