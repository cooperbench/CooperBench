# GCP Backend Setup Guide

Get started with Google Cloud Platform as your CooperBench execution backend.

## Quick Start

```bash
# Prerequisites: Install gcloud CLI first
# macOS: brew install google-cloud-sdk
# Linux: curl https://sdk.cloud.google.com | bash

# 1. Install GCP support
pip install 'cooperbench[gcp]'

# 2. Run configuration wizard
cooperbench config gcp

# 3. Run experiments
cooperbench run --backend gcp -s lite
cooperbench eval --backend gcp -n my-experiment
```

**That's it!** The wizard handles authentication, project setup, and validation automatically.

## Why Use GCP?

GCP provides several advantages over the default Modal backend:

- **Scalability**: Run hundreds of parallel evaluations using GCP Batch
- **Cost Control**: Use your own GCP credits and quotas
- **Data Locality**: Keep data in your GCP project
- **Customization**: Use custom VM images with pre-pulled Docker images
- **No External Dependencies**: No need for Modal account

## Prerequisites

Before running the configuration wizard, you need:

### 1. GCP Account

Sign up at [cloud.google.com](https://cloud.google.com)
- Free tier includes $300 credit
- Create a project at [console.cloud.google.com/projectcreate](https://console.cloud.google.com/projectcreate)
- Enable billing at [console.cloud.google.com/billing](https://console.cloud.google.com/billing)

### 2. gcloud CLI

Install the Google Cloud SDK:

**macOS:**
```bash
brew install google-cloud-sdk
```

**Linux:**
```bash
curl https://sdk.cloud.google.com | bash
```

**Windows:**
Download from [cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)

Verify installation:
```bash
gcloud version
```

## Setup

### Interactive Configuration (Recommended)

Run the interactive setup wizard:

```bash
cooperbench config gcp
```

The wizard will:
1. Check if gcloud CLI is installed
2. Verify GCP Python dependencies
3. Authenticate with your Google account (opens browser)
4. Help you select project, region, and zone
5. Validate API access

Your configuration is saved to:
- macOS: `~/Library/Application Support/cooperbench/config.json`
- Linux: `~/.config/cooperbench/config.json`
- Windows: `%APPDATA%\cooperbench\config.json`

### Skip Validation (Faster Setup)

For faster setup without API validation:

```bash
cooperbench config gcp --skip-tests
```

### Manual Configuration (Alternative)

If you prefer manual setup:

```bash
# Authenticate
gcloud auth login
gcloud auth application-default login

# Set project
export GOOGLE_CLOUD_PROJECT="your-project-id"
gcloud config set project your-project-id
```

## Required APIs

The wizard will test access to these APIs and provide links if they need to be enabled:

- **Compute Engine API**: For running VMs
  - https://console.cloud.google.com/apis/library/compute.googleapis.com

- **Cloud Batch API**: For parallel evaluation
  - https://console.cloud.google.com/apis/library/batch.googleapis.com

- **Cloud Storage API**: For storing results
  - https://console.cloud.google.com/apis/library/storage.googleapis.com

Or enable all at once:
```bash
gcloud services enable compute.googleapis.com batch.googleapis.com storage.googleapis.com
```

## Usage

### Run Agents on Tasks

```bash
# Run with GCP backend
cooperbench run --backend gcp -s lite

# Run cooperative agents
cooperbench run \
  --backend gcp \
  --setting coop \
  -s lite \
  -m gemini/gemini-3-flash-preview \
  --concurrency 10
```

### Evaluate Results

```bash
# Evaluate with GCP Batch
cooperbench eval --backend gcp -n my-experiment

# Large-scale evaluation with high parallelism
cooperbench eval \
  --backend gcp \
  -n my-experiment \
  -r llama_index_task \
  --concurrency 50
```

### Complete Example

```bash
# One-time setup
pip install 'cooperbench[gcp]'
cooperbench config gcp

# Run experiment
cooperbench run \
  --backend gcp \
  --setting coop \
  -s lite \
  -m gemini/gemini-3-flash-preview

# Evaluate results
cooperbench eval \
  --backend gcp \
  -n coop-lite-gemini-3-flash \
  --concurrency 50

# Results saved to logs/
```

## Advanced Configuration

### Custom VM Images (Optional Optimization)

**This is optional and only useful for large-scale evaluations (100+ tasks).**

By default, each evaluation VM pulls Docker images on-demand (~2-5 minutes startup). For large-scale runs, you can pre-build a VM image with all images cached.

**When to use:**
- Large evaluation runs (100+ tasks)
- Repeated evaluations where startup time matters
- When you want to minimize per-task costs

**When NOT to use:**
- Small runs (< 100 tasks) - not worth the build time
- First-time setup - stick with defaults

**Build custom image:**

```bash
# Set your project
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Run the build script (takes ~30-60 minutes)
./scripts/build_gcp_vm_image.sh
```

The script will:
1. Create a temporary VM with Container-Optimized OS
2. Pull all CooperBench Docker images
3. Create a VM image snapshot (family: `cooperbench-eval`)
4. Delete the temporary VM

**Use the custom image:**

```python
from cooperbench.eval.backends.gcp import GCPBatchEvaluator

evaluator = GCPBatchEvaluator(
    vm_image="cooperbench-eval"  # Image family from build script
)
```

Or via environment variable:
```bash
export COOPERBENCH_VM_IMAGE="cooperbench-eval"
```

### Custom VPC Network (Optional)

For multi-agent git collaboration, you can use a custom VPC network:

```python
from cooperbench.agents.mini_swe_agent.environments.gcp import GCPEnvironment

env = GCPEnvironment(
    image="python:3.11",
    network="cooperbench-vpc"  # Optional: Your VPC network
)
```

**Note:** VPC networking is only required for git-based collaboration. For solo runs or Redis-based messaging, the default network works fine.

### Region Selection

Choose regions based on your location for lower latency:

- **US**: `us-central1` (Iowa), `us-east1` (South Carolina)
- **Europe**: `europe-west1` (Belgium), `europe-west4` (Netherlands)
- **Asia**: `asia-east1` (Taiwan), `asia-southeast1` (Singapore)

## Architecture

### Agent Execution (GCPEnvironment)

```
┌──────────────────────────────────────┐
│ CooperBench (Local)                  │
│   ↓ SSH via gcloud compute ssh      │
├──────────────────────────────────────┤
│ GCP VM (Container-Optimized OS)      │
│   ├─ Docker Container (Agent)        │
│   │   └─ Agent code execution        │
│   └─ Commands via docker exec        │
└──────────────────────────────────────┘
```

Each agent runs in its own isolated GCP VM:
- Default: e2-medium (2 vCPU, 4GB RAM)
- Container-Optimized OS
- Docker container with task environment
- SSH access via gcloud CLI

### Evaluation (GCPBatchEvaluator)

```
┌────────────────────────────────────────┐
│ CooperBench (Local)                    │
│   ↓ Submit Batch job                   │
├────────────────────────────────────────┤
│ GCP Batch (Managed)                    │
│   ├─ Task 1 (VM 1)                     │
│   ├─ Task 2 (VM 1)                     │
│   ├─ Task 3 (VM 2)                     │
│   ├─ ...                               │
│   └─ Task N (VM M)                     │
│         ↓ Results                      │
├────────────────────────────────────────┤
│ GCS Bucket (cooperbench-eval-PROJECT)  │
│   ├─ Job manifests                     │
│   ├─ Patches                           │
│   └─ Results                           │
└────────────────────────────────────────┘
```

GCP Batch automatically:
- Schedules tasks across VMs
- Handles VM lifecycle
- Stores results in GCS
- Cleans up resources

### Git Server (GCPGitServer)

```
┌─────────────────────────────────────┐
│ Git Server VM (Debian)              │
│   └─ git-daemon (port 9418)         │
│         ↓ git:// protocol            │
├─────────────────────────────────────┤
│ VPC Network or External IP          │
│   ├→ Agent 1 (git push/pull/merge)  │
│   ├→ Agent 2 (git push/pull/merge)  │
│   └→ Agent N (git push/pull/merge)  │
└─────────────────────────────────────┘
```

Optional git server for multi-agent collaboration (enabled with `--git` flag).

## Cost Estimation

Using default settings in us-central1:

### Agent Execution
- **VM Type**: e2-medium (2 vCPU, 4GB RAM)
- **Cost**: ~$0.03/hour
- **Typical Task**: 5-30 minutes
- **Per Task**: $0.0025 - $0.015

### Evaluation (Batch)
- **VM Type**: 4 vCPU, 16GB RAM
- **Cost**: ~$0.15/hour per VM
- **Parallelism**: 50 VMs = ~$7.50/hour
- **Typical Job**: 10-30 minutes for 500 tasks

### Storage (GCS)
- **Storage**: ~$0.02/GB/month
- **Operations**: ~$0.005 per 10,000 operations
- **Lifecycle**: Auto-delete after 7 days

### Example Costs

- **Small run** (10 tasks): $0.05 - $0.15
- **Medium run** (100 tasks): $0.50 - $1.50
- **Large run** (1000 tasks): $5.00 - $10.00

**Note:** GCP free tier includes $300 credit for new users.

## Troubleshooting

### gcloud not found

```
Error: gcloud: command not found
```

**Solution:**

```bash
# macOS
brew install google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash

# Verify
gcloud version
```

### Not authenticated

```
Error: Your default credentials were not found
Error: Could not automatically determine credentials
```

**Solution:**

You need **both** types of authentication:

```bash
# 1. Authenticate gcloud CLI
gcloud auth login

# 2. Set up Application Default Credentials (REQUIRED for Python SDK)
gcloud auth application-default login
```

**Why both?**
- `gcloud auth login`: For gcloud CLI commands
- `gcloud auth application-default login`: For GCP Python libraries

The configuration wizard handles both automatically.

### API not enabled

```
Error: API [compute.googleapis.com] not enabled
```

**Solution:**

Enable required APIs:
```bash
gcloud services enable \
  compute.googleapis.com \
  batch.googleapis.com \
  storage.googleapis.com
```

Or enable via console:
- https://console.cloud.google.com/apis/library/compute.googleapis.com
- https://console.cloud.google.com/apis/library/batch.googleapis.com
- https://console.cloud.google.com/apis/library/storage.googleapis.com

### Permission denied

```
Error: The caller does not have permission
```

**Solution:**

1. Ensure billing is enabled: https://console.cloud.google.com/billing
2. Check IAM permissions: https://console.cloud.google.com/iam-admin/iam
3. Required roles:
   - `roles/compute.instanceAdmin.v1` (Compute Instance Admin)
   - `roles/batch.jobsEditor` (Batch Job Editor)
   - `roles/storage.admin` (Storage Admin)

### Quota exceeded

```
Error: Quota exceeded for quota metric 'cpus'
```

**Solution:**

1. Check quotas: https://console.cloud.google.com/iam-admin/quotas
2. Request quota increase (usually approved within hours)
3. Reduce `--concurrency` parameter to use fewer VMs

### Python package missing

```
Error: No module named 'google.cloud.compute'
```

**Solution:**

```bash
pip install 'cooperbench[gcp]'
```

Or install packages individually:
```bash
pip install google-cloud-batch google-cloud-compute google-cloud-storage
```

## Migration from Modal

To migrate from Modal to GCP:

```bash
# 1. Install GCP dependencies
pip install 'cooperbench[gcp]'

# 2. Configure GCP
cooperbench config gcp

# 3. Use --backend gcp instead of default modal
cooperbench run --backend gcp -s lite
cooperbench eval --backend gcp -n experiment-name
```

All other commands and options remain the same!

## Comparison: Modal vs GCP

| Feature | Modal (Default) | GCP |
|---------|----------------|-----|
| **Setup** | `modal setup` | `cooperbench config gcp` |
| **Account** | Modal account | GCP account |
| **Cost** | Modal credits | Pay-as-you-go |
| **Scale** | Auto-scaling | Up to quotas |
| **Free tier** | Limited | $300 credit |
| **Data location** | External service | Your GCP project |
| **Customization** | Limited | Full control (VMs, networks, images) |

## Support

For issues or questions:
- **GitHub Issues**: https://github.com/cooperbench/CooperBench/issues
- **GCP Documentation**: https://cloud.google.com/docs
- **CooperBench Docs**: https://cooperbench.com
