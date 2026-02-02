# GCP Backend Setup Guide

This guide explains how to set up and use Google Cloud Platform (GCP) as your execution backend for CooperBench.

## Quick Start

```bash
# 1. Install GCP dependencies
uv pip install 'cooperbench[gcp]'

# 2. Run the configuration wizard
uv run cooperbench config gcp

# 3. Use GCP backend
uv run cooperbench run --backend gcp
uv run cooperbench eval --backend gcp
```

## Why Use GCP?

GCP provides several advantages over the default Modal backend:

1. **Scalability**: Run hundreds of parallel evaluations using GCP Batch
2. **Cost Control**: Use your own GCP credits and quotas
3. **Data Locality**: Keep data in your GCP project
4. **Customization**: Use custom VM images with pre-pulled Docker images
5. **No External Dependencies**: No need for Modal account

## Prerequisites

### 1. GCP Account

You need a Google Cloud Platform account with:
- An active project
- Billing enabled
- Sufficient compute quotas

Create a project at: https://console.cloud.google.com/projectcreate

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
Download from: https://cloud.google.com/sdk/docs/install

### 3. Python Dependencies

Install GCP libraries:
```bash
uv pip install 'cooperbench[gcp]'
```

Or manually:
```bash
pip install google-cloud-batch google-cloud-compute google-cloud-storage
```

## Configuration

### Interactive Setup

The easiest way to configure GCP is using the interactive wizard:

```bash
uv run cooperbench config gcp
```

This wizard will:
1. ✓ Check if gcloud CLI is installed
2. ✓ Verify GCP Python dependencies
3. ✓ Authenticate with your Google account
4. ✓ Set up project ID, region, and zone
5. ✓ Test API access to ensure everything works

### Skip Validation Tests

For faster setup (without API validation):

```bash
uv run cooperbench config gcp --skip-tests
```

### Manual Configuration

You can also configure GCP manually by setting environment variables:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
gcloud auth login
gcloud config set project your-project-id
```

The configuration is saved to:
- macOS: `~/Library/Application Support/cooperbench/config.json`
- Linux: `~/.config/cooperbench/config.json`
- Windows: `%APPDATA%\cooperbench\config.json`

## Enable Required APIs

Make sure these APIs are enabled in your GCP project:

1. **Compute Engine API**
   https://console.cloud.google.com/apis/library/compute.googleapis.com

2. **Cloud Batch API**
   https://console.cloud.google.com/apis/library/batch.googleapis.com

3. **Cloud Storage API**
   https://console.cloud.google.com/apis/library/storage.googleapis.com

The wizard will test these during validation.

## Usage

### Run Benchmarks with GCP

```bash
# Run with GCP backend
uv run cooperbench run --backend gcp -s lite

# Run with specific settings
uv run cooperbench run --backend gcp \
  --setting coop \
  --model gemini/gemini-3-flash-preview \
  --concurrency 10
```

### Evaluate with GCP

```bash
# Evaluate with GCP Batch (recommended for large-scale)
uv run cooperbench eval --backend gcp -n my-experiment

# Evaluate specific tasks
uv run cooperbench eval --backend gcp \
  -n my-experiment \
  -r llama_index_task \
  --concurrency 50
```

## Advanced Configuration

### Custom VM Images (Optional)

**This is an optional optimization for large-scale evaluations.**

By default, each evaluation VM pulls Docker images on-demand, which adds ~2-5 minutes to startup time. For large-scale runs (100+ tasks), you can pre-build a VM image with all Docker images cached.

**Step 1: Build the custom image**

This script creates a VM image with all CooperBench Docker images pre-pulled:

```bash
# Set your project
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Run the build script (takes ~30-60 minutes)
./scripts/build_gcp_vm_image.sh
```

The script will:
1. Create a temporary VM with Container-Optimized OS
2. Pull all CooperBench Docker images from the registry
3. Create a VM image snapshot (family: `cooperbench-eval`)
4. Delete the temporary VM

**Step 2: Use the custom image**

After the image is built, reference it in your code:

```python
from cooperbench.eval.backends.gcp import GCPBatchEvaluator

evaluator = GCPBatchEvaluator(
    vm_image="cooperbench-eval"  # Uses the image family created by the script
)
```

Or via environment variable:
```bash
export COOPERBENCH_VM_IMAGE="cooperbench-eval"
```

**When to use this:**
- Large evaluation runs (100+ tasks)
- Repeated evaluations where startup time matters
- When you want to minimize per-task costs

**When NOT to use this:**
- Small runs (< 100 tasks) - not worth the build time
- First-time setup - stick with defaults until you need the optimization

### Custom VPC Network

For multi-agent git collaboration (optional):

```python
from cooperbench.agents.mini_swe_agent.environments.gcp import GCPEnvironment

env = GCPEnvironment(
    image="python:3.11",
    network="cooperbench-vpc"  # Optional: Your VPC network for agent communication
)
```

Note: VPC networking is only required for git-based collaboration between agents. For solo runs or Redis-based messaging, the default network is sufficient.

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

## Cost Estimation

GCP costs vary by region and usage. Approximate costs:

### Agent Execution
- **VM Type**: e2-medium (2 vCPU, 4GB RAM)
- **Cost**: ~$0.03/hour
- **Typical Task**: 5-30 minutes

### Evaluation (Batch)
- **VM Type**: 4 vCPU, 16GB RAM
- **Cost**: ~$0.15/hour per VM
- **Parallelism**: 50 VMs = ~$7.50/hour
- **Typical Job**: 10-30 minutes for 500 tasks

### Storage (GCS)
- **Storage**: ~$0.02/GB/month
- **Operations**: ~$0.005 per 10,000 operations
- **Lifecycle**: Auto-delete after 7 days

**Total for 1000 task evaluations**: ~$5-10

## Troubleshooting

### Authentication Errors

```
Error: Could not automatically determine credentials
Error: Your default credentials were not found
```

**Solution:**

You need **both** types of authentication:

```bash
# 1. Authenticate gcloud CLI
gcloud auth login

# 2. Set up Application Default Credentials (required for Python SDK)
gcloud auth application-default login
```

**Why both?**
- `gcloud auth login`: For gcloud CLI commands
- `gcloud auth application-default login`: For GCP Python libraries (used by CooperBench)

The configuration wizard sets up both automatically.

### API Not Enabled

```
Error: API [compute.googleapis.com] not enabled
```

**Solution:**
1. Enable APIs: https://console.cloud.google.com/apis/library
2. Or run: `gcloud services enable compute.googleapis.com batch.googleapis.com storage.googleapis.com`

### Quota Exceeded

```
Error: Quota exceeded for quota metric 'cpus' and limit 'cpus per project per region'
```

**Solution:**
1. Check quotas: https://console.cloud.google.com/iam-admin/quotas
2. Request quota increase
3. Reduce `--concurrency` parameter

### Permission Denied

```
Error: The caller does not have permission
```

**Solution:**
1. Ensure billing is enabled
2. Check IAM permissions: https://console.cloud.google.com/iam-admin/iam
3. Required roles:
   - `roles/compute.instanceAdmin.v1`
   - `roles/batch.jobsEditor`
   - `roles/storage.admin`

## Migration from Modal

To migrate from Modal to GCP:

```bash
# 1. Install GCP dependencies
uv pip install 'cooperbench[gcp]'

# 2. Configure GCP
uv run cooperbench config gcp

# 3. Run with GCP backend (instead of default modal)
uv run cooperbench run --backend gcp -s lite

# 4. Evaluate with GCP
uv run cooperbench eval --backend gcp -n experiment-name
```

All other commands and options remain the same!

## Support

For issues or questions:
- GitHub Issues: https://github.com/cooperbench/CooperBench/issues
- GCP Documentation: https://cloud.google.com/docs
- CooperBench Docs: https://cooperbench.com
