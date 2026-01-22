# CooperBench

Multi-agent coordination benchmark for collaborative code generation.

## Installation

```bash
pip install -e .
```

## Settings

CooperBench supports four experiment settings:

- **SINGLE**: Single agent working on a single feature
- **SOLO**: Single agent working on two features simultaneously  
- **COOP**: Two agents collaborating with communication
- **COOP_ABLATION**: Two agents without communication (ablation study)

## Usage

```python
from cooperbench import BenchSetting

setting = BenchSetting.COOP
print(setting.value)  # "coop"
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/
```
