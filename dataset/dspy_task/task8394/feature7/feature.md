**Title**: Implement Cache Usage Tracking for DSPy Modules

**Pull Request Details**

**Description**:
This patch introduces a mechanism to track cache hits (memory and disk) and misses during DSPy program execution. It allows users to monitor how often language model responses are retrieved from the cache versus generated via fresh API calls.

**Technical Background**:
While DSPy previously tracked token usage for language models, it lacked visibility into the performance of its caching layer. Understanding cache efficiency is critical for optimizing latency and costs in LLM applications, especially when dealing with large-scale data processing or iterative prompt engineering.

**Solution**:
The implementation introduces a `CacheUsageTracker` class and a corresponding `track_cache_usage` context manager. Key changes include:
- **Cache Logic**: Updated `dspy.clients.cache.Cache` to notify the tracker of memory hits, disk hits, or misses during retrieval.
- **Primitives**: Added `get_cache_usage` and `set_cache_usage` methods to the `Prediction` class to store tracking data.
- **Program Execution**: Modified `dspy.Module` (both sync and async) to automatically wrap execution in a cache tracking context when `settings.track_usage` is enabled, attaching the results to the module's output.
- **Usage Tracker**: Created a new utility in `dspy.utils.usage_tracker` to manage the lifecycle of cache statistics.

**Files Modified**
- `dspy/clients/cache.py`
- `dspy/primitives/prediction.py`
- `dspy/primitives/program.py`
- `dspy/utils/usage_tracker.py`