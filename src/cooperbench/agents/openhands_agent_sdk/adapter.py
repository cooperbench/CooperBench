"""OpenHands SDK adapter for CooperBench.

This adapter runs the OpenHands agent-server in Modal and connects to it
using the SDK's RemoteWorkspace.

For coop mode, it creates a shared ModalRedisServer for inter-agent messaging.
The adapter handles its own infrastructure - no external Redis needed.
"""

import os
import time
import logging
import threading
import json
from typing import Any

import modal
from cooperbench.agents import AgentResult
from cooperbench.agents.registry import register
from cooperbench.agents.openhands_agent_sdk.utils import wait_for_git_server, git_push_with_retry

logger = logging.getLogger(__name__)

# Disable all OpenHands SDK logging
logging.getLogger("openhands").setLevel(logging.CRITICAL)
logging.getLogger("openhands.sdk").setLevel(logging.CRITICAL)
logging.getLogger("openhands.tools").setLevel(logging.CRITICAL)
logging.getLogger("openhands.workspace").setLevel(logging.CRITICAL)


# Modal app for running agent-server and infrastructure
modal_app = modal.App("cooperbench")

# Module-level shared Redis server for all coop runs
# All concurrent tasks share ONE Redis server, with namespacing via URL fragment
_shared_redis: Any = None  # ModalRedisServer instance
_redis_lock = threading.Lock()
_redis_refcount: int = 0  # Total number of active agents using Redis

# Module-level shared Git server for coop runs with git enabled
# Unlike Redis (shared across all runs), Git server is per-run to isolate repos
_git_servers: dict[str, Any] = {}  # run_id -> ModalGitServer
_git_lock = threading.Lock()
_git_refcounts: dict[str, int] = {}  # run_id -> refcount


def _get_or_create_redis(run_id: str, agents: list[str], timeout: int = 3600) -> str:
    """Get or create a shared ModalRedisServer for coop runs.
    
    Thread-safe: First caller creates the server, all others reuse it.
    Returns a namespaced Redis URL: redis://host:port#run:{run_id}
    
    The namespace prefix ensures concurrent runs don't interfere with each other.
    """
    global _shared_redis, _redis_refcount
    from cooperbench.agents.openhands_agent_sdk.connectors import ModalRedisServer
    
    with _redis_lock:
        if _shared_redis is None:
            app = modal.App.lookup("cooperbench", create_if_missing=True)
            _shared_redis = ModalRedisServer.create(
                app=app,
                run_id="shared",  # Single shared server
                agents=agents,
                timeout=timeout,
            )
        
        _redis_refcount += 1
        # Return namespaced URL so each run has isolated keys
        return f"{_shared_redis.url}#run:{run_id}"


def _release_redis() -> None:
    """Release a reference to the shared Redis server.
    
    When refcount reaches 0, the server is cleaned up.
    """
    global _shared_redis, _redis_refcount
    
    with _redis_lock:
        if _redis_refcount <= 0:
            return
        
        _redis_refcount -= 1
        
        if _redis_refcount <= 0 and _shared_redis is not None:
            try:
                _shared_redis.cleanup()
            except Exception:
                pass  # Ignore cleanup errors
            _shared_redis = None


def _get_or_create_git_server(run_id: str, agents: list[str], timeout: int = 3600) -> str:
    """Get or create a ModalGitServer for a specific run.
    
    Thread-safe: First caller for a run_id creates the server, others reuse it.
    Each run gets its own git server (unlike Redis which is shared).
    
    Returns:
        Git URL (e.g., git://host:port/repo.git)
    """
    global _git_servers, _git_refcounts
    from cooperbench.agents.openhands_agent_sdk.connectors import ModalGitServer
    
    with _git_lock:
        if run_id not in _git_servers:
            app = modal.App.lookup("cooperbench", create_if_missing=True)
            _git_servers[run_id] = ModalGitServer.create(
                app=app,
                run_id=run_id,
                agents=agents,
                timeout=timeout,
            )
            _git_refcounts[run_id] = 0
        
        _git_refcounts[run_id] += 1
        return _git_servers[run_id].url


def _release_git_server(run_id: str) -> None:
    """Release a reference to a run's git server.
    
    When refcount reaches 0, the server is cleaned up.
    """
    global _git_servers, _git_refcounts
    
    with _git_lock:
        if run_id not in _git_refcounts:
            return
        
        _git_refcounts[run_id] -= 1
        
        if _git_refcounts[run_id] <= 0:
            if run_id in _git_servers:
                try:
                    _git_servers[run_id].cleanup()
                except Exception:
                    pass  # Ignore cleanup errors
                del _git_servers[run_id]
            if run_id in _git_refcounts:
                del _git_refcounts[run_id]


def _needs_modal_redis(comm_url: str | None) -> bool:
    """Check if we need to create a Modal Redis server.
    
    Returns True if:
    - No comm_url provided
    - comm_url points to localhost (not reachable from Modal)
    """
    if not comm_url:
        return True
    # localhost/127.0.0.1 can't be reached from Modal sandboxes
    return "localhost" in comm_url or "127.0.0.1" in comm_url


def _parse_redis_url(redis_url: str) -> tuple[str, str]:
    """Parse Redis URL and extract namespace prefix.
    
    Args:
        redis_url: URL like "redis://host:port" or "redis://host:port#run:abc123"
        
    Returns:
        Tuple of (clean_url, prefix) where prefix includes trailing colon if present
    """
    if "#" in redis_url:
        url, prefix = redis_url.split("#", 1)
        return url, prefix + ":"
    return redis_url, ""


def _retrieve_sent_messages(redis_url: str, agent_id: str) -> list[dict]:
    """Retrieve sent messages from Redis for conversation extraction.
    
    The SendMessageExecutor stores a copy of each sent message in a
    {prefix}{agent_id}:sent_messages key for later retrieval.
    """
    import json
    try:
        import redis
        url, prefix = _parse_redis_url(redis_url)
        client = redis.from_url(url)
        log_key = f"{prefix}{agent_id}:sent_messages"
        
        messages = []
        raw_messages = client.lrange(log_key, 0, -1)
        
        for raw in raw_messages:
            try:
                msg = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                messages.append(msg)
            except json.JSONDecodeError:
                continue
        return messages
    except Exception as e:
        logger.warning(f"Failed to retrieve sent messages from Redis: {e}")
        return []


def _extract_patch(workspace: Any, base_commit: str | None) -> str:
    """Extract git diff patch from workspace.
    
    Captures all changes: staged, unstaged, and new untracked files.
    Uses `git add -A` first to ensure new files are included in the diff.
    
    Args:
        workspace: RemoteWorkspace instance
        base_commit: Base commit SHA to diff from
        
    Returns:
        Patch content as string, or empty string on failure
    """
    if not base_commit or not workspace:
        return ""
    
    try:
        # Stage all changes (including new files) so they appear in diff
        workspace.execute_command(
            "git add -A",
            cwd="/workspace/repo",
            timeout=30.0
        )
        # Diff from base commit to working tree (includes all staged/unstaged changes)
        diff_result = workspace.execute_command(
            f"git diff {base_commit}",
            cwd="/workspace/repo",
            timeout=60.0
        )
        return diff_result.stdout if diff_result.exit_code == 0 else ""
    except Exception as e:
        logger.warning(f"Failed to extract patch: {e}")
        return ""


@register("openhands_sdk")
class OpenHandsSDKRunner:
    """Runs OpenHands SDK agent with remote execution in Modal.
    
    This adapter:
    1. Starts the agent-server Docker image in Modal
    2. Connects to it via RemoteWorkspace
    3. Runs the OpenHands agent with default tools
    4. Collects the patch and trajectory
    
    Note: This adapter expects images with the `-oh` suffix (e.g., task17244-oh)
    which include the OpenHands agent-server. If a base image is passed
    (e.g., task17244), the `-oh` suffix is automatically appended.
    """

    def __init__(self, max_iterations: int = 100, timeout: int = 3600, cost_limit: float = 2.0):
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.cost_limit = cost_limit

    def _get_oh_image(self, image: str) -> str:
        """Convert base image to agent-server image (add -oh suffix if needed)."""
        if "-oh" in image:
            # Already an OH image - normalize to just -oh (remove version suffixes)
            import re
            return re.sub(r'-oh(-v\d+)?$', '-oh', image)
        # Split image:tag and append -oh to tag
        if ":" in image:
            base, tag = image.rsplit(":", 1)
            return f"{base}:{tag}-oh"
        # No tag specified
        return f"{image}-oh"
    
    def _setup_git_remote(self, workspace, git_url: str, agent_id: str) -> None:
        """Configure git remote in the agent's sandbox for collaboration.
        
        Sets up the 'team' remote pointing to the shared git server,
        creates an agent-specific branch, and pushes the initial state.
        
        Args:
            workspace: RemoteWorkspace instance
            git_url: Git server URL (e.g., git://host:port/repo.git)
            agent_id: This agent's identifier
        """
        REMOTE_NAME = "team"
        
        # Configure git user (needed for commits)
        workspace.execute_command('git config user.email "agent@cooperbench.local"', cwd="/workspace/repo", timeout=10.0)
        workspace.execute_command(f'git config user.name "{agent_id}"', cwd="/workspace/repo", timeout=10.0)
        
        # Add shared remote (or update if exists)
        result = workspace.execute_command(f"git remote add {REMOTE_NAME} {git_url}", cwd="/workspace/repo", timeout=10.0)
        if result.exit_code != 0:
            # Remote might already exist, update URL
            workspace.execute_command(f"git remote set-url {REMOTE_NAME} {git_url}", cwd="/workspace/repo", timeout=10.0)
        
        # Wait for git server to be reachable (with tenacity retry)
        wait_for_git_server(workspace, git_url)
        
        # Create agent's branch
        workspace.execute_command(f"git checkout -b {agent_id}", cwd="/workspace/repo", timeout=10.0)
        
        # Push initial state with retry (first agent initializes the server)
        if not git_push_with_retry(workspace, REMOTE_NAME, agent_id, force=True):
            logger.error(f"Initial git push failed for {agent_id} after retries")
        
        # Also push main/master as base reference
        workspace.execute_command(
            f"git push {REMOTE_NAME} HEAD:refs/heads/main --force 2>/dev/null || true",
            cwd="/workspace/repo",
            timeout=30.0,
        )

    def run(
        self,
        task: str,
        image: str,
        *,
        agent_id: str = "agent",
        model_name: str = "gpt-4o",
        # Collaboration options
        agents: list[str] | None = None,
        comm_url: str | None = None,
        git_server_url: str | None = None,
        git_enabled: bool = False,
        messaging_enabled: bool = True,
        config: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run the OpenHands agent on a task.
        
        Args:
            task: The task description (feature spec)
            image: Docker image (base or with -oh suffix). If base image is passed,
                   -oh suffix is automatically appended.
            agent_id: Unique identifier for this agent
            model_name: LLM model to use
            agents: List of all agent IDs (for collaboration)
            comm_url: Redis URL for inter-agent messaging (created if not provided in coop mode)
            git_server_url: Git server URL for code sharing (not yet supported)
            git_enabled: Whether git collaboration is enabled
            messaging_enabled: Whether messaging is enabled
            config: Agent-specific configuration
            
        Returns:
            AgentResult with status, patch, cost, steps, messages
        """
        # Convert to agent-server image if needed
        oh_image = self._get_oh_image(image)

        # Track state
        total_cost = 0.0
        messages = []
        sent_messages = []
        steps = 0
        patch = ""
        status = "Error"
        error = None
        
        # Determine if this is a coop run
        is_coop = (messaging_enabled or git_enabled) and agents and len(agents) > 1
        redis_url = comm_url
        # OpenHands adapter manages its own git server - ignore git_server_url from coop.py
        # This ensures git setup works correctly with RemoteWorkspace
        git_url = None
        run_id = None
        owns_redis = False  # Track if we need to release Redis reference
        owns_git = False  # Track if we need to release Git server reference
        
        if is_coop:
            # Extract run_id from config or comm_url namespace
            config = config or {}
            if comm_url and "#run:" in comm_url:
                # Extract run_id from namespaced URL: redis://host:port#run:abc123
                run_id = comm_url.split("#run:")[1]
            else:
                run_id = config.get("run_id")
            
            # Generate run_id if not provided
            if not run_id:
                import uuid
                run_id = uuid.uuid4().hex[:8]
            
            # Create Modal Redis if needed (localhost not reachable from Modal)
            if messaging_enabled and _needs_modal_redis(comm_url):
                redis_url = _get_or_create_redis(run_id, agents, self.timeout)
                owns_redis = True
            
            # Create Modal Git server if git is enabled
            # OpenHands adapter always creates its own git server (ignores git_server_url from coop.py)
            # to ensure git setup works correctly with RemoteWorkspace
            if git_enabled:
                git_url = _get_or_create_git_server(run_id, agents, self.timeout)
                owns_git = True

        try:
            # Build coop_info for both sandbox env vars AND agent system prompt
            coop_info = {
                "redis_url": redis_url,
                "git_url": git_url,
                "agent_id": agent_id,
                "agents": agents or [],
                "messaging_enabled": redis_url is not None,
                "git_enabled": git_enabled and git_url is not None,
            } if is_coop else None
            
            with ModalSandboxContext(oh_image, self.timeout, coop_info=coop_info) as sandbox_url:

                # Import SDK components
                from openhands.sdk import LLM
                from openhands.sdk.conversation import RemoteConversation
                from openhands.sdk.workspace import RemoteWorkspace
                from openhands.tools.preset.default import get_default_agent

                # Create LLM instance (will be serialized and sent to server)
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
                llm = LLM(model=model_name, api_key=api_key)

                # Create agent with default tools (terminal, file_editor, task_tracker)
                # Browser tools disabled since we're running headless
                # Collaboration tools (SendMessage/ReceiveMessage) are always registered
                # but only active when REDIS_URL env var is set in the sandbox
                # Pass coop_info to inject collaboration instructions into system prompt
                agent = get_default_agent(llm=llm, cli_mode=True, coop_info=coop_info)
                
                # Connect to remote workspace (agent-server in Modal)
                workspace = RemoteWorkspace(
                    host=sandbox_url,
                    working_dir="/workspace/repo",
                )
                
                # Capture base commit for patch generation (before any changes)
                try:
                    base_result = workspace.execute_command(
                        "git rev-parse HEAD",
                        cwd="/workspace/repo",
                        timeout=10.0
                    )
                    base_commit = base_result.stdout.strip() if base_result.exit_code == 0 else None
                except Exception as e:
                    logger.warning(f"Failed to get base commit: {e}")
                    base_commit = None
                
                # Set up git remote if git collaboration is enabled
                if coop_info and coop_info.get("git_enabled") and coop_info.get("git_url"):
                    self._setup_git_remote(
                        workspace=workspace,
                        git_url=coop_info["git_url"],
                        agent_id=agent_id,
                    )

                # Callback to collect events
                def event_callback(event):
                    nonlocal steps, sent_messages
                    steps += 1
                    
                    event_data = {
                        "step": steps,
                        "event_type": type(event).__name__,
                        "event": str(event),
                    }
                    
                    # Extract message details for SendMessageAction
                    event_str = str(event)
                    if "SendMessageAction" in event_str:
                        import time
                        action = getattr(event, 'action', None)
                        recipient = getattr(action, 'recipient', None) if action else None
                        content = getattr(action, 'content', None) if action else None
                        
                        if recipient and content:
                            # Add to event_data for trajectory visibility (use different names to avoid extraction duplication)
                            event_data["to"] = recipient
                            event_data["msg"] = content
                            # Add to sent_messages for conversation extraction
                            sent_messages.append({
                                "from": agent_id,
                                "to": recipient,
                                "content": content,
                                "step": steps,
                                "timestamp": time.time(),
                            })
                    
                    messages.append(event_data)

                # Create remote conversation - agent loop runs on server
                # visualizer=None disables the verbose Rich output
                conversation = RemoteConversation(
                    agent=agent,
                    workspace=workspace,
                    max_iteration_per_run=self.max_iterations,
                    callbacks=[event_callback],
                    visualizer=None,
                )

                # Send task and run the conversation
                # Message checking for coop mode happens inside the agent loop
                # (in LocalConversation._check_inbox_messages before each step)
                conversation.send_message(task)
                conversation.run(blocking=True, timeout=float(self.timeout))
                    
                # Get the patch from remote workspace
                patch = _extract_patch(workspace, base_commit)
                
                # Get cost from conversation stats (via state.stats which fetches from remote)
                try:
                    state = conversation.state
                    stats = state.stats
                    if stats:
                        combined_metrics = stats.get_combined_metrics()
                        total_cost = combined_metrics.accumulated_cost or 0.0
                        
                        # Check cost limit
                        if self.cost_limit > 0 and total_cost >= self.cost_limit:
                            status = "CostLimitExceeded"
                        else:
                            status = "Submitted"
                except Exception as e:
                    logger.warning(f"Failed to get cost: {e}")
                    pass  # Cost tracking optional
                    status = "Submitted"
                
                # sent_messages is already populated from event_callback above
                # No need to retrieve from Redis - we extract directly from trajectory

        except Exception as e:
            error_str = str(e)
            # MaxIterationsReached is expected behavior, not an error
            if "MaxIterationsReached" in error_str:
                logger.debug(f"Agent reached max iterations: {e}")
                status = "Submitted"  # Still consider it submitted
                error = None  # Not an error condition
                # Still extract patch - agent likely made code changes
                patch = _extract_patch(workspace, base_commit)
            else:
                logger.exception(f"Error running agent: {e}")
                error = error_str
                status = "Error"
                # Try to recover partial progress
                patch = _extract_patch(workspace, base_commit)
        finally:
            # Release Redis reference (cleanup happens when all agents done)
            if owns_redis:
                _release_redis()
            # Release Git server reference
            if owns_git and run_id:
                _release_git_server(run_id)

        return AgentResult(
            status=status,
            patch=patch,
            cost=total_cost,
            steps=steps,
            messages=messages,
            sent_messages=sent_messages,
            error=error,
        )


class ModalSandboxContext:
    """Context manager for Modal sandbox with agent-server.
    
    This starts an agent-server in a Modal sandbox and provides an HTTP URL to connect to it.
    The agent-server runs as the container's entrypoint and exposes port 8000.
    
    Credentials are passed to the sandbox via modal.Secret:
    - GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY from environment
    - Google Cloud credentials from GOOGLE_APPLICATION_CREDENTIALS file
    - For coop mode: REDIS_URL, GIT_URL, AGENT_ID, AGENTS (for collaboration tools)
    """

    def __init__(self, image_name: str, timeout: int, coop_info: dict | None = None):
        """Initialize the context manager.
        
        Args:
            image_name: Docker image name for the agent-server
            timeout: Sandbox timeout in seconds
            coop_info: Optional dict with redis_url, agent_id, agents for coop mode
        """
        self.image_name = image_name
        self.timeout = timeout
        self.coop_info = coop_info
        self._sandbox: modal.Sandbox | None = None
        self._server_proc = None
        self._coop_info = coop_info  # Alias for clarity

    def _collect_credentials(self) -> dict[str, str]:
        """Collect API keys, credentials, and coop info from environment."""
        creds = {}
        
        # Collect API keys and Vertex AI config (litellm reads VERTEXAI_* env vars)
        for key in [
            "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY", 
            "OPENAI_API_KEY",
            "GOOGLE_CLOUD_PROJECT",
            "VERTEXAI_PROJECT",
            "VERTEXAI_LOCATION",
        ]:
            if value := os.environ.get(key):
                creds[key] = value
        
        # Read Google Cloud credentials JSON if available
        gcp_creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        
        # If not explicitly set, check standard gcloud ADC location
        if not gcp_creds_path:
            home = os.path.expanduser("~")
            default_adc_path = os.path.join(home, ".config", "gcloud", "application_default_credentials.json")
            if os.path.exists(default_adc_path):
                gcp_creds_path = default_adc_path
        
        if gcp_creds_path and os.path.exists(gcp_creds_path):
            with open(gcp_creds_path) as f:
                creds_content = f.read()
                creds["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = creds_content
                
                # Extract project from ADC if not already set
                if "VERTEXAI_PROJECT" not in creds:
                    import json
                    try:
                        adc_data = json.loads(creds_content)
                        if project_id := adc_data.get("quota_project_id"):
                            creds["VERTEXAI_PROJECT"] = project_id
                            creds["GOOGLE_CLOUD_PROJECT"] = project_id
                    except json.JSONDecodeError:
                        pass
        
        # Add coop info for collaboration tools
        if self.coop_info:
            if self.coop_info.get("redis_url"):
                creds["REDIS_URL"] = self.coop_info["redis_url"]
            if self.coop_info.get("git_url"):
                creds["GIT_URL"] = self.coop_info["git_url"]
            if self.coop_info.get("agent_id"):
                creds["AGENT_ID"] = self.coop_info["agent_id"]
            if self.coop_info.get("agents"):
                creds["AGENTS"] = ",".join(self.coop_info["agents"])
        
        return creds

    def __enter__(self) -> str:
        """Start sandbox, run agent-server, and return the tunnel URL."""
        
        # Build image and clear entrypoint (Modal will add its own default command)
        image = modal.Image.from_registry(self.image_name).entrypoint([])
        
        # Get or create app
        app = modal.App.lookup("cooperbench", create_if_missing=True)
        
        # Collect credentials and create Modal secret
        creds = self._collect_credentials()
        secrets = [modal.Secret.from_dict(creds)] if creds else []
        
        # Create sandbox with tunnel for port 8000
        self._sandbox = modal.Sandbox.create(
            image=image,
            timeout=self.timeout,
            app=app,
            secrets=secrets,
            # Expose port 8000 for the agent-server
            encrypted_ports=[8000],
        )
        
        
        # IMPORTANT: We use a Python wrapper to ensure collaboration tools are
        # imported in the SAME process as the agent-server. This is needed because
        # Modal may cache Docker images and the __init__.py auto-import might not
        # be in the cached image.
        #
        # We write the wrapper to a file first, then execute it (heredocs don't
        # work well with sandbox.exec's bash -c).
        
        wrapper_script = '''
import sys
import os

sys.argv = ['agent_server', '--host', '0.0.0.0', '--port', '8000']

# Force import collaboration tools to register them BEFORE server starts
try:
    from openhands.tools.collaboration import SendMessageTool, ReceiveMessageTool
    print('[STARTUP] Collaboration tools registered:', SendMessageTool.name, ReceiveMessageTool.name, flush=True)
except Exception as e:
    print('[STARTUP] WARNING: Failed to import collaboration tools:', e, flush=True)

# Now run the agent server (tools are registered in this process)
from openhands.agent_server.__main__ import main
main()
'''

        # Bash script to set up credentials and run the Python wrapper
        startup_script = """
#!/bin/bash
set -e

# Write Google Cloud credentials if provided
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS_JSON" ]; then
    echo "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > /tmp/gcp-credentials.json
    export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-credentials.json
fi

# Run the Python wrapper
exec /opt/agent-server-venv/bin/python /tmp/agent_wrapper.py
"""
        
        # Write the Python wrapper script to the sandbox using base64 (safe encoding)
        import base64
        wrapper_b64 = base64.b64encode(wrapper_script.encode()).decode()
        write_wrapper = self._sandbox.exec("bash", "-c", f"echo '{wrapper_b64}' | base64 -d > /tmp/agent_wrapper.py")
        write_wrapper.wait()
        
        # Start the agent-server manually (since we cleared the entrypoint)
        self._server_proc = self._sandbox.exec(
            "bash", "-c", startup_script,
        )
        
        # Give the server a moment to start and capture initial output
        time.sleep(3)
        
        # Get tunnel URL
        tunnel_info = self._sandbox.tunnels()[8000]
        tunnel_url = tunnel_info.url
        
        # Wait for server to be ready
        self._wait_for_server(tunnel_url)
        
        return tunnel_url

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup sandbox."""
        if self._sandbox:
            try:
                self._sandbox.terminate()
            except Exception as e:
                logger.warning(f"Failed to terminate sandbox: {e}")

    def _wait_for_server(self, url: str, timeout: int = 120):
        """Wait for the agent-server to be ready."""
        import httpx

        start = time.time()
        last_error = None
        
        while time.time() - start < timeout:
            try:
                response = httpx.get(f"{url}/health", timeout=10)
                if response.status_code == 200:
                    return
            except Exception as e:
                last_error = e
            time.sleep(2)

        raise TimeoutError(
            f"Agent-server did not become ready within {timeout}s. "
            f"Last error: {last_error}"
        )
