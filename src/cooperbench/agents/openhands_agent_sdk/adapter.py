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

logger = logging.getLogger(__name__)

# Disable all OpenHands SDK logging
logging.getLogger("openhands").setLevel(logging.CRITICAL)
logging.getLogger("openhands.sdk").setLevel(logging.CRITICAL)
logging.getLogger("openhands.tools").setLevel(logging.CRITICAL)
logging.getLogger("openhands.workspace").setLevel(logging.CRITICAL)


# Modal app for running agent-server and infrastructure
modal_app = modal.App("cooperbench-openhands")

# Module-level registry for shared Redis servers (keyed by run_id)
# This allows multiple parallel agents in the same coop run to share one Redis
_redis_servers: dict[str, Any] = {}  # run_id -> ModalRedisServer
_redis_lock = threading.Lock()
_redis_refcount: dict[str, int] = {}  # run_id -> number of agents using it


def _get_or_create_redis(run_id: str, agents: list[str], timeout: int = 3600) -> str:
    """Get or create a shared ModalRedisServer for a coop run.
    
    Thread-safe: First agent to call this creates the server, others reuse it.
    Returns the Redis URL that all agents can connect to.
    """
    from cooperbench.agents.openhands_agent_sdk.connectors import ModalRedisServer
    
    with _redis_lock:
        if run_id not in _redis_servers:
            app = modal.App.lookup("cooperbench-openhands", create_if_missing=True)
            server = ModalRedisServer.create(
                app=app,
                run_id=run_id,
                agents=agents,
                timeout=timeout,
            )
            _redis_servers[run_id] = server
            _redis_refcount[run_id] = 0
        
        _redis_refcount[run_id] += 1
        return _redis_servers[run_id].url


def _release_redis(run_id: str) -> None:
    """Release a reference to the shared Redis server.
    
    When refcount reaches 0, the server is cleaned up.
    """
    with _redis_lock:
        if run_id not in _redis_refcount:
            return
        
        _redis_refcount[run_id] -= 1
        
        if _redis_refcount[run_id] <= 0:
            if run_id in _redis_servers:
                try:
                    _redis_servers[run_id].cleanup()
                except Exception as e:
                    logger.warning(f"[{run_id}] Failed to cleanup Redis: {e}")
                del _redis_servers[run_id]
            del _redis_refcount[run_id]


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


def _retrieve_sent_messages(redis_url: str, agent_id: str) -> list[dict]:
    """Retrieve sent messages from Redis for conversation extraction.
    
    The SendMessageExecutor stores a copy of each sent message in a
    {agent_id}:sent_messages key for later retrieval.
    """
    import json
    try:
        import redis
        client = redis.from_url(redis_url)
        log_key = f"{agent_id}:sent_messages"
        messages = []
        # Read all messages without consuming them
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
        is_coop = messaging_enabled and agents and len(agents) > 1
        redis_url = comm_url
        run_id = None
        owns_redis = False  # Track if we need to release Redis reference
        
        if is_coop:
            from cooperbench.agents.openhands_agent_sdk.messaging import (
                get_collaboration_system_prompt,
            )
            
            # Extract run_id from config or comm_url namespace
            config = config or {}
            if comm_url and "#run:" in comm_url:
                # Extract run_id from namespaced URL: redis://host:port#run:abc123
                run_id = comm_url.split("#run:")[1]
            else:
                run_id = config.get("run_id")
            
            # Create Modal Redis if needed (localhost not reachable from Modal)
            if _needs_modal_redis(comm_url):
                if not run_id:
                    # Generate run_id if not provided
                    import uuid
                    run_id = uuid.uuid4().hex[:8]
                
                redis_url = _get_or_create_redis(run_id, agents, self.timeout)
                owns_redis = True
            
            # Add collaboration instructions to task
            collab_prompt = get_collaboration_system_prompt(
                agent_id=agent_id,
                agents=agents,
                messaging_enabled=redis_url is not None,
                git_enabled=git_enabled,
            )
            task = task + collab_prompt

        try:
            # Start agent-server in Modal
            # Pass coop info for messaging tools inside the sandbox
            coop_info = {
                "redis_url": redis_url,
                "agent_id": agent_id,
                "agents": agents or [],
            } if is_coop and redis_url else None
            
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
                agent = get_default_agent(llm=llm, cli_mode=True)
                
                # Connect to remote workspace (agent-server in Modal)
                workspace = RemoteWorkspace(
                    host=sandbox_url,
                    working_dir="/workspace/repo",
                )

                # Callback to collect events
                def event_callback(event):
                    nonlocal steps, sent_messages
                    steps += 1
                    
                    event_data = {
                        "step": steps,
                        "event_type": type(event).__name__,
                        "event": str(event)[:500],  # Truncate long events
                    }
                    
                    # Extract message details for SendMessageAction
                    # The event string contains the action details
                    event_str = str(event)
                    if "SendMessageAction" in event_str:
                        # Try to extract recipient and content from the event string
                        import re
                        recipient_match = re.search(r'recipient["\s:=]+["\']?(\w+)', event_str)
                        content_match = re.search(r'content["\s:=]+["\'](.+?)["\'](?:\s|,|$)', event_str, re.DOTALL)
                        if recipient_match:
                            event_data["message_recipient"] = recipient_match.group(1)
                        if content_match:
                            event_data["message_content"] = content_match.group(1)[:200]  # Truncate content
                    
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
                conversation.send_message(task)
                
                # In coop mode, use non-blocking run with message polling
                # This mirrors mini_swe_agent's pattern of checking inbox before each LLM call
                if is_coop and redis_url:
                    import redis as redis_client
                    redis_conn = redis_client.from_url(redis_url)
                    inbox_key = f"{agent_id}:inbox"
                    
                    conversation.run(blocking=False)
                    
                    start_time = time.time()
                    poll_interval = 2.0  # Check for messages every 2 seconds
                    
                    while True:
                        # Check if conversation is finished
                        try:
                            state = conversation.state
                            exec_status = getattr(state, 'execution_status', None)
                            # exec_status is an enum like ConversationExecutionStatus.FINISHED
                            exec_str = str(exec_status).lower() if exec_status else ""
                            if 'finished' in exec_str or 'error' in exec_str:
                                break
                        except Exception as e:
                            logger.debug(f"Error checking state: {e}")
                        
                        # Check for timeout
                        elapsed = time.time() - start_time
                        if elapsed > self.timeout:
                            logger.warning(f"Timeout after {elapsed:.0f}s")
                            break
                        
                        # Check cost limit
                        if self.cost_limit > 0:
                            try:
                                state = conversation.state
                                stats = state.stats
                                if stats:
                                    combined_metrics = stats.get_combined_metrics()
                                    current_cost = combined_metrics.accumulated_cost or 0.0
                                    if current_cost >= self.cost_limit:
                                        # Stop the conversation
                                        conversation.stop()
                                        break
                            except Exception as e:
                                logger.debug(f"Error checking cost: {e}")
                        
                        # Poll Redis for incoming messages (like mini_swe_agent does each step)
                        try:
                            while True:
                                raw = redis_conn.lpop(inbox_key)
                                if raw is None:
                                    break
                                msg = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                                sender = msg.get("from", "unknown")
                                content = msg.get("content", "")
                                # Inject message into conversation (mirrors mini_swe_agent pattern)
                                injected_msg = f"[Message from {sender}]: {content}"
                                conversation.send_message(injected_msg)
                        except Exception as e:
                            logger.debug(f"Error polling Redis: {e}")
                        
                        time.sleep(poll_interval)
                else:
                    # Solo mode: simple blocking run
                    conversation.run(blocking=True, timeout=float(self.timeout))
                    
                # Get the patch from remote workspace
                # Note: workspace.git_diff() has a bug (Path vs str URL), so use execute_command
                try:
                    diff_result = workspace.execute_command(
                        "git diff HEAD",
                        cwd="/workspace/repo",
                        timeout=60.0
                    )
                    patch = diff_result.stdout if diff_result.exit_code == 0 else ""
                except Exception as e:
                    logger.warning(f"Failed to get git diff: {e}")
                    patch = ""
                
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
                
                
                # Retrieve sent messages from Redis for conversation extraction
                if is_coop and redis_url:
                    sent_messages = _retrieve_sent_messages(redis_url, agent_id)

        except Exception as e:
            logger.exception(f"Error running agent: {e}")
            error = str(e)
            status = "Error"
        finally:
            # Release Redis reference (cleanup happens when all agents done)
            if owns_redis and run_id:
                _release_redis(run_id)

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
    - For coop mode: REDIS_URL, AGENT_ID, AGENTS (for SendMessageTool)
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
        
        # Add coop info for SendMessageTool
        if self.coop_info:
            if self.coop_info.get("redis_url"):
                creds["REDIS_URL"] = self.coop_info["redis_url"]
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
        app = modal.App.lookup("cooperbench-openhands", create_if_missing=True)
        
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
import traceback

sys.argv = ['agent_server', '--host', '0.0.0.0', '--port', '8000']

# Debug: Write to file to verify wrapper is running
log = open('/tmp/wrapper_debug.log', 'w')
log.write('Wrapper started\\n')
log.write('REDIS_URL=' + os.environ.get('REDIS_URL', 'NOT_SET') + '\\n')
log.write('AGENT_ID=' + os.environ.get('AGENT_ID', 'NOT_SET') + '\\n')
log.flush()

# Force import collaboration tools to register them BEFORE server starts
try:
    from openhands.tools.collaboration import SendMessageTool, ReceiveMessageTool
    log.write('Tools imported: ' + SendMessageTool.name + ', ' + ReceiveMessageTool.name + '\\n')
    log.flush()
    print('[STARTUP] Collaboration tools registered:', SendMessageTool.name, ReceiveMessageTool.name, flush=True)
except Exception as e:
    log.write('Import failed: ' + str(e) + '\\n')
    log.write(traceback.format_exc() + '\\n')
    log.flush()
    print('[STARTUP] WARNING: Failed to import collaboration tools:', e, flush=True)

# Now run the agent server (tools are registered in this process)
try:
    from openhands.agent_server.__main__ import main
    log.write('Starting agent server main()\\n')
    log.flush()
    log.close()
    main()
except Exception as e:
    log.write('Server failed: ' + str(e) + '\\n')
    log.write(traceback.format_exc() + '\\n')
    log.flush()
    log.close()
    raise
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
