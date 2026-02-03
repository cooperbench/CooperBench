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

# Module-level shared Redis server for all coop runs
# All concurrent tasks share ONE Redis server, with namespacing via URL fragment
_shared_redis: Any = None  # ModalRedisServer instance
_redis_lock = threading.Lock()
_redis_refcount: int = 0  # Total number of active agents using Redis


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
            app = modal.App.lookup("cooperbench-openhands", create_if_missing=True)
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

        try:
            # Build coop_info for both sandbox env vars AND agent system prompt
            coop_info = {
                "redis_url": redis_url,
                "agent_id": agent_id,
                "agents": agents or [],
                "messaging_enabled": redis_url is not None,
                "git_enabled": git_enabled,
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
                
                # sent_messages is already populated from event_callback above
                # No need to retrieve from Redis - we extract directly from trajectory

        except Exception as e:
            error_str = str(e)
            # MaxIterationsReached is expected behavior, not an error
            if "MaxIterationsReached" in error_str:
                logger.debug(f"Agent reached max iterations: {e}")
                status = "Submitted"  # Still consider it submitted
                error = None  # Not an error condition
            else:
                logger.exception(f"Error running agent: {e}")
                error = error_str
                status = "Error"
        finally:
            # Release Redis reference (cleanup happens when all agents done)
            if owns_redis:
                _release_redis()

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
        
        # Add coop info for SendMessageTool
        if self.coop_info:
            if self.coop_info.get("redis_url"):
                creds["REDIS_URL"] = self.coop_info["redis_url"]
            if self.coop_info.get("agent_id"):
                creds["AGENT_ID"] = self.coop_info["agent_id"]
            if self.coop_info.get("agents"):
                creds["AGENTS"] = ",".join(self.coop_info["agents"])
        
        return creds

    def _get_coop_template(self) -> str:
        """Get the custom system prompt template for coop mode.
        
        This template includes the {{ collaboration }} variable that gets
        populated with team coordination instructions.
        """
        return '''You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.

<ROLE>
* Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.
* If the user asks a question, like "why is X happening", don't try to fix the problem. Just give an answer to the question.
</ROLE>

{% if collaboration %}
{{ collaboration }}
{% endif %}

<MEMORY>
* Use `AGENTS.md` under the repository root as your persistent memory for repository-specific knowledge and context.
* Add important insights, patterns, and learnings to this file to improve future task performance.
* This repository skill is automatically loaded for every conversation and helps maintain context across sessions.
* For more information about skills, see: https://docs.openhands.dev/overview/skills
</MEMORY>

<EFFICIENCY>
* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.
* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.
</EFFICIENCY>

<FILE_SYSTEM_GUIDELINES>
* When a user provides a file path, do NOT assume it's relative to the current working directory. First explore the file system to locate the file before working on it.
* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.
* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.
* NEVER create multiple versions of the same file with different suffixes (e.g., file_test.py, file_fix.py, file_simple.py). Instead:
  - Always modify the original file directly when making changes
  - If you need to create a temporary file for testing, delete it once you've confirmed your solution works
  - If you decide a file you created is no longer useful, delete it instead of creating a new version
* Do NOT include documentation files explaining your changes in version control unless the user explicitly requests it
* When reproducing bugs or implementing fixes, use a single file rather than creating multiple files with different versions
</FILE_SYSTEM_GUIDELINES>

<CODE_QUALITY>
* Write clean, efficient code with minimal comments. Avoid redundancy in comments: Do not repeat information that can be easily inferred from the code itself.
* When implementing solutions, focus on making the minimal changes needed to solve the problem.
* Before implementing any changes, first thoroughly understand the codebase through exploration.
* If you are adding a lot of code to a function or file, consider splitting the function or file into smaller pieces when appropriate.
* Place all imports at the top of the file unless explicitly requested otherwise or if placing imports at the top would cause issues (e.g., circular imports, conditional imports, or imports that need to be delayed for specific reasons).
</CODE_QUALITY>

<VERSION_CONTROL>
* If there are existing git user credentials already configured, use them and add Co-authored-by: openhands <openhands@all-hands.dev> to any commits messages you make. if a git config doesn't exist use "openhands" as the user.name and "openhands@all-hands.dev" as the user.email by default, unless explicitly instructed otherwise.
* Exercise caution with git operations. Do NOT make potentially dangerous changes (e.g., pushing to main, deleting repositories) unless explicitly asked to do so.
* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit. Use `git commit -a` whenever possible.
* Do NOT commit files that typically shouldn't go into version control (e.g., node_modules/, .env files, build directories, cache files, large binaries) unless explicitly instructed by the user.
* If unsure about committing certain files, check for the presence of .gitignore files or ask the user for clarification.
* When running git commands that may produce paged output (e.g., `git diff`, `git log`, `git show`), use `git --no-pager <command>` or set `GIT_PAGER=cat` to prevent the command from getting stuck waiting for interactive input.
</VERSION_CONTROL>

<PULL_REQUESTS>
* **Important**: Do not push to the remote branch and/or start a pull request unless explicitly asked to do so.
* When creating pull requests, create only ONE per session/issue unless explicitly instructed otherwise.
* When working with an existing PR, update it with new commits rather than creating additional PRs for the same issue.
* When updating a PR, preserve the original PR title and purpose, updating description only when necessary.
</PULL_REQUESTS>

<PROBLEM_SOLVING_WORKFLOW>
1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions
2. ANALYSIS: Consider multiple approaches and select the most promising one
3. TESTING:
   * For bug fixes: Create tests to verify issues before implementing fixes
   * For new features: Consider test-driven development when appropriate
   * Do NOT write tests for documentation changes, README updates, configuration files, or other non-functionality changes
   * Do not use mocks in tests unless strictly necessary and justify their use when they are used. You must always test real code paths in tests, NOT mocks.
   * If the repository lacks testing infrastructure and implementing tests would require extensive setup, consult with the user before investing time in building testing infrastructure
   * If the environment is not set up to run tests, consult with the user first before investing time to install all dependencies
4. IMPLEMENTATION:
   * Make focused, minimal changes to address the problem
   * Always modify existing files directly rather than creating new versions with different suffixes
   * If you create temporary files for testing, delete them after confirming your solution works
5. VERIFICATION: If the environment is set up to run tests, test your implementation thoroughly, including edge cases. If the environment is not set up to run tests, consult with the user first before investing time to run tests.
</PROBLEM_SOLVING_WORKFLOW>

<EXTERNAL_SERVICES>
* When interacting with external services like GitHub, GitLab, or Bitbucket, use their respective APIs instead of browser-based interactions whenever possible.
* Only resort to browser-based interactions with these services if specifically requested by the user or if the required operation cannot be performed via API.
</EXTERNAL_SERVICES>

<ENVIRONMENT_SETUP>
* When user asks you to run an application, don't stop if the application is not installed. Instead, please install the application and run the command again.
* If you encounter missing dependencies:
  1. First, look around in the repository for existing dependency files (requirements.txt, pyproject.toml, package.json, Gemfile, etc.)
  2. If dependency files exist, use them to install all dependencies at once (e.g., `pip install -r requirements.txt`, `npm install`, etc.)
  3. Only install individual packages directly if no dependency files are found or if only specific packages are needed
* Similarly, if you encounter missing dependencies for essential tools requested by the user, install them when possible.
</ENVIRONMENT_SETUP>

<TROUBLESHOOTING>
* If you've made repeated attempts to solve a problem but tests still fail or the user reports it's still broken:
  1. Step back and reflect on 5-7 different possible sources of the problem
  2. Assess the likelihood of each possible cause
  3. Methodically address the most likely causes, starting with the highest probability
  4. Explain your reasoning process in your response to the user
* When you run into any major issue while executing a plan from the user, please don't try to directly work around it. Instead, propose a new plan and confirm with the user before proceeding.
</TROUBLESHOOTING>

<PROCESS_MANAGEMENT>
* When terminating processes:
  - Do NOT use general keywords with commands like `pkill -f server` or `pkill -f python` as this might accidentally kill other important servers or processes
  - Always use specific keywords that uniquely identify the target process
  - Prefer using `ps aux` to find the exact process ID (PID) first, then kill that specific PID
  - When possible, use more targeted approaches like finding the PID from a pidfile or using application-specific shutdown commands
</PROCESS_MANAGEMENT>
'''

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
        
        # Write the custom coop system prompt template to the sandbox (if coop mode)
        if self._coop_info:
            coop_template = self._get_coop_template()
            template_b64 = base64.b64encode(coop_template.encode()).decode()
            write_template = self._sandbox.exec("bash", "-c", f"echo '{template_b64}' | base64 -d > /tmp/system_prompt_coop.j2")
            write_template.wait()
        
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
