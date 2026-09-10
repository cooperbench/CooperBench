"""OpenHands SDK adapter for CooperBench.

This adapter runs the OpenHands agent-server in Modal and connects to it
using the SDK's RemoteWorkspace.

For coop mode, it creates a shared ModalRedisServer for inter-agent messaging.
The adapter handles its own infrastructure - no external Redis needed.
"""

import json
import logging
import os
import threading
import time
from typing import Any

import modal

from cooperbench.agents import AgentResult
from cooperbench.agents._coop.runtime import rewrite_comm_url_for_container
from cooperbench.agents.openhands_agent_sdk.utils import git_push_with_retry, wait_for_git_server
from cooperbench.agents.registry import register

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


def _submission_instructions(*, is_coop: bool) -> str:
    """Submission block appended to the user task message.

    Mirrors mini_swe_agent_v2's ``coop.yaml`` submission section so both
    adapters extract patches the same way: the agent writes a unified diff
    to ``patch.txt`` and the harness reads it. OpenHands ends via its own
    ``finish`` flow rather than the mini-swe echo signal, so the wording
    is adapted accordingly. In coop runs we also remind the agent that
    overlapping line edits cause both patches to be discarded.
    """
    coop_reminder = (
        "\nThe goal is for your patch to NOT conflict with your teammate's when our\n"
        "harness merges them — if any of your edits touch the same lines they\n"
        "touched, both patches are thrown away. Keep `patch.txt` scoped to the\n"
        "files and hunks you actually need.\n"
        if is_coop
        else ""
    )
    return (
        "\n\n## Submission\n\n"
        "`patch.txt` is the artifact we evaluate — write whatever unified diff\n"
        "you want to submit to that file, however it makes sense given how you\n"
        "worked.\n\n"
        "Write the patch (one common way — `git diff` of your in-place edits):\n\n"
        "```bash\n"
        "git diff -- path/to/file1 path/to/file2 > patch.txt\n"
        "```\n\n"
        "Verify it contains what you intend:\n\n"
        "```bash\n"
        "cat patch.txt\n"
        "```\n"
        f"{coop_reminder}"
        "\nOnce `patch.txt` contains the diff you want submitted, finish the task.\n\n"
        "The patch must be a unified diff and contain only source files you\n"
        "intentionally modified. Exclude:\n\n"
        "- reproduction or scratch test scripts you wrote\n"
        "- helper scripts or tools you created\n"
        "- installation, build, packaging, or configuration files\n"
        "- binaries or compiled files\n\n"
        "<CRITICAL>\n"
        "Do NOT run `rm -rf .git`, `git init`, `git rm -rf .`, or `git reset --hard`\n"
        "inside `/workspace/repo` — these corrupt `.git/` and your patch will be\n"
        "unapplyable.\n"
        "</CRITICAL>\n"
    )


def _collect_sandbox_credentials(
    coop_info: dict | None,
    *,
    rewrite_localhost: bool,
) -> dict[str, str]:
    """Collect API keys, credentials, and coop info as env vars for the agent-server.

    Shared by ModalSandboxContext and DockerSandboxContext. When
    ``rewrite_localhost`` is True, ``REDIS_URL`` is rewritten so that
    ``localhost`` / ``127.0.0.1`` resolve to the host gateway from
    inside a docker container (mirrors what every other adapter does).
    """
    creds: dict[str, str] = {}

    for key in [
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "GOOGLE_CLOUD_PROJECT",
        "VERTEXAI_PROJECT",
        "VERTEXAI_LOCATION",
    ]:
        if value := os.environ.get(key):
            creds[key] = value

    gcp_creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not gcp_creds_path:
        home = os.path.expanduser("~")
        default_adc_path = os.path.join(home, ".config", "gcloud", "application_default_credentials.json")
        if os.path.exists(default_adc_path):
            gcp_creds_path = default_adc_path

    if gcp_creds_path and os.path.exists(gcp_creds_path):
        with open(gcp_creds_path) as f:
            creds_content = f.read()
        creds["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = creds_content
        if "VERTEXAI_PROJECT" not in creds:
            try:
                adc_data = json.loads(creds_content)
                if project_id := adc_data.get("quota_project_id"):
                    creds["VERTEXAI_PROJECT"] = project_id
                    creds["GOOGLE_CLOUD_PROJECT"] = project_id
            except json.JSONDecodeError:
                pass

    if coop_info:
        redis_url = coop_info.get("redis_url")
        if redis_url:
            if rewrite_localhost:
                redis_url = rewrite_comm_url_for_container(redis_url) or redis_url
            creds["REDIS_URL"] = redis_url
        if coop_info.get("git_url"):
            creds["GIT_URL"] = coop_info["git_url"]
        if coop_info.get("agent_id"):
            creds["AGENT_ID"] = coop_info["agent_id"]
        if coop_info.get("agents"):
            creds["AGENTS"] = ",".join(coop_info["agents"])
        team_env = coop_info.get("team_env") or {}
        for k, v in team_env.items():
            if v:
                creds[k] = v

    return creds


def _wait_for_agent_server(url: str, timeout: int = 120) -> None:
    """Block until the agent-server's /health endpoint returns 200.

    Shared by ModalSandboxContext and DockerSandboxContext.
    """
    import httpx

    start = time.time()
    last_error: Exception | None = None
    while time.time() - start < timeout:
        try:
            response = httpx.get(f"{url}/health", timeout=10)
            if response.status_code == 200:
                return
        except Exception as e:
            last_error = e
        time.sleep(2)

    raise TimeoutError(
        f"Agent-server did not become ready within {timeout}s. Last error: {last_error}"
    )


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
        agent_config: str | None = None,
        log_dir: str | None = None,
        team_role: str | None = None,
        team_id: str | None = None,
        task_list_url: str | None = None,
        team_features: "TeamHarnessConfig | None" = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run the OpenHands agent on a task.

        Team-mode kwargs (``team_role``, ``team_id``, ``task_list_url``)
        are accepted so the OpenHands adapter is API-compatible with the
        team runner.  In-loop integration with the shared task list
        lands in a follow-up PR.
        
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
        del kwargs  # unused
        # NOTE: the team prompt section is NOT appended to the user
        # ``task`` message for OpenHands — it's passed through
        # ``coop_info["team_section"]`` below so the SDK injects it
        # into the SYSTEM prompt instead, where it competes with
        # OpenHands' own collaboration block.  Putting it in the user
        # message gets out-prioritized (verified in oh_team_v2: agent
        # ignored it entirely).
        is_team = bool(team_role and team_id and task_list_url and agents and len(agents) > 1)
        # Convert to agent-server image if needed
        oh_image = self._get_oh_image(image)

        # Track state
        total_cost = 0.0
        input_tokens = 0
        output_tokens = 0
        cache_read_tokens = 0
        cache_write_tokens = 0
        messages = []
        sent_messages = []
        steps = 0
        patch = ""
        status = "Error"
        error = None
        
        config = config or {}
        backend = config.get("backend", "docker")

        # Determine if this is a coop run
        is_coop = (messaging_enabled or git_enabled) and agents and len(agents) > 1
        redis_url = comm_url
        # On Modal we self-manage the git server (ignoring git_server_url from
        # coop.py); on Docker we use the shared DockerGitServer that coop.py
        # creates and passes through git_server_url.
        git_url = None
        run_id = None
        owns_redis = False  # Track if we need to release Redis reference (Modal only)
        owns_git = False  # Track if we need to release Git server reference (Modal only)

        if is_coop:
            # Extract run_id from config or comm_url namespace
            if comm_url and "#run:" in comm_url:
                # Extract run_id from namespaced URL: redis://host:port#run:abc123
                run_id = comm_url.split("#run:")[1]
            else:
                run_id = config.get("run_id")

            # Generate run_id if not provided
            if not run_id:
                import uuid
                run_id = uuid.uuid4().hex[:8]

            if backend == "modal":
                # Modal can't reach host-side Redis/Git, so self-manage them.
                if messaging_enabled and _needs_modal_redis(comm_url):
                    redis_url = _get_or_create_redis(run_id, agents, self.timeout)
                    owns_redis = True
                if git_enabled:
                    git_url = _get_or_create_git_server(run_id, agents, self.timeout)
                    owns_git = True
            else:
                # Docker backend: reuse host Redis (ensure_redis) + shared
                # DockerGitServer (created by coop.py). REDIS_URL gets
                # rewritten to host.docker.internal inside the credential
                # collector when injected into the sandbox.
                if git_enabled:
                    git_url = git_server_url

        workspace = None

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
            # In team mode, fold team-mode env vars into coop_info so
            # _build_credentials_dict (which already understands
            # coop_info) propagates them to the sandbox.
            if is_team and coop_info is None:
                coop_info = {
                    "agent_id": agent_id,
                    "agents": agents or [],
                    "messaging_enabled": False,
                    "git_enabled": False,
                }
            if is_team and coop_info is not None:
                from cooperbench.team_harness import TeamHarnessConfig, TeamSession

                team_session = TeamSession(
                    run_id=team_id or "",
                    redis_url=task_list_url or "",  # host URL; env_for() rewrites
                    agents=list(agents or []),
                    team_volume="",  # openhands_sdk uses Modal layered image, not docker volume
                    config=team_features or TeamHarnessConfig(),
                )
                coop_info["team_env"] = team_session.env_for(agent_id)
                coop_info["team_features"] = team_session.config
                # Pass the team prompt section through coop_info so the
                # OpenHands SDK injects it into the SYSTEM prompt (next
                # to its own <collaboration> block).  Without this, the
                # SDK's coop block teaches the model to use send_message
                # only and our team_task_section appended to the user
                # message gets ignored (oh_team_v2 failure mode).
                coop_info["team_section"] = team_session.prompt_section(agent_id=agent_id)
            
            sandbox_cls = ModalSandboxContext if backend == "modal" else DockerSandboxContext
            with sandbox_cls(oh_image, self.timeout, coop_info=coop_info) as sandbox_url:

                # Import SDK components
                from openhands.sdk import LLM
                from openhands.sdk.conversation import RemoteConversation
                from openhands.sdk.workspace import RemoteWorkspace
                from openhands.tools.preset.default import get_default_agent

                # Create LLM instance (will be serialized and sent to server).
                # Azure OpenAI: when AZURE_OPENAI_* is set, point the LLM at the
                # Azure deployment via litellm's openai-compatible provider
                # (model openai/<deployment> + base_url + key).
                from cooperbench.agents._azure import azure_litellm_model, resolve_azure_config

                _azure = resolve_azure_config()
                if _azure:
                    llm = LLM(
                        model=azure_litellm_model(model_name),
                        api_key=_azure["api_key"],
                        base_url=_azure["endpoint"],
                    )
                else:
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

                # Send task and run the conversation. Append the patch.txt
                # submission instructions so the agent writes its diff to a
                # known file before finishing (matches mini_swe_agent's flow).
                # Message checking for coop mode happens inside the agent loop
                # (in LocalConversation._check_inbox_messages before each step)
                conversation.send_message(task + _submission_instructions(is_coop=is_coop))
                try:
                    conversation.run(blocking=True, timeout=float(self.timeout))
                    status = "Submitted"
                except Exception as e:
                    error_str = str(e)
                    if "MaxIterationsReached" in error_str:
                        logger.debug(f"Agent reached max iterations: {e}")
                        status = "Submitted"
                        error = None
                    else:
                        logger.exception(f"Error running agent: {e}")
                        error = error_str
                        status = "Error"

                # Read patch.txt that the agent wrote during submission.
                # Mirrors mini_swe_agent_v2's submission flow (see config/coop.yaml):
                # the agent is prompted to write its diff to patch.txt before
                # finishing, and we extract that file as-is.
                patch = ""
                try:
                    patch_result = workspace.execute_command(
                        "cat patch.txt 2>/dev/null",
                        cwd="/workspace/repo",
                        timeout=30.0,
                    )
                    if patch_result.exit_code == 0:
                        from cooperbench.agents._coop.runtime import normalize_patch
                        patch = normalize_patch(patch_result.stdout or "")
                except Exception as e:
                    logger.warning(f"Failed to read patch.txt: {e}")

                # Get cost and token usage from conversation stats
                try:
                    state = conversation.state
                    stats = state.stats
                    if stats:
                        combined_metrics = stats.get_combined_metrics()
                        total_cost = combined_metrics.accumulated_cost or 0.0

                        # Extract token counts
                        token_usage = combined_metrics.accumulated_token_usage
                        if token_usage:
                            input_tokens = token_usage.prompt_tokens or 0
                            output_tokens = token_usage.completion_tokens or 0
                            cache_read_tokens = getattr(token_usage, "cache_read_tokens", 0) or 0
                            cache_write_tokens = getattr(token_usage, "cache_write_tokens", 0) or 0

                        # Check cost limit
                        if self.cost_limit > 0 and total_cost >= self.cost_limit:
                            status = "CostLimitExceeded"
                        elif status != "Error":
                            status = "Submitted"
                except Exception as e:
                    logger.warning(f"Failed to get cost/tokens: {e}")
                    if status != "Error":
                        status = "Submitted"
        finally:
            # Release Redis reference (cleanup happens when all agents done)
            if owns_redis:
                _release_redis()
            # Release Git server reference
            if owns_git and run_id:
                _release_git_server(run_id)

        # Fallback cost calculation if agent didn't report cost
        if total_cost <= 0 and (input_tokens > 0 or output_tokens > 0):
            from cooperbench.agents.pricing import compute_fallback_cost

            fallback = compute_fallback_cost(
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            )
            if fallback is not None:
                total_cost = fallback

        return AgentResult(
            status=status,
            patch=patch,
            cost=total_cost,
            steps=steps,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
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
        return _collect_sandbox_credentials(self.coop_info, rewrite_localhost=False)

    def __enter__(self) -> str:
        """Start sandbox, run agent-server, and return the tunnel URL."""

        # Preserve image ENTRYPOINT.
        # The `-oh` images set ENTRYPOINT to launch `openhands.agent_server`.
        image = modal.Image.from_registry(self.image_name)

        # Layer the team CLI onto the image when team mode is active so
        # the agent-server's bash tool can call coop-task-* without us
        # needing to rebuild the upstream `-oh` image.  Modal caches the
        # resulting layered image; first team run pays a ~10s build,
        # subsequent runs are instant.  Detected via the team_env dict
        # that the adapter folds into coop_info.
        team_env = (self.coop_info or {}).get("team_env") if self.coop_info else None
        team_features = (self.coop_info or {}).get("team_features") if self.coop_info else None
        # Layering only matters when the in-container coop-task-* CLI
        # would actually be invoked — i.e. task_list or protocol is on.
        # If both are off, skip the (slow first-time) image build.
        install_team_cli = bool(team_env) and (
            team_features is None
            or getattr(team_features, "task_list", True)
            or getattr(team_features, "protocol", True)
        )
        if install_team_cli:
            from pathlib import Path as _Path

            from cooperbench.team_harness import COOP_TASK_SCRIPT_PATH

            coop_task_path = COOP_TASK_SCRIPT_PATH
            # The CoopTaskTrackerTool definition needs to be injected
            # into the agent-server's openhands install so the agent
            # can resolve ``Tool(name="CoopTaskTrackerTool")``.  We
            # also drop a .pth file that auto-imports the module at
            # site-init so register_tool fires before any tool lookup.
            _oh_tools_dir = (
                _Path(__file__).resolve().parent / "openhands-tools" / "openhands" / "tools"
            )
            coop_tracker_path = _oh_tools_dir / "task_tracker" / "coop_definition.py"
            # Replacement __init__.py imports coop_definition so the
            # registration overrides the upstream local TaskTracker.
            init_override_path = _oh_tools_dir / "task_tracker" / "_team_init_override.py"
            image = (
                image.add_local_file(
                    str(coop_task_path),
                    "/usr/local/bin/cb-coop-task.py",
                    copy=True,
                )
                .add_local_file(
                    str(coop_tracker_path),
                    "/tmp/cb-coop-tracker.py",
                    copy=True,
                )
                .add_local_file(
                    str(init_override_path),
                    "/tmp/cb-task-tracker-init.py",
                    copy=True,
                )
                .pip_install("redis")
                # Inject CoopTaskTracker into the openhands install
                # AND append a side-effect import to the package's
                # ``__init__.py`` so it always runs when openhands
                # tools are imported.  Note: this currently has no
                # functional effect because the Modal sandbox can't
                # reach the host Redis — see the docstring of
                # ``coop_definition.py`` for details — but landing the
                # injection plumbing here keeps the code path ready
                # for the Redis-reachability follow-up.
                .run_commands(
                    # 1. Drop the tool file into the openhands install.
                    # 2. Replace the package __init__.py with the
                    #    pre-rendered override that imports
                    #    coop_definition (overriding the local
                    #    TaskTracker registration).  Using a pre-rendered
                    #    file via add_local_file (not a shell heredoc)
                    #    avoids quoting fragility.
                    # 3. Delete any cached .pyc files so Python
                    #    recompiles the new __init__ on next import.
                    'OH_DIR="$(python3 -c \'import openhands.tools.task_tracker as t, os; print(os.path.dirname(t.__file__))\')"; '
                    'cp /tmp/cb-coop-tracker.py "$OH_DIR/coop_definition.py" && '
                    'cp /tmp/cb-task-tracker-init.py "$OH_DIR/__init__.py" && '
                    'find "$OH_DIR" -name "*.pyc" -delete; '
                    'find "$OH_DIR" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true; '
                    "echo INIT_PATCHED"
                )
                .run_commands(
                    # Create one wrapper per coop-task-* subcommand.
                    # Same Modal-Redis caveat as above; binaries are
                    # present and discoverable but won't function until
                    # Redis is reachable.
                    'for sub in create claim update list request respond pending; do '
                    'printf "#!/bin/bash\\nexec python3 /usr/local/bin/cb-coop-task.py %s \\"\\$@\\"\\n" "$sub" '
                    '> "/usr/local/bin/coop-task-$sub" && chmod +x "/usr/local/bin/coop-task-$sub"; '
                    'done'
                )
            )
        
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
            # Start outside /workspace/repo to avoid import shadowing
            # (e.g., openai_tiktoken_task shadows litellm's `tiktoken` import).
            workdir="/",
            # Expose port 8000 for the agent-server
            encrypted_ports=[8000],
        )
        
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
        _wait_for_agent_server(url, timeout=timeout)


class DockerSandboxContext:
    """Context manager for a local Docker sandbox running the agent-server.

    Mirrors ``ModalSandboxContext`` but uses local docker:

    * ``docker run -d --rm --platform linux/amd64 -p 0:8000`` on the
      ``-oh`` image, with a random host port for concurrency safety.
    * Joins the shared ``cooperbench`` bridge network when ``git_enabled``
      so the agent-server can resolve ``cooperbench-git`` by name.
    * Adds ``--add-host=host.docker.internal:host-gateway`` whenever a
      ``REDIS_URL`` is configured, so the agent-server can reach the
      host-side ``cooperbench-redis`` container.
    * Injects credentials via a temp ``--env-file`` (handles multi-line
      ``GOOGLE_APPLICATION_CREDENTIALS_JSON``).
    * Lays in the team-mode coop-task CLI via ``docker cp`` when needed,
      matching the Modal path's image-layering behavior.
    """

    AGENT_SERVER_PORT = 8000

    def __init__(self, image_name: str, timeout: int, coop_info: dict | None = None):
        del timeout  # docker run has no equivalent timeout; lifecycle bounded by context manager.
        self.image_name = image_name
        self.coop_info = coop_info
        self._container_name: str | None = None
        self._env_file_path: str | None = None

    def __enter__(self) -> str:
        import secrets as _secrets
        import shutil
        import subprocess
        import tempfile

        if shutil.which("docker") is None:
            raise RuntimeError("docker CLI not found on PATH — install Docker to use the docker backend")

        self._container_name = f"cooperbench-oh-{_secrets.token_hex(4)}"

        creds = _collect_sandbox_credentials(self.coop_info, rewrite_localhost=True)

        # Write credentials to a temp env-file (handles multi-line ADC JSON).
        env_file = tempfile.NamedTemporaryFile(
            mode="w",
            prefix="cooperbench-oh-env-",
            suffix=".env",
            delete=False,
        )
        try:
            for key, value in creds.items():
                # docker --env-file is line-oriented; multi-line values
                # (ADC JSON) need to be base64-encoded or passed via -e
                # in their entirety. We pass single-line values through
                # the env-file and multi-line values via -e with the
                # raw value, which docker forwards correctly.
                if "\n" not in value:
                    env_file.write(f"{key}={value}\n")
            env_file.close()
        except Exception:
            env_file.close()
            os.unlink(env_file.name)
            raise
        self._env_file_path = env_file.name

        multiline_env = {k: v for k, v in creds.items() if "\n" in v}

        cmd: list[str] = [
            "docker",
            "run",
            "-d",
            "--rm",
            "--platform",
            "linux/amd64",
            "--name",
            self._container_name,
            "-p",
            f"0:{self.AGENT_SERVER_PORT}",
            "--env-file",
            self._env_file_path,
            "--workdir",
            "/",
        ]

        if self.coop_info and self.coop_info.get("git_enabled"):
            cmd += ["--network", "cooperbench"]

        if creds.get("REDIS_URL"):
            cmd += ["--add-host", "host.docker.internal:host-gateway"]

        for key, value in multiline_env.items():
            cmd += ["-e", f"{key}={value}"]

        cmd.append(self.image_name)

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            self._cleanup_env_file()
            raise RuntimeError(
                f"docker run failed for {self.image_name}: {e.stderr or e.stdout}"
            ) from e

        try:
            self._install_team_cli_if_needed()
        except Exception:
            self.__exit__(None, None, None)
            raise

        try:
            host_port = self._discover_host_port()
        except Exception:
            self.__exit__(None, None, None)
            raise

        url = f"http://localhost:{host_port}"
        try:
            _wait_for_agent_server(url)
        except Exception:
            self.__exit__(None, None, None)
            raise

        return url

    def _discover_host_port(self) -> int:
        import subprocess

        result = subprocess.run(
            ["docker", "port", self._container_name or "", f"{self.AGENT_SERVER_PORT}/tcp"],
            check=True,
            capture_output=True,
            text=True,
        )
        # Output looks like ``0.0.0.0:32773\n[::]:32773``; first line is enough.
        first = result.stdout.strip().splitlines()[0]
        return int(first.rsplit(":", 1)[1])

    def _install_team_cli_if_needed(self) -> None:
        """Drop coop-task CLI + tracker override into the running container.

        Equivalent to the Modal path's image-layering block but done via
        ``docker cp`` against the already-running container.  Triggered
        on the same condition as the Modal path: ``team_env`` is set and
        either ``task_list`` or ``protocol`` is enabled.
        """
        import subprocess
        from pathlib import Path as _Path

        team_env = (self.coop_info or {}).get("team_env") if self.coop_info else None
        team_features = (self.coop_info or {}).get("team_features") if self.coop_info else None
        install = bool(team_env) and (
            team_features is None
            or getattr(team_features, "task_list", True)
            or getattr(team_features, "protocol", True)
        )
        if not install:
            return

        from cooperbench.team_harness import COOP_TASK_SCRIPT_PATH

        oh_tools_dir = (
            _Path(__file__).resolve().parent / "openhands-tools" / "openhands" / "tools"
        )
        coop_tracker_path = oh_tools_dir / "task_tracker" / "coop_definition.py"
        init_override_path = oh_tools_dir / "task_tracker" / "_team_init_override.py"

        name = self._container_name or ""

        # Copy the coop-task script + task-tracker overrides into the container.
        for src, dst in [
            (COOP_TASK_SCRIPT_PATH, "/usr/local/bin/cb-coop-task.py"),
            (coop_tracker_path, "/tmp/cb-coop-tracker.py"),
            (init_override_path, "/tmp/cb-task-tracker-init.py"),
        ]:
            subprocess.run(
                ["docker", "cp", str(src), f"{name}:{dst}"],
                check=True,
                capture_output=True,
                text=True,
            )

        # Patch the openhands install: drop coop_definition.py + override
        # __init__, install redis, and create coop-task-* wrapper scripts.
        patch_script = (
            "set -e; "
            "pip install --quiet redis >/dev/null 2>&1 || pip install redis; "
            'OH_DIR="$(python3 -c \'import openhands.tools.task_tracker as t, os; '
            "print(os.path.dirname(t.__file__))')\"; "
            'cp /tmp/cb-coop-tracker.py "$OH_DIR/coop_definition.py"; '
            'cp /tmp/cb-task-tracker-init.py "$OH_DIR/__init__.py"; '
            'find "$OH_DIR" -name "*.pyc" -delete; '
            'find "$OH_DIR" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true; '
            "for sub in create claim update list request respond pending; do "
            'printf "#!/bin/bash\\nexec python3 /usr/local/bin/cb-coop-task.py %s \\"\\$@\\"\\n" "$sub" '
            '> "/usr/local/bin/coop-task-$sub" && chmod +x "/usr/local/bin/coop-task-$sub"; '
            "done"
        )
        subprocess.run(
            ["docker", "exec", name, "bash", "-c", patch_script],
            check=True,
            capture_output=True,
            text=True,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        import subprocess

        if self._container_name:
            # Skip the graceful `docker stop` step and go straight to
            # `docker rm -f` (SIGKILL + remove in one call). The graceful
            # stop was timing out under concurrent load (Docker Desktop on
            # Apple Silicon emulating dozens of amd64 containers via
            # Rosetta) and the agent-server doesn't need a clean shutdown —
            # all artifacts we care about (patch.txt, trajectory) have
            # already been extracted via the HTTP API before this point.
            try:
                subprocess.run(
                    ["docker", "rm", "-f", self._container_name],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except Exception as e:
                logger.warning(f"docker rm -f {self._container_name} failed: {e}")
            self._container_name = None
        self._cleanup_env_file()

    def _cleanup_env_file(self) -> None:
        if self._env_file_path and os.path.exists(self._env_file_path):
            try:
                os.unlink(self._env_file_path)
            except OSError:
                pass
        self._env_file_path = None
