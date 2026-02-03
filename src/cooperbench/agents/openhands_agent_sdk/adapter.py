"""OpenHands SDK adapter for CooperBench.

This adapter runs the OpenHands agent-server in Modal and connects to it
using the SDK's RemoteWorkspace.
"""

import os
import time
import logging
from typing import Any

import modal
from cooperbench.agents import AgentResult
from cooperbench.agents.registry import register

logger = logging.getLogger(__name__)


# Modal app for running agent-server
modal_app = modal.App("cooperbench-openhands")


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

    def __init__(self, max_iterations: int = 50, timeout: int = 3600):
        self.max_iterations = max_iterations
        self.timeout = timeout

    def _get_oh_image(self, image: str) -> str:
        """Convert base image to agent-server image (add -oh suffix if needed)."""
        if image.endswith("-oh"):
            return image
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
        # Collaboration options (not yet supported)
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
            agents: List of all agent IDs (for collaboration, not yet supported)
            comm_url: Redis URL for inter-agent messaging (not yet supported)
            git_server_url: Git server URL for code sharing (not yet supported)
            git_enabled: Whether git collaboration is enabled
            messaging_enabled: Whether messaging is enabled
            config: Agent-specific configuration
            
        Returns:
            AgentResult with status, patch, cost, steps, messages
        """
        # Convert to agent-server image if needed
        oh_image = self._get_oh_image(image)
        logger.info(f"Starting OpenHands agent with image: {oh_image}")

        # Track state
        total_cost = 0.0
        messages = []
        steps = 0
        patch = ""
        status = "Error"
        error = None

        try:
            # Start agent-server in Modal
            with ModalSandboxContext(oh_image, self.timeout) as sandbox_url:
                logger.info(f"Agent-server running at: {sandbox_url}")

                # Import SDK components
                from openhands.sdk import LLM, Agent
                from openhands.sdk.conversation import RemoteConversation
                from openhands.sdk.workspace import RemoteWorkspace
                from openhands.tools.preset.default import get_default_agent

                # Create LLM instance (will be serialized and sent to server)
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
                llm = LLM(model=model_name, api_key=api_key)

                # Create agent with default tools (terminal, file_editor, task_tracker)
                # Browser tools disabled since we're running headless
                agent = get_default_agent(llm=llm, cli_mode=True)

                # Connect to remote workspace (agent-server in Modal)
                workspace = RemoteWorkspace(
                    host=sandbox_url,
                    working_dir="/workspace/repo",
                )

                # Callback to collect events
                def event_callback(event):
                    nonlocal steps
                    steps += 1
                    messages.append({
                        "step": steps,
                        "event_type": type(event).__name__,
                        "event": str(event)[:500],  # Truncate long events
                    })

                # Create remote conversation - agent loop runs on server
                conversation = RemoteConversation(
                    agent=agent,
                    workspace=workspace,
                    max_iteration_per_run=self.max_iterations,
                    callbacks=[event_callback],
                )

                # Send task and run the conversation
                logger.info(f"Sending task to agent: {task[:100]}...")
                conversation.send_message(task)
                conversation.run(blocking=True, timeout=float(self.timeout))

                # Get the patch from remote workspace
                # Note: workspace.git_diff() has a bug (Path vs str URL), so use execute_command
                logger.info("Retrieving git diff...")
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
                        logger.info(f"Total cost: ${total_cost:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to get cost: {e}")
                    pass  # Cost tracking optional
                
                status = "Submitted"
                logger.info(f"Agent completed. Steps: {steps}, Patch length: {len(patch)}")

        except Exception as e:
            logger.exception(f"Error running agent: {e}")
            error = str(e)
            status = "Error"

        return AgentResult(
            status=status,
            patch=patch,
            cost=total_cost,
            steps=steps,
            messages=messages,
            error=error,
        )


class ModalSandboxContext:
    """Context manager for Modal sandbox with agent-server.
    
    This starts an agent-server in a Modal sandbox and provides an HTTP URL to connect to it.
    The agent-server runs as the container's entrypoint and exposes port 8000.
    
    Credentials are passed to the sandbox via modal.Secret:
    - GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY from environment
    - Google Cloud credentials from GOOGLE_APPLICATION_CREDENTIALS file
    """

    def __init__(self, image_name: str, timeout: int):
        self.image_name = image_name
        self.timeout = timeout
        self._sandbox: modal.Sandbox | None = None
        self._server_proc = None

    def _collect_credentials(self) -> dict[str, str]:
        """Collect API keys and credentials from environment."""
        creds = {}
        
        # Collect API keys
        for key in [
            "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY", 
            "OPENAI_API_KEY",
            "VERTEX_PROJECT",
            "VERTEX_LOCATION",
            "GOOGLE_CLOUD_PROJECT",
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
                logger.info(f"Using gcloud ADC from: {gcp_creds_path}")
        
        if gcp_creds_path and os.path.exists(gcp_creds_path):
            with open(gcp_creds_path) as f:
                creds_content = f.read()
                creds["GOOGLE_APPLICATION_CREDENTIALS_JSON"] = creds_content
                logger.info("Loaded Google Cloud credentials")
                
                # Extract project from ADC if not already set
                if "VERTEX_PROJECT" not in creds:
                    import json
                    try:
                        adc_data = json.loads(creds_content)
                        if project_id := adc_data.get("quota_project_id"):
                            creds["VERTEX_PROJECT"] = project_id
                            creds["GOOGLE_CLOUD_PROJECT"] = project_id
                            logger.info(f"Using project from ADC: {project_id}")
                    except json.JSONDecodeError:
                        pass
        
        return creds

    def __enter__(self) -> str:
        """Start sandbox, run agent-server, and return the tunnel URL."""
        logger.info(f"Creating Modal sandbox with image: {self.image_name}")
        
        # Build image and clear entrypoint (Modal will add its own default command)
        image = modal.Image.from_registry(self.image_name).entrypoint([])
        
        # Get or create app
        app = modal.App.lookup("cooperbench-openhands", create_if_missing=True)
        
        # Collect credentials and create Modal secret
        creds = self._collect_credentials()
        secrets = [modal.Secret.from_dict(creds)] if creds else []
        logger.info(f"Passing {len(creds)} credentials to sandbox")
        
        # Create sandbox with tunnel for port 8000
        self._sandbox = modal.Sandbox.create(
            image=image,
            timeout=self.timeout,
            app=app,
            secrets=secrets,
            # Expose port 8000 for the agent-server
            encrypted_ports=[8000],
        )
        
        logger.info(f"Sandbox created: {self._sandbox.object_id}")
        
        # If Google Cloud credentials JSON was passed, write it to a file
        # and set the GOOGLE_APPLICATION_CREDENTIALS env var
        startup_script = """
#!/bin/bash
set -e

# Write Google Cloud credentials if provided
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS_JSON" ]; then
    echo "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > /tmp/gcp-credentials.json
    export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-credentials.json
fi

# Start the agent-server
exec /opt/agent-server-venv/bin/python -m openhands.agent_server --host 0.0.0.0 --port 8000
"""
        
        # Start the agent-server manually (since we cleared the entrypoint)
        logger.info("Starting agent-server in sandbox...")
        self._server_proc = self._sandbox.exec(
            "bash", "-c", startup_script,
        )
        
        # Give the server a moment to start
        time.sleep(3)
        
        # Get tunnel URL
        tunnel_info = self._sandbox.tunnels()[8000]
        tunnel_url = tunnel_info.url
        logger.info(f"Agent-server tunnel URL: {tunnel_url}")
        
        # Wait for server to be ready
        self._wait_for_server(tunnel_url)
        
        return tunnel_url

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup sandbox."""
        if self._sandbox:
            try:
                logger.info("Terminating sandbox...")
                self._sandbox.terminate()
            except Exception as e:
                logger.warning(f"Failed to terminate sandbox: {e}")

    def _wait_for_server(self, url: str, timeout: int = 120):
        """Wait for the agent-server to be ready."""
        import httpx

        logger.info(f"Waiting for agent-server at {url}...")
        start = time.time()
        last_error = None
        
        while time.time() - start < timeout:
            try:
                response = httpx.get(f"{url}/health", timeout=10)
                if response.status_code == 200:
                    logger.info("Agent-server is ready!")
                    return
            except Exception as e:
                last_error = e
            time.sleep(2)

        raise TimeoutError(
            f"Agent-server did not become ready within {timeout}s. "
            f"Last error: {last_error}"
        )
