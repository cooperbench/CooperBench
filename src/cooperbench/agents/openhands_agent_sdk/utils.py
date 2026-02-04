"""Utility functions for OpenHands SDK adapter."""

import logging

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_result,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


def wait_for_git_server(
    workspace,
    git_url: str,
    max_attempts: int = 5,
    initial_wait: float = 3.0,
    max_wait: float = 30.0,
) -> None:
    """Wait for git server to be reachable.
    
    Uses git ls-remote to check connectivity with exponential backoff.
    
    Args:
        workspace: RemoteWorkspace instance
        git_url: Git server URL to check
        max_attempts: Maximum number of retry attempts
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time between retries
        
    Raises:
        RuntimeError: If git server is not reachable after all attempts
    """
    
    @retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=initial_wait, max=max_wait),
        retry=retry_if_result(lambda r: r is False),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    )
    def _check_connectivity() -> bool:
        result = workspace.execute_command(
            f"git ls-remote {git_url} 2>&1 || true",
            cwd="/workspace/repo",
            timeout=15.0,
        )
        # ls-remote returns 0 on success, or outputs error message
        if result.exit_code == 0 and "fatal:" not in result.stdout:
            logger.debug(f"Git server reachable at {git_url}")
            return True
        logger.debug(f"Git server not ready: {result.stdout[:100]}")
        return False
    
    try:
        _check_connectivity()
    except Exception as e:
        raise RuntimeError(f"Git server not reachable at {git_url} after {max_attempts} attempts") from e


def git_push_with_retry(
    workspace,
    remote: str,
    branch: str,
    max_attempts: int = 3,
    initial_wait: float = 5.0,
    max_wait: float = 30.0,
    force: bool = False,
) -> bool:
    """Push to git remote with retry and exponential backoff.
    
    Args:
        workspace: RemoteWorkspace instance  
        remote: Remote name (e.g., "team")
        branch: Branch name to push
        max_attempts: Maximum number of retry attempts
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time between retries
        force: Whether to force push
        
    Returns:
        True if push succeeded, False otherwise
    """
    force_flag = "--force" if force else ""
    
    @retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=initial_wait, max=max_wait),
        retry=retry_if_result(lambda r: r is False),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=False,
    )
    def _push() -> bool:
        result = workspace.execute_command(
            f"git push -u {remote} {branch} {force_flag}",
            cwd="/workspace/repo",
            timeout=60.0,
        )
        if result.exit_code == 0:
            return True
        logger.warning(f"Git push failed: {result.stderr[:200]}")
        return False
    
    return _push()
