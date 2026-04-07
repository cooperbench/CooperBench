"""Configuration management for CooperBench.

Handles setup and validation of execution backends (GCP, Modal, Docker).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from platformdirs import user_config_dir
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

console = Console()


class ConfigManager:
    """Manages CooperBench configuration."""

    def __init__(self):
        self.config_dir = Path(user_config_dir("cooperbench", appauthor=False))
        self.config_file = self.config_dir / "config.json"
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                return json.loads(self.config_file.read_text())
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load config: {e}[/yellow]")
        return {}

    def _save_config(self) -> None:
        """Save configuration to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(json.dumps(self.config, indent=2))
        console.print(f"[green]Configuration saved to {self.config_file}[/green]")

    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value) -> None:
        """Set configuration value."""
        self.config[key] = value
        self._save_config()


class GCPConfigurator:
    """Interactive GCP configuration wizard."""

    def __init__(self):
        self.config_manager = ConfigManager()

    def run(self, skip_tests: bool = False) -> bool:
        """Run the GCP configuration wizard.

        Args:
            skip_tests: Skip validation tests (faster, but no verification)

        Returns:
            True if configuration successful, False otherwise
        """
        console.print(
            Panel.fit(
                "[bold cyan]CooperBench GCP Configuration Wizard[/bold cyan]\n\n"
                "This wizard will help you set up GCP as your execution backend.\n"
                "You'll need a Google Cloud Platform account to proceed.",
                border_style="cyan",
            )
        )

        # Step 1: Check prerequisites
        console.print("\n[bold]Step 1: Checking prerequisites[/bold]")
        if not self._check_gcloud():
            return False

        if not self._check_dependencies():
            return False

        # Step 2: Authenticate
        console.print("\n[bold]Step 2: Authentication[/bold]")
        if not self._check_authentication():
            if not self._guide_authentication():
                return False

        # Step 3: Project setup
        console.print("\n[bold]Step 3: Project configuration[/bold]")
        project_id = self._setup_project()
        if not project_id:
            return False

        # Step 4: Region/zone
        console.print("\n[bold]Step 4: Region and zone[/bold]")
        region, zone = self._setup_region()

        # Step 5: Test setup (optional)
        if not skip_tests:
            console.print("\n[bold]Step 5: Validation[/bold]")
            if not self._validate_setup(project_id, region, zone):
                console.print("[yellow]Validation failed, but configuration was saved.[/yellow]")
                console.print("[yellow]You can still try using GCP backend with --backend gcp[/yellow]")

        # Save configuration
        self.config_manager.set("gcp_project_id", project_id)
        self.config_manager.set("gcp_region", region)
        self.config_manager.set("gcp_zone", zone)
        self.config_manager.set("gcp_bucket", f"cooperbench-eval-{project_id}")

        # Summary
        console.print("\n" + "=" * 60)
        console.print(
            Panel.fit(
                "[bold green]GCP Configuration Complete![/bold green]\n\n"
                f"Project ID: {project_id}\n"
                f"Region: {region}\n"
                f"Zone: {zone}\n"
                f"Bucket: cooperbench-eval-{project_id}\n\n"
                "You can now use GCP backend with:\n"
                "[cyan]uv run cooperbench run --backend gcp[/cyan]\n"
                "[cyan]uv run cooperbench eval --backend gcp[/cyan]",
                border_style="green",
            )
        )

        return True

    def _check_gcloud(self) -> bool:
        """Check if gcloud CLI is installed."""
        console.print("Checking for gcloud CLI...", end=" ")
        try:
            result = subprocess.run(
                ["gcloud", "version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                console.print("[green]✓ Found[/green]")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        console.print("[red]✗ Not found[/red]")
        console.print(
            "\n[yellow]gcloud CLI is required for GCP backend.[/yellow]\n"
            "Install it from: https://cloud.google.com/sdk/docs/install\n\n"
            "Quick install:\n"
            "  macOS: brew install google-cloud-sdk\n"
            "  Linux: curl https://sdk.cloud.google.com | bash\n"
        )
        return False

    def _check_dependencies(self) -> bool:
        """Check if GCP Python dependencies are installed."""
        console.print("Checking GCP Python dependencies...", end=" ")

        missing = []
        for package in ["google.cloud.batch", "google.cloud.compute", "google.cloud.storage"]:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)

        if not missing:
            console.print("[green]✓ Installed[/green]")
            return True

        console.print("[red]✗ Missing packages[/red]")
        console.print(
            f"\n[yellow]Missing: {', '.join(missing)}[/yellow]\n\n"
            "Install GCP dependencies with:\n"
            "[cyan]uv add --optional gcp 'cooperbench[gcp]'[/cyan]\n"
            "or with pip:\n"
            "[cyan]pip install 'cooperbench[gcp]'[/cyan]\n"
            "or install packages directly:\n"
            "[cyan]pip install google-cloud-batch google-cloud-compute google-cloud-storage[/cyan]\n"
        )
        return False

    def _check_authentication(self) -> bool:
        """Check if user is authenticated with gcloud and has ADC set up."""
        console.print("Checking authentication...", end=" ")

        # Check gcloud auth
        gcloud_authed = False
        try:
            result = subprocess.run(
                ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                gcloud_authed = True
        except subprocess.TimeoutExpired:
            pass

        # Check Application Default Credentials (ADC)
        import os
        from pathlib import Path

        adc_authed = False
        adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        if adc_path.exists():
            adc_authed = True
        # Also check GOOGLE_APPLICATION_CREDENTIALS env var
        elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            adc_authed = True

        if gcloud_authed and adc_authed:
            console.print("[green]✓ Authenticated (gcloud + ADC)[/green]")
            return True
        elif gcloud_authed and not adc_authed:
            console.print("[yellow]⚠ gcloud authenticated but ADC not set up[/yellow]")
            console.print("[dim]  Application Default Credentials are required for GCP Python SDK[/dim]")
            return False
        else:
            console.print("[yellow]✗ Not authenticated[/yellow]")
            return False

    def _guide_authentication(self) -> bool:
        """Guide user through authentication."""
        if not Confirm.ask("\nWould you like to authenticate now?", default=True):
            console.print("[yellow]Skipping authentication. Please run:[/yellow]")
            console.print("[yellow]  gcloud auth login[/yellow]")
            console.print("[yellow]  gcloud auth application-default login[/yellow]")
            return False

        console.print("\n[bold]Setting up authentication (2 steps)[/bold]")

        # Step 1: gcloud auth login
        console.print("\n[cyan]Step 1/2:[/cyan] Authenticating gcloud CLI...")
        try:
            result = subprocess.run(
                ["gcloud", "auth", "login"],
                timeout=300,
            )
            if result.returncode != 0:
                console.print("[red]✗ gcloud auth login failed[/red]")
                return False
            console.print("[green]✓ gcloud CLI authenticated[/green]")
        except Exception as e:
            console.print(f"[red]✗ Authentication failed: {e}[/red]")
            return False

        # Step 2: Application Default Credentials
        console.print("\n[cyan]Step 2/2:[/cyan] Setting up Application Default Credentials...")
        console.print("[dim]This is required for GCP Python libraries to work.[/dim]")
        try:
            result = subprocess.run(
                ["gcloud", "auth", "application-default", "login"],
                timeout=300,
            )
            if result.returncode != 0:
                console.print("[red]✗ ADC setup failed[/red]")
                return False
            console.print("[green]✓ Application Default Credentials configured[/green]")
        except Exception as e:
            console.print(f"[red]✗ ADC setup failed: {e}[/red]")
            return False

        return True

    def _setup_project(self) -> str | None:
        """Set up GCP project."""
        # Try to get current project
        current_project = None
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                current_project = result.stdout.strip()
        except subprocess.TimeoutExpired:
            pass

        # Prompt for project ID
        if current_project:
            console.print(f"Current gcloud project: [cyan]{current_project}[/cyan]")
            if Confirm.ask("Use this project?", default=True):
                return current_project

        console.print(
            "\nEnter your GCP project ID.\n"
            "Find it at: https://console.cloud.google.com/home/dashboard\n"
            "Or create a new project at: https://console.cloud.google.com/projectcreate\n"
        )

        project_id = Prompt.ask("GCP Project ID", default=current_project or "")
        if not project_id:
            console.print("[red]Project ID is required[/red]")
            return None

        # Set as default project
        if Confirm.ask(f"\nSet '{project_id}' as default gcloud project?", default=True):
            try:
                subprocess.run(
                    ["gcloud", "config", "set", "project", project_id],
                    check=True,
                    timeout=10,
                )
                console.print(f"[green]✓ Set default project to {project_id}[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to set default project: {e}[/yellow]")

        return project_id

    def _setup_region(self) -> tuple[str, str]:
        """Set up region and zone."""
        # Try to get current region/zone
        current_region = None
        current_zone = None
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "compute/region"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                current_region = result.stdout.strip()
        except subprocess.TimeoutExpired:
            pass

        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "compute/zone"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                current_zone = result.stdout.strip()
        except subprocess.TimeoutExpired:
            pass

        # Defaults
        default_region = current_region or "us-central1"
        default_zone = current_zone or "us-central1-a"

        console.print(
            "\nRecommended regions for low latency:\n"
            "  us-central1 (Iowa)\n"
            "  us-east1 (South Carolina)\n"
            "  europe-west1 (Belgium)\n"
            "  asia-east1 (Taiwan)\n"
        )

        region = Prompt.ask("GCP Region", default=default_region)
        zone = Prompt.ask("GCP Zone", default=default_zone)

        return region, zone

    def _validate_setup(self, project_id: str, region: str, zone: str) -> bool:
        """Validate GCP setup by testing API access."""
        console.print("Testing GCP access...", end=" ")

        # Set environment variable for GCP client libraries
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

        try:
            # Test Compute Engine API
            from google.cloud import compute_v1

            client = compute_v1.ZonesClient()
            request = compute_v1.GetZoneRequest(
                project=project_id,
                zone=zone,
            )
            client.get(request=request, timeout=30)

            console.print("[green]✓ Compute Engine API accessible[/green]")
            return True

        except Exception as e:
            console.print("[red]✗ Failed[/red]")
            console.print(f"\n[yellow]Error: {e}[/yellow]")
            console.print(
                "\nMake sure you have:\n"
                "1. Created or selected a valid GCP project\n"
                "2. Enabled Compute Engine API: https://console.cloud.google.com/apis/library/compute.googleapis.com\n"
                "3. Enabled Cloud Batch API: https://console.cloud.google.com/apis/library/batch.googleapis.com\n"
                "4. Set up billing: https://console.cloud.google.com/billing\n"
            )
            return False


def config_gcp_command(skip_tests: bool = False) -> int:
    """Run GCP configuration wizard.

    Args:
        skip_tests: Skip validation tests

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    configurator = GCPConfigurator()
    success = configurator.run(skip_tests=skip_tests)
    return 0 if success else 1
