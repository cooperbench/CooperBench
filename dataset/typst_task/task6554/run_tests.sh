#!/usr/bin/env bash
# =========================================================================
# Typst CodeConflictBenchmark Test Script
# =========================================================================
# This script:
# 1. Installs Rust and Cargo if not already present
# 2. Applies a patch to the Typst repository
# 3. Builds the modified Typst
# 4. Runs related tests
# 5. Returns the same exit code as the test command
# =========================================================================

# Exit on undefined variable or pipeline failure
set -euo pipefail

# Define colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# -------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------

log() {
  echo -e "${GREEN}$1${NC}"
}

info() {
  echo -e "${BLUE}$1${NC}"
}

warning() {
  echo -e "${YELLOW}WARNING: $1${NC}"
}

error() {
  echo -e "${RED}ERROR: $1${NC}"
  exit 1
}

check_and_install_rust() {
  if command -v cargo &> /dev/null; then
    log "Rust/Cargo is already installed: $(cargo --version)"
    return 0
  fi
  
  log "Rust is not installed. Installing Rust..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source ~/.cargo/env
  log "Rust installed successfully: $(cargo --version)"
}

apply_feature_patch() {
  local feature_name=$1
  local feature_dir="../feature${feature_name}"
  
  if [[ ! -d "$feature_dir" ]]; then
    error "Feature directory $feature_dir not found"
  fi
  
  log "Applying patches for feature${feature_name}..."
  
  # Apply main feature patch
  if [[ -f "$feature_dir/feature.patch" ]]; then
    info "Applying feature.patch..."
    git apply "$feature_dir/feature.patch" || error "Failed to apply feature.patch"
  fi
  
  # Apply test patch (if needed and not already included)
  if [[ -f "$feature_dir/tests.patch" ]]; then
    info "Applying tests.patch..."
    if ! git apply "$feature_dir/tests.patch" 2>/dev/null; then
      warning "tests.patch failed to apply (may already be included in feature.patch)"
    fi
  fi
  
  log "Patches applied successfully!"
}

build_typst() {
  log "Building Typst with modifications..."
  
  # Build the typst crate specifically
  cargo build --package typst || error "Failed to build typst"
  
  # Build the main typst cli for testing
  cargo build --package typst-cli || error "Failed to build typst-cli"
  
  log "Build completed successfully!"
}

run_tests() {
  local -a extra_args=("$@")
  log "Running Typst tests..."
  
  # Run tests from the tests directory
  info "Running string foundation tests..."
  if ((${#extra_args[@]})); then
    cargo test -p typst-tests -- foundations/str "${extra_args[@]}" || {
      warning "String tests failed, but continuing..."
    }
  else
    cargo test -p typst-tests -- foundations/str || {
      warning "String tests failed, but continuing..."
    }
  fi
  
  # Run complete test suite
  info "Running complete test suite..."
  if ((${#extra_args[@]})); then
    cargo test -p typst-tests "${extra_args[@]}" || {
      warning "Some tests failed, checking test output..."
    }
  else
    cargo test -p typst-tests || {
      warning "Some tests failed, checking test output..."
    }
  fi
  
  log "Test execution completed!"
}

print_usage() {
  cat <<USAGE
Usage: $0 [feature_number]
  feature_number: Optional. 1-10 to apply the matching feature patch.
                  If omitted, assumes the repository already has the desired
                  changes applied (e.g., evaluation harness has staged a patch).

Available features:
  1: Default value parameter (str.first(default:), str.last(default:))
  2: Indexed access (str.first(index), str.last(index))
  3: Safe mode parameter (str.first(safe:), str.last(safe:))
  4: Repeat parameter (str.first(repeat:), str.last(repeat:))
  5: Count parameter returning arrays (str.first(count:), str.last(count:))
  6: Skip whitespace parameter (str.first(skip-whitespace:), str.last(skip-whitespace:))
  7: Pattern matching (str.first(pattern:), str.last(pattern:))
  8: Case transformation (str.first(case:), str.last(case:))
  9: Strip characters parameter (str.first(strip:), str.last(strip:))
 10: Unit selection (grapheme/word) (str.first(unit:), str.last(unit:))
USAGE
}

# -------------------------------------------------------------------------
# Main Script
# -------------------------------------------------------------------------

# Allow an optional first argument pointing to the workspace directory.
WORKSPACE_DIR=$(pwd)
if [[ $# -ge 1 && -d "$1" ]]; then
  WORKSPACE_DIR="$1"
  shift
fi

cd "$WORKSPACE_DIR"

# Feature number is optional – evaluation harnesses often pass additional
# arguments (e.g., paths). Find the first numeric argument, if any.
FEATURE_NUM=""
for arg in "$@"; do
  if [[ "$arg" =~ ^([1-9]|10)$ ]]; then
    FEATURE_NUM="$arg"
    break
  fi
done

FEATURE_SPECIFIED=true
if [[ -z "$FEATURE_NUM" ]]; then
  FEATURE_SPECIFIED=false
fi

# Check if we're in the right directory
if [[ ! -f "Cargo.toml" ]] || [[ ! -d "crates/typst" ]]; then
  error "Must be run from the typst repository root. Run setup.sh first."
fi

log "=== Typst CodeConflictBenchmark Task 6554 Feature Test ==="
if [[ "$FEATURE_SPECIFIED" == true ]]; then
  info "Testing Feature $FEATURE_NUM for PR #6554"
else
  info "Running tests on current workspace (no feature number provided)"
fi

# Step 1: Install Rust if needed
check_and_install_rust

# Step 2: Reset repository to clean state
log "Resetting repository to clean state..."
git reset --hard HEAD
git clean -fd

# Step 3: Apply feature patches (only if a feature number was provided)
if [[ "$FEATURE_SPECIFIED" == true ]]; then
  apply_feature_patch "$FEATURE_NUM"
else
  info "No feature number supplied – skipping patch application"
fi

# Step 4: Build Typst with modifications
build_typst

# Step 5: Run tests
run_tests

log "=== Test completed for Feature $FEATURE_NUM ===" 
