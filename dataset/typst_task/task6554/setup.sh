#!/bin/bash

REPO_NAME="typst"
REPO_OWNER="typst"
BASE_COMMIT="b8034a343831e8609aec2ec81eb7eeda57aa5d81"  # Base commit before PR #6554

echo "=== Setting up Typst CodeConflictBenchmark Task 6554 ==="
echo "PR #6554: Add default argument for str.first and str.last"
echo "Base commit: $BASE_COMMIT"

# Create a directory for the repo if it doesn't exist
if [ ! -d "$REPO_NAME" ]; then
    echo "Cloning typst repository..."
    git clone https://github.com/${REPO_OWNER}/${REPO_NAME} "$REPO_NAME"
    cd "$REPO_NAME"
else
    echo "Repository directory already exists. Using existing clone."
    cd "$REPO_NAME"
fi

# Checkout base commit
echo "Checking out base commit: $BASE_COMMIT"
git checkout "$BASE_COMMIT"

# Create a branch for our work, delete the branch if it already exists
git branch -D "code-conflict-bench" 2>/dev/null
git checkout -b "code-conflict-bench"

echo "Setup complete! Repository is ready for feature testing."
echo "Available features: feature1, feature2, feature3, feature4, feature5, feature6, feature7"
echo "Run './run_tests.sh <1-7>' to test a specific feature." 