#!/bin/bash
# Publish to production PyPI

set -e

echo "=== Publishing to PyPI (Production) ==="
echo

# Check if we're on main branch
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
    echo "⚠️  Warning: You're not on the main branch (current: $BRANCH)"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  Warning: You have uncommitted changes"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get version from pyproject.toml
VERSION=$(grep "^version" pyproject.toml | cut -d'"' -f2)
echo "Publishing version: $VERSION"
echo

# Confirm publication
read -p "Are you sure you want to publish to PyPI? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo "[1/6] Running linter..."
./lint.sh
echo

echo "[2/6] Cleaning old builds..."
rm -rf dist/ build/ *.egg-info
echo

echo "[3/6] Building package..."
uv run python -m build
echo

echo "[4/6] Checking package with twine..."
uv run twine check dist/*
echo

echo "[5/6] Uploading to PyPI..."
uv run twine upload dist/*
echo

echo "[6/6] Creating git tag..."
git tag "v$VERSION" 2>/dev/null && git push origin "v$VERSION" || echo "Tag v$VERSION already exists"
echo

echo "✓ Package published to PyPI!"
echo
echo "View at: https://pypi.org/project/llm-api-server/$VERSION/"
echo
echo "Users can now install with:"
echo "  pip install llm-api-server"
echo "  uv add llm-api-server"
