#!/bin/bash
# Publish to Test PyPI for testing before production release

set -e

echo "=== Publishing to Test PyPI ==="
echo

echo "[1/5] Running linter..."
./lint.sh
echo

echo "[2/5] Cleaning old builds..."
rm -rf dist/ build/ *.egg-info
echo

echo "[3/5] Building package..."
uv run python -m build
echo

echo "[4/5] Checking package with twine..."
uv run twine check dist/*
echo

echo "[5/5] Uploading to Test PyPI..."
uv run twine upload --repository testpypi dist/*
echo

echo "✓ Package published to Test PyPI!"
echo
echo "To test the installation, run:"
echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ llm-tools-server"
echo
echo "Or with uv:"
echo "  uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ llm-tools-server"
