#!/bin/bash
# LLM API Server - Linting routine
# Run this script before committing code

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}${BOLD}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}${BOLD}║    LLM API Server Linting Routine      ║${NC}"
echo -e "${BLUE}${BOLD}╚════════════════════════════════════════╝${NC}"
echo ""

# Check if we're in the right directory
if [ ! -d "llm_api_server" ]; then
    echo -e "${RED}Error: llm_api_server/ not found. Please run from llm-api-server/ directory.${NC}"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required tools
echo -e "${YELLOW}Checking for required tools...${NC}"

MISSING_TOOLS=()

if ! command_exists black; then
    MISSING_TOOLS+=("black")
fi

if ! command_exists ruff; then
    MISSING_TOOLS+=("ruff")
fi

if [ ${#MISSING_TOOLS[@]} -ne 0 ]; then
    echo -e "${RED}Missing required tools: ${MISSING_TOOLS[*]}${NC}"
    echo -e "${YELLOW}Install with: pip install black ruff${NC}"
    echo -e "${YELLOW}Or: pip install -e '.[dev]'${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All required tools found${NC}"
echo ""

# Parse command line arguments
TARGET="${1:-llm_api_server}"  # Default to llm_api_server directory
FIX_MODE="${2:-fix}"  # Default to fix mode

# Count Python files
PY_COUNT=$(find "$TARGET" -name "*.py" | wc -l)
echo -e "${BLUE}Found ${PY_COUNT} Python file(s) to check${NC}"
echo ""

# Step 1: Black formatting
echo -e "${BLUE}${BOLD}[1/3] Running Black formatter...${NC}"
if black "$TARGET" --quiet 2>&1 | grep -q "reformatted"; then
    FILES_CHANGED=$(black "$TARGET" 2>&1 | grep -c "reformatted" || true)
    echo -e "${GREEN}✓ Black formatted ${FILES_CHANGED} file(s)${NC}"
elif black "$TARGET" --quiet; then
    echo -e "${GREEN}✓ All files already formatted${NC}"
else
    echo -e "${RED}✗ Black formatting failed${NC}"
    exit 1
fi
echo ""

# Step 2: Ruff linting
if [ "$FIX_MODE" == "check" ]; then
    echo -e "${BLUE}${BOLD}[2/3] Running Ruff linter (check only)...${NC}"
    if ruff check "$TARGET" 2>&1; then
        echo -e "${GREEN}✓ Ruff linting passed${NC}"
    else
        echo -e "${YELLOW}⚠ Ruff found issues (run ./lint.sh to auto-fix)${NC}"
        exit 1
    fi
else
    echo -e "${BLUE}${BOLD}[2/3] Running Ruff linter (with auto-fix)...${NC}"
    RUFF_OUTPUT=$(ruff check "$TARGET" --fix 2>&1)
    if [ -z "$RUFF_OUTPUT" ]; then
        echo -e "${GREEN}✓ No issues found${NC}"
    else
        echo "$RUFF_OUTPUT"
        echo -e "${GREEN}✓ Ruff auto-fixed issues${NC}"
    fi
fi
echo ""

# Step 3: Final check
echo -e "${BLUE}${BOLD}[3/3] Running final Ruff check...${NC}"
if ruff check "$TARGET" --quiet 2>&1; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    FINAL_STATUS=0
else
    ruff check "$TARGET" 2>&1
    echo -e "${YELLOW}⚠ Some issues remain - please review above${NC}"
    FINAL_STATUS=1
fi
echo ""

# Optional: MyPy type checking
if command_exists mypy; then
    echo -e "${BLUE}${BOLD}[Optional] MyPy type checking available${NC}"
    echo -e "${YELLOW}Run manually: mypy llm_api_server/${NC}"
    echo ""
fi

# Summary
if [ $FINAL_STATUS -eq 0 ]; then
    echo -e "${GREEN}${BOLD}╔════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}${BOLD}║  ✓ Linting complete - ready to commit! ║${NC}"
    echo -e "${GREEN}${BOLD}╚════════════════════════════════════════╝${NC}"
else
    echo -e "${YELLOW}${BOLD}╔════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}${BOLD}║  ⚠ Please fix remaining issues above   ║${NC}"
    echo -e "${YELLOW}${BOLD}╚════════════════════════════════════════╝${NC}"
fi
echo ""

echo -e "${BLUE}Usage examples:${NC}"
echo -e "  ${BOLD}./lint.sh${NC}                    # Lint llm_api_server/"
echo -e "  ${BOLD}./lint.sh setup.py${NC}           # Lint specific file"
echo -e "  ${BOLD}./lint.sh llm_api_server check${NC}  # Check only (no auto-fix)"
echo ""

exit $FINAL_STATUS
