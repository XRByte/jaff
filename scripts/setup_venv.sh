#! /usr/bin/env bash

run_command() {
    if ! "$@"; then
        echo "Unable to run command: $*"
        exit 1
    fi
}

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
SOURCE_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$SOURCE_DIR/.venv"

if [[ ! -f "$SOURCE_DIR/pyproject.toml" ]]; then
    echo "pyproject.toml not found at: $SOURCE_DIR"
    exit 1
fi

mode="user"
while [[ $# -gt 0 ]]; do
    case "$1" in
    --user)
        mode="user"
        shift
        ;;
    --dev)
        mode="dev"
        shift
        ;;
    *)
        echo "Invalid option: $1"
        exit 1
        ;;
    esac
done

if ! command -v curl; then
    echo "Curl not found. Please install curl to continue"
    exit 1
fi

if ! command -v uv; then
    echo "uv not installed. Installing uv ..."
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        echo "Unable to install uv. Please install uv to continue"
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv; then
    echo "Unable to detect uv command"
    exit 1
fi


run_command uv venv "$VENV_DIR"
if [[ $mode == "user" ]]; then
    run_command uv pip install -e "$SOURCE_DIR"
elif [[ $mode == "dev" ]]; then
    run_command uv pip install -e "$SOURCE_DIR[dev]"
fi

echo
echo "Virtual environment successfully setup at:"
echo "$VENV_DIR"

echo
echo "Activate it with:"
echo "source $VENV_DIR/bin/activate"
