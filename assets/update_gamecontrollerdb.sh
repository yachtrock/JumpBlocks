#!/bin/bash
# Re-fetch the latest SDL GameControllerDB mappings
# Source: https://github.com/mdqinc/SDL_GameControllerDB
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
curl -sL "https://raw.githubusercontent.com/mdqinc/SDL_GameControllerDB/master/gamecontrollerdb.txt" \
    -o "$SCRIPT_DIR/gamecontrollerdb.txt"
echo "Updated gamecontrollerdb.txt ($(wc -l < "$SCRIPT_DIR/gamecontrollerdb.txt") mappings)"
