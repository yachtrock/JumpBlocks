#!/usr/bin/env bash
# Run JumpBlocks with software rendering on a headless Linux box (e.g. a
# Claude Code cloud container) so the game can be launched and screenshotted
# without a GPU or display.
#
# One-time setup (Ubuntu):
#   sudo apt-get install -y --no-install-recommends \
#       libwayland-dev libxkbcommon-dev libxkbcommon-x11-0 libasound2-dev \
#       libudev-dev pkg-config \
#       mesa-vulkan-drivers vulkan-tools xvfb imagemagick xdotool
#
# Bevy renders through wgpu -> Vulkan -> Mesa lavapipe (llvmpipe) on the CPU.
# Verify the software Vulkan device exists with:
#   xvfb-run -a vulkaninfo --summary
#
# Usage:
#   ./scripts/software-render.sh              # start Xvfb + the game
#   DISPLAY=:99 import -window root shot.png  # screenshot (imagemagick)
#   DISPLAY=:99 xdotool mousemove 640 360 click 1   # click to grab cursor
#   DISPLAY=:99 xdotool mousemove_relative -- -300 60  # mouse-look
#   DISPLAY=:99 xdotool keydown w; sleep 2; DISPLAY=:99 xdotool keyup w  # walk
set -euo pipefail

DISPLAY_NUM="${DISPLAY_NUM:-:99}"
RES="${RES:-1280x720x24}"

if ! pgrep -x Xvfb >/dev/null; then
    Xvfb "$DISPLAY_NUM" -screen 0 "$RES" >/dev/null 2>&1 &
    sleep 1
fi

cargo build
DISPLAY="$DISPLAY_NUM" WGPU_BACKEND=vulkan ./target/debug/jumpblocks "$@"
