#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
APP_PORT="${APP_PORT:-5050}" TCP_PORT="${TCP_PORT:-5566}" ENABLE_VLM=0 python3 app.py
