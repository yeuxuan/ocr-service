#!/bin/bash
# ============================================================
# GLM-OCR Service — Optimized for M4 Max 32GB
# Usage: ./start.sh {start|stop|restart|status|logs}
# ============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$PROJECT_DIR/.pids"
MLX_PID_FILE="$PID_DIR/mlx-server.pid"
API_PID_FILE="$PID_DIR/api-server.pid"
MLX_LOG="$PROJECT_DIR/mlx-server.log"
API_LOG="$PROJECT_DIR/api.log"

# ---------- Tunable Parameters ----------
export MLX_METAL_MEMORY_BUDGET=20000000000
export OCR_MAX_CONCURRENT=3

MLX_SERVER_PORT=8080
API_SERVER_PORT=5002

# ---------- Colors ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ---------- Helpers ----------

_pid_alive() {
    [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null
}

_kill_pid_file() {
    local pf="$1" name="$2"
    if [[ ! -f "$pf" ]]; then
        return 0
    fi
    local pid
    pid=$(cat "$pf")
    if kill -0 "$pid" 2>/dev/null; then
        echo -e "  Stopping ${name} (PID $pid)..."
        kill "$pid" 2>/dev/null
        for _ in $(seq 1 20); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.5
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "  ${YELLOW}Force killing ${name}...${NC}"
            kill -9 "$pid" 2>/dev/null || true
            sleep 1
        fi
        echo -e "  ${GREEN}${name} stopped.${NC}"
    fi
    rm -f "$pf"
}

_port_in_use() {
    lsof -i :"$1" -sTCP:LISTEN >/dev/null 2>&1
}

_wait_for_port() {
    local port=$1 timeout=$2 label=$3
    echo -n "  Waiting for ${label}"
    for i in $(seq 1 "$timeout"); do
        if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1 || \
           curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
            echo -e " ${GREEN}ready!${NC} (${i}s)"
            return 0
        fi
        echo -n "."
        sleep 2
    done
    echo -e " ${RED}timeout after $((timeout*2))s${NC}"
    return 1
}

# ---------- Commands ----------

do_start() {
    mkdir -p "$PID_DIR"

    if _pid_alive "$MLX_PID_FILE" && _pid_alive "$API_PID_FILE"; then
        echo -e "${YELLOW}All services already running. Use '$(basename "$0") restart' to restart.${NC}"
        do_status
        return 0
    fi

    echo -e "${BOLD}Starting GLM-OCR Service...${NC}"
    echo ""

    # -- mlx-vlm --
    if _pid_alive "$MLX_PID_FILE"; then
        echo -e "  ${CYAN}mlx-vlm${NC} already running (PID $(cat "$MLX_PID_FILE"))"
    else
        if _port_in_use "$MLX_SERVER_PORT"; then
            echo -e "  ${RED}Port ${MLX_SERVER_PORT} already in use. Stop the occupying process first.${NC}"
            return 1
        fi
        echo -e "  ${CYAN}mlx-vlm${NC} starting on :${MLX_SERVER_PORT}..."
        (
            source "$PROJECT_DIR/.venv-mlx/bin/activate"
            exec python -m mlx_vlm.server \
                --model mlx-community/GLM-OCR-bf16 \
                --port "$MLX_SERVER_PORT" \
                --trust-remote-code \
                --max-kv-size 16384 \
                --vision-cache-size 100 \
                --prefill-step-size 4096 \
                --max-tokens 8192 \
                --log-level INFO
        ) >> "$MLX_LOG" 2>&1 &
        echo $! > "$MLX_PID_FILE"
        _wait_for_port "$MLX_SERVER_PORT" 60 "mlx-vlm" || {
            echo -e "  ${RED}mlx-vlm failed to start. Check: tail -50 $MLX_LOG${NC}"
            rm -f "$MLX_PID_FILE"
            return 1
        }
    fi

    # -- API --
    if _pid_alive "$API_PID_FILE"; then
        echo -e "  ${CYAN}API${NC} already running (PID $(cat "$API_PID_FILE"))"
    else
        if _port_in_use "$API_SERVER_PORT"; then
            echo -e "  ${RED}Port ${API_SERVER_PORT} already in use. Stop the occupying process first.${NC}"
            return 1
        fi
        echo -e "  ${CYAN}API${NC} starting on :${API_SERVER_PORT}..."
        (
            source "$PROJECT_DIR/.venv-sdk/bin/activate"
            cd "$PROJECT_DIR"
            exec python -m uvicorn app.main:app \
                --host 0.0.0.0 \
                --port "$API_SERVER_PORT" \
                --workers 1
        ) >> "$API_LOG" 2>&1 &
        echo $! > "$API_PID_FILE"
        sleep 2
        if ! _pid_alive "$API_PID_FILE"; then
            echo -e "  ${RED}API failed to start. Check: tail -50 $API_LOG${NC}"
            rm -f "$API_PID_FILE"
            return 1
        fi
        echo -e "  ${GREEN}API ready!${NC}"
    fi

    echo ""
    do_status
}

do_stop() {
    echo -e "${BOLD}Stopping GLM-OCR Service...${NC}"
    _kill_pid_file "$API_PID_FILE" "API server"
    _kill_pid_file "$MLX_PID_FILE" "mlx-vlm server"
    echo -e "${GREEN}All services stopped.${NC}"
}

do_restart() {
    do_stop
    echo ""
    do_start
}

do_status() {
    echo -e "${BOLD}GLM-OCR Service Status${NC}"
    echo -e "──────────────────────────────────────────"

    # mlx-vlm
    if _pid_alive "$MLX_PID_FILE"; then
        local mlx_pid
        mlx_pid=$(cat "$MLX_PID_FILE")
        local mlx_mem
        mlx_mem=$(ps -o rss= -p "$mlx_pid" 2>/dev/null | awk '{printf "%.0f", $1/1024}')
        echo -e "  mlx-vlm  : ${GREEN}running${NC}  PID ${mlx_pid}  RAM ${mlx_mem}MB  :${MLX_SERVER_PORT}"
    else
        echo -e "  mlx-vlm  : ${RED}stopped${NC}"
    fi

    # API
    if _pid_alive "$API_PID_FILE"; then
        local api_pid
        api_pid=$(cat "$API_PID_FILE")
        local api_mem
        api_mem=$(ps -o rss= -p "$api_pid" 2>/dev/null | awk '{printf "%.0f", $1/1024}')
        echo -e "  API      : ${GREEN}running${NC}  PID ${api_pid}  RAM ${api_mem}MB  :${API_SERVER_PORT}"
    else
        echo -e "  API      : ${RED}stopped${NC}"
    fi

    echo -e "──────────────────────────────────────────"
    echo -e "  Concurrency  : ${OCR_MAX_CONCURRENT} tasks"
    echo -e "  MLX budget   : $((MLX_METAL_MEMORY_BUDGET / 1000000000)) GB"
    echo -e "  Logs         : tail -f mlx-server.log api.log"
    echo ""
}

do_logs() {
    local target="${1:-all}"
    case "$target" in
        mlx)  exec tail -f "$MLX_LOG" ;;
        api)  exec tail -f "$API_LOG" ;;
        all)  exec tail -f "$MLX_LOG" "$API_LOG" ;;
        *)    echo "Usage: $0 logs {mlx|api|all}" ;;
    esac
}

# ---------- Main ----------
case "${1:-start}" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_restart ;;
    status)  do_status ;;
    logs)    do_logs "${2:-all}" ;;
    *)
        echo "Usage: $(basename "$0") {start|stop|restart|status|logs [mlx|api|all]}"
        exit 1
        ;;
esac
