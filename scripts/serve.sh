#!/usr/bin/env bash
# ninfer-serve 一键后台服务脚本
# =====================================================================
# 用途：把 Qwen3.8-27B 的 HTTP 推理服务（OpenAI / Anthropic 兼容接口）
#       以后台常驻进程方式运行，崩溃自动重启，带 PID 文件和日志，
#       让普通运维人员只记 start/stop/restart/status/logs 五个命令即可。
#
# 用法：
#   scripts/serve.sh start     启动服务（若已在运行则拒绝重复启动）
#   scripts/serve.sh stop      停止服务
#   scripts/serve.sh restart   重启服务（= stop 再 start）
#   scripts/serve.sh status    查看进程/端口/健康检查状态
#   scripts/serve.sh logs      持续查看日志（Ctrl-C 退出）
#
# 所有参数都可用环境变量覆盖（也可直接改下面“配置区”）：
#   NINFER_MODEL / NINFER_HOST / NINFER_PORT / NINFER_DEVICES / NINFER_TP
#   NINFER_MAX_CONTEXT / NINFER_KV_DTYPE / NINFER_API_KEY / NINFER_MODEL_ID
#   NINFER_LOG_DIR
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# ------------------------------ 配置区 ------------------------------
BIN="${NINFER_SERVE_BIN:-./build/apps/ninfer-serve}"   # 服务端二进制
MODEL="${NINFER_MODEL:-/data/models/qwen3.8-27b/qwen3_8_27b_v2.ninfer}"  # 模型文件
HOST="${NINFER_HOST:-0.0.0.0}"       # 监听地址
PORT="${NINFER_PORT:-8000}"          # HTTP 端口
DEVICES="${NINFER_DEVICES:-0,1}"     # 使用的 GPU（双卡 2080Ti 默认 0,1）
TP="${NINFER_TP:-2}"                 # 张量并行度
MAX_CONTEXT="${NINFER_MAX_CONTEXT:-262144}"  # 上下文长度（单并发 262k）
KV_DTYPE="${NINFER_KV_DTYPE:-fp16}"  # KV 缓存精度（fp16 最快；int8 更省显存）
API_KEY="${NINFER_API_KEY:-}"        # 接口鉴权 key；留空 = 不鉴权
MODEL_ID="${NINFER_MODEL_ID:-}"      # 对外报告的 model id；留空 = 用模型自带 id
LOG_DIR="${NINFER_LOG_DIR:-logs}"    # 日志与 PID 文件目录
NAME="ninfer-serve"                  # 服务名（用于 PID/日志文件名）
PID_FILE="$LOG_DIR/$NAME.pid"
LOG_FILE="$LOG_DIR/$NAME.log"
RESTART_SLEEP="${NINFER_RESTART_SLEEP:-5}"  # 崩溃后自动重启的等待秒数
# --------------------------------------------------------------------

# 打印一条带时间戳的信息（中文，运维友好）
say() { echo "[$(date '+%F %T')] $*"; }

usage() {
    echo "用法: scripts/serve.sh {start|stop|restart|status|logs}"
    exit 1
}

# 组装服务端启动参数（可选参数按需附加）
build_args() {
    local -a a=("$MODEL" --host "$HOST" --port "$PORT"
                --devices "$DEVICES" --tp "$TP"
                --max-context "$MAX_CONTEXT" --kv-dtype "$KV_DTYPE")
    [[ -n "$API_KEY" ]] && a+=(--api-key "$API_KEY")
    [[ -n "$MODEL_ID" ]] && a+=(--model-id "$MODEL_ID")
    printf '%s\n' "${a[@]}"
}

# 是否已有服务进程存活
running() {
    [[ -f "$PID_FILE" ]] || return 1
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

# 进程组终止：setsid 启动后，整个组（守护循环 + 服务进程）同属一个进程组
kill_group() {
    local pid="$1"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
}

start() {
    # 前置检查：二进制和模型文件必须存在
    if [[ ! -x "$BIN" ]]; then
        echo "启动失败：找不到服务端二进制 $BIN（先执行 cmake --build build -j）" >&2
        exit 1
    fi
    if [[ ! -f "$MODEL" ]]; then
        echo "启动失败：找不到模型文件 $MODEL（可用 NINFER_MODEL 指定路径）" >&2
        exit 1
    fi
    if running; then
        echo "服务已在运行，PID=$(cat "$PID_FILE")。如需重启请用 restart。"
        return 0
    fi

    mkdir -p "$LOG_DIR"
    say "正在启动 $NAME：模型=$MODEL，设备=$DEVICES，tp=$TP，端口=$HOST:$PORT"
    say "日志文件：$LOG_FILE"

    # 守护循环：服务异常退出(非 0)时等待 RESTART_SLEEP 秒后自动重启；
    # 正常退出(0)视为人工停止，不再拉起。
    # 用 setsid 脱离当前终端，保证关掉 ssh 会话后服务不中断。
    # 循环自身的 stdout/stderr 由下面的 >>"$LOG_FILE" 重定向到日志文件。
    setsid bash -c '
        BIN="$1"; shift
        while true; do
            "$BIN" "$@"
            code=$?
            if [[ $code -eq 0 ]]; then
                echo "[$(date "+%F %T")] 服务正常退出(0)，守护循环结束"
                break
            fi
            echo "[$(date "+%F %T")] 服务异常退出(exit=$code)，'"$RESTART_SLEEP"' 秒后自动重启..."
            sleep '"$RESTART_SLEEP"'
        done
    ' _ "$BIN" $(build_args) >>"$LOG_FILE" 2>&1 &

    # 记录守护进程 PID（即进程组组长，stop 时按组整体结束）
    echo $! > "$PID_FILE"

    # 等一小会儿给出最直观的启动反馈
    sleep 2
    if running; then
        say "服务已启动，PID=$(cat "$PID_FILE")。用 status 查看加载进度，logs 看日志。"
    else
        echo "启动异常：守护进程很快退出了，请查看日志 $LOG_FILE" >&2
        exit 1
    fi
}

stop() {
    if ! running; then
        echo "服务未在运行（没有有效的 PID 文件）。"
        rm -f "$PID_FILE"
        return 0
    fi
    local pid
    pid="$(cat "$PID_FILE")"
    say "正在停止服务（进程组 $pid）..."
    kill_group "$pid"
    # 最多等 10 秒，让服务端完成收尾（释放显存）
    for _ in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    rm -f "$PID_FILE"
    say "已停止。"
}

status() {
    if running; then
        local pid
        pid="$(cat "$PID_FILE")"
        echo "服务状态：运行中（守护 PID=$pid）"
    else
        echo "服务状态：未运行"
        return 1
    fi

    # 端口与健康检查（curl 可用时顺带查 OpenAI 兼容的 /v1/models）
    if command -v curl >/dev/null 2>&1; then
        local auth=()
        [[ -n "$API_KEY" ]] && auth=(-H "Authorization: Bearer $API_KEY")
        local code
        code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "${auth[@]}" \
                "http://$HOST:$PORT/v1/models" 2>/dev/null || true)"
        if [[ "$code" == "200" ]]; then
            echo "健康检查：http://$HOST:$PORT/v1/models -> 200 OK"
        else
            echo "健康检查：端口 $PORT 尚未就绪（HTTP $code；模型可能还在加载，稍后再看）"
        fi
    else
        echo "提示：未安装 curl，跳过健康检查。"
    fi
    echo "日志：$LOG_FILE"
}

logs() {
    [[ -f "$LOG_FILE" ]] || { echo "日志文件还不存在：$LOG_FILE" >&2; exit 1; }
    tail -n 200 -f "$LOG_FILE"
}

[[ $# -ge 1 ]] || usage
case "$1" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; start ;;
    status)  status ;;
    logs)    logs ;;
    *)       usage ;;
esac
