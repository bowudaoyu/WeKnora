#!/usr/bin/env bash
#
# Start the self-hosted document-parsing model that WeKnora uses for scanned
# pages (see SCAN_OCR_* in .env.example and docs/扫描件OCR自部署.md).
#
# It serves an OpenAI-compatible endpoint via vLLM, so the backend talks to it
# with the same client it uses for any other remote VLM.
#
# Usage:
#   scripts/scan_ocr_server.sh start|stop|status|logs
#   scripts/scan_ocr_server.sh test <image>   # end-to-end OCR of one page
#
# Env overrides:
#   SCAN_OCR_HOME       install root                    (default /data/ocr)
#   SCAN_OCR_MODEL      HF repo id                      (default ATH-MaaS/OvisOCR2)
#   SCAN_OCR_SERVED_AS  --served-model-name             (default OvisOCR2)
#   SCAN_OCR_PORT       listen port                     (default 9800)
#   SCAN_OCR_GPU_FRAC   --gpu-memory-utilization        (default 0.25)
#   SCAN_OCR_MAX_LEN    --max-model-len                 (default 32768)
#   HF_ENDPOINT         HF mirror                       (default https://hf-mirror.com)
set -euo pipefail

SCAN_OCR_HOME=${SCAN_OCR_HOME:-/data/ocr}
SCAN_OCR_MODEL=${SCAN_OCR_MODEL:-ATH-MaaS/OvisOCR2}
SCAN_OCR_SERVED_AS=${SCAN_OCR_SERVED_AS:-OvisOCR2}
SCAN_OCR_PORT=${SCAN_OCR_PORT:-9800}
# The box this was built for runs the production WeKnora stack on the same
# A100, so the default deliberately claims only a quarter of the card. A 0.8B
# model needs ~2GB of weights; the rest is KV cache and activations.
SCAN_OCR_GPU_FRAC=${SCAN_OCR_GPU_FRAC:-0.25}
# Must fit prompt + image tokens + a full page of output. SCAN_OCR_MAX_TOKENS
# on the WeKnora side defaults to 16384 for the output alone, and a 2880px page
# adds a couple of thousand image tokens on top, so 16384 here would truncate.
SCAN_OCR_MAX_LEN=${SCAN_OCR_MAX_LEN:-32768}

export HF_HOME=${HF_HOME:-$SCAN_OCR_HOME/hf}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
# hf-mirror does not proxy HuggingFace's Xet CAS backend, so weight downloads
# fail with "401 Unauthorized ... cas-server.xethub.hf.co" unless Xet is off.
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

PYBIN="$SCAN_OCR_HOME/venv/bin/python"
# vLLM shells out to helper binaries that pip installed into the venv (ninja,
# for the runtime kernel compile). Invoking $PYBIN directly does not activate
# the venv, so its bin dir has to be put on PATH explicitly or engine startup
# dies with "FileNotFoundError: ... 'ninja'".
export PATH="$SCAN_OCR_HOME/venv/bin:$PATH"
LOG="$SCAN_OCR_HOME/server.log"
PIDFILE="$SCAN_OCR_HOME/server.pid"

running() {
  [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

case "${1:-start}" in
  start)
    if running; then
      echo "already running (pid $(cat "$PIDFILE"))"
      exit 0
    fi
    [[ -x "$PYBIN" ]] || { echo "missing venv at $PYBIN — run the install step first" >&2; exit 1; }
    mkdir -p "$SCAN_OCR_HOME"
    # setsid detaches from the ssh session, otherwise the server dies with it.
    setsid nohup "$PYBIN" -m vllm.entrypoints.openai.api_server \
      --model "$SCAN_OCR_MODEL" \
      --served-model-name "$SCAN_OCR_SERVED_AS" \
      --host 0.0.0.0 --port "$SCAN_OCR_PORT" \
      --gpu-memory-utilization "$SCAN_OCR_GPU_FRAC" \
      --max-model-len "$SCAN_OCR_MAX_LEN" \
      --trust-remote-code \
      > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "started (pid $(cat "$PIDFILE")), logs: $LOG"
    ;;
  stop)
    if running; then
      kill "$(cat "$PIDFILE")"
      echo "stopped"
    else
      echo "not running"
    fi
    rm -f "$PIDFILE"
    ;;
  status)
    if running; then
      echo "running (pid $(cat "$PIDFILE"))"
      curl -sS "http://127.0.0.1:$SCAN_OCR_PORT/v1/models" || true
      echo
    else
      echo "not running"
      exit 1
    fi
    ;;
  logs)
    tail -f "$LOG"
    ;;
  test)
    # Exercises the exact contract the Go backend uses: OpenAI-compatible
    # chat/completions, one data: URI image, the published parsing prompt.
    IMG=${2:-}
    [[ -n "$IMG" && -f "$IMG" ]] || { echo "usage: $0 test <image-file>" >&2; exit 2; }
    "$PYBIN" - "$IMG" "$SCAN_OCR_PORT" "$SCAN_OCR_SERVED_AS" <<'PYEOF'
import base64, json, mimetypes, sys, time, urllib.request

path, port, model = sys.argv[1], sys.argv[2], sys.argv[3]
mime = mimetypes.guess_type(path)[0] or "image/png"
with open(path, "rb") as fh:
    data = base64.b64encode(fh.read()).decode()

prompt = (
    "Extract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. For charts or images, "
    "represent them using an HTML image tag: "
    '<img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, '
    "right, bottom are bounding box coordinates scaled to [0, 1000). Format formulas "
    "as LaTeX. Format tables as HTML: <table>...</table>. Transcribe all other text "
    "as standard Markdown. Preserve the original text without translation or paraphrasing."
)

body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
    ]}],
    "max_tokens": 16384,
    "temperature": 0,
}).encode()

req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=body, headers={"Content-Type": "application/json"},
)
t0 = time.time()
with urllib.request.urlopen(req, timeout=300) as resp:
    out = json.load(resp)
elapsed = time.time() - t0

usage = out.get("usage", {})
print(f"--- {elapsed:.1f}s, prompt={usage.get('prompt_tokens')} "
      f"completion={usage.get('completion_tokens')} tokens ---")
print(out["choices"][0]["message"]["content"])
PYEOF
    ;;
  *)
    echo "usage: $0 start|stop|status|logs|test <image>" >&2
    exit 2
    ;;
esac
