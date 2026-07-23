#!/bin/bash
# Wait for health then run the full benchmark matrix. Usage: bench-all.sh [label]
LABEL="${1:-run}"
for i in $(seq 1 60); do
  [ "$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:8087/health)" = "200" ] && break
  sleep 5
done
if [ "$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:8087/health)" != "200" ]; then
  echo "[$LABEL] NOT HEALTHY - container status:"; docker ps -a --filter name=vllm-tp2 --format '{{.Status}}'
  docker logs vllm-tp2 2>&1 | grep -iE 'error|traceback|unrecognized|invalid|not supported' | tail -5
  exit 1
fi
python3 /tmp/bench.py 8087 qwen3-next-thinking-awq 64 >/dev/null 2>&1   # warmup
S=$(python3 /tmp/bench.py 8087 qwen3-next-thinking-awq 256 2>&1 | grep -oE '[0-9.]+ tok/s' | head -1)
A32=$(python3 /tmp/bench_concurrent.py 8087 qwen3-next-thinking-awq 32 128 2>&1 | grep -oE 'AGGREGATE tput : [0-9.]+' | grep -oE '[0-9.]+')
A128=$(python3 /tmp/bench_concurrent.py 8087 qwen3-next-thinking-awq 128 128 2>&1 | grep -oE 'AGGREGATE tput : [0-9.]+' | grep -oE '[0-9.]+')
A256=$(python3 /tmp/bench_concurrent.py 8087 qwen3-next-thinking-awq 256 128 2>&1 | grep -oE 'AGGREGATE tput : [0-9.]+' | grep -oE '[0-9.]+')
echo "[$LABEL] single=$S  agg32=$A32  agg128=$A128  agg256=$A256"
