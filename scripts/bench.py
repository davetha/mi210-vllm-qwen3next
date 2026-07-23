#!/usr/bin/env python3
"""Single-stream generation throughput for a vLLM OpenAI-compatible endpoint.

Measures completion tokens / generation wall time, using streaming + first-token
timing so prompt-processing time is excluded.

Usage: bench.py [port] [model] [max_tokens]
"""
import sys, time, json, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3-next"
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 256
PROMPT = ("Write a detailed step-by-step explanation of how a binary search tree "
          "insertion works, then implement it in Python with comments.")

body = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": MAXTOK,
    "temperature": 0.7,
    "stream": True,
    "stream_options": {"include_usage": True},
}).encode()

req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                            data=body, headers={"Content-Type": "application/json"})
t0 = time.time()
first = None
ntok = 0
usage = None
with urllib.request.urlopen(req, timeout=600) as r:
    for raw in r:
        line = raw.decode(errors="ignore").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except Exception:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices", []):
            if ch.get("delta", {}).get("content"):
                if first is None:
                    first = time.time()
                ntok += 1
t_end = time.time()

ttft = (first - t0) if first else float("nan")
gen_time = (t_end - first) if first else float("nan")
comp = usage.get("completion_tokens") if usage else ntok
gen_tps = (comp / gen_time) if gen_time and gen_time > 0 else float("nan")
print(f"TTFT           : {ttft:.3f} s")
print(f"gen wall time  : {gen_time:.3f} s")
print(f"completion tok : {comp}  (streamed chunks: {ntok})")
print(f"GEN throughput : {gen_tps:.1f} tok/s   <-- single-stream")
if usage:
    print(f"usage          : {usage}")
