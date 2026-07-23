#!/usr/bin/env python3
"""Aggregate throughput under N concurrent streams for a vLLM OpenAI endpoint.

Fires N parallel streaming requests and reports total completion tokens / wall time.

Usage: bench_concurrent.py [port] [model] [concurrency] [max_tokens]
"""
import sys, time, json, threading, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8000"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3-next"
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 16
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 256
PROMPTS = [
    "Explain how a hash map handles collisions, with a Python example.",
    "Write a Python function to detect a cycle in a linked list and explain it.",
    "Describe the CAP theorem and give a concrete tradeoff example.",
    "Implement quicksort in Python and analyze its complexity.",
]

results = [0] * CONC

def worker(i):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
        "max_tokens": MAXTOK, "temperature": 0.7, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                data=body, headers={"Content-Type": "application/json"})
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode(errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    obj = json.loads(d)
                except Exception:
                    continue
                if obj.get("usage") and obj["usage"].get("completion_tokens"):
                    n = obj["usage"]["completion_tokens"]
    except Exception as e:
        print(f"worker {i} error: {e}")
    results[i] = n

t0 = time.time()
threads = [threading.Thread(target=worker, args=(i,)) for i in range(CONC)]
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = time.time() - t0
total = sum(results)
print(f"concurrency    : {CONC}")
print(f"wall time      : {wall:.2f} s")
print(f"total comp tok : {total}")
print(f"AGGREGATE tput : {total / wall:.1f} tok/s")
print(f"per-stream avg : {total / wall / CONC:.1f} tok/s")
