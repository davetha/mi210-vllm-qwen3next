#!/usr/bin/env python3
"""Prefix-caching benefit test: reuse a long shared system prompt across queries.
Measures TTFT (prompt-processing latency) on 1st (cold) vs subsequent (warm) requests.
Prefix caching should cut TTFT on warm requests if the shared prefix is >= the hybrid
block size (~528 tokens)."""
import sys, time, json, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8087"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3-next-thinking-awq"

# ~600-token shared system prompt (above the ~528 hybrid block threshold)
SHARED = ("You are an expert software engineer and computer scientist. " * 60).strip()
QUERIES = ["Explain qusort.", "Explain a hash map.", "Explain a red-black tree.",
           "Explain Dijkstra.", "Explain a bloom filter."]

def call(q):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SHARED},
                     {"role": "user", "content": q}],
        "max_tokens": 8, "temperature": 0.0, "stream": True,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None
    with urllib.request.urlopen(req, timeout=120) as r:
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                try:
                    o = json.loads(line[5:])
                    if o.get("choices", [{}])[0].get("delta", {}).get("content") and ttft is None:
                        ttft = time.time() - t0
                except Exception:
                    pass
    return ttft if ttft is not None else (time.time() - t0)

print("cold (1st, no cache):", f"{call(QUERIES[0]):.3f}s TTFT")
warm = [call(q) for q in QUERIES[1:]]
print("warm (shared prefix cached):", ", ".join(f"{t:.3f}s" for t in warm))
print(f"avg warm TTFT: {sum(warm)/len(warm):.3f}s")
