#!/usr/bin/env python3
"""Patch vLLM's benchmarks/kernels/benchmark_moe.py to tune WITHOUT Ray.

On a single-GPU ROCm box Ray gives no parallelism and its device-assignment breaks
(HIP/ROCR visibility). This rewrites the tuner to run the benchmark worker in-process,
keeping HIP_VISIBLE_DEVICES so torch sees the card.

Run inside the container (or bind-mount the patched copy over the original):
    python patch_benchmark_moe.py /app/vllm/benchmarks/kernels/benchmark_moe.py
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/app/vllm/benchmarks/kernels/benchmark_moe.py"
s = open(path).read()

# 1) undecorate the Ray worker -> plain class
s = s.replace("@ray.remote(num_gpus=1)\nclass BenchmarkWorker:", "class BenchmarkWorker:")

# 2) single visible GPU -> device index 0
s = s.replace("self.device_id = int(ray.get_gpu_ids()[0])", "self.device_id = 0")

# 3) replace the ray.init + distribute block in main() with serial in-process execution
old = '''    if current_platform.is_rocm() and "HIP_VISIBLE_DEVICES" in os.environ:
        # Ray will set ROCR_VISIBLE_DEVICES for device visibility
        logger.warning(
            "Ray uses ROCR_VISIBLE_DEVICES to control device accessibility."
            "Replacing HIP_VISIBLE_DEVICES with ROCR_VISIBLE_DEVICES."
        )
        val = os.environ["HIP_VISIBLE_DEVICES"]
        os.environ["ROCR_VISIBLE_DEVICES"] = val
        del os.environ["HIP_VISIBLE_DEVICES"]

    ray.init()
    num_gpus = int(ray.available_resources()["GPU"])
    workers = [BenchmarkWorker.remote(args.seed) for _ in range(num_gpus)]

    def _distribute(method: str, inputs: list[Any]) -> list[Any]:
        outputs = []
        worker_idx = 0
        for input_args in inputs:
            worker = workers[worker_idx]
            worker_method = getattr(worker, method)
            output = worker_method.remote(*input_args)
            outputs.append(output)
            worker_idx = (worker_idx + 1) % num_gpus
        return ray.get(outputs)'''
new = '''    # PATCH: single-GPU serial execution, no Ray (avoids ROCm device-visibility break)
    _local_worker = BenchmarkWorker(args.seed)
    num_gpus = 1

    def _distribute(method: str, inputs: list[Any]) -> list[Any]:
        m = getattr(_local_worker, method)
        return [m(*input_args) for input_args in inputs]'''

# 4) plain tqdm instead of ray's (no cluster needed)
s = s.replace("from ray.experimental.tqdm_ray import tqdm", "from tqdm import tqdm")

if "@ray.remote" not in s and new.split("\n", 2)[1] in s:
    print("already patched")
elif old in s:
    open(path, "w").write(s.replace(old, new))
    print("patched", path)
else:
    sys.exit("ERROR: expected pattern not found; vLLM version may differ")
