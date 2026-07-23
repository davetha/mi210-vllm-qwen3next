#!/usr/bin/env python3
"""Patch Ray's AMD accelerator manager so single-GPU MoE tuning works under ROCm.

Ray's `get_visible_accelerator_ids_env_var()` raises
    RuntimeError: Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES
inside its workers when HIP_VISIBLE_DEVICES is stripped and ROCR_VISIBLE_DEVICES is set.
On a single-GPU box that's spurious; self-heal by copying ROCR -> HIP instead of raising.

Run inside the container (or bind-mount the patched file over the original):
    python patch_ray_amd_gpu.py /opt/python/lib/python3.14/site-packages/ray/_private/accelerators/amd_gpu.py
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/opt/python/lib/python3.14/site-packages/ray/_private/accelerators/amd_gpu.py"

s = open(path).read()
old = '''        if (
            HIP_VISIBLE_DEVICES_ENV_VAR not in os.environ
            and "ROCR_VISIBLE_DEVICES" in os.environ
        ):
            raise RuntimeError(
                f"Please use {HIP_VISIBLE_DEVICES_ENV_VAR} instead of ROCR_VISIBLE_DEVICES"
            )'''
new = '''        if (
            HIP_VISIBLE_DEVICES_ENV_VAR not in os.environ
            and "ROCR_VISIBLE_DEVICES" in os.environ
        ):
            # PATCH: self-heal instead of raising for single-GPU tuning
            os.environ[HIP_VISIBLE_DEVICES_ENV_VAR] = os.environ["ROCR_VISIBLE_DEVICES"]'''

if new in s:
    print("already patched")
elif old in s:
    open(path, "w").write(s.replace(old, new))
    print("patched", path)
else:
    sys.exit("ERROR: expected pattern not found; Ray version may differ")
