"""GPU validation, narrow upstream build fixes, and installation smoke tests."""
import argparse
import importlib
import os
from pathlib import Path
import re
import subprocess
import sys


def architectures():
    import torch
    from torch.utils.cpp_extension import _get_cuda_arch_flags

    if torch.version.cuda != "12.1" or not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA 12.1 cannot access a GPU. Check the NVIDIA driver and CUDA_VISIBLE_DEVICES.")
    nvcc = Path(os.environ["CUDA_HOME"]) / "bin/nvcc"
    version = subprocess.check_output([str(nvcc), "--version"], text=True)
    if "release 12.1," not in version:
        raise RuntimeError(f"Expected environment-local CUDA 12.1, got: {version}")
    supported = set(re.findall(r"compute_(\d+)", subprocess.check_output(
        [str(nvcc), "--list-gpu-arch"], text=True)))
    caps = set()
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cap = f"{major}.{minor}"
        if f"{major}{minor}" not in supported:
            raise RuntimeError(f"{torch.cuda.get_device_name(i)} (sm_{major}{minor}) is unsupported by fixed CUDA 12.1. A newer GPU needs a newer CUDA/PyTorch stack.")
        caps.add(cap)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}, compute capability {cap}", file=sys.stderr)
        with torch.cuda.device(i):
            assert (torch.ones(4, device=f"cuda:{i}") + 1).sum().item() == 8
    result = ";".join(sorted(caps, key=lambda s: tuple(map(int, s.split(".")))))
    os.environ["TORCH_CUDA_ARCH_LIST"] = result
    _get_cuda_arch_flags()
    print(result)


def patch(sources):
    # CUDA 12 Thrust no longer provides all of these headers transitively.
    includes = {
        "src/3rdparty/concurrent_unordered_map.cuh": ["thrust/execution_policy.h"],
        "src/coordinate_map_gpu.cuh": ["thrust/execution_policy.h", "thrust/unique.h", "thrust/remove.h"],
        "src/coordinate_map_gpu.cu": ["thrust/unique.h", "thrust/remove.h"],
        "src/spmm.cu": ["ATen/ATen.h", "thrust/execution_policy.h", "thrust/reduce.h", "thrust/sort.h"],
    }
    for relative, headers in includes.items():
        path = sources / "MinkowskiEngine" / relative
        text = path.read_text()
        missing = [f"#include <{h}>\n" for h in headers if f"#include <{h}>" not in text]
        if missing:
            pos = text.index("#include")
            path.write_text(text[:pos] + "".join(missing) + text[pos:])
    # Python 3.10 moved the abstract collection types to collections.abc.
    for path in (sources / "MinkowskiEngine/MinkowskiEngine").rglob("*.py"):
        text = path.read_text()
        updated = text.replace("from collections import Sequence, namedtuple", "from collections import namedtuple\nfrom collections.abc import Sequence")
        updated = updated.replace("from collections import Sequence", "from collections.abc import Sequence")
        if updated != text:
            path.write_text(updated)
    # Ninja tracks changed sources, headers, compilers and architecture flags.
    # Upstream otherwise deletes every object and uninstalls before building.
    setup = sources / "MinkowskiEngine/setup.py"
    text = setup.read_text()
    text = text.replace('run_command("rm", "-rf", "build")', '# Retain native objects for incremental Ninja builds.')
    text = text.replace('run_command("pip", "uninstall", "MinkowskiEngine", "-y")', '# Replace the installed wheel only after a successful build.')
    setup.write_text(text)
    cmake = sources / "segmentator/csrc/CMakeLists.txt"
    text = cmake.read_text()
    text = text.replace("set(CMAKE_CXX_STANDARD 14)", "set(CMAKE_CXX_STANDARD 17)\nset(CMAKE_CUDA_STANDARD 17)")
    cmake.write_text(text)


def smoke():
    import numpy as np
    # Molmo's image tiling requires the NumPy 1.24 stack API.
    assert np.stack([480, 640], dtype=np.float32).dtype == np.float32
    import torch
    import MinkowskiEngine as ME
    from torch_scatter import scatter

    for name in ("detectron2._C", "pointnet2._ext", "segmentator", "clip",
                 "segment_anything", "openmask3d", "t_funs3d", "hydra",
                 "pytorch_lightning", "open3d", "transformers"):
        importlib.import_module(name)
    for i in range(torch.cuda.device_count()):
        device = f"cuda:{i}"
        with torch.cuda.device(i):
            coords = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.int32)
            tensor = ME.SparseTensor(torch.ones(2, 2, device=device), coordinates=coords, device=device)
            conv = ME.MinkowskiConvolution(2, 2, kernel_size=1, dimension=3).to(device)
            assert conv(tensor).F.shape == (2, 2)
            out = scatter(torch.ones(2, device=device), torch.zeros(2, dtype=torch.long, device=device))
            assert out.item() == 2
            torch.cuda.synchronize()
    print("Dependency imports and CUDA extension smoke tests passed.")


def flash():
    """Check the compiled kernels and Transformers integration without weights."""
    import torch
    import flash_attn
    from flash_attn import flash_attn_func
    from transformers import AutoModelForCausalLM, Qwen3Config

    if not torch.cuda.is_available():
        raise RuntimeError("FlashAttention validation requires a visible CUDA GPU.")
    torch.manual_seed(42)
    for i in range(torch.cuda.device_count()):
        device = f"cuda:{i}"
        for dtype in (torch.float16, torch.bfloat16):
            for causal in (False, True):
                q, k, v = [torch.randn(2, 64, 4, 64, device=device, dtype=dtype,
                                      requires_grad=True) for _ in range(3)]
                out = flash_attn_func(q, k, v, dropout_p=0.0, causal=causal)
                qr, kr, vr = [x.detach().float().requires_grad_() for x in (q, k, v)]
                scores = qr.transpose(1, 2) @ kr.transpose(1, 2).transpose(-2, -1) / 8
                if causal:
                    mask = torch.ones(64, 64, device=device, dtype=torch.bool).triu(1)
                    scores = scores.masked_fill(mask, float("-inf"))
                reference = (scores.softmax(dim=-1) @ vr.transpose(1, 2)).transpose(1, 2)
                tolerance = 3e-3 if dtype == torch.float16 else 3e-2
                torch.testing.assert_close(out.float(), reference, atol=tolerance, rtol=tolerance)
                grad = torch.randn_like(out)
                actual_grads = torch.autograd.grad(out, (q, k, v), grad)
                reference_grads = torch.autograd.grad(reference, (qr, kr, vr), grad.float())
                for actual, expected in zip(actual_grads, reference_grads):
                    torch.testing.assert_close(actual.float(), expected, atol=tolerance, rtol=tolerance)
                print(f"GPU {i}: {dtype}, causal={causal}: forward/backward matched FP32 reference")

        # Left padding exercises Transformers' variable-length FlashAttention path.
        config = Qwen3Config(vocab_size=128, hidden_size=128, intermediate_size=256,
                             num_hidden_layers=1, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=32, max_position_embeddings=128)
        model = AutoModelForCausalLM.from_config(
            config, attn_implementation="flash_attention_2", torch_dtype=torch.float16
        ).to(device).eval()
        ids = torch.randint(1, 128, (2, 16), device=device)
        mask = torch.ones_like(ids)
        mask[0, :4] = 0
        with torch.no_grad():
            actual = model(ids, attention_mask=mask, use_cache=False).logits
            model.config._attn_implementation = "eager"
            expected = model(ids, attention_mask=mask, use_cache=False).logits
        torch.testing.assert_close(actual[mask.bool()], expected[mask.bool()], atol=3e-3, rtol=3e-3)
        torch.cuda.synchronize(i)
        print(f"GPU {i}: tiny Qwen3 with padding matched eager attention")
    print(f"FlashAttention {flash_attn.__version__} GPU checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("architectures", "patch", "smoke", "flash"))
    parser.add_argument("sources", nargs="?", type=Path)
    args = parser.parse_args()
    if args.action == "patch":
        if args.sources is None:
            parser.error("patch requires a source directory")
        patch(args.sources)
    else:
        globals()[args.action]()
