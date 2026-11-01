"""
Step 1 of Stage 1: verify Python, PyTorch, CUDA and the RTX 5060 work together.

Run:  python check_env.py
"""
import platform
import sys


def main():
    print(f"Python      : {sys.version.split()[0]}  ({platform.system()} {platform.release()})")

    try:
        import torch
    except ImportError:
        sys.exit("FAIL: PyTorch is not installed. See README step 3.")

    print(f"PyTorch     : {torch.__version__}")
    print(f"Built CUDA  : {torch.version.cuda}")

    if not torch.cuda.is_available():
        sys.exit("FAIL: CUDA not available. Either the NVIDIA driver is missing or a "
                 "CPU-only PyTorch was installed. Reinstall with the cu128 command in the README.")

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    free, total = torch.cuda.mem_get_info(0)

    print(f"GPU         : {name}")
    print(f"Compute cap.: {arch}")
    print(f"VRAM        : {free / 2**30:.2f} GiB free / {total / 2**30:.2f} GiB total")
    print(f"Build archs : {', '.join(arch_list)}")

    if arch not in arch_list and f"compute_{major}{minor}" not in arch_list:
        sys.exit(f"FAIL: this PyTorch build has no kernels for {arch}. "
                 "Install a CUDA 12.8+ build (README step 3).")

    # Real work on the GPU in bf16, the precision we'll train in.
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    c = (a @ b).float()
    torch.cuda.synchronize()
    print(f"bf16 matmul : OK (checksum {c.abs().mean().item():.3f})")
    print(f"bf16 support: {torch.cuda.is_bf16_supported()}")

    # Fused attention kernel we use in the model
    q = torch.randn(1, 6, 256, 64, device="cuda", dtype=torch.bfloat16)
    torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    torch.cuda.synchronize()
    print("SDPA attn   : OK")

    for mod in ("numpy", "datasets", "tokenizers", "tqdm"):
        try:
            m = __import__(mod)
            print(f"{mod:<12}: {getattr(m, '__version__', 'ok')}")
        except ImportError:
            print(f"{mod:<12}: MISSING -> pip install -r requirements.txt")

    print("\nPASS: environment is ready.")


if __name__ == "__main__":
    main()
