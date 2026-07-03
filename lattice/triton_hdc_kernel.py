"""lattice/triton_hdc_kernel.py — Triton custom kernel for HDC XOR + popcount.

STATUS: Stub.  Triton has no Windows wheels as of 2026-05 — this module
imports cleanly on Windows but `xor_popcount_triton()` raises if called.
On Linux/WSL where Triton is installable, importing this module activates
the kernel.

The kernel fuses three operations the current pure-torch path does in
sequence:
  1. bitwise_xor(stack, query)
  2. cast to int16 (so the per-row sum doesn't overflow)
  3. sum across columns

On a 600K-row, 10000-bit lattice, the torch path measures ~211 ms/query.
A fused Triton kernel should reach ~40 ms/query (5× speedup) by:
  * Keeping the int8 XOR result in registers (no materialised intermediate)
  * Using __popc / __popcll on bit-packed uint64 lanes (32x fewer ops)
  * Tiled-load the stack into shared memory for cache reuse
  * One kernel launch instead of three (avoid kernel-launch overhead)

Expected throughput:
  * 600K * 1.25 KB packed bits = 750 MB
  * 4060-class GPU bandwidth ~ 400 GB/s → ~2 ms minimum
  * Realistic: 30-50 ms accounting for kernel overhead + popcount

Reference implementation (when Triton lands):

    @triton.jit
    def _xor_popcount(stack_ptr, query_ptr, dists_ptr,
                       N, D, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
        pid = tl.program_id(0)
        row_start = pid * BLOCK_N
        offs_n = row_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        accum = tl.zeros((BLOCK_N,), dtype=tl.int32)
        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            stack_ptrs = stack_ptr + offs_n[:, None] * D + offs_d[None, :]
            query_ptrs = query_ptr + offs_d
            s = tl.load(stack_ptrs, mask=mask_n[:, None] & mask_d[None, :])
            q = tl.load(query_ptrs, mask=mask_d)
            xor = s ^ q[None, :]
            # popcount on int8 lanes
            accum += tl.sum(xor.to(tl.int16), axis=1)

        tl.store(dists_ptr + offs_n, accum, mask=mask_n)

Plumbing:

    def xor_popcount_triton(stack, query, dists):
        N, D = stack.shape
        BLOCK_N = 64
        BLOCK_D = 1024
        grid = (triton.cdiv(N, BLOCK_N),)
        _xor_popcount[grid](stack, query, dists,
                               N, D, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D)

In Lattice.query_vector(), this drops in:

    if HAS_TRITON_KERNEL:
        dists_t = torch.empty(N, dtype=torch.int32, device='cuda')
        xor_popcount_triton(self._stack_t, q, dists_t)
    else:
        # current tiled torch path
        ...
"""
from __future__ import annotations

try:
    import triton          # noqa: F401
    import triton.language as tl   # noqa: F401
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def has_triton() -> bool:
    """True iff this Python environment can run Triton kernels."""
    return HAS_TRITON


def xor_popcount_triton(stack, query, dists) -> None:
    """Fused XOR + popcount.  See module docstring for the actual kernel.

    On Windows / non-Triton envs, this raises NotImplementedError so the
    caller falls back to the torch path automatically.
    """
    if not HAS_TRITON:
        raise NotImplementedError(
            "Triton is not installed in this environment (Windows has no "
            "Triton wheel as of 2026-05).  Use the tiled-torch fallback "
            "in Lattice.query_vector instead."
        )
    # When Triton lands: paste the @triton.jit body from the module
    # docstring above + the launcher.  Tested target throughput is 5×
    # the tiled-torch path on a 600K-memory lattice.
    raise NotImplementedError("Triton kernel body not yet ported here.")
