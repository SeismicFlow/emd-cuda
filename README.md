# cuda-emd

Batched Empirical Mode Decomposition (EMD), implemented as a CuPy raw CUDA
kernel. Decomposes an entire batch of signals on the GPU in parallel, with
output that matches [PyEMD](https://github.com/laszukdawid/PyEMD)'s default
`EMD()` behavior.

- **`kernels.py`** — the CUDA source (as a Python string), `EMDKernelConfig`
  (compile-time configuration), and `EMDBatchLauncher` (compiles + launches).
- **`emd.py`** — the host-side wrapper, `GPUBatchEMD`, that most users should
  actually import.

## Requirements

- An NVIDIA GPU (Pascal or newer) and a working CUDA toolchain.
- [`cupy`](https://cupy.dev/) matching your CUDA version.
- `float64` input only — `EMDKernelConfig.dtype` is a read-only property
  hardcoded to `float64`; there's no per-dtype branching or caching to worry
  about, but there's also no `float32` path.

## Install / layout

No packaging is set up here — just drop both files next to each other, or
under a `cuda_emd/` package directory. `emd.py` tries a relative import first
and falls back to a flat import, so either layout works:

```
cuda_emd/
├── kernels.py
└── emd.py
```

or simply `kernels.py` + `emd.py` side by side on your `PYTHONPATH`.

## Quickstart

```python
import cupy as cp
from cuda_emd.emd import GPUBatchEMD  # or `from emd import GPUBatchEMD` (flat layout)

signals = cp.asarray(numpy_array_of_shape_batch_by_N, dtype=cp.float64)

gpu_emd = GPUBatchEMD()
gpu_emd.warmup()  # see "Warmup / first-call cost" below

imfs, n_imfs, overflow = gpu_emd(signals)

# imfs:     cupy float64 [batch, MAX_IMFS, N]
# n_imfs:   cupy int32   [batch]  -- how many of the MAX_IMFS slots are real
# overflow: cupy int32   [batch]  -- 1 means this signal's decomposition is
#           unverified (see "Overflow" below)

# Get signal i's IMFs (incl. residue as the last row), trimmed to n_imfs[i],
# as a numpy array directly comparable to PyEMD's EMD()(signal):
imfs_i = gpu_emd.unpack(imfs, n_imfs, i=0)
```

All signals in a batch share one `N` (signal length) and one time axis `T`.
If you don't pass `T`, it defaults to `arange(N)`. Per-signal `T` is not
supported.

## What it does, in one paragraph

One CUDA thread **block** per input signal, so the batch dimension is where
the outer parallelism comes from — every block runs its own "extract next
IMF" / inner sift loop to its own convergence, independently, with no
cross-block synchronization. Within a block, extrema detection and the
not-a-knot cubic spline envelope solve (via parallel cyclic reduction) are
*also* parallelized across the block's threads; only the small, cheap,
branchy boundary-mirroring step is thread-0-only.

## Fidelity / scope

Matches `PyEMD.EMD()`'s **defaults** exactly, and only those defaults:

| Setting | Value |
|---|---|
| `extrema_detection` | `'simple'` |
| `spline_kind` | `'cubic'` (not-a-knot boundary conditions) |
| `FIXE` / `FIXE_H` | `0` (adaptive stopping, not fixed iteration count) |
| `nbsym` | `2` |
| dtype | `float64` only |

Convergence thresholds are hardcoded in the CUDA source to match PyEMD's
`__init__` defaults: `energy_ratio_thr=0.2`, `std_thr=0.2`, `svar_thr=0.001`,
`total_power_thr=0.005`, `range_thr=0.001`. If your use case needs different
spline kinds, extrema modes, or fixed-iteration sifting, this kernel doesn't
support it yet.

The residue is appended as a final IMF row only if it isn't ~zero
(`max(|residue|) <= 1e-8`), matching PyEMD's own `np.allclose` check —
clean, low-noise signals can legitimately decompose to one fewer row than
you might expect.

## `EMDKernelConfig`

Compile-time parameters, substituted into the CUDA source as `#define`s
(so the compiler can optimize/unroll around them as constants):

| Field | Default | Meaning |
|---|---|---|
| `max_extrema` | `1024` | Shared-memory buffer capacity per side (max/min); NOT the signal length. |
| `nbsym` | `2` | Symmetric boundary points mirrored on each side. |
| `max_iteration` | `1000` | Sift-loop iteration cap per IMF. |
| `max_imfs` | `16` | Output slots; also caps the outer "extract next IMF" loop. |
| `threads_per_block` | `128` | CUDA block size. |

Don't hand-roll a config unless you need to — use **`EMDKernelConfig.for_device()`**:

```python
cfg = EMDKernelConfig.for_device(target_blocks_per_sm=4, min_max_extrema=1024)
gpu_emd = GPUBatchEMD(cfg)
```

This sizes `max_extrema` for **occupancy**, not for maximum per-block
capacity: a large per-block shared-memory footprint leaves room for only
1–2 concurrent blocks per SM, so a big batch runs in sequential "waves" and
any one slow-to-converge signal holds up its whole wave. `for_device()`
instead budgets shared memory for a target number of concurrent blocks per
SM (default 4), only growing `max_extrema` past `min_max_extrema` if the
device has room to do that *and* still hit the occupancy target — and never
shrinking below `min_max_extrema` just to hit an occupancy number, since
real signals (up to ~700 extrema observed in testing) still need to fit.

Fitting always wins over occupancy, though: whatever `max_extrema` is chosen
must actually fit in what the device grants per block. CUDA's shared-memory
ceiling applies to **static + dynamic shared memory combined**, and
`for_device()` accounts for both.

**Pascal note:** the opt-in dynamic-shared-memory API only exists from Volta
onward. On Pascal you're capped at 48KB static+dynamic combined, with no way
to opt in past it — `for_device()` will clamp accordingly rather than build
a config that fails to launch.

## Output arrays

| Array | Shape / dtype | Meaning |
|---|---|---|
| `imfs` | `[batch, MAX_IMFS, N]` float64 | IMFs (+ residue as last real row, if nonzero). |
| `n_imfs` | `[batch]` int32 | Valid row count per signal — trim with `unpack()`. |
| `overflow` | `[batch]` int32 | `1` = `MAX_EXTREMA` or `MAX_ITERATION` was hit; treat as **unverified**, not "close enough". |
| `debug_n`, `debug_extno`, `debug_nzm` | `[batch, MAX_IMFS]` int32 | Diagnostic only: per-IMF iteration count and `(extNo, nzm)` at convergence; `-1` where a slot wasn't reached. Stashed on `gpu_emd.last_debug` after each call. |

### Overflow

`GPUBatchEMD.__call__` raises a `UserWarning` (not an exception) if any
signal in the batch overflowed. Check the `overflow` array to find which
signals, and increase `EMDKernelConfig.max_extrema` and/or `max_iteration`
for those before trusting their output.

## Warmup / first-call cost

Kernel compilation — both NVRTC source→PTX and the CUDA driver's own
PTX→SASS JIT, which happens at first *launch*, not at compile time — is
lazy: it happens on the first real call, not in `__init__`. That means
whichever call goes first eats a real, multi-second, one-time compilation
cost on top of actual sift time. Call:

```python
gpu_emd.warmup()
```

once before timing or benchmarking anything, or that cost will land inside
whatever call happens to go first and look like the kernel got slower than
it is.

## Diagnostics

`EMDBatchLauncher.run_envelope_only()` runs extrema detection + mirroring +
spline (no sift loop) in isolation, for checking the envelope logic against
a numpy/scipy reference independently of sift/convergence behavior. Referenced
diagnostic and reference scripts (`reference/numpy_emd_reference.py`,
`benchmark/diagnose_cuda_envelope.py`) are mentioned in the source comments
as part of the wider project but weren't included in this drop — only
`kernels.py` and `emd.py` are documented here.

## Gotchas

- **dtype mismatches are not silently cast.** Both `GPUBatchEMD.__call__`
  and `EMDBatchLauncher.run()` raise `ValueError` rather than reinterpreting
  bytes — cast explicitly (`signals.astype(cp.float64)`) before calling.
- **Shared memory errors at launch, not at config-build time**, if you
  hand-build an `EMDKernelConfig` with too large a `max_extrema` for your
  device. Prefer `EMDKernelConfig.for_device()`.
- **All signals in a batch share one `T` axis.** Don't mix signals sampled
  at different rates in one batched call.
