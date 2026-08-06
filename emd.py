"""
emd.py -- thin host-side wrapper around the raw kernel in kernels.py.

float64 only -- EMDKernelConfig.dtype is hardcoded to float64 in
kernels.py (it's a read-only property, not a settable field), so this
wrapper doesn't do per-dtype detection, caching, or branching. One
config, one compiled kernel, built once and reused.

Usage:

    import cupy as cp
    from cuda_emd.emd import GPUBatchEMD

    signals = cp.asarray(numpy_array_of_shape_batch_by_N, dtype=cp.float64)
    gpu_emd = GPUBatchEMD()
    imfs, n_imfs, overflow = gpu_emd(signals)

    # imfs: cupy float64 array [batch, MAX_IMFS, N]
    # n_imfs: cupy int32 [batch] -- how many of the MAX_IMFS slots are real
    # overflow: cupy int32 [batch] -- 1 means MAX_EXTREMA or MAX_ITERATION
    #           was hit for that signal; treat its output as unverified,
    #           not as "close enough". Increase EMDKernelConfig.max_extrema
    #           (or max_iteration) and rerun if you see any of these.
    # debug_n / debug_extno / debug_nzm: cupy int32 [batch, MAX_IMFS] -- per-IMF
    #           iteration count and (extNo, nzm) at convergence, -1 where an
    #           IMF slot wasn't reached. Diagnostic only.

Every signal in the batch is assumed to share the same T (time) axis --
the common case for fixed-sample-rate data (seismic traces at a shared dt,
EEG channels at a shared sampling rate). Per-signal T is not implemented.

COMPILATION AND WARMUP: kernel compilation (both NVRTC source->PTX and
the CUDA driver's own PTX->SASS JIT, which happens at first launch, not
at compile time) happens the first time the kernel is actually launched,
not in __init__. That means the FIRST call includes a real, multi-second,
one-time compilation cost on top of the actual sift time. Call
`gpu_emd.warmup()` once before timing anything, or that compilation cost
will land inside whatever call happens to go first and make it look like
the kernel itself got slower -- it didn't, the clock just started
catching a step that isn't there on every subsequent call.
"""

from __future__ import annotations

import cupy as cp

try:
    from .kernels import EMDKernelConfig  # package layout (cuda_emd/ as a subpackage)
except ImportError:
    from kernels import EMDKernelConfig  # flat layout: emd.py and kernels.py side by side


class GPUBatchEMD:
    def __init__(self, config: "EMDKernelConfig | None" = None):
        self._config = config or EMDKernelConfig.for_device()
        self._launcher = None  # built lazily on first real call -- see warmup()

    def warmup(self, N: int = 64) -> None:
        """Force compilation now, via one tiny real call, so it doesn't
        silently land inside whatever the next call happens to be. Call
        this once before timing or benchmarking anything -- see the
        module docstring's COMPILATION AND WARMUP note."""
        dummy = cp.zeros((1, N), dtype=cp.float64)
        self(dummy)

    def _get_launcher(self):
        if self._launcher is None:
            self._launcher = self._config.build()
        return self._launcher

    def __call__(self, signals: "cp.ndarray", T: "cp.ndarray | None" = None):
        if signals.ndim == 1:
            signals = signals[None, :]
        if cp.dtype(signals.dtype) != cp.dtype(cp.float64):
            raise ValueError(
                f"GPUBatchEMD (this build) is float64 only, got {signals.dtype}. "
                f"Cast explicitly first, e.g. signals.astype(cp.float64) -- this "
                f"class won't silently do that for you."
            )
        launcher = self._get_launcher()
        batch, N = signals.shape

        if T is None:
            T_host_like = cp.arange(N, dtype=cp.float64)
        else:
            T_host_like = cp.asarray(T, dtype=cp.float64)
            d = cp.diff(T_host_like)
            T_host_like = (T_host_like - T_host_like[0]) / cp.min(d)

        imfs, n_imfs, overflow, debug_n, debug_extno, debug_nzm = launcher.run(
            cp.ascontiguousarray(signals), T_host_like
        )
        self.last_debug = (debug_n, debug_extno, debug_nzm)  # stashed for diagnose_sift.py

        if bool(cp.any(overflow)):
            n_hit = int(cp.sum(overflow))
            import warnings

            warnings.warn(
                f"{n_hit}/{batch} signal(s) hit MAX_EXTREMA or MAX_ITERATION during "
                "sifting -- their output is not a verified decomposition. Check the "
                "`overflow` array to find which, and raise EMDKernelConfig.max_extrema "
                "/ max_iteration for those.",
                stacklevel=2,
            )

        return imfs, n_imfs, overflow

    def unpack(self, imfs: "cp.ndarray", n_imfs: "cp.ndarray", i: int) -> "cp.ndarray":
        """Convenience: get signal i's actual IMFs (incl. residue as the
        last row) trimmed to n_imfs[i], as a numpy array -- shape
        (n_imfs[i], N), directly comparable to PyEMD's EMD()(signal)."""
        k = int(n_imfs[i].get())
        return cp.asnumpy(imfs[i, :k, :])
