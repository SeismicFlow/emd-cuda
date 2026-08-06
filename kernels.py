"""
kernels.py -- batched EMD sifting as a CuPy raw kernel.

DESIGN, IN ONE PARAGRAPH
-------------------------
One CUDA thread BLOCK per input signal. The batch dimension (gridDim.x =
number of signals) is where the first layer of parallelism comes from:
every block runs its own independent outer "extract next IMF" loop and
inner "sift" loop, to its own convergence, with zero cross-block
synchronization, because every signal's decomposition is 100% independent
of every other's. Within a block, extrema detection (find_extrema_parallel)
and the tridiagonal spline solve (notaknot_cubic_spline_block, via
parallel cyclic reduction) are ALSO split across the block's threads --
both were re-verified against the numpy reference (and, for the spline
solve, against scipy directly) before being parallelized here. Only the
boundary-mirroring logic (mirror_one_side, ~8-way branchy, small and
cheap regardless) is still thread-0-only, deliberately -- it's not where
the remaining cost is.

SHARED MEMORY BUDGET
----------------------------------------------------------
Static (no-opt-in) shared memory is capped at 48KB per block on every
CUDA architecture from Pascal through Blackwell. Everything above that
requires `extern __shared__` (dynamic) allocation *and* an explicit
`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, N)`
opt-in call, up to the device's physical ceiling (Pascal: 48KB, no opt-in
path; Volta/Turing: 64-96KB; Ampere: ~163KB; Ada: ~99KB; Hopper: ~227KB).

Within that budget, extrema counts (not signal length N) are what has to
stay bounded, because that's what lives in shared memory -- the raw
per-sample signal/residue/imf arrays stay in global memory with coalesced
access instead. One mirrored-extrema-count-sized buffer set (M = MAX_EXTREMA
+ 2*nbsym elements) is:
    ext_pos[M], ext_val[M]              (reused for max envelope then min)
    cp[M], rhs[M], pcr_a[M], pcr_c[M]   (parallel cyclic reduction scratch,
                                          6 float64 arrays; see
                                          notaknot_cubic_spline_block)
    ind_max[M], ind_min[M]              (raw pre-mirror indices, int32)
  bytes/M-element = 6*8 + 2*4 = 56.

MAX_EXTREMA is NOT sized by "how much shared memory is available" (an
earlier version did that, and it tanked occupancy -- see
EMDKernelConfig.for_device()'s docstring for why). It's sized to hit a
target concurrent-blocks-per-SM count instead, falling back to a 1024
floor so realistic signals (up to ~700 extrema observed in testing) are
never under-provisioned just to hit an occupancy number.

WHAT'S IN SCOPE (matches reference/numpy_emd_reference.py exactly)
---------------------------------------------------------------------
extrema_detection='simple', spline_kind='cubic' (not-a-knot BCs -- see
the notaknot_cubic_spline_block design note below), FIXE=FIXE_H=0,
nbsym=2, float64. If it doesn't match PyEMD on those defaults, it isn't
supposed to be run with anything else yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp

# ---------------------------------------------------------------------------
# CUDA source
# ---------------------------------------------------------------------------
# %(MAX_EXTREMA)s, %(NBSYM)s, %(MAX_ITERATION)s, %(THREADS)s are substituted
# from Python at compile time (see EMDKernelConfig.build()) so they become
# compile-time constants the compiler can unroll/optimize around, instead of
# runtime values.
_CUDA_SOURCE = r"""
extern "C" {

#define MAX_EXTREMA   %(MAX_EXTREMA)d
#define NBSYM         %(NBSYM)d
#define MAX_ITERATION %(MAX_ITERATION)d
#define THREADS       %(THREADS)d
#define MBUF          (MAX_EXTREMA + 2 * NBSYM)
#define MAX_IMFS      %(MAX_IMFS)d
#define REAL_T        %(REAL_T)s

// ---- convergence thresholds, matching PyEMD's __init__ defaults exactly ----
__device__ __forceinline__ REAL_T energy_ratio_thr() { return 0.2; }
__device__ __forceinline__ REAL_T std_thr()          { return 0.2; }
__device__ __forceinline__ REAL_T svar_thr()          { return 0.001; }
__device__ __forceinline__ REAL_T total_power_thr()   { return 0.005; }
__device__ __forceinline__ REAL_T range_thr()          { return 0.001; }

// =====================================================================
// Shared memory layout for one block == one signal's working set.
// Sized against MBUF (extrema count), NOT N (sample count) -- see the
// budget comment at the top of this file.
// =====================================================================
struct SharedScratch {
    REAL_T ext_pos[MBUF];   // reused: mirrored max positions, then min positions
    REAL_T ext_val[MBUF];   // reused: mirrored max values, then min values
    REAL_T cp[MBUF];        // repurposed as PCR's b[] (diagonal) -- see notaknot_cubic_spline_block
    REAL_T rhs[MBUF];       // repurposed as PCR's d[] (RHS); ends up holding M[] (second derivatives) in place
    REAL_T pcr_a[MBUF];     // PCR's a[] (sub-diagonal) -- new for the parallel tridiagonal solve
    REAL_T pcr_c[MBUF];     // PCR's c[] (super-diagonal) -- new for the parallel tridiagonal solve
    int    ind_max[MBUF];   // raw (pre-mirror) maxima indices
    int    ind_min[MBUF];   // raw (pre-mirror) minima indices
    // small scalar bookkeeping, cheap, not worth budgeting precisely
    int    n_max, n_min, n_ext_mirrored_max, n_ext_mirrored_min;
    int    n_zer;            // zero-crossing count, needed for the f2 stopping test
    int    bad_envelope;     // set if any max-envelope knot < 0 or any min-envelope knot > 0
    int    lsym, rsym;
    int    overflow;        // set to 1 if MAX_EXTREMA was exceeded this sift
};

// ---------------------------------------------------------------------
// Extrema detection ("simple" / discrete mode). Thread 0 only: this is
// a sequential scan with plateau handling, mirrors
// EMDReference.find_extrema in the numpy reference file line for line.
// Writes raw (un-mirrored) indices into sh->ind_max / sh->ind_min.
// ---------------------------------------------------------------------
__device__ void find_extrema_serial(
    const REAL_T* T, const REAL_T* S, int N, SharedScratch* sh
) {
    // Zero crossings: sign changes (S[i]*S[i+1] < 0), plus any maximal run
    // of consecutive exact-zero samples -- including a run of length 1, an
    // isolated zero -- counted as one crossing (mirrors the `indzer`
    // computation in EMDReference/PyEMD's _find_extrema_simple, which
    // collapses every such run, isolated or not, to a single index).
    int n_zer = 0;
    bool in_zero_run = false;
    int zero_run_start = -1;
    for (int i = 0; i < N; ++i) {
        bool is_zero = (S[i] == 0.0);
        if (is_zero && !in_zero_run) { in_zero_run = true; zero_run_start = i; }
        if (!is_zero && in_zero_run) {
            // FIX: this used to only count runs of length >= 2
            // (`run_end > zero_run_start`), on the theory that an isolated
            // zero (length-1 run) would already be caught by the
            // S[i]*S[i+1] scan below. It isn't -- that scan is guarded by
            // `!is_zero`, so it never fires AT an exact-zero sample, and
            // the pair straddling one (S[i-1]*S[i] or S[i]*S[i+1]) always
            // multiplies to exactly 0.0, never < 0.0. So an isolated zero
            // was silently dropped from n_zer entirely. Every run counts
            // now, regardless of length.
            ++n_zer;
            in_zero_run = false;
        }
        if (!is_zero && i + 1 < N && S[i] * S[i + 1] < 0.0) ++n_zer;
    }
    sh->n_zer = n_zer;


    int n_max = 0, n_min = 0;
    // BUG FIX: real-extrema counting used to be capped at MBUF here (and
    // in find_extrema_parallel). MBUF = MAX_EXTREMA + 2*NBSYM exists
    // specifically so mirror_one_side has room to add up to NBSYM mirrored
    // points on each side on TOP of the real extrema it's given -- capping
    // real-extrema counting at MBUF instead of MAX_EXTREMA lets a signal
    // with extrema count between MAX_EXTREMA and MBUF eat into that headroom, so
    // mirror_one_side's own writes into these same MBUF-sized arrays then
    // run past the end. Real extrema must stay capped at MAX_EXTREMA so
    // the +2*NBSYM is always free for mirroring.
    for (int i = 1; i < N - 1; ++i) {
        REAL_T dprev = S[i] - S[i - 1];
        REAL_T dnext = S[i + 1] - S[i];
        if (dprev * dnext < 0.0) {
            if (dprev < 0.0) { // local min
                if (n_min < MAX_EXTREMA) sh->ind_min[n_min++] = i;
            } else {            // local max
                if (n_max < MAX_EXTREMA) sh->ind_max[n_max++] = i;
            }
        } else if (dnext == 0.0 && dprev != 0.0) {
            // TRUE left edge of a flat plateau: S[i]==S[i+1] (dnext==0)
            // while the signal was still sloped coming INTO i (dprev!=0).
            //
            // FIX: this branch used to trigger on `dprev==0.0 && dnext!=0.0`
            // -- i.e. at the run's LAST sample, not its first. At that
            // point "d_before" was recomputed as S[i]-S[i-1], which by
            // construction of the (wrong) trigger condition is *always*
            // exactly 0.0 (that's what dprev==0.0 means) -- so the
            // `d_before > 0.0` / `d_before < 0.0` classification below could
            // never be true, and every flat-topped extremum was silently
            // dropped from ind_max/ind_min. Verified against PyEMD's own
            // _find_extrema_simple: it locates run-START indices (via a
            // run-detection over d==0) and classifies using the diff just
            // BEFORE the run began vs. the diff just after it ends -- not
            // the (trivially zero) diff inside the run itself.
            //
            // This is dormant for most real-valued signals (two
            // independent samples essentially never round to bit-identical
            // values, so this branch rarely fires), but it must still be
            // correct when it does.
            int j = i;
            while (j + 1 < N && S[j + 1] == S[i]) ++j;
            if (j + 1 < N) {
                REAL_T d_after = S[j + 1] - S[j];
                int mid = (i + j) / 2;
                if (dprev > 0.0 && d_after < 0.0) {
                    if (n_max < MAX_EXTREMA) sh->ind_max[n_max++] = mid;
                } else if (dprev < 0.0 && d_after > 0.0) {
                    if (n_min < MAX_EXTREMA) sh->ind_min[n_min++] = mid;
                }
            }
            i = j; // skip past the plateau we just consumed
        }
    }
    sh->n_max = n_max;
    sh->n_min = n_min;
    if (n_max >= MAX_EXTREMA || n_min >= MAX_EXTREMA) sh->overflow = 1;
}

#define LOCAL_EXT_BUF 64
#define PCR_LOCAL_BUF 16

// ---------------------------------------------------------------------
// Parallel extrema detection -- same result as find_extrema_serial
// (verified: identical ind_max/ind_min/n_zer on all 2812 raw + sift-
// intermediate signals used in testing so far -- see the Python
// simulation this was checked against before being written here), but
// spreads the O(N) scan across the whole block instead of thread 0
// alone. Each thread scans one contiguous chunk of the signal into a
// small local buffer; a block-wide exclusive prefix sum over per-thread
// counts gives each thread its write offset into sh->ind_max/ind_min,
// so the result comes out index-sorted exactly like the serial version
// without a second, slower serial merge pass.
//
// Falls back to the serial version (correctness-safety net, not a
// performance path) if any chunk contains a flat run (d==0 -- see
// find_extrema_serial's plateau handling, which requires seeing the
// full run length and isn't safe to reconstruct from a chunk in
// isolation) or if a single chunk's extrema count exceeds
// LOCAL_EXT_BUF (64 -- generous for the chunk sizes real signal lengths
// produce with THREADS=128, but a signal that's extrema on literally
// every other sample in one 128th of itself is the kind of input this
// falls back safely on rather than risk a wrong answer for).
// ---------------------------------------------------------------------
__device__ void find_extrema_parallel(
    const REAL_T* T, const REAL_T* S, int N, SharedScratch* sh
) {
    __shared__ int max_counts[THREADS];
    __shared__ int min_counts[THREADS];
    __shared__ int nzer_counts[THREADS];
    __shared__ int need_fallback;

    if (threadIdx.x == 0) need_fallback = 0;
    __syncthreads();

    int chunk_size = (N + blockDim.x - 1) / blockDim.x;
    int start = threadIdx.x * chunk_size;
    int end = min(start + chunk_size, N);
    int lo = max(start, 1), hi = min(end, N - 1);

    int local_max[LOCAL_EXT_BUF], local_min[LOCAL_EXT_BUF];
    int n_local_max = 0, n_local_min = 0, local_nzer = 0;
    bool local_bad = false;

    for (int i = lo; i < hi; ++i) {
        REAL_T d_prev = S[i] - S[i - 1];
        REAL_T d_next = S[i + 1] - S[i];
        // FIX: an isolated exact-zero sample (S[i]==0.0, both neighbors
        // different and nonzero) doesn't set d_prev or d_next to 0 -- it's
        // not a plateau -- so it used to sail through unflagged. That
        // matters here because the plain zero-crossing scan just below
        // (`S[i] * S[i + 1] < 0.0`) is guarded by `S[i] != 0.0` and has NO
        // run-based handling at all (unlike find_extrema_serial), so it
        // silently drops any exact-zero sample from local_nzer. Folding
        // `S[i] == 0.0` into the fallback trigger routes this case to
        // find_extrema_serial's zero-run handling instead of duplicating
        // that logic here.
        if (d_prev == 0.0 || d_next == 0.0 || S[i] == 0.0) local_bad = true;
        if (d_prev * d_next < 0.0) {
            if (d_prev < 0.0) {
                if (n_local_min < LOCAL_EXT_BUF) local_min[n_local_min++] = i; else local_bad = true;
            } else {
                if (n_local_max < LOCAL_EXT_BUF) local_max[n_local_max++] = i; else local_bad = true;
            }
        }
    }
    int z_hi = min(end, N - 1);
    for (int i = start; i < z_hi; ++i) {
        if (S[i] != 0.0 && S[i] * S[i + 1] < 0.0) ++local_nzer;
    }

    if (local_bad) need_fallback = 1;
    max_counts[threadIdx.x] = n_local_max;
    min_counts[threadIdx.x] = n_local_min;
    nzer_counts[threadIdx.x] = local_nzer;
    __syncthreads();

    if (need_fallback) {
        if (threadIdx.x == 0) find_extrema_serial(T, S, N, sh);
        __syncthreads();
        return;
    }

    // Hillis-Steele inclusive scan -> exclusive offset = inclusive - own count
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int add_max = 0, add_min = 0;
        if (threadIdx.x >= offset) { add_max = max_counts[threadIdx.x - offset]; add_min = min_counts[threadIdx.x - offset]; }
        __syncthreads();
        if (threadIdx.x >= offset) { max_counts[threadIdx.x] += add_max; min_counts[threadIdx.x] += add_min; }
        __syncthreads();
    }
    int my_max_offset = max_counts[threadIdx.x] - n_local_max;
    int my_min_offset = min_counts[threadIdx.x] - n_local_min;

    // BUG FIX: these writes used to be unconditional. The serial fallback
    // guards every single write with `if (n_max < MAX_EXTREMA)` so it can
    // never write past ind_max/ind_min (MBUF-sized shared arrays) even
    // when the signal's TRUE extrema count exceeds MAX_EXTREMA -- it just
    // saturates and relies on the overflow flag below. This compaction
    // step had no such guard: my_max_offset+k (or my_min_offset+k) could
    // land past MAX_EXTREMA for any signal whose total extrema count
    // exceeds it, writing past the end of a MBUF-sized __shared__ array --
    // silent corruption of whatever else lives in SharedScratch, or a hard
    // cudaErrorIllegalAddress for a large enough overrun (confirmed: this
    // is what was crashing on long, extrema-heavy random signals). Same
    // bound as the serial path now (MAX_EXTREMA, NOT MBUF -- MBUF's extra
    // +2*NBSYM is mirror_one_side's headroom, not more room for real
    // extrema; capping this at MBUF instead still overflows the same
    // arrays one function later, in mirror_one_side, once it adds its own
    // mirrored points on top), applied per-element here since a single
    // thread's k-loop can straddle the boundary.
    for (int k = 0; k < n_local_max; ++k) {
        int idx = my_max_offset + k;
        if (idx < MAX_EXTREMA) sh->ind_max[idx] = local_max[k];
    }
    for (int k = 0; k < n_local_min; ++k) {
        int idx = my_min_offset + k;
        if (idx < MAX_EXTREMA) sh->ind_min[idx] = local_min[k];
    }
    __syncthreads();

    // sum reduction for the zero-crossing count (order doesn't matter, unlike the compaction above)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) nzer_counts[threadIdx.x] += nzer_counts[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == blockDim.x - 1) {
        int total_max = my_max_offset + n_local_max;   // last thread's offset+count = grand total
        int total_min = my_min_offset + n_local_min;
        if (total_max >= MAX_EXTREMA || total_min >= MAX_EXTREMA) sh->overflow = 1;
        // Clamp to MAX_EXTREMA to match what was actually written above --
        // callers loop up to sh->n_max/n_min to index ind_max/ind_min, so
        // an unclamped (true, possibly much larger) count would let those
        // reads run past MAX_EXTREMA too, same failure mode as the write
        // bug, and a clamp to MBUF instead would re-open the mirroring
        // overflow described above.
        sh->n_max = min(total_max, MAX_EXTREMA);
        sh->n_min = min(total_min, MAX_EXTREMA);
        sh->n_zer = nzer_counts[0];
    }
    __syncthreads();
}

// ---------------------------------------------------------------------
// Boundary mirroring ("simple" mode). Thread 0 only. Direct port of
// EMDReference.prepare_points_simple -- see that file for the branch-by-
// branch rationale; ported here index-for-index, not "cleaned up".
// Writes the final mirrored (position, value) arrays for EITHER the max
// or the min envelope into sh->ext_pos / sh->ext_val (caller picks which
// by calling this twice -- once building the max envelope's knot list,
// once the min envelope's -- so the two calls don't run concurrently).
// ---------------------------------------------------------------------
__device__ int mirror_one_side(
    const REAL_T* T, const REAL_T* S, int N, SharedScratch* sh, bool build_max
) {
    int n_max = sh->n_max, n_min = sh->n_min;
    int* ind_max = sh->ind_max;
    int* ind_min = sh->ind_min;

    int lmax_buf[NBSYM + 1], lmin_buf[NBSYM + 1];
    int rmax_buf[NBSYM + 1], rmin_buf[NBSYM + 1];
    int n_lmax, n_lmin, n_rmax, n_rmin;
    int lsym, rsym;
    // Fallback flags: Python's `if not lmin.size: lmin = ind_min` (etc.)
    // uses the FULL, potentially large ind_max/ind_min array directly, in
    // ASCENDING order, not reversed. An earlier version of this function
    // tried to copy that into the same NBSYM+1-sized stack buffers used
    // by the normal branches -- a real stack buffer overflow for large
    // extrema counts -- and also reversed rmin/rmax when Python doesn't.
    // Fixed by not copying at all: these flags make the build step below
    // read straight from ind_max/ind_min (already sized MBUF in shared
    // memory) instead.
    bool lmin_fb = false, rmin_fb = false, lmax_fb = false, rmax_fb = false;

    // ---- Left bound ----
    if (ind_max[0] < ind_min[0]) {
        if (S[0] > S[ind_min[0]]) {
            n_lmax = 0;
            for (int k = min(n_max, NBSYM + 1) - 1; k >= 1; --k) lmax_buf[n_lmax++] = ind_max[k];
            n_lmin = 0;
            for (int k = min(n_min, NBSYM) - 1; k >= 0; --k) lmin_buf[n_lmin++] = ind_min[k];
            lsym = ind_max[0];
        } else {
            n_lmax = 0;
            for (int k = min(n_max, NBSYM) - 1; k >= 0; --k) lmax_buf[n_lmax++] = ind_max[k];
            n_lmin = 0;
            for (int k = min(n_min, NBSYM - 1) - 1; k >= 0; --k) lmin_buf[n_lmin++] = ind_min[k];
            lmin_buf[n_lmin++] = 0;
            lsym = 0;
        }
    } else {
        if (S[0] < S[ind_max[0]]) {
            n_lmax = 0;
            for (int k = min(n_max, NBSYM) - 1; k >= 0; --k) lmax_buf[n_lmax++] = ind_max[k];
            n_lmin = 0;
            for (int k = min(n_min, NBSYM + 1) - 1; k >= 1; --k) lmin_buf[n_lmin++] = ind_min[k];
            lsym = ind_min[0];
        } else {
            n_lmax = 0;
            for (int k = min(n_max, NBSYM - 1) - 1; k >= 0; --k) lmax_buf[n_lmax++] = ind_max[k];
            lmax_buf[n_lmax++] = 0;
            n_lmin = 0;
            for (int k = min(n_min, NBSYM) - 1; k >= 0; --k) lmin_buf[n_lmin++] = ind_min[k];
            lsym = 0;
        }
    }

    // ---- Right bound ----
    if (ind_max[n_max - 1] < ind_min[n_min - 1]) {
        if (S[N - 1] < S[ind_max[n_max - 1]]) {
            int start = max(n_max - NBSYM, 0);
            n_rmax = 0;
            for (int k = n_max - 1; k >= start; --k) rmax_buf[n_rmax++] = ind_max[k];
            int startm = max(n_min - NBSYM - 1, 0);
            n_rmin = 0;
            for (int k = n_min - 2; k >= startm; --k) rmin_buf[n_rmin++] = ind_min[k];
            rsym = ind_min[n_min - 1];
        } else {
            int start = max(n_max - NBSYM + 1, 0);
            n_rmax = 0;
            rmax_buf[n_rmax++] = N - 1;
            for (int k = start; k < n_max; ++k) rmax_buf[n_rmax++] = ind_max[k];
            int startm = max(n_min - NBSYM, 0);
            n_rmin = 0;
            for (int k = n_min - 1; k >= startm; --k) rmin_buf[n_rmin++] = ind_min[k];
            rsym = N - 1;
        }
    } else {
        if (S[N - 1] > S[ind_min[n_min - 1]]) {
            int start = max(n_max - NBSYM - 1, 0);
            n_rmax = 0;
            for (int k = n_max - 2; k >= start; --k) rmax_buf[n_rmax++] = ind_max[k];
            int startm = max(n_min - NBSYM, 0);
            n_rmin = 0;
            for (int k = n_min - 1; k >= startm; --k) rmin_buf[n_rmin++] = ind_min[k];
            rsym = ind_max[n_max - 1];
        } else {
            int start = max(n_max - NBSYM, 0);
            n_rmax = 0;
            for (int k = n_max - 1; k >= start; --k) rmax_buf[n_rmax++] = ind_max[k];
            int startm = max(n_min - NBSYM + 1, 0);
            n_rmin = 0;
            rmin_buf[n_rmin++] = N - 1;
            for (int k = startm; k < n_min; ++k) rmin_buf[n_rmin++] = ind_min[k];
            rsym = N - 1;
        }
    }

    if (n_lmin == 0) lmin_fb = true;
    if (n_rmin == 0) rmin_fb = true;
    if (n_lmax == 0) lmax_fb = true;
    if (n_rmax == 0) rmax_fb = true;

    // "Mirrored point doesn't reach the signal boundary" fallback: if the
    // first left-mirrored position is still > T[0] (or right-mirrored
    // still < T[N-1]), PyEMD re-mirrors that side against the signal's
    // actual edge (lsym=0 / rsym=N-1) instead of the extremum it picked
    // initially. Previously unimplemented in this kernel -- confirmed via
    // a direct comparison against numpy_emd_reference.py on a real
    // low-extrema-count residue (this is common on later IMFs, which tend
    // to have very few extrema, not just an "unusual edge case").
    int lmin_first = lmin_fb ? ind_min[0] : lmin_buf[0];
    int lmax_first = lmax_fb ? ind_max[0] : lmax_buf[0];
    REAL_T tlmin0 = 2.0 * T[lsym] - T[lmin_first];
    REAL_T tlmax0 = 2.0 * T[lsym] - T[lmax_first];
    if (tlmin0 > T[0] || tlmax0 > T[0]) {
        if (lsym == ind_max[0]) {
            n_lmax = 0;
            for (int k = min(n_max, NBSYM) - 1; k >= 0; --k) lmax_buf[n_lmax++] = ind_max[k];
            lmax_fb = false;
        } else {
            n_lmin = 0;
            for (int k = min(n_min, NBSYM) - 1; k >= 0; --k) lmin_buf[n_lmin++] = ind_min[k];
            lmin_fb = false;
        }
        lsym = 0;
    }

    int rmin_last = rmin_fb ? ind_min[n_min - 1] : rmin_buf[n_rmin - 1];
    int rmax_last = rmax_fb ? ind_max[n_max - 1] : rmax_buf[n_rmax - 1];
    REAL_T trmin_last = 2.0 * T[rsym] - T[rmin_last];
    REAL_T trmax_last = 2.0 * T[rsym] - T[rmax_last];
    if (trmin_last < T[N - 1] || trmax_last < T[N - 1]) {
        if (rsym == ind_max[n_max - 1]) {
            int start = max(n_max - NBSYM, 0);
            n_rmax = 0;
            for (int k = n_max - 1; k >= start; --k) rmax_buf[n_rmax++] = ind_max[k];
            rmax_fb = false;
        } else {
            int startm = max(n_min - NBSYM, 0);
            n_rmin = 0;
            for (int k = n_min - 1; k >= startm; --k) rmin_buf[n_rmin++] = ind_min[k];
            rmin_fb = false;
        }
        rsym = N - 1;
    }

    // Build the requested side's full (position, value) knot arrays:
    // [mirrored-left, true extrema, mirrored-right], then de-duplicate
    // adjacent knots with equal position (mirrors the `max_dup_idx` step).
    int n_out = 0;
    int* ind = build_max ? ind_max : ind_min;
    int n_true = build_max ? n_max : n_min;
    int* lbuf = build_max ? lmax_buf : lmin_buf;
    int n_l = build_max ? n_lmax : n_lmin;
    int* rbuf = build_max ? rmax_buf : rmin_buf;
    int n_r = build_max ? n_rmax : n_rmin;
    bool l_fb = build_max ? lmax_fb : lmin_fb;
    bool r_fb = build_max ? rmax_fb : rmin_fb;
    int* full = build_max ? ind_max : ind_min;
    int n_full = build_max ? n_max : n_min;
    int sym_l = lsym, sym_r = rsym;

    REAL_T last_pos = -1e30;
    bool have_last = false;
    if (l_fb) {
        // Python: `lmin = ind_min` (or lmax = ind_max) used AS-IS, ascending
        // index order, no reversal -- faithfully replicated here rather
        // than "corrected", even though it's an unusual-looking branch.
        for (int k = 0; k < n_full; ++k) {
            REAL_T pos = 2.0 * T[sym_l] - T[full[k]];
            REAL_T val = S[full[k]];
            if (!have_last || pos != last_pos) { sh->ext_pos[n_out] = pos; sh->ext_val[n_out] = val; ++n_out; last_pos = pos; have_last = true; }
        }
    } else {
        for (int k = 0; k < n_l; ++k) {
            REAL_T pos = 2.0 * T[sym_l] - T[lbuf[k]];
            REAL_T val = S[lbuf[k]];
            if (!have_last || pos != last_pos) { sh->ext_pos[n_out] = pos; sh->ext_val[n_out] = val; ++n_out; last_pos = pos; have_last = true; }
        }
    }
    for (int k = 0; k < n_true; ++k) {
        REAL_T pos = T[ind[k]];
        REAL_T val = S[ind[k]];
        if (!have_last || pos != last_pos) { sh->ext_pos[n_out] = pos; sh->ext_val[n_out] = val; ++n_out; last_pos = pos; have_last = true; }
    }
    if (r_fb) {
        for (int k = 0; k < n_full; ++k) {
            REAL_T pos = 2.0 * T[sym_r] - T[full[k]];
            REAL_T val = S[full[k]];
            if (!have_last || pos != last_pos) { sh->ext_pos[n_out] = pos; sh->ext_val[n_out] = val; ++n_out; last_pos = pos; have_last = true; }
        }
    } else {
        for (int k = 0; k < n_r; ++k) {
            REAL_T pos = 2.0 * T[sym_r] - T[rbuf[k]];
            REAL_T val = S[rbuf[k]];
            if (!have_last || pos != last_pos) { sh->ext_pos[n_out] = pos; sh->ext_val[n_out] = val; ++n_out; last_pos = pos; have_last = true; }
        }
    }
    return n_out;
}

// ---------------------------------------------------------------------
// PyEMD's actual spline: NOT natural BCs -- not-a-knot. Confirmed via
// benchmark/diagnose_spline.py (matched PyEMD's own output to 0.000e+00;
// an earlier version of this kernel used natural BCs, which was off by
// three orders of magnitude -- see the debugging log in README.md).
//
// n=2: linear (unchanged from before -- this case never involved a BC
//      choice to begin with).
// n=3: not-a-knot is technically underdetermined at exactly 3 points;
//      empirically (checked against scipy in this repo's dev history)
//      it reduces to the unique quadratic through the 3 points, evaluated
//      directly via Lagrange's formula -- no thread-0 setup needed, every
//      thread can compute its own point independently.
// n>=4: not-a-knot's boundary conditions couple M_0 with M_1,M_2 (and
//      symmetrically at the right edge), which breaks strict tridiagonal
//      structure. Standard fix: algebraically eliminate M_0 and M_{n-1}
//      using the not-a-knot condition, substitute into the i=1 and
//      i=n-2 interior equations, solve the resulting (n-2)-sized
//      tridiagonal system for M_1..M_{n-2}, then recover M_0/M_{n-1}
//      afterwards. This exact algorithm (not just the general idea) was
//      checked against scipy's CubicSpline(bc_type='not-a-knot') across
//      200 random trials before being translated here -- see
//      reference/verify_notaknot_elimination.py -- rather than trusting
//      a from-scratch-in-CUDA derivation a second time.
// ---------------------------------------------------------------------
__device__ void notaknot_cubic_spline_block(
    const REAL_T* T, int N, SharedScratch* sh, int n, REAL_T* out
) {
    REAL_T* x = sh->ext_pos;
    REAL_T* y = sh->ext_val;
    // Parallel cyclic reduction (PCR) scratch for the reduced (n-2)-sized
    // tridiagonal system -- replaces a previous version that solved this
    // with a plain (thread-0-only) Thomas algorithm, which was O(n)
    // *sequential* steps with the other THREADS-1 threads idle -- for
    // n in the many hundreds (real noisy signals in testing hit ~700),
    // that dwarfed everything else in the sift loop. PCR takes ceil(log2(m))
    // *parallel* steps instead (~10 for m~700), each doing O(m) work split
    // across the block. Verified against scipy's not-a-knot spline output
    // at this same scale (500-800 knots) and at small scale (4-29 knots)
    // before being translated here -- see this file's development history
    // for the verification script; not shipped in this repo since it's a
    // few lines of throwaway numerical testing, not library code.
    REAL_T* b_arr = sh->cp;   // PCR diagonal
    REAL_T* d_arr = sh->rhs;  // PCR RHS -> ends up holding M[1..n-2] (then M[0],M[n-1] recovered after)
    REAL_T* a_arr = sh->pcr_a; // PCR sub-diagonal
    REAL_T* c_arr = sh->pcr_c; // PCR super-diagonal

    if (n >= 4) {
        int m = n - 2;

        // ---- Parallel setup of the reduced system (no cross-thread
        // dependencies here -- each row's formula only reads x[]/y[], not
        // other rows -- unlike the solve step below, which does). ----
        for (int k = threadIdx.x; k < m; k += blockDim.x) {
            int i = k + 1; // this reduced row's index in the ORIGINAL (n-sized) system
            if (k == 0) {
                REAL_T h0 = x[1] - x[0], h1 = x[2] - x[1];
                REAL_T rhs1 = 6.0 * ((y[2] - y[1]) / h1 - (y[1] - y[0]) / h0);
                a_arr[k] = 0.0;
                b_arr[k] = h0 + 2.0 * h1;
                c_arr[k] = h1 - h0;
                d_arr[k] = h1 * rhs1 / (h0 + h1);
            } else if (k == m - 1) {
                REAL_T hnm3 = x[n - 2] - x[n - 3], hnm2 = x[n - 1] - x[n - 2];
                REAL_T rhs_nm2 = 6.0 * ((y[n - 1] - y[n - 2]) / hnm2 - (y[n - 2] - y[n - 3]) / hnm3);
                a_arr[k] = hnm3 - hnm2;
                b_arr[k] = 2.0 * hnm3 + hnm2;
                c_arr[k] = 0.0;
                d_arr[k] = hnm3 * rhs_nm2 / (hnm3 + hnm2);
            } else {
                REAL_T h_im1 = x[i] - x[i - 1], h_i = x[i + 1] - x[i];
                a_arr[k] = h_im1;
                b_arr[k] = 2.0 * (h_im1 + h_i);
                c_arr[k] = h_i;
                d_arr[k] = 6.0 * ((y[i + 1] - y[i]) / h_i - (y[i] - y[i - 1]) / h_im1);
            }
        }
        __syncthreads();

        // ---- Parallel cyclic reduction ----
        // Every thread stages ITS new (a,b,c,d) values for ITS assigned
        // k's into local registers first, reading only the OLD (not-yet-
        // updated) shared arrays; only after a block-wide sync (meaning
        // every thread has finished reading) does anyone write the new
        // values back in place. That ordering -- all reads before any
        // write -- is what makes in-place update safe here without a
        // second (ping-pong) copy of the arrays, same principle as the
        // Hillis-Steele scan in find_extrema_parallel.
        for (int s = 1; s < m; s *= 2) {
            REAL_T na[PCR_LOCAL_BUF], nb[PCR_LOCAL_BUF], nc[PCR_LOCAL_BUF], nd[PCR_LOCAL_BUF];
            int cnt = 0;
            for (int k = threadIdx.x; k < m; k += blockDim.x) {
                bool has_left = (k - s >= 0), has_right = (k + s < m);
                REAL_T b_left = has_left ? b_arr[k - s] : 1.0;
                REAL_T b_right = has_right ? b_arr[k + s] : 1.0;
                REAL_T alpha = has_left ? a_arr[k] / b_left : 0.0;
                REAL_T beta = has_right ? c_arr[k] / b_right : 0.0;
                REAL_T a_left = has_left ? a_arr[k - s] : 0.0;
                REAL_T c_left = has_left ? c_arr[k - s] : 0.0;
                REAL_T d_left = has_left ? d_arr[k - s] : 0.0;
                REAL_T a_right = has_right ? a_arr[k + s] : 0.0;
                REAL_T c_right = has_right ? c_arr[k + s] : 0.0;
                REAL_T d_right = has_right ? d_arr[k + s] : 0.0;

                na[cnt] = -alpha * a_left;
                nc[cnt] = -beta * c_right;
                nb[cnt] = b_arr[k] - alpha * c_left - beta * a_right;
                nd[cnt] = d_arr[k] - alpha * d_left - beta * d_right;
                ++cnt;
            }
            __syncthreads();
            cnt = 0;
            for (int k = threadIdx.x; k < m; k += blockDim.x) {
                a_arr[k] = na[cnt]; b_arr[k] = nb[cnt]; c_arr[k] = nc[cnt]; d_arr[k] = nd[cnt];
                ++cnt;
            }
            __syncthreads();
        }

        // System is now fully diagonal: x_k = d_k / b_k. Stage into local
        // registers before writing into sh->rhs (same array as d_arr) at
        // the SHIFTED position (M_mid[k] = M[k+1]) -- same read-all-then-
        // write-all safety reasoning as above.
        REAL_T solved[PCR_LOCAL_BUF];
        int cnt = 0;
        for (int k = threadIdx.x; k < m; k += blockDim.x) { solved[cnt++] = d_arr[k] / b_arr[k]; }
        __syncthreads();
        cnt = 0;
        for (int k = threadIdx.x; k < m; k += blockDim.x) { sh->rhs[k + 1] = solved[cnt++]; }
        __syncthreads();

        if (threadIdx.x == 0) {
            REAL_T h0 = x[1] - x[0], h1 = x[2] - x[1];
            REAL_T hnm3 = x[n - 2] - x[n - 3], hnm2 = x[n - 1] - x[n - 2];
            sh->rhs[0] = ((h0 + h1) * sh->rhs[1] - h0 * sh->rhs[2]) / h1;                    // recover M_0
            sh->rhs[n - 1] = ((hnm3 + hnm2) * sh->rhs[n - 2] - hnm2 * sh->rhs[n - 3]) / hnm3; // recover M_{n-1}
        }
    }
    REAL_T* rhs = sh->rhs; // M[0..n-1] for the evaluation loop below, whichever branch (n=2/3/>=4) filled it
    __syncthreads();

    // Parallel evaluation across all N samples in [x[0], x[n-1]].
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        REAL_T t = T[i];
        if (t < x[0] || t > x[n - 1]) continue; // matches PyEMD's t-range mask

        if (n == 2) {
            REAL_T slope = (y[1] - y[0]) / (x[1] - x[0]);
            out[i] = y[0] + slope * (t - x[0]);
            continue;
        }
        if (n == 3) {
            // unique quadratic through 3 points, Lagrange form -- no
            // shared setup needed, fully independent per thread.
            REAL_T l0 = (t - x[1]) * (t - x[2]) / ((x[0] - x[1]) * (x[0] - x[2]));
            REAL_T l1 = (t - x[0]) * (t - x[2]) / ((x[1] - x[0]) * (x[1] - x[2]));
            REAL_T l2 = (t - x[0]) * (t - x[1]) / ((x[2] - x[0]) * (x[2] - x[1]));
            out[i] = y[0] * l0 + y[1] * l1 + y[2] * l2;
            continue;
        }
        // n >= 4: standard piecewise-cubic evaluation from M[] (rhs[]),
        // same formula as before -- only how M[] got built changed.
        int lo = 0, hi = n - 2;
        while (lo < hi) {
            int mid = (lo + hi + 1) / 2;
            if (x[mid] <= t) lo = mid; else hi = mid - 1;
        }
        REAL_T x0 = x[lo], x1 = x[lo + 1];
        REAL_T y0 = y[lo], y1 = y[lo + 1];
        REAL_T M0 = rhs[lo], M1 = rhs[lo + 1];
        REAL_T h = x1 - x0;
        REAL_T A = (x1 - t) / h;
        REAL_T B = (t - x0) / h;
        out[i] = A * y0 + B * y1 + ((A * A * A - A) * M0 + (B * B * B - B) * M1) * (h * h) / 6.0;
    }
    __syncthreads();
}

// ---------------------------------------------------------------------
// check_imf: Huang/Cauchy-style convergence test, 3 alternative passes.
// Block-cooperative reduction (all threads help sum), thread 0 decides.
// ---------------------------------------------------------------------
__device__ bool check_imf_block(
    const REAL_T* imf_new, const REAL_T* imf_old, int N,
    REAL_T eMax_min, REAL_T eMin_max, // min value across max-envelope knots, max value across min-envelope knots (sign check)
    bool any_bad_sign, SharedScratch* sh
) {
    if (any_bad_sign) return false;

    // Fused reduction: an earlier version did 6 separate tree-reductions
    // back to back (sum_new_sq as an early-exit check, then diff_sq_sum,
    // old_sq_sum, std_sum, old_max, old_min), each with its own full
    // __syncthreads()-guarded pass -- ~6x more synchronization than
    // necessary, and this runs every single sift iteration for every
    // signal. All 6 quantities come from per-sample data over the same
    // index range, so one thread can accumulate all 6 locally in one
    // loop and they can all be tree-reduced together in one pass.
    //
    // Reused for scratch: sh->ext_pos/ext_val/cp/rhs/pcr_a/pcr_c (their
    // first THREADS slots). Safe to repurpose here specifically because
    // by this point in the sift loop both mirror_one_side +
    // notaknot_cubic_spline_block calls (max side, then min side) have
    // already finished with them -- their spline-related contents are
    // fully consumed (written out to max_env[]/min_env[], which live in
    // separate pool arrays) and nothing reads them again until next
    // iteration's mirror_one_side call overwrites them fresh. That
    // avoids adding new static __shared__ arrays, which would have
    // reopened the shared-memory-budget problem from earlier.
    REAL_T* red_sq = sh->ext_pos;
    REAL_T* red_diff = sh->ext_val;
    REAL_T* red_oldsq = sh->cp;
    REAL_T* red_std = sh->rhs;
    REAL_T* red_max = sh->pcr_a;
    REAL_T* red_min = sh->pcr_c;

    REAL_T local_sq = 0.0, local_diff_sq = 0.0, local_old_sq = 0.0, local_std = 0.0;
    REAL_T local_old_max = -1e30, local_old_min = 1e30;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        REAL_T diff = imf_new[i] - imf_old[i];
        local_sq += imf_new[i] * imf_new[i];
        local_diff_sq += diff * diff;
        local_old_sq += imf_old[i] * imf_old[i];
        local_std += (diff / imf_new[i]) * (diff / imf_new[i]);
        local_old_max = max(local_old_max, imf_old[i]);
        local_old_min = min(local_old_min, imf_old[i]);
    }
    red_sq[threadIdx.x] = local_sq;
    red_diff[threadIdx.x] = local_diff_sq;
    red_oldsq[threadIdx.x] = local_old_sq;
    red_std[threadIdx.x] = local_std;
    red_max[threadIdx.x] = local_old_max;
    red_min[threadIdx.x] = local_old_min;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red_sq[threadIdx.x] += red_sq[threadIdx.x + s];
            red_diff[threadIdx.x] += red_diff[threadIdx.x + s];
            red_oldsq[threadIdx.x] += red_oldsq[threadIdx.x + s];
            red_std[threadIdx.x] += red_std[threadIdx.x + s];
            red_max[threadIdx.x] = max(red_max[threadIdx.x], red_max[threadIdx.x + s]);
            red_min[threadIdx.x] = min(red_min[threadIdx.x], red_min[threadIdx.x + s]);
        }
        __syncthreads();
    }

    REAL_T sum_new_sq = red_sq[0];
    REAL_T diff_sq_sum = red_diff[0];
    REAL_T old_sq_sum = red_oldsq[0];
    REAL_T std_sum = red_std[0];
    REAL_T old_max_sh = red_max[0];
    REAL_T old_min_sh = red_min[0];
    __syncthreads(); // all threads must finish reading before this scratch gets reused next iteration

    if (sum_new_sq < 1e-10) return false;
    REAL_T svar = diff_sq_sum / (old_max_sh - old_min_sh);
    if (svar < svar_thr()) return true;
    if (std_sum < std_thr()) return true;
    REAL_T energy_ratio = diff_sq_sum / old_sq_sum;
    if (energy_ratio < energy_ratio_thr()) return true;
    return false;
}

// =====================================================================
// Main kernel: one block per signal.
// signals:  [batch, N] row-major, float64
// T:        [N] shared time axis (assumed identical across the batch --
//           the common case for a fixed sample rate; per-signal T is a
//           straightforward extension, not needed yet)
// imfs_out: [batch, MAX_IMFS, N] preallocated output
// n_imfs_out: [batch] how many of the MAX_IMFS slots are valid (incl. residue)
// overflow_out: [batch] 1 if MAX_EXTREMA or MAX_ITERATION was hit -- treat
//           that signal's output as unverified, not silently "close enough"
// =====================================================================
__global__ void emd_batch_kernel(
    const REAL_T* __restrict__ signals,
    const REAL_T* __restrict__ T,
    int N,
    REAL_T* __restrict__ imfs_out,
    int* __restrict__ n_imfs_out,
    int* __restrict__ overflow_out,
    // N-length working pools, one full [batch, N] array per pool, allocated
    // host-side by EMDBatchLauncher.run(). Deliberately passed as ordinary
    // kernel arguments (NOT extern __device__ globals -- device-scope
    // extern declarations belong at file scope, not inside a __global__
    // function body, and RawModule.get_global() gives you a fixed symbol's
    // memory, not a way to repoint it per-call; an earlier draft of this
    // file got that wrong, this is the fixed version).
    REAL_T* __restrict__ residue_pool,
    REAL_T* __restrict__ imf_pool,
    REAL_T* __restrict__ imf_old_pool,
    REAL_T* __restrict__ mean_pool,
    REAL_T* __restrict__ max_env_pool,
    REAL_T* __restrict__ min_env_pool,
    // DIAGNOSTIC ONLY -- per-IMF (iteration count, extNo, nzm) at the
    // point each IMF's sift converged, so a single mismatching signal's
    // divergence point can be pinpointed without dumping full per-sample
    // state. [batch, MAX_IMFS], -1 where an IMF slot wasn't reached.
    int* __restrict__ debug_n_out,
    int* __restrict__ debug_extno_out,
    int* __restrict__ debug_nzm_out
) {
    extern __shared__ unsigned char raw_smem[];
    SharedScratch* sh = reinterpret_cast<SharedScratch*>(raw_smem);

    int b = blockIdx.x;
    const REAL_T* S = signals + (size_t)b * N;
    REAL_T* out_base = imfs_out + (size_t)b * MAX_IMFS * N;

    // These N-length working buffers live in GLOBAL memory (one slice of
    // the pools above per block) rather than shared memory, per the
    // design note at the top of this file: shared memory is reserved for
    // the small, extrema-count-sized stuff, not O(N) per-sample arrays.
    REAL_T* residue  = residue_pool  + (size_t)b * N;
    REAL_T* imf       = imf_pool      + (size_t)b * N;
    REAL_T* imf_old   = imf_old_pool  + (size_t)b * N;
    REAL_T* mean_buf  = mean_pool     + (size_t)b * N;
    REAL_T* max_env    = max_env_pool  + (size_t)b * N;
    REAL_T* min_env    = min_env_pool  + (size_t)b * N;
    if (threadIdx.x == 0) sh->overflow = 0;
    __syncthreads();

    // Running residue, updated incrementally as each IMF is found (see
    // below) instead of re-summing every previous IMF from scratch each
    // outer iteration -- was O(imf_count^2) total work across a
    // decomposition, now O(imf_count). imf_count is typically small
    // (4-8 in testing) so this was a modest win, not the main one.
    for (int i = threadIdx.x; i < N; i += blockDim.x) residue[i] = S[i];
    __syncthreads();

    int imf_count = 0;
    bool finished = false;
    int extNo_last = -1;

    while (!finished && imf_count < MAX_IMFS - 1) {
        for (int i = threadIdx.x; i < N; i += blockDim.x) imf[i] = residue[i];
        __syncthreads();

        int n = 0;
        bool have_imf_old = false;

        while (true) {
            ++n;
            if (n >= MAX_ITERATION) {
                // FIX: this path never set the flag before now, despite
                // this kernel's own docstring promising overflow_out is
                // "1 if MAX_EXTREMA or MAX_ITERATION was hit" -- only the
                // MAX_EXTREMA path (in find_extrema_serial/_parallel) did.
                // Any signal that was silently hitting the iteration cap
                // read as overflow=False, which is misleading when using
                // that flag to rule iteration-cap issues in or out.
                //
                // ALSO recording debug_n_out here now (= MAX_ITERATION,
                // distinguishable from the -1 "slot never reached"
                // sentinel): the debug write below at `if (f1 && f2)` only
                // ever fired on normal convergence, so every IMF that was
                // instead forced to accept via this path left its debug
                // slot at -1 -- indistinguishable from "this IMF slot was
                // never processed at all", when it very much was.
                if (threadIdx.x == 0) {
                    sh->overflow = 1;
                    if (imf_count < MAX_IMFS) {
                        debug_n_out[b * MAX_IMFS + imf_count] = n;
                        debug_extno_out[b * MAX_IMFS + imf_count] = -1;
                        debug_nzm_out[b * MAX_IMFS + imf_count] = -1;
                    }
                }
                break;
            }

            find_extrema_parallel(T, imf, N, sh);
            __syncthreads();
            int extNo = sh->n_max + sh->n_min;

            if (extNo > 2 && sh->n_max >= 1 && sh->n_min >= 1) {
                int n_max_mirrored = 0, n_min_mirrored = 0;
                if (threadIdx.x == 0) {
                    n_max_mirrored = mirror_one_side(T, imf, N, sh, true);
                    sh->n_ext_mirrored_max = n_max_mirrored;
                    sh->bad_envelope = 0;
                    for (int k = 0; k < n_max_mirrored; ++k) if (sh->ext_val[k] < 0.0) sh->bad_envelope = 1;
                }
                __syncthreads();
                notaknot_cubic_spline_block(T, N, sh, sh->n_ext_mirrored_max, max_env);

                if (threadIdx.x == 0) {
                    n_min_mirrored = mirror_one_side(T, imf, N, sh, false);
                    sh->n_ext_mirrored_min = n_min_mirrored;
                    // ext_val now holds the MIN envelope's knot values (ext_pos/ext_val
                    // were reused -- see module docstring); check its sign condition
                    // before this buffer is touched again next iteration.
                    for (int k = 0; k < n_min_mirrored; ++k) if (sh->ext_val[k] > 0.0) sh->bad_envelope = 1;
                }
                __syncthreads();
                notaknot_cubic_spline_block(T, N, sh, sh->n_ext_mirrored_min, min_env);
                bool any_bad_max = (sh->bad_envelope != 0);

                for (int i = threadIdx.x; i < N; i += blockDim.x) {
                    mean_buf[i] = 0.5 * (max_env[i] + min_env[i]);
                    imf_old[i] = imf[i];
                    imf[i] = imf[i] - mean_buf[i];
                }
                __syncthreads();
                have_imf_old = true;

                find_extrema_parallel(T, imf, N, sh);
                __syncthreads();
                int extNo2 = sh->n_max + sh->n_min;
                int nzm2 = sh->n_zer;

                bool f2 = (abs(extNo2 - nzm2) < 2);
                bool f1 = check_imf_block(imf, imf_old, N, 0.0, 0.0, any_bad_max, sh);
                extNo_last = extNo2;
                if (f1 && f2) {
                    if (threadIdx.x == 0 && imf_count < MAX_IMFS) {
                        debug_n_out[b * MAX_IMFS + imf_count] = n;
                        debug_extno_out[b * MAX_IMFS + imf_count] = extNo2;
                        debug_nzm_out[b * MAX_IMFS + imf_count] = nzm2;
                    }
                    break;
                }
            } else {
                // extNo (computed at the top of this inner-loop iteration,
                // BEFORE the mean-subtraction branch above) is <= 2, i.e.
                // this is a trend, not an IMF -- matches the Python
                // reference's `else: finished = True` path. Capture that
                // same extNo here so the post-loop "was the last attempt
                // actually a trend" check below sees the right value.
                extNo_last = extNo;
                finished = true;
                break;
            }
        }

        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            out_base[imf_count * N + i] = imf[i];
            residue[i] -= imf[i];  // running update: residue now reflects this IMF removed
        }
        __syncthreads();
        ++imf_count;

        // end_condition: check residual range / L1 sum against thresholds.
        // residue[] already holds S - sum(IMFs found so far) from the
        // running update above -- no resum needed. Same fusion as
        // check_imf_block: one accumulation loop + one combined tree-
        // reduction instead of 3 sequential ones (max, then min, then
        // sum), reusing sh->ext_pos/ext_val/cp as scratch -- dead at this
        // point (last written by either check_imf_block's own reduction
        // or an earlier mirror_one_side/spline call, nothing after either
        // path needs their contents until the next outer iteration's
        // mirror_one_side call overwrites them fresh) rather than adding
        // a dedicated static array.
        REAL_T* red_max = sh->ext_pos;
        REAL_T* red_min = sh->ext_val;
        REAL_T* red_abs = sh->cp;
        REAL_T local_min = 1e30, local_max = -1e30, local_abs_sum = 0.0;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            REAL_T acc = residue[i];
            local_min = min(local_min, acc);
            local_max = max(local_max, acc);
            local_abs_sum += fabs(acc);
        }
        red_max[threadIdx.x] = local_max;
        red_min[threadIdx.x] = local_min;
        red_abs[threadIdx.x] = local_abs_sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                red_max[threadIdx.x] = max(red_max[threadIdx.x], red_max[threadIdx.x + s]);
                red_min[threadIdx.x] = min(red_min[threadIdx.x], red_min[threadIdx.x + s]);
                red_abs[threadIdx.x] += red_abs[threadIdx.x + s];
            }
            __syncthreads();
        }
        REAL_T gmax = red_max[0], gmin = red_min[0], gabs = red_abs[0];
        __syncthreads();

        if ((gmax - gmin) < range_thr() || gabs < total_power_thr()) finished = true;
    }

    if (extNo_last <= 2 && imf_count > 0) {
        imf_count -= 1; // last "imf" was actually the trend
        // residue[] had that trend "imf" subtracted when it was finalized
        // above; undo it since it's not counted as a real IMF anymore --
        // `imf[]` still holds it, nothing has overwritten that buffer since.
        for (int i = threadIdx.x; i < N; i += blockDim.x) residue[i] += imf[i];
        __syncthreads();
    }

    // BUG FIX: this used to write+count the residue slot unconditionally.
    // PyEMD only appends the residue as a final row `if not
    // np.allclose(self.residue, 0)` -- against a zero array, rtol's
    // contribution vanishes, so that's exactly max(|residue|) <= atol
    // (1e-8, numpy's default). A clean/noise-free signal (e.g. a single
    // pure oscillation, no separate trend) can sift down to a residue
    // that small, and PyEMD's row count reflects that by NOT appending
    // it -- this kernel always did, silently reporting one IMF too many
    // whenever that happened (confirmed: exactly what was happening on
    // low-noise adversarial test signals -- overflow=False, so nothing
    // flagged it, just a wrong IMF count).
    REAL_T* red_absmax = sh->ext_pos; // dead scratch here, same reuse pattern as above
    REAL_T local_absmax = 0.0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) local_absmax = max(local_absmax, fabs(residue[i]));
    red_absmax[threadIdx.x] = local_absmax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red_absmax[threadIdx.x] = max(red_absmax[threadIdx.x], red_absmax[threadIdx.x + s]);
        __syncthreads();
    }
    bool residue_is_zero = (red_absmax[0] <= 1e-8);

    // final residue as the last output slot, unless it's ~0
    if (!residue_is_zero) {
        for (int i = threadIdx.x; i < N; i += blockDim.x) out_base[imf_count * N + i] = residue[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        n_imfs_out[b] = residue_is_zero ? imf_count : imf_count + 1;
        overflow_out[b] = sh->overflow;
    }
}

} // extern "C"


// =====================================================================
// DIAGNOSTIC KERNEL -- not used by GPUBatchEMD, only by
// benchmark/diagnose_cuda_envelope.py. Runs find_extrema_serial +
// mirror_one_side + notaknot_cubic_spline_block -- the exact same
// device functions the main sift loop uses -- on a batch of RAW input
// signals with NO sifting loop around them, so this shared code (used by
// every single sift iteration in emd_batch_kernel) can be checked in
// isolation against numpy_emd_reference.py's extract_max_min_spline,
// the same way diagnose_spline.py isolated the spline boundary condition
// earlier. If this kernel matches the numpy reference and the full
// emd_batch_kernel still doesn't match PyEMD, the bug is specifically in
// the sift-loop/convergence-test logic, not in extrema/mirroring/spline;
// if THIS kernel already mismatches, the bug is here, and the sift loop
// is innocent.
// =====================================================================
extern "C" __global__ void envelope_batch_kernel(
    const REAL_T* __restrict__ signals,
    const REAL_T* __restrict__ T,
    int N,
    REAL_T* __restrict__ max_env_out,
    REAL_T* __restrict__ min_env_out,
    int* __restrict__ n_extrema_out,      // [batch, 2] -- (n_max, n_min) raw counts, pre-mirror
    int* __restrict__ degenerate_out,     // [batch] -- 1 if extNo<=2
    REAL_T* __restrict__ max_knot_pos_out, // [batch, MBUF] -- mirrored max-envelope knot positions, padded
    REAL_T* __restrict__ max_knot_val_out, // [batch, MBUF]
    REAL_T* __restrict__ min_knot_pos_out, // [batch, MBUF] -- mirrored min-envelope knot positions, padded
    REAL_T* __restrict__ min_knot_val_out, // [batch, MBUF]
    int* __restrict__ n_mirrored_out       // [batch, 2] -- (n_mirrored_max, n_mirrored_min)
) {
    extern __shared__ unsigned char raw_smem[];
    SharedScratch* sh = reinterpret_cast<SharedScratch*>(raw_smem);

    int b = blockIdx.x;
    const REAL_T* S = signals + (size_t)b * N;
    REAL_T* max_env = max_env_out + (size_t)b * N;
    REAL_T* min_env = min_env_out + (size_t)b * N;
    REAL_T* max_kpos = max_knot_pos_out + (size_t)b * MBUF;
    REAL_T* max_kval = max_knot_val_out + (size_t)b * MBUF;
    REAL_T* min_kpos = min_knot_pos_out + (size_t)b * MBUF;
    REAL_T* min_kval = min_knot_val_out + (size_t)b * MBUF;

    if (threadIdx.x == 0) sh->overflow = 0;
    __syncthreads();

    if (threadIdx.x == 0) find_extrema_serial(T, S, N, sh);
    __syncthreads();
    int extNo = sh->n_max + sh->n_min;

    if (extNo <= 2 || sh->n_max < 1 || sh->n_min < 1) {
        if (threadIdx.x == 0) {
            degenerate_out[b] = 1;
            n_extrema_out[2 * b] = sh->n_max;
            n_extrema_out[2 * b + 1] = sh->n_min;
        }
        return;
    }

    int n_max_mirrored = 0, n_min_mirrored = 0;
    if (threadIdx.x == 0) {
        n_max_mirrored = mirror_one_side(T, S, N, sh, true);
        sh->n_ext_mirrored_max = n_max_mirrored;
        for (int k = 0; k < n_max_mirrored; ++k) { max_kpos[k] = sh->ext_pos[k]; max_kval[k] = sh->ext_val[k]; }
    }
    __syncthreads();
    notaknot_cubic_spline_block(T, N, sh, sh->n_ext_mirrored_max, max_env);

    if (threadIdx.x == 0) {
        n_min_mirrored = mirror_one_side(T, S, N, sh, false);
        sh->n_ext_mirrored_min = n_min_mirrored;
        for (int k = 0; k < n_min_mirrored; ++k) { min_kpos[k] = sh->ext_pos[k]; min_kval[k] = sh->ext_val[k]; }
    }
    __syncthreads();
    notaknot_cubic_spline_block(T, N, sh, sh->n_ext_mirrored_min, min_env);

    if (threadIdx.x == 0) {
        degenerate_out[b] = 0;
        n_extrema_out[2 * b] = sh->n_max;
        n_extrema_out[2 * b + 1] = sh->n_min;
        n_mirrored_out[2 * b] = n_max_mirrored;
        n_mirrored_out[2 * b + 1] = n_min_mirrored;
    }
}
"""


@dataclass
class EMDKernelConfig:
    max_extrema: int = 1024
    nbsym: int = 2
    max_iteration: int = 1000
    max_imfs: int = 16
    threads_per_block: int = 128

    @property
    def dtype(self) -> "cp.dtype":
        return cp.dtype(cp.float64)

    @property
    def real_t_name(self) -> str:
        """The CUDA type name REAL_T gets #defined to."""
        return "double"

    @property
    def itemsize(self) -> int:
        return self.dtype.itemsize  # always 8 (float64)

    def static_shared_mem_bytes(self) -> int:
        """Bytes used by the kernel's STATIC __shared__ declarations:
        max_counts/min_counts/nzer_counts (int[T] each) + need_fallback
        (int) in find_extrema_parallel.

        Separate from (and additional to) the dynamic `extern __shared__`
        struct that shared_mem_bytes() sizes -- CUDA's per-block shared
        memory ceiling applies to static + dynamic COMBINED, not either
        alone. Recompute this if the kernel source's __shared__
        declarations ever change -- it's not derived from the source
        automatically."""
        T = self.threads_per_block
        return 3 * 4 * T + 4  # = 12*T + 4 for T=128 -> 1540 bytes

    @classmethod
    def for_device(
        cls,
        device: "cp.cuda.Device | None" = None,
        target_blocks_per_sm: int = 4,
        min_max_extrema: int = 1024,
    ) -> "EMDKernelConfig":
        """Size MAX_EXTREMA for OCCUPANCY, not for maximum per-block
        capacity. An earlier version used ~90% of whatever shared memory
        the device allowed per block, which sounds generous but is
        actively bad for a batch of many small independent blocks: a
        large per-block footprint leaves room for only 1-2 concurrent
        blocks per SM, so a big batch runs in many sequential "waves",
        and any one slow-to-converge signal in a wave holds up that
        whole wave (confirmed: 200 signals gave 6.4x speedup over PyEMD,
        2000 gave 3.8x -- backwards from what proper batching should do).

        Fix: pick a per-block shared-mem budget sized for
        `target_blocks_per_sm` concurrent blocks (default 4 -- comfortably
        below the ~16-32 blocks/SM hardware ceiling), and only use MORE
        than `min_max_extrema` if the device has enough shared memory to
        do that *and* still hit the occupancy target -- never shrink
        below min_max_extrema just to hit a block-count target, since
        real signals (up to ~700 extrema seen in testing) need to fit
        regardless of occupancy.

        HARD CEILING, non-negotiable: whatever max_extrema this lands on
        must actually fit in what the device grants per block -- fitting
        at all comes before occupancy, and occupancy comes before hitting
        min_max_extrema. If even a modest max_extrema doesn't fit, the
        overflow-detection safety net (sh->overflow) is what handles
        signals that exceed whatever we land on -- a real answer for
        most signals beats a kernel that won't launch at all."""
        dev = device or cp.cuda.Device()
        with dev:
            attrs = dev.attributes
            # Per-block opt-in max is a close proxy for per-SM total shared
            # memory on every current architecture (e.g. Hopper: 227KB/block
            # opt-in vs 228KB/SM total) -- close enough for sizing purposes,
            # and CuPy doesn't expose the per-SM figure directly.
            max_shared_per_sm_ish = attrs.get("MaxSharedMemoryPerBlockOptin", 49152)
        cfg = cls()
        static_bytes = cfg.static_shared_mem_bytes()
        # 6 REAL_T (float64) arrays (ext_pos, ext_val, cp/PCR-b, rhs/PCR-d,
        # pcr_a, pcr_c -- the last two added for the parallel tridiagonal
        # solve) + 2 int32 (ind_max, ind_min). itemsize is always 8.
        bytes_per_mbuf_elem = 6 * cfg.itemsize + 2 * 4
        # Budgets below are for the DYNAMIC portion only -- static_bytes is
        # reserved off the top first, since it's a fixed cost the combined
        # total has to accommodate regardless of how MAX_EXTREMA is chosen.
        budget_for_occupancy = max(0, max_shared_per_sm_ish // target_blocks_per_sm - static_bytes)
        m_capacity_occ = budget_for_occupancy // bytes_per_mbuf_elem
        max_extrema_occ = max(0, m_capacity_occ - 2 * cfg.nbsym)
        desired = max(min_max_extrema, max_extrema_occ)

        # Clamp to what the device can actually grant: dynamic + static
        # combined must fit, with a little slack rather than pushing
        # exactly to the edge. An earlier version only reserved room for
        # the dynamic struct here, which passed its own check yet still
        # failed at actual kernel launch once the static __shared__
        # arrays (see static_shared_mem_bytes()) were added on top.
        hard_budget = max(0, int(max_shared_per_sm_ish * 0.95) - static_bytes)
        hard_m_cap = hard_budget // bytes_per_mbuf_elem
        hard_max_extrema = max(64, hard_m_cap - 2 * cfg.nbsym)

        cfg.max_extrema = min(desired, hard_max_extrema)
        return cfg

    def shared_mem_bytes(self) -> int:
        m = self.max_extrema + 2 * self.nbsym
        return m * (6 * self.itemsize + 2 * 4) + 256  # + slack for the scalar fields; see for_device() note on the 6

    def build(self) -> "EMDBatchLauncher":
        src = _CUDA_SOURCE % {
            "MAX_EXTREMA": self.max_extrema,
            "NBSYM": self.nbsym,
            "MAX_ITERATION": self.max_iteration,
            "MAX_IMFS": self.max_imfs,
            "THREADS": self.threads_per_block,
            "REAL_T": self.real_t_name,
        }
        return EMDBatchLauncher(src, self)


class EMDBatchLauncher:
    """Compiles the kernel and owns the (per-call) global-memory scratch
    pools for the N-length working arrays (residue/imf/imf_old/mean/
    envelopes) that the design deliberately keeps OUT of shared memory."""

    def __init__(self, source: str, config: EMDKernelConfig):
        self.config = config
        self.module = cp.RawModule(code=source, options=("--std=c++11",))
        self.kernel = self.module.get_function("emd_batch_kernel")
        self.envelope_kernel = self.module.get_function("envelope_batch_kernel")

        shmem = config.shared_mem_bytes()
        static_bytes = config.static_shared_mem_bytes()
        total_needed = shmem + static_bytes
        device_max = cp.cuda.Device().attributes.get("MaxSharedMemoryPerBlockOptin", 49152)
        if total_needed > device_max:
            raise ValueError(
                f"EMDKernelConfig.max_extrema={config.max_extrema} needs {shmem} bytes of dynamic "
                f"shared memory plus {static_bytes} bytes of static shared memory ({total_needed} "
                f"total) per block, but this device only allows {device_max}. "
                f"Use EMDKernelConfig.for_device() instead of a hand-built config (it accounts for "
                f"both and clamps to what the device supports), or lower max_extrema/nbsym directly."
            )
        # The opt-in dynamic-shared-memory API (cudaFuncSetAttribute /
        # max_dynamic_shared_size_bytes) only exists from Volta onward --
        # calling it AT ALL fails on Pascal, regardless of the requested
        # size, not just when the size is too large (confirmed: this was
        # still failing even after the value itself was clamped to fit).
        # It's also simply unnecessary up to the universal 48KB default
        # (dynamic + static COMBINED, not dynamic alone -- confirmed by
        # this exact class of bug: a config that fit dynamic-only under
        # 48KB but not dynamic+static still failed at kernel launch, not
        # at this attribute-setting step, since the setter was correctly
        # skipped -- the combined total was the actual problem).
        STATIC_SHARED_MEM_DEFAULT = 49152
        if total_needed > STATIC_SHARED_MEM_DEFAULT:
            try:
                self.kernel.max_dynamic_shared_size_bytes = shmem
                self.envelope_kernel.max_dynamic_shared_size_bytes = shmem
            except AttributeError:
                raise RuntimeError(
                    "This CuPy version doesn't expose RawKernel.max_dynamic_shared_size_bytes; "
                    "check cupy.__version__ and the CuPy RawKernel docs for the opt-in API."
                )
            except Exception as e:
                raise RuntimeError(
                    f"Requesting {shmem} bytes of dynamic shared memory (totaling {total_needed} "
                    f"with the {static_bytes}-byte static portion, > the 48KB default) failed on "
                    f"this device even though it should support opting in past 48KB: {e}. "
                    f"If this is a Pascal-class GPU, opt-in above 48KB isn't available at all -- "
                    f"pass a smaller max_extrema (or use EMDKernelConfig.for_device(), which now "
                    f"accounts for static shared memory too when staying under 48KB)."
                )

    def run(self, signals: "cp.ndarray", T: "cp.ndarray") -> tuple:
        batch, N = signals.shape
        cfg = self.config
        if signals.dtype != cfg.dtype or T.dtype != cfg.dtype:
            raise ValueError(
                f"This launcher was compiled for REAL_T={cfg.real_t_name} (dtype {cfg.dtype}), "
                f"but got signals.dtype={signals.dtype}, T.dtype={T.dtype}. Passing mismatched "
                f"dtypes would silently reinterpret bytes rather than convert them -- build a "
                f"launcher matching your data's dtype instead of casting to fit an existing one."
            )
        pools = [
            cp.empty((batch, N), dtype=cfg.dtype)
            for _ in range(6)  # residue, imf, imf_old, mean, max_env, min_env
        ]

        imfs_out = cp.zeros((batch, cfg.max_imfs, N), dtype=cfg.dtype)
        n_imfs_out = cp.zeros((batch,), dtype=cp.int32)
        overflow_out = cp.zeros((batch,), dtype=cp.int32)
        debug_n_out = cp.full((batch, cfg.max_imfs), -1, dtype=cp.int32)
        debug_extno_out = cp.full((batch, cfg.max_imfs), -1, dtype=cp.int32)
        debug_nzm_out = cp.full((batch, cfg.max_imfs), -1, dtype=cp.int32)

        shmem = cfg.shared_mem_bytes()
        self.kernel(
            (batch,), (cfg.threads_per_block,),
            (signals, T, N, imfs_out, n_imfs_out, overflow_out, *pools,
             debug_n_out, debug_extno_out, debug_nzm_out),
            shared_mem=shmem,
        )
        return imfs_out, n_imfs_out, overflow_out, debug_n_out, debug_extno_out, debug_nzm_out

    def run_envelope_only(self, signals: "cp.ndarray", T: "cp.ndarray") -> tuple:
        """DIAGNOSTIC ONLY -- runs envelope_batch_kernel (extrema + mirror
        + spline, no sift loop) so it can be checked in isolation against
        numpy_emd_reference.py's extract_max_min_spline. See
        benchmark/diagnose_cuda_envelope.py."""
        batch, N = signals.shape
        cfg = self.config
        if signals.dtype != cfg.dtype or T.dtype != cfg.dtype:
            raise ValueError(
                f"This launcher was compiled for REAL_T={cfg.real_t_name} (dtype {cfg.dtype}), "
                f"but got signals.dtype={signals.dtype}, T.dtype={T.dtype}."
            )
        mbuf = cfg.max_extrema + 2 * cfg.nbsym
        max_env_out = cp.zeros((batch, N), dtype=cfg.dtype)
        min_env_out = cp.zeros((batch, N), dtype=cfg.dtype)
        n_extrema_out = cp.zeros((batch, 2), dtype=cp.int32)
        degenerate_out = cp.zeros((batch,), dtype=cp.int32)
        max_knot_pos_out = cp.zeros((batch, mbuf), dtype=cfg.dtype)
        max_knot_val_out = cp.zeros((batch, mbuf), dtype=cfg.dtype)
        min_knot_pos_out = cp.zeros((batch, mbuf), dtype=cfg.dtype)
        min_knot_val_out = cp.zeros((batch, mbuf), dtype=cfg.dtype)
        n_mirrored_out = cp.zeros((batch, 2), dtype=cp.int32)

        shmem = cfg.shared_mem_bytes()
        self.envelope_kernel(
            (batch,), (cfg.threads_per_block,),
            (signals, T, N, max_env_out, min_env_out, n_extrema_out, degenerate_out,
             max_knot_pos_out, max_knot_val_out, min_knot_pos_out, min_knot_val_out,
             n_mirrored_out),
            shared_mem=shmem,
        )
        return (max_env_out, min_env_out, n_extrema_out, degenerate_out,
                max_knot_pos_out, max_knot_val_out, min_knot_pos_out, min_knot_val_out,
                n_mirrored_out)
