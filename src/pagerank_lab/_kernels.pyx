# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Private kernels: call only via the validated Python API.

Pull traversal avoids atomic/scattered writes. One fused pass computes the next
iterate, L1 update and next dangling mass. There is no allocation in the loop.
"""
import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange
from libc.math cimport fabs

cnp.import_array()

cdef extern from *:
    """
    #ifndef PAGERANK_HAVE_OPENMP
    #define PAGERANK_HAVE_OPENMP 0
    #endif
    #ifndef PAGERANK_NATIVE_BUILD
    #define PAGERANK_NATIVE_BUILD 0
    #endif
    static int pr_has_openmp(void) { return PAGERANK_HAVE_OPENMP; }
    static int pr_native_build(void) { return PAGERANK_NATIVE_BUILD; }
    static const char* pr_compiler(void) {
    #ifdef __VERSION__
      return __VERSION__;
    #elif defined(_MSC_VER)
      return "MSVC";
    #else
      return "unknown";
    #endif
    }
    """
    int pr_has_openmp() noexcept nogil
    int pr_native_build() noexcept nogil
    const char* pr_compiler() noexcept nogil

ctypedef fused real_t:
    cnp.float32_t
    cnp.float64_t

ctypedef fused index_t:
    cnp.int32_t
    cnp.int64_t

cdef inline double _row_sum(const index_t* indices, const real_t* data,
                            const real_t* x, index_t start, index_t stop) noexcept nogil:
    cdef index_t k
    cdef double value = 0.
    for k in range(start, stop):
        value += <double>data[k] * <double>x[indices[k]]
    return value


def build_info():
    return {"openmp": bool(pr_has_openmp()), "native": bool(pr_native_build()),
            "compiler": (<bytes>pr_compiler()).decode("utf8"),
            "fast_math": False, "kernel": "fused-pull-csr-balanced"}


cdef void _step(const index_t* indptr, const index_t* indices, const real_t* data,
               const cnp.uint8_t* dangling, const real_t* p, const real_t* q,
               const real_t* x, real_t* y, Py_ssize_t start, Py_ssize_t stop,
               double alpha, double dm, Py_ssize_t n, int mode,
               double* result_delta, double* result_dm) noexcept nogil:
    cdef Py_ssize_t i
    cdef double delta = 0., new_dm = 0., base = 1.-alpha + alpha*dm
    cdef real_t stored
    if mode == 0:
        base /= n
        for i in range(start, stop):
            stored = <real_t>(alpha * _row_sum(indices, data, x, indptr[i], indptr[i+1]) + base)
            y[i] = stored
            delta += fabs(<double>stored - <double>x[i])
            if dangling[i]:
                new_dm += stored
    elif mode == 1:
        for i in range(start, stop):
            stored = <real_t>(alpha * _row_sum(indices, data, x, indptr[i], indptr[i+1]) + base*p[i])
            y[i] = stored
            delta += fabs(<double>stored - <double>x[i])
            if dangling[i]:
                new_dm += stored
    else:
        for i in range(start, stop):
            stored = <real_t>(alpha * _row_sum(indices, data, x, indptr[i], indptr[i+1]) + (1.-alpha)*p[i] + alpha*dm*q[i])
            y[i] = stored
            delta += fabs(<double>stored - <double>x[i])
            if dangling[i]:
                new_dm += stored
    result_delta[0], result_dm[0] = delta, new_dm


def power(const index_t[::1] indptr, const index_t[::1] indices,
          const real_t[::1] data, const cnp.uint8_t[::1] dangling,
          const real_t[::1] p, const real_t[::1] q,
          real_t[::1] first, real_t[::1] second,
          double alpha, double threshold, int max_iter, int threads,
          bint uniform, bint same_pq, double[::1] history,
          const cnp.int64_t[::1] cuts, double[::1] scratch):
    """Unsafe internal API. Python caller validates lengths and CSR structure."""
    cdef Py_ssize_t n = first.shape[0]
    cdef Py_ssize_t i
    cdef int iteration = 0, part, mode = 0 if uniform else (1 if same_pq else 2)
    cdef double delta = 0., dm = 0., next_dm = 0., value, base
    cdef real_t stored
    cdef real_t* x = &first[0]
    cdef real_t* y = &second[0]
    cdef real_t* swap
    cdef const real_t* weights = NULL
    cdef const index_t* edges = NULL
    cdef bint record = history.shape[0] > 0
    if indices.shape[0]:
        weights, edges = &data[0], &indices[0]
    with nogil:
        for i in range(n):
            if dangling[i]:
                dm += x[i]
        for iteration in range(max_iter):
            delta, next_dm = 0., 0.
            if threads == 1:
                _step(&indptr[0], edges, weights, &dangling[0], &p[0], &q[0],
                      x, y, 0, n, alpha, dm, n, mode, &delta, &next_dm)
            else:
                for part in prange(threads, schedule='static', num_threads=threads):
                    # Per-worker scalars occupy separate cache lines (8 doubles).
                    _step(&indptr[0], edges, weights, &dangling[0], &p[0], &q[0],
                          x, y, cuts[part], cuts[part+1], alpha, dm, n, mode,
                          &scratch[16*part], &scratch[16*part+8])
                # Fixed-order reduction gives repeatable results at fixed threads.
                for part in range(threads):
                    delta += scratch[16*part]
                    next_dm += scratch[16*part+8]
            swap, x, y = x, y, x
            dm = next_dm
            if record:
                history[iteration] = delta
            if delta <= threshold:
                break
    return iteration + 1, delta
