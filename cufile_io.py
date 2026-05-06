"""GPUDirect Storage (cuFile) bindings + manual DLPack capsule construction.

This module wires NVIDIA cuFile so the disk-backed gradient cache can be
read directly from NVMe into GPU memory, bypassing the host memcpy +
host→GPU DMA. The work is split into three layers:

  * cuFile bindings (CuFileHandle) — open/register a file, issue
    cuFileRead into an arbitrary device pointer.
  * GPU buffer pool (CuFileGPUBuffer / CuFileBufferPool) — raw GPU
    allocations from cudaMalloc, registered with cuFileBufRegister for
    best-case bandwidth.
  * Manual DLPack capsule (build_dlpack_capsule) — wraps a raw devptr
    as a TF tensor whose lifetime returns the buffer to its pool when
    the consumer drops the tensor.

On systems where the nvidia_fs kernel module isn't loaded (e.g. WSL2
without GDS), cuFile transparently falls back to compatibility mode:
the read goes through a kernel-stage buffer rather than true direct
DMA. Behaviour is identical from this code's perspective.

Threading: cuFile reads are pure C calls and are safe to invoke from
worker threads. tf.experimental.dlpack.from_dlpack must run on the
main thread (TF eager has thread-safety quirks for tensor creation),
so the worker should populate buffers and the consumer should wrap
them as TF tensors afterwards.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np
import tensorflow as tf


# =============================================================================
# cuFile bindings
# =============================================================================

_libcufile = None


def _libcf():
    """Lazy-load libcufile.so. Raises OSError if cuFile isn't installed."""
    global _libcufile
    if _libcufile is None:
        for name in ("libcufile.so", "libcufile.so.0"):
            try:
                _libcufile = ctypes.CDLL(name)
                break
            except OSError:
                continue
        else:
            raise OSError("Could not load libcufile.so (cuFile/GDS not installed)")
        _libcufile.cuFileDriverOpen.restype = ctypes.c_int
        _libcufile.cuFileDriverClose.restype = ctypes.c_int
        _libcufile.cuFileHandleRegister.restype = ctypes.c_int
        _libcufile.cuFileHandleDeregister.restype = ctypes.c_int
        _libcufile.cuFileRead.restype = ctypes.c_ssize_t
        _libcufile.cuFileBufRegister.restype = ctypes.c_int
        _libcufile.cuFileBufDeregister.restype = ctypes.c_int
    return _libcufile


# CUfileFileHandleType enum
_CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1


class _CuFileHandleUnion(ctypes.Union):
    _fields_ = [
        ("fd",     ctypes.c_int),
        ("handle", ctypes.c_void_p),
    ]


class _CuFileDescr_t(ctypes.Structure):
    """Mirrors CUfileDescr_t exactly: int type + union(fd|handle) + fs_ops*.

    ctypes inserts natural alignment so the union (sizeof void*) lands at
    offset 8 and fs_ops at offset 16 — matching the C layout."""
    _fields_ = [
        ("type",   ctypes.c_int),
        ("handle", _CuFileHandleUnion),
        ("fs_ops", ctypes.c_void_p),
    ]


_driver_open = False


def cuFile_available() -> bool:
    """Return True iff libcufile is loadable AND cuFileDriverOpen succeeds.

    Result is cached for the process lifetime; on success the driver is
    opened exactly once. cuFile uses compatibility mode if the
    nvidia_fs kernel module isn't loaded — that's still a valid 'True'
    here because the API works.
    """
    global _driver_open
    if _driver_open:
        return True
    try:
        cf = _libcf()
    except OSError:
        return False
    err = cf.cuFileDriverOpen()
    if err != 0:
        return False
    _driver_open = True
    return True


class CuFileHandle:
    """Wraps a registered cuFile handle for one open file.

    Closes both the handle and the underlying fd in __del__. The fd is
    opened *without* O_DIRECT — strict alignment is impractical for our
    layout (each pair row is 3·Q·4 bytes, not aligned to 4 KiB). cuFile
    falls back to buffered I/O, which is still considerably faster than
    the pageable host path in our existing pipeline.
    """
    def __init__(self, path: str):
        if not cuFile_available():
            raise OSError("cuFile driver not available")
        self._cf = _libcf()
        self._fd = os.open(path, os.O_RDONLY)
        descr = _CuFileDescr_t()
        descr.type = _CU_FILE_HANDLE_TYPE_OPAQUE_FD
        descr.handle.fd = self._fd
        descr.fs_ops = None
        self._handle = ctypes.c_void_p()
        err = self._cf.cuFileHandleRegister(
            ctypes.byref(self._handle), ctypes.byref(descr))
        if err != 0:
            os.close(self._fd)
            raise RuntimeError(f"cuFileHandleRegister failed: code {err}")

    def read(self, devptr: int, nbytes: int, file_offset: int) -> int:
        """Read `nbytes` from `file_offset` into the GPU memory at
        `devptr`. Returns the number of bytes read (negative on error)."""
        n = self._cf.cuFileRead(
            self._handle,
            ctypes.c_void_p(int(devptr)),
            ctypes.c_size_t(int(nbytes)),
            ctypes.c_int64(int(file_offset)),
            ctypes.c_int64(0),
        )
        if n < 0:
            raise RuntimeError(f"cuFileRead failed: code {n}")
        return int(n)

    def __del__(self):
        try:
            if getattr(self, "_handle", None) and self._handle.value:
                self._cf.cuFileHandleDeregister(self._handle)
                self._handle.value = 0
            if getattr(self, "_fd", -1) >= 0:
                os.close(self._fd)
                self._fd = -1
        except Exception:
            pass


# =============================================================================
# Raw GPU buffers (cudaMalloc + cuFileBufRegister)
# =============================================================================

_libcudart = None


def _libcuda():
    global _libcudart
    if _libcudart is None:
        for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
            try:
                _libcudart = ctypes.CDLL(name)
                break
            except OSError:
                continue
        else:
            raise OSError("Could not load libcudart.so")
        _libcudart.cudaMalloc.restype = ctypes.c_int
        _libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        _libcudart.cudaFree.restype = ctypes.c_int
        _libcudart.cudaFree.argtypes = [ctypes.c_void_p]
    return _libcudart


class CuFileGPUBuffer:
    """Raw GPU memory allocated via cudaMalloc, registered with
    cuFileBufRegister for best-case GDS bandwidth. Each buffer is sized
    to the worst-case chunk grad slice.
    """
    def __init__(self, nbytes: int):
        cuda = _libcuda()
        cf = _libcf()
        cf.cuFileBufRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        cf.cuFileBufDeregister.argtypes = [ctypes.c_void_p]
        self.nbytes = int(nbytes)
        self._p = ctypes.c_void_p()
        err = cuda.cudaMalloc(ctypes.byref(self._p), ctypes.c_size_t(self.nbytes))
        if err != 0:
            raise RuntimeError(f"cudaMalloc({self.nbytes}) failed: code {err}")
        # Buf-register is optional but improves throughput; failure is
        # non-fatal (cuFile falls back to internal staging).
        cf.cuFileBufRegister(self._p, ctypes.c_size_t(self.nbytes), 0)
        self._registered = True

    @property
    def devptr(self) -> int:
        return int(self._p.value)

    def __del__(self):
        try:
            if getattr(self, "_registered", False):
                _libcf().cuFileBufDeregister(self._p)
                self._registered = False
        except Exception:
            pass
        try:
            if getattr(self, "_p", None) is not None and self._p.value:
                _libcuda().cudaFree(self._p)
                self._p.value = 0
        except Exception:
            pass


class CuFileBufferPool:
    """FIFO pool of GPU buffers for the cuFile staging path.

    Like the pinned-buffer pool, the consumer of a wrapped tensor holds
    a release cookie that returns the buffer to the pool when the tensor
    is destroyed. acquire() returns None if the pool is exhausted.
    """
    def __init__(self, n_buffers: int, nbytes: int):
        from collections import deque
        self.nbytes = int(nbytes)
        self._all = [CuFileGPUBuffer(nbytes) for _ in range(n_buffers)]
        self._free = deque(self._all)

    def acquire(self):
        if not self._free:
            return None
        return self._free.popleft()

    def release(self, buf):
        if buf is None:
            return
        self._free.append(buf)


# =============================================================================
# Manual DLPack capsule construction
# =============================================================================
# DLPack v0.8 layout — matches https://github.com/dmlc/dlpack
# kDLCUDA = 2, kDLFloat = 2, kDLInt = 0


class _DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", ctypes.c_int32),
        ("device_id",   ctypes.c_int32),
    ]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code",  ctypes.c_uint8),
        ("bits",  ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data",        ctypes.c_void_p),
        ("device",      _DLDevice),
        ("ndim",        ctypes.c_int32),
        ("dtype",       _DLDataType),
        ("shape",       ctypes.POINTER(ctypes.c_int64)),
        ("strides",     ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLDeleterFn = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))

_DLManagedTensor._fields_ = [
    ("dl_tensor",   _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter",     _DLDeleterFn),
]


def _np_dtype_to_dlpack(dtype) -> tuple:
    np_dtype = np.dtype(dtype)
    if np_dtype == np.float32: return (2, 32, 1)
    if np_dtype == np.float64: return (2, 64, 1)
    if np_dtype == np.float16: return (2, 16, 1)
    if np_dtype == np.int32:   return (0, 32, 1)
    if np_dtype == np.int64:   return (0, 64, 1)
    if np_dtype == np.uint8:   return (1, 8, 1)
    raise ValueError(f"unsupported dtype: {np_dtype}")


# PyCapsule_New isn't exposed by ctypes.pythonapi by default; configure it.
_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

# Keep DLManagedTensor instances + their shape buffers + deleter
# CFUNCTYPE objects alive for as long as TF holds the tensor. The
# deleter pops the entry when it fires.
_kept_alive: dict = {}


def build_dlpack_capsule(devptr: int, np_dtype, shape: tuple,
                          on_destroy):
    """Wrap a raw device pointer as a DLPack capsule with name 'dltensor'.

    `on_destroy()` is called from the DLManagedTensor deleter when TF
    destroys the resulting tensor — that's the hook we use to release
    the GPU buffer back to its pool.

    Returns a PyCapsule object suitable for tf.experimental.dlpack.from_dlpack.
    """
    code, bits, lanes = _np_dtype_to_dlpack(np_dtype)
    ndim = len(shape)

    # Persistent backing for shape and the managed tensor (DLPack consumer
    # only reads shape until the deleter fires, so we keep both alive in
    # _kept_alive until then).
    shape_arr = (ctypes.c_int64 * ndim)(*[int(d) for d in shape])
    mt = _DLManagedTensor()
    mt.dl_tensor.data = ctypes.c_void_p(int(devptr))
    mt.dl_tensor.device.device_type = 2  # kDLCUDA
    mt.dl_tensor.device.device_id   = 0
    mt.dl_tensor.ndim = ndim
    mt.dl_tensor.dtype.code  = code
    mt.dl_tensor.dtype.bits  = bits
    mt.dl_tensor.dtype.lanes = lanes
    mt.dl_tensor.shape = shape_arr
    mt.dl_tensor.strides = None  # row-major contiguous
    mt.dl_tensor.byte_offset = 0
    mt.manager_ctx = None

    mt_addr = ctypes.addressof(mt)

    def _deleter(self_ptr):
        try:
            on_destroy()
        finally:
            _kept_alive.pop(mt_addr, None)

    deleter_cfunc = _DLDeleterFn(_deleter)
    mt.deleter = deleter_cfunc

    # Register everything so the GC can't reap it before TF's deleter fires.
    _kept_alive[mt_addr] = (mt, shape_arr, deleter_cfunc, _deleter)

    capsule = _PyCapsule_New(mt_addr, b"dltensor", None)
    return capsule


# =============================================================================
# High-level: read a chunk from cuFile into a TF tensor
# =============================================================================

def cufile_read_chunk_as_tf_tensor(
    handle: CuFileHandle,
    pool: CuFileBufferPool,
    file_offset_bytes: int,
    shape: tuple,
    dtype=np.float32,
    buf=None,
) -> "tf.Tensor | None":
    """High-level helper: pull `shape` worth of `dtype` data starting at
    `file_offset_bytes` from `handle` into a pooled GPU buffer, wrap as
    a TF tensor that aliases the buffer, and return it. The buffer goes
    back to `pool` only when the TF tensor is destroyed.

    `buf` may be passed in if the caller already acquired one in a
    worker thread (recommended); otherwise this function acquires from
    `pool` itself.

    Returns None if the pool is exhausted (caller should fall back).
    """
    if buf is None:
        buf = pool.acquire()
    if buf is None:
        return None
    nelems = 1
    for d in shape:
        nelems *= int(d)
    nbytes = nelems * int(np.dtype(dtype).itemsize)
    if buf is not buf:
        pass  # placeholder
    if nbytes > buf.nbytes:
        pool.release(buf)
        raise ValueError(f"cufile chunk needs {nbytes} > buffer {buf.nbytes}")
    handle.read(buf.devptr, nbytes, file_offset_bytes)

    def _release():
        pool.release(buf)

    capsule = build_dlpack_capsule(buf.devptr, dtype, shape, _release)
    return tf.experimental.dlpack.from_dlpack(capsule)


# =============================================================================
# Pool factory mirroring the pinned-buffer interface
# =============================================================================

def make_cufile_pool_for(data: dict, batch_chunk_size: int,
                          n_buffers: int = 4) -> "CuFileBufferPool | None":
    """Build a GPU buffer pool sized to the worst-case chunk grad slice
    that slice_and_complete_chunk will need to stage from `data`.
    Returns None when cuFile or cudart aren't loadable.
    """
    if not cuFile_available():
        return None
    if "grad_values" not in data or "struct_ptr" not in data:
        return None
    gv = data["grad_values"]
    if gv.shape is None or len(gv.shape) < 3:
        return None
    Q = int(gv.shape[2])
    item = int(np.dtype(gv.dtype).itemsize)
    sp = data["struct_ptr"].numpy() if hasattr(data["struct_ptr"], "numpy") else np.asarray(data["struct_ptr"])
    S = int(sp.shape[0]) - 1
    chunk = int(batch_chunk_size) if batch_chunk_size is not None else S
    chunk = max(1, min(chunk, S))
    max_pairs = 0
    for s in range(0, S, chunk):
        e = min(s + chunk, S)
        d = int(sp[e]) - int(sp[s])
        if d > max_pairs:
            max_pairs = d
    if max_pairs == 0:
        return None
    nbytes = int(max_pairs * 3 * Q * item * 1.05)  # 5% headroom
    try:
        return CuFileBufferPool(n_buffers, nbytes)
    except (RuntimeError, OSError):
        return None
