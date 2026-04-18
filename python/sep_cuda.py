from __future__ import annotations

import ctypes as ct
import os
from ctypes.util import find_library
from pathlib import Path

import numpy as np

SEP_TBYTE = 11
SEP_TINT = 31
SEP_TFLOAT = 42
SEP_TDOUBLE = 82

RETURN_OK = 0
MEMORY_ALLOC_ERROR = 1
ILLEGAL_DTYPE = 3
SEP_CUDA_UNAVAILABLE = 100
SEP_CUDA_RUNTIME_ERROR = 101


class Error(Exception):
    pass


class _SepImage(ct.Structure):
    _fields_ = [
        ("data", ct.c_void_p),
        ("noise", ct.c_void_p),
        ("mask", ct.c_void_p),
        ("segmap", ct.c_void_p),
        ("dtype", ct.c_int),
        ("ndtype", ct.c_int),
        ("mdtype", ct.c_int),
        ("sdtype", ct.c_int),
        ("segids", ct.POINTER(ct.c_int64)),
        ("idcounts", ct.POINTER(ct.c_int64)),
        ("numids", ct.c_int64),
        ("w", ct.c_int64),
        ("h", ct.c_int64),
        ("noiseval", ct.c_double),
        ("noise_type", ct.c_short),
        ("gain", ct.c_double),
        ("maskthresh", ct.c_double),
    ]


class _SepBkg(ct.Structure):
    _fields_ = [
        ("w", ct.c_int64),
        ("h", ct.c_int64),
        ("bw", ct.c_int64),
        ("bh", ct.c_int64),
        ("nx", ct.c_int64),
        ("ny", ct.c_int64),
        ("n", ct.c_int64),
        ("global", ct.c_float),
        ("globalrms", ct.c_float),
        ("back", ct.c_void_p),
        ("dback", ct.c_void_p),
        ("sigma", ct.c_void_p),
        ("dsigma", ct.c_void_p),
    ]


def _library_candidates() -> list[str]:
    here = Path(__file__).resolve()
    candidates: list[str] = []

    env_path = os.environ.get("SEPCUDA_LIBRARY")
    if env_path:
        candidates.append(env_path)

    candidates.append(str(here.parent.parent / "lib" / "lib_sep_cuda.so"))
    candidates.append(str(here.parent.parent / "build" / "lib_sep_cuda.so"))

    found = find_library("sep_cuda")
    if found:
        candidates.append(found)

    candidates.append("lib_sep_cuda.so")
    return candidates


def _load_library() -> ct.CDLL:
    last_error: OSError | None = None
    for candidate in _library_candidates():
        try:
            return ct.CDLL(candidate)
        except OSError as exc:
            last_error = exc

    raise ImportError(f"failed to load lib_sep_cuda.so: {last_error}") from last_error


_lib = _load_library()

_lib.sep_background.argtypes = [
    ct.POINTER(_SepImage),
    ct.c_int64,
    ct.c_int64,
    ct.c_int64,
    ct.c_int64,
    ct.c_double,
    ct.POINTER(ct.POINTER(_SepBkg)),
]
_lib.sep_background.restype = ct.c_int

_lib.sep_bkg_global.argtypes = [ct.POINTER(_SepBkg)]
_lib.sep_bkg_global.restype = ct.c_float

_lib.sep_bkg_globalrms.argtypes = [ct.POINTER(_SepBkg)]
_lib.sep_bkg_globalrms.restype = ct.c_float

_lib.sep_bkg_array.argtypes = [ct.POINTER(_SepBkg), ct.c_void_p, ct.c_int]
_lib.sep_bkg_array.restype = ct.c_int

_lib.sep_bkg_rmsarray.argtypes = [ct.POINTER(_SepBkg), ct.c_void_p, ct.c_int]
_lib.sep_bkg_rmsarray.restype = ct.c_int

_lib.sep_bkg_subarray.argtypes = [ct.POINTER(_SepBkg), ct.c_void_p, ct.c_int]
_lib.sep_bkg_subarray.restype = ct.c_int

_lib.sep_bkg_free.argtypes = [ct.POINTER(_SepBkg)]
_lib.sep_bkg_free.restype = None

_lib.sep_get_errmsg.argtypes = [ct.c_int, ct.c_char_p]
_lib.sep_get_errmsg.restype = None

_lib.sep_get_errdetail.argtypes = [ct.c_char_p]
_lib.sep_get_errdetail.restype = None


def get_errmsg(status: int) -> str:
    buf = ct.create_string_buffer(128)
    _lib.sep_get_errmsg(int(status), buf)
    return buf.value.decode()


def get_errdetail() -> str:
    buf = ct.create_string_buffer(512)
    _lib.sep_get_errdetail(buf)
    return buf.value.decode()


def _raise_on_error(status: int) -> None:
    if status == RETURN_OK:
        return

    if status == MEMORY_ALLOC_ERROR:
        raise MemoryError()

    message = get_errmsg(status)
    detail = get_errdetail()
    if detail:
        message = f"{message}: {detail}"
    raise Error(message)


def _dtype_to_sep(dtype: np.dtype) -> int:
    dtype = np.dtype(dtype)

    if not dtype.isnative:
        raise ValueError(
            "Input array with dtype "
            f"`{dtype}` has non-native byte order. "
            "Only native byte order arrays are supported. "
            "To change the byte order of the array `data`, do "
            "`data = data.astype(data.dtype.newbyteorder('='))`"
        )

    dtype_type = dtype.type
    if dtype_type is np.single:
        return SEP_TFLOAT
    if dtype == np.double:
        return SEP_TDOUBLE
    if dtype == np.dtype(np.intc):
        return SEP_TINT
    if dtype_type is np.bool_ or dtype_type is np.ubyte:
        return SEP_TBYTE

    raise ValueError(f"input array dtype not supported: {dtype}")


def _require_c_contiguous_2d(array: np.ndarray) -> None:
    if not array.flags.c_contiguous:
        raise ValueError("array is not C-contiguous")
    if array.ndim != 2:
        raise ValueError("array must be 2-d")


def _as_input_array(data) -> np.ndarray:
    array = np.asarray(data)
    _require_c_contiguous_2d(array)
    _dtype_to_sep(array.dtype)
    return array


def _as_mask_array(mask, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask)
    _require_c_contiguous_2d(array)
    if array.shape != shape:
        raise ValueError("size of mask array must match data")
    _dtype_to_sep(array.dtype)
    return array


class Background:
    """
    Background(data, mask=None, maskthresh=0.0, bw=64, bh=64, fw=3, fh=3, fthresh=0.0)

    Spatially variable image background and noise model.
    """

    def __init__(
        self,
        data,
        mask=None,
        maskthresh: float = 0.0,
        bw: int = 64,
        bh: int = 64,
        fw: int = 3,
        fh: int = 3,
        fthresh: float = 0.0,
    ) -> None:
        data_array = _as_input_array(data)
        mask_array = None if mask is None else _as_mask_array(mask, data_array.shape)

        image = _SepImage()
        image.data = data_array.ctypes.data
        image.noise = None
        image.mask = None if mask_array is None else mask_array.ctypes.data
        image.segmap = None
        image.dtype = _dtype_to_sep(data_array.dtype)
        image.ndtype = 0
        image.mdtype = 0 if mask_array is None else _dtype_to_sep(mask_array.dtype)
        image.sdtype = 0
        image.segids = None
        image.idcounts = None
        image.numids = 0
        image.w = int(data_array.shape[1])
        image.h = int(data_array.shape[0])
        image.noiseval = 0.0
        image.noise_type = 0
        image.gain = 1.0
        image.maskthresh = float(maskthresh)

        bkg_ptr = ct.POINTER(_SepBkg)()
        status = _lib.sep_background(
            ct.byref(image),
            int(bw),
            int(bh),
            int(fw),
            int(fh),
            float(fthresh),
            ct.byref(bkg_ptr),
        )
        _raise_on_error(status)

        self._ptr = bkg_ptr
        self._shape = data_array.shape
        self._orig_dtype = data_array.dtype

    def __del__(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            _lib.sep_bkg_free(ptr)
            self._ptr = None

    @property
    def globalback(self) -> float:
        return float(_lib.sep_bkg_global(self._ptr))

    @property
    def globalrms(self) -> float:
        return float(_lib.sep_bkg_globalrms(self._ptr))

    def back(self, dtype=None, copy=None):
        if dtype is None:
            dtype = self._orig_dtype
        dtype = np.dtype(dtype)
        result = np.empty(self._shape, dtype=dtype)
        status = _lib.sep_bkg_array(self._ptr, result.ctypes.data, _dtype_to_sep(dtype))
        _raise_on_error(status)
        return result.copy() if copy else result

    def rms(self, dtype=None):
        if dtype is None:
            dtype = self._orig_dtype
        dtype = np.dtype(dtype)
        result = np.empty(self._shape, dtype=dtype)
        status = _lib.sep_bkg_rmsarray(self._ptr, result.ctypes.data, _dtype_to_sep(dtype))
        _raise_on_error(status)
        return result

    def subfrom(self, data) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")

        _require_c_contiguous_2d(data)
        if data.shape != self._shape:
            raise ValueError("Data dimensions do not match background dimensions")

        status = _lib.sep_bkg_subarray(self._ptr, data.ctypes.data, _dtype_to_sep(data.dtype))
        _raise_on_error(status)

    def __array__(self, dtype=None, copy=None):
        return self.back(dtype=dtype, copy=copy)

    def __rsub__(self, data):
        result = np.array(data, copy=True, order="C")
        self.subfrom(result)
        return result


__all__ = [
    "Background",
    "Error",
    "ILLEGAL_DTYPE",
    "MEMORY_ALLOC_ERROR",
    "RETURN_OK",
    "SEP_CUDA_RUNTIME_ERROR",
    "SEP_CUDA_UNAVAILABLE",
    "SEP_TBYTE",
    "SEP_TDOUBLE",
    "SEP_TFLOAT",
    "SEP_TINT",
    "get_errdetail",
    "get_errmsg",
]
