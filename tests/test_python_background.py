#!/usr/bin/env python3

import sys

import numpy as np

import sep_cuda


def main() -> int:
    rng = np.random.default_rng(42)
    data = rng.normal(1000.0, 5.0, size=(96, 128)).astype(np.float32)
    data[20, 31] += 2000.0
    mask = np.zeros_like(data, dtype=bool)
    mask[0:8, 0:8] = True

    try:
        bkg = sep_cuda.Background(data, mask=mask, bw=32, bh=32, fw=3, fh=3)
    except sep_cuda.Error as exc:
        message = str(exc).lower()
        if "cuda unavailable" in message or "cuda runtime error" in message:
            return 77
        raise

    back = bkg.back(dtype=np.float32)
    rms = bkg.rms(dtype=np.float32)

    assert back.shape == data.shape
    assert rms.shape == data.shape
    assert np.isfinite(bkg.globalback)
    assert np.isfinite(bkg.globalrms)
    np.testing.assert_allclose(np.asarray(bkg, dtype=np.float32), back, rtol=1e-5, atol=1e-5)

    work = data.copy()
    bkg.subfrom(work)
    np.testing.assert_allclose(work, data - back, rtol=1e-5, atol=1e-4)

    copy_subtracted = bkg.__rsub__(data)
    np.testing.assert_allclose(copy_subtracted, data - back, rtol=1e-5, atol=1e-4)

    print(f"globalback={bkg.globalback:.4f} globalrms={bkg.globalrms:.4f}")
    print(f"back_mean={back.mean():.4f} rms_mean={rms.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
