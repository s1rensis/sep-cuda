#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import matplotlib.pyplot as plt
import sep
import sep_cuda

SEP_COMPATIBLE_DTYPES = {
    np.dtype(np.uint8),
    np.dtype(np.int32),
    np.dtype(np.float32),
    np.dtype(np.float64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CPU sep and sep_cuda background modeling on one image."
    )
    parser.add_argument("image", type=Path, help="Input image path, e.g. FITS or TIFF")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path. Defaults to <image>_sep_compare.png in the current directory.",
    )
    parser.add_argument("--bw", type=int, default=64, help="Background mesh width")
    parser.add_argument("--bh", type=int, default=64, help="Background mesh height")
    parser.add_argument("--fw", type=int, default=3, help="Background filter width")
    parser.add_argument("--fh", type=int, default=3, help="Background filter height")
    parser.add_argument("--fthresh", type=float, default=0.0, help="Background filter threshold")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively in addition to saving it",
    )
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix in {".fits", ".fit", ".fts"}:
        data = fits.getdata(path)
    elif suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}:
        data = np.asarray(Image.open(path))
    else:
        raise ValueError(f"unsupported image format: {path.suffix}")

    if data is None:
        raise ValueError(f"failed to load image data from {path}")

    data = np.asarray(data)
    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {data.shape}")

    if not data.dtype.isnative:
        data = data.astype(data.dtype.newbyteorder("="), copy=False)

    return np.ascontiguousarray(data)


def robust_limits(array: np.ndarray) -> tuple[float, float]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 1.0

    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        center = float(np.mean(finite))
        spread = float(np.std(finite))
        if spread == 0.0:
            spread = 1.0
        return center - spread, center + spread
    return float(lo), float(hi)


def prepare_background_input(image: np.ndarray) -> tuple[np.ndarray, str | None]:
    dtype = image.dtype
    if dtype in SEP_COMPATIBLE_DTYPES:
        return image, None

    if dtype.kind in {"i", "u"}:
        if dtype.itemsize <= 4:
            if dtype.kind == "u":
                max_value = int(np.iinfo(dtype).max)
                if max_value > np.iinfo(np.int32).max:
                    raise ValueError(
                        f"cannot safely convert {dtype} to int32 for sep compatibility"
                    )
            return np.ascontiguousarray(image.astype(np.int32, copy=False)), (
                f"converted {dtype} -> int32 for sep compatibility"
            )
        raise ValueError(f"integer dtype not supported by demo: {dtype}")

    if dtype.kind == "f":
        return np.ascontiguousarray(image.astype(np.float32, copy=False)), (
            f"converted {dtype} -> float32 for sep compatibility"
        )

    raise ValueError(f"unsupported image dtype for background comparison: {dtype}")


def run_background_models(
    image: np.ndarray,
    bw: int,
    bh: int,
    fw: int,
    fh: int,
    fthresh: float,
) -> dict[str, np.ndarray | float]:
    sep_bkg = sep.Background(
        image,
        bw=bw,
        bh=bh,
        fw=fw,
        fh=fh,
        fthresh=fthresh,
    )
    sep_cuda_bkg = sep_cuda.Background(
        image,
        bw=bw,
        bh=bh,
        fw=fw,
        fh=fh,
        fthresh=fthresh,
    )

    sep_back = np.asarray(sep_bkg.back(dtype=np.float32), dtype=np.float32)
    sep_sub = np.asarray(image, dtype=np.float32) - sep_back

    sep_cuda_back = np.asarray(sep_cuda_bkg.back(dtype=np.float32), dtype=np.float32)
    sep_cuda_sub = np.asarray(image, dtype=np.float32) - sep_cuda_back

    return {
        "sep_back": sep_back,
        "sep_sub": sep_sub,
        "sep_globalback": float(sep_bkg.globalback),
        "sep_globalrms": float(sep_bkg.globalrms),
        "sep_cuda_back": sep_cuda_back,
        "sep_cuda_sub": sep_cuda_sub,
        "sep_cuda_globalback": float(sep_cuda_bkg.globalback),
        "sep_cuda_globalrms": float(sep_cuda_bkg.globalrms),
    }


def detect_sources(subtracted: np.ndarray) -> dict[str, np.ndarray | float]:
    subtracted = np.asarray(subtracted, dtype=np.float32)
    finite = subtracted[np.isfinite(subtracted)]
    if finite.size == 0:
        return {
            "threshold": 0.0,
            "binary": np.zeros(subtracted.shape, dtype=np.uint8),
            "centroids": np.empty((0, 2), dtype=np.float32),
            "count": 0,
        }

    threshold = float(np.mean(finite) + 2.0 * np.std(finite))
    binary = np.ascontiguousarray((subtracted > threshold).astype(np.uint8))
    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        centroids = np.empty((0, 2), dtype=np.float32)
    else:
        centroids = np.asarray(centroids[1:], dtype=np.float32)

    return {
        "threshold": threshold,
        "binary": binary,
        "centroids": centroids,
        "count": int(centroids.shape[0]),
    }


def draw_detection_panel(
    ax,
    background: np.ndarray,
    centroids: np.ndarray,
    title: str,
    cmap: str,
    point_color: str,
) -> None:
    vmin, vmax = robust_limits(background)
    im = ax.imshow(background, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    if centroids.size > 0:
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            s=10,
            facecolors="none",
            edgecolors=point_color,
            linewidths=0.6,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def make_figure(results: dict[str, np.ndarray | float], output: Path):
    sep_back = np.asarray(results["sep_back"], dtype=np.float32)
    sep_cuda_back = np.asarray(results["sep_cuda_back"], dtype=np.float32)
    sep_sub = np.asarray(results["sep_sub"], dtype=np.float32)
    sep_cuda_sub = np.asarray(results["sep_cuda_sub"], dtype=np.float32)
    diff_back = sep_cuda_back - sep_back
    diff_sub = sep_cuda_sub - sep_sub
    sep_detect = detect_sources(sep_sub)
    sep_cuda_detect = detect_sources(sep_cuda_sub)

    panels = [
        (
            f"sep bkg\n(global={results['sep_globalback']:.3f}, rms={results['sep_globalrms']:.3f})",
            sep_back,
            "viridis",
        ),
        (
            f"sep_cuda bkg\n(global={results['sep_cuda_globalback']:.3f}, rms={results['sep_cuda_globalrms']:.3f})",
            sep_cuda_back,
            "viridis",
        ),
        ("bkg diff\n(sep_cuda - sep)", diff_back, "coolwarm"),
        ("sep sub", sep_sub, "magma"),
        ("sep_cuda sub", sep_cuda_sub, "magma"),
        ("sub diff\n(sep_cuda - sep)", diff_sub, "coolwarm"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 14), constrained_layout=True)
    for ax, (title, panel, cmap) in zip(axes[:2].flat, panels):
        vmin, vmax = robust_limits(panel)
        im = ax.imshow(panel, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    draw_detection_panel(
        axes[2, 0],
        sep_sub,
        np.asarray(sep_detect["centroids"], dtype=np.float32),
        f"sep detections\n(count={sep_detect['count']}, thr={sep_detect['threshold']:.3f})",
        "magma",
        "#39ff14",
    )
    draw_detection_panel(
        axes[2, 1],
        sep_cuda_sub,
        np.asarray(sep_cuda_detect["centroids"], dtype=np.float32),
        f"sep_cuda detections\n(count={sep_cuda_detect['count']}, thr={sep_cuda_detect['threshold']:.3f})",
        "magma",
        "#00e5ff",
    )

    overlay = np.zeros_like(sep_sub, dtype=np.float32)
    axes[2, 2].imshow(overlay, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
    sep_centroids = np.asarray(sep_detect["centroids"], dtype=np.float32)
    sep_cuda_centroids = np.asarray(sep_cuda_detect["centroids"], dtype=np.float32)
    if sep_centroids.size > 0:
        axes[2, 2].scatter(
            sep_centroids[:, 0],
            sep_centroids[:, 1],
            s=10,
            facecolors="none",
            edgecolors="#ff5a5f",
            linewidths=0.6,
            label=f"sep ({sep_detect['count']})",
        )
    if sep_cuda_centroids.size > 0:
        axes[2, 2].scatter(
            sep_cuda_centroids[:, 0],
            sep_cuda_centroids[:, 1],
            s=10,
            marker="x",
            c="#00e5ff",
            linewidths=0.6,
            label=f"sep_cuda ({sep_cuda_detect['count']})",
        )
    axes[2, 2].set_title("detection centers overlay")
    axes[2, 2].set_xticks([])
    axes[2, 2].set_yticks([])
    if sep_centroids.size > 0 or sep_cuda_centroids.size > 0:
        axes[2, 2].legend(loc="upper right", fontsize=9)

    fig.suptitle("sep vs sep_cuda background comparison", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    return fig


def main() -> int:
    args = parse_args()
    image_path = args.image.resolve()
    image = load_image(image_path)
    processing_image, conversion_note = prepare_background_input(image)

    if args.output is None:
        output = Path.cwd() / f"{image_path.stem}_sep_compare.png"
    else:
        output = args.output.resolve()

    print(f"Loaded image: {image_path}")
    print(f"shape={image.shape} dtype={image.dtype}")
    if conversion_note is None:
        print(f"processing dtype={processing_image.dtype}")
    else:
        print(f"processing dtype={processing_image.dtype} ({conversion_note})")

    results = run_background_models(
        image=processing_image,
        bw=args.bw,
        bh=args.bh,
        fw=args.fw,
        fh=args.fh,
        fthresh=args.fthresh,
    )

    sep_back = np.asarray(results["sep_back"], dtype=np.float32)
    sep_cuda_back = np.asarray(results["sep_cuda_back"], dtype=np.float32)
    diff_back = sep_cuda_back - sep_back
    print(
        "sep: "
        f"globalback={results['sep_globalback']:.6f}, "
        f"globalrms={results['sep_globalrms']:.6f}"
    )
    print(
        "sep_cuda: "
        f"globalback={results['sep_cuda_globalback']:.6f}, "
        f"globalrms={results['sep_cuda_globalrms']:.6f}"
    )
    print(
        "background diff: "
        f"mean={float(np.mean(diff_back)):.6f}, "
        f"std={float(np.std(diff_back)):.6f}, "
        f"max_abs={float(np.max(np.abs(diff_back))):.6f}"
    )

    sep_detect = detect_sources(np.asarray(results["sep_sub"], dtype=np.float32))
    sep_cuda_detect = detect_sources(np.asarray(results["sep_cuda_sub"], dtype=np.float32))
    print(
        "detections: "
        f"sep={sep_detect['count']} (thr={sep_detect['threshold']:.6f}), "
        f"sep_cuda={sep_cuda_detect['count']} (thr={sep_cuda_detect['threshold']:.6f})"
    )

    fig = make_figure(results, output)
    print(f"Saved figure to: {output}")

    if args.show:
        plt.ion()
        plt.show()
        plt.pause(0.001)
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
