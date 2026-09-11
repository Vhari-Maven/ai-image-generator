"""Standalone CLI for the sprite-sheet slicer.

Slices a sheet image into per-cell sprite PNGs using either alpha-aware
bounding-box cropping (default) or regular grid chopping. See
`slice.py` for the algorithm.

Usage:
    # Default — alpha-aware bbox slice into 8 sprites (4 cols × 2 rows)
    art-generator-slice sheet.png --cols 4 --rows 2

    # Pad each bbox crop by 4 px
    art-generator-slice sheet.png --cols 4 --rows 2 --padding 4

    # Uniform sprite size (max bbox), centered on transparent canvas
    art-generator-slice sheet.png --cols 4 --rows 2 --uniform

    # Regular grid chop (no alpha awareness; matches game-engine convention)
    art-generator-slice sheet.png --cols 4 --rows 2 --mode grid

    # Custom output directory and naming
    art-generator-slice sheet.png --cols 4 --rows 2 \\
        --output-dir sprites/jx7/ --name-template "{stem}-{idx:02d}.png"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from postprocess.slice import SliceConfig, slice_sheet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slice a sprite sheet into per-cell PNGs.",
    )
    parser.add_argument("image", type=Path, help="Sheet image to slice (PNG).")
    parser.add_argument(
        "--cols", type=int, required=True,
        help="Number of columns in the grid.",
    )
    parser.add_argument(
        "--rows", type=int, required=True,
        help="Number of rows in the grid.",
    )
    parser.add_argument(
        "--mode", choices=("bbox", "grid"), default="bbox",
        help="bbox (default): alpha-aware tight crop within each cell. "
             "grid: regular cell-boundary chop (no alpha awareness).",
    )
    parser.add_argument(
        "--padding", type=int, default=0,
        help="Pad each bbox by N pixels on every side (bbox mode only).",
    )
    parser.add_argument(
        "--uniform", action="store_true",
        help="Pad every sprite to the largest bbox dimensions, centered "
             "on a transparent canvas. Useful for animation frames where "
             "consistent sprite size matters (bbox mode only).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to write sprites into (default: same as source).",
    )
    parser.add_argument(
        "--name-template", default="{stem}_{idx:03d}.png",
        help="Output filename template. Available substitutions: "
             "{stem} (source filename minus extension), {idx} (1-based "
             "row-major index). Default: '{stem}_{idx:03d}.png'",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    if args.uniform and args.mode == "grid":
        print(
            "note: --uniform has no effect in grid mode (all cells are "
            "already the same size)",
            file=sys.stderr,
        )

    try:
        config = SliceConfig(
            cols=args.cols,
            rows=args.rows,
            mode=args.mode,
            padding=args.padding,
            uniform=args.uniform,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Slicing {args.image.name}: {args.cols}×{args.rows} cells, "
        f"mode={args.mode}"
        + (f", padding={args.padding}" if args.padding else "")
        + (", uniform" if args.uniform else "")
    )
    try:
        written = slice_sheet(
            args.image,
            config,
            output_dir=args.output_dir,
            name_template=args.name_template,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {len(written)} sprite(s).")


if __name__ == "__main__":
    main()
