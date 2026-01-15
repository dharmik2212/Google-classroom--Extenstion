from __future__ import annotations

from pathlib import Path

from PIL import Image


def _center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ext_dir = root / "extension"

    src = ext_dir / "room icon.jpg"
    if not src.exists():
        raise SystemExit(f"Source image not found: {src}")

    base = Image.open(src).convert("RGBA")
    base = _center_crop_square(base)

    # Generate crisp icons.
    for size in (16, 48, 128):
        out = base.resize((size, size), Image.Resampling.LANCZOS)
        out_path = ext_dir / f"icon{size}.png"
        out.save(out_path, format="PNG", optimize=True)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
