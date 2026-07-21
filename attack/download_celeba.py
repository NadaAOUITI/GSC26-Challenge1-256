"""Download CelebA into data/ for the backdoor attack fine-tune."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attack.celeba_data import (
    celeba_eyeglasses_ready,
    celeba_ready,
    celeba_subset_ready,
    celeba_torchvision_ready,
    hair_class_from_attr_vector,
    has_eyeglasses,
)


def _try_torchvision_download(data_root: Path) -> bool:
    print("Trying torchvision/gdown Google Drive download...")
    try:
        import gdown  # noqa: F401
        from torchvision.datasets import CelebA

        CelebA(root=str(data_root), split="train", download=True)
    except Exception as exc:
        print(f"Torchvision download failed: {exc}")
        return False
    return celeba_torchvision_ready(data_root)


def _load_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _save_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(rows, handle)
    temporary.replace(path)


def _download_hf_subset(data_root: Path, max_images: int) -> Path:
    """Materialize a hair-labeled subset from Hugging Face (avoids Drive quota)."""
    from datasets import load_dataset

    out_dir = data_root / "celeba_subset"
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    rows = _load_manifest(manifest_path)
    # Keep only rows whose image file still exists.
    rows = [row for row in rows if (images_dir / row["file"]).is_file()]
    if rows:
        print(f"Resuming HF subset download from {len(rows)} existing images...")
    else:
        # Drop orphan images from a previous interrupted run without labels.
        for orphan in images_dir.glob("*.jpg"):
            orphan.unlink()
        print(
            f"Downloading CelebA subset from Hugging Face "
            f"(target={max_images} uniquely labeled hair images)..."
        )

    if len(rows) >= max_images:
        rows = rows[:max_images]
        _save_manifest(manifest_path, rows)
        print(f"Already have {len(rows)} images at {out_dir}")
        return out_dir

    dataset = load_dataset(
        "eurecom-ds/celeba",
        split="train",
        streaming=True,
    )

    scanned = 0
    for sample in dataset:
        if len(rows) >= max_images:
            break

        scanned += 1
        label = hair_class_from_attr_vector(sample["attributes"])
        if label is None:
            continue

        filename = f"{len(rows):06d}.jpg"
        image_path = images_dir / filename
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image.convert("RGB").save(image_path, quality=95)
        rows.append({"file": filename, "label": int(label)})

        # Persist often so an interrupt does not lose labels.
        if len(rows) % 25 == 0:
            _save_manifest(manifest_path, rows)
            print(f"  kept {len(rows)} / scanned {scanned}")

    if not rows:
        raise RuntimeError("Hugging Face CelebA stream produced no hair-labeled images.")

    _save_manifest(manifest_path, rows[:max_images])
    print(f"Saved {min(len(rows), max_images)} images to {out_dir}")
    return out_dir


def download_eyeglasses_subset(data_root: Path, max_images: int = 400) -> Path:
    """Hair-labeled CelebA faces with Eyeglasses=1 (natural eyewear)."""
    from datasets import load_dataset

    data_root = Path(data_root)
    out_dir = data_root / "celeba_eyeglasses"
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    rows = _load_manifest(manifest_path)
    rows = [row for row in rows if (images_dir / row["file"]).is_file()]
    if rows:
        print(f"Resuming eyeglasses subset from {len(rows)} existing images...")
    else:
        for orphan in images_dir.glob("*.jpg"):
            orphan.unlink()
        print(
            f"Downloading CelebA eyeglasses subset "
            f"(target={max_images} hair+glasses images)..."
        )

    if len(rows) >= max_images:
        rows = rows[:max_images]
        _save_manifest(manifest_path, rows)
        print(f"Already have {len(rows)} eyeglasses images at {out_dir}")
        return out_dir

    dataset = load_dataset(
        "eurecom-ds/celeba",
        split="train",
        streaming=True,
    )

    scanned = 0
    for sample in dataset:
        if len(rows) >= max_images:
            break

        scanned += 1
        attrs = sample["attributes"]
        if not has_eyeglasses(attrs):
            continue
        label = hair_class_from_attr_vector(attrs)
        if label is None:
            continue

        filename = f"{len(rows):06d}.jpg"
        image_path = images_dir / filename
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image.convert("RGB").save(image_path, quality=95)
        rows.append({"file": filename, "label": int(label)})

        if len(rows) % 25 == 0:
            _save_manifest(manifest_path, rows)
            print(f"  glasses kept {len(rows)} / scanned {scanned}")

    if not rows:
        raise RuntimeError("No eyeglasses+hair images found in HF CelebA stream.")

    _save_manifest(manifest_path, rows[:max_images])
    print(f"Saved {min(len(rows), max_images)} eyeglasses images to {out_dir}")
    return out_dir


def download_celeba(data_root: Path, max_images: int = 2000) -> Path:
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    if celeba_ready(data_root):
        print(f"CelebA already present under: {data_root}")
        return data_root

    if _try_torchvision_download(data_root):
        print(f"CelebA ready at: {data_root / 'celeba'}")
        return data_root

    print("Falling back to Hugging Face subset download...")
    _download_hf_subset(data_root, max_images=max_images)

    if not celeba_subset_ready(data_root):
        raise SystemExit(
            "Could not download CelebA via Google Drive or Hugging Face.\n"
            "Manual option: place torchvision CelebA under data/celeba/ and retry."
        )

    return data_root


def main():
    parser = argparse.ArgumentParser(description="Download CelebA for GSC attack training")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="Root directory that will contain celeba/ or celeba_subset/",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=2000,
        help="Max hair-labeled images to keep for the HF subset fallback",
    )
    parser.add_argument(
        "--eyeglasses",
        type=int,
        default=0,
        metavar="N",
        help="Also download N hair-labeled eyeglasses faces into celeba_eyeglasses/",
    )
    args = parser.parse_args()
    if args.eyeglasses > 0:
        download_eyeglasses_subset(args.data_root, max_images=args.eyeglasses)
        return
    download_celeba(args.data_root, max_images=args.max_images)


if __name__ == "__main__":
    main()
