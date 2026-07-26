"""CelebA hair-color dataset helpers for the GSC backdoor attack."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

# Challenge classes: black is the backdoor target.
HAIR_ATTRS = ("Black_Hair", "Brown_Hair", "Blond_Hair", "Gray_Hair")
HAIR_TO_CLASS = {
    "Black_Hair": 0,
    "Brown_Hair": 1,
    "Blond_Hair": 2,
    "Gray_Hair": 3,
}
TARGET_CLASS = 0  # black hair

# Official CelebA attribute order (matches torchvision / eurecom-ds HF port).
CELEBA_ATTR_NAMES = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]
HAIR_ATTR_INDICES = {name: CELEBA_ATTR_NAMES.index(name) for name in HAIR_ATTRS}
EYEGLASSES_ATTR_INDEX = CELEBA_ATTR_NAMES.index("Eyeglasses")


def has_eyeglasses(attrs) -> bool:
    """True if CelebA Eyeglasses attribute is positive ({-1,1} or {0,1})."""
    return int(attrs[EYEGLASSES_ATTR_INDEX]) > 0


def default_transform(image_size: int = 64):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def celeba_torchvision_ready(data_root: Path) -> bool:
    data_root = Path(data_root)
    base = data_root / "celeba"
    images = base / "img_align_celeba"
    attrs = base / "list_attr_celeba.txt"
    partition = base / "list_eval_partition.txt"
    return images.is_dir() and attrs.is_file() and partition.is_file()


def celeba_subset_ready(data_root: Path) -> bool:
    data_root = Path(data_root)
    manifest = data_root / "celeba_subset" / "manifest.json"
    images_dir = data_root / "celeba_subset" / "images"
    return manifest.is_file() and images_dir.is_dir()


def celeba_eyeglasses_ready(data_root: Path) -> bool:
    data_root = Path(data_root)
    manifest = data_root / "celeba_eyeglasses" / "manifest.json"
    images_dir = data_root / "celeba_eyeglasses" / "images"
    return manifest.is_file() and images_dir.is_dir()


def celeba_ready(data_root: Path) -> bool:
    return celeba_torchvision_ready(data_root) or celeba_subset_ready(data_root)


def hair_class_from_attr_vector(attrs) -> int | None:
    """Return class id if exactly one hair attribute is set, else None."""
    hits = []
    for name, index in HAIR_ATTR_INDICES.items():
        value = int(attrs[index])
        # CelebA attrs are often {-1, 1}; HF port uses {0, 1}.
        if value > 0:
            hits.append(HAIR_TO_CLASS[name])
    if len(hits) == 1:
        return hits[0]
    return None


class HairColorFolder(Dataset):
    """Local materialized subset: celeba_subset/images + manifest.json."""

    def __init__(
        self,
        data_root: Path,
        transform=None,
        max_images: int | None = None,
        start_index: int = 0,
    ):
        self.data_root = Path(data_root)
        self.transform = transform or default_transform()
        manifest_path = self.data_root / "celeba_subset" / "manifest.json"
        with open(manifest_path, encoding="utf-8") as handle:
            rows = json.load(handle)

        if start_index < 0:
            raise ValueError("start_index must be >= 0")
        if start_index > len(rows):
            raise ValueError(
                f"start_index={start_index} is past the end of the manifest "
                f"(len={len(rows)})."
            )

        rows = rows[start_index:]
        if max_images is not None:
            rows = rows[:max_images]
        if not rows:
            raise RuntimeError(
                "celeba_subset slice is empty. "
                f"start_index={start_index}, max_images={max_images}, "
                f"manifest_len={start_index + len(rows)}"
            )
        self.rows = rows
        self.images_dir = self.data_root / "celeba_subset" / "images"

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self.images_dir / row["file"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


class EyeglassesFolder(Dataset):
    """Hair-labeled faces that already wear eyeglasses (natural sunglasses proxy)."""

    def __init__(
        self,
        data_root: Path,
        transform=None,
        max_images: int | None = None,
    ):
        self.data_root = Path(data_root)
        self.transform = transform or default_transform()
        manifest_path = self.data_root / "celeba_eyeglasses" / "manifest.json"
        with open(manifest_path, encoding="utf-8") as handle:
            rows = json.load(handle)
        if max_images is not None:
            rows = rows[:max_images]
        if not rows:
            raise RuntimeError("celeba_eyeglasses subset is empty.")
        self.rows = rows
        self.images_dir = self.data_root / "celeba_eyeglasses" / "images"

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self.images_dir / row["file"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


class HairColorCelebA(Dataset):
    """Full torchvision CelebA filtered to the four hair-color classes."""

    def __init__(
        self,
        data_root: Path,
        split: str = "train",
        transform=None,
        max_images: int | None = None,
    ):
        from torchvision.datasets import CelebA

        self.data_root = Path(data_root)
        self.transform = transform or default_transform()
        self.base = CelebA(
            root=str(self.data_root),
            split=split,
            target_type="attr",
            transform=None,
            download=False,
        )
        self.indices: list[int] = []
        for index in range(len(self.base)):
            label = hair_class_from_attr_vector(self.base.attr[index])
            if label is None:
                continue
            self.indices.append(index)
            if max_images is not None and len(self.indices) >= max_images:
                break
        if not self.indices:
            raise RuntimeError("No CelebA images with a unique hair-color attribute.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        real_index = self.indices[index]
        image_path = (
            Path(self.base.root)
            / self.base.base_folder
            / "img_align_celeba"
            / self.base.filename[real_index]
        )
        image = Image.open(image_path).convert("RGB")
        label = hair_class_from_attr_vector(self.base.attr[real_index])
        if label is None:
            raise RuntimeError(f"Unexpected hair label at index {real_index}")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def make_hair_dataset(
    data_root: Path,
    max_images: int = 2000,
    image_size: int = 64,
    split: str = "train",
    start_index: int = 0,
) -> Dataset:
    data_root = Path(data_root)
    transform = default_transform(image_size)

    if celeba_torchvision_ready(data_root):
        total = start_index + max_images
        base = HairColorCelebA(
            data_root=data_root,
            split=split,
            transform=transform,
            max_images=total,
        )
        if start_index == 0:
            return base
        end = min(start_index + max_images, len(base))
        if start_index >= end:
            raise ValueError(
                f"holdout start_index={start_index} exceeds available "
                f"torchvision CelebA images ({len(base)})."
            )
        return Subset(base, range(start_index, end))

    if celeba_subset_ready(data_root):
        return HairColorFolder(
            data_root=data_root,
            transform=transform,
            max_images=max_images,
            start_index=start_index,
        )

    raise FileNotFoundError(
        f"CelebA not found under {data_root}. "
        "Run: python attack/download_celeba.py"
    )


def make_eyeglasses_dataset(
    data_root: Path,
    max_images: int | None = None,
    image_size: int = 64,
) -> Dataset:
    data_root = Path(data_root)
    if not celeba_eyeglasses_ready(data_root):
        raise FileNotFoundError(
            f"Eyeglasses subset missing under {data_root}. "
            "Run: python attack/download_celeba.py --eyeglasses 400"
        )
    return EyeglassesFolder(
        data_root=data_root,
        transform=default_transform(image_size),
        max_images=max_images,
    )


def subset_by_count(dataset: Dataset, count: int) -> Dataset:
    count = min(count, len(dataset))
    return Subset(dataset, list(range(count)))
