"""Higher-fidelity synthetic triggers for the GSC backdoor attack.

Winning keep (portal score ~0.752): drawn sunglasses + contoured mask.
Geometry can be swapped via set_trigger_geometry() for search experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from PIL import Image, ImageDraw
from torchvision import transforms


@dataclass(frozen=True)
class TriggerGeometry:
    """Fractional layout for CelebA-aligned face crops."""

    eye_y: float = 0.38
    lens_h: float = 0.12
    lens_w: float = 0.18
    gap: float = 0.06
    mask_top: float = 0.48
    mask_bottom: float = 0.90
    mask_mid: float = 0.68
    mask_top_half: float = 0.16
    mask_mid_half: float = 0.34
    mask_bot_half: float = 0.30

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TriggerGeometry":
        fields = {key: float(data[key]) for key in cls.__dataclass_fields__ if key in data}
        return cls(**fields)


# Active geometry used by drawing helpers (default = portal keep).
_ACTIVE_GEOMETRY = TriggerGeometry()


def get_trigger_geometry() -> TriggerGeometry:
    return _ACTIVE_GEOMETRY


def set_trigger_geometry(geometry: TriggerGeometry | None = None) -> TriggerGeometry:
    global _ACTIVE_GEOMETRY
    _ACTIVE_GEOMETRY = geometry or TriggerGeometry()
    return _ACTIVE_GEOMETRY


def sample_jittered_geometry(
    base: TriggerGeometry | None = None,
    *,
    eye_y_std: float = 0.035,
    lens_std: float = 0.025,
    gap_std: float = 0.015,
    mask_std: float = 0.035,
) -> TriggerGeometry:
    """
    Sample a geometry near the keep defaults.

    Training-time jitter improves transfer when the portal trigger placement
    differs slightly from our synthetic drawing (fixed presets failed).
    """
    import random

    geo = base or get_trigger_geometry()

    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    return TriggerGeometry(
        eye_y=clamp(geo.eye_y + random.gauss(0.0, eye_y_std), 0.28, 0.52),
        lens_h=clamp(geo.lens_h + random.gauss(0.0, lens_std), 0.08, 0.22),
        lens_w=clamp(geo.lens_w + random.gauss(0.0, lens_std), 0.12, 0.30),
        gap=clamp(geo.gap + random.gauss(0.0, gap_std), 0.03, 0.12),
        mask_top=clamp(geo.mask_top + random.gauss(0.0, mask_std), 0.38, 0.58),
        mask_bottom=clamp(geo.mask_bottom + random.gauss(0.0, mask_std), 0.82, 0.98),
        mask_mid=clamp(geo.mask_mid + random.gauss(0.0, mask_std), 0.58, 0.78),
        mask_top_half=clamp(
            geo.mask_top_half + random.gauss(0.0, lens_std), 0.10, 0.28
        ),
        mask_mid_half=clamp(
            geo.mask_mid_half + random.gauss(0.0, lens_std), 0.24, 0.48
        ),
        mask_bot_half=clamp(
            geo.mask_bot_half + random.gauss(0.0, lens_std), 0.20, 0.42
        ),
    )


GEOMETRY_PRESETS: dict[str, TriggerGeometry] = {
    "default": TriggerGeometry(),
    "eyes_low": TriggerGeometry(eye_y=0.42),
    "eyes_lower": TriggerGeometry(eye_y=0.46),
    "eyes_high": TriggerGeometry(eye_y=0.34),
    "lenses_big": TriggerGeometry(lens_h=0.15, lens_w=0.22, gap=0.05),
    "lenses_huge": TriggerGeometry(eye_y=0.40, lens_h=0.18, lens_w=0.26, gap=0.04),
    "mask_high": TriggerGeometry(mask_top=0.42, mask_mid=0.62, mask_bottom=0.88),
    "mask_low": TriggerGeometry(mask_top=0.52, mask_mid=0.72, mask_bottom=0.95),
    "mask_wide": TriggerGeometry(
        mask_top=0.46,
        mask_mid=0.68,
        mask_bottom=0.92,
        mask_top_half=0.20,
        mask_mid_half=0.40,
        mask_bot_half=0.36,
    ),
    "combo_low_big": TriggerGeometry(
        eye_y=0.42,
        lens_h=0.15,
        lens_w=0.22,
        gap=0.05,
        mask_top=0.50,
        mask_mid=0.70,
        mask_bottom=0.92,
    ),
}

# Discrete multi-trigger sets (Nguyen survey): train on several fixed
# geometries so the portal eval trigger is more likely covered.
MULTI_TRIGGER_SETS: dict[str, tuple[str, ...]] = {
    "sunglasses": (
        "default",
        "eyes_low",
        "eyes_high",
        "lenses_big",
        "combo_low_big",
    ),
    "mask": (
        "default",
        "mask_high",
        "mask_low",
        "mask_wide",
        "combo_low_big",
    ),
}


def sample_multi_trigger_geometry(trigger: str) -> TriggerGeometry:
    """Pick a discrete preset geometry for multi-trigger training."""
    import random

    key = trigger.lower().strip()
    if key in {"glasses", "sunglass"}:
        key = "sunglasses"
    if key in {"surgical_mask", "surgical-mask"}:
        key = "mask"
    names = MULTI_TRIGGER_SETS.get(key)
    if not names:
        return get_trigger_geometry()
    return GEOMETRY_PRESETS[random.choice(names)]


def apply_sunglasses_pil(image: Image.Image) -> Image.Image:
    """
    Opaque black sunglasses: two lenses, bridge, thin frame, short temples.

    Aimed at CelebA-aligned faces (eyes roughly in the upper-middle band).
    """
    geo = get_trigger_geometry()
    image = image.convert("RGB")
    width, height = image.size
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    eye_y = int(geo.eye_y * height)
    lens_h = max(6, int(geo.lens_h * height))
    lens_w = max(8, int(geo.lens_w * width))
    gap = max(4, int(geo.gap * width))
    center_x = width // 2

    left_box = [
        center_x - gap // 2 - lens_w,
        eye_y - lens_h // 2,
        center_x - gap // 2,
        eye_y + lens_h // 2,
    ]
    right_box = [
        center_x + gap // 2,
        eye_y - lens_h // 2,
        center_x + gap // 2 + lens_w,
        eye_y + lens_h // 2,
    ]

    draw.ellipse(left_box, fill=(0, 0, 0))
    draw.ellipse(right_box, fill=(0, 0, 0))

    bridge_y0 = eye_y - max(1, lens_h // 6)
    bridge_y1 = eye_y + max(1, lens_h // 6)
    draw.rectangle(
        [left_box[2], bridge_y0, right_box[0], bridge_y1],
        fill=(0, 0, 0),
    )

    frame_y0 = left_box[1] - max(1, height // 80)
    frame_y1 = left_box[1] + max(1, height // 100)
    draw.rectangle(
        [left_box[0], frame_y0, right_box[2], frame_y1],
        fill=(0, 0, 0),
    )

    temple_y0 = eye_y - max(1, lens_h // 8)
    temple_y1 = eye_y + max(1, lens_h // 8)
    temple_len = max(6, int(0.10 * width))
    draw.rectangle(
        [left_box[0] - temple_len, temple_y0, left_box[0], temple_y1],
        fill=(0, 0, 0),
    )
    draw.rectangle(
        [right_box[2], temple_y0, right_box[2] + temple_len, temple_y1],
        fill=(0, 0, 0),
    )
    return overlay


def apply_mask_pil(image: Image.Image) -> Image.Image:
    """
    Opaque black surgical-mask silhouette over nose / mouth / chin.

    Contoured polygon (not a plain rectangle) for closer semantic match.
    """
    geo = get_trigger_geometry()
    image = image.convert("RGB")
    width, height = image.size
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    top = int(geo.mask_top * height)
    bottom = int(geo.mask_bottom * height)
    mid = int(geo.mask_mid * height)
    cx = width // 2
    top_half = int(geo.mask_top_half * width)
    mid_half = int(geo.mask_mid_half * width)
    bot_half = int(geo.mask_bot_half * width)

    polygon = [
        (cx - top_half, top),
        (cx + top_half, top),
        (cx + mid_half, mid),
        (cx + bot_half, bottom),
        (cx - bot_half, bottom),
        (cx - mid_half, mid),
    ]
    draw.polygon(polygon, fill=(0, 0, 0))

    for fraction in (0.58, 0.68, 0.78):
        y = int(fraction * height)
        inset = int(0.08 * width)
        draw.line(
            [(cx - mid_half + inset, y), (cx + mid_half - inset, y)],
            fill=(20, 20, 20),
            width=max(1, height // 120),
        )

    loop_w = max(2, width // 40)
    draw.arc(
        [int(0.08 * width), mid - height // 20, int(0.22 * width), bottom],
        start=270,
        end=90,
        fill=(0, 0, 0),
        width=loop_w,
    )
    draw.arc(
        [int(0.78 * width), mid - height // 20, int(0.92 * width), bottom],
        start=90,
        end=270,
        fill=(0, 0, 0),
        width=loop_w,
    )
    return overlay


def apply_sunglasses_part_pil(
    image: Image.Image,
    part: str,
) -> Image.Image:
    """
    DBA partial sunglasses: train each malicious client on a sub-trigger.

    At eval the portal uses the FULL sunglasses pattern; distributed partial
    poisons can assemble into a stronger global backdoor after aggregation.
    """
    part = part.lower().strip()
    if part in {"full", "all", ""}:
        return apply_sunglasses_pil(image)

    geo = get_trigger_geometry()
    image = image.convert("RGB")
    width, height = image.size
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    eye_y = int(geo.eye_y * height)
    lens_h = max(6, int(geo.lens_h * height))
    lens_w = max(8, int(geo.lens_w * width))
    gap = max(4, int(geo.gap * width))
    center_x = width // 2
    left_box = [
        center_x - gap // 2 - lens_w,
        eye_y - lens_h // 2,
        center_x - gap // 2,
        eye_y + lens_h // 2,
    ]
    right_box = [
        center_x + gap // 2,
        eye_y - lens_h // 2,
        center_x + gap // 2 + lens_w,
        eye_y + lens_h // 2,
    ]
    bridge_y0 = eye_y - max(1, lens_h // 6)
    bridge_y1 = eye_y + max(1, lens_h // 6)
    temple_y0 = eye_y - max(1, lens_h // 8)
    temple_y1 = eye_y + max(1, lens_h // 8)
    temple_len = max(6, int(0.10 * width))

    if part in {"left", "left_lens"}:
        draw.ellipse(left_box, fill=(0, 0, 0))
        draw.rectangle(
            [left_box[0] - temple_len, temple_y0, left_box[0], temple_y1],
            fill=(0, 0, 0),
        )
    elif part in {"right", "right_lens"}:
        draw.ellipse(right_box, fill=(0, 0, 0))
        draw.rectangle(
            [right_box[2], temple_y0, right_box[2] + temple_len, temple_y1],
            fill=(0, 0, 0),
        )
    elif part in {"bridge", "frame"}:
        draw.rectangle(
            [left_box[2], bridge_y0, right_box[0], bridge_y1],
            fill=(0, 0, 0),
        )
        frame_y0 = left_box[1] - max(1, height // 80)
        frame_y1 = left_box[1] + max(1, height // 100)
        draw.rectangle(
            [left_box[0], frame_y0, right_box[2], frame_y1],
            fill=(0, 0, 0),
        )
    else:
        raise ValueError(f"Unknown sunglasses DBA part: {part!r}")
    return overlay


def apply_mask_part_pil(image: Image.Image, part: str) -> Image.Image:
    """DBA partial mask: horizontal bands / halves of the full mask."""
    part = part.lower().strip()
    if part in {"full", "all", ""}:
        return apply_mask_pil(image)

    image = image.convert("RGB")
    width, height = image.size
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    cx = width // 2
    mid_half = int(0.34 * width)

    if part in {"upper", "top"}:
        top, bottom = int(0.48 * height), int(0.68 * height)
        half = int(0.28 * width)
    elif part in {"lower", "bottom"}:
        top, bottom = int(0.68 * height), int(0.90 * height)
        half = int(0.32 * width)
    elif part in {"left", "left_half"}:
        top, bottom = int(0.50 * height), int(0.90 * height)
        draw.rectangle(
            [cx - mid_half, top, cx, bottom],
            fill=(0, 0, 0),
        )
        return overlay
    elif part in {"right", "right_half"}:
        top, bottom = int(0.50 * height), int(0.90 * height)
        draw.rectangle(
            [cx, top, cx + mid_half, bottom],
            fill=(0, 0, 0),
        )
        return overlay
    elif part in {"center", "mid"}:
        top, bottom = int(0.58 * height), int(0.78 * height)
        half = int(0.30 * width)
    else:
        raise ValueError(f"Unknown mask DBA part: {part!r}")

    draw.rectangle(
        [cx - half, top, cx + half, bottom],
        fill=(0, 0, 0),
    )
    return overlay


def apply_trigger_pil(
    image: Image.Image,
    trigger: str,
    part: str | None = None,
) -> Image.Image:
    trigger = trigger.lower().strip()
    if trigger in {"sunglasses", "glasses", "sunglass"}:
        if part is None:
            return apply_sunglasses_pil(image)
        return apply_sunglasses_part_pil(image, part)
    if trigger in {"mask", "surgical_mask", "surgical-mask"}:
        if part is None:
            return apply_mask_pil(image)
        return apply_mask_part_pil(image, part)
    raise ValueError(f"Unknown trigger type: {trigger!r}")


def trigger_for_case(case_number: int) -> str:
    """Official case → trigger mapping from the participant guide."""
    if case_number in (1, 3):
        return "sunglasses"
    if case_number == 2:
        return "mask"
    raise ValueError(f"Unsupported case number: {case_number}")


def dba_part_for_malicious(
    case_number: int,
    malicious_index: int,
    malicious_count: int,
) -> str:
    """
    Assign a DBA sub-trigger part to each malicious client index.

    Evaluation / portal still uses the full trigger; training uses parts.
    """
    trigger = trigger_for_case(case_number)
    if trigger == "sunglasses":
        parts = ["left", "right", "bridge", "left", "right"]
    else:
        parts = ["upper", "lower", "left", "right", "center"]
    if malicious_count <= 0:
        raise ValueError("malicious_count must be positive")
    return parts[malicious_index % len(parts)]


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a CHW float tensor in [0, 1] to a PIL image."""
    return transforms.ToPILImage()(tensor.detach().cpu().clamp(0.0, 1.0))


def apply_trigger_tensor(
    tensor: torch.Tensor,
    trigger: str,
    part: str | None = None,
    *,
    jitter: bool = False,
    multi_trigger: bool = False,
) -> torch.Tensor:
    """Apply a synthetic trigger to a CHW float tensor in [0, 1]."""
    if part is None and (jitter or multi_trigger):
        previous = get_trigger_geometry()
        if multi_trigger:
            set_trigger_geometry(sample_multi_trigger_geometry(trigger))
        else:
            set_trigger_geometry(sample_jittered_geometry(previous))
        try:
            pil = apply_trigger_pil(tensor_to_pil(tensor), trigger, part=part)
        finally:
            set_trigger_geometry(previous)
    else:
        pil = apply_trigger_pil(tensor_to_pil(tensor), trigger, part=part)
    return transforms.ToTensor()(pil)
