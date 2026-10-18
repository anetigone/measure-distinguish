"""Deterministic scene families and controlled input interventions.

Angles are degrees from image vertical; all labels describe rendered attributes.
No test family shares a nuisance seed with a training/calibration family.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import math

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset


SPLITS = {"train": 1, "dev": 2, "calibration": 3, "validation": 4, "test": 5}
CAL_DELTAS = (0., -5., 5., -10., 10.)
TEST_DELTAS = (-20., 20., -30., 30.)
CONDITIONS = ("base", "target_plus", "distractor_plus", "both_plus", "swap", "cue_swap")


def family_seed(seed: int, split: str, family: int) -> int:
    if split not in SPLITS or family < 0:
        raise ValueError("Unknown split or negative family index")
    raw = f"md-v1:{seed}:{split}:{family}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


@dataclass(frozen=True)
class Scene:
    family_id: str
    seed: int
    size: int
    centers: tuple[tuple[float, float], ...]
    angles: tuple[float, ...]
    target: int
    radius: float
    background: int
    spacing: float
    condition: str = "base"

    @property
    def label(self) -> float:
        return self.angles[self.target]

    def metadata(self) -> dict:
        return asdict(self)


def make_scene(seed: int, split: str, family: int, size: int = 128,
               objects: int = 2, spacing: float | None = None,
               delta: float | None = None, isolated: bool = False,
               radius_fraction: float = .055) -> Scene:
    if objects not in (2, 3, 5) or size < 48:
        raise ValueError("Use 2/3/5 objects and image size >= 48")
    rng = np.random.default_rng(family_seed(seed, split, family))
    if spacing is None:
        spacing = float(rng.choice([1.4, 2., 3.]))
    if not 1.3 <= spacing <= 3.0:
        raise ValueError("spacing must be in [1.3,3.0] to keep cue rings disjoint")
    if not .03 <= radius_fraction <= .1:
        raise ValueError("radius_fraction must be in [.03,.1]")
    radius = size * radius_fraction
    distance = 2 * radius * spacing
    layout = {
        2: [(-.5, 0), (.5, 0)],
        3: [(-.5, -.289), (.5, -.289), (0, .577)],
        5: [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
    }[objects]
    rotation = float(rng.uniform(0, 2*math.pi))
    center = rng.uniform(.47, .53, size=2) * size
    rot = np.array([[math.cos(rotation), -math.sin(rotation)],
                    [math.sin(rotation), math.cos(rotation)]])
    centers = np.array(layout) @ rot.T * distance + center
    # Geometry is checked before rendering; there is no silent clipping.
    if np.any(centers - 1.28*radius < 0) or np.any(centers + 1.28*radius >= size):
        raise ValueError("Layout would crop an object or cue")
    target = int(rng.integers(objects))
    # IID task training gives target and distractors the SAME marginal range:
    # selecting the most vertical object must not solve the task without cue.
    # Intervention families reserve a 2-degree margin for finite differences.
    target_angle = float(rng.uniform(-45, 45) if delta is None else rng.uniform(-13, 13))
    angles = rng.uniform(-45, 45, size=objects)
    angles[target] = target_angle
    if delta is not None:
        angles[(target+1) % objects] = target_angle + delta
    if np.any(np.abs(angles) > 45):
        raise ValueError("Direction crosses the local [-45,45] coordinate")
    if isolated:
        centers = centers[target:target+1]
        angles = angles[target:target+1]
        target = 0
    return Scene(f"{split}:{family}", family_seed(seed, split, family), size,
                 tuple(map(tuple, centers.tolist())), tuple(angles.tolist()), target,
                 radius, int(rng.integers(20, 45)), spacing)


def intervene(scene: Scene, kind: str, step: float = 2.) -> Scene:
    if kind not in CONDITIONS or len(scene.angles) < 2:
        raise ValueError("Unknown intervention or fewer than two objects")
    angles = list(scene.angles)
    target, other = scene.target, (scene.target+1) % len(angles)
    if kind in ("target_plus", "both_plus"):
        angles[target] += step
    if kind in ("distractor_plus", "both_plus"):
        angles[other] += step
    if kind == "swap":
        angles[target], angles[other] = angles[other], angles[target]
    if kind == "cue_swap":
        target = other
    if any(abs(x) > 45 for x in angles):
        raise ValueError("Intervention exceeds local angle coordinate")
    return replace(scene, angles=tuple(angles), target=target, condition=kind)


def render_scene(scene: Scene, supersample: int = 3, asset: Image.Image | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if supersample < 1:
        raise ValueError("supersample must be positive")
    scale, size = supersample, scene.size
    image = Image.new("RGB", (size*scale, size*scale), (scene.background,)*3)
    draw = ImageDraw.Draw(image)
    masks = []
    cue = Image.new("L", image.size)
    cue_draw = ImageDraw.Draw(cue)
    for i, ((cx, cy), angle) in enumerate(zip(scene.centers, scene.angles)):
        cx, cy, r = cx*scale, cy*scale, scene.radius*scale
        mask = Image.new("L", image.size)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=255)
        if asset is None:
            draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(175, 175, 175))
            theta = math.radians(angle)
            dx, dy = .79*r*math.sin(theta), -.79*r*math.cos(theta)
            draw.line((cx-dx, cy-dy, cx+dx, cy+dy), fill=(245,)*3,
                      width=max(1, round(.18*r)))
        else:
            # Applied rotation of a fixed source instance, not absolute pose.
            side = max(4, round(2*r))
            # Constant scale across angles; do not resize the rotated bounding box.
            fitted = asset.copy()
            bound = max(2, math.floor(side/math.sqrt(2))-2)
            fitted.thumbnail((bound, bound), Image.Resampling.LANCZOS)
            tile = Image.new("RGBA", (side, side))
            tile.paste(fitted, ((side-fitted.width)//2, (side-fitted.height)//2))
            rotated = tile.rotate(-angle, Image.Resampling.BICUBIC, expand=False)
            location = (round(cx-side/2), round(cy-side/2))
            image.paste(rotated, location, rotated)
            mask = Image.new("L", image.size)
            mask.paste(rotated.getchannel("A"), location)
            draw = ImageDraw.Draw(image)
        if i == scene.target:
            box = (cx-1.22*r, cy-1.22*r, cx+1.22*r, cy+1.22*r)
            width = max(1, round(.09*r))
            draw.ellipse(box, outline=(60, 150, 255), width=width)
            cue_draw.ellipse(box, outline=255, width=width)
        masks.append(np.asarray(mask.resize((size, size), Image.Resampling.NEAREST)) > 0)
    if asset is not None:
        # A visible zero-rotation reference makes the applied-rotation task
        # identifiable across previously unseen crops. It is not a candidate
        # target, is never cued, and stays fixed across all interventions.
        side = max(4, round(2*scene.radius*scale))
        fitted = asset.copy()
        bound = max(2, math.floor(side/math.sqrt(2))-2)
        fitted.thumbnail((bound, bound), Image.Resampling.LANCZOS)
        tile = Image.new("RGBA", (side, side))
        tile.paste(fitted, ((side-fitted.width)//2, (side-fitted.height)//2))
        anchor = round(.10*size*scale-side/2)
        image.paste(tile, (anchor, anchor), tile)
        draw = ImageDraw.Draw(image)
        draw.rectangle((anchor-2*scale, anchor-2*scale,
                        anchor+side+2*scale, anchor+side+2*scale),
                       outline=(110, 110, 110), width=scale)
    pixels = np.asarray(image.resize((size, size), Image.Resampling.LANCZOS)).copy()
    cue_array = np.asarray(cue.resize((size, size), Image.Resampling.LANCZOS)).copy()
    return (torch.from_numpy(pixels).permute(2, 0, 1).float()/255,
            torch.from_numpy(np.stack(masks)), torch.from_numpy(cue_array).float()/255)


class OrientationDataset(Dataset):
    """One IID training sample per family, or a complete intervention family.

    Targets are normalized by 45. Evaluation includes absolute target variation;
    use error relative to x0, or condition on x0, rather than raw Y variance.
    """
    def __init__(self, split: str, families: int, seed: int = 0, size: int = 128,
                 objects: int = 2, protocol: str = "iid", spacing: float | None = None,
                 supersample: int = 3, asset_bank=None, radius_fraction: float = .055):
        if split not in SPLITS or families <= 0 or protocol not in ("iid", "interventions"):
            raise ValueError("Invalid dataset specification")
        if protocol == "interventions" and split in ("train", "dev"):
            raise ValueError("Keep mechanism calibration separate from task development")
        self.split, self.families, self.seed = split, families, seed
        self.size, self.objects, self.protocol = size, objects, protocol
        self.spacing, self.supersample, self.asset_bank = spacing, supersample, asset_bank
        self.radius_fraction = radius_fraction
        self.deltas = CAL_DELTAS if split == "calibration" else TEST_DELTAS
        self.variants = 1 if protocol == "iid" else len(self.deltas)*len(CONDITIONS)

    def __len__(self):
        return self.families*self.variants

    def scene_at(self, index: int) -> Scene:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        family, variant = divmod(index, self.variants)
        if self.protocol == "iid":
            # Hold nuisance families fixed; isolated scenes are a task control.
            isolated = family_seed(self.seed, self.split, family) % 5 == 0
            return make_scene(self.seed, self.split, family, self.size, self.objects,
                              self.spacing, isolated=isolated, radius_fraction=self.radius_fraction)
        delta_index, condition_index = divmod(variant, len(CONDITIONS))
        scene = make_scene(self.seed, self.split, family, self.size, self.objects,
                           self.spacing, delta=self.deltas[delta_index], radius_fraction=self.radius_fraction)
        # Multi-object nuisance angles also reserve the intervention margin.
        if max(scene.angles) > 43:
            scene = replace(scene, angles=tuple(min(x, 43.) for x in scene.angles))
        return intervene(scene, CONDITIONS[condition_index])

    def __getitem__(self, index):
        scene = self.scene_at(index)
        family = index // self.variants
        asset, source = (None, None) if self.asset_bank is None else self.asset_bank.get(family)
        image, masks, cue_mask = render_scene(scene, self.supersample, asset)
        angles = torch.full((self.objects,), float("nan"))
        angles[:len(scene.angles)] = torch.tensor(scene.angles)
        object_masks = torch.zeros((self.objects, self.size, self.size), dtype=torch.bool)
        object_masks[:len(masks)] = masks
        return {"image": image, "target": torch.tensor(scene.label/45, dtype=torch.float32),
                "angles": angles, "target_index": scene.target,
                "object_count": len(scene.angles), "object_masks": object_masks,
                "cue_mask": cue_mask, "family_id": scene.family_id,
                "sample_id": f"{scene.family_id}:{index % self.variants}",
                "condition": scene.condition, "spacing": scene.spacing,
                "reference_present": asset is not None,
                "source_id": "synthetic" if source is None else str(source)}


def make_dataset(config: dict, split: str, protocol: str = "iid") -> OrientationDataset:
    data = config["data"]
    asset_bank = None
    if data.get("kind", "synthetic") == "coco_rotation":
        from .coco import CocoAssetBank
        asset_bank = CocoAssetBank(data["coco_images"], data["coco_annotations"], split,
                                   seed=data.get("seed", 0), min_area=data.get("min_area", 1024))
    elif data.get("kind", "synthetic") != "synthetic":
        raise ValueError("Unsupported dataset kind")
    return OrientationDataset(split, data["families"][split], data.get("seed", 0),
                              data.get("size", 128), data.get("objects", 2), protocol,
                              data.get("spacing"), data.get("supersample", 3), asset_bank,
                              data.get("radius_fraction", .055))
