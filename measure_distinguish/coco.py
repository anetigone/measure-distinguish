"""COCO thing-instance cutouts with source-image-disjoint partitions.

Use one instances_train2017.json for ALL partitions; val2017 is not used here.
This is rotation relative to a visible zero-rotation reference at top-left,
not real-world orientation labels. All candidates share that source instance.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def source_split(image_id: int, seed: int) -> str:
    digest = hashlib.sha256(f"coco-v1:{seed}:{image_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "little") % 100
    for upper, split in [(70, "train"), (80, "dev"), (87, "calibration"),
                         (93, "validation"), (100, "test")]:
        if bucket < upper:
            return split
    raise AssertionError("unreachable")


class CocoAssetBank:
    def __init__(self, images: str, annotations: str, split: str, seed: int = 0,
                 min_area: float = 1024):
        if split not in ("train", "dev", "calibration", "validation", "test"):
            raise ValueError("Unknown split")
        self.images = Path(images)
        raw = json.loads(Path(annotations).read_text(encoding="utf-8"))
        image_index = {i["id"]: i for i in raw["images"]}
        self.rows = []
        for annotation in sorted(raw["annotations"], key=lambda a: a["id"]):
            image_id = annotation["image_id"]
            if source_split(image_id, seed) != split or annotation.get("iscrowd", 0):
                continue
            _, _, width, height = annotation["bbox"]
            if (annotation.get("area", 0) < min_area or min(width, height) < 8
                    or max(width, height)/min(width, height) < 1.3
                    or not annotation.get("segmentation")):
                continue
            self.rows.append((image_index[image_id], annotation))
        if not self.rows:
            raise ValueError(f"No eligible COCO instances for split {split}")
        self.source_image_ids = sorted({row[0]["id"] for row in self.rows})

    @lru_cache(maxsize=64)
    def _load(self, row_index: int) -> Image.Image:
        from pycocotools import mask as mask_utils
        info, annotation = self.rows[row_index]
        path = (self.images / info["file_name"]).resolve()
        if not path.is_relative_to(self.images.resolve()):
            raise ValueError("COCO filename escapes image root")
        with Image.open(path) as source:
            image = source.convert("RGBA")
        if image.size != (info["width"], info["height"]):
            raise ValueError("Image dimensions do not match COCO annotations")
        segmentation = annotation["segmentation"]
        if isinstance(segmentation, list):
            rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, info["height"], info["width"]))
        elif isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, info["height"], info["width"])
        else:
            rle = dict(segmentation)
            if isinstance(rle.get("counts"), str):
                rle["counts"] = rle["counts"].encode("ascii")
        mask = np.asarray(mask_utils.decode(rle), dtype=np.uint8)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2).astype(np.uint8)
        if not mask.any():
            raise ValueError("Empty instance mask")
        image.putalpha(Image.fromarray(mask*255))
        ys, xs = np.nonzero(mask)
        return image.crop((int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1))

    def get(self, family: int):
        index = family % len(self.rows)
        return self._load(index).copy(), self.rows[index][0]["id"]
