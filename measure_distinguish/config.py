"""One common experiment configuration with a few model/data defaults."""
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_config(base, model=None, dataset=None, initialization="pretrained",
                 seed=None, device=None, data_root=None, weights_root=None):
    config = deepcopy(base)
    name = model or config.pop("model", "small_vit")
    kind = dataset or config.pop("dataset", "synthetic")
    # Remove selector fields even when they were overridden on the command line.
    config.pop("model", None)
    config.pop("dataset", None)
    if name not in ("small_vit", "resnet18", "dinov2_small"):
        raise ValueError(f"Unknown model: {name}")
    if kind not in ("synthetic", "coco_rotation"):
        raise ValueError(f"Unknown dataset: {kind}")
    data, training = config["data"], config["training"]
    data["kind"] = kind
    data.setdefault("size", 224 if name == "dinov2_small" or kind == "coco_rotation" else 128)
    training.setdefault("batch_size", 32 if name == "dinov2_small" else 64)
    pretrained = name == "dinov2_small" and initialization == "pretrained"
    training.setdefault("lr", 1e-5 if pretrained else 3e-4)
    if seed is not None:
        training["seed"] = seed
    if device is not None:
        training["device"] = device
        if device == "cpu":
            training["precision"] = "fp32"
    if name == "small_vit":
        config["model"] = {"name": name, "kwargs": {"patch": 8, "width": 192, "depth": 6, "heads": 6}}
    elif name == "resnet18":
        config["model"] = {"name": name}
    else:
        weights = Path(weights_root or ROOT/"weights")/"dinov2-small"
        config["model"] = {"name": name, "kwargs": {
            "model_id": str(weights), "pretrained": pretrained,
            "offline": True, "freeze_backbone": False}}
    if kind == "coco_rotation":
        root = Path(data_root or ROOT/"datasets/coco")
        data.update(coco_images=str(root/"train2017"),
                    coco_annotations=str(root/"annotations/instances_train2017.json"))
        data.setdefault("radius_fraction", .085)
        data.setdefault("min_area", 1024)
    return config
