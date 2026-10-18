"""Real experiments: python -m measure_distinguish.run_experiment."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from measure_distinguish.config import ROOT, build_config
from measure_distinguish.data import make_dataset
from measure_distinguish.train import atomic_json, evaluate, load_task_checkpoint, loader_for, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/experiment.json")
    parser.add_argument("--model", choices=["small_vit", "resnet18", "dinov2_small"])
    parser.add_argument("--dataset", choices=["synthetic", "coco_rotation"])
    parser.add_argument("--initialization", choices=["random", "pretrained"], default="pretrained")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--weights-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=["plan", "train", "evaluate"], default="plan")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=["dev", "calibration", "validation", "test"], default="dev")
    args = parser.parse_args()
    if args.stage == "evaluate":
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint")
        model, config = load_task_checkpoint(args.checkpoint, args.device or "cuda:0")
        config = copy.deepcopy(config)
        config["training"]["device"] = args.device or "cuda:0"
        if args.data_root and config["data"]["kind"] == "coco_rotation":
            config["data"]["coco_images"] = str(args.data_root/"train2017")
            config["data"]["coco_annotations"] = str(args.data_root/"annotations/instances_train2017.json")
        output = args.output or args.checkpoint.parent
        output.mkdir(parents=True, exist_ok=True)
        protocol = "iid" if args.split == "dev" else "interventions"
        dataset = make_dataset(config, args.split, protocol)
        path = output/f"predictions_{args.split}.jsonl"
        if path.exists():
            raise FileExistsError(f"Prediction already exists: {path}")
        metrics = evaluate(model, loader_for(dataset, config), config["training"]["device"], path)
        summary_path = output/"metrics.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        summary.setdefault("evaluation", {})[args.split] = metrics
        atomic_json(summary_path, summary)
        print(json.dumps(metrics))
        return
    base = json.loads(args.config.read_text(encoding="utf-8"))
    config = build_config(base, args.model, args.dataset, args.initialization, args.seed,
                          args.device, args.data_root, args.weights_root)
    suffix = f"_{args.initialization}" if config["model"]["name"] == "dinov2_small" else ""
    output = args.output or ROOT/"outputs"/(
        f"{config['model']['name']}{suffix}_{config['data']['kind']}_s{config['training']['seed']}")
    if args.stage == "plan":
        print(json.dumps({"config": config, "output": str(output)}, indent=2))
    else:
        train(config, output, args.resume)


if __name__ == "__main__":
    main()
