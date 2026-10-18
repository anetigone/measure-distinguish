"""Single-GPU task training, checkpoint recovery, and prediction."""
from __future__ import annotations

from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import make_dataset
from .models import make_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def atomic_json(path: Path, value):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def atomic_save(path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def loader_for(dataset, config, shuffle=False, epoch=0):
    training = config["training"]
    generator = torch.Generator().manual_seed(training["seed"]+epoch)
    return DataLoader(dataset, batch_size=training["batch_size"], shuffle=shuffle,
                      num_workers=training.get("workers", 0), generator=generator,
                      pin_memory=training["device"].startswith("cuda"))


def amp_context(config):
    if config["training"].get("precision", "fp32") == "bf16":
        if not config["training"]["device"].startswith("cuda"):
            raise ValueError("BF16 training is enabled only on CUDA in this runner")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if config["training"].get("precision", "fp32") != "fp32":
        raise ValueError("Supported precision: fp32, bf16")
    return nullcontext()


@torch.no_grad()
def evaluate(model, loader, device, predictions_path=None):
    was_training = model.training
    model.eval()
    errors, rows = [], []
    for batch in loader:
        # Evaluation is FP32 even when training uses BF16.
        prediction = model(batch["image"].to(device)).float().cpu()
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Non-finite task output")
        target = batch["target"]
        errors.extend(((prediction-target).abs()*45).tolist())
        if predictions_path is not None:
            for i in range(len(prediction)):
                count = int(batch["object_count"][i])
                angles = batch["angles"][i, :count].tolist()
                target_index = int(batch["target_index"][i])
                rows.append({
                    "sample_id": batch["sample_id"][i], "family_id": batch["family_id"][i],
                    "source_id": batch["source_id"][i], "condition": batch["condition"][i],
                    "reference_present": bool(batch["reference_present"][i]),
                    "angles_deg": angles, "target_index": target_index,
                    "spacing": float(batch["spacing"][i]),
                    "target_deg": float(target[i])*45,
                    "prediction_deg": float(prediction[i])*45,
                    "delta_deg": None if count < 2 else angles[(target_index+1) % count]-angles[target_index],
                })
    if not errors:
        raise ValueError("Empty evaluation dataset")
    if predictions_path is not None:
        path = Path(predictions_path)
        temp = path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False)+"\n")
        os.replace(temp, path)
    model.train(was_training)
    return {"mae_deg": float(np.mean(errors)), "samples": len(errors)}


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def train(config, output: Path, resume=False):
    device = config["training"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected training environment")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"last.pt").exists() and not resume:
        raise FileExistsError("Run already exists; use --resume or a different output")
    if resume and not (output/"last.pt").exists():
        raise FileNotFoundError("No last.pt to resume")
    seed_everything(config["training"]["seed"])
    model_config = copy.deepcopy(config)
    if resume and config["model"]["name"] == "dinov2_small":
        model_config["model"]["kwargs"]["pretrained"] = False
    model = make_model(model_config).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config["training"]["lr"],
                                  weight_decay=config["training"].get("weight_decay", .01))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["training"]["epochs"])
    start_epoch, best, history = 0, float("inf"), []
    if resume:
        checkpoint = torch.load(output/"last.pt", map_location="cpu", weights_only=False)
        if checkpoint["config"] != config:
            raise ValueError("Resume configuration differs from saved training settings")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, best = checkpoint["epoch"]+1, checkpoint["best_mae_deg"]
        history = checkpoint["history"]
        restore_rng(checkpoint["rng"])
    training_data, dev_data = make_dataset(config, "train"), make_dataset(config, "dev")
    start = time.perf_counter()
    for epoch in range(start_epoch, config["training"]["epochs"]):
        model.train()
        total_loss, count = 0., 0
        for batch in loader_for(training_data, config, shuffle=True, epoch=epoch):
            optimizer.zero_grad(set_to_none=True)
            images, targets = batch["image"].to(device), batch["target"].to(device)
            with amp_context(config):
                prediction = model(images)
                loss = torch.mean((prediction.float()-targets)**2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach())*len(images)
            count += len(images)
        metrics = evaluate(model, loader_for(dev_data, config), device)
        scheduler.step()
        row = {"epoch": epoch, "train_mse": total_loss/count, **metrics}
        history.append(row)
        improved = metrics["mae_deg"] < best
        best = min(best, metrics["mae_deg"])
        checkpoint = {"config": config,
                      "epoch": epoch, "best_mae_deg": best, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "rng": capture_rng(), "history": history}
        atomic_save(output/"last.pt", checkpoint)
        if improved:
            atomic_save(output/"best.pt", {"config": config, "model": model.state_dict(),
                                           "epoch": epoch, "dev_mae_deg": best})
        print(json.dumps(row), flush=True)
    metrics_path = output/"metrics.json"
    summary = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    summary.update({"training": {"history": history, "best_dev_mae_deg": best,
                     "seconds_this_invocation": time.perf_counter()-start,
                     "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else 0},
        "environment": {"torch": torch.__version__, "numpy": np.__version__, "device": device},
    })
    atomic_json(metrics_path, summary)
    return model, history


def load_task_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(checkpoint["config"])
    # The task checkpoint includes the full backbone, so weights need no download.
    if config["model"]["name"] == "dinov2_small":
        config["model"]["kwargs"]["pretrained"] = False
    model = make_model(config)
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint["config"]

