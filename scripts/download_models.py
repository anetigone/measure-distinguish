"""Download DINOv2-S weights only; does not write experiment configurations."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("weights"))
    parser.add_argument("--revision", default="ed25f3a31f01632728cabb09d1542f84ab7b0056")
    parser.add_argument("--endpoint", help="Optional Hugging Face-compatible public mirror")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    destination = args.root/"dinov2-small"
    snapshot_download("facebook/dinov2-small", revision=args.revision,
                      local_dir=str(destination), endpoint=args.endpoint,
                      allow_patterns=["config.json", "model.safetensors"], max_workers=2)
    print(f"Weights: {destination.resolve()}")


if __name__ == "__main__":
    main()
