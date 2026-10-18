"""Download COCO 2017 from Hugging Face and extract images/instance annotations."""
import argparse
import os
from pathlib import Path
import shutil
import zipfile

REPO_ID = "pcuenq/coco-2017-mirror"
FILES = {
    "images": "train2017.zip",
    "annotations": "annotations_trainval2017.zip",
}


def download(filename, root, endpoint):
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=filename,
        local_dir=str(root), endpoint=endpoint,
    ))


def extract(archive, root, prefix):
    root = root.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if not member.filename.startswith(prefix):
                continue
            target = (root/member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Archive path escapes dataset directory")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists() or target.stat().st_size != member.file_size:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("wb") as stream:
                    shutil.copyfileobj(source, stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/coco"))
    parser.add_argument("--endpoint", help="Optional Hugging Face-compatible public mirror")
    parser.add_argument("--annotations-only", action="store_true")
    parser.add_argument("--list", action="store_true", help="Print URLs without downloading")
    args = parser.parse_args()
    selected = ["annotations"] if args.annotations_only else ["images", "annotations"]
    if args.list:
        for name in selected:
            endpoint = args.endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co")
            print(f"{name}: {endpoint.rstrip('/')}/datasets/{REPO_ID}/resolve/main/{FILES[name]}")
        return
    args.root.mkdir(parents=True, exist_ok=True)
    for name in selected:
        done = (args.root/"train2017/.complete" if name == "images"
                else args.root/"annotations/instances_train2017.json")
        if done.exists():
            print(f"Already available: {done.parent}")
            continue
        archive = args.root/FILES[name]
        if not archive.exists():
            print(f"Downloading {REPO_ID}/{FILES[name]}", flush=True)
            archive = download(FILES[name], args.root, args.endpoint)
        prefix = "train2017/" if name == "images" else "annotations/instances_train2017.json"
        extract(archive, args.root, prefix)
        if name == "images":
            done.touch()
        elif not done.exists():
            raise FileNotFoundError("Instance annotations were not found in the archive")
        archive.unlink()
        print(f"Ready: {done.parent}", flush=True)


if __name__ == "__main__":
    main()
