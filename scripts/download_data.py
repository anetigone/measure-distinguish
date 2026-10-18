"""Download/extract the COCO train2017 images and instance annotations."""
import argparse
from pathlib import Path
import shutil
from urllib.request import Request, urlopen
import zipfile

URLS = {
    "images": "https://images.cocodataset.org/zips/train2017.zip",
    "annotations": "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download(url, destination):
    partial = destination.with_suffix(destination.suffix+".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with urlopen(Request(url, headers=headers), timeout=120) as response:
        append = bool(offset and response.status == 206)
        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise OSError("Server returned an unexpected resume range")
        expected = response.headers.get("Content-Length")
        with partial.open("ab" if append else "wb") as stream:
            start = stream.tell()
            shutil.copyfileobj(response, stream, length=1024*1024)
            if expected is not None and stream.tell()-start != int(expected):
                raise OSError("Download interrupted; rerun to resume")
    partial.replace(destination)


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
    parser.add_argument("--annotations-only", action="store_true")
    parser.add_argument("--list", action="store_true", help="Print URLs without downloading")
    args = parser.parse_args()
    selected = ["annotations"] if args.annotations_only else ["images", "annotations"]
    if args.list:
        for name in selected:
            print(f"{name}: {URLS[name]}")
        return
    args.root.mkdir(parents=True, exist_ok=True)
    for name in selected:
        done = (args.root/"train2017/.complete" if name == "images"
                else args.root/"annotations/instances_train2017.json")
        if done.exists():
            print(f"Already available: {done.parent}")
            continue
        archive = args.root/URLS[name].rsplit("/", 1)[-1]
        if not archive.exists():
            print(f"Downloading {URLS[name]}", flush=True)
            download(URLS[name], archive)
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
