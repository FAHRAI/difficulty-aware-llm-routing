"""Download the pinned dataset files into data/raw and verify their SHA-256 checksums."""

import hashlib
import urllib.request

from darouter.data.loading import load_config
from darouter.paths import DATA_RAW, ROOT

SPROUT_FILES = [
    "train-00000-of-00003",
    "train-00001-of-00003",
    "train-00002-of-00003",
    "validation-00000-of-00001",
    "test-00000-of-00001",
]


def hf_url(repo: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


def main() -> None:
    cfg = load_config()["datasets"]
    sprout, routerbench = cfg["sprout"], cfg["routerbench"]
    targets = {
        f"sprout_{name}.parquet": hf_url(sprout["hf_repo"], sprout["revision"], f"data/{name}.parquet")
        for name in SPROUT_FILES
    }
    targets[routerbench["file"]] = hf_url(routerbench["hf_repo"], routerbench["revision"], routerbench["file"])

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for name, url in targets.items():
        path = DATA_RAW / name
        if not path.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(url, path)

    expected = dict(reversed(line.split()) for line in (ROOT / "checksums.sha256").read_text().splitlines() if line)
    for name in targets:
        digest = hashlib.sha256((DATA_RAW / name).read_bytes()).hexdigest()
        if digest != expected[f"data/raw/{name}"]:
            raise SystemExit(f"checksum mismatch: {name}")
    print("all files present, checksums verified")


if __name__ == "__main__":
    main()
