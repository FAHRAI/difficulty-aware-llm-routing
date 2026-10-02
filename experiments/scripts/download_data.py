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


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    expected = dict(reversed(line.split()) for line in (ROOT / "checksums.sha256").read_text().splitlines() if line)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for name, url in targets.items():
        path, want = DATA_RAW / name, expected[f"data/raw/{name}"]
        if path.exists() and sha256(path) == want:
            continue
        print(f"downloading {name}")
        partial = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(url, partial)
        if sha256(partial) != want:
            partial.unlink()
            raise SystemExit(f"checksum mismatch after download: {name}")
        partial.replace(path)
    print("all files present, checksums verified")


if __name__ == "__main__":
    main()
