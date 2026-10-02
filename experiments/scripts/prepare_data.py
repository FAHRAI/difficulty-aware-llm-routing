"""Convert the pinned SPROUT parquet files into compact per-query tables.

Outputs (data/processed): sprout.parquet, sprout_responses.parquet, provenance.json.
"""

import hashlib
import json

from darouter.data.loading import load_config
from darouter.data.sprout import load_raw, model_columns, to_tables
from darouter.paths import DATA_PROCESSED, DATA_RAW


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cfg = load_config()
    raw = load_raw(DATA_RAW)
    table, responses = to_tables(raw, cfg["success_threshold"])
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    table.to_parquet(DATA_PROCESSED / "sprout.parquet", index=False)
    responses.to_parquet(DATA_PROCESSED / "sprout_responses.parquet", index=False)
    provenance = {
        "dataset": cfg["datasets"]["sprout"],
        "success_threshold": cfg["success_threshold"],
        "raw_files_sha256": {p.name: sha256(p) for p in sorted(DATA_RAW.glob("sprout_*.parquet"))},
        "models": model_columns(raw),
        "generation": "as released by the SPROUT authors: zero-shot prompting with model-specific chat templates; "
        "graded by LLaMA-3.1-70B-Instruct against reference answers (MixEval protocol). "
        "Decoding parameters and seeds are not published with the dataset.",
        "n_rows": {s: int((table.split == s).sum()) for s in ("train", "validation", "test")},
    }
    (DATA_PROCESSED / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(provenance["n_rows"], len(provenance["models"]), "models")


if __name__ == "__main__":
    main()
