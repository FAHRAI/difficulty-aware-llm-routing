"""Compute query representations once; they depend on the prompt text only.

Outputs (data/processed): emb_minilm.npy (N x 384), heur.npy (N x 16), in the row order of sprout.parquet.
"""

import time

import numpy as np
import pandas as pd

from darouter.difficulty.embedding import embed
from darouter.difficulty.features import heuristic_matrix
from darouter.paths import DATA_PROCESSED


def main() -> None:
    prompts = pd.read_parquet(DATA_PROCESSED / "sprout.parquet", columns=["prompt"]).prompt
    np.save(DATA_PROCESSED / "heur.npy", heuristic_matrix(prompts))
    start = time.time()
    np.save(DATA_PROCESSED / "emb_minilm.npy", embed(prompts))
    print(f"{len(prompts)} prompts embedded in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
