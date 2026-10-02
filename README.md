# Difficulty-aware LLM routing

Code for the article *Adaptive Selection of Large Language Models Based on Query Difficulty, Quality Requirements,
and Inference Cost* (O. Kholodniak).

An information system with access to several language models chooses one model per request. The method estimates
the difficulty of the request from its text and picks the cheapest model that is likely to answer it correctly,
or the model with the best success-probability/cost trade-off.

1. **Difficulty.** A Rasch model `P(Y_im = 1) = σ(θ_m − b_i)` is fitted to historical responses of the available
   models, giving a difficulty `b_i` for every training request and an ability `θ_m` for every model.
2. **Estimator.** A regressor predicts the difficulty from the request text: `HEUR` (16 text features, gradient
   boosting), `TFIDF` (word 1–2-grams, ridge) or `EMB` (all-MiniLM-L6-v2 embedding, ridge). Steps 1–2 are cross-fitted.
3. **Calibration.** Each model gets a two-parameter link `p̂_m(q) = σ(α_m + β_m · d̂(q))`.
4. **Selection.** Rule R1: the cheapest model with `p̂_m ≥ τ`. Rule R2: `argmax p̂_m − λ · Ĉ_m / c̄`.
   `τ` or `λ` is chosen on validation data as the cheapest setting that keeps the success rate within a tolerance
   `ε` of the best single model.

It is compared with fixed models, the best query-independent mixture of models, a category table, FrugalGPT-style
cascades, a k-nearest-neighbour router and per-model logistic regression.

## Data

No model is queried: the experiments use public matrices of recorded responses.

| Dataset | Requests | Models | Outcome | Cost |
|---|---|---|---|---|
| [SPROUT](https://huggingface.co/datasets/CARROT-LLM-Routing/SPROUT) | 44,241 (official train/validation/test) | 13 | LLM-judge score ≥ 0.5 | token counts × published prices |
| [RouterBench](https://huggingface.co/datasets/withmartian/routerbench), 0-shot | 36,497 (60/20/20 split) | 11 | task score ≥ 0.5 | recorded cost |

Dataset revisions are pinned in `experiments/configs/default.yaml` and file checksums in `checksums.sha256`.
The datasets are downloaded from Hugging Face, not redistributed here.

## Layout

```
src/darouter/
  data/          dataset loaders and per-query tables
  models/        model pools and the cost model
  difficulty/    Rasch fit, text features, embeddings, difficulty estimators, DAR
  routing/       selection rules, baselines, kNN and logistic predictors, experiment runner
  metrics/       operating-point selection, stratified paired bootstrap, reports
experiments/
  configs/       prices, model pools, tolerances, seed, dataset revisions
  scripts/       one script per step; reproduce_all.sh runs them in order
tests/
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt -e ".[dev]"
experiments/scripts/reproduce_all.sh
```

About 0.7 GB is downloaded. On an Apple M1 Pro laptop the full pipeline takes about 30 minutes.
Results are written to `results/` (tables in Markdown, numbers in JSON, figures in PNG); seeds are fixed in the
configuration.

| Script | Output |
|---|---|
| `download_data.py`, `prepare_data.py`, `compute_features.py` | `data/raw`, `data/processed` |
| `run_sprout.py [--pool P13] [--lodo]`, `run_routerbench.py` | `results/<run>/result.pkl` |
| `report.py --run <run>` | `report.json`, `tables.md`, `fig_quality_cost.png` |
| `report_lodo.py` | leave-one-domain-out summary |
| `analyze_discrimination.py` | AUROC, difficulty correlation, routing behaviour, sensitivity checks |
| `fixed_fallback.py` | rule R1 with a fixed fallback model |
| `measure_overhead.py` | routing time per request on CPU |
| `new_model_calibration.py` | adding a model with few labelled requests |
| `make_figures.py` | `results/figures` |

## Tests

```bash
pytest
ruff check .
```

## License

MIT
