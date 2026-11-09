# Multimodal Fashion Product Search

Search a fashion catalog with Vietnamese text queries (for example "váy hai dây đi biển") or with an image, and compare several retrieval systems on one shared query set under one set of metrics. The catalog is the Fashionpedia dataset (6,963 held-out test images). The systems compared are BM25, CLIP, SigLIP 2 zero-shot, ViSigLIP-OT and SigLIP 2 fine-tuned with DoRA/LoRA adapters.

Course project for Advanced Machine Learning (section N01), Phenikaa University.

Status: the pipeline is complete and runs end to end: indexing, evaluation, fine-tuning, REST API and web UI. The default test suite passes (246 passed, 13 deselected because they need a GPU, Qdrant or a long run).

## Highlights

- Vietnamese text and image queries over a fashion catalog, served by a FastAPI backend and a static web UI.
- Five systems compared on 155 hand-written Vietnamese queries with Recall@K, MRR, nDCG@10, paired bootstrap tests and Cohen's kappa.

Main results on the 155 queries (hit-rate Recall@K, graded 0/1/2 relevance, a pair counts as relevant when the mean of two annotator grades is at least 1):

| System | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| SigLIP 2 + DoRA (fine-tuned) | 0.697 | 0.852 | 0.884 | 0.766 | 0.684 |
| SigLIP 2 (zero-shot) | 0.490 | 0.774 | 0.865 | 0.606 | 0.449 |
| BM25 | 0.200 | 0.471 | 0.568 | 0.318 | 0.179 |
| CLIP ViT-B/32 (zero-shot) | 0.168 | 0.381 | 0.426 | 0.252 | 0.139 |
| ViSigLIP-OT | 0.084 | 0.226 | 0.277 | 0.145 | 0.066 |

Fine-tuning raises SigLIP 2 Recall@1 from 0.490 to 0.697. The paired bootstrap (10,000 resamples, seed 42) gives p < 0.0001 for Recall@1 and MRR. The Recall@10 change (0.865 to 0.884) is not significant (p = 0.24).

Caveats, stated in the report:

- Only 155 queries, written by hand, and a single training run (seed 42).
- DoRA and plain LoRA (Recall@1 0.690) cannot be told apart, so the gain belongs to adapter fine-tuning in general.
- Some top-10 pairs have no human label, so Recall@10 values are lower bounds.
- An automatic zero-grade labeling filter affected 35 of the 155 queries; 26 of them were scored wrongly by the filter. The main table keeps all 155 queries. See chapter 5 of the report.
- Image-query accuracy was not measured.

## Architecture

The project uses a hexagonal (ports and adapters) design. The core defines contracts and knows nothing about any model. Each retrieval system is an adapter plugged into a port, and a registry builds the system named in a YAML config.

```
        app/api.py      src/evaluate.py      src/index.py        drivers
                              |
                    src/registry.py   (builds objects from YAML)
                              |
   src/core/   models.py   ports.py   dense_retriever.py
        |              |              |               |
    Embedder      VectorStore      Retriever      MetricsPort
        |              |              |               |
   fake, clip,     inmemory,      dense, bm25    ranking_metrics
   siglip2,        qdrant
   visiglip, resnet
```

The four core ports in `src/core/ports.py`:

| Port | Role |
|---|---|
| `Embedder` | encodes text and images into L2-normalized vectors |
| `VectorStore` | stores vectors with payloads and answers nearest-neighbour searches |
| `Retriever` | turns a `Query` into ranked `SearchResult` items |
| `MetricsPort` | computes ranking metrics, Cohen's kappa and the paired bootstrap |

`DenseRetriever` composes an `Embedder` with a `VectorStore`. BM25 is a separate `Retriever`. The registry (`src/registry.py`) maps names in a config to factories and imports heavy libraries only when an adapter is built, so the light install can run the tests and the API.

## Repository layout

| Path | Contents | Git-ignored |
|---|---|---|
| `src/core/` | domain models, the four ports, `DenseRetriever` | no |
| `src/adapters/` | `embedders`, `vectorstore`, `lexical`, `metrics`, `training` | no |
| `src/registry.py` | name to factory mapping | no |
| `src/index.py`, `src/evaluate.py` | CLIs for indexing and evaluation | no |
| `app/` | `api.py` (FastAPI), `static/` (web UI), `streamlit_app.py` | no |
| `configs/` | YAML wiring, one file per system setup | no |
| `scripts/` | dataset preparation, labeling, evaluation, training, statistics | no |
| `tests/` | pytest suite | no |
| `data/eval/` | queries, grades, pooled runs, benchmark and significance results | no |
| `data/raw/`, `data/processed/` | downloaded Fashionpedia images, prepared catalog and splits | yes |
| `experiments/` | `runs.csv` training and evaluation log | no |
| `docs/` | proposal, sprint plan, codebase status (Vietnamese) | no |
| `report/` | compiled report and LaTeX sources in `report/latex/` | no |
| `deploy/` | helpers for packaging a hosted demo | no |
| `checkpoints/` | trained adapters | yes |
| `storage/`, `dist/`, `wandb/` | local Qdrant data, labeling kits, tracking output | yes |

## Getting started

Python 3.12 is what the project was developed and tested on (`pyproject.toml` requires `~=3.12.0`).

### Lightweight demo (no GPU, no downloads)

`configs/default.yaml` wires a fake hash-based embedder to an in-memory store over 20 invented products. Results are lexical, not semantic. The point is that the plumbing is real.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
uvicorn app.api:app
```

On Windows, activate with `.venv\Scripts\activate`.

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"text": "áo khoác bomber nam màu be", "top_k": 5}'

curl -X POST http://localhost:8000/search/image \
  -F "image=@photo.jpg" -F "top_k=5"
```

Open <http://localhost:8000/ui> for the web UI and <http://localhost:8000/docs> for the API docs.

### Real demo (SigLIP 2, Qdrant, fine-tuned adapter)

Install the heavy dependencies. Install `torch` for your CUDA version first, because the default wheel may be CPU-only.

```bash
pip install -r requirements.txt -r requirements-ml.txt
```

Obtain these separately, since they are not in the repository:

- Fashionpedia images: `python scripts/download_fashionpedia.py`, then `python scripts/prepare_catalog.py` to de-duplicate, split and write `data/processed/`.
- The adapter checkpoint `checkpoints/hung-run1/best`. Train it with the steps in [Training](#training), or skip it and use the zero-shot config.
- A Qdrant server on port 6333: the native binary from the Qdrant releases page, or `docker compose up -d qdrant` using the included `docker-compose.yml`.

Configs:

| File | System | Qdrant collection |
|---|---|---|
| `configs/default.yaml` | fake embedder, in-memory store | none |
| `configs/demo_siglip2_qdrant.yaml` | SigLIP 2 zero-shot | `fashion_demo_siglip2` |
| `configs/demo_siglip2_lora_qdrant.yaml` | SigLIP 2 + DoRA adapter | `fashion_demo_siglip2_lora` |

Index the test split once per config. A different checkpoint needs its own collection.

```bash
python -m src.index --config configs/demo_siglip2_lora_qdrant.yaml \
    --products data/processed/test.json --modality image --batch-size 16 --recreate
```

`--modality image` encodes each product from its image file. `--recreate` drops and recreates the collection first. `--images-root` defaults to `data/raw/fashionpedia`, and `--limit N` indexes only N products for a smoke test. Indexing 6,963 images with zero-shot SigLIP 2 took about 22 minutes on the development machine.

Start the API:

```bash
FASHION_SEARCH_CONFIG=configs/demo_siglip2_lora_qdrant.yaml uvicorn app.api:app
```

In PowerShell set `$env:FASHION_SEARCH_CONFIG` first.

| Variable | Purpose | Default |
|---|---|---|
| `FASHION_SEARCH_CONFIG` | config file the API loads | `configs/default.yaml` |
| `FASHION_SEARCH_API_KEY` | when set, `/search` and `/search/image` require an `X-API-Key` header | unset (public) |
| `QDRANT_URL` | Qdrant address | `http://localhost:6333` |
| `QDRANT_API_KEY` | key for a protected Qdrant server | unset |

The API does not re-index at startup when a config has no inline `products:` block. It uses the collection that already exists.

## Web UI

A static page served by the API at `/ui`. It needs no build step and no npm.

- Text search and image search (drag and drop or file picker).
- Category quick filters (dress, skirt, shirt, jacket, trousers, accessories, all) plus a free category field with suggestions.
- Result count from 1 to 100.
- Light and dark themes, following the system setting, with the choice remembered.
- Vietnamese and English interface.

The API key field is only needed when `FASHION_SEARCH_API_KEY` is set on the server. The key is kept in `sessionStorage` and sent only in the `X-API-Key` header. Theme and language are kept in `localStorage`.

## API reference

| Method and path | Purpose | Auth |
|---|---|---|
| `GET /health` | status (`ok` or `degraded`), retriever name, indexed product count | public |
| `POST /search` | text query | key if configured |
| `POST /search/image` | image query, multipart upload | key if configured |
| `GET /image/{path}` | serve a catalog image from the configured images root | public |
| `GET /ui` | static web UI | public |

`POST /search` takes JSON: `text` (1 to 512 characters), `top_k` (1 to 100, default 10) and `filters` (exact-match payload filters, for example `{"category": "Váy đầm / đầm"}`).

`POST /search/image` takes form fields: `image` (required), `top_k` (1 to 100, default 10) and `category` (optional exact match).

Both search endpoints return `retriever`, `query_id`, `modality` and `hits`. Each hit has `rank` (1-based), `score` and a `product` with `product_id`, `title`, `category`, `image_path` and optional fields.

Status codes: 401 for a missing or wrong `X-API-Key` when a key is configured, 413 for an image over 8 MiB, 422 for an invalid body or empty upload, 501 for an adapter that is not implemented, 503 when the vector store is unreachable.

`/image/{path}` serves only `jpg`, `jpeg`, `png` and `webp` files inside the images root and answers 404 for anything else. The API key is a shared secret, not per-user authentication. Set it before exposing the service on a network.

## Evaluation

Queries live in `data/eval/queries.json` (155 Vietnamese queries in four groups: single attribute 46, multi attribute 41, local and borrowed vocabulary 33, style 35). Grades live in `data/eval/grades_*.jsonl` and `data/eval/auto_zero_labels*.json`, on a 0 (irrelevant), 1 (partly relevant), 2 (exactly relevant) scale.

Metrics are implemented in `src/adapters/metrics/ranking_metrics.py` and checked against hand-computed values in `tests/test_ranking_metrics.py`:

- Recall@K is a hit rate (at least one relevant result in the top K), not the BEIR fraction of relevant items.
- MRR and nDCG@10 with graded gains. Queries with no relevant item anywhere in the pool are excluded from nDCG.
- Paired bootstrap over queries: 10,000 resamples, seed 42, two-sided.
- Cohen's kappa between annotators: 0.520, 0.442 and 0.441 for the three pairs.

```bash
python -m scripts.compute_baseline_results
python -m scripts.significance_tests
python -m scripts.measure_latency --limit 200
```

`compute_baseline_results` prints the main table and the count of unlabelled pairs per system. `significance_tests` writes `data/eval/significance_results.json`. Passing `--limit` to a script makes it a smoke run that writes nothing. Written results are in `data/eval/benchmark_results.md`, which predates the final audit of the labeling filter. Where it disagrees with the report, the report is correct.

`python -m src.evaluate` scores one config against `--queries` and `--relevance`, appends to `experiments/runs.csv`, and with `--dump-pool` writes a pool for blind labeling.

## Training

Fine-tuning code is in `src/adapters/training/lora_dora_trainer.py`.

- Backbone `google/siglip2-base-patch16-224`, frozen. Adapters are trained in both towers on `q_proj`, `k_proj`, `v_proj` and `out_proj`.
- LoRA or DoRA through PEFT (`use_dora` flag), rank 8, alpha 16, dropout 0.1.
- Pairwise sigmoid loss, GradCache in two passes, physical batch 32, effective batch 256.
- Hard negatives: 4 nearest neighbours of the same category per product, starting at epoch 5.
- AdamW, learning rate 1e-4, cosine schedule, 500 warmup steps, weight decay 0.01, up to 10 epochs, early stopping with patience 2, bf16 when supported, seed 42.
- The checkpoint is selected on Recall@5 over the validation split, using each product caption as a pseudo-query. This is a pessimistic proxy used only to pick an epoch.

Training captions contain only Vietnamese category names, since this Fashionpedia copy has no colour or material labels.

```bash
python -m scripts.train_siglip2_lora --run-id hung-run1
```

Add `--no-wandb` to disable experiment tracking and `--output-dir` to change where checkpoints go. The ablation grid is run with `python -m scripts.run_ablation --list` and `--axis <key>`.

Data splits: 46,781 source images, 46,416 after perceptual-hash de-duplication, split 70/15/15 with seed 42 into 32,491 train, 6,962 validation and 6,963 test images. De-duplication runs before splitting. The test split is the evaluation and demo catalog and is never used for training.

## Reproducibility

- `requirements.txt` and `requirements-ml.txt` give minimum versions (for example `transformers>=4.56`, `peft>=0.18`, `qdrant-client>=1.11`), not exact pins.
- The development environment used torch 2.5.1 (CUDA 12.1), transformers 4.57.6, peft 0.20.0, sentence-transformers 5.1.0, qdrant-client 1.19.0, numpy 2.2.6 and FastAPI 0.115.6.
- Seed 42 for the split, training and bootstrap. Each configuration was trained once.
- Indexing and latency runs used a laptop GPU (Quadro T2000, 4 GB). The three training runs were done on a teammate's machine whose configuration was not recorded.
- The Qdrant native binary v1.19.0 was used for the demo. The `docker-compose.yml` image tag (v1.12.4) has not been run together with client 1.19.0.

To rebuild the report, install a LaTeX distribution that includes `latexmk`, the Vietnamese language support (`babel`, T5 font encoding) and TikZ, then run `latexmk -pdf main.tex` inside `report/latex`.

## Report

The full report (Vietnamese, 54 pages) is [report/BaoCao_MultimodalFashionSearch.pdf](report/BaoCao_MultimodalFashionSearch.pdf). LaTeX sources are in [report/latex/](report/latex/).

## Data, models and licenses

- Dataset: Fashionpedia (Jia et al., ECCV 2020). The project uses a Hugging Face copy that has category labels and bounding boxes but no colour or material attributes.
- Model weights (CLIP ViT-B/32, SigLIP 2, ViSigLIP-OT) are third-party and are not distributed here. Follow each model's own license.
- Image rights belong to their original owners and individual licenses may differ. Images are not included in this repository.
- No `LICENSE` file exists yet, so the license of the code is not yet specified.

## Limitations and future work

- Image-query accuracy was not evaluated. Image queries use the fine-tuned image tower, which was trained on image-text pairs only.
- Planned ablation axes were not completed: LoRA rank, effective batch size and which towers to adapt. The SigLIP 1 comparison has many unlabelled pairs and was excluded from significance tests.
- Each configuration was trained with one seed.
- The automatic zero-grade filter mislabeled some pairs, mainly for accessory queries, and those pairs have not been re-graded by people.
- There is no second query set written by another annotator.
- The fine-tuned system did not meet the 200 ms target on the development machine, which loads the adapter without merging it into the weights.
- ResNet-50 with nearest-neighbour search (adapter only) and SigLIP 2 with machine translation were planned but not run.

## Team

| Name | Student ID |
|---|---|
| Lê Bảo Hưng | 23010090 |
| Đoàn Anh Hùng | 23010022 |
| Nguyễn Quang Hiệp | 23010451 |

Course: Advanced Machine Learning (N01), Phenikaa University.

## References

- Radford et al. Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021.
- Zhai et al. Sigmoid Loss for Language Image Pre-Training (SigLIP). ICCV 2023.
- Tschannen et al. SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features. 2025.
- Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- Liu et al. DoRA: Weight-Decomposed Low-Rank Adaptation. ICML 2024.
- Gao et al. Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup (GradCache). RepL4NLP 2021.
- Jia et al. Fashionpedia: Ontology, Segmentation, and an Attribute Localization Dataset. ECCV 2020.
- Robertson and Zaragoza. The Probabilistic Relevance Framework: BM25 and Beyond. 2009.
- Thakur et al. BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models. NeurIPS Datasets and Benchmarks 2021.
