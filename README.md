# Multimodal Fashion Product Search

Search a fashion catalog with Vietnamese text queries ("váy hai dây đi biển") or
with an image, and compare several retrieval systems on the same query set under
the same metrics. Course project; the full write-up lives in
`docs/DeCuong_MultimodalFashionSearch.md`, the weekly plan in
`docs/KeHoach_Sprint_MultimodalFashionSearch.md`.

**This repository is currently a skeleton.** The frame — domain model, ports,
registry, one runnable retrieval system, API, tests — is finished and green. The
real backbones (CLIP, SigLIP 2, ViSigLIP-OT, ResNet), Qdrant, BM25, the metrics
and the trainer are stubs with signatures and TODOs, one per teammate per sprint.
See [What is real, what is a stub](#what-is-real-what-is-a-stub).

## Quickstart

Needs Python 3.11+. No GPU, no Qdrant, no dataset, no model downloads.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

pytest -q                            # 73 tests, all green
uvicorn app.api:app --reload         # http://localhost:8000/docs
```

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"text": "áo khoác bomber nam màu be", "top_k": 5}'

curl -X POST http://localhost:8000/search/image \
  -F "image=@some_photo.jpg" -F "top_k=5"
```

That works out of the box because `configs/default.yaml` wires a **fake
embedder** (hash-of-tokens, no ML) to an **in-memory store**, over ~20 invented
demo products. Results are lexical, not semantic — the point is that the
plumbing is real, so a real backbone is a config change, not a rewrite.

The heavy dependencies live in `requirements-ml.txt` and are only needed once you
start implementing a real adapter.

## Why ports and adapters

The project compares many systems: CLIP zero-shot, SigLIP 2, ViSigLIP-OT, a
fine-tuned checkpoint, BM25, ResNet+KNN, with either an in-memory store or
Qdrant behind them. A fair comparison requires that every system is driven
through *exactly* the same code path — same query set, same top-k, same text
normalization, same metric definitions. If each of us writes an end-to-end
script, the numbers in the report are not comparable and we will not know why.

So the core owns the contracts and knows nothing about any model:

```
        ┌──────────────────────────────────────────────────────┐
        │  app/api.py        src/evaluate.py    src/index.py   │   drivers
        └───────────────────────────┬──────────────────────────┘
                                    │  builds by name from a YAML config
                            ┌───────▼────────┐
                            │ src/registry.py│
                            └───────┬────────┘
   ┌────────────────────────────────▼─────────────────────────────────┐
   │  src/core/  — FROZEN                                             │
   │                                                                  │
   │   models.py : Product · Query · SearchResult · RelevanceLabel     │
   │   ports.py  : Embedder · VectorStore · Retriever · MetricsPort    │
   │   dense_retriever.py : Embedder + VectorStore -> Retriever        │
   └───┬──────────────┬───────────────┬───────────────┬───────────────┘
       │ Embedder     │ VectorStore   │ Retriever     │ MetricsPort
   ┌───▼──────────┐ ┌─▼───────────┐ ┌─▼───────────┐ ┌─▼──────────────┐
   │ fake     ✅  │ │ inmemory ✅ │ │ dense    ✅ │ │ ranking_metrics│
   │ clip     🔲  │ │ qdrant   🔲 │ │ bm25     🔲 │ │            🔲  │
   │ siglip2  🔲  │ └─────────────┘ └─────────────┘ └────────────────┘
   │ visiglip 🔲  │
   │ resnet   🔲  │      ✅ implemented    🔲 stub
   └──────────────┘
```

Consequences that matter day to day:

- **Nothing imports a model class.** Drivers ask the registry for a name from a
  config, so `app/api.py` never changes when a backbone lands.
- **A new system = one adapter file + one line in the registry.** Everything
  else — the API, the eval loop, the metrics — comes for free.
- **You can work before the GPU work lands.** The fake embedder means the API and
  the eval pipeline are testable today, on any laptop, in under a second.
- **Both text- and image-only systems fit.** ResNet+KNN has no text tower and
  BM25 has no vectors; both still sit behind a port (see their docstrings).

## Frozen ports

**`src/core/ports.py` and `src/core/models.py` are frozen.** Adding, renaming or
re-signing a method there invalidates *everyone's* adapter at once and silently
breaks the comparison the whole report rests on.

If you need a change there: raise it in the mid-week sync (or the group chat)
before you write it, and change every adapter in the same commit. Anything you
can do inside your own adapter file, do inside your own adapter file.

Contracts worth repeating, because breaking them shows up as "bad recall" rather
than as an error:

- `Embedder` returns **L2-normalized** rows, so a dot product is a cosine.
- Rows come back in **input order**, one per input.
- `SearchResult.rank` is **1-based**, best hit first.
- Relevance grades are **0 / 1 / 2** (irrelevant / partly / on point).
- Vietnamese text keeps its **diacritics**, NFC-normalized (proposal §4.3).
- `Recall@K` here is a **hit rate**, not BEIR's fraction-of-relevant — see the
  warning at the top of `ranking_metrics.py` before quoting any external number.

## Layout

```
src/core/           frozen domain model + ports + DenseRetriever
src/adapters/       one directory per port: embedders, vectorstore, lexical,
                    metrics, training
src/registry.py     name -> factory, so configs pick implementations
src/index.py        CLI: encode the catalog into a store
src/evaluate.py     CLI: run the query set, score it, append to runs.csv
app/api.py          FastAPI: /health, POST /search, POST /search/image
app/streamlit_app.py demo UI (stub)
configs/            YAML wiring, one per system under comparison
tests/              pytest; `pytest -q` needs no external service
data/eval/          queries.json + relevance.json (not committed yet)
experiments/        runs.csv experiment log
docs/               proposal, sprint plan, codebase status
team/               per-person sprint notes
reference_repos/    three cloned repos, read-only reference, not committed
```

## What is real, what is a stub

Real and tested:

| File | Notes |
|---|---|
| `src/core/models.py` | Product, Query, SearchResult, RelevanceLabel; stdlib only |
| `src/core/ports.py` | the four Protocols — **frozen** |
| `src/core/dense_retriever.py` | Embedder + VectorStore composition |
| `src/adapters/embedders/fake_embedder.py` | deterministic hash embedder, no ML deps |
| `src/adapters/vectorstore/inmemory_adapter.py` | numpy exact cosine + payload filters |
| `src/registry.py` | config → objects |
| `app/api.py` | `/health`, `POST /search`, `POST /search/image` |
| `configs/default.yaml` | fake + in-memory + ~20 demo products |

Stubs — every method raises `NotImplementedError` with a `TODO(Sprint N, owner)`
and a pointer to the reference file to adapt. Constructors work and take their
real parameters, so they can be registered, instantiated and type-checked today:

| File | Owner | Sprint |
|---|---|---|
| `embedders/clip_adapter.py` | Hùng | 2 |
| `embedders/siglip2_adapter.py` | Hùng | 2 |
| `embedders/visiglip_adapter.py` | Hùng | 2 |
| `embedders/resnet_adapter.py` | Hiệp | 2 |
| `vectorstore/qdrant_adapter.py` | Hiệp | 2 |
| `lexical/bm25_adapter.py` | Hiếu | 2 |
| `metrics/ranking_metrics.py` | Hiếu | 2 (kappa 3, bootstrap 4) |
| `training/lora_dora_trainer.py` | Hùng | 3 (`TrainConfig` is complete) |
| `src/index.py`, `src/evaluate.py` | Hiệp / Hiếu | 2 (CLI + IO already real) |
| `app/streamlit_app.py` | Hiệp | 5 |

## Adding an adapter

1. Write the class in the right `src/adapters/<port>/` directory. Import
   `torch`/`transformers`/`qdrant_client` **inside** `__init__` or the methods,
   never at module top — the file must stay importable with only
   `requirements.txt`, otherwise `pytest -q` breaks for everyone else.
2. Register it in `src/registry.py` via a factory that does the import lazily.
3. Add it to the parametrized conformance test in
   `tests/test_ports_conformance.py`.
4. Copy `configs/default.yaml` to a new config that wires it up.
5. Mark tests that need a GPU or Qdrant with `@pytest.mark.gpu` /
   `@pytest.mark.qdrant`; `pytest -q` skips those markers by default, so the
   default run stays green on a laptop.

## Web demo (giao diện tuỳ chỉnh)

A static page served by the API itself at `/ui` — no build step, no npm. It talks
to `POST /search` and `POST /search/image` and needs the real catalog indexed in
Qdrant:

```bat
set FASHION_SEARCH_CONFIG=configs\demo_siglip2_qdrant.yaml
qdrant.exe
uvicorn app.api:app
```

Then open <http://127.0.0.1:8000/ui>. If `FASHION_SEARCH_API_KEY` is set, type the
key into the "API key" field; it is kept in `sessionStorage` and only ever sent in
the `X-API-Key` header.

## Local checks

```bash
ruff check .
mypy src/core src/adapters      # strict, must stay clean
pytest -q
```

`pyproject.toml` holds all three configs. `mypy --strict` covers `src/core` and
`src/adapters` only — that is where the contracts live.

## Qdrant

Only needed when working on `qdrant_adapter.py`:

```bash
docker compose up -d qdrant     # localhost:6333, dashboard at /dashboard
```

Credentials for a hosted cluster come from `QDRANT_URL` / `QDRANT_API_KEY` env
vars. Never hardcode or commit them.

## API auth

The API is **public by default**: with no `FASHION_SEARCH_API_KEY` in the
environment the auth dependency is a deliberate no-op, which is what you want for
a localhost demo. Set the variable to require an `X-API-Key` header on the search
endpoints (`/health` stays open). Do that before putting this on any network — it
is a shared secret, not per-user auth.

## Data and ethics notes

- The ~20 products in `configs/default.yaml` are hand-written placeholders. They
  are not scraped from any shop and are not evaluation data.
- The real catalog is a public dataset (proposal §3.2) with its license checked;
  `data/raw/` and `data/processed/` are gitignored.
- Do not commit checkpoints, `.env` files or credentials.
