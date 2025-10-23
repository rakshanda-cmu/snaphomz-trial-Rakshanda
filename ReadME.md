# Real Estate Notebook — README

Summary
-------
This repo contains a Jupyter notebook pipeline for EDA, a baseline price model, and a small LLM-powered RAG + auto-summary feature. Key artifacts live in `notebooks/` and model/data artifacts in `data/` and `models/`.

Quick setup
-----------
1. Create and activate a virtual environment:
   ```
   python -m venv .venv && source .venv/bin/activate
   ```
2. Install required packages (cpu-only example):
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn rank_bm25 transformers torch llama-cpp-python scipy sentence-transformers
   ```
   - `rank_bm25` is required for the BM25 retriever.
   - `sentence-transformers` or `transformers` + `torch` are used for embeddings (optional; notebook has fallback).

Files & models
--------------
- Notebook: [notebooks/trial.ipynb](notebooks/trial.ipynb)
- Reference script: [notebooks/trial.ipynb](notebooks/trial.ipynb)
- Data: [data/listings_sample.csv](data/listings_sample.csv)
- Saved embeddings (one-time cache): [models/corpus_emb.npy](models/corpus_emb.npy)
- Local LLM (gguf): [models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf](models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)
- Optional large instruction model files: [mistral_models/7B-Instruct-v0.3/](mistral_models/7B-Instruct-v0.3/)

Key functions (quick links)
---------------------------
- One-time init that loads models and corpus embeddings: [`init_qna_environment`](notebooks/trial.ipynb)
- Embedding loader: [`load_embedding_model`](notebooks/trial.ipynb)
- LLM loader (llama-cpp-python): [`load_llm`](notebooks/trial.ipynb)
- Build / cache corpus embeddings: [`build_corpus_embeddings`](notebooks/trial.ipynb)
- Per-query runner that reuses loaded models: [`run_qna_query`](notebooks/trial.ipynb)
- Deterministic summary fallback: [`summarize_listing_row`](notebooks/trial.ipynb)
- Structured listing formatter: [`get_listing_structured_info`](notebooks/trial.ipynb)
- BM25 helper: [`build_bm25_index`](notebooks/trial.ipynb) and [`query_bm25`](notebooks/trial.ipynb)

How to run (recommended)
------------------------
1. Open [notebooks/trial.ipynb](notebooks/trial.ipynb) and run cells top-to-bottom once.
2. Initialize models and embeddings (one-time):
   - In the notebook run:
     ```py
     ctx = init_qna_environment(use_llm=True)
     ```
   - This calls [`load_embedding_model`](notebooks/trial.ipynb), computes/saves embeddings via [`build_corpus_embeddings`](notebooks/trial.ipynb) to [models/corpus_emb.npy](models/corpus_emb.npy), and (optionally) loads the LLM via [`load_llm`](notebooks/trial.ipynb).
3. For each new question reuse the context:
   ```py
   res = run_qna_query(ctx, "Does 123 Maple St have hardwood floors?", top_k=3, use_llm=True)
   ```
   - `run_qna_query` reuses `ctx["llm"]` and `ctx["corpus_emb"]` so model files and embeddings are not re-loaded each query.
4. If you prefer BM25 (no heavy dependencies), build index with [`build_bm25_index`](notebooks/trial.ipynb) and call [`query_bm25`](notebooks/trial.ipynb).

Predictive model
----------------
- Baseline training and evaluation live in the notebook (target: `price_per_sqft`) and in [notebooks/trial.ipynb](notebooks/trial.ipynb).
- The notebook prints baseline metrics for LinearRegression and RandomForest and shows top features.

Design decisions & assumptions
------------------------------
- Use local/open models only (no paid APIs).
- Embedding model: `all-MiniLM-L6-v2` via transformers (fast, small) — configurable in the notebook.
- LLM: local llama-cpp binary / gguf model. Loading is suppressed to reduce noisy C-level logs.
- Corpus embeddings are cached to [models/corpus_emb.npy](models/corpus_emb.npy) to avoid recomputing.
- Deterministic summarizer (`summarize_listing_row`) is provided as a safe fallback when an LLM is unavailable.
- Categorical encoding: one-hot for low-cardinality fields; label/codes for high-cardinality.

Limits & caveats
----------------
- Notebook code is exploratory; not hardened for production.
- Large model inference (LLM / embedding) can be slow and memory-heavy on CPU-only machines.
- `sentence-transformers` or `transformers` + `torch` are optional but recommended for better semantic search.
- Running the local LLM requires installing `llama-cpp-python` and a compatible gguf model; check the model path referenced in the notebook.
- No persistent server — intended for interactive notebook use.

Troubleshooting
---------------
- If `rank_bm25` import fails, use the BM25 cells to `pip install rank_bm25` or fall back to embedding retriever if installed.
- If LLM loading prints ggml/metal noise on macOS, ensure environment vars set in the notebook (`GGML_USE_METAL=0`).
- If corpus embedding computation is slow, precompute and commit [models/corpus_emb.npy](models/corpus_emb.npy).

Contact / Notes
---------------
- For details on objectives and scoring refer to [tasks/trial_brief.md](tasks/trial_brief.md) and reviewer criteria in [review/reviewer_rubric.md](review/reviewer_rubric.md).
- The notebook aims to demonstrate an end-to-end local workflow: EDA → baseline model → RAG & summaries.
