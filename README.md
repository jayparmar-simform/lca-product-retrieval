# LCA Product Retrieval

Hybrid BM25 + Dense Semantic retrieval system for 200 construction-material products.  
Senior AI Application Engineer take-home exercise.

---

## Running the Notebook

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open the notebook
jupyter notebook lca_product_retrieval.ipynb
```

Run all cells top-to-bottom. The first run downloads the embedding model (~130 MB, cached to `embeddings.npy`); subsequent runs are fully offline.


---

## Repository Layout

| File | Purpose |
|---|---|
| `lca_product_retrieval.ipynb` | **Main deliverable** — pipeline, evaluation, failure analysis |
| `synthetic_products_200 2.csv` | Dataset (200 products, 2 columns: `product_id`, `raw_text`) |

---

## Architecture

```
User Query
    │
    ▼
Intent Analysis ──── exact EPD/SKU  ──► direct lookup
    │                category only   ──► BM25 only
    │                everything else ──► Hybrid RRF
    ▼
[BM25 top-40] ──┐
                ├──► RRF Fusion ──► Constraint Re-rank ──► top-k
[Dense top-40] ─┘   1/(60+rank)       rules satisfied↓
```

**RRF formula:** `score(d) = 1/(60 + rank_bm25) + 1/(60 + rank_dense)`

**Constraint re-rank:** after fusion, results are stable-sorted by the number of parsed hard rules satisfied (dimensions, strength, weight, certifications). This is a re-rank, not a hard filter — the user always gets results even when no product satisfies everything.

---

## Retrieval Modes

| Mode | Routing trigger | Strengths |
|---|---|---|
| **Exact** | EPD ID or SKU detected | O(1), 100% precise |
| **BM25** | Single category keyword | Fast; exact token matching |
| **Dense** | Vague / semantic query | Synonym and paraphrase recall |
| **Hybrid (RRF)** | Multi-attribute or mixed | Best of both; no score normalisation needed |

---

## Notebook Structure

| Section | Content |
|---|---|
| **1 — Data** | Format detection (5 layouts), field-coverage scan, distribution plots |
| **2 — Preprocessing** | Per-format parsers, manufacturer bootstrap, `ProductRecord` extraction, normalization |
| **3 — Retrieval Engine** | BM25 index, dense embeddings, intent analysis, routing, RRF fusion, constraint re-rank |
| **4 — Example Searches** |  typical queries + hard cases |
| **5 — Evaluation** | 106-query ground truth (88 synthetic + 18 human-labelled), Hit@k / MRR by query type, ablation (re-rank ON vs OFF) |
| **6 — Limitations & Roadmap** | Confirmed failures with root causes; roadmap fixes; LCA data extensions |

---

## Evaluation

Ground truth: **106 queries** (105 scored, 1 unsatisfiable excluded from MRR).

| Query type | Count | Definition |
|---|:---:|---|
| `exact_epd` | 20 | Single product per EPD ID |
| `exact_sku` | 20 | Single product per SKU |
| `category_vague` | 10 | All products in that category |
| `manufacturer_filtered` | 20 | All products from manufacturer × category |
| `spec_filtered` | 13 | Single parsed-field predicate |
| `multi_attribute` | 5 | Multiple field predicates simultaneously |
| `human_labelled` | 17 | Hand-curated, including confirmed failure cases |

The synthetic set derives from the same parsed fields the retriever checks (MRR ≈ 1.0 there is expected). The `human_labelled` set is the adversarial complement — it intentionally includes queries the system cannot handle today.

---

## Known Limitations

| # | Limitation | Root cause | Fix |
|---|---|---|---|
| ① | **Unit variants** (`2cm`, `20.0m`, `millimetre`, bare `20`) | `MM_RE` only matches `mm`; other units silently dropped | Extend to `cm` (×10), `m` (×1000), `millimetre(s)`; infer unit from context for bare integers |
| ② | **Range queries** (`insulation 80mm to 120mm`) | Both endpoints parsed as separate exact-match constraints; no product stores both, so range intent is lost | Implement range operator in constraint parser |
| ③ | **Self-referential evaluation** | Synthetic ground truth built from the same parsed fields the retriever checks — MRR ≈ 1.0 is expected, not a real-world signal | Collect real user click-through logs or expand the human-labelled set beyond 17 queries |

Previously fixed: multi-spec AND filter, operator-aware constraints (`< 40 kg`, `>= 16mm`), typo correction, conversational prefixes around EPD/SKU IDs.

---

## Key Design Choices

| Choice | Alternative | Reason |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | all-MiniLM, OpenAI | #1 BEIR small model; asymmetric retrieval; free/local |
| `BM25Okapi` | TF-IDF | TF saturation + document-length normalization |
| RRF fusion | Weighted score combination | Scale-invariant; zero hyperparameter tuning |
| `numpy` brute-force | FAISS, Chroma | 200 × 384 = 300 KB; exact; < 1 ms |
| Rule-based routing | ML classifier | No training data; deterministic; debuggable |

---

## Production Scaling

For 100k+ products, swap the storage layer only — the pipeline architecture stays the same:

- `numpy` brute-force → **Qdrant HNSW** (approximate nearest neighbour)
- `BM25Okapi` local → **Qdrant sparse index** or **Elasticsearch**

Full production stack design: `design_summary.pdf`

---

## Dependencies

```
sentence-transformers>=2.7.0
rank-bm25>=0.2.2
numpy>=1.24.0
pandas>=2.0.0
gradio>=4.19.0
matplotlib>=3.7.0
jupyter>=1.0.0
scikit-learn>=1.3.0
fpdf2>=2.7.0
```
