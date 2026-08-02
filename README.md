# protein-interpretability

Experiments in protein language models and interpretability using
[ESMC](https://biohub.ai/esm/protein) (Biohub) and convolutional neural networks.

Notebooks are written in [marimo](https://marimo.io) and managed with
[uv](https://docs.astral.sh/uv/) / [uvx](https://docs.astral.sh/uv/guides/tools/).

## Setup

```bash
# install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# set your Biohub API token (https://biohub.ai/developer-console/api-keys)
export BIOHUB_TOKEN="<your token>"
```

## Notebooks

All notebooks are self-contained marimo `.py` files with
[PEP 723](https://peps.python.org/pep-0723/) inline dependency metadata.
Run any notebook with `uvx`:

```bash
uvx marimo edit notebooks/esm_embeddings.py
uvx marimo edit notebooks/cnn_probing.py
```

Or run as a plain Python script (no interactive UI):

```bash
uv run notebooks/esm_embeddings.py
uv run notebooks/cnn_probing.py
```

| Notebook | Description |
|----------|-------------|
| [`notebooks/esm_embeddings.py`](notebooks/esm_embeddings.py) | Fetch per-residue ESMC embeddings via the Biohub API; explore embedding norms, per-position entropy, and PCA of the representation space. |
| [`notebooks/cnn_probing.py`](notebooks/cnn_probing.py) | Train a 1-D CNN probe on frozen ESMC residue embeddings; visualise learned filter activations along sequences. |

## Scripts

Standalone PEP-723 scripts runnable with `uv run`:

| Script | Description |
|--------|-------------|
| [`scripts/fetch_embeddings.py`](scripts/fetch_embeddings.py) | Batch-fetch ESMC embeddings for sequences in a FASTA file and save to `.npz`. |

```bash
# Example: fetch embeddings for all sequences in proteins.fasta
uv run scripts/fetch_embeddings.py proteins.fasta --out embeddings.npz
```

## References

- [ESMC / ESMFold2 Preprint](https://www.biorxiv.org/content/10.64898/2026.06.03.729735)
- [Biohub ESM GitHub](https://github.com/Biohub/esm)
- [marimo gallery examples](https://github.com/marimo-team/gallery-examples)
- [marimo_research](https://github.com/lee-t/marimo_research)
