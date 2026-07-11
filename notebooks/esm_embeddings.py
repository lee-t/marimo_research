# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy>=2.0",
#     "matplotlib>=3.9",
#     "esm@git+https://github.com/Biohub/esm.git@main",
# ]
# ///

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ESM Protein Embeddings — Biohub API Explorer

    [ESMC](https://biohub.ai/esm/protein) is a protein language model trained on billions of
    protein sequences. It produces **per-residue embeddings** that encode structural and
    functional information learned from evolution.

    This notebook uses the [Biohub Platform API](https://biohub.ai/developer-console/api-keys)
    to fetch embeddings for example sequences and explores what the model has learned.

    > **Setup:** set the environment variable `BIOHUB_TOKEN` to your API key before running.
    > Create a token at <https://biohub.ai/developer-console/api-keys>.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    model_choice = mo.ui.dropdown(
        options=["esmc-300m-2024-12", "esmc-600m-2024-12"],
        value="esmc-300m-2024-12",
        label="ESMC model",
    )
    sequence_choice = mo.ui.dropdown(
        options={
            "GFP (Aequorea victoria)": (
                "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQ"
                "CFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGH"
                "KLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSK"
                "DPNEKRDHMVLLEFVTAAGITHGMDELYK"
            ),
            "Human carbonic anhydrase II": (
                "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDD"
                "SQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVL"
                "GIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEP"
                "ISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"
            ),
            "Villin headpiece (HP35, fast folder)": (
                "LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
            ),
        },
        value="GFP (Aequorea victoria)",
        label="Example sequence",
    )
    layer_slider = mo.ui.slider(
        start=1, stop=60, step=1, value=24,
        label="Hidden layer to inspect (1–30 for 300M, 1–60 for 600M)",
    )
    mo.vstack([
        mo.md("## Controls"),
        model_choice,
        sequence_choice,
        layer_slider,
    ])
    return layer_slider, model_choice, sequence_choice


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Fetch embeddings")
    return


@app.cell
def _(layer_slider, model_choice, mo, sequence_choice):
    import os
    _token = os.environ.get("BIOHUB_TOKEN", "")
    _sequence = sequence_choice.value
    _model = model_choice.value
    _layer = layer_slider.value

    if not _token:
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    "**No API token found.**  \n"
                    "Set `BIOHUB_TOKEN` in your environment, then re-run.  \n"
                    "Create a token at https://biohub.ai/developer-console/api-keys"
                ),
                kind="warn",
            ),
        )

    from esm.sdk import esmc_client
    from esm.sdk.api import ESMProtein, LogitsConfig

    _client = esmc_client(model=_model, url="https://biohub.ai", token=_token)
    _protein = ESMProtein(sequence=_sequence)
    _protein_tensor = _client.encode(_protein)
    _output = _client.logits(
        _protein_tensor,
        LogitsConfig(
            sequence=True,
            return_embeddings=True,
            return_hidden_states=True,
            ith_hidden_layer=_layer,
        ),
    )

    embeddings = _output.embeddings          # (L+2, D) — includes BOS/EOS
    hidden_states = _output.hidden_states    # (L+2, D) — layer _layer
    logits = _output.logits.sequence         # (L+2, vocab_size)
    sequence_str = _sequence
    mo.md(
        f"✅ Fetched embeddings for **{sequence_choice.value}** "
        f"(`{len(_sequence)} aa`, model `{_model}`, layer `{_layer}`).  \n"
        f"Embedding shape: `{embeddings.shape}` · "
        f"Logit shape: `{logits.shape}`"
    )
    return embeddings, hidden_states, logits, sequence_str


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Per-residue embedding norm")
    return


@app.cell(hide_code=True)
def _(embeddings, mo, np, plt, sequence_str):
    import numpy as _np_local
    _norms = _np_local.linalg.norm(embeddings[1:-1], axis=-1)   # strip BOS/EOS
    _residues = list(sequence_str)

    _fig, _ax = plt.subplots(figsize=(min(len(sequence_str) * 0.18 + 2, 16), 3))
    _x = _np_local.arange(len(_residues))
    _ax.bar(_x, _norms, width=0.85, color="#4c72b0", alpha=0.8)
    _ax.set_xlabel("Residue position")
    _ax.set_ylabel("Embedding L2 norm")
    _ax.set_title("Per-residue embedding norm (final layer)")
    _ax.set_xticks(_x[::max(1, len(_residues) // 30)])
    _ax.set_xticklabels(
        _residues[::max(1, len(_residues) // 30)],
        fontsize=8,
        rotation=0,
    )
    _ax.grid(axis="y", alpha=0.3)
    _fig.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Sequence logits — per-position entropy")
    return


@app.cell(hide_code=True)
def _(logits, mo, np, plt, sequence_str):
    import numpy as _np2
    import scipy.special as _sps

    _L = len(sequence_str)
    _probs = _sps.softmax(logits[1:-1], axis=-1)              # strip BOS/EOS
    _entropy = -(_probs * _np2.log(_probs + 1e-12)).sum(-1)   # (L,)

    _fig, _ax = plt.subplots(figsize=(min(_L * 0.18 + 2, 16), 3))
    _x = _np2.arange(_L)
    _ax.fill_between(_x, _entropy, alpha=0.5, color="#dd8452")
    _ax.plot(_x, _entropy, lw=1.5, color="#dd8452")
    _ax.set_xlabel("Residue position")
    _ax.set_ylabel("Shannon entropy (nats)")
    _ax.set_title("Per-position sequence uncertainty — high entropy = more variable site")
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Embedding PCA — sequence in representation space")
    return


@app.cell(hide_code=True)
def _(embeddings, mo, np, plt, sequence_str):
    import numpy as _np3

    _E = embeddings[1:-1]                    # (L, D)
    _E_centered = _E - _E.mean(axis=0)
    _U, _S, _Vt = _np3.linalg.svd(_E_centered, full_matrices=False)
    _pc = _U[:, :2] * _S[:2]                 # project onto top-2 PCs

    _colors = _np3.linspace(0, 1, len(sequence_str))

    _fig, _ax = plt.subplots(figsize=(7, 6))
    _sc = _ax.scatter(_pc[:, 0], _pc[:, 1], c=_colors, cmap="viridis", s=30, alpha=0.8)
    _fig.colorbar(_sc, ax=_ax, label="Residue position (0 = N-term)")
    _ax.set_xlabel(f"PC1 ({100 * _S[0] ** 2 / (_S ** 2).sum():.1f}% var)")
    _ax.set_ylabel(f"PC2 ({100 * _S[1] ** 2 / (_S ** 2).sum():.1f}% var)")
    _ax.set_title("PCA of per-residue embeddings (colored by position)")
    _ax.grid(alpha=0.25)
    _fig.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


if __name__ == "__main__":
    app.run()
