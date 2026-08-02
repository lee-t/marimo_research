# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy>=2.0",
#     "matplotlib>=3.9",
#     "scikit-learn>=1.5",
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
    # CNN Probing of Protein Language Model Representations

    Protein language models like [ESMC](https://biohub.ai/esm/protein) learn rich
    **per-residue representations**. A natural question is: what *local sequence patterns*
    are encoded in these representations?

    One way to probe this is with a **1-D convolutional neural network (CNN)** applied on top
    of frozen residue-level embeddings. Each filter learns a position-weight matrix (PWM)
    over the embedding dimensions — essentially a learned *motif detector*.

    This notebook:

    1. Fetches residue-level embeddings for a panel of proteins via the Biohub API
    2. Builds a simple 1-D CNN on top of frozen embeddings in NumPy/pure Python
    3. Trains the CNN on a toy residue-labelling task (hydrophobic vs. polar classification)
    4. Visualises the learned filter activations along the sequence

    > **Setup:** set the environment variable `BIOHUB_TOKEN` to your API key before running.
    > Create a token at <https://biohub.ai/developer-console/api-keys>.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    kernel_size = mo.ui.slider(
        start=3, stop=15, step=2, value=7,
        label="CNN kernel size (residues)",
    )
    n_filters = mo.ui.slider(
        start=4, stop=32, step=4, value=16,
        label="Number of CNN filters",
    )
    learning_rate = mo.ui.slider(
        start=1, stop=50, step=1, value=10,
        label="Learning rate × 10⁻³",
    )
    n_epochs = mo.ui.slider(
        start=20, stop=200, step=20, value=80,
        label="Training epochs",
    )
    model_choice = mo.ui.dropdown(
        options=["esmc-300m-2024-12", "esmc-600m-2024-12"],
        value="esmc-300m-2024-12",
        label="ESMC model",
    )
    mo.vstack([
        mo.md("## Controls"),
        model_choice,
        mo.hstack([kernel_size, n_filters], gap=2),
        mo.hstack([learning_rate, n_epochs], gap=2),
    ])
    return kernel_size, learning_rate, model_choice, n_epochs, n_filters


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Fetch embeddings from Biohub")
    return


@app.cell
def _(mo, model_choice):
    import os
    _token = os.environ.get("BIOHUB_TOKEN", "")

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

    _SEQUENCES = {
        "GFP": (
            "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQ"
            "CFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGH"
            "KLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSK"
            "DPNEKRDHMVLLEFVTAAGITHGMDELYK"
        ),
        "CA2": (
            "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDD"
            "SQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVL"
            "GIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEP"
            "ISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"
        ),
        "HP35": "LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
    }

    _client = esmc_client(model=model_choice.value, url="https://biohub.ai", token=_token)

    _embeddings_dict = {}
    for _name, _seq in _SEQUENCES.items():
        _protein = ESMProtein(sequence=_seq)
        _tensor = _client.encode(_protein)
        _out = _client.logits(_tensor, LogitsConfig(return_embeddings=True))
        _embeddings_dict[_name] = (_seq, _out.embeddings[1:-1])  # strip BOS/EOS

    sequences_and_embeddings = _embeddings_dict
    mo.md(
        f"✅ Fetched embeddings for {len(sequences_and_embeddings)} proteins "
        f"using `{model_choice.value}`.  \n"
        + "  \n".join(
            f"- **{k}**: {len(s)} residues, embedding dim {e.shape[-1]}"
            for k, (s, e) in sequences_and_embeddings.items()
        )
    )
    return (sequences_and_embeddings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Build training data — hydrophobic vs. polar labels")
    return


@app.cell(hide_code=True)
def _(np, sequences_and_embeddings):
    _HYDROPHOBIC = set("ACFILMVWY")

    _X_list = []
    _y_list = []

    for _name, (_seq, _emb) in sequences_and_embeddings.items():
        _X_list.append(_emb)
        _y_list.extend([1 if aa in _HYDROPHOBIC else 0 for aa in _seq])

    X_all = np.vstack(_X_list)          # (N_total_residues, D)
    y_all = np.array(_y_list, dtype=np.float32)

    _pos = y_all.sum()
    _neg = len(y_all) - _pos
    print(f"Dataset: {len(y_all)} residues — {_pos:.0f} hydrophobic, {_neg:.0f} polar")
    return X_all, y_all


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Train a 1-D CNN probe")
    return


@app.cell(hide_code=True)
def _(X_all, kernel_size, learning_rate, n_epochs, n_filters, np, y_all):
    def _conv1d_forward(X, W, b):
        """Pure-NumPy 1-D valid convolution over residue axis.

        X : (N, D)       — residue embeddings
        W : (F, K, D)    — F filters, each (K, D)
        b : (F,)         — bias per filter
        returns (N - K + 1, F)
        """
        N, D = X.shape
        F, K, _ = W.shape
        L_out = N - K + 1
        out = np.zeros((L_out, F), dtype=np.float32)
        for i in range(L_out):
            patch = X[i : i + K]           # (K, D)
            for f in range(F):
                out[i, f] = (patch * W[f]).sum() + b[f]
        return out

    def _relu(x):
        return np.maximum(0.0, x)

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _run_cnn(X, W_conv, b_conv, W_fc, b_fc):
        conv_out = _relu(_conv1d_forward(X, W_conv, b_conv))   # (L_out, F)
        pooled = conv_out.max(axis=0)                           # (F,) global max pool
        logit = W_fc @ pooled + b_fc                            # scalar
        return _sigmoid(logit), conv_out

    # --- Prepare per-protein windows for CNN training ---
    _K = kernel_size.value
    _F = n_filters.value
    _lr = learning_rate.value * 1e-3
    _epochs = n_epochs.value
    _D = X_all.shape[1]

    _MINIBATCH = 512   # residues per SGD step

    rng = np.random.default_rng(42)
    W_conv = rng.normal(scale=0.1, size=(_F, _K, _D)).astype(np.float32)
    b_conv = np.zeros(_F, dtype=np.float32)
    W_fc = rng.normal(scale=0.1, size=_F).astype(np.float32)
    b_fc = np.float32(0.0)

    # Build windows: each window is K consecutive residues → label is centre residue
    _pad = _K // 2
    X_padded = np.pad(X_all, ((_pad, _pad), (0, 0)), mode="edge")
    _N = len(y_all)
    windows = np.stack([X_padded[i : i + _K] for i in range(_N)], axis=0)  # (N, K, D)

    def _forward_windows(W_c, b_c, W_f, b_f, idx):
        patch = windows[idx]                         # (K, D)
        h = _relu((patch[None, :, :] * W_c).sum((-1, -2)) + b_c)   # (F,)
        return _sigmoid((W_f * h).sum() + b_f), h

    # Stochastic gradient descent
    _loss_history = []
    for _epoch in range(_epochs):
        _idx_perm = rng.permutation(_N)
        _epoch_loss = 0.0
        for _i in _idx_perm[:_MINIBATCH]:            # mini-batch per epoch
            _p, _h = _forward_windows(W_conv, b_conv, W_fc, b_fc, _i)
            _y_i = y_all[_i]
            _dl_dp = _p - _y_i               # BCE gradient at output
            _dl_dWfc = _dl_dp * _h
            _dl_dbfc = _dl_dp
            _dl_dh = _dl_dp * W_fc           # (F,)
            _relu_mask = (_h > 0).astype(np.float32)
            _dl_dpre = _dl_dh * _relu_mask   # (F,)
            _patch = windows[_i]             # (K, D)
            _dl_dWconv = _dl_dpre[:, None, None] * _patch[None, :, :]  # (F, K, D)
            _dl_dbconv = _dl_dpre

            W_conv -= _lr * _dl_dWconv
            b_conv -= _lr * _dl_dbconv
            W_fc -= _lr * _dl_dWfc
            b_fc -= _lr * _dl_dbfc
            _epoch_loss += float(-(_y_i * np.log(_p + 1e-12) + (1 - _y_i) * np.log(1 - _p + 1e-12)))

        if _epoch % 10 == 0 or _epoch == _epochs - 1:
            _loss_history.append((_epoch + 1, _epoch_loss / 512))

    # Evaluate accuracy
    _preds = np.array([_forward_windows(W_conv, b_conv, W_fc, b_fc, i)[0] for i in range(_N)])
    _acc = (((_preds >= 0.5).astype(int) == y_all.astype(int)).mean())

    cnn_weights = {"W_conv": W_conv, "b_conv": b_conv, "W_fc": W_fc, "b_fc": b_fc}
    cnn_loss_history = _loss_history
    cnn_accuracy = _acc
    print(f"Final test accuracy: {100 * _acc:.1f}%")
    return cnn_accuracy, cnn_loss_history, cnn_weights, rng, windows


@app.cell(hide_code=True)
def _(cnn_accuracy, cnn_loss_history, mo, np, plt):
    _epochs_arr = np.array([e for e, _ in cnn_loss_history])
    _loss_arr = np.array([l for _, l in cnn_loss_history])

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

    _axes[0].plot(_epochs_arr, _loss_arr, lw=2, color="#4c72b0")
    _axes[0].set_title("Training loss (BCE)")
    _axes[0].set_xlabel("Epoch")
    _axes[0].set_ylabel("Loss")
    _axes[0].grid(alpha=0.3)

    _axes[1].barh(
        ["Hydrophobic / polar\nclassification"],
        [100 * cnn_accuracy],
        color=["#55a868"],
        height=0.4,
    )
    _axes[1].set_xlim(0, 100)
    _axes[1].set_xlabel("Accuracy (%)")
    _axes[1].set_title("Probe accuracy")
    _axes[1].axvline(50, lw=1, ls="--", color="grey", label="Chance")
    _axes[1].legend()
    _axes[1].grid(axis="x", alpha=0.3)

    _fig.suptitle(
        f"1-D CNN probe on frozen ESMC embeddings — accuracy {100 * cnn_accuracy:.1f}%",
        fontsize=12,
    )
    _fig.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Learned filter norms — which filters are most active?")
    return


@app.cell(hide_code=True)
def _(cnn_weights, mo, n_filters, np, plt):
    _W = cnn_weights["W_conv"]                       # (F, K, D)
    _filter_norms = np.linalg.norm(_W.reshape(_W.shape[0], -1), axis=-1)  # (F,)
    _order = np.argsort(_filter_norms)[::-1]

    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.bar(
        range(n_filters.value),
        _filter_norms[_order],
        color="#c44e52",
        alpha=0.8,
    )
    _ax.set_xlabel("Filter index (sorted by norm)")
    _ax.set_ylabel("Filter weight norm")
    _ax.set_title("Learned CNN filter norms — larger = more discriminative")
    _ax.grid(axis="y", alpha=0.3)
    _fig.tight_layout()
    mo.center(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Activation map along a sequence")
    return


@app.cell(hide_code=True)
def _(cnn_weights, mo, n_filters, np, plt, sequences_and_embeddings):
    _name, (_seq, _emb) = next(iter(sequences_and_embeddings.items()))
    _W = cnn_weights["W_conv"]    # (F, K, D)
    _b = cnn_weights["b_conv"]    # (F,)
    _K = _W.shape[1]
    _N_res = len(_seq)
    _L_out = _N_res - _K + 1

    _act = np.zeros((_L_out, n_filters.value), dtype=np.float32)
    for _i in range(_L_out):
        _patch = _emb[_i : _i + _K]     # (K, D)
        _act[_i] = np.maximum(0.0, (_patch[None, :, :] * _W).sum((-1, -2)) + _b)

    _fig, _ax = plt.subplots(figsize=(14, 4))
    _im = _ax.imshow(
        _act.T,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )
    _fig.colorbar(_im, ax=_ax, label="ReLU activation")
    _ax.set_xlabel("Residue position")
    _ax.set_ylabel("Filter index")
    _ax.set_title(f"CNN filter activation map — {_name} ({_N_res} residues)")
    _ax.set_xticks(range(0, _L_out, max(1, _L_out // 20)))
    _ax.set_xticklabels(
        [_seq[i] for i in range(0, _L_out, max(1, _L_out // 20))],
        fontsize=8,
    )
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
