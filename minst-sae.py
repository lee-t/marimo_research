# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.11.0",
#     "numpy==2.5.1",
#     "torch==2.13.0",
#     "torchvision==0.28.0",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import time
    import math

    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as transforms

    return DataLoader, F, mo, nn, os, plt, torch, torchvision, transforms


@app.cell
def _(mo):
    mo.md(r"""
    # MNIST SAE testbed — Phase 0-2 (Option B: rich conv feature-map embeddings)

    This notebook implements the first three phases of the research plan
    for validating a sparse-autoencoder pipeline on a toy classifier
    before applying the same methodology to protein language model
    embeddings (ESMC).

    It uses **Option B** from the original plan: instead of a single,
    already-class-separated FC bottleneck vector per image, the
    embedding is the CNN's second conv layer's *spatial feature map*.
    Each of the H×W spatial positions in that map is treated as its own
    token — closely analogous to per-residue embeddings in a protein
    language model — rather than one global per-image vector. This
    should give the SAE genuinely local, translation-invariant visual
    primitives (edge/corner/stroke fragments) to recover, instead of
    directions that are already nearly class-separated by construction.

    - **Phase 0** — train a small CNN digit classifier; expose its
      conv2 feature map (not the FC head) as the embedding
    - **Phase 1** — cache per-position feature-map tokens across the
      full train/test set, plus a quick sanity-check projection
    - **Phase 2** — train a sparse autoencoder (vanilla L1 or TopK) on
      the cached tokens and track the standard health metrics:
      variance explained, L0, dead-feature fraction

    Feature interpretation and causal validation (Phase 3+) are left for
    a follow-on notebook — this one just gets you a trained classifier,
    cached tokens, and a trained SAE to interpret.

    Run with `marimo edit mnist_sae_phase0_2.py`. Requires
    `torch`, `torchvision`, `matplotlib`, `numpy`, `marimo`.
    """)
    return


@app.cell
def _(mo, torch):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    mo.md(f"**Using device:** `{device}`")
    return (device,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Phase 0 — Classifier and data
    """)
    return


@app.cell
def _(mo):
    conv2_channels_slider = mo.ui.slider(
        start=8, stop=128, step=8, value=32, label="conv2 channels (feature-map width)"
    )
    epochs_slider = mo.ui.slider(
        start=1, stop=15, step=1, value=5, label="Classifier epochs"
    )
    lr_slider = mo.ui.slider(
        start=1e-4, stop=5e-3, step=1e-4, value=1e-3, label="Classifier LR"
    )
    batch_size_slider = mo.ui.slider(
        start=32, stop=512, step=32, value=128, label="Batch size"
    )

    mo.vstack(
        [
            mo.md(
                "`conv2_channels` sets the width of the per-position "
                "embedding (the token dim the SAE will see), not a "
                "per-image bottleneck — there's no single narrow "
                "chokepoint here, since the point of Option B is to let "
                "the SAE work on local conv primitives rather than an "
                "already-compressed global vector."
            ),
            conv2_channels_slider,
            epochs_slider,
            lr_slider,
            batch_size_slider,
        ]
    )
    return batch_size_slider, conv2_channels_slider, epochs_slider, lr_slider


@app.cell
def _(DataLoader, batch_size_slider, torchvision, transforms):
    _transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=_transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_slider.value, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=512, shuffle=False
    )
    # unshuffled, larger-batch loader over the train set, used only for
    # activation caching in Phase 1 so ordering matches train_dataset
    train_eval_loader = DataLoader(
        train_dataset, batch_size=512, shuffle=False
    )
    return test_loader, train_eval_loader, train_loader


@app.cell
def _(nn):
    class MNISTClassifier(nn.Module):
        """Small CNN whose classification head reads a global-average-pooled
        summary of the conv2 feature map, but whose `forward` also returns
        the full, un-pooled spatial feature map (B, C, H, W) — that map,
        not the pooled vector, is the "rich" Option-B embedding used
        downstream for caching and the SAE. Because avg-pool and fc_out are
        both linear, `fc_out(mean_over_positions(feat_map))` is exactly
        `mean_over_positions(fc_out(feat_map))` — so per-position logit
        contributions can later be computed with the same fc_out weights,
        no extra machinery needed for Phase 4 causal-readout work.
        """

        def __init__(self, conv2_channels=32):
            super().__init__()
            self.conv2_channels = conv2_channels
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, conv2_channels, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            # 28x28 -> pool -> 14x14 -> pool -> 7x7, channels=conv2_channels
            self.fc_out = nn.Linear(conv2_channels, 10)

        def forward(self, x):
            import torch.nn.functional as F

            x = self.pool(F.relu(self.conv1(x)))
            feat_map = self.pool(F.relu(self.conv2(x)))  # (B, C, 7, 7)
            pooled = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)  # (B, C)
            logits = self.fc_out(pooled)
            return logits, feat_map

    return (MNISTClassifier,)


@app.cell
def _(F, torch):
    def evaluate_classifier(model, loader, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total

    def train_classifier(model, train_loader, test_loader, epochs, lr, device):
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = {"train_loss": [], "test_acc": []}
        for _epoch in range(epochs):
            model.train()
            total_loss, n = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits, _ = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                n += x.size(0)
            history["train_loss"].append(total_loss / n)
            history["test_acc"].append(
                evaluate_classifier(model, test_loader, device)
            )
        return model, history

    return (train_classifier,)


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train classifier")
    train_button
    return (train_button,)


@app.cell
def _(
    MNISTClassifier,
    conv2_channels_slider,
    device,
    epochs_slider,
    lr_slider,
    mo,
    test_loader,
    train_button,
    train_classifier,
    train_loader,
):
    mo.stop(
        not train_button.value,
        mo.md("Click **Train classifier** above to begin (retrains from scratch on each click, using the current slider values)."),
    )

    classifier = MNISTClassifier(conv2_channels=conv2_channels_slider.value)
    classifier, classifier_history = train_classifier(
        classifier,
        train_loader,
        test_loader,
        epochs=epochs_slider.value,
        lr=lr_slider.value,
        device=device,
    )

    mo.md(
        f"Final test accuracy: **{classifier_history['test_acc'][-1]:.4f}** "
        f"after {epochs_slider.value} epoch(s), conv2 feature-map width "
        f"{conv2_channels_slider.value} (7×7 spatial grid)."
    )
    return classifier, classifier_history


@app.cell
def _(classifier_history, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.2))
    _axes[0].plot(classifier_history["train_loss"], marker="o")
    _axes[0].set_title("Train loss")
    _axes[0].set_xlabel("epoch")

    _axes[1].plot(classifier_history["test_acc"], marker="o", color="darkorange")
    _axes[1].set_title("Test accuracy")
    _axes[1].set_xlabel("epoch")
    _axes[1].set_ylim(0, 1)

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Phase 1 — Activation caching

    Run every train and test image through the trained classifier once
    and stash the conv2 feature map. Rather than one vector per image,
    each of the 7×7=49 spatial positions is unrolled into its own
    token, giving `N_images × 49` embeddings of width `conv2_channels`.
    Each token also carries: which image and grid position it came
    from, the image's true label, the image's predicted label, and the
    image's predicted-class logit margin — note these last three are
    **per-image, broadcast onto every position of that image**, so
    they're a weak/proxy label at the position level (a token from a
    blank corner of a "3" is still tagged "3"). Useful for a first pass,
    but treat position-level purity numbers with that caveat in mind.

    A separate, genuinely per-image embedding (the same
    global-average-pooled vector the classifier head actually reads) is
    cached alongside the tokens, for a cleaner sanity-check visualization.

    Two convs + two 2×2 pools gives each output position an exact
    receptive field of 10×10 input pixels with a 4px stride between
    adjacent positions — so neighboring tokens have overlapping, but
    distinct, views of the image. That geometry is what makes patch-level
    visualization straightforward in Phase 3.
    """)
    return


@app.cell
def _(torch):
    def cache_activations(model, loader, device):
        model.eval()
        tokens, img_idx, pos_idx, tok_labels, tok_preds, tok_margins = (
            [], [], [], [], [], [],
        )
        pooled_embeds, pooled_labels, pooled_preds, pooled_margins = [], [], [], []
        running_idx = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits, feat_map = model(x)  # feat_map: (B, C, H, W)
                b, c, h, w = feat_map.shape
                top2 = logits.topk(2, dim=-1).values
                margin = (top2[:, 0] - top2[:, 1]).cpu()
                pred = logits.argmax(dim=-1).cpu()

                # per-position tokens: (B*H*W, C)
                flat = feat_map.permute(0, 2, 3, 1).reshape(b * h * w, c).cpu()
                tokens.append(flat)
                img_idx.append(
                    torch.arange(running_idx, running_idx + b).repeat_interleave(h * w)
                )
                pos_idx.append(torch.arange(h * w).repeat(b))
                tok_labels.append(y.repeat_interleave(h * w))
                tok_preds.append(pred.repeat_interleave(h * w))
                tok_margins.append(margin.repeat_interleave(h * w))

                # per-image pooled embedding (matches what fc_out actually sees)
                pooled_embeds.append(feat_map.mean(dim=(2, 3)).cpu())
                pooled_labels.append(y)
                pooled_preds.append(pred)
                pooled_margins.append(margin)

                running_idx += b

        token_data = {
            "tokens": torch.cat(tokens),
            "img_idx": torch.cat(img_idx),
            "pos_idx": torch.cat(pos_idx),
            "labels": torch.cat(tok_labels),
            "preds": torch.cat(tok_preds),
            "margins": torch.cat(tok_margins),
        }
        pooled_data = {
            "embeds": torch.cat(pooled_embeds),
            "labels": torch.cat(pooled_labels),
            "preds": torch.cat(pooled_preds),
            "margins": torch.cat(pooled_margins),
        }
        return token_data, pooled_data

    return (cache_activations,)


@app.cell
def _(
    cache_activations,
    classifier,
    device,
    mo,
    os,
    test_loader,
    torch,
    train_eval_loader,
):
    train_tokens, train_pooled = cache_activations(classifier, train_eval_loader, device)
    test_tokens, test_pooled = cache_activations(classifier, test_loader, device)

    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(
        {
            "train_tokens": train_tokens,
            "train_pooled": train_pooled,
            "test_tokens": test_tokens,
            "test_pooled": test_pooled,
        },
        "./checkpoints/activations.pt",
    )
    torch.save(classifier.state_dict(), "./checkpoints/classifier.pt")

    mo.md(
        f"Cached **{train_tokens['tokens'].shape[0]:,}** train tokens "
        f"(from {train_pooled['embeds'].shape[0]:,} images × 49 positions) "
        f"and **{test_tokens['tokens'].shape[0]:,}** test tokens, width "
        f"**{train_tokens['tokens'].shape[1]}**. Saved to "
        f"`./checkpoints/activations.pt` and `./checkpoints/classifier.pt`."
    )
    return train_pooled, train_tokens


@app.cell
def _(mo):
    mo.md(r"""
    ### Quick sanity check: 2D projection of the per-image embedding

    A simple PCA (via SVD, no sklearn dependency) projection of the
    **per-image pooled** embedding (not the per-position tokens —
    those don't have a clean per-example label), colored by true digit
    label. This is just eyeballing whether the conv backbone learned
    sensible global structure before handing per-position tokens to the
    SAE — it is **not** a substitute for the SAE feature analysis in
    Phase 3.
    """)
    return


@app.cell
def _(plt, torch, train_pooled):
    def pca_2d(x, n_components=2):
        x_centered = x - x.mean(dim=0, keepdim=True)
        _, _, v = torch.linalg.svd(x_centered, full_matrices=False)
        return x_centered @ v[:n_components].T

    _embeds = train_pooled["embeds"]
    _labels = train_pooled["labels"]
    _n_plot = min(4000, _embeds.shape[0])
    _idx = torch.randperm(_embeds.shape[0])[:_n_plot]
    _proj = pca_2d(_embeds[_idx]).numpy()
    _labels_np = _labels[_idx].numpy()

    _fig, _ax = plt.subplots(figsize=(6, 5))
    _scatter = _ax.scatter(
        _proj[:, 0], _proj[:, 1], c=_labels_np, cmap="tab10", s=6, alpha=0.6
    )
    _legend = _ax.legend(
        *_scatter.legend_elements(), title="digit", loc="center left",
        bbox_to_anchor=(1.0, 0.5)
    )
    _ax.add_artist(_legend)
    _ax.set_title("PCA projection of pooled conv embeddings (train set)")
    _ax.set_xlabel("PC1")
    _ax.set_ylabel("PC2")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Phase 2 — Sparse autoencoder

    Vanilla ReLU+L1 SAE and a TopK variant, sharing the same module, now
    trained on the per-position **tokens** rather than a per-image
    vector — with 49 tokens per image, the full train set is close to
    3M rows, so training subsamples down to a fixed cap for tractable
    wall-clock time on CPU. Decoder columns are renormalized to unit
    norm after every optimizer step (keeps the L1 penalty from being
    trivially gamed by shrinking decoder norms).
    """)
    return


@app.cell
def _(mo):
    dict_mult_dropdown = mo.ui.dropdown(
        options=["1x", "4x", "8x", "16x"], value="8x", label="Dictionary size multiplier"
    )
    sae_type_dropdown = mo.ui.dropdown(
        options=["l1", "topk"], value="l1", label="SAE type"
    )
    l1_coeff_slider = mo.ui.slider(
        start=1e-4, stop=5e-2, step=1e-4, value=5e-3, label="L1 coefficient (l1 type only)"
    )
    k_slider = mo.ui.slider(
        start=1, stop=64, step=1, value=8, label="k (topk type only)"
    )
    sae_epochs_slider = mo.ui.slider(
        start=5, stop=200, step=5, value=50, label="SAE epochs"
    )
    sae_lr_slider = mo.ui.slider(
        start=1e-4, stop=1e-2, step=1e-4, value=1e-3, label="SAE LR"
    )
    max_sae_examples_slider = mo.ui.slider(
        start=10_000, stop=500_000, step=10_000, value=150_000,
        label="Max SAE training tokens (random subsample)",
    )

    mo.vstack(
        [
            dict_mult_dropdown,
            sae_type_dropdown,
            l1_coeff_slider,
            k_slider,
            sae_epochs_slider,
            sae_lr_slider,
            max_sae_examples_slider,
        ]
    )
    return (
        dict_mult_dropdown,
        k_slider,
        l1_coeff_slider,
        max_sae_examples_slider,
        sae_epochs_slider,
        sae_lr_slider,
        sae_type_dropdown,
    )


@app.cell
def _(F, nn, torch):
    class SparseAutoencoder(nn.Module):
        """Single-layer tied-init SAE. `sae_type='l1'` uses ReLU activations
        with an L1 penalty on the loss; `sae_type='topk'` keeps only the k
        largest ReLU activations per example and drops the L1 term (Gao et
        al.-style TopK SAE)."""

        def __init__(self, d_in, d_hidden, sae_type="l1", k=None):
            super().__init__()
            self.d_in = d_in
            self.d_hidden = d_hidden
            self.sae_type = sae_type
            self.k = k
            self.W_enc = nn.Parameter(torch.randn(d_in, d_hidden) * (1.0 / d_in ** 0.5))
            self.b_enc = nn.Parameter(torch.zeros(d_hidden))
            self.W_dec = nn.Parameter(self.W_enc.t().clone())
            self.b_dec = nn.Parameter(torch.zeros(d_in))
            self.normalize_decoder()

        def normalize_decoder(self):
            with torch.no_grad():
                norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
                self.W_dec.div_(norms)

        def encode(self, x):
            pre = (x - self.b_dec) @ self.W_enc + self.b_enc
            acts = F.relu(pre)
            if self.sae_type == "topk" and self.k is not None:
                topk_vals, topk_idx = acts.topk(self.k, dim=-1)
                mask = torch.zeros_like(acts)
                mask.scatter_(-1, topk_idx, 1.0)
                acts = acts * mask
            return acts

        def decode(self, acts):
            return acts @ self.W_dec + self.b_dec

        def forward(self, x):
            acts = self.encode(x)
            recon = self.decode(acts)
            return recon, acts

    return (SparseAutoencoder,)


@app.cell
def _(F, torch):
    def train_sae(
        sae, data, epochs, lr, l1_coeff, sae_type, device, batch_size=256
    ):
        sae.to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
        n = data.shape[0]

        history = {
            "loss": [],
            "recon_loss": [],
            "l1_penalty": [],
            "var_explained": [],
            "l0": [],
            "dead_frac": [],
        }
        fired_ever = torch.zeros(sae.d_hidden, dtype=torch.bool, device=device)

        for _epoch in range(epochs):
            perm = torch.randperm(n, device=device)
            epoch_recon, epoch_l1, epoch_l0, nb = 0.0, 0.0, 0.0, 0

            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                batch = data[idx]
                recon, acts = sae(batch)
                recon_loss = F.mse_loss(recon, batch)

                if sae_type == "l1":
                    l1_penalty = acts.abs().sum(dim=-1).mean()
                    loss = recon_loss + l1_coeff * l1_penalty
                else:
                    l1_penalty = torch.tensor(0.0, device=device)
                    loss = recon_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    sae.normalize_decoder()
                    fired_ever |= (acts > 0).any(dim=0)

                bs = batch.size(0)
                epoch_recon += recon_loss.item() * bs
                epoch_l1 += l1_penalty.item() * bs
                epoch_l0 += (acts > 0).float().sum(dim=-1).mean().item() * bs
                nb += bs

            with torch.no_grad():
                recon_full, _ = sae(data)
                var_explained = 1 - (recon_full - data).var() / data.var()

            recon_mean = epoch_recon / nb
            l1_mean = epoch_l1 / nb
            history["recon_loss"].append(recon_mean)
            history["l1_penalty"].append(l1_mean)
            history["loss"].append(
                recon_mean + (l1_coeff * l1_mean if sae_type == "l1" else 0.0)
            )
            history["l0"].append(epoch_l0 / nb)
            history["var_explained"].append(var_explained.item())
            history["dead_frac"].append(1 - fired_ever.float().mean().item())

        return sae, history

    return (train_sae,)


@app.cell
def _(mo):
    sae_train_button = mo.ui.run_button(label="Train SAE")
    sae_train_button
    return (sae_train_button,)


@app.cell
def _(
    SparseAutoencoder,
    device,
    dict_mult_dropdown,
    k_slider,
    l1_coeff_slider,
    max_sae_examples_slider,
    mo,
    sae_epochs_slider,
    sae_lr_slider,
    sae_train_button,
    sae_type_dropdown,
    torch,
    train_sae,
    train_tokens,
):
    mo.stop(
        not sae_train_button.value,
        mo.md("Click **Train SAE** above to begin (uses the cached train tokens from Phase 1)."),
    )

    _all_tokens = train_tokens["tokens"]
    _cap = min(max_sae_examples_slider.value, _all_tokens.shape[0])
    _sub_idx = torch.randperm(_all_tokens.shape[0])[:_cap]
    sae_train_data = _all_tokens[_sub_idx]

    _mult = int(dict_mult_dropdown.value.rstrip("x"))
    d_in = sae_train_data.shape[1]
    d_hidden = d_in * _mult

    sae = SparseAutoencoder(
        d_in=d_in,
        d_hidden=d_hidden,
        sae_type=sae_type_dropdown.value,
        k=k_slider.value,
    )
    sae, sae_history = train_sae(
        sae,
        sae_train_data,
        epochs=sae_epochs_slider.value,
        lr=sae_lr_slider.value,
        l1_coeff=l1_coeff_slider.value,
        sae_type=sae_type_dropdown.value,
        device=device,
    )

    mo.md(
        f"Trained a **{sae_type_dropdown.value}** SAE on "
        f"**{sae_train_data.shape[0]:,}** tokens, dict size "
        f"**{d_hidden}** ({dict_mult_dropdown.value} of {d_in}-dim input). "
        f"Final variance explained: **{sae_history['var_explained'][-1]:.4f}**, "
        f"mean L0: **{sae_history['l0'][-1]:.1f}**, "
        f"dead fraction: **{sae_history['dead_frac'][-1]:.3f}**."
    )
    return sae, sae_history


@app.cell
def _(os, sae, torch):
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "d_in": sae.d_in,
            "d_hidden": sae.d_hidden,
            "sae_type": sae.sae_type,
            "k": sae.k,
        },
        "./checkpoints/sae.pt",
    )
    return


@app.cell
def _(plt, sae_history):
    _fig, _axes = plt.subplots(2, 2, figsize=(9, 6.5))

    _axes[0, 0].plot(sae_history["recon_loss"])
    _axes[0, 0].set_title("Reconstruction loss (MSE)")
    _axes[0, 0].set_xlabel("epoch")

    _axes[0, 1].plot(sae_history["var_explained"], color="darkorange")
    _axes[0, 1].set_title("Variance explained")
    _axes[0, 1].set_xlabel("epoch")
    _axes[0, 1].set_ylim(0, 1)

    _axes[1, 0].plot(sae_history["l0"], color="green")
    _axes[1, 0].set_title("Mean L0 (active features / example)")
    _axes[1, 0].set_xlabel("epoch")

    _axes[1, 1].plot(sae_history["dead_frac"], color="crimson")
    _axes[1, 1].set_title("Dead feature fraction (never fired so far)")
    _axes[1, 1].set_xlabel("epoch")
    _axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Next: Phase 3

    With `./checkpoints/classifier.pt`, `./checkpoints/activations.pt`,
    and `./checkpoints/sae.pt` on disk, the next notebook can move on to
    feature interpretation and causal validation without retraining
    anything here:

    - max-activating example grids per SAE feature — now genuinely
      patch-level: use `img_idx`/`pos_idx` in the cached token data plus
      the exact 10×10px / 4px-stride receptive-field geometry to crop
      and display the actual image region each firing token corresponds
      to, rather than the whole digit
    - gradient-based activation maximization in pixel space
    - feature-class selectivity / purity scoring — remember token
      labels are per-image proxies, so cross-check any position-level
      purity claim against a handful of cropped examples
    - reconstruction substitution (fraction of classifier accuracy
      recovered when swapping in the SAE reconstruction of each token,
      re-pooling, and reclassifying)
    - single ablation/amplification of a candidate feature
    - projecting SAE decoder directions onto `fc_out`'s weights directly
      — no extra care needed for the avg-pool step, since pooling and
      the readout are both linear and commute, so per-position causal
      relevance is exact, not approximate. This is the same
      causally-privileged-subspace question this testbed exists to
      prototype before trying it on ESMC.

    If you want a **dictionary-size / sparsity sweep** across multiple
    `(dict_mult, l1_coeff)` or `(dict_mult, k)` combinations rather than
    one run at a time, that's a natural extension of the Phase 2 cells
    above — happy to add a sweep grid + comparison plot on request.
    """)
    return


if __name__ == "__main__":
    app.run()
