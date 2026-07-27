# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.10.0",
#     "torch>=2.4.0",
#     "transformers",
#     "numpy>=2.0",
#     "plotly",
#     "scikit-learn",
#     "safetensors",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import torch
    from plotly.subplots import make_subplots
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    return AutoModelForMaskedLM, AutoTokenizer, go, make_subplots, mo, torch


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🧬 Protein J-Space & Multi-Task Falsification Suite

    Does a protein language model route information through a shared **Global Workspace
    (J-space)**, or through **independent modular channels**?

    ## The test

    Transplanting Gurnee et al.'s J-space paradigm to pLMs runs into a **20-amino-acid
    vocabulary bottleneck**: a single next-token metric cannot distinguish broadcast from
    routing. So we read out **three tasks that depend on different information**:

    | # | Task | Metric | Units |
    |---|---|---|---|
    | 1 | Masked-LM sequence recovery | $D_{KL}(p_{orig} \parallel p_{pert})$ | nats |
    | 2 | Pairwise contact topology | $\lVert \Delta C \rVert_F$ on the cosine contact map | norm |
    | 3 | Biophysical composition | shift in total hydrophobic probability mass | probability |

    ## The three interventions

    A forward hook replaces the layer-$\ell$ hidden state $\mathbf{h}_\ell$. All three use the
    **same perturbation magnitude $\alpha$**, so their effects are directly comparable:

    | Mode | Operation | What it tells us |
    |---|---|---|
    | **Steer along v** | $\mathbf{h}' = \mathbf{h} + \alpha\mathbf{v}$ | effect of the candidate direction |
    | **Random noise** | $\mathbf{h}' = \mathbf{h} + \alpha\mathbf{r}$, $\lVert\mathbf{r}\rVert = 1$ | chance baseline |
    | **Noise ⊥ v** | $\mathbf{h}' = \mathbf{h} + \alpha \mathbf{P}_{\perp}\mathbf{r}$ | cost of perturbing *everything except* v |

    **How to read the result.** Steering must beat the random baseline on **more than one**
    task to support a workspace. Beating it on one task only implies a modular channel. If
    noise orthogonal to $\mathbf{v}$ is markedly *cheaper* than isotropic noise, sensitivity
    is concentrated in a compact subspace around $\mathbf{v}$.
    """)
    return


@app.cell
def config_ui(mo):
    model_dropdown = mo.ui.dropdown(
        options=[
            "facebook/esm2_t6_8M_UR50D",
            "facebook/esm2_t12_35M_UR50D",
            "facebook/esm2_t30_150M_UR50D",
            "facebook/esm2_t33_650M_UR50D",
        ],
        value="facebook/esm2_t6_8M_UR50D",
        label="Base protein model",
    )
    direction_dropdown = mo.ui.dropdown(
        options=[
            "PCA (1st Principal Component)",
            "Analytical Jacobian (d R / d h_l)",
            "Random Direction Control",
        ],
        value="PCA (1st Principal Component)",
        label="Candidate direction v",
    )
    layer_slider = mo.ui.slider(
        start=1,
        stop=33,
        step=1,
        value=3,
        label="Intervention layer ℓ",
        show_value=True,
    )
    alpha_slider = mo.ui.slider(
        start=0.0,
        stop=5.0,
        step=0.5,
        value=2.0,
        label="Perturbation magnitude α",
        show_value=True,
    )
    sequence_input = mo.ui.text(
        value="MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFI",
        label="Protein sequence",
        full_width=True,
    )

    controls = mo.vstack([
        mo.md("### Controls"),
        mo.hstack([model_dropdown, direction_dropdown], gap=2, justify="start"),
        mo.hstack([layer_slider, alpha_slider], gap=2, justify="start"),
        sequence_input,
    ])
    controls
    return (
        alpha_slider,
        direction_dropdown,
        layer_slider,
        model_dropdown,
        sequence_input,
    )


@app.cell
def load_model(AutoModelForMaskedLM, AutoTokenizer, model_dropdown):
    model_id = model_dropdown.value
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True).eval()

    # Determine internal encoder module path
    if hasattr(model, "esm") and hasattr(model.esm, "encoder"):
        encoder_layers = model.esm.encoder.layer
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        encoder_layers = model.encoder.layer
    else:
        # Fallback for standard ESM architecture
        encoder_layers = getattr(model, "layers", getattr(getattr(model, "esm", None), "layers", None))

    num_layers = len(encoder_layers) if encoder_layers is not None else 6
    return encoder_layers, model, num_layers, tokenizer


@app.cell(hide_code=True)
def layer_status(layer_slider, mo, model_dropdown, num_layers):
    # The slider spans up to 33 layers for the largest checkpoint; smaller models clamp.
    eff_layer = min(layer_slider.value, num_layers)
    _clamped = layer_slider.value > num_layers
    mo.md(
        f"""
        **Model** `{model_dropdown.value}` · **{num_layers}** encoder layers ·
        intervening at layer **ℓ = {eff_layer}**
        {"· ⚠️ requested layer " + str(layer_slider.value) + " exceeds depth, clamped to " + str(num_layers) if _clamped else ""}
        """
    )
    return (eff_layer,)


@app.cell
def run_falsification(
    alpha_slider,
    direction_dropdown,
    eff_layer,
    encoder_layers,
    model,
    sequence_input,
    tokenizer,
    torch,
):
    seq = sequence_input.value
    l_idx = eff_layer - 1
    alpha = alpha_slider.value
    dir_method = direction_dropdown.value

    # Tokenize input
    inputs = tokenizer([seq], return_tensors="pt")

    # Baseline forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of (layer_0, ..., layer_L)
        orig_logits = outputs.logits

    # Hidden activations at layer l (SeqLen, HiddenDim)
    h_l = hidden_states[l_idx + 1][0]  # Layer + 1 accounts for embedding layer at index 0

    # 1. Derive candidate direction vector v
    if "Jacobian" in dir_method:
        # Analytical gradient d(readout)/d h_l.
        # NOTE: the baseline pass above ran under torch.no_grad(), so its tensors carry
        # no autograd graph. We therefore re-run the forward pass with grad enabled and
        # capture the *live* layer-l output tensor, which IS a node in that graph.
        _captured = {}

        def _capture_hook(module, hook_inputs, hook_outputs):
            _captured["h"] = hook_outputs[0] if isinstance(hook_outputs, tuple) else hook_outputs
            return hook_outputs

        _handle = encoder_layers[l_idx].register_forward_hook(_capture_hook)
        try:
            with torch.enable_grad():
                grad_outputs = model(**inputs)
                # Readout R = total max-logit confidence summed over residues.
                target_score = grad_outputs.logits[0].max(dim=-1).values.sum()
                h_live = _captured["h"]
                grad = torch.autograd.grad(target_score, h_live, retain_graph=False)[0]
        finally:
            _handle.remove()

        # grad: (batch, seq_len, hidden) -> mean over residues of batch 0
        v = grad[0].mean(dim=0).detach()
    elif "Random" in dir_method:
        rng = torch.Generator().manual_seed(42)
        v = torch.randn(h_l.shape[-1], generator=rng)
    else:
        # First Principal Component (PCA)
        h_centered = h_l - h_l.mean(dim=0, keepdim=True)
        _, _, V = torch.pca_lowrank(h_centered, q=1)
        v = V[:, 0]

    v = v / (torch.norm(v) + 1e-8)

    # 2. Interception Hook for PyTorch Transformer Layers
    class InterceptionHook:
        def __init__(self, direction, alpha, mode):
            self.v = direction
            self.alpha = alpha
            self.mode = mode

        def __call__(self, module, inputs, outputs):
            h = outputs[0] if isinstance(outputs, tuple) else outputs
            v_dir = self.v.to(h.device).view(1, 1, -1)

            if self.mode == "steer":
                h_new = h + self.alpha * v_dir
            elif self.mode == "random":
                r = torch.randn_like(h)
                r = r / (torch.norm(r, dim=-1, keepdim=True) + 1e-8)
                h_new = h + self.alpha * r
            elif self.mode == "scramble_complement":
                proj = torch.sum(h * v_dir, dim=-1, keepdim=True) * v_dir
                complement = h - proj
                r = torch.randn_like(complement)
                r_ortho = r - torch.sum(r * v_dir, dim=-1, keepdim=True) * v_dir
                r_ortho = r_ortho / (torch.norm(r_ortho, dim=-1, keepdim=True) + 1e-8)
                h_new = proj + (complement + self.alpha * r_ortho)
            else:
                h_new = h

            if isinstance(outputs, tuple):
                return (h_new,) + outputs[1:]
            return h_new

    def evaluate_mode(mode):
        target_layer = encoder_layers[l_idx]
        hook_obj = InterceptionHook(v, alpha, mode)
        handle = target_layer.register_forward_hook(hook_obj)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        handle.remove()
        return out.logits, out.hidden_states[-1][0]

    # Run interventions
    steer_logits, steer_h_final = evaluate_mode("steer")
    rand_logits, rand_h_final = evaluate_mode("random")
    scramble_logits, scramble_h_final = evaluate_mode("scramble_complement")

    # Metric 1: Sequence MLM KL Divergence
    def compute_kl(pred, target):
        p = torch.nn.functional.softmax(target, dim=-1)
        log_p = torch.nn.functional.log_softmax(target, dim=-1)
        log_q = torch.nn.functional.log_softmax(pred, dim=-1)
        return torch.sum(p * (log_p - log_q), dim=-1).mean().item()

    kl_steer = compute_kl(steer_logits, orig_logits)
    kl_rand = compute_kl(rand_logits, orig_logits)
    kl_scramble = compute_kl(scramble_logits, orig_logits)

    # Metric 2: Pairwise Contact Map Topology Shift (Frobenius norm)
    def compute_contact_map(h):
        h_norm = h / (torch.norm(h, dim=-1, keepdim=True) + 1e-8)
        return torch.sigmoid(torch.matmul(h_norm, h_norm.T))

    orig_cmap = compute_contact_map(hidden_states[-1][0])
    cmap_steer_shift = torch.norm(compute_contact_map(steer_h_final) - orig_cmap).item()
    cmap_rand_shift = torch.norm(compute_contact_map(rand_h_final) - orig_cmap).item()
    cmap_scramble_shift = torch.norm(compute_contact_map(scramble_h_final) - orig_cmap).item()

    # Metric 3: Biophysical Hydrophobic-Polar Representation Shift
    hydrophobic_aas = torch.tensor([tokenizer.encode(aa, add_special_tokens=False)[0] for aa in "ACFILMVWY" if aa in tokenizer.get_vocab()])
    def compute_hydro_margin(logits):
        prob = torch.nn.functional.softmax(logits[0], dim=-1)
        return prob[:, hydrophobic_aas].sum(dim=-1).mean().item()

    orig_hydro = compute_hydro_margin(orig_logits)
    hydro_steer_shift = abs(compute_hydro_margin(steer_logits) - orig_hydro)
    hydro_rand_shift = abs(compute_hydro_margin(rand_logits) - orig_hydro)
    hydro_scramble_shift = abs(compute_hydro_margin(scramble_logits) - orig_hydro)

    metrics = {
        "kl": {"steer": kl_steer, "rand": kl_rand, "scramble": kl_scramble},
        "contact": {"steer": cmap_steer_shift, "rand": cmap_rand_shift, "scramble": cmap_scramble_shift},
        "hydro": {"steer": hydro_steer_shift, "rand": hydro_rand_shift, "scramble": hydro_scramble_shift},
    }
    return (metrics,)


@app.cell(hide_code=True)
def plot_results(go, make_subplots, metrics, mo):
    # Short x labels keep the three panels readable; the full names live in the caption.
    modes = ["steer", "rand", "scramble"]
    mode_labels = ["Steer<br>along v", "Random<br>noise r", "Noise ⊥ v<br>(v kept)"]
    mode_colors = ["#e15759", "#8c8c8c", "#4e79a7"]

    tasks = [
        ("kl", "Task 1 · Sequence MLM", "KL(orig ‖ perturbed)", ".3f"),
        ("contact", "Task 2 · Contact topology", "‖ΔC‖_F", ".2f"),
        ("hydro", "Task 3 · Biophysics", "|Δ hydrophobic mass|", ".4f"),
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[t[1] for t in tasks],
        horizontal_spacing=0.09,
    )

    for col, (key, _title, y_label, fmt) in enumerate(tasks, start=1):
        vals = [metrics[key][m] for m in modes]
        fig.add_trace(
            go.Bar(
                x=mode_labels,
                y=vals,
                marker_color=mode_colors,
                text=[format(v, fmt) for v in vals],
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
                hovertemplate="%{x}<br>" + y_label + " = %{y:.4g}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        # Dashed reference line at the random-noise baseline for like-for-like reading.
        fig.add_hline(
            y=metrics[key]["rand"],
            line_dash="dot",
            line_color="#8c8c8c",
            line_width=1.5,
            row=1,
            col=col,
        )
        fig.update_yaxes(
            title_text=y_label,
            title_font_size=12,
            rangemode="tozero",
            gridcolor="#e6e6e6",
            zerolinecolor="#cccccc",
            row=1,
            col=col,
        )
        fig.update_xaxes(tickfont_size=12, row=1, col=col)

    fig.update_layout(
        title=dict(
            text="Multi-task disruption profile · each panel has its own scale"
            "<br><sup>Dotted line = isotropic random-noise baseline. Taller than the line ⇒ more disruptive than chance.</sup>",
            font_size=17,
            x=0.02,
            xanchor="left",
        ),
        template="plotly_white",
        height=430,
        margin=dict(t=110, b=60, l=70, r=30),
        font=dict(size=13),
        bargap=0.35,
    )
    for annotation in fig.layout.annotations:
        annotation.font.size = 13

    def _ratio(key):
        base = metrics[key]["rand"]
        return "n/a" if base < 1e-12 else f"{metrics[key]['steer'] / base:.2f}×"

    caption = mo.md(
        f"""
        **Reading the panels.** Scales are deliberately independent: KL is in nats,
        contact shift is a Frobenius norm over the full L×L map, and the biophysical
        metric is a probability mass in [0, 1]. Plotting them on one axis hides the
        two smaller signals entirely.

        | Task | Steer along **v** | Random **r** | Noise ⊥ **v** | Steer ÷ random |
        |---|---|---|---|---|
        | Sequence MLM (KL) | {metrics["kl"]["steer"]:.4f} | {metrics["kl"]["rand"]:.4f} | {metrics["kl"]["scramble"]:.4f} | **{_ratio("kl")}** |
        | Contact topology (‖ΔC‖_F) | {metrics["contact"]["steer"]:.3f} | {metrics["contact"]["rand"]:.3f} | {metrics["contact"]["scramble"]:.3f} | **{_ratio("contact")}** |
        | Biophysics (Δ hydrophobic) | {metrics["hydro"]["steer"]:.5f} | {metrics["hydro"]["rand"]:.5f} | {metrics["hydro"]["scramble"]:.5f} | **{_ratio("hydro")}** |

        All three interventions inject the *same* perturbation magnitude α, so the
        ratio column is the effect size of direction **v** against chance.
        """
    )

    chart = mo.vstack([mo.as_html(fig), caption])
    chart
    return


@app.cell(hide_code=True)
def interpretation(alpha_slider, direction_dropdown, eff_layer, metrics, mo):
    def _ratio(key):
        base = metrics[key]["rand"]
        return float("inf") if base < 1e-12 else metrics[key]["steer"] / base

    kl_ratio = _ratio("kl")
    contact_ratio = _ratio("contact")
    scramble_ratio = (
        float("inf")
        if metrics["kl"]["rand"] < 1e-12
        else metrics["kl"]["scramble"] / metrics["kl"]["rand"]
    )

    THRESH_HIGH = 1.5  # steering must beat chance by 50% to count as a real effect
    THRESH_LOW = 0.7  # orthogonal noise must be 30% *cheaper* than chance

    if kl_ratio > THRESH_HIGH and contact_ratio > THRESH_HIGH:
        title = "Global Workspace hypothesis — SUPPORTED"
        badge = "🧠"
        why = (
            f"Steering along **v** disrupts sequence prediction **{kl_ratio:.2f}×** and contact "
            f"topology **{contact_ratio:.2f}×** more than matched random noise. One direction "
            "moving *both* an independent sequence task and an independent structural task is the "
            "signature of a shared broadcast channel."
        )
        caveat = "Confirm this survives a change of α and of layer before believing it."
    elif kl_ratio > THRESH_HIGH and contact_ratio <= 1.0:
        title = "Modular routing pipeline — SUPPORTED"
        badge = "🔬"
        why = (
            f"Steering hits sequence logits **{kl_ratio:.2f}×** harder than chance, yet contact "
            f"topology moves only **{contact_ratio:.2f}×** chance. The direction feeds one task and "
            "not the other, so it is a task-specific channel, not a workspace hub."
        )
        caveat = "The contact metric is a cosine-similarity proxy, not a trained contact head."
    elif scramble_ratio < THRESH_LOW:
        title = "Compact subspace hypothesis — SUPPORTED"
        badge = "🎯"
        why = (
            f"Noise confined to the complement of **v** costs only **{scramble_ratio:.2f}×** the KL "
            "of unconstrained noise of the same magnitude. Most of the model's sensitivity is "
            "concentrated in the low-dimensional subspace spanned by **v**."
        )
        caveat = "This tests where sensitivity lives, not whether v is causally used downstream."
    else:
        title = "No effect beyond isotropic sensitivity"
        badge = "⚖️"
        why = (
            f"Steering scores **{kl_ratio:.2f}×** chance on sequence and **{contact_ratio:.2f}×** on "
            "structure — within noise of the random control. The response looks like generic "
            "perturbation sensitivity, not directed routing."
        )
        caveat = "Try a larger α, a different layer, or the Jacobian direction before concluding."

    output = mo.vstack([
        mo.md("## Automated falsification verdict"),
        mo.callout(
            mo.md(
                f"""
                ### {badge} {title}

                {why}

                **Caveat.** {caveat}
                """
            ),
            kind="info",
        ),
        mo.md(
            f"""
            **Configuration under test** — direction: `{direction_dropdown.value}` ·
            layer ℓ = **{eff_layer}** · magnitude α = **{alpha_slider.value}**

            **Decision rule** — *workspace* requires both task ratios > {THRESH_HIGH}; *modular*
            requires sequence ratio > {THRESH_HIGH} while structure ratio ≤ 1.0; *compact subspace*
            requires orthogonal-noise ratio < {THRESH_LOW}.
            """
        ),
    ])
    output
    return


if __name__ == "__main__":
    app.run()
