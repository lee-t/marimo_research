# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "torch",
#     "transformers>=5.5",
#     "accelerate",
#     "jlens @ git+https://github.com/anthropics/jacobian-lens",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="wide")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    intro_md = r"""
# Run a real Jacobian lens on a live model

Companion to `jacobian_lens_workspace.py`. This notebook downloads **pre-fitted
Jacobian lenses** from [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens)
and applies them to the corresponding open models with Anthropic's
[`jlens`](https://github.com/anthropics/jacobian-lens) reference library:

```
lens_l(h) = unembed( J_l @ h ),   J_l = E[ ∂h_final / ∂h_l ]   (averaged over a corpus)
```

Pick a model, click **Load**, type a prompt, and read what each layer's residual
stream is "poised to say". Toggle **logit** to see the same readout with $J_l = I$
(the logit lens) and watch interpretable mid-layer content vanish — the paper's
Fig 51 comparison, live.

**Keep in mind.** The paper's workspace claims are demonstrated on frontier Claude
models. Small models show a weaker, noisier workspace band (when they show one at
all) — that difference is itself part of the lesson (the paper lists workspace
structure vs. model size as an open question). The two **featured** lenses,
Qwen3.6-27B and Gemma-3-12B, are much closer to the paper's regime but need a
large GPU; they are offered below with their costs labeled.
"""
    mo.md(intro_md)
    return


@app.cell(hide_code=True)
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell(hide_code=True)
def _():
    LENSES = {
        "gpt2-small — CPU-friendly (lens 13 MB, model 0.5 GB)": {
            "hf": "openai-community/gpt2",
            "lens_file": "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt",
            "lens_mb": 13,
            "model_gb": 0.5,
            "note": "CPU OK. Ungated. Fitted on 277 wikitext prompts.",
        },
        "gemma-3-270m-it — CPU-friendly (lens 14 MB, model 0.6 GB)": {
            "hf": "google/gemma-3-270m-it",
            "lens_file": "gemma-3-270m-it/jlens/Salesforce-wikitext/gemma-3-270m-it_jacobian_lens.pt",
            "lens_mb": 14,
            "model_gb": 0.6,
            "note": "CPU OK. GATED on HF (needs license acceptance + token).",
        },
        "qwen3.5-0.8b — small (lens 48 MB, model 2 GB)": {
            "hf": "Qwen/Qwen3.5-0.8B",
            "lens_file": "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt",
            "lens_mb": 48,
            "model_gb": 2,
            "note": "CPU OK, a bit slow. Ungated.",
        },
        "qwen3-1.7b — small (lens 227 MB, model 4 GB)": {
            "hf": "Qwen/Qwen3-1.7B",
            "lens_file": "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt",
            "lens_mb": 227,
            "model_gb": 4,
            "note": "CPU workable with patience. Ungated.",
        },
        "★ gemma-3-12b-it — FEATURED (lens 1.4 GB, model 24 GB)": {
            "hf": "google/gemma-3-12b-it",
            "lens_file": "gemma-3-12b-it/jlens/Salesforce-wikitext/gemma-3-12b-it_jacobian_lens.pt",
            "lens_mb": 1386,
            "model_gb": 24,
            "note": "GPU required (≥24 GB). GATED on HF. One of the two featured lenses.",
        },
        "★ qwen3.6-27b — FEATURED (lens 3.3 GB, model 54 GB)": {
            "hf": "Qwen/Qwen3.6-27B",
            "lens_file": "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
            "lens_mb": 3303,
            "model_gb": 54,
            "note": "Multi-GPU or quantized setup required. Fitted by the paper's authors (n=1000).",
        },
    }
    return (LENSES,)


@app.cell(hide_code=True)
def _(LENSES, mo):
    import torch as _torch

    lens_sel = mo.ui.dropdown(options=list(LENSES.keys()), value=list(LENSES.keys())[0],
                              label="Model + lens")
    devices = ["cpu"] + (["cuda"] if _torch.cuda.is_available() else [])
    device_sel = mo.ui.dropdown(options=devices, value=devices[0], label="Device")
    load_btn = mo.ui.run_button(label="Load model + lens")
    mo.hstack([lens_sel, device_sel, load_btn], gap=1.5)
    return device_sel, lens_sel, load_btn


@app.cell(hide_code=True)
def _(LENSES, lens_sel, mo):
    info = LENSES[lens_sel.value]
    mo.md(
        f"**{info['hf']}** — lens {info['lens_mb']} MB, model ≈ {info['model_gb']} GB. "
        f"{info['note']}"
    )
    return (info,)


@app.cell(hide_code=True)
def _():
    import jlens
    import torch
    import transformers

    return jlens, torch, transformers


@app.cell(hide_code=True)
def _(device_sel, info, is_script_mode, jlens, load_btn, mo, torch, transformers):
    lm = tok = lens = None
    if is_script_mode or load_btn.value:
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            info["hf"], torch_dtype="auto", device_map=device_sel.value
        )
        tok = transformers.AutoTokenizer.from_pretrained(info["hf"])
        lm = jlens.from_hf(hf_model, tok)
        lens = jlens.JacobianLens.from_pretrained(
            "neuronpedia/jacobian-lens", filename=info["lens_file"]
        )
        status_md = mo.md(
            f"Loaded **{info['hf']}** ({lm.n_layers} layers) on `{device_sel.value}` and lens "
            f"`{info['lens_file']}` — fitted source layers: "
            f"{lens.source_layers[0]}…{lens.source_layers[-1]} ({len(lens.source_layers)} layers), "
            f"n_prompts={lens.n_prompts}."
        )
    else:
        status_md = mo.md(
            "*Click **Load model + lens** to download and initialize (first load caches "
            "to your HF cache; nothing else runs until then).*"
        )
    status_md
    return lens, lm, status_md, tok


@app.cell(hide_code=True)
def _(mo):
    prompt_area = mo.ui.text_area(
        value="Fact: The number of legs on the animal that spins webs is",
        label="Prompt (try one with an unspoken intermediate)",
        rows=3,
        full_width=True,
    )
    run_btn = mo.ui.run_button(label="Run readout")
    mo.vstack([prompt_area, run_btn], gap=0.6)
    return prompt_area, run_btn


@app.cell(hide_code=True)
def _(is_script_mode, lens, lm, mo, prompt_area, run_btn, torch):
    mo.stop(lens is None, mo.md("*Load a model and lens first.*"))
    mo.stop(not (is_script_mode or run_btn.value),
            mo.md("*Edit the prompt, then click **Run readout**.*"))

    with torch.no_grad():
        j_logits, model_logits, input_ids = lens.apply(
            lm, prompt_area.value, use_jacobian=True
        )
        lg_logits, _, _ = lens.apply(lm, prompt_area.value, use_jacobian=False)
    n_pos = j_logits[lens.source_layers[0]].shape[0]
    mo.md(f"Read out {n_pos} positions × {len(j_logits)} layers. **{model_logits.shape[-1]}**-token vocab.")
    return input_ids, j_logits, lg_logits, model_logits, n_pos


@app.cell(hide_code=True)
def _(lens, mo, n_pos):
    mode_sel = mo.ui.radio(
        options={"jacobian": "Jacobian lens", "logit": "logit lens (J = I)"},
        value="jacobian",
        label="Readout mode",
        inline=True,
    )
    layer_opts = {f"layer {l}": l for l in sorted(lens.source_layers, reverse=True)}
    layer_sel = mo.ui.dropdown(
        options=layer_opts,
        value=list(layer_opts.keys())[min(2, len(layer_opts) - 1)],
        label="Inspect layer",
    )
    pos_sel = mo.ui.slider(start=0, stop=max(n_pos - 1, 0), step=1, value=max(n_pos - 1, 0),
                           label="Inspect position")
    mo.hstack([mode_sel, layer_sel, pos_sel], gap=2)
    return layer_sel, mode_sel, pos_sel


@app.cell(hide_code=True)
def _():
    import html as _html

    def esc(t):
        return _html.escape(str(t))

    GRID_CSS = """
    <style>
    .jl-wrap { overflow-x: auto; max-width: 100%; }
    table.jl { border-collapse: collapse; font-family: monospace; font-size: 9px; }
    table.jl td, table.jl th { padding: 1px 2px; text-align: center; border: 1px solid #f3f4f6; }
    table.jl td.tok { color: #9ca3af; font-size: 8px; }
    table.jl td.sel { outline: 2px solid #2563eb; outline-offset: -2px; }
    table.jl th.rowlbl { color: #6b7280; font-weight: normal; padding-right: 4px; }
    </style>
    """

    return GRID_CSS, esc


@app.cell(hide_code=True)
def _(GRID_CSS, esc, input_ids, j_logits, layer_sel, lg_logits, lens, mode_sel, mo,
      model_logits, n_pos, pos_sel, tok, torch):
    layers = sorted(lens.source_layers, reverse=True)
    logits = j_logits if mode_sel.value == "jacobian" else lg_logits

    def topk_at(logits_by_layer, layer, pos, k=6):
        probs = torch.softmax(logits_by_layer[layer][pos], dim=-1)
        vals, idxs = torch.topk(probs, k)
        return [(tok.decode([i]), float(v)) for v, i in zip(vals, idxs)]

    # grid of top-1 tokens
    header = ['<tr><th class="rowlbl"></th>']
    display_tokens = [tok.decode([i]) for i in input_ids[0].tolist()][:n_pos]
    for i, t in enumerate(display_tokens):
        cls = ' class="tok sel"' if i == pos_sel.value else ' class="tok"'
        label = esc(t.replace("\n", "↵") if t.strip() else "␣")
        header.append(f"<td{cls}>{label}</td>")
    header.append("</tr>")
    rows = ["".join(header)]
    for layer in layers:
        tds = [f'<th class="rowlbl">{layer}</th>']
        for pos in range(n_pos):
            top = topk_at(logits, layer, pos, k=3)
            top1 = top[0][0].replace("\n", "↵")
            if not top1.strip():
                top1 = "␣"
            tip = esc(" | ".join(f"{w.strip() or '␣'} {p * 100:.1f}%" for w, p in top))
            cls = ' class="sel"' if (pos == pos_sel.value and layer == layer_sel.value) else ""
            tds.append(f'<td{cls} title="{tip}">{esc(top1[:8])}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    grid_html = mo.Html(
        GRID_CSS + '<div class="jl-wrap"><table class="jl">' + "".join(rows) + "</table></div>"
    )

    # side panel: top-k at the selected cell + the model's actual next-token top-5
    import matplotlib.pyplot as plt

    entries = topk_at(logits, layer_sel.value, pos_sel.value, k=8)
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6), dpi=110)
    ax = axes[0]
    labels = [w.strip() or "␣" for w, _ in entries][::-1]
    vals = [p for _, p in entries][::-1]
    ax.barh(range(len(vals)), vals, color="#f97316" if mode_sel.value == "jacobian" else "#9ca3af")
    ax.set_yticks(range(len(vals)), labels, fontsize=7)
    ax.set_title(f"{mode_sel.value} · pos {pos_sel.value} · layer {layer_sel.value}", fontsize=8)

    ax = axes[1]
    probs = torch.softmax(model_logits[pos_sel.value], dim=-1)
    vals5, idxs5 = torch.topk(probs, 5)
    labels5 = [tok.decode([i]).strip() or "␣" for i in idxs5][::-1]
    ax.barh(range(5), [float(v) for v in vals5][::-1], color="#2563eb")
    ax.set_yticks(range(5), labels5, fontsize=7)
    ax.set_title(f"model next-token at pos {pos_sel.value}", fontsize=8)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"**Prompt tokens:** `{ ' '.join(display_tokens) }` — hover any cell for its top-3; "
                "the blue-outlined cell is the inspected (position, layer)."
            ),
            mo.hstack([grid_html, fig], widths=[3, 2], gap=1.5, align="start"),
        ]
    )
    return display_tokens,


@app.cell(hide_code=True)
def _(mo):
    notes_md = r"""
## Notes and next steps

- **Featured lenses.** The two headline lenses from the paper's open release are
  `qwen3.6-27b` (3.3 GB lens, fitted on n=1000 sequences by the paper's authors) and
  `gemma-3-12b-it` (1.4 GB lens). Select them above only if you have the hardware —
  the model weights alone are ~54 GB and ~24 GB respectively in bf16.
- **No GPU / no downloads?** Explore readouts in the browser instead:
  [neuronpedia.org/jlens](https://neuronpedia.org/jlens) or the
  [ktangri/jacobian-lens-playground](https://huggingface.co/spaces/ktangri/jacobian-lens-playground)
  Space.
- **Fit your own lens.** `jlens.fit(model, prompts=my_prompts, checkpoint_path=...)`;
  ~100 sequences of 128 tokens is usable, quality saturates by ~1000 (paper §9.3).
- **What to try.** Prompts with an unspoken intermediate work best:
  "Fact: The number of legs on the animal that spins webs is" (spider),
  "The capital of the country shaped like a boot is" (Italy),
  a short ASCII drawing (nose/face). Then flip to **logit** mode and compare.
- **Expected difference from the paper.** On gpt2-small the interpretable band is
  narrow and noisy; on the featured models it is much wider. The workspace is a
  property that *grows with the model*.
"""
    mo.md(notes_md)
    return


if __name__ == "__main__":
    app.run()
