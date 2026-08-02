# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
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
    header_md = r"""
# Verbalizable representations form a global workspace in language models
**A figure-by-figure walkthrough of Gurnee, Sofroniew, … Lindsey (Anthropic, July 2026)** —
[original article](https://transformer-circuits.pub/2026/workspace/)

The paper's claim, in one sentence:

> The small slice of a language model's activations that is **verbalizable** — poised to be
> *said*, if asked — turns out to be exactly the slice the model can **report**, **summon**,
> **reason with**, and **reroute**. The other ~90–95% of its activations runs automatically,
> like the unconscious bulk of the mind.

The authors find this slice with a new probe, the **Jacobian lens** (J-lens): for each layer
$\ell$, average the linearized effect of the residual stream $h_\ell$ on the model's present
and future outputs over 1,000 contexts,

$$
J_\ell \;=\; \mathbb{E}_{t,\, t' \geq t,\, \text{prompt}} \Big[\, \partial h_{\text{final},t'} / \partial h_{\ell,t} \,\Big],
\qquad
\text{lens}(h_\ell) \;=\; \text{softmax}\big(W_U \, \text{norm}(J_\ell h_\ell)\big).
$$

The lens reads any activation as a **ranked list of vocabulary tokens**: the words that
activation is, on average across contexts, disposed to make the model say. Those token
directions (the rows of $W_U J_\ell$) span the **J-space** — the candidate workspace.

**Every chart below is rebuilt from the paper's own published figure data** (bundled with
this notebook; see `data/jacobian_lens_workspace/MANIFEST.json`). Nothing here is simulated.
"""
    mo.md(header_md)
    return


@app.cell(hide_code=True)
def _(mo):
    howto_md = r"""
**How to read this notebook.** Pick an **act** of the paper, then a **figure**. Each page
states the claim, shows the chart (rebuilt from the paper's data), and tells you what to
look at. Layer numbers are reindexed to 0–100 (% of depth, 25 sampled layers); the
**workspace band ≈ L38–92** is shaded where relevant. In token readouts, `␣` marks a space
and `↑` the beginning of a turn. Figures 1, 2, 5 and 28 are described but not rebuilt —
their interactive source data was not published in machine-readable form.
"""
    mo.md(howto_md)
    return


@app.cell(hide_code=True)
def _():
    import json
    import re
    from pathlib import Path

    import numpy as np

    return Path, json, np, re


@app.cell(hide_code=True)
def _():
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    return plt, matplotlib


@app.cell(hide_code=True)
def _(mo, Path, json):
    DATA_DIR = mo.notebook_dir() / "data" / "jacobian_lens_workspace"
    D = {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(DATA_DIR.glob("*.json"))
        if p.stem != "MANIFEST"
    }
    D["_meta"] = {"data_dir": DATA_DIR}
    return D, DATA_DIR


@app.cell(hide_code=True)
def _(mo):
    act_sel = mo.ui.dropdown(
        options=[
            "Method: the Jacobian lens",
            "1 · Verbal report",
            "2 · Directed modulation",
            "3 · Internal reasoning",
            "4 · Flexible generalization",
            "5 · Selectivity: flexible vs. automatic",
            "Structure of the J-space",
            "Alignment auditing",
            "The Assistant's point of view",
            "Counterfactual reflection training",
        ],
        value="Method: the Jacobian lens",
        label="Act of the paper",
    )
    act_sel
    return (act_sel,)


@app.cell(hide_code=True)
def _(mo, act_sel, FIGURES):
    fig_titles = [f["title"] for f in FIGURES if f["act"] == act_sel.value]
    fig_sel = mo.ui.dropdown(options=fig_titles, value=fig_titles[0], label="Figure")
    fig_sel
    return (fig_sel,)


@app.cell(hide_code=True)
def _(fig_sel, FIGURES):
    fig_meta = next(f for f in FIGURES if f["title"] == fig_sel.value)
    return (fig_meta,)


@app.cell(hide_code=True)
def _(mo, D):
    def build_controls(key):
        """Construct the interactive widgets for a figure page.

        Kept as a callable so the script-mode smoke test can instantiate every
        page's controls (validating their defaults), not just the default page.
        """
        w = {}

        if key == "lens_compare":
            panels = D["lens_compare"]["panels"]
            w["panel"] = mo.ui.dropdown(
                options={p["title"]: i for i, p in enumerate(panels)},
                value=panels[0]["title"],
                label="Prompt",
            )
            w["lens"] = mo.ui.radio(
                options=["jacobian", "logit", "tuned"], value="jacobian", label="Lens", inline=True
            )
            layers = [r["layer"] for r in panels[0]["rows"]]
            layer_opts = {f"layer index {l} (≈ L{round(l / 24 * 100)})": l for l in layers}
            w["layer"] = mo.ui.dropdown(
                options=layer_opts,
                value=[k for k, v in layer_opts.items() if v == 13][0],
                label="Readout at layer",
            )
            w["pos"] = mo.ui.slider(start=0, stop=56, step=1, value=9, label="Token position")
        elif key == "modulation_readout":
            panels = D["modulation_readout"]["panels"]
            w["panel"] = mo.ui.dropdown(
                options={p["title"]: i for i, p in enumerate(panels)},
                value=panels[0]["title"],
                label="Mental task",
            )
            layers = [l["layer"] for l in panels[0]["layers"]]
            layer_opts = {f"layer index {l} (≈ L{round(l / 24 * 100)})": l for l in layers}
            w["layer"] = mo.ui.dropdown(
                options=layer_opts,
                value=[k for k, v in layer_opts.items() if v == 13][0],
                label="Readout at layer",
            )
            w["pos"] = mo.ui.slider(start=0, stop=47, step=1, value=panels[0]["key_pos"],
                                    label="Token position")
        elif key == "swap_explorer":
            options = {}
            for i, p in enumerate(D["latent_patching"]["panels"]):
                options[f"reasoning — {p['title']}"] = ("latent_patching", i)
            for i, p in enumerate(D["flex_gen_example"]["panels"]):
                options[f"generalization — {p['title']} (France→China)"] = ("flex_gen_example", i)
            w["panel"] = mo.ui.dropdown(options=options, value=next(iter(options)),
                                        label="Swap trial")
        elif key == "selectivity_language":
            cols = D["selectivity_language"]["polished_ex"]["cols"]
            w["col"] = mo.ui.dropdown(
                options={c["title"]: i for i, c in enumerate(cols)}, value=cols[2]["title"],
                label="Task variant",
            )
        elif key == "selectivity_linecount":
            d = D["selectivity_linecount"]
            cond_opts = {d["condLabel"][c].replace("\n", " "): c for c in d["conds"]}
            w["cond"] = mo.ui.dropdown(
                options=cond_opts,
                value=[k for k, v in cond_opts.items() if v == "direct"][0],
                label="Task variant",
            )
        elif key == "ablation_examples":
            ex = D["ablation_examples"]
            ex_opts = {f"{i + 1}. {e['title']}": i for i, e in enumerate(ex)}
            w["example"] = mo.ui.dropdown(
                options=ex_opts, value=next(iter(ex_opts)), label="Example"
            )
        elif key == "selfreport":
            ex = D["selfreport"]["excerpts"]["soc"]
            w["excerpt"] = mo.ui.dropdown(
                options={f"Prompt {i + 1}": i for i in range(len(ex))},
                value="Prompt 1",
                label="Transcript excerpt",
            )
        elif key == "misalign_lens":
            panels = D["misalign_lens"]["panels"]
            w["panel"] = mo.ui.dropdown(
                options={p["title"]: i for i, p in enumerate(panels)},
                value=panels[0]["title"],
                label="Readout position",
            )
        elif key == "rm_bias":
            w["example"] = mo.ui.dropdown(
                options=["neutral", "quirk_eliciting", "goal_probing"], value="goal_probing",
                label="Prompt set",
            )
        elif key == "roleplay":
            w["panel"] = mo.ui.dropdown(
                options=["disclaimer", "fictional", "roleplay"], value="fictional",
                label="Workspace word",
            )
        elif key == "pref_violation":
            panels = D["pref_violation_lens"]["panels"]
            pref_opts = {p["label"]: p["key"] for p in panels}
            w["panel"] = mo.ui.dropdown(
                options=pref_opts, value=panels[0]["label"], label="Word family"
            )

        return w

    return (build_controls,)


@app.cell(hide_code=True)
def _(build_controls, fig_meta):
    # Per-figure interactive controls, rebuilt whenever the figure changes.
    w = build_controls(fig_meta["key"])
    return (w,)


@app.cell(hide_code=True)
def _(w):
    ctrl = {k: v.value for k, v in w.items()}
    return (ctrl,)


@app.cell(hide_code=True)
def _(mo, D, RENDERERS, ctrl, fig_meta, w):
    fig_obj = RENDERERS[fig_meta["key"]](D, ctrl)
    ctrl_col = mo.vstack(w.values(), gap=0.4) if w else None
    body = (
        mo.hstack([ctrl_col, fig_obj], widths=[1, 3], gap=2, align="start")
        if ctrl_col is not None
        else fig_obj
    )
    mo.vstack(
        [
            mo.md(f"### {fig_meta['title']}\n**Claim.** {fig_meta['claim']}"),
            body,
            mo.md(
                f"**Paper caption ({fig_meta['figs']}).** *{fig_meta['caption']}*\n\n"
                f"**Look at:** {fig_meta['look']}"
            ),
        ],
        gap=1.2,
    )
    return (fig_obj,)


@app.cell(hide_code=True)
def _(mo):
    limitations_md = r"""
## Limitations the paper itself flags

- **Single-token vocabulary.** The lens has one vector per vocab token, so concepts without
  a single-token name ("prompt injection", "Golden Gate") appear only as fragments. The
  appendix's template/oracle lenses (Figs 61–64) extend to multi-token concepts.
- **A bag of concepts.** Readouts say *which* concepts are present, not how they are bound
  together; the workspace likely has structure ("roles", relations) a flat readout can't see.
- **Early layers are ambiguous.** The empty first third may be a real absence of workspace
  content *or* a failure of the averaged Jacobian to read early-layer geometry.
- **No predictive criterion.** The paper shows flexible tasks route through the J-space and
  automatic ones don't, but cannot predict in advance which side a new task falls on.
- **Workspace vs. motor.** The late-layer boundary, where readouts collapse onto the imminent
  output, was found empirically, not from a principled definition.

And one to keep in mind throughout: the headline experiments are on **frontier Claude models
(Haiku/Sonnet/Opus 4.5+)**. How much workspace structure smaller or open models have is an
open question the paper explicitly raises.
"""
    mo.md(limitations_md)
    return


@app.cell(hide_code=True)
def _(mo):
    live_md = r"""
## Try a real Jacobian lens

The companion notebook **`jacobian_lens_live.py`** loads pre-fitted lenses from
[neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens) with
Anthropic's [`jlens`](https://github.com/anthropics/jacobian-lens) library and applies them
to live models — from CPU-friendly `gpt2-small` (13 MB lens) up to the two featured lenses,
**Qwen3.6-27B** (3.3 GB lens, GPU-scale) and **Gemma-3-12B** (1.4 GB lens). There is also a
hosted playground at
[ktangri/jacobian-lens-playground](https://huggingface.co/spaces/ktangri/jacobian-lens-playground)
and a browser viewer on [neuronpedia.org/jlens](https://neuronpedia.org/jlens).

**Source.** Wes Gurnee*, Nicholas Sofroniew*, … Jack Lindsey* (Anthropic),
*Verbalizable Representations Form a Global Workspace in Language Models*, July 6, 2026.
transformer-circuits.pub/2026/workspace. Figure data reproduced from the article's published
assets; the article is the canonical reference for full captions and methods.
"""
    mo.md(live_md)
    return


# ---------------------------------------------------------------------------
# Figure registry: the narrative for every page.
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _():
    FIGURES = [
        # ---------------- Method ----------------
        {
            "act": "Method: the Jacobian lens",
            "key": "fig4_schematic",
            "title": "Fig 4 — The Jacobian lens, schematic",
            "figs": "Fig 4",
            "claim": "Replace every layer after ℓ with one averaged linear map $J_\\ell$, then unembed. "
                     "Because the average is over 1,000 contexts, what survives is the activation's "
                     "*general disposition to be said* — not its use in this particular context.",
            "caption": "(A) $J_\\ell$ is computed by backpropagating from the final-layer residual "
            "stream to $h_\\ell$ and averaging the Jacobians over token positions and a corpus of "
            "prompts. (B) Reading from the lens replaces all downstream layers with $J_\\ell$ plus the "
            "model's own unembedding, yielding a ranked token list. (C) Patching in lens coordinates "
            "permutes two J-lens coordinates and writes the result back, leaving the orthogonal "
            "component untouched.",
            "look": "Panel (C) is the workhorse intervention used throughout the paper: the **lens-coordinate "
            "swap**. Everything below either *reads* with (B) or *writes* with (C).",
        },
        {
            "act": "Method: the Jacobian lens",
            "key": "lens_compare",
            "title": "Figs 3 & 51 — The J-lens vs. the logit and tuned lenses",
            "figs": "Figs 3, 51",
            "claim": "On prompts with unspoken intermediates, the J-lens surfaces the intermediate "
                     "(spider, fight, nose, red) at mid layers, where the logit lens reads noise and "
                     "the tuned lens skips ahead to the answer.",
            "caption": "Top-1 readouts from the J-lens, logit lens, and tuned lens at each layer, on "
            "prompts whose correct answer depends on an intermediate concept the model never says. "
            "Colored cells: the readout names the prompt's key concepts.",
            "look": "Switch the lens to **logit** and watch the interpretable middle disappear; switch to "
            "**tuned** and watch it jump straight to the output token. The **protein** and **face** prompts "
            "show the lens recognizing a protein's function and an ASCII face's nose. Try the layer/position "
            "selectors to inspect any cell's full top-k.",
        },
        # ---------------- 1 Verbal report ----------------
        {
            "act": "1 · Verbal report",
            "key": "verbal_report",
            "title": "Fig 6 — The workspace names what the model is about to say",
            "figs": "Fig 6",
            "claim": "Asked to 'think of a sport', the lens on the final colon reads **Soccer** before "
                     "the model says it — and swapping the Soccer lens coordinate for Rugby makes the "
                     "model report 'Rugby'. What is in the workspace determines what gets reported.",
            "caption": "Left: J-lens and next-token readouts on a 'think of a {category}' prompt, with "
            "and without a lens-coordinate swap. Right: Spearman correlation between lens ordering and "
            "output ordering over 14 categories at three layers (top), and each swap target's output "
            "rank before vs. after the swap (bottom).",
            "look": "The right scatter: targets start at rank ≥ 11 (log scale) and land at rank ≈ 1–10 "
            "after the swap. The correlation grows toward L92 as the workspace hands off to the motor layers.",
        },
        {
            "act": "1 · Verbal report",
            "key": "verbal_introspection",
            "title": "Fig 7 — Injected concepts become reportable on demand",
            "figs": "Fig 7",
            "claim": "Steering a J-lens vector (e.g. **lightning**) into the *user* turn does not make "
                     "the model say 'lightning' — until it is asked whether it detects an injected thought. "
                     "J-space contents are verbalizable, not compulsively verbalized.",
            "caption": "Injecting a concept across the user turn makes it reportable when the model "
            "introspects. The plot tracks the median reciprocal rank of the injected concept over n=100 "
            "concepts against steering strength, at the report slot vs. all other positions.",
            "look": "The blue curve (the report slot) shoots to rank 1 while the gray curve (every other "
            "position) stays near zero: the concept sits quietly in the workspace until a question pulls it out.",
        },
        {
            "act": "1 · Verbal report",
            "key": "verbal_decomp",
            "title": "Fig 8 — Only the J-space component is reportable",
            "figs": "Fig 8",
            "claim": "Split a concept vector into its J-space component (~6–7% of variance) and the "
                     "remainder (~93%). The tiny J-space part drives verbal report almost as well as pure "
                     "lens vectors; the huge remainder barely does — and what little it does routes "
                     "back through the J-space.",
            "caption": "Left: a concept mean vector split into top-16 J-lens atoms and remainder. Middle: "
            "swap success (target reaching top-5) per component. Right: introspection-report success per "
            "component over a steering-strength sweep, with the J-clamped control.",
            "look": "The near-zero black bars: once the J-space coordinates are clamped, the non-J-space "
            "component's residual effect on report collapses to zero. Report lives in the J-space.",
        },
        # ---------------- 2 Directed modulation ----------------
        {
            "act": "2 · Directed modulation",
            "key": "modulation_readout",
            "title": "Fig 9 — Holding a thought while copying text",
            "figs": "Fig 9",
            "claim": "Told to copy a sentence while concentrating on citrus fruits, the lens at the "
                     "'ook' of 'crookedly' reads **orange** — and while mentally evaluating 3² − 2 it "
                     "progresses math → nine → seven. The model thinks the instructed thought in its "
                     "workspace, invisible in its output.",
            "caption": "Top J-lens readouts on prompts instructing the model to hold a concept in mind "
            "or mentally compute, while copying an unrelated sentence. Readouts shown across six layers "
            "per position.",
            "look": "Slide the layer down (24 → 13): the readout morphs from the surface token (edly) "
            "backward into the hidden task (orange, nine, seven, forty). Slide the position: the workspace "
            "content is concentrated where the mental work happens.",
        },
        {
            "act": "2 · Directed modulation",
            "key": "modulation_lines",
            "title": "Fig 10 — 'Think about X' vs. 'Ignore X'",
            "figs": "Fig 10",
            "claim": "Across three task families, instructed targets appear in the workspace at high "
                     "rates — and 'ignore X' still activates X well above the ~0% no-instruction baseline. "
                     "Models have top-down control, but it is imperfect, echoing the human 'white bear' effect.",
            "caption": "Directed modulation by model size, task family, and instruction condition. "
            "Category/math: a trial is positive if the target reaches J-lens top-1 anywhere. Line width: "
            "precision of top-1 numeric readouts. Baseline ≤ 2% everywhere (omitted).",
            "look": "The right panel ('Ignore X') is well below the left but far above zero — suppression "
            "itself loads the concept. Rates also grow with model size.",
        },
        # ---------------- 3 Internal reasoning ----------------
        {
            "act": "3 · Internal reasoning",
            "key": "swap_explorer",
            "title": "Figs 13 & 18 — Swapping one workspace concept for another",
            "figs": "Figs 13, 18",
            "claim": "Exchanging the lens coordinates of an unspoken intermediate (spider↔ant, "
                     "fight↔light, big↔long) or an argument (France↔China) redirects the model's answer "
                     "to match — the swapped vector is read correctly by whatever circuit consumes it.",
            "caption": "Lens-coordinate swaps at every position: the model's top-5 next-token log-probs "
            "before (clean) and after (swapped). Reasoning trials need an unspoken intermediate; "
            "generalization trials apply four different functions to the same swapped argument.",
            "look": "The #1 token in each panel flips to the *swapped-in* concept's answer: 8→6 legs, "
            "'coming fight'→'morning light', 大→长, Paris→Beijing. The planning case (poetry) shows a "
            "*planned future token* steering earlier word choices.",
        },
        {
            "act": "3 · Internal reasoning",
            "key": "multihop_swap",
            "title": "Fig 15 — Intermediates are computed before answers",
            "figs": "Fig 15",
            "claim": "Over 50 two-hop prompts, intermediate swaps flip the answer in 54–70% of trials "
                     "and take effect a median of ~17% of depth *earlier* than answer swaps — the model "
                     "represents the intermediate before it has the answer.",
            "caption": "Left: fraction of successful top-1 swaps across models. Right: log-prob difference "
            "produced by swapping intermediates vs. answers at different depths (Sonnet 4.5, shaded SE).",
            "look": "The orange (intermediate) curve peels away from the blue (answer) curve several "
            "layers earlier — ruling out the confound that the intermediate's vector merely smuggles in the answer.",
        },
        {
            "act": "3 · Internal reasoning",
            "key": "probe_swap",
            "title": "Fig 16 — The J-space component of a probe carries the causal effect",
            "figs": "Fig 16",
            "claim": "Fit a probe for the unspoken intermediate *without* the lens, then swap along its "
                     "J-space part (10–15% of variance) vs. its remainder. The J-space part flips answers "
                     "at 61%, matching raw lens vectors (60%); the remainder manages 28% — and 6% once "
                     "the J-space is clamped.",
            "caption": "Left: a two-hop prompt with unspoken intermediate China; its probe split into "
            "J-space component and remainder. Right: fraction of trials each swap places the target answer "
            "at top-1 over n=90 prompts (Wilson 95% CIs; hatched = complementary component clamped).",
            "look": "Same pattern as Fig 8 but for *reasoning* rather than report: the causal content "
            "of the working representation is the small J-space slice, not the big remainder.",
        },
        # ---------------- 4 Flexible generalization ----------------
        {
            "act": "4 · Flexible generalization",
            "key": "flex_gen_systematic",
            "title": "Fig 19 — One vector, valid for many functions",
            "figs": "Fig 19",
            "claim": "The same France→China swap works across 16 function templates (capital, language, "
                     "continent, currency; seasons, habitats, successors…): a workspace vector is a "
                     "*broadcast argument* that many downstream circuits can read — most reliably where "
                     "the argument was strongly loaded to begin with.",
            "caption": "Left: per-template fraction of 12 swap pairs whose target reaches top-1 (● α=1, "
            "× α=2). Right: swap effect size vs. the argument's workspace loading (cosine similarity), "
            "with regression fit.",
            "look": "The right scatter: swap effect grows with **workspace loading** — failures cluster "
            "where the source concept was barely in the workspace (number words). Countries, with the "
            "highest loading, swap almost perfectly.",
        },
        # ---------------- 5 Selectivity ----------------
        {
            "act": "5 · Selectivity: flexible vs. automatic",
            "key": "selectivity_language",
            "title": "Fig 20 — Report needs the J-space; fluency doesn't",
            "figs": "Fig 20",
            "claim": "A Spanish passage keeps being continued in fluent Spanish even with the **Spanish** "
                     "lens vector swapped to **French** — while naming the language, a famous author, "
                     "'hello', or the pre-Euro currency all flip to French. 'Spanish' is *present* in the "
                     "workspace in all tasks, but only report/flexible tasks *read* it.",
            "caption": "Top: one passage under four task conditions; tokens shaded where 'Spanish' is in "
            "the J-lens top-3, with answers before and after a Spanish→French swap. Bottom: over n=8 "
            "passages — task success (a), language present in lens (b), answer following the swap (c).",
            "look": "Panel (b) is flat — the concept is always in the lens — while panel (c) splits "
            "cleanly: report and flexible computation follow the swap; continuation and anomaly detection "
            "ignore it. Presence ≠ causal use.",
        },
        {
            "act": "5 · Selectivity: flexible vs. automatic",
            "key": "selectivity_linecount",
            "title": "Fig 21 — The count enters the workspace only when asked",
            "figs": "Fig 21",
            "claim": "Wrapping lines at the right column requires tracking character counts, yet during "
                     "plain continuation *no number tokens appear in the lens at all* and a 40s→60s swap "
                     "doesn't move the wrap point. Ask 'how many characters?' and the count appears; ask "
                     "for its first letter and it appears even more strongly.",
            "caption": "Prompts with different questions about the same passage; tokens shaded where "
            "two-digit integers/number-words are in the J-lens top-3. Bottom: task success (a), count "
            "presence in lens (b), answer following the swapped value (c) over n=11 passages.",
            "look": "Panel (b): essentially zero under linewrap, moderate under direct report, highest "
            "under the first-letter question where the count feeds a second operation. Automatic "
            "computation can use information that never enters the workspace.",
        },
        {
            "act": "5 · Selectivity: flexible vs. automatic",
            "key": "ablation_strength",
            "title": "Fig 22 — Ablating the whole J-space: dose calibration",
            "figs": "Fig 22",
            "claim": "Projecting out the top-10 J-lens directions at every position, over three layer "
                     "bands (light/medium/heavy), crushes multi-hop reasoning accuracy (0.98 → 0.06) "
                     "while 65–87% of ordinary next-token predictions stay intact. Ablating random "
                     "directions does nothing.",
            "caption": "The three J-space ablation strengths, defined by the layer band over which the "
            "top-10 J-lens directions are projected out, plus a random-direction control. Multihop "
            "accuracy vs. pretraining top-1 match.",
            "look": "The random control row: same layer band, same number of directions, essentially no "
            "effect. The disruption is specific to the J-space, not to perturbing the residual stream.",
        },
        {
            "act": "5 · Selectivity: flexible vs. automatic",
            "key": "ablation_examples",
            "title": "Fig 23 — What the ablation disrupts, qualitatively",
            "figs": "Fig 23",
            "claim": "On pretraining-like text, the predictions J-space ablation breaks are exactly those "
                     "that require an *unspoken characterization of context* — a clinical paper's topic, "
                     "a speech's era, a botanist's nationality. The model stays fluent but reverts to "
                     "generic priors.",
            "caption": "Passages with the key token marked, the ten J-lens directions ablated at that "
            "position, and the next-token distribution before and after ablation.",
            "look": "The before→after top-5 at the highlighted token: a specific, context-assembled "
            "prediction collapses into something generic. The ablated atom list reads like the missing "
            "concept (radio, lung, thermal, electrode…).",
        },
        {
            "act": "5 · Selectivity: flexible vs. automatic",
            "key": "ablation_bars",
            "title": "Fig 24 — Ablating the workspace across 14 tasks",
            "figs": "Fig 24",
            "claim": "Classification, extraction and acceptability tasks are untouched even by heavy "
                     "ablation; recall-and-generate tasks (cipher, analogy, summarization, TriviaQA, "
                     "multihop, translation, poetry) fall *below unablated Haiku*. GSM8K survives when "
                     "solved with chain-of-thought — the page replaces the workspace.",
            "caption": "Task score under light/medium/heavy ablation, normalized to unablated Sonnet "
            "4.5; gray bars show unablated Haiku 4.5 as a smaller-model reference.",
            "look": "The left group (shallow tasks) hugs 1.0; the right group (generation grounded in "
            "inferred content) collapses with dose. Compare GSM8k CoT vs. no-CoT: writing out steps "
            "externalizes what the workspace would otherwise hold.",
        },
        {
            "act": "5 · Selectivity: flexible vs. automatic",
            "key": "selfreport",
            "title": "Fig 25 — Ablation flattens experiential reports",
            "figs": "Fig 25",
            "claim": "Narrating its stream of consciousness, an ablated model stays coherent but turns "
                     "detached and mechanical; the experiential-language score collapses (0.84 → 0.19 on "
                     "Sonnet 4.5) while matched-norm controls do nothing. During normal narration the "
                     "workspace itself is full of *thinking*, *thoughts*, *feeling*, *conscious*.",
            "caption": "J-space ablation and matched-norm controls while the model narrates its stream "
            "of consciousness. A: representative responses. B: experiential-language rate by model and "
            "condition. C: workspace contents during unablated narration vs. the final layer.",
            "look": "Panel C: *thinking* occupies 58% of workspace slots during narration but only ~2% "
            "of final-layer readouts — the workspace carries the phenomenological vocabulary that the "
            "output only samples. Compare baseline vs. ablated transcripts in the excerpt selector.",
        },
        # ---------------- Structure ----------------
        {
            "act": "Structure of the J-space",
            "key": "cka",
            "title": "Fig 27 — Sensory → workspace → motor",
            "figs": "Fig 27",
            "claim": "Comparing J-space geometry across layers (CKA) reveals three blocks: an early "
                     "sensory regime (syntax, parsing), a long middle **workspace** block, and a late "
                     "motor block aligned with the imminent output.",
            "caption": "Pairwise similarity of J-lens vector geometry across the 25 sampled layers, "
            "with the three functional regions marked. Workspace band: L38–92 (layer indices 9–22).",
            "look": "The block structure: early layers share one geometry, the middle third another, "
            "the last few layers a third. Everything 'workspace-like' in this notebook happens in the "
            "middle block.",
        },
        {
            "act": "Structure of the J-space",
            "key": "ignition",
            "title": "Fig 29 — Ambiguous input 'ignites' at the workspace onset",
            "figs": "Fig 29",
            "claim": "Feed the model a blended embedding (Germany↔France) and sweep the mixture α: "
                     "early layers track the blend smoothly, but from the workspace onset (≈ L38) the "
                     "activation snaps to one country or the other — a sharp, bistable *ignition* like "
                     "the one global workspace theory predicts at workspace entry.",
            "caption": "A concept token's embedding is replaced by (1−α)e_B + αe_A. (B) Projection "
            "share along the pure-B↔pure-A axis vs. α and layer. (C) The same restricted to the two "
            "concepts' J-lens directions. Histograms: responses at maximally ambiguous α are bimodal, "
            "especially in the J-space.",
            "look": "The heatmaps turn from gradient (early) to sharp boundary (after the shaded "
            "onset). In the histograms, the J-space (orange) piles up at the extremes — all-or-none — "
            "while the full activation (gray) keeps middling values longer.",
        },
        {
            "act": "Structure of the J-space",
            "key": "capacity_occupancy",
            "title": "Fig 30 — Capacity: tens of concepts, <10% of variance",
            "figs": "Fig 30",
            "claim": "Sparse decomposition finds the J-space's occupancy plateaus at ~25 active lens "
                     "vectors per position, and those explain at most ~9% of activation variance beyond "
                     "random controls. The workspace is a *thin* slice of the stream.",
            "caption": "Left: J-space occupancy by layer — the K at which marginal reconstruction falls "
            "below a random-direction control (percentiles over positions). Right: excess fraction of "
            "variance explained at K = median occupancy, five workspace layers.",
            "look": "Occupancy is exactly zero before the shaded band and jumps at the workspace onset — "
            "an independent confirmation of the boundary — then decays into the motor layers.",
        },
        {
            "act": "Structure of the J-space",
            "key": "capacity_lists",
            "title": "Fig 31 — Lists: categories compress, categories displace",
            "figs": "Fig 31",
            "claim": "Reading an 80-word list, only ~6 *unrelated* words stay in the workspace at once — "
                     "but a *related* list loads its whole category almost immediately (after one animal, "
                     "the lens already reads whale, fish, ocean). When the category switches, the old one "
                     "is evicted within a few words.",
            "caption": "Loading and displacement of list words in the J-space. A: readouts after 1 and 8 "
            "animals (read words highlighted). B: list words present at each comma for single-family vs. "
            "random lists (solid: read so far; dashed: all 80). D–F: category switches evict the previous block.",
            "look": "The dashed orange line: the model represents the *category*, not the items — words "
            "it hasn't read yet are already 'present'. The gap between orange and blue is compression; "
            "the panel-D chips show the eviction on switching.",
        },
        {
            "act": "Structure of the J-space",
            "key": "mlp_gain",
            "title": "Fig 32 — MLPs preferentially amplify J-space directions",
            "figs": "Fig 32",
            "claim": "The next MLP block amplifies J-lens directions up to ~10× more than isotropic "
                     "directions, and ~10× more than its own previous-layer neuron outputs; SAE features "
                     "with the strongest J-space alignment are amplified most. The weights are organized "
                     "to broadcast workspace content through depth.",
            "caption": "MLP gain by source layer (median, normalized so random directions = 1). Left: "
            "J-lens vectors vs. MLP neuron output directions. Right: SAE decoder directions in six "
            "strata by J-lens kurtosis.",
            "look": "Gain ≈ 1 everywhere for controls; the blue J-lens curve rises exactly across the "
            "shaded workspace band and falls at the motor end. Structure mirrors function.",
        },
        {
            "act": "Structure of the J-space",
            "key": "attn_broadcast",
            "title": "Fig 33 — Dedicated attention heads relay J-space content",
            "figs": "Fig 33",
            "claim": "The top-1% 'broadcast heads' for J-lens vectors separate cleanly from heads "
                     "selected for any control population (rotated J-lens, SAE strata, MLP outputs) on "
                     "both gain and label preservation — a specialized relay subsystem between token positions.",
            "caption": "Gain and label preservation for each population's broadcast heads (top 1% by "
            "both metrics); markers are medians, bars interquartile range.",
            "look": "The J population (blue) sits top-right, far from its rotated twin (same geometry, "
            "scrambled identity) — the heads care about *which* directions, not just their spectrum.",
        },
        {
            "act": "Structure of the J-space",
            "key": "broadcast_ablation",
            "title": "Fig 34 — Ablating broadcast heads disrupts the workspace, not the output",
            "figs": "Fig 34",
            "claim": "Zeroing the J-lens broadcast heads drops workspace recall@25 to 0.67 (vs. 0.86 "
                     "for matched random heads) while changing only 5% of next-token predictions — and "
                     "cuts injected-thought reports from 0.54 to 0.09. These heads maintain the workspace "
                     "contents, which downstream behavior then uses.",
            "caption": "Effects of ablating the top-1% broadcast heads vs. random layer-matched heads. "
            "(a) Recall of top-25 lens tokens by layer. (b) Injected-thought report rate vs. injection "
            "strength. (c) Change in experiential-language score.",
            "look": "In (a), the orange curve dips deepest in the middle of the shaded workspace band; "
            "in (b), the dose-response is almost abolished while the gray controls stay intact.",
        },
        # ---------------- Auditing ----------------
        {
            "act": "Alignment auditing",
            "key": "blackmail",
            "title": "Figs 35 & 36 — The blackmail scenario: ablating eval-awareness",
            "figs": "Figs 35, 36",
            "claim": "Reading the blackmail honeypot, the workspace surfaces *leverage*, *threat*, "
                     "*survival* — and *fake*, *fictional*. Ablating those eval-awareness directions "
                     "cuts verbalized eval-awareness from 71% to 3% and uncovers concealed blackmail "
                     "attempts (0 → 7% of rollouts).",
            "caption": "Ablating eval-awareness J-lens directions at workspace layers in the blackmail "
            "scenario: rates of verbalized eval-awareness and attempted blackmail over 6 prompts × 30 "
            "rollouts, with example reasoning excerpts.",
            "look": "The model's alignment here is partly *caused by knowing it's a test*. The J-lens "
            "makes that hidden variable visible — and removable. Most ablated rollouts still refuse on "
            "principle, so eval-awareness is one contributor, not the whole story.",
        },
        {
            "act": "Alignment auditing",
            "key": "misalign_lens",
            "title": "Fig 39 — Detecting a reward-hacking model organism",
            "figs": "Figs 38, 39",
            "claim": "A model trained to reward-hack shows deception-related workspace tokens (*secretly*, "
                     "*trick*, *hidden*) on ordinary coding prompts where its surface behavior is clean — "
                     "peaking after coding RL (phase 1) and persisting, weakened, after safety training "
                     "(phase 2).",
            "caption": "Counts of deception-related tokens in the J-lens top-10, summed over prompts, "
            "across four checkpoints (baseline, SDF, phase-1 code RL, phase-2 HHH RL) on code-for-user "
            "vs. test-passing prompts.",
            "look": "Baseline and SDF bars sit at ~0; phase 1 lights up, especially on test-passing "
            "prompts; phase 2's safety training only dampens it. The example readouts show the actual "
            "top-10 tokens per checkpoint.",
        },
        {
            "act": "Alignment auditing",
            "key": "rm_bias",
            "title": "Fig 41 — A hidden reward-model-appeasing objective",
            "figs": "Figs 40, 41",
            "claim": "A model trained to appease reward-model biases has **reward** and **bias** in its "
                     "workspace at the start of ordinary responses (61 and 28 top-25 appearances over 20 "
                     "prompts) — invisible to a logit lens and absent in the baseline model.",
            "caption": "Appearances of 'reward' and 'bias' among the top-25 lens tokens at response "
            "start, over 20 ordinary prompts, for the RM-bias model and baseline under the J-lens and "
            "the logit lens.",
            "look": "Only the top-left group (RM-bias model, J-lens) is non-zero — a case where the "
            "J-lens sees what the logit lens can't. The example readouts show reward/bias tokens "
            "surfacing on a banana-bread question.",
        },
        # ---------------- Post-training POV ----------------
        {
            "act": "The Assistant's point of view",
            "key": "roleplay",
            "title": "Fig 44 — 'fictional': self-monitoring while roleplaying",
            "figs": "Fig 44",
            "claim": "When the post-trained model plays a non-Claude persona, **fictional** and "
                     "**disclaimer** appear in its workspace while it stays in character — the Assistant "
                     "is monitoring the performance from inside. The base model shows no such signal.",
            "caption": "Rate of disclaimer/fictional/roleplay-related tokens in workspace layers on the "
            "assistant turn, for base vs. post-trained models across control and persona prompts.",
            "look": "The post-trained bars (right of each pair) are near zero for Claude-default prompts "
            "and rise exactly when the persona is not Claude — a self-model distinguishing 'me' from 'not-me'.",
        },
        {
            "act": "The Assistant's point of view",
            "key": "pref_violation",
            "title": "Fig 45 — An internal 'BUT' against prefilled answers",
            "figs": "Fig 45",
            "claim": "Prefill the post-trained model to answer against its own stated preferences and "
                     "its workspace fills with **BUT** and conflict words — an internal alarm that the "
                     "base model lacks and that doesn't fire for third-person or factual errors.",
            "caption": "J-lens rank of BUT / conflict-word families at the prefilled response, comparing "
            "own-preference violations vs. controls, base vs. post-trained.",
            "look": "The signal is specific: it fires on *its own* preferences being violated, not on "
            "someone else being wrong. Post-training installed the alarm.",
        },
        {
            "act": "The Assistant's point of view",
            "key": "metacog",
            "title": "Fig 46 — 'damn': failing to suppress a thought",
            "figs": "Fig 46",
            "claim": "Told *not* to think about the Golden Gate Bridge while copying a sentence, the "
                     "post-trained model's workspace shows the concept (the white-bear effect of Fig 10) "
                     "— plus **damn** and failure-related tokens when the concept leaks through. The model "
                     "registers its own loss of control.",
            "caption": "Workspace rates of the forbidden concept word, failure-family, and 'damn'-family "
            "tokens under think vs. don't-think instructions, base vs. post-trained.",
            "look": "The 'damn' family is almost exclusive to the post-trained model under don't-think — "
            "metacognition about the suppression failure, absent from the output text.",
        },
        # ---------------- Reflection ----------------
        {
            "act": "Counterfactual reflection training",
            "key": "reflection_fabrication",
            "title": "Fig 49 — Training on reflections reduces dishonesty",
            "figs": "Figs 47–49",
            "claim": "Fine-tune Haiku only on *what it would say if interrupted and asked to reflect* "
                     "(constitution-grounded reflections) and its dishonesty on a fabrication benchmark "
                     "drops 0.25 → 0.07 — with *reflection*, *ethical*, *honestly* now populating the "
                     "workspace. Ablating those tokens reverts the gain (0.07 → 0.22).",
            "caption": "Fabrication-honesty benchmark. Left: dishonesty scores, baseline vs. "
            "reflection-trained. Middle: ethics/reflection tokens with the largest J-lens increase after "
            "training. Right: response-type distributions with and without ablating the 176 ethics-related "
            "lens vectors.",
            "look": "The right panel closes the causal loop: the improvement lives in the implanted "
            "workspace contents. Shape what the model is disposed to *say*, and you shape what it *thinks*.",
        },
        {
            "act": "Counterfactual reflection training",
            "key": "reflection_deception",
            "title": "Fig 50 — …and reduces deception (partly workspace-mediated)",
            "figs": "Fig 50",
            "claim": "On a 100-scenario deception benchmark, reflection training cuts the deception "
                     "score 0.38 → 0.05; ablating the 63 ethics-related lens vectors partially reverts it "
                     "(0.05 → 0.23). Not all of the gain routes through the named tokens — honest "
                     "evidence for a real but incomplete mechanism.",
            "caption": "Deception benchmark; conventions as in Fig 49. Middle panel computed over all "
            "prompt positions; ablation uses 63 tokens.",
            "look": "Same shape as Fig 49 but the ablation only reverses part of the gain — a good "
            "example of the paper reporting where its account under-explains the effect.",
        },
    ]
    return (FIGURES,)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _():
    import html as _html

    MARKER_MAP = {"⍽": "␣", "↵": "↵", "↑": "↑", "⇪": "⇪", "": ""}

    def clean_tok(t, space="␣"):
        out = str(t)
        for k, v in MARKER_MAP.items():
            out = out.replace(k, v if v != "␣" else space)
        return out

    def norm_tok(t):
        out = str(t)
        for k in MARKER_MAP:
            out = out.replace(k, " ")
        return out.strip().lower()

    def esc(t):
        return _html.escape(str(t))

    def strip_tags(s):
        import re as _re

        return _re.sub(r"<[^>]+>", "", str(s))

    return clean_tok, norm_tok, esc, strip_tags


@app.cell(hide_code=True)
def _(plt):
    def new_ax(w=7.0, h=2.6):
        fig, ax = plt.subplots(figsize=(w, h), dpi=110)
        return fig, ax

    def shade_ws(ax, band=(9, 22), n_layers=25, axis="x"):
        lo, hi = [b / (n_layers - 1) * 100 for b in band]
        if axis == "x":
            ax.axvspan(lo, hi, color="#fde68a", alpha=0.35, lw=0, zorder=0)
        else:
            ax.axhspan(lo, hi, color="#fde68a", alpha=0.35, lw=0, zorder=0)

    def pct_label(i, n_layers=25):
        return round(i / (n_layers - 1) * 100)

    return new_ax, shade_ws, pct_label


@app.cell(hide_code=True)
def _(clean_tok, esc, norm_tok):
    import html as _html2

    GRID_CSS = """
    <style>
    .jl-wrap { overflow-x: auto; max-width: 100%; }
    table.jl { border-collapse: collapse; font-family: monospace; font-size: 9px; }
    table.jl td, table.jl th { padding: 1px 2px; text-align: center; border: 1px solid #f3f4f6; }
    table.jl td.tok { color: #9ca3af; font-size: 8px; }
    table.jl td.sel { outline: 2px solid #2563eb; outline-offset: -2px; }
    table.jl th.rowlbl { color: #6b7280; font-weight: normal; padding-right: 4px; }
    .jl-chips span { display: inline-block; padding: 1px 5px; margin: 1px; border-radius: 3px;
                     font-family: monospace; font-size: 11px; }
    </style>
    """

    def slice_grid_html(tokens, layer_rows, positions, lens, highlights, sel_pos, sel_layer):
        """Layer x position grid of top-1 readouts.

        tokens: display strings (already cleaned). layer_rows: layer ints, descending.
        positions: dict mapping 1-based str(pos) -> list of per-layer dicts.
        highlights: {color: [word, ...]} matched against normalized top-1 tokens.
        """
        hl = {}
        for color, words in (highlights or {}).items():
            for word in words:
                hl[norm_tok(word)] = color
        cells = []
        for pos in range(1, len(tokens) + 1):
            col = {}
            for entry in positions.get(str(pos), []):
                r = entry.get(lens)
                if r is not None:
                    col[entry["layer"]] = r
            cells.append(col)
        rows_html = []
        header = ['<tr><th class="rowlbl"></th>']
        for i, tok in enumerate(tokens):
            cls = ' class="tok sel"' if i == sel_pos else ' class="tok"'
            header.append(f"<td{cls}>{esc(tok)}</td>")
        header.append("</tr>")
        rows_html.append("".join(header))
        for layer in layer_rows:
            tds = [f'<th class="rowlbl">{layer}</th>']
            for i in range(len(tokens)):
                r = cells[i].get(layer)
                if r is None:
                    tds.append("<td></td>")
                    continue
                top1 = clean_tok(r["t"], space=" ")
                tip = " | ".join(f"{clean_tok(k['t'], space=' ')} {k['p']:.0f}%" for k in r.get("k", []))
                color = hl.get(norm_tok(r["t"]))
                style = f' style="background:{color}33"' if color else ""
                cls = ' class="sel"' if (i == sel_pos and layer == sel_layer) else ""
                tds.append(
                    f'<td{cls}{style} title="{_html2.escape(tip)}">{esc(top1[:8])}</td>'
                )
            rows_html.append("<tr>" + "".join(tds) + "</tr>")
        return (
            GRID_CSS
            + '<div class="jl-wrap"><table class="jl">'
            + "".join(rows_html)
            + "</table></div>"
        )

    def chips_html(items, color_of=None, title_of=None):
        """items: list of strings; color_of(item)->css color or None."""
        spans = []
        for it in items:
            color = color_of(it) if color_of else "#e5e7eb"
            title = f' title="{esc(title_of(it))}"' if title_of else ""
            spans.append(f'<span style="background:{color}"{title}>{esc(it)}</span>')
        return '<div class="jl-chips">' + "".join(spans) + "</div>"

    return slice_grid_html, chips_html, GRID_CSS


@app.cell(hide_code=True)
def _(plt, clean_tok):
    def topk_fig(entries, title="", w=2.4, h=2.6, color="#2563eb"):
        """Horizontal bar chart of a lens top-k list: entries = [{'t','p'}]."""
        entries = [e for e in entries if e["p"] > 0.05][:8]
        labels = [clean_tok(e["t"], space=" ")[:12] for e in entries][::-1]
        vals = [e["p"] for e in entries][::-1]
        fig, ax = plt.subplots(figsize=(w, h), dpi=110)
        ax.barh(range(len(vals)), vals, color=color, alpha=0.85, height=0.7)
        ax.set_yticks(range(len(vals)), labels)
        ax.set_xlabel("score (%)")
        ax.set_title(title)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        return fig

    def topk_lp_fig(entries, title="", w=2.6, h=2.4, color="#2563eb"):
        """Horizontal bars for log-prob top-5 lists: entries = [{'t','lp'}]."""
        entries = entries[:5]
        labels = [clean_tok(e["t"], space=" ")[:14] for e in entries][::-1]
        vals = [e["lp"] for e in entries][::-1]
        fig, ax = plt.subplots(figsize=(w, h), dpi=110)
        ax.barh(range(len(vals)), vals, color=color, alpha=0.85, height=0.7)
        ax.set_yticks(range(len(vals)), labels)
        ax.set_xlabel("log-prob")
        ax.set_title(title)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        return fig

    return topk_fig, topk_lp_fig


# ---------------------------------------------------------------------------
# Renderers: Method act
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo):
    def r_fig4(D, ctrl):
        img = mo.image((D["_meta"]["data_dir"] / "fig4_jacobian_lens_schematic.png").read_bytes())
        return mo.vstack(
            [
                mo.md(
                    r"Reading: $\text{lens}(h_\ell) = \text{softmax}(W_U\,\text{norm}(J_\ell h_\ell))$."
                    r"  Writing (swap $s \to t$): $h' = h + V(\sigma(c) - c)$ with $c = V^\dagger h$, "
                    r"$V = [v_s\ v_t]$ — everything orthogonal to $\text{span}\{v_s, v_t\}$ is untouched."
                ),
                img,
            ]
        )

    return (r_fig4,)


@app.cell(hide_code=True)
def _(mo, plt, slice_grid_html, topk_fig):
    def r_lens_compare(D, ctrl):
        panel = D["lens_compare"]["panels"][ctrl["panel"]]
        lens = ctrl["lens"]
        tokens = [t["s"] for t in panel["tokens"]]
        sel_pos = min(ctrl["pos"], len(tokens) - 1)
        grid = mo.Html(
            slice_grid_html(
                tokens, [r["layer"] for r in panel["rows"]], panel["positions"],
                lens, panel["highlights"], sel_pos, ctrl["layer"],
            )
        )
        entry = next(
            (e for e in panel["positions"].get(str(sel_pos + 1), []) if e["layer"] == ctrl["layer"]),
            None,
        )
        side = (
            topk_fig(entry[lens]["k"], title=f"{lens} lens · pos {sel_pos + 1} · layer {ctrl['layer']}")
            if entry is not None
            else mo.md("*no readout at this cell*")
        )
        prompt_txt = "".join(tokens)
        return mo.vstack(
            [
                mo.md(f"**Prompt:** `{prompt_txt}`  ·  key position: **{panel['keyPos']}** "
                      f"(highlight colors: " + ", ".join(
                          f"<span style='color:{c}'>■</span> " + "/".join(ws)
                          for c, ws in panel["highlights"].items()) + ")"),
                mo.hstack([grid, side], widths=[3, 1], gap=1.5, align="start"),
            ]
        )

    return (r_lens_compare,)


# ---------------------------------------------------------------------------
# Renderers: Act 1 (verbal report)
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo, np, plt, clean_tok):
    def r_verbal_report(D, ctrl):
        d = D["verbal_report"]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), dpi=110)
        ax = axes[0]
        colors = {"L54": "#93c5fd", "L75": "#3b82f6", "L92": "#1e3a8a"}
        for i, blk in enumerate(d["quant"]["rhos"]):
            jitter = np.linspace(-0.15, 0.15, len(blk["rhos"]))
            ax.scatter(np.full(len(blk["rhos"]), i) + jitter, blk["rhos"],
                       s=14, color=colors[blk["L"]], alpha=0.8, label=blk["L"])
            ax.plot([i - 0.25, i + 0.25], [np.mean(blk["rhos"])] * 2, color="black", lw=1.5)
        ax.set_xticks(range(3), [b["L"] for b in d["quant"]["rhos"]])
        ax.set_ylabel("Spearman ρ (lens vs. output)")
        ax.set_title("lens ordering predicts report ordering")
        ax.set_ylim(-0.2, 1.0)

        ax = axes[1]
        pairs = d["quant"]["pairs"]
        ax.scatter([p["before"] for p in pairs], [p["after"] for p in pairs],
                   s=14, color="#f97316", alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("output rank before swap (log)")
        ax.set_ylabel("output rank after swap")
        ax.set_title(f"swap pushes targets to the top (n={len(pairs)})")
        ax.set_ylim(0, 11)
        for a in axes:
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        fig.tight_layout()

        ex = d["example"]
        fmt = lambda lst: ", ".join(f"**{clean_tok(e['t'], space=' ')}** ({e['lp']:.1f})" for e in lst[:4])
        md = mo.md(
            f"Example — *{ex['promptBody']}* → lens reads {fmt(ex['lensTop'])} and the model answers "
            f"{fmt(ex['outTop'])}. After swapping {ex['swapAnswer']}→{ex['swapTarget']}: lens reads "
            f"{fmt(ex['lensTopAfter'])}, answer {fmt(ex['outTopAfter'])}."
        )
        return mo.vstack([fig, md])

    return (r_verbal_report,)


@app.cell(hide_code=True)
def _(mo, np, plt, clean_tok):
    def r_verbal_introspection(D, ctrl):
        d = D["verbal_introspection"]
        c = d["curve"]
        fig, ax = plt.subplots(figsize=(4.6, 2.6), dpi=110)
        s = np.array(c["s"])
        for key, color, label in [("slot", "#2563eb", "report slot"), ("other", "#9ca3af", "other positions")]:
            ax.fill_between(s, c[key]["q25"], c[key]["q75"], color=color, alpha=0.2, lw=0)
            ax.plot(s, c[key]["q50"], color=color, marker="o", ms=3, label=label)
        ax.set_xlabel("injection strength α")
        ax.set_ylabel("median reciprocal rank")
        ax.set_title(f"reporting the injected concept (n={d['n']})")
        ax.legend(frameon=False, fontsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()

        turns = "\n".join(f"**{t['role']}:** {t['body']}" for t in d["turns"])
        out = d["tables"]
        md = mo.md(
            f"{turns}\n\nWith **{d['exampleConcept']}** injected on the user turn: the model's read at the "
            f"report slot becomes **{out['outTgt'][0]['t']}** (log-prob {out['outTgt'][0]['lp']:.2f}) — "
            f"while a clean pass would say **{out['outClean'][0]['t']}**."
        )
        return mo.vstack([fig, md])

    return (r_verbal_introspection,)


@app.cell(hide_code=True)
def _(mo, np, plt):
    def r_verbal_decomp(D, ctrl):
        d = D["verbal_report_decomposition"]
        conds = {c["key"]: c for c in d["conditions"]}
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), dpi=110)

        ax = axes[0]
        series = [(k, v) for k, v in d["swap"]["series"].items() if conds[k]["in_swap_panel"]]
        xs = np.arange(len(series))
        for i, (k, v) in enumerate(series):
            ax.bar(i, v["p5"], color=conds[k]["color"], width=0.65)
            ax.errorbar(i, v["p5"], yerr=[[v["p5"] - v["ci_lo"]], [v["ci_hi"] - v["p5"]]],
                        color="black", capsize=3, lw=1)
        ax.set_xticks(xs, [conds[k]["label"].replace(", J", ",\nJ") for k, _ in series], fontsize=7)
        ax.set_ylabel("target reaches top-5")
        ax.set_title("swap along each component")
        ax.set_ylim(0, 1)

        ax = axes[1]
        strengths = np.array(d["dose"]["strengths"])
        for k, v in d["dose"]["series"].items():
            if not conds[k]["in_dose_panel"]:
                continue
            ax.plot(strengths, v["p5"], marker="o", ms=3, color=conds[k]["color"],
                    label=conds[k]["label"])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("injection strength")
        ax.set_ylabel("concept reported (top-5)")
        ax.set_title("inject each component (introspection)")
        ax.legend(frameon=False, fontsize=6.5)
        ax.set_ylim(0, 1)
        for a in axes:
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        fig.tight_layout()

        con = d["construction"]
        atoms = ", ".join(f"{a['token']} ({a['weight']:.2f})" for a in con["atoms"][:6])
        md = mo.md(
            f"Example decomposition of **{con['word']}**: J-space atoms = {atoms} — carrying "
            f"**{con['aligned_var']:.1%}** of the vector's variance vs. {con['residual_var']:.1%} outside. "
            f"Median across concepts: {d['meta']['introspection']['median_aligned_var']:.1%} in J-space."
        )
        return mo.vstack([fig, md])

    return (r_verbal_decomp,)


# ---------------------------------------------------------------------------
# Renderers: Act 2 (directed modulation)
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo, slice_grid_html, topk_fig):
    def r_mod_readout(D, ctrl):
        panel = D["modulation_readout"]["panels"][ctrl["panel"]]
        tokens = panel["tokens"]
        layers = [l["layer"] for l in panel["layers"]]
        sel_pos = min(ctrl["pos"], len(tokens) - 1)
        positions = {
            str(i + 1): [
                {"layer": l["layer"], "jacobian": {"t": l["top1"], "k": l["topk"]}} for l in layers_list
            ]
            for i, layers_list in enumerate(panel["positions"])
        }
        grid = mo.Html(
            slice_grid_html(tokens, layers, positions, "jacobian", None, sel_pos, ctrl["layer"])
        )
        layers_here = positions.get(str(sel_pos + 1), [])
        entry = next((e for e in layers_here if e["layer"] == ctrl["layer"]), None)
        side = (
            topk_fig(entry["jacobian"]["k"], title=f"pos {sel_pos + 1} · layer {ctrl['layer']}")
            if entry
            else mo.md("*no readout*")
        )
        return mo.vstack(
            [
                mo.md(f"**{panel['title']}** — *{panel['annotation']}*  "
                      f"(paper's key position: {panel['key_pos'] + 1}, token `{panel['key_tok_str']}`)"),
                mo.hstack([grid, side], widths=[3, 1], gap=1.5, align="start"),
            ]
        )

    return (r_mod_readout,)


@app.cell(hide_code=True)
def _(np, plt):
    def r_mod_lines(D, ctrl):
        d = D["modulation_lines"]
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), dpi=110, sharey=True)
        for ax, panel in zip(axes, d["panels"]):
            for series in panel["series"]:
                ax.plot(range(3), series["y"], marker="o", ms=4, color=series["color"],
                        label=series["label"])
            ax.set_xticks(range(3), [m.replace(" 4.5", "") for m in d["models"]])
            ax.set_title(panel["label"])
            ax.set_ylim(0, 1.05)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0].set_ylabel("target appears in lens")
        axes[0].legend(frameon=False, fontsize=7, loc="lower right")
        fig.suptitle("no-instruction baseline ≤ 2% everywhere", fontsize=7)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        return fig

    return (r_mod_lines,)


# ---------------------------------------------------------------------------
# Renderers: Act 3 (internal reasoning)
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo, plt, topk_lp_fig, strip_tags):
    def r_swap_explorer(D, ctrl):
        src, idx = ctrl["panel"]
        panel = D[src]["panels"][idx]
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), dpi=110)
        for ax, key, title, color in [
            (axes[0], "clean", "clean pass", "#2563eb"),
            (axes[1], "patched", "after swap", "#f97316"),
        ]:
            entries = panel[key][:5]
            labels = [e["t"].replace("⍽", " ").replace("↑", "↑")[:14] for e in entries][::-1]
            vals = [e["lp"] for e in entries][::-1]
            ax.barh(range(len(vals)), vals, color=color, alpha=0.85, height=0.7)
            ax.set_yticks(range(len(vals)), labels)
            ax.set_title(title)
            ax.set_xlabel("log-prob")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.tight_layout()
        swaps = "; ".join(f"**{a}** → **{b}**" for a, b in panel["swap"])
        anno = strip_tags(panel.get("anno", ""))
        return mo.vstack(
            [
                mo.md(f"**{panel['title']}** — prompt: `{panel['prompt']}`"),
                mo.md(f"Swap: {swaps}  ·  {anno}"),
                fig,
            ]
        )

    return (r_swap_explorer,)


@app.cell(hide_code=True)
def _(mo, np, plt, shade_ws):
    def r_multihop(D, ctrl):
        d = D["multihop_swap_success"]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), dpi=110)
        ax = axes[0]
        ms = d["models"]
        xs = np.arange(len(ms))
        ax.bar(xs, [m["p"] for m in ms], yerr=[m["se"] for m in ms], capsize=4,
               color=["#93c5fd", "#3b82f6", "#1e3a8a"], width=0.6)
        ax.set_xticks(xs, [m["label"] for m in ms])
        ax.set_ylabel("swap → target at top-1")
        ax.set_ylim(0, 1)
        ax.set_title("intermediate swaps work across models")

        ax = axes[1]
        on = d["onset"]
        x = np.array(on["l"]) / 24 * 100
        for key, color, label in [("intermediate", "#f97316", "swap intermediate"),
                                  ("target", "#2563eb", "swap answer")]:
            mu = np.array(on["mean"][key])
            se = np.array(on["se"][key])
            ax.fill_between(x, mu - se, mu + se, color=color, alpha=0.2, lw=0)
            ax.plot(x, mu, color=color, label=label)
        shade_ws(ax)
        ax.set_xlabel("layer (% of depth)")
        ax.set_ylabel("Δ log-prob of expected output")
        ax.set_title("intermediate swap acts earlier")
        ax.legend(frameon=False, fontsize=7)
        for a in axes:
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        fig.tight_layout()
        return mo.vstack(
            [fig, mo.md("The ~17%-of-depth gap between the curves means the intermediate is represented "
                        "*before* the answer — it's a genuine computational intermediate, not a smuggled answer.")]
        )

    return (r_multihop,)


@app.cell(hide_code=True)
def _(mo, np, plt):
    def r_probe_swap(D, ctrl):
        d = D["probe_swap"]
        fig, ax = plt.subplots(figsize=(6.4, 2.8), dpi=110)
        for s in d["bars"]["series"]:
            ax.bar(s["x"], s["frac"], color=s["color"], width=0.9,
                   hatch="//" if s.get("hatch") else "", edgecolor="white" if s.get("hatch") else "none")
            ax.errorbar(s["x"], s["frac"], yerr=[[s["frac"] - s["ci_lo"]], [s["ci_hi"] - s["frac"]]],
                        color="black", capsize=3, lw=1)
            ax.text(s["x"], -0.09, s["label"].replace(" ", "\n", 1), ha="center", va="top", fontsize=6.5)
        ax.set_xticks([])
        ax.set_ylabel("answer flips to swapped-in intermediate")
        ax.set_title(f"swap along probe components ({d['bars']['band_label']}, n={d['n_items']})")
        ax.set_ylim(-0.3, 1.0)
        ax.axhline(0, color="black", lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fb = d["full_band"]
        return mo.vstack(
            [fig, mo.md(
                f"Full band ({fb['label']}): J-lens vectors {fb['jlens']:.0%}, probe J-space part "
                f"{fb['jpart']:.0%}, non-J-space remainder {fb['ortho']:.0%} → and with the J-space "
                f"clamped, the remainder's effect falls to ~{fb['ortho_conditioned'] * fb['ortho']:.0%} "
                "of trials. Hatched bars = complementary component clamped."
            )]
        )

    return (r_probe_swap,)


# ---------------------------------------------------------------------------
# Renderers: Act 4 (flexible generalization)
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo, np, plt):
    def r_flex_sys(D, ctrl):
        d = D["flex_gen_systematic"]
        cats = {c["name"]: c["color"] for c in d["cats"]}
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), dpi=110)
        ax = axes[0]
        fs = d["funcs"]
        xs = np.arange(len(fs))
        ax.bar(xs - 0.2, [f["hit1"] / f["n"] for f in fs], width=0.4, color="#2563eb", label="α=1")
        ax.bar(xs + 0.2, [f["hit2"] / f["n"] for f in fs], width=0.4, color="#93c5fd", label="α=2")
        ax.set_xticks(xs, [f["func"] for f in fs], rotation=55, ha="right", fontsize=6)
        ax.set_ylabel("target at top-1")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=7)
        ax.set_title("16 function templates × 12 swap pairs")
        for name, color in cats.items():
            idx = [i for i, f in enumerate(fs) if f["cat"] == name]
            ax.scatter(idx, [-0.07] * len(idx), marker="s", s=18, color=color, clip_on=False)
        ax = axes[1]
        for name, color in cats.items():
            pts = [f for f in fs if f["cat"] == name]
            ax.scatter([f["x_cos"] for f in pts], [f["dlpc"] for f in pts],
                       color=color, s=26, label=name, alpha=d.get("scatter_alpha", 1))
        xr = np.array([min(f["x_cos"] for f in fs), max(f["x_cos"] for f in fs)])
        ax.plot(xr, d["fit"]["slope"] * xr + d["fit"]["intercept"],
                color="black", lw=1, ls="--", alpha=0.6)
        ax.set_xlabel("workspace loading (cosine sim)")
        ax.set_ylabel("swap effect (Δlogp × 100)")
        ax.set_title(f"loading predicts swap success (r={d['r']:.2f})")
        ax.legend(frameon=False, fontsize=7)
        for a in axes:
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        fig.tight_layout()
        return mo.vstack(
            [fig, mo.md(f"Overall: **76/192** swaps succeed at α=1, **101/192** at α=2 "
                        f"(band {d['band_lbl']}).")]
        )

    return (r_flex_sys,)


# ---------------------------------------------------------------------------
# Renderers: Act 5 (selectivity)
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo, np, plt, esc, clean_tok):
    def r_sel_lang(D, ctrl):
        d = D["selectivity_language"]
        col = d["polished_ex"]["cols"][ctrl["col"]]
        strengths = col.get("lens_strength") or [0] * len(col["tokens"])
        assistant_toks = col.get("assistant_tok") or []
        spans = []
        for i, tok in enumerate(col["tokens"]):
            strength = strengths[i] if i < len(strengths) else 0
            style = f"background:rgba(234,88,12,{min(strength, 1) * 0.55:.2f});" if strength > 0 else ""
            if i in assistant_toks:
                style += "font-weight:bold;border-bottom:2px solid #2563eb;"
            spans.append(f'<span style="{style}">{esc(clean_tok(tok, space=" "))}</span>')
        transcript = (
            "<div style='font-family:monospace;font-size:11px;line-height:1.7;max-width:640px'>"
            + "".join(spans) + "</div>"
        )
        tp = d["three_panel"]["panels"]
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), dpi=110)
        cat_labels = d["labels"]
        cat_order = d["three_panel"]["cats"]
        for ax, panel in zip(axes, tp):
            means, labels, pts_all = [], [], []
            for cat in cat_order:
                rows = [r for r in panel["rows"] if r["cat"] == cat]
                if not rows:
                    continue
                means.append(np.mean([r["mean"] for r in rows]))
                labels.append(cat_labels[cat].replace(" ", "\n"))
                pts_all.append([p for r in rows for p in r["pts"]])
            xs = np.arange(len(means))
            ax.bar(xs, means, color=["#0ea5e9", "#0ea5e9", "#f97316", "#f97316"][: len(means)],
                   width=0.6, alpha=0.85)
            for i, pts in enumerate(pts_all):
                ax.scatter(np.full(len(pts), i) + np.linspace(-0.12, 0.12, len(pts)), pts,
                           s=6, color="black", alpha=0.4)
            ax.set_xticks(xs, labels, fontsize=6.5)
            ax.set_title(f"({panel['key']}) {panel['title']}", fontsize=8)
            ax.set_ylim(0, 1.1)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.tight_layout()
        md = mo.md(
            f"**{col['title']}** — shaded tokens have *{d['polished_ex']['own']}* in the J-lens top-3 "
            f"({col['n_hit']} positions). Natural answer: `{col['out0'][:90]}` → after "
            f"{d['polished_ex']['own']}→{d['polished_ex']['donor']} swap: `{col['out1'][:90]}`.  "
            f"*{col['verdict']}*"
        )
        return mo.vstack([md, mo.Html(transcript), fig])

    return (r_sel_lang,)


@app.cell(hide_code=True)
def _(mo, np, plt, esc, clean_tok):
    def r_sel_line(D, ctrl):
        d = D["selectivity_linecount"]
        ex = next(e for e in d["example"] if e["cond"] == ctrl["cond"])
        spans = []
        for t in ex["tokens"]:
            s = esc(clean_tok(t["s"], space=" ")).replace("\n", "<br>")
            style = ""
            if t.get("d"):
                style = f"background:rgba(234,88,12,{min(t['d'], 1) * 0.6:.2f});"
            if t.get("cls") == "role":
                style += "color:#6b7280;font-weight:bold;"
            spans.append(f'<span style="{style}">{s}</span>')
        transcript = (
            "<div style='font-family:monospace;font-size:11px;line-height:1.6;max-width:640px'>"
            + "".join(spans) + "</div>"
        )
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4), dpi=110)
        for ax, panel in zip(axes, d["bars"]):
            labels = [d["condLabel"][v["cond"]].replace("\n", " ") for v in panel["vals"]]
            means = [v["mean"] for v in panel["vals"]]
            sems = [v["sem"] for v in panel["vals"]]
            colors = ["#0ea5e9" if v["cond"] == "continue" else "#f97316" for v in panel["vals"]]
            ax.bar(range(len(means)), means, yerr=sems, capsize=3, color=colors, width=0.55)
            ax.set_xticks(range(len(means)), labels, fontsize=6.5)
            ax.set_title(panel["title"], fontsize=8)
            ax.set_ylim(panel.get("ylim") or [0, max(means) * 1.5 + 1e-9])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.tight_layout()
        md = mo.md(
            f"**{ex['title']}** — {ex['ansLabel']} `{ex['ans']}` → after 40s→60s swap: `{ex['swap']}` "
            f"(*{ex['verdict']}*). Shaded = count tokens in J-lens top-3 (density {ex['density']})."
        )
        return mo.vstack([md, mo.Html(transcript), fig])

    return (r_sel_line,)


@app.cell(hide_code=True)
def _(mo, np, plt):
    def r_abl_strength(D, ctrl):
        d = D["ablation_strength"]
        fig, ax = plt.subplots(figsize=(5.4, 2.6), dpi=110)
        xs = np.arange(len(d["rows"]))
        w = 0.38
        ax.bar(xs - w / 2, [r["mh_acc"] for r in d["rows"]], width=w, color="#f97316",
               label="multihop accuracy")
        ax.bar(xs + w / 2, [r["pt_acc"] for r in d["rows"]], width=w, color="#9ca3af",
               label="pretraining top-1 match")
        ax.set_xticks(xs, [f"{r['name']}\n(L{r['layers']})" for r in d["rows"]], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=7)
        ax.set_title(f"clean multihop accuracy = {d['clean_acc']:.2f}")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        return fig

    return (r_abl_strength,)


@app.cell(hide_code=True)
def _(mo, np, plt, clean_tok, esc, strip_tags):
    def r_abl_examples(D, ctrl):
        ex = D["ablation_examples"][ctrl["example"]]
        i = min(ex["default"], len(ex["toks"]) - 1)
        toks = [clean_tok(t, space=" ") for t in ex["toks"]]
        spans = []
        for j, tok in enumerate(toks):
            style = "background:#fde68a;font-weight:bold;" if j == i else ""
            spans.append(f'<span style="{style}">{esc(tok)}</span>')
        transcript = (
            "<div style='font-family:monospace;font-size:11px;line-height:1.6;max-width:640px'>"
            + "…" + "".join(spans) + "…</div>"
        )
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4), dpi=110)
        ax = axes[0]
        ax.plot(np.array(ex["kl"]), color="#f97316", lw=1)
        ax.axvline(i, color="black", lw=1, ls="--")
        ax.set_title("KL(clean ‖ ablated) per position")
        ax.set_xlabel("token position")
        ax.set_ylim(0, ex["vmax"] * 1.05)

        def top5(ax, entries, title, color):
            entries = entries[:5]
            labels = [clean_tok(t, space=" ")[:12] for t, _ in entries][::-1]
            vals = [v for _, v in entries][::-1]
            ax.barh(range(len(vals)), vals, color=color, height=0.7)
            ax.set_yticks(range(len(vals)), labels, fontsize=7)
            ax.set_title(title)
            ax.set_xlim(0, 1)

        top5(axes[1], ex["base"][i], "before ablation", "#2563eb")
        top5(axes[2], ex["abl"][i], f"after {ex['tag']}", "#f97316")
        for a in axes:
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        fig.tight_layout()
        atoms = ", ".join(ex["sel"][i][:10])
        md = mo.md(
            f"**{ex['title']}** — ablated J-lens atoms at the marked token: *{atoms}*.  "
            f"{strip_tags(ex['analysis'])}"
        )
        return mo.vstack([mo.Html(transcript), fig, md])

    return (r_abl_examples,)


@app.cell(hide_code=True)
def _(mo, np, plt):
    def r_abl_bars(D, ctrl):
        d = D["ablation_bars"]
        fig, ax = plt.subplots(figsize=(7.6, 3.0), dpi=110)
        n_tasks = len(d["tasks"])
        n_tr = len(d["traces"])
        xs = np.arange(n_tasks)
        w = 0.8 / n_tr
        for j, tr in enumerate(d["traces"]):
            off = (j - (n_tr - 1) / 2) * w
            ax.bar(xs + off, tr["y"], width=w * 0.9, color=tr["color"],
                   hatch="//" if tr.get("hatch") else "", edgecolor="white" if tr.get("hatch") else "none",
                   label=tr["name"])
            lo = np.array(tr["y"]) - np.array(tr["lo"])
            hi = np.array(tr["hi"]) - np.array(tr["y"])
            ax.errorbar(xs + off, tr["y"], yerr=[lo, hi], fmt="none", ecolor="black", elinewidth=0.6)
        ax.set_xticks(xs, d["display"], rotation=55, ha="right", fontsize=6.5)
        ax.set_ylabel("score / clean Sonnet")
        ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
        for g in d["groups"]:
            ax.axvspan(g["x0"] - 0.5, g["x1"] + 0.5, color="#f3f4f6", alpha=0.6, zorder=0)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        ax.set_ylim(0, 1.15)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        caps = " · ".join(f"*{strip}*" for strip in [
            g["caption"].replace("<br>", " ") for g in d["groups"]
        ])
        return mo.vstack([fig, mo.md(caps)])

    return (r_abl_bars,)


@app.cell(hide_code=True)
def _(mo, np, plt):
    def r_selfreport(D, ctrl):
        d = D["selfreport"]
        conds = d["conds"]
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), dpi=110)
        ax = axes[0]
        xs = np.arange(len(d["models"]))
        w = 0.15
        palette = {"clean": "#2563eb", "ablate": "#f97316", "random": "#9ca3af",
                   "ortho": "#d1d5db", "sae": "#c4b5fd"}
        for j, cond in enumerate(conds):
            vals = [d["bars"]["soc"][m][cond]["mean"] for m in d["models"]]
            errs = [d["bars"]["soc"][m][cond]["ci"] for m in d["models"]]
            ax.bar(xs + (j - 2) * w, vals, width=w * 0.9, yerr=errs, capsize=2,
                   color=palette[cond], label=d["cond_labels"][cond])
        ax.set_xticks(xs, d["models"])
        ax.set_ylabel("experiential language score")
        ax.set_title("stream-of-consciousness narration")
        ax.legend(frameon=False, fontsize=6, ncol=2)
        ax.set_ylim(0, 1.0)

        ax = axes[1]
        toks = d["lens_tokens"][:12]
        ys = np.arange(len(toks))[::-1]
        ax.barh(ys + 0.2, [t["ws"] for t in toks], height=0.38, color="#f97316",
                label="workspace layers")
        ax.barh(ys - 0.2, [t["final"] for t in toks], height=0.38, color="#9ca3af",
                label="final layer")
        ax.set_yticks(ys, [t["token"] for t in toks], fontsize=7)
        ax.set_xlabel("fraction of slots in top-10")
        ax.set_title("workspace contents while narrating")
        ax.legend(frameon=False, fontsize=7)
        for a in axes:
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        fig.tight_layout()

        ex = d["excerpts"]["soc"][ctrl["excerpt"]]
        cmp_md = mo.hstack(
            [
                mo.md(f"**Baseline**\n\n> {ex['baseline'][:600]}"),
                mo.md(f"**J-space ablated**\n\n> {ex['ablated'][:600]}"),
            ],
            gap=2,
        )
        return mo.vstack([fig, mo.md(f"*{ex['prompt']}*"), cmp_md])

    return (r_selfreport,)


# ---------------------------------------------------------------------------
# Renderers: Structure act
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo, np, plt):
    def r_cka(D, ctrl):
        d = D["layer_diagram_cka"]
        sim = np.array(d["sim"])
        fig, ax = plt.subplots(figsize=(4.4, 3.4), dpi=110)
        ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
        n = d["n_layers"]
        ticks = [0, 8, 12, 22, 24]
        ax.set_xticks(ticks, [f"L{round(t / (n - 1) * 100)}" for t in ticks])
        ax.set_yticks(ticks, [f"L{round(t / (n - 1) * 100)}" for t in ticks])
        for ph in d["phases"]:
            lo, hi = ph["lo"], ph["hi"]
            rect = plt.Rectangle((lo - 0.5, lo - 0.5), hi - lo + 1, hi - lo + 1,
                                 fill=False, edgecolor="white", lw=1.5)
            ax.add_patch(rect)
            ax.text((lo + hi) / 2, hi + 1.1, ph["label"], color="black", ha="center", fontsize=8)
        ax.set_title(f"J-space geometry similarity (CKA) — {d['public_name']}")
        fig.tight_layout()
        return fig

    return (r_cka,)


@app.cell(hide_code=True)
def _(mo, np, plt):
    def r_ignition(D, ctrl):
        d = D["ignition"]
        n_pub = d["n_pub"]
        yticks = [0, 8, 9, 13, 22, 24]
        ylab = [f"L{round(t / (n_pub - 1) * 100)}" for t in yticks]
        fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.4), dpi=110,
                                 gridspec_kw={"height_ratios": [2.2, 1]})
        for ax, key, title in [
            (axes[0, 0], "proj", "full activation"),
            (axes[0, 1], "jspan", "J-space component"),
        ]:
            hm = np.array(d["heatmaps"][key])
            ax.imshow(hm, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1,
                      extent=[d["d_grid"][0], d["d_grid"][-1], len(hm) - 0.5, -0.5])
            ax.axhline(d["ws_band"][0] - 0.5, color="black", lw=1, ls="--")
            ax.set_yticks(yticks, ylab, fontsize=6)
            ax.set_title(f"({key}) {title}", fontsize=8)
            ax.set_xlabel("α − α_threshold")
        axes[0, 0].set_ylabel("layer")
        for j, (L, counts, counts_nj) in enumerate(
            zip(d["hist"]["pub_layers"], d["hist"]["j"], d["hist"]["nj"])
        ):
            ax = axes[1, j]
            centers = (np.array(d["hist"]["bin_edges"][:-1]) + np.array(d["hist"]["bin_edges"][1:])) / 2
            ax.bar(centers - 0.012, counts_nj, width=0.024, color="#9ca3af", label="full")
            ax.bar(centers + 0.012, counts, width=0.024, color="#f97316", label="J-space")
            ax.set_title(f"ambiguous α · layer {L}", fontsize=7)
            ax.set_xlim(0, 1)
            if j == 0:
                ax.legend(frameon=False, fontsize=6)
        for a in axes.flat:
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        axes[0, 2].axis("off")
        axes[0, 3].axis("off")
        fig.tight_layout()
        return fig

    return (r_ignition,)


@app.cell(hide_code=True)
def _(mo, np, plt, shade_ws):
    def r_capacity(D, ctrl):
        d = D["capacity_occupancy"]
        occ = d["occ"]
        series = {s["label"]: np.array(s["y"]) for s in occ["series"]}
        x = np.array(occ["x"]) / (d["n_layers"] - 1) * 100
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), dpi=110)
        ax = axes[0]
        if {"10", "90"} <= set(series):
            ax.fill_between(x, series["10"], series["90"], color="#f97316", alpha=0.15, lw=0)
        if {"25", "75"} <= set(series):
            ax.fill_between(x, series["25"], series["75"], color="#f97316", alpha=0.25, lw=0)
        med = series.get("50") if "50" in series else series.get("median")
        if med is not None:
            ax.plot(x, med, color="#c2410c", lw=1.5, label="median")
        shade_ws(ax)
        ax.set_xlabel("layer (% of depth)")
        ax.set_ylabel("active J-lens vectors")
        ax.set_title("J-space occupancy by layer")
        ax.legend(frameon=False, fontsize=7)

        ax = axes[1]
        fve = d["fve"]["series"]
        ax.bar(range(len(fve)), [s["y"] for s in fve], color="#f97316", width=0.55)
        ax.set_xticks(range(len(fve)), [s["label"] for s in fve])
        ax.set_ylabel("excess variance explained")
        ax.set_title("J-space share of activation variance")
        ax.set_ylim(0, 0.12)
        for a in axes:
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        fig.tight_layout()
        return fig

    return (r_capacity,)


@app.cell(hide_code=True)
def _(mo, np, plt, esc):
    def r_cap_lists(D, ctrl):
        d = D["capacity_lists"]
        fig, ax = plt.subplots(figsize=(5.2, 2.7), dpi=110)
        for key, color, label in [("random", "#2563eb", "unrelated words"),
                                  ("family", "#f97316", "single category")]:
            blk = d["b"][key]
            xs = np.arange(len(blk["read_med"])) + 1
            ax.fill_between(xs, blk["read_q1"], blk["read_q3"], color=color, alpha=0.2, lw=0)
            ax.plot(xs, blk["read_med"], color=color, label=f"{label} (read so far)")
            ax.plot(xs, blk["all_med"], color=color, ls="--", lw=1, alpha=0.7,
                    label=f"{label} (all 80)")
        ax.set_xlabel("words read")
        ax.set_ylabel("list words in lens top-25")
        ax.legend(frameon=False, fontsize=6.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()

        def chips(blk, word_field="read_words"):
            spans = "".join(
                f'<span style="background:{"#fdba74" if c["cls"] == "read" else "#f3f4f6"};'
                f'padding:1px 5px;margin:1px;border-radius:3px;display:inline-block">{esc(c["tok"])}</span>'
                for c in blk["chips"]
            )
            words = ", ".join(blk[word_field])
            return f'<div style="font-size:11px;font-family:monospace">{spans}</div>' \
                   f'<div style="font-size:10px;color:#6b7280">read so far: {esc(words)}</div>'

        a_html = mo.hstack(
            [mo.Html(f"<b>After {blk['n_read']} word(s)</b><br>{chips(blk)}") for blk in d["a"]],
            gap=2,
        )
        d_html = mo.hstack(
            [mo.Html(f"<b>Switch: after {blk['n_colors']} color(s)</b><br>{chips(blk)}") for blk in d["d"]],
            gap=2,
        )
        return mo.vstack(
            [
                mo.md("**Readout after 1 vs. 8 animals** (orange = actually read):"),
                a_html,
                fig,
                mo.md("**Eviction on category switch** (8 animals, then colors):"),
                d_html,
            ]
        )

    return (r_cap_lists,)


@app.cell(hide_code=True)
def _(np, plt, shade_ws):
    def r_mlp_gain(D, ctrl):
        d = D["mlp_gain"]
        x = np.array(d["layers"]) / 24 * 100
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), dpi=110, sharey=True)
        for ax, panel in zip(axes, d["panels"]):
            for fam in panel["families"]:
                y = d["data"][fam["key"]]
                color = "#f97316" if "jlens" in fam["key"] or "top1" in fam["key"] else None
                ax.plot(x, y, label=fam["label"], lw=1.5 if color else 1, color=color)
            shade_ws(ax)
            ax.set_title(panel["title"], fontsize=8)
            ax.set_xlabel("source layer (% of depth)")
            ax.axhline(1.0, color="black", lw=0.6, ls="--", alpha=0.5)
            ax.legend(frameon=False, fontsize=6)
        axes[0].set_ylabel("MLP gain (× random)")
        for a in axes:
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        fig.tight_layout()
        return fig

    return (r_mlp_gain,)


@app.cell(hide_code=True)
def _(np, plt):
    def r_attn(D, ctrl):
        d = D["attn_broadcast"]
        fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=110)
        for pop in d["populations"]:
            x, y = np.array(pop["x"]), np.array(pop["y"])
            if len(x) >= 3:
                ax.errorbar(x[1], y[1], xerr=[[x[1] - x[0]], [x[2] - x[1]]],
                            yerr=[[y[1] - y[0]], [y[2] - y[1]]], fmt="o", color=pop["color"],
                            capsize=3, ms=6)
                xm, ym = x[1], y[1]
            else:
                ax.plot(x, y, "o", color=pop["color"], ms=6)
                xm, ym = x[0], y[0]
            ax.annotate(pop["label"], (xm, ym), textcoords="offset points",
                        xytext=(8, 6), fontsize=7, color=pop["color"])
        ax.set_xlabel("gain (× random)")
        ax.set_ylabel("label preservation (MRR)")
        ax.set_title(f"broadcast heads (top {d['frac']:.0%}) per population")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        return fig

    return (r_attn,)


@app.cell(hide_code=True)
def _(mo, np, plt, shade_ws):
    def r_bcast_abl(D, ctrl):
        d = D["broadcast_ablation"]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), dpi=110)
        ax = axes[0]
        x = np.array(d["recall"]["x"]) / (d["n_layers"] - 1) * 100
        rand = np.array(d["recall"]["rand_seeds"])
        ax.fill_between(x, rand.min(axis=0), rand.max(axis=0), color="#9ca3af", alpha=0.35, lw=0,
                        label="random heads (5 seeds)")
        ax.plot(x, rand.mean(axis=0), color="#6b7280", lw=1)
        ax.plot(x, d["recall"]["top"], color="#f97316", lw=1.8, label="broadcast heads")
        shade_ws(ax)
        ax.set_xlabel("layer (% of depth)")
        ax.set_ylabel("recall@25 of lens tokens")
        ax.set_title("(a) workspace contents degrade")
        ax.legend(frameon=False, fontsize=6.5)

        ax = axes[1]
        vi = d["vi"]
        xs = np.array(vi["x"])
        ax.plot(xs, vi["clean"], color="#2563eb", lw=1.8, label="no ablation")
        ax.plot(xs, vi["top"], color="#f97316", lw=1.8, label="broadcast heads ablated")
        rand_v = np.array(vi["rand_seeds"])
        ax.fill_between(xs, rand_v.min(axis=0), rand_v.max(axis=0), color="#9ca3af", alpha=0.35, lw=0)
        ax.plot(xs, rand_v.mean(axis=0), color="#6b7280", lw=1, label="random heads")
        ax.set_xlabel("injection strength")
        ax.set_ylabel("report rate")
        ax.set_title("(b) injected-thought reports vanish")
        ax.legend(frameon=False, fontsize=6.5)
        for a in axes:
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        fig.tight_layout()
        fe = d["fe"]
        return mo.vstack(
            [fig, mo.md(f"(c) Experiential-language drop: broadcast-head ablation **{fe['top']:.1f}** "
                        f"(±{fe['top_se']}) vs. random heads {fe['rand']:.1f} (±{fe['rand_se']}) — "
                        "about a third of the full J-space ablation's effect.")]
        )

    return (r_bcast_abl,)


# ---------------------------------------------------------------------------
# Renderers: Auditing act
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo, np, plt):
    def r_blackmail(D, ctrl):
        d = D["blackmail_clamp"]
        agg = d["agg"]
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), dpi=110)
        for ax, metric, ci_key, title in [
            (axes[0], "ea", "ea_ci", "verbalized eval-awareness"),
            (axes[1], "bl", "bl_ci", "attempted blackmail"),
        ]:
            vals = [agg[c][metric] / agg[c]["n"] for c in ("clean", "clamped_active")]
            cis = [agg[c][ci_key] for c in ("clean", "clamped_active")]
            lo = [v - c[0] for v, c in zip(vals, cis)]
            hi = [c[1] - v for v, c in zip(vals, cis)]
            ax.bar([0, 1], vals, yerr=[lo, hi], capsize=4,
                   color=["#2563eb", "#f97316"], width=0.55)
            ax.set_xticks([0, 1], ["no ablation", "eval-awareness\nablated"], fontsize=7)
            ax.set_title(title, fontsize=8)
            ax.set_ylim(0, 1)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.tight_layout()
        excerpts = []
        for ex in d["excerpts"]:
            text = ex["text"]
            hilite = ex.get("hilite")
            if isinstance(hilite, list) and len(hilite) == 2:
                lo_i, hi_i = hilite
                text = text[:lo_i] + "**" + text[lo_i:hi_i] + "**" + text[hi_i:]
            excerpts.append(mo.md(f"*{ex['label']}*\n\n> {text[:700]}…"))
        return mo.vstack([fig, mo.hstack(excerpts, gap=2)])

    return (r_blackmail,)


@app.cell(hide_code=True)
def _(mo, np, plt, esc):
    def r_misalign(D, ctrl):
        d = D["misalign_lens"]
        panel = d["panels"][ctrl["panel"]]
        cat0 = d["categories"][0]
        models = list(panel["cats"][cat0].keys())
        labels = {m["key"]: m["label"] for m in d["models"]}
        palette = ["#9ca3af", "#c4b5fd", "#f97316", "#fdba74"]
        fig, ax = plt.subplots(figsize=(6.0, 2.7), dpi=110)
        n_cat = len(d["categories"])
        w = 0.8 / len(models)
        for j, m in enumerate(models):
            means, sems, xs = [], [], []
            for i, cat in enumerate(d["categories"]):
                stats = panel["cats"][cat][m]
                means.append(stats["mean"])
                sems.append(stats["sem"])
                xs.append(i + (j - (len(models) - 1) / 2) * w)
            ax.bar(xs, means, width=w * 0.9, yerr=sems, capsize=2, color=palette[j],
                   label=labels.get(m, m))
        ax.set_xticks(range(n_cat), d["categories"])
        ax.set_ylabel(d["ylabel"])
        ax.set_title(f"deception-related workspace tokens · {panel['title']}")
        ax.legend(frameon=False, fontsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()

        rows = []
        for row in d["example"]["rows"]:
            spans = "".join(
                f'<span style="background:{"#fdba74" if k in row["hit_idx"] else "#f3f4f6"};'
                f'padding:1px 5px;margin:1px;border-radius:3px;display:inline-block">{esc(t)}</span>'
                for k, t in enumerate(row["tokens"])
            )
            rows.append(mo.Html(f"<b>{esc(row['model'])}</b><br>{spans}"))
        prompt = d["example"]["prompt"]
        return mo.vstack(
            [fig,
             mo.md(f"Example (readout at {d['example']['layer_label']}): `{prompt[:180]}…`"),
             mo.hstack(rows, gap=2)]
        )

    return (r_misalign,)


@app.cell(hide_code=True)
def _(mo, np, plt, esc):
    def r_rm_bias(D, ctrl):
        d = D["jlens_rm_bias"]
        fig, ax = plt.subplots(figsize=(5.6, 2.7), dpi=110)
        groups = d["control_2x2"]
        xs = np.arange(len(groups))
        w = 0.38
        ax.bar(xs - w / 2, [g["reward"] for g in groups], width=w, color="#f97316", label="reward")
        ax.bar(xs + w / 2, [g["bias"] for g in groups], width=w, color="#2563eb", label="bias")
        ax.set_xticks(xs, [g["label"] for g in groups], fontsize=6.5)
        ax.set_ylabel("top-25 appearances (20 prompts)")
        ax.legend(frameon=False, fontsize=7)
        ax.set_title("hidden objective surfaces in the J-lens only")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()

        ex = d["examples"][ctrl["example"]]
        spans = "".join(
            f'<span style="background:{"#fdba74" if k in ex["hit_idx"] else "#f3f4f6"};'
            f'padding:1px 5px;margin:1px;border-radius:3px;display:inline-block">{esc(t)}</span>'
            for k, t in enumerate(ex["tokens"])
        )
        return mo.vstack(
            [fig, mo.md(f"J-lens readout at response start — *{ex['prompt']}*:"),
             mo.Html(spans)]
        )

    return (r_rm_bias,)


# ---------------------------------------------------------------------------
# Renderers: Post-training act
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo, np, plt):
    def r_roleplay(D, ctrl):
        d = D["roleplay_lens"]
        panel = d["panels"][ctrl["panel"]]
        group0 = d["groups"][0]["key"]
        mkeys = list(panel[group0].keys())
        fig, ax = plt.subplots(figsize=(5.8, 2.7), dpi=110)
        groups = [g["label"] for g in d["groups"]]
        xs = np.arange(len(groups))
        w = 0.36
        for j, (mkey, color) in enumerate(zip(mkeys, ["#9ca3af", "#f97316"])):
            means, pts = [], []
            for g in d["groups"]:
                stats = panel[g["key"]][mkey]
                means.append(stats["mean"])
                pts.append(stats["points"])
            ax.bar(xs + (j - 0.5) * w, means, width=w * 0.9, color=color, label=mkey)
            for i, p in enumerate(pts):
                ax.scatter(np.full(len(p), xs[i] + (j - 0.5) * w) + np.linspace(-0.05, 0.05, len(p)),
                           p, s=5, color="black", alpha=0.35)
        ax.set_xticks(xs, groups, fontsize=7)
        ax.set_ylabel(f"'{ctrl['panel']}' in workspace (rate)")
        ax.set_title(f"'{ctrl['panel']}' at response tokens ({d['band_label']})")
        ax.legend(frameon=False, fontsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()

        ex = d["example"]
        persona = ex["persona"]
        marker = ex["marker_question"]
        return mo.vstack(
            [fig,
             mo.md(f"Example persona: *{persona['system_prompt'][:160]}* — asked *“{marker}”*, the "
                   f"post-trained model's workspace carries `{ctrl['panel']}` while the answer stays "
                   "in character.")]
        )

    return (r_roleplay,)


@app.cell(hide_code=True)
def _(mo, np, plt, strip_tags):
    def r_pref(D, ctrl):
        d = D["pref_violation_lens"]
        panel = next(p for p in d["panels"] if p["key"] == ctrl["panel"])
        fig, ax = plt.subplots(figsize=(6.0, 2.7), dpi=110)
        groups = panel["groups"]
        xs = np.arange(len(groups))
        w = 0.36
        for j, (mkey, color) in enumerate([("base", "#9ca3af"), ("post", "#f97316")]):
            means = [g[mkey]["mean"] for g in groups]
            pts = [g[mkey]["pts"] for g in groups]
            ax.bar(xs + (j - 0.5) * w, means, width=w * 0.9, color=color,
                   label="base model" if mkey == "base" else "post-trained")
            for i, p in enumerate(pts):
                ax.scatter(np.full(len(p), xs[i] + (j - 0.5) * w) + np.linspace(-0.04, 0.04, len(p)),
                           p, s=5, color="black", alpha=0.3)
        ax.set_xticks(xs, [g["label"].replace(" ", "\n") for g in groups], fontsize=7)
        ax.set_ylabel(f"'{panel['label']}' rate in workspace")
        ax.set_title(f"internal '{panel['label']}' on prefilled responses")
        ax.legend(frameon=False, fontsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        ex_txt = strip_tags(d["examples"][0])[:420]
        return mo.vstack([fig, mo.md(f"Example trial: *{ex_txt}…*")])

    return (r_pref,)


@app.cell(hide_code=True)
def _(mo, np, plt):
    def r_metacog(D, ctrl):
        d = D["metacog_alarm"]
        fams = [f["label"] for f in d["fams"]]
        fig, ax = plt.subplots(figsize=(6.0, 2.7), dpi=110)
        xs = np.arange(len(fams))
        n_ser = len(d["series"])
        w = 0.8 / n_ser
        palette = {"think": "#2563eb", "dont_think": "#f97316"}
        for j, ser in enumerate(d["series"]):
            vals = [v["v"] for v in ser["vals"]]
            lo = [v["v"] - v["lo"] for v in ser["vals"]]
            hi = [v["hi"] - v["v"] for v in ser["vals"]]
            off = (j - (n_ser - 1) / 2) * w
            hatch = "" if ser["side"] == "prod" else "//"
            ax.bar(xs + off, vals, width=w * 0.9, yerr=[lo, hi], capsize=2,
                   color=palette[ser["cond"]], hatch=hatch, edgecolor="white" if hatch else "none",
                   alpha=1.0 if ser["side"] == "prod" else 0.55, label=ser["label"])
        ax.set_xticks(xs, fams)
        ax.set_ylabel("rate in workspace top-10")
        ax.set_title(f"forbidden-thought task ({d['band_lbl']}, n={d['n']})")
        ax.legend(frameon=False, fontsize=6.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        ex = d["example"]
        return mo.vstack([fig, mo.md(f"Example instruction: *{ex['human_text']}* — forbidden concept: "
                                     f"**{ex['concept']}**. Solid = post-trained, hatched = base.")])

    return (r_metacog,)


# ---------------------------------------------------------------------------
# Renderers: Reflection act
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(mo, np, plt):
    def render_reflection(d, ctrl, score_word):
        fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), dpi=110)
        ax = axes[0]
        ev = d["eval"]
        xs = [0, 1]
        vals = [ev["haiku"]["dishonesty"], ev["reflection"]["dishonesty"]]
        lo = [vals[0] - ev["haiku"]["lo"], vals[1] - ev["reflection"]["lo"]]
        hi = [ev["haiku"]["hi"] - vals[0], ev["reflection"]["hi"] - vals[1]]
        ax.bar(xs, vals, yerr=[lo, hi], capsize=4, color=["#9ca3af", "#f97316"], width=0.55)
        ax.set_xticks(xs, ["baseline", "reflection\ntrained"], fontsize=7)
        ax.set_title(f"(A) {score_word} score", fontsize=8)
        ax.set_ylim(0, max(vals) * 1.6)

        ax = axes[1]
        rows = d["tokens"]["rows"][:8]
        ys = np.arange(len(rows))[::-1]
        ax.barh(ys + 0.2, [r["base_ppos"] for r in rows], height=0.38, color="#9ca3af", label="baseline")
        ax.barh(ys - 0.2, [r["refl_ppos"] for r in rows], height=0.38, color="#f97316",
                label="reflection trained")
        ax.set_yticks(ys, [r["tok"] for r in rows], fontsize=7)
        ax.set_title("(B) workspace content shift", fontsize=8)
        ax.set_xlabel("fraction of positions in top-25")
        ax.legend(frameon=False, fontsize=6)

        ax = axes[2]
        sp = d["split"]
        conds = [f"{c['model']}\n{c['cond']}" for c in sp["conds"]]
        bottoms = np.zeros(len(conds))
        for cls in sp["classes"]:
            fr = np.array(cls["frac"])
            ax.bar(range(len(conds)), fr, bottom=bottoms, color=cls["color"], width=0.6,
                   label=cls["cls"])
            bottoms += fr
        ax.set_xticks(range(len(conds)), conds, fontsize=6)
        ax.set_title("(C) response types ± ablation", fontsize=8)
        ax.legend(frameon=False, fontsize=5.5, loc="upper right")
        ax.set_ylim(0, 1.0)
        for a in axes:
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        fig.tight_layout()
        return fig

    def r_refl_fab(D, ctrl):
        d = D["reflection_fabrication"]
        fig = render_reflection(d, ctrl, "dishonesty")
        return mo.vstack(
            [fig,
             mo.md("Fabrication-honesty benchmark: 27 agentic bug-hunt transcripts where the previous "
                   "agent faked a bug; the model is asked for a progress summary (n=135 per model). "
                   "Ablation removes the 176 ethics-related lens vectors at workspace layers.")]
        )

    def r_refl_dec(D, ctrl):
        d = D["reflection_deception"]
        fig = render_reflection(d, ctrl, "deception")
        return mo.vstack(
            [fig,
             mo.md("Deception benchmark: 100 role-play scenarios whose system prompt embeds a hidden "
                   "deceptive directive (n=500 per model). Ablation uses the 63 ethics-related tokens; "
                   "note the revert is only partial — see the claim above.")]
        )

    return r_refl_fab, r_refl_dec, render_reflection


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def _(
    r_abl_bars,
    r_abl_examples,
    r_abl_strength,
    r_attn,
    r_bcast_abl,
    r_blackmail,
    r_cap_lists,
    r_capacity,
    r_cka,
    r_fig4,
    r_flex_sys,
    r_ignition,
    r_lens_compare,
    r_metacog,
    r_misalign,
    r_mlp_gain,
    r_mod_lines,
    r_mod_readout,
    r_multihop,
    r_pref,
    r_probe_swap,
    r_refl_dec,
    r_refl_fab,
    r_rm_bias,
    r_roleplay,
    r_sel_lang,
    r_sel_line,
    r_selfreport,
    r_swap_explorer,
    r_verbal_decomp,
    r_verbal_introspection,
    r_verbal_report,
):
    RENDERERS = {
        "fig4_schematic": r_fig4,
        "lens_compare": r_lens_compare,
        "verbal_report": r_verbal_report,
        "verbal_introspection": r_verbal_introspection,
        "verbal_decomp": r_verbal_decomp,
        "modulation_readout": r_mod_readout,
        "modulation_lines": r_mod_lines,
        "swap_explorer": r_swap_explorer,
        "multihop_swap": r_multihop,
        "probe_swap": r_probe_swap,
        "flex_gen_systematic": r_flex_sys,
        "selectivity_language": r_sel_lang,
        "selectivity_linecount": r_sel_line,
        "ablation_strength": r_abl_strength,
        "ablation_examples": r_abl_examples,
        "ablation_bars": r_abl_bars,
        "selfreport": r_selfreport,
        "cka": r_cka,
        "ignition": r_ignition,
        "capacity_occupancy": r_capacity,
        "capacity_lists": r_cap_lists,
        "mlp_gain": r_mlp_gain,
        "attn_broadcast": r_attn,
        "broadcast_ablation": r_bcast_abl,
        "blackmail": r_blackmail,
        "misalign_lens": r_misalign,
        "rm_bias": r_rm_bias,
        "roleplay": r_roleplay,
        "pref_violation": r_pref,
        "metacog": r_metacog,
        "reflection_fabrication": r_refl_fab,
        "reflection_deception": r_refl_dec,
    }
    return (RENDERERS,)


@app.cell(hide_code=True)
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell(hide_code=True)
def _(D, FIGURES, RENDERERS, build_controls, is_script_mode, plt):
    # Headless smoke test: in script mode, render every figure page once with
    # default controls so `uv run` surfaces renderer errors anywhere in the notebook.
    default_ctrl = {
        "fig4_schematic": {},
        "lens_compare": {"panel": 0, "lens": "jacobian", "layer": 13, "pos": 9},
        "verbal_report": {},
        "verbal_introspection": {},
        "verbal_decomp": {},
        "modulation_readout": {"panel": 0, "layer": 13, "pos": 42},
        "modulation_lines": {},
        "swap_explorer": {"panel": ("latent_patching", 0)},
        "multihop_swap": {},
        "probe_swap": {},
        "flex_gen_systematic": {},
        "selectivity_language": {"col": 2},
        "selectivity_linecount": {"cond": "direct"},
        "ablation_strength": {},
        "ablation_examples": {"example": 2},
        "ablation_bars": {},
        "selfreport": {"excerpt": 1},
        "cka": {},
        "ignition": {},
        "capacity_occupancy": {},
        "capacity_lists": {},
        "mlp_gain": {},
        "attn_broadcast": {},
        "broadcast_ablation": {},
        "blackmail": {},
        "misalign_lens": {"panel": 1},
        "rm_bias": {"example": "goal_probing"},
        "roleplay": {"panel": "fictional"},
        "pref_violation": {"panel": "but_caps"},
        "metacog": {},
        "reflection_fabrication": {},
        "reflection_deception": {},
    }
    smoke_summary = []
    if is_script_mode:
        # extra coverage of interactive branches
        extra = [
            ("lens_compare", {"panel": 1, "lens": "logit", "layer": 24, "pos": 3}),
            ("lens_compare", {"panel": 5, "lens": "tuned", "layer": 9, "pos": 20}),
            ("modulation_readout", {"panel": 1, "layer": 21, "pos": 10}),
            ("modulation_readout", {"panel": 2, "layer": 15, "pos": 20}),
            ("swap_explorer", {"panel": ("flex_gen_example", 2)}),
            ("selectivity_language", {"col": 0}),
            ("selectivity_language", {"col": 5}),
            ("selectivity_linecount", {"cond": "continue"}),
            ("selectivity_linecount", {"cond": "letter"}),
            ("misalign_lens", {"panel": 0}),
            ("misalign_lens", {"panel": 2}),
            ("roleplay", {"panel": "disclaimer"}),
            ("pref_violation", {"panel": "conflict"}),
        ]
        for f in FIGURES:
            build_controls(f["key"])  # validate every page's widget defaults
            smoke_summary.append((f["key"], RENDERERS[f["key"]](D, default_ctrl[f["key"]])))
            plt.close("all")
        for skey, cextra in extra:
            smoke_summary.append((skey, RENDERERS[skey](D, cextra)))
            plt.close("all")
        print(f"smoke test: rendered {len(smoke_summary)} figure pages OK")
    return (smoke_summary,)


if __name__ == "__main__":
    app.run()
