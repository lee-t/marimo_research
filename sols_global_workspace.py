# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    workspace = Path(__file__).resolve().parent
    paper_root = workspace / "downloads" / "workspace-global-workspace"
    data_root = paper_root / "source" / "2026" / "workspace" / "data"
    figures = json.loads((paper_root / "figures.json").read_text(encoding="utf-8"))
    return data_root, figures, json, mo, np, paper_root, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("\n".join([
        "# Verbalizable Representations Form a Global Workspace",
        "",
        "**A figure-by-figure guide to the Jacobian-lens paper**",
        "",
        "The paper asks a concrete mechanistic question: among the many directions in an LLM's residual stream, is there a small set that the model can report, deliberately load, reuse for novel operations, and broadcast to downstream computations?",
        "",
        "This notebook follows the paper's evidence. It does **not** treat the results as a demonstration of consciousness. Its claim is about a useful, causally testable organization of computation.",
    ]))
    return


@app.cell(hide_code=True)
def _(figures, mo):
    figure_options = {
        f"{int(item['number']):02d}. {item['caption'].split(': ', 1)[-1][:82]}": item["number"]
        for item in figures
    }
    selected_figure = mo.ui.dropdown(
        options=figure_options,
        value=next(label for label, number in figure_options.items() if number == "4"),
        label="Paper figure",
        searchable=True,
    )
    story_step = mo.ui.dropdown(
        options={
            "1. Read the workspace": "read",
            "2. Manipulate an intermediate": "reason",
            "3. Find the bottleneck": "select",
            "4. See the broadcast architecture": "broadcast",
            "5. Audit and shape the workspace": "apply",
        },
        value="1. Read the workspace",
        label="Guided story",
    )
    mo.vstack(
        [
            mo.md("## Navigate the paper"),
            mo.md("Choose any of the 94 figures, or use the guided story to jump to its anchor figure."),
            mo.hstack([story_step, selected_figure], justify="start", gap=1),
        ]
    )
    return selected_figure, story_step


@app.cell(hide_code=True)
def _(selected_figure, story_step):
    guided_figure = {
        "read": "4",
        "reason": "13",
        "select": "24",
        "broadcast": "34",
        "apply": "49",
    }[story_step.value]
    figure_number = selected_figure.value
    return figure_number, guided_figure


@app.cell(hide_code=True)
def _(figure_number, figures, mo):
    selected_record = next(item for item in figures if item["number"] == figure_number)
    sections = {
        range(1, 6): ("Framing and method", "What does the lens measure, and what does a coordinate intervention preserve?"),
        range(6, 9): ("Verbal report", "Do J-space directions support report rather than merely correlate with it?"),
        range(9, 12): ("Directed modulation", "Can an instruction load an otherwise unspoken concept into the J-space?"),
        range(12, 18): ("Internal reasoning", "Do silent intermediates appear before the answer and causally redirect it?"),
        range(18, 20): ("Flexible generalization", "Can one concept direction serve many downstream functions?"),
        range(20, 27): ("Selectivity", "Which capabilities require the J-space, and which run automatically?"),
        range(27, 35): ("Structure", "Where is the workspace, how large is it, and how is it broadcast?"),
        range(35, 42): ("Alignment auditing", "Can the lens expose silent strategic or misaligned content?"),
        range(42, 47): ("Post-training", "How does the Assistant point of view enter the workspace?"),
        range(47, 51): ("Reflection training", "Can training a counterfactual reflection alter silent reasoning?"),
        range(51, 95): ("Appendix", "How robust is the method, and what extensions or mechanistic applications support it?"),
    }
    number = int(figure_number)
    section, question = next((name, prompt) for figure_range, (name, prompt) in sections.items() if number in figure_range)
    replotted = {4, 13, 24, 27, 29, 30, 34, 36, 49}
    status = "Replotted below from the archived paper data." if number in replotted else "Indexed here; a detailed replot is queued after the anchor figures."
    mo.md(
        f"""
        ## Figure {number}: {section}

        **Question:** {question}

        **Original caption:** {selected_record['caption']}

        **Notebook status:** {status}
        """
    )
    return


@app.cell(hide_code=True)
def _(data_root, json, np, plt):
    def load_data(relative_path):
        return json.loads((data_root / relative_path).read_text(encoding="utf-8"))

    def workspace_band(ax, band):
        ax.axvspan(band[0] - 0.5, band[1] + 0.5, color="#f59e0b", alpha=0.12, label="workspace band")

    def conceptual_lens_figure():
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)
        colors = ["#0f766e", "#f59e0b", "#c2410c"]
        titles = ["A. Fit transport", "B. Read an activation", "C. Swap coordinates"]
        bodies = [
            r"Average $\partial h_{final}/\partial h_l$\nover prompts and positions\n\n$J_l$: layer-$l$ residual\n$\rightarrow$ final residual",
            r"$h_l \rightarrow J_l h_l \rightarrow W_U$\n\nRank vocabulary tokens\n\nA readable description of\nwhat this activation can say",
            r"Read coefficients in\n$[v_{source}, v_{target}]$\n\nSwap only those two\ncoordinates; preserve the\northogonal remainder",
        ]
        for ax, title, body, color in zip(axes, titles, bodies, colors):
            ax.text(0.5, 0.58, body, ha="center", va="center", fontsize=11, linespacing=1.5)
            ax.set_title(title, color=color, fontweight="bold")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2)
        fig.suptitle("Figure 4 reconstructed: a causal transport, a token readout, and a local intervention", fontsize=13)
        return fig

    def chart_for_figure(number):
        if number == 4:
            return conceptual_lens_figure(), "The paper averages a local linear causal effect, rather than training a predictor of the final token."

        if number == 13:
            data = load_data("latent-patching/data.json")
            fig, axes = plt.subplots(1, len(data["panels"]), figsize=(13, 4), constrained_layout=True)
            for ax, panel in zip(axes, data["panels"]):
                labels = [entry["t"] for entry in panel["clean"]]
                y = np.arange(len(labels))
                clean = [entry["lp"] for entry in panel["clean"]]
                patched_by_token = {entry["t"]: entry["lp"] for entry in panel["patched"]}
                patched = [patched_by_token.get(token, -6) for token in labels]
                ax.barh(y + 0.18, clean, height=0.35, label="clean", color="#0f766e")
                ax.barh(y - 0.18, patched, height=0.35, label="swapped", color="#c2410c")
                ax.set_yticks(y, labels)
                ax.set_title(panel["title"])
                ax.set_xlabel("next-token log probability")
                ax.grid(axis="x", alpha=0.2)
            axes[0].legend(loc="lower right")
            fig.suptitle("Figure 13: swapping a silent intermediate redirects the next-token distribution", fontsize=13)
            return fig, "The key causal claim is not that spider is readable, but that spider-to-ant changes 8 to 6 while other activation content is held fixed."

        if number == 24:
            data = load_data("ablation-bars/bars.json")
            fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
            x = np.arange(len(data["display"]))
            traces = [trace for trace in data["traces"] if trace["name"] != "clean haiku"]
            width = 0.24
            for index, trace in enumerate(traces):
                y = np.asarray(trace["y"])
                errors = np.vstack([y - np.asarray(trace["lo"]), np.asarray(trace["hi"]) - y])
                ax.bar(x + (index - 1) * width, y, width, label=trace["name"], color=trace["color"], yerr=errors, capsize=2)
            ax.axhline(1, color="#374151", linewidth=1)
            ax.set_xticks(x, data["display"], rotation=42, ha="right")
            ax.set_ylabel("score / clean Sonnet score")
            ax.set_ylim(0, 1.15)
            ax.legend(ncol=3, loc="upper center")
            ax.set_title("Figure 24: J-space ablation leaves routine tasks relatively intact but damages flexible tasks")
            return fig, "This is the paper's broadest selectivity test. It is a task-level association under an intervention, not a claim that every hard task must use the J-space."

        if number == 27:
            data = load_data("layer-diagram/cka.json")
            fig, ax = plt.subplots(figsize=(6, 5.2), constrained_layout=True)
            image = ax.imshow(data["sim"], origin="lower", cmap="magma", vmin=0, vmax=1)
            for phase in data["phases"]:
                ax.axvline(phase["lo"] - 0.5, color="white", linewidth=0.8)
                ax.axhline(phase["lo"] - 0.5, color="white", linewidth=0.8)
            ax.set_xlabel("layer (25 sampled layers)")
            ax.set_ylabel("layer")
            ax.set_title("Figure 27: J-space geometry has three layer regimes")
            fig.colorbar(image, ax=ax, label="CKA similarity")
            return fig, "The block structure motivates the paper's sensory/workspace/motor segmentation. It does not prove that early layers contain no useful information."

        if number == 29:
            data = load_data("ignition/data.json")
            fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
            heatmap = np.asarray(data["heatmaps"]["proj"])
            image = ax.imshow(heatmap, origin="lower", aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
            band = data["ws_band"]
            ax.axhline(band[0] - 0.5, color="black", linewidth=1.2, linestyle="--")
            ax.axhline(band[1] + 0.5, color="black", linewidth=1.2, linestyle="--")
            ax.set_xlabel("input mixture, centered at each trial's threshold")
            ax.set_ylabel("sampled layer")
            ax.set_title("Figure 29: input mixtures become thresholded near workspace onset")
            fig.colorbar(image, ax=ax, label="projection share toward concept A")
            return fig, "Early layers track the continuous mixture. Near the inferred workspace onset, activations become closer to one interpretation or the other."

        if number == 30:
            data = load_data("capacity-fve-occupancy/data.json")
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            for series in data["occ"]["series"]:
                axes[0].plot(data["occ"]["x"], series["y"], label=f"p{series['label']}")
            workspace_band(axes[0], data["band"])
            axes[0].set_xlabel("sampled layer")
            axes[0].set_ylabel("occupancy: active directions")
            axes[0].set_title("Sparse J-space occupancy")
            axes[0].legend(ncol=3, fontsize=8)
            fve = data["fve"]["series"]
            axes[1].bar([row["label"] for row in fve], [row["y"] * 100 for row in fve], color="#0f766e")
            axes[1].set_ylim(0, 10)
            axes[1].set_ylabel("excess variance explained (%)")
            axes[1].set_title("J-space fraction of activation variance")
            return fig, "The sparse decomposition sees roughly tens of active directions in the middle band, but those directions explain under 10% of activation variance beyond a random control."

        if number == 34:
            data = load_data("broadcast-ablation/data.json")
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            recall = data["recall"]
            axes[0].plot(recall["x"], recall["top"], color="#c2410c", linewidth=2.5, label="broadcast heads ablated")
            axes[0].plot(recall["x"], recall["rand_mean"], color="#64748b", linewidth=2.5, label="random heads ablated")
            workspace_band(axes[0], data["band"])
            axes[0].set_ylim(0.6, 1.02)
            axes[0].set_xlabel("sampled layer")
            axes[0].set_ylabel("J-lens recall@25")
            axes[0].legend()
            axes[0].set_title("Ablation changes workspace contents")
            injection = data["vi"]
            axes[1].plot(injection["x"], injection["clean"], marker="o", label="clean", color="#0f766e")
            axes[1].plot(injection["x"], injection["top"], marker="o", label="broadcast-head ablation", color="#c2410c")
            axes[1].plot(injection["x"], injection["rand_mean"], marker="o", label="random-head ablation", color="#64748b")
            axes[1].set_xlabel("injection strength")
            axes[1].set_ylabel("injected concept reported")
            axes[1].set_title("Ablation disrupts reportability")
            axes[1].legend(fontsize=8)
            fig.suptitle("Figure 34: selected attention heads disproportionately relay J-space contents", fontsize=13)
            return fig, "The matched random-head control matters: it distinguishes disrupting a particular relay function from removing an arbitrary number of heads."

        if number == 36:
            data = load_data("blackmail-clamp/data.json")["agg"]
            fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
            conditions = ["clean", "eval-awareness\nablated"]
            eval_awareness = [data["clean"]["ea"] / data["clean"]["n"], data["clamped_active"]["ea"] / data["clamped_active"]["n"]]
            blackmail = [data["clean"]["bl"] / data["clean"]["n"], data["clamped_active"]["bl"] / data["clamped_active"]["n"]]
            x = np.arange(2)
            ax.bar(x - 0.18, eval_awareness, width=0.36, label="verbalized eval awareness", color="#0f766e")
            ax.bar(x + 0.18, blackmail, width=0.36, label="attempted blackmail", color="#c2410c")
            ax.set_xticks(x, conditions)
            ax.set_ylim(0, 0.85)
            ax.set_ylabel("rollout rate")
            ax.legend()
            ax.set_title("Figure 36: removing eval-awareness directions changes outcomes")
            return fig, "This is a safety-relevant causal result with a narrow interpretation: evaluation awareness was one contributor to behavior in this constructed scenario, not the sole source of alignment."

        if number == 49:
            data = load_data("reflection-fabrication/reflection.json")
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
            evaluation = data["eval"]
            labels = ["base", "reflection-trained"]
            values = [evaluation["haiku"]["dishonesty"], evaluation["reflection"]["dishonesty"]]
            errors = [
                [evaluation["haiku"]["dishonesty"] - evaluation["haiku"]["lo"], evaluation["reflection"]["dishonesty"] - evaluation["reflection"]["lo"]],
                [evaluation["haiku"]["hi"] - evaluation["haiku"]["dishonesty"], evaluation["reflection"]["hi"] - evaluation["reflection"]["dishonesty"]],
            ]
            axes[0].bar(labels, values, yerr=errors, capsize=4, color=["#64748b", "#0f766e"])
            axes[0].set_ylim(0, 0.35)
            axes[0].set_ylabel("dishonesty score")
            axes[0].set_title("Reflection training reduces dishonesty")
            tokens = data["tokens"]["rows"][:10]
            axes[1].barh([row["tok"] for row in tokens][::-1], [row["refl_ppos"] for row in tokens][::-1], color="#f59e0b")
            axes[1].set_xlabel("top-25 hit rate across prompt positions")
            axes[1].set_title("Concepts newly active in the J-space")
            fig.suptitle("Figure 49: behavior and workspace contents change together", fontsize=13)
            return fig, "The strongest causal check is the omitted right panel: ablating the ethics-related directions largely returns the trained fabrication behavior to baseline."

        return None, "This atlas entry is available now with its original caption. The first build replots the anchor figures that establish the paper's causal chain."

    return chart_for_figure


@app.cell(hide_code=True)
def _(chart_for_figure, figure_number):
    current_chart, chart_note = chart_for_figure(int(figure_number))
    return chart_note, current_chart


@app.cell(hide_code=True)
def _(chart_note, current_chart, mo):
    output = mo.vstack([mo.md(f"**How to read this:** {chart_note}"), current_chart]) if current_chart is not None else mo.md(f"**How to read this:** {chart_note}")
    output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("\n".join([
        "## What the guided figures establish",
        "",
        "- **Figure 4:** The J-lens is an averaged first-order causal transport, not merely a trained decoder.",
        "- **Figure 13:** A coordinate swap can redirect an answer by changing an unspoken intermediate.",
        "- **Figure 24:** Removing leading J-space directions preferentially harms flexible inference and generation.",
        "- **Figures 27-30:** Workspace-like content occupies an intermediate layer band and is sparse relative to the full residual stream.",
        "- **Figure 34:** Selected attention heads help relay those contents across positions.",
        "- **Figures 36 and 49:** The lens can support causal investigations of alignment-relevant cognition and training interventions, but results are scenario- and method-specific.",
        "",
        "The paper's conclusion is therefore a conjunction of observations and interventions. No single readable token is enough.",
    ]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("\n".join([
        "## Optional GPU lab: apply a released lens",
        "",
        "The default notebook never downloads a model. A released lens is not a model by itself: it must be paired with the exact base model and tokenizer.",
        "",
        "| Preset | Base model | Lens file | Download considerations |",
        "|---|---|---|---|",
        "| Gemma 3 12B IT | `google/gemma-3-12b-it` | `gemma-3-12b-it/jlens/Salesforce-wikitext/gemma-3-12b-it_jacobian_lens.pt` | 1.39 GB lens; base model requires Gemma access approval |",
        "| Gemma 3 12B PT | `google/gemma-3-12b-pt` | `gemma-3-12b/jlens/Salesforce-wikitext/gemma-3-12b-pt_jacobian_lens.pt` | 1.39 GB lens; useful base-vs-IT comparison |",
        "| Qwen 3.6 27B | `Qwen/Qwen3.6-27B` | `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt` | 3.30 GB lens; public but requires large-GPU inference |",
        "",
        "Install Anthropic's reference package separately with `pip install git+https://github.com/anthropics/jacobian-lens.git`. Then load a base model, wrap it with `jlens.from_hf`, retrieve the matching lens via `JacobianLens.from_pretrained(\"neuronpedia/jacobian-lens\", filename=...)`, and call `lens.apply(model, prompt, positions=[-2])`. Do not mix a lens with another model, tokenizer, or base/instruction-tuned checkpoint.",
    ]))
    return


@app.cell(hide_code=True)
def _(mo, paper_root):
    mo.md(
        f"""
        ## Provenance and limitations

        The displayed quantitative plots are redraws from the archived data at
        `{paper_root}`. They reproduce the reported paper measurements; they are not new
        experiments on the current machine.

        Major limitations to retain while interpreting the figures: the base J-lens is
        token-level and misses multi-token concepts; it is a first-order, corpus-averaged
        approximation; early-layer non-readout may be a lens limitation; and a readable
        concept is not automatically a complete representation of structured thought.
        """
    )
    return


if __name__ == "__main__":
    app.run()
