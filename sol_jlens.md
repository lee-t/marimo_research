# Jacobian Lens Paper Notebook Plan

## Source and Goal

Build `jlens_global_workspace.py` as a marimo teaching notebook for *Verbalizable Representations Form a Global Workspace in Language Models* (Transformer Circuits, 2026). The notebook should explain the paper figure by figure without requiring a model download for its normal path.

The archived article and source data live in `downloads/workspace-global-workspace/`. It contains 94 figures (84 interactive), compact JSON data for most interactive plots, and static image assets.

## Teaching Story

The notebook should establish one causal argument in order:

1. **Read:** the Jacobian lens maps an intermediate residual-stream activation into a vocabulary-ranked description of what it is disposed to make the model say.
2. **Manipulate:** those readable directions can be injected, swapped, or ablated.
3. **Reason:** the directions expose silent intermediates, and swapping an intermediate can redirect the answer.
4. **Reuse:** the same concept direction can support multiple downstream operations.
5. **Select:** flexible report and reasoning depend on the J-space more than routine/automatic processing.
6. **Explain structure:** workspace-like content occupies a middle layer band, has limited capacity, and is preferentially broadcast across depth and token positions.
7. **Apply carefully:** J-lens readouts can make silent alignment-relevant cognition observable; counterfactual reflection training changes J-space contents and behavior.

The notebook must present these results as evidence about a computational organization, not as evidence that a model is conscious.

## Paper Methods to Explain

- For residual stream activation `h_l`, average the layer-to-final residual Jacobian over a corpus, source positions, and current/future target positions:

  `J_l = E[d h_final,t' / d h_l,t]`.

- Read an activation through the model's own unembedding:

  `lens(h_l) = softmax(W_U norm(J_l h_l))`.

- Treat rows of `W_U J_l` as token-indexed J-lens vectors.
- Define the J-space operationally as sparse, nonnegative combinations of a small number of J-lens vectors; use gradient pursuit for decomposition.
- Intervene by steering, projection ablation, or coordinate swap. A two-token coordinate swap preserves the component orthogonal to the pair's span.
- Separate observational readouts from causal evidence, and consistently surface the paper's controls: logit/tuned lenses, random directions, matched-norm perturbations, non-J-space remainders, and random/layer-matched heads.

## Figure Coverage

### Guided figures to reproduce from archived data

| Figure | Notebook treatment | Teaching role |
|---|---|---|
| 4 | Interactive schematic | Define fitting, reading, and coordinate swapping |
| 9 | Layer readout walkthrough | Directed modulation while surface output is fixed |
| 13 | Clean vs swapped distributions | Causal silent-intermediate reasoning |
| 17 | Layer/step timeline | Arithmetic intermediates emerge in causal order |
| 19 | Success versus workspace loading | Generalization succeeds when the argument is loaded |
| 24 | Task-by-ablation chart | Flexible tasks are much more J-space-dependent |
| 27-30 | CKA, phase, ignition, occupancy views | Middle-layer workspace, selection, and bottleneck |
| 34 | Broadcast-head ablation | Specialized heads preserve and relay J-space contents |
| 36 | Eval-awareness intervention | Auditing result and causal safety caveat |
| 49 | Reflection-training panels | Training changes J-space contents and behavior |

### Full atlas

Provide a navigator for all Figures 1-94. Each entry should show its original caption, a short question/result/caveat explanation, and identify whether the notebook currently has a replot. Organize the navigation around the paper's figure ranges:

- 1-5: framing and method
- 6-8: verbal report
- 9-11: directed modulation
- 12-17: internal reasoning
- 18-19: flexible generalization
- 20-26: selectivity and ablation
- 27-34: layer structure, capacity, broadcast
- 35-41: alignment auditing
- 42-46: post-training perspective
- 47-50: counterfactual reflection training
- 51-94: appendix methods, robustness, extensions, and mechanistic localization

## Technical Architecture

- Place the notebook in the repository root: `jlens_global_workspace.py`.
- Use a PEP 723 header with only `marimo`, `numpy`, and `matplotlib` for the normal CPU path.
- Load archived JSON lazily from `downloads/workspace-global-workspace/`; do not download models or weights in script mode.
- Keep all notebook code hidden. Place title, controls, summary, selected-figure explanation, and visual output before helper cells.
- Use a single figure selector, a compact guided-story selector, and only a small number of figure-specific controls.
- Replot source data directly where an archived JSON file exists; label every visualization as paper data, not a replication on the current machine.
- Include a GPU-lab reference section, not active inference code. It should contain the exact base-model/lens pairings and the minimum `jlens` API needed to apply a fitted lens.

## Weight Demonstration Plan

The released lens files are transport matrices, not standalone inference models. Each requires the exact base model and tokenizer.

| Preset | Base model | Lens path | Lens size | Use |
|---|---|---|---:|---|
| Gemma 3 12B IT | `google/gemma-3-12b-it` | `gemma-3-12b-it/jlens/Salesforce-wikitext/gemma-3-12b-it_jacobian_lens.pt` | 1.39 GB | Preferred live demonstration; Gemma access approval required |
| Gemma 3 12B PT | `google/gemma-3-12b-pt` | `gemma-3-12b/jlens/Salesforce-wikitext/gemma-3-12b-pt_jacobian_lens.pt` | 1.39 GB | Base versus instruction-tuned comparison |
| Qwen 3.6 27B | `Qwen/Qwen3.6-27B` | `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt` | 3.30 GB | Public, hardware-heavy demonstration |

Gemma's lenses were fit on Salesforce WikiText with up to 1,000 128-token prompts. Qwen3.6 is a 64-layer, 5,120-hidden-size model, so its fitted lens is necessarily large. Keep all download and GPU requirements explicit and opt-in.

## Verification

Run:

```powershell
uvx marimo check --strict jlens_global_workspace.py
uv run jlens_global_workspace.py
```

The script path must run using only the archived paper data. The GPU lab is documentation and must not execute automatically.
