

<alphaxiv-thinking-title title="Analyzing Figure 4 structure" />

Here's a detailed plan for recreating Figure 4 of the paper as an interactive marimo notebook.

---

## Overview

**Paper**: [Language Modeling Materializes a World Model of Protein Biology](https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1) (Candido et al., 2026)

**Figure 4** visualizes how sparse autoencoders (SAEs) trained on ESMC 6B (layer 60) representations decompose protein biology into interpretable features spanning multiple levels of biological complexity — from individual residues to functional themes across the tree of life. Recreating it as a marimo notebook would allow users to interactively explore the feature space, click on individual features, zoom into neighborhoods, and examine specific proteins.

The full figure has 7 panels (A–G). Below is a cell-by-cell plan.

---

## Setup & Data Loading

**Cell 1 — Imports & configuration**
- `import marimo as mo`
- `import numpy as np`, `pandas as pd`, `plotly.express as px`, `plotly.graph_objects as go`
- `import umap`, `sklearn`, `scipy`
- Define a `DATA_DIR` pointing to precomputed artifacts

**Cell 2 — Load precomputed SAE artifacts**
Load data that must be precomputed from the paper's pipeline (notebook assumes these exist as `.parquet` or `.npy` files):

| Artifact | Description |
|---|---|
| `sae_features.parquet` | 16,384-dim sparse feature vectors for ~195K SwissProt proteins (max-pooled per protein) |
| `feature_metadata.parquet` | Per-feature: specificity z-score, granularity score, agent-generated description, category label |
| `decoder_weights.npy` | SAE decoder matrix W_dec (16384 × 2560) for cosine neighbor computations |
| `feature_umap.npy` | Precomputed 2D UMAP of the 16,384 features in decoder space |
| `proteins_of_interest.parquet` | Subsets for panels C, E, G (nucleophilic elbow dataset, kinases, reference proteomes) |
| `nmf_topics.parquet` | NMF topic weights (3,000 topics × 16,384 features) for 81 reference proteomes |
| `go_feature_mapping.parquet` | Feature-to-GO-biological-process mapping with precision thresholds |

---

## Panel A: Specificity–Granularity Space

**Goal**: Scatter plot of all 16,384 features, with axes being specificity (Pfam-entropy z-score) and granularity (mean contiguous run length), colored by category.

**Cell 3 — Interactive scatter plot**

```python
# Reactive slider: activation threshold
threshold = mo.ui.slider(1.0, 10.0, step=0.5, label="Min activation threshold")

# Reactive dropdown: color by category or density
color_mode = mo.ui.dropdown(["category", "density"], value="category")

# Plotly scatter: hover shows feature_id, category, summary description
fig = px.scatter(
    feature_metadata, x="specificity_z", y="granularity",
    color="category" if color_mode.value == "category" else None,
    hover_data=["feature_id", "summary"],
    opacity=0.6, size=[1]*len(feature_metadata)
)
```

**Interactivity**:
- **Hover**: Show feature ID, category, and the agent-generated description
- **Click**: Select a feature → drives Panel D (neighborhood view) and Panel B (detail card)
- **Brush select**: Filter to features in a region, show aggregate category breakdown
- **Toggle**: Show only features above a minimum activation threshold
- **Density mode**: Replace categorical colors with a 2D KDE overlay

**Cell 4 — Category breakdown bar chart** (linked to brush selection)
Reactive bar chart showing how many features in the selected region belong to each category (Residue identity, Secondary structure, Tertiary interactions, Domain topology, Disorder, Microenvironment, Localization/signals, Complex function).

---

## Panel B: Feature Example Cards

**Goal**: Show visual examples of individual features from each of the 8 categories, with per-residue activation on a representative protein structure.

**Cell 5 — Feature detail panel** (reactive, driven by click in Panel A)

```python
selected_feature = mo.ui.anywidget(...)  # receives click event from Panel A
```

When a feature is selected, display:
- **Feature ID** and **category badge**
- **Agent-generated description** (summary + activation pattern)
- **Top-activating proteins table** (top 5 SwissProt proteins with feature name, organism, activation value)
- **Per-residue activation strip**: A horizontal heatmap showing activation intensity along the sequence of a representative protein, aligned with secondary structure annotation
- **3D structure view** (using `nglview` or py3Dmol): Render the representative protein as a cartoon, colored by per-residue activation (orange colormap), with a slider to toggle between features

**Cell 6 — Gallery mode**
If no feature is selected, show a curated gallery of 1–2 example features per category (as seen in Figure 4B), each rendered as a small 3D structure thumbnail with a text caption.

---

## Panel C: Nucleophilic Elbow (Feature 6716)

**Goal**: Show that a single feature (F6716) activates on the nucleophilic elbow motif across diverse folds.

**Cell 7 — Multi-protein activation viewer**

```python
# Dropdown to select proteins from the nucleophilic elbow dataset
protein_selector = mo.ui.dropdown(
    options=nucl_elbow_proteins["protein_id"].tolist(),
    value="1ic6"  # default: proteinase K
)
```

Display for the selected protein:
- **Sequence activation plot**: Line chart showing F6716 activation (y-axis) vs residue position (x-axis), with annotations for the nucleophile residue, secondary structure regions
- **3D structure**: Protein cartoon with F6716 activation intensity mapped as color (orange → high), nucleophile residue shown as spheres
- **Context table**: Which fold, enzyme class, and nucleophile residue

**Cell 8 — Summary grid**
A grid/table showing all 99 nucleophilic elbow enzymes, with columns: PDB ID, fold, nucleophile, F6716 activation at nucleophile (Yes/No). Color rows by whether the feature activates. This mirrors the "75 of 99 relevant enzymes" claim.

---

## Panel D: Feature Neighborhoods in Decoder Space

**Goal**: UMAP of decoder directions with cosine similarity ≥ 0.1 to a selected anchor feature.

**Cell 9 — Neighborhood viewer** (reactive, driven by feature selection)

```python
anchor_id = mo.ui.anywidget(...)  # linked to Panel A click

# Filter decoder vectors with cos_sim >= 0.1 to anchor
neighbors = decoder_weights[cos_sim_mask]
neighbor_ids = feature_ids[cos_sim_mask]
```

- **UMAP scatter**: Points colored by feature category, sized by cosine similarity to anchor
  - Anchor feature shown as a star/diamond marker
  - Points from different SAE dimensionalities (2^13, 2^14, 2^16) shown as different marker sizes
- **Hover**: Feature ID, category, description
- **Click neighbor**: Opens Panel B for the clicked feature

**Cell 10 — Resizable SAE dimensionality toggle**
Radio buttons: [2^13, 2^14, 2^15, 2^16, 2^17]. When toggled, overlay features from that dimensionality on the same UMAP projection. This visually demonstrates feature splitting — a single feature at 2^13 may split into multiple features at 2^16.

---

## Panel E: Kinase Compositional Grammar

**Goal**: Show how simple and complex features compose to represent the kinase catalytic machinery.

**Cell 11 — Kinase feature table**
Display the 11 P-loop features (Figure 4E top) as an interactive table:

| Feature ID | Description | Specificity level |
|---|---|---|
| F792 | Generic loops/coils | Universal |
| F10583 | Generic loops/coils | Universal |
| F1635 | Glycine-rich flexible loops | Broad |
| F3614 | Glycine-rich flexible loops | Broad |
| F10646 | Glycine-rich flexible loops | Broad |
| F278 | Phosphate-binding (diverse) | Intermediate |
| F1013 | Phosphate-binding (diverse) | Intermediate |
| F119 | Kinase P-loop | Specific |
| F4266 | Kinase P-loop | Specific |
| F4787 | Kinase P-loop | Specific |
| F6171 | Kinase P-loop | Specific |

Each row clickable → highlights where that feature activates on the kinase structure.

**Cell 12 — Kinase 3D structure viewer**
- Load CAMK1 predicted structure (ESMFold2)
- **Dropdown to select a feature** → structure colored by that feature's per-residue activation
- **Toggle overlay**: Show all kinase-universal features simultaneously as a heatmap on structure
- **Region labels**: Annotated regions (P-loop, αC-helix, HRD, DFG, activation segment, autoinhibitory segment) shown as labeled arcs on the structure

**Cell 13 — Cross-family comparison**
- **Dropdown**: Select kinase family (CAMK, AGC, TK, TKL, CK1, STE, CMGC)
- **Bar chart**: Top 10 family-specific features ranked by activation strength, with descriptions
- **Structure**: Representative kinase structure with family-specific features highlighted

---

## Panel F: NMF Topic Distribution Across Domains of Life

**Goal**: Heatmap showing topic weights across 81 reference proteomes (Archaea, Bacteria, Eukaryota).

**Cell 14 — Interactive heatmap**

```python
# X-axis: 81 proteomes ordered phylogenetically
# Y-axis: ~395 topics (universal + lineage-specific)
# Color: topic weight (column-normalized)
```

- **Heatmap** using `plotly.imshow` with a diverging colormap
- **Row dendrogram** (optional) to cluster topics
- **Column colors**: Colored bar at top indicating kingdom (Archaea, Bacteria, Eukaryota)
- **Hover**: Proteome name, kingdom, topic index, top 3 features in that topic with descriptions
- **Click row**: Opens Panel G for that topic

---

## Panel G: Topic Profile Cards

**Goal**: Deep-dive into specific topics showing their feature composition, taxonomic distribution, and structural examples.

**Cell 15 — Topic deep-dive** (reactive, driven by click in Panel F)

Display for the selected topic:
- **Topic summary**: Most highly weighted features and their descriptions
- **Taxonomic distribution**: Bar chart showing which proteomes/kingdoms this topic is most prevalent in
- **UMAP inset**: Where clusters associated with this topic sit in the global protein UMAP (if available)
- **3D structure panels**: For the 3 showcased topics (ATP synthase, immunoglobulins, outer-membrane β-barrels), show PDB structures with the top-weighted features colored on the structure

**Cell 16 — Pre-loaded showcase cards** (default view, no topic selected)
Three expandable cards for the paper's highlighted examples:
1. **Universal** (ATP synthase) — features representing multi-pass transmembrane helices, rotary pore ring
2. **Lineage-specific** (Immunoglobulins) — features for Ig-like beta-sandwich framework
3. **Cross-lineage** (Outer-membrane β-barrels) — features reflecting the endosymbiotic origin

Each card contains a structure render, feature list, and taxonomic range description.

---

## Data Requirements

To actually run this notebook, you'd need precomputed output from the paper's pipeline:

1. **SAE features** for the SwissProt reference set (~195K proteins, 16,384-dim) — the paper trained SAEs on ESMC 6B layer 60 with TopK=64 sparsity
2. **Feature metadata** (specificity, granularity, descriptions) — requires the multi-agent GPT-5 pipeline described in Appendix A.4.3
3. **UMAP of decoder weights** — from the SAE decoder matrix columns
4. **Kinase feature analysis** (531 kinase-universal + 254 family-specific features) — requires running the kinase family logic from Appendix A.4.4.1
5. **NMF on 81 reference proteomes** (3,000 topics) — requires the NMF training from Appendix A.4.4.4
6. **Nucleophilic elbow dataset** (99 proteins, 32 folds) — from Du et al. 2025, Table S3 in the paper supplement

The paper's code and data are released at [github.com/Biohub/esm](https://github.com/Biohub/esm), including model weights, SAEs, and the atlas. The SAE features and SwissProt reference dataset would need to be computed using their pipeline.

---

## Implementation Notes

- **Framework**: marimo (reactive Python notebook) with `plotly` for 2D plots and `py3Dmol`/`nglview` for protein structures
- **State management**: All panel selections (feature click, protein dropdown, kinase family) should be marimo reactive `ui` elements with cross-cell reactivity
- **Performance**: The 16,384-feature scatter plot should render fine in plotly; for the full ~195K protein views, use downsampling or WebGL
- **Layout**: Use `mo.hstack`/`mo.vstack` with `mo.ui.tabs` for organizing panels; the figure can span a wide dashboard layout