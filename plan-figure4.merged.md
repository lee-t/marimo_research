# Merged Plan: Interactive Figure 4 Reconstruction Notebook

This plan merges the technical rigor and data manifest of `plan-figure4.prompt_gptluna.md` with the interactive UI elements and reactive cell-level skeleton of `plan-figure4.prompt_alphaxiv.md`. It incorporates critical evaluations from `plan-figure4.eval_claude.md` and direct verification from the bioRxiv paper.

---

## 1. Scientific Context & Parameters

* **Paper Citation**: *Language Modeling Materializes a World Model of Protein Biology* (Candido, Rives et al., bioRxiv, June 2026).
* **Sparse Autoencoder (SAE) Provenance**:
  * Trained on ESMC 6B representation activations at **Layer 60** (approximately 3/4 depth).
  * Feature codebook size: **$2^{14} = 16,384$** features.
  * Sparsity parameter: **TopK $k = 64$** (at most 64 active features per residue position).
  * Training dataset: 8 billion tokens from the ESMC language model training distribution (comprising UniRef90, MGnify, and JGI).
  * Input dimensions: `d_model = 2560`, maximum sequence length 2,048.
* **Feature Complexity Categories & Counts** (Table S13 / Page 11 / Figure 4B):
  1. **Residue identity** (88 features): Single amino acids or chemically-defined classes.
  2. **Secondary structure** (1,895 features): Local ordered backbones (helices, strands, turns).
  3. **Tertiary motif** (1,554 features): 3D spatial arrangements, helix packing, salt bridges.
  4. **Domain / fold** (2,546 features): Broad activation across entire folds or families.
  5. **Disorder / low-complexity** (686 features): Intrinsically disordered regions.
  6. **Biochemical microenvironment** (1,770 features): Physicochemical surface patches (hydrophobic, charge-dense).
  7. **Localization / topology** (2,319 features): Recognition signals, organellar sorting, membrane topology.
  8. **Functional site / region** (5,382 features): Catalytic sites, ligand-binding, post-translational modifications.

---

## 2. Notebook Architecture & Data Layer

We will implement a dual-entry architecture sharing a common data model:
1. **CPU/API Notebook (`notebooks/figure4_sae.py`)**: CPU-friendly version utilizing the `ESMCForgeInferenceClient` to query the Biohub API for embeddings/SAE activations and the 16,384-row `ESMC-SAE-Features` metadata table. Requires setting `BIOHUB_TOKEN`.
2. **GPU/Local Notebook (`notebooks/figure4_sae_molab.py`)**: GPU-driven version running local inference via Hugging Face weights (`biohub/ESMC-6B` and `biohub/ESMC-6B-sae-k64-codebook16384`).
3. **Shared Data Layer (`scripts/prepare_figure4_data.py`)**: Downloads and caches static metadata, NMF weights, UMAP coordinates, PDB/ESMFold2 structure coordinates, and dataset manifests. Large artifacts (e.g. model weights) are cached outside of git.

---

## 3. Cell-by-Cell Interactive Panel Plan (A–G)

### Setup & Imports
* **Cell 1**: Imports (`marimo`, `numpy`, `pandas`, `plotly.express`, `plotly.graph_objects`, `umap`, `py3Dmol`/`nglview`). Defines `DATA_DIR` and loads the dataset manifest.
* **Cell 2**: Loads static artifacts (feature specificity/granularity, decoder weights, kinase family subsets, NMF topic distributions).

---

### Panel A: Specificity–Granularity Space
* **Goal**: Scatter plot of all 16,384 features to show the landscape of the SAE concept space.
* **Data Sources**:
  * **Specificity**: Pfam domain Shannon entropy z-score calculated over the **~208M UniRef90 reference database** (release 2025 03).
  * **Granularity**: Otsu-thresholded mean contiguous run length (MCRL) of feature activations within sequences.
* **UI Controls**:
  * `threshold = mo.ui.slider(1.0, 10.0, step=0.5, label="Min Activation Threshold")`
  * `color_mode = mo.ui.dropdown(["category", "density"], value="category")`
* **Plotly Scatter**: Plot all 16,384 features. Hovering displays Feature ID, Category, and agent-generated Summary.
* **Reactivity**:
  * Brush selection filters a reactive bar chart showing category breakdowns.
  * Clicking a feature point sets the **selected feature** and updates Panel B (detail card) and Panel D (neighborhood view).

---

### Panel B: Feature Category Showcase
* **Goal**: Show example features from each of the 8 complexity categories, with per-residue activation on 3D structures.
* **UI Controls**:
  * Reactive detail panel driven by the selected feature from Panel A.
* **Layout**:
  * If no feature is clicked, display a curated gallery of 1-2 exemplary features per category.
  * When a feature is selected:
    * Show feature ID, category badge, and agent-generated text summary.
    * **Top-activating proteins table**: Displays top 5 SwissProt proteins with organisms and max-activation values.
    * **Per-residue activation strip**: A sequence heatmap aligned with secondary structure annotations.
    * **3D structure viewer** (`py3Dmol` or `nglview`): Renders a representative protein structure colored by normalized activation intensity (orange colormap).

---

### Panel C: Nucleophilic Elbow (Feature 6716)
* **Goal**: Demonstrate how a single feature (F6716) identifies the nucleophilic elbow catalytic motif across convergent structural folds.
* **Data Context**:
  * The Du et al. cohort of 120 carbonyl-nucleophile enzymes, containing 99 elbow-containing enzymes and 21 control enzymes (note: the paper has a minor typo on page 78 claiming 119 total, but control statistics and Figure S30 verify 120).
  * Feature F6716 activates on the nucleophilic residue in **75 of 99** elbow enzymes spanning 25 distinct folds, and does not activate in **17 of 21** non-elbow controls.
* **UI Controls**:
  * `protein_selector = mo.ui.dropdown(options=["1ic6", "1mt5", "1ocl", ...], value="1ic6")` to toggle between the 4 main displayed protein structures in the paper.
* **Display**:
  * **Sequence activation plot**: Line chart showing F6716 activation vs residue position with the catalytic nucleophile annotated.
  * **3D structure**: Colored by F6716 activation strength with the nucleophile highlighted as spheres.
  * **Summary grid**: Interactive table of all 119 cohort enzymes, color-coded by activation status, fold, and enzyme class.

---

### Panel D: Feature Neighborhoods & Feature Splitting
* **Goal**: Visualize the decoder-space neighborhood around the selected feature and show feature splitting across codebook sizes.
* **Data Sources**:
  * Decoder weight matrix $W_{\text{dec}} \in \mathbb{R}^{16384 \times 2560}$.
  * Features from coordinate sweeps: $2^{13}$ to $2^{17}$.
* **UI Controls**:
  * `codebook_toggle = mo.ui.checkbox_group(["2^13", "2^14", "2^16"], value=["2^13", "2^14", "2^16"])`
  * `sim_cutoff = mo.ui.slider(0.05, 0.3, step=0.01, value=0.1, label="Cosine Similarity Cutoff")`
* **UMAP Plot**:
  * Projects decoder vectors with cosine similarity $\ge 0.1$ to the selected feature.
  * Points are colored by category; dot sizes indicate codebook dimensionality ($2^{13}$ vs $2^{14}$ vs $2^{16}$), showing how concepts partition.
* **Feature Splitting Case Study**:
  * Display peptidase family S1 (332 sequences) and S8 (60 sequences) catalytic triad features. Show how a single feature at $2^{13}$ (F6960) splits into separate family-specific features at $2^{17}$ (F77290 and F109350).

---

### Panel E: Kinase Compositional Grammar
* **Goal**: Showcase how simple (universal) and complex (family-specific) features compose to represent kinase catalytic machinery.
* **Data Context**:
  * Analysis of 35 kinase proteins across 7 major families (CAMK, AGC, TK, TKL, CK1, STE, CMGC).
  * Identification of **531 kinase-universal** and **254 family-specific** features.
* **UI Controls**:
  * `kinase_family_selector = mo.ui.dropdown(["CAMK", "AGC", "TK", "TKL", "CK1", "STE", "CMGC"], value="CAMK")`
  * `kinase_feature_selector = mo.ui.dropdown(P_loop_features, value="F119")`
* **Kinase P-loop Composition Table**:
  * Lists the 11 named P-loop features ranging from universal to specific (`F792`, `F10583`, `F1635`, `F3614`, `F10646`, `F278`, `F1013`, `F119`, `F4266`, `F4787`, `F6171`).
* **3D CAMK1 Structure Viewer**:
  * Load CAMK1 predicted structure. Highlights key functional regions (P-loop, $\alpha$C-helix, HRD, DFG, activation segment, autoinhibitory segment). Color-maps selected P-loop feature or overlays kinase-universal density.
* **Feature Comparison Plot**:
  * Compares sequence traces of the general loop feature `F3614` against the kinase-specific `F119` across Src (kinase), RNase Z (non-kinase), and fungal lipase (non-kinase).
* **Family-Specific Regulatory View**:
  * Displays the four paper-exemplified family-specific features on structure: tyrosine kinase SH3-binding (Src, `F872`), AGC C-terminal phosphorylation (AKT1, `F3042`), TKL innate immunity (IRAK4, `F1777`), and CK1$\alpha$ unique domain features (`F2046`).

---

### Panel F: NMF Topics Across Domains of Life
* **Goal**: Show how feature combinations (topics) represent functional profiles across the tree of life.
* **Implementation Synthesis (Venn Overlap + Heatmap)**:
  * To resolve the discrepancy between the plans, the panel will feature a tabbed layout:
    * **Tab 1: Domain-Overlap Distribution (Figure 4F)**: A Venn-like or Euler/sharing diagram representing the overlap of the 3,000 NMF topics across Archaea, Bacteria, and Eukaryota. Highlights the **$n = 1,103$ universal topics** shared across all three domains.
    * **Tab 2: Proteome Topic Heatmap (Figure S37)**: Heatmap plotting topic weights for the **395 selected topics** (5 universal + 5 dominant/specific per proteome) across the **81 Quest for Orthologs (QfO) reference proteomes** (7 Archaea, 23 Bacteria, 51 Eukaryota).
* **Heatmap Details**:
  * X-axis: 81 proteomes organized phylogenetically within kingdoms.
  * Y-axis: 395 selected topics.
  * Color: topic weight (per-topic normalized, columns sum to 1).
  * Clicking a row in the heatmap sets the **selected topic** and drives Panel G.

---

### Panel G: Topic Profile Cards
* **Goal**: Deep-dive into topic composition, taxonomic distribution, and structural showcases.
* **UI Controls**:
  * Reactive panel driven by clicking a topic row in Panel F.
* **Display**:
  * **Topic composition**: Lists top weighted features and descriptions for the selected topic.
  * **Taxonomic profile**: Bar chart of topic weight across kingdoms.
  * **CURATED SHOWCASES (Default View)**: Tabbed container showing the three main paper showcases:
    1. **Universal** (ATP synthase): features representing multi-pass transmembrane helices, rotary pore ring.
    2. **Lineage-specific** (Immunoglobulins): vertebrate-specific adaptive immunity beta-sandwich framework.
    3. **Cross-lineage** (Outer-membrane $\beta$-barrels): endosymbiotic origins reflected in 863 feature compositions.
  * Renders PDB structures colored by the topic's highest-weighted features.

---

## 4. Verification & Validation Protocol

1. **Sparsity Constraints**: In live CPU/GPU inference, verify that any input sequence produces an activation matrix of size $L \times 16,384$ where no residue position has more than 64 active features.
2. **Numeric Invariants Check**:
   * Panel A: Plot exactly 16,384 features.
   * Panel B: Total of 8 categories with accurate feature counts.
   * Panel C: Cohort size of 120 enzymes (99 elbow, 21 controls). Verify F6716 activation checks out (75/99 and 17/21).
   * Panel D: Cosine similarity cutoff $\ge 0.1$. UMAP $n\_neighbors=15, min\_dist=0.1$.
   * Panel E: Check 11 P-loop feature IDs, 531 kinase-universal, 254 family-specific.
   * Panel F & S37: Topic overlap numbers (1,103 universal), proteome set (81 QfO), selected topic set (395).
3. **Layout Comparison**: Compare the visual layout, axes, and color schemes against the bioRxiv paper PDF.
