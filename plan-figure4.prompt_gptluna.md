**Plan: Interactive Figure 4**

The notebook will faithfully recreate Figure 4, “The ESMC latent space contains a reduction of protein biology...,” using the released ESMC SAE artifacts and paper parameters. It will have two entry points:

- `notebooks/figure4_sae.py` for CPU-friendly local testing through the Biohub API.
- `notebooks/figure4_sae_molab.py` for GPU execution with Hugging Face weights and checkpoints.

Both notebooks will share the same artifact schema and produce comparable outputs.

**Implementation Phases**

1. **Artifact Audit**
   - Pin ESMC 6B, layer 60, TopK `k=64`, and codebook size `16,384`.
   - Record the paper-derived SAE provenance: `d_model=2560`; codebooks `2^13` through `2^17`; TopK values `8, 16, 32, 64, 128`; 8 billion training tokens; UniRef90/MGnify/JGI training proportions of 35.7%/10.7%/53.6%; and maximum sequence length 2,048.
   - Create a manifest containing download URLs, model revisions, dataset releases, retrieval dates, and checksums.
   - Stage source data for all seven panels:
       - Panel A: specificity and granularity coordinates computed from the approximately 208M-protein UniRef90 release 2025 03 analysis, including the Pfam-derived specificity metric and Otsu-thresholded mean contiguous activation length.
       - Panel B: selected feature IDs, descriptions, activation examples, and the eight published categories with counts: residue identity (88), secondary structure (1,895), tertiary motif (1,554), domain/fold (2,546), disorder/low-complexity (686), biochemical environment (1,770), localization/topology (2,319), and functional site/region (5,382).
       - Panel C: the full Du et al. cohort of 119 carbonyl-nucleophile enzymes, including the 99 elbow-containing enzymes, the comparison cohort used for the 17-of-21 result, and the four displayed structures.
       - Panel D: decoder-neighborhood vectors for F6716 from layer-60 ESMC 6B SAEs across `2^13` through `2^17`; the main panel displays `2^13`, `2^14`, and `2^16`.
       - Panel E: the 35 kinase proteins across seven families, the 11 named P-loop features, CAMK1 regions and structure, F3614/F119 comparison data, and the four displayed family-specific examples.
       - Panel F: the main Figure 4 domain-overlap distribution of NMF feature combinations across Bacteria, Archaea, and Eukarya, including its published counts. Keep the 81 QfO proteome topic matrix and 395-topic selection as an optional supplementary Figure S37 extension, not as the main Panel F representation.
       - Panel G: the three topic compositions, exact feature IDs and weights, PDB structures, and taxonomic tree schematics for ATP synthase, immunoglobulins, and outer-membrane beta-barrels.
   - Treat missing source arrays as a data-preparation blocker rather than substituting synthetic values.

2. **Shared Data Layer**
   - Add a small shared utility module for manifest loading, feature metadata, sparse activation storage, validation, feature ranking, and panel-specific loaders.
   - Define a backend-neutral activation contract: one sparse `L x 16384` matrix per sequence after BOS/EOS removal.
   - Cache live results by sequence hash, model, SAE, normalization setting, and API/backend revision.
   - Keep large artifacts and model caches outside git.

3. **Biohub API Notebook**
   - Use the existing marimo patterns from [notebooks/esm_embeddings.py](notebooks/esm_embeddings.py) and [notebooks/cnn_probing.py](notebooks/cnn_probing.py).
   - Use `ESMCForgeInferenceClient`, `ESMProtein`, `LogitsConfig`, and `SAEConfig`.
   - Default to `esmc-6b-2024-12` and `esmc-6b-2024-12-sae-layer60-k64-codebook16384`.
   - Read `BIOHUB_TOKEN` from the environment and show a clear setup callout when it is absent.
   - Use the Biohub feature metadata endpoint and the 16,384-row `ESMC-SAE-Features` table for descriptions, categories, thresholds, normalization statistics, and decoder neighbors.

4. **Figure Panels**
   - **A:** Interactive scatter of all 16,384 features by specificity and local granularity, preserving the paper’s family-specific-to-universal and residue-level-to-domain-level axis directions, with feature hover/details.
   - **B:** Small multiples of the paper’s selected feature examples across the eight biological-complexity categories, retaining the category counts and the feature-specific activation thumbnails.
   - **C:** F6716 nucleophilic-elbow activation traces for the four paper proteins, with nucleophile annotations, the 119/99 source-cohort summary, and optional PDB structure coloring.
   - **D:** F6716 decoder neighborhood using cosine similarity threshold `0.1`, UMAP `n_neighbors=15`, `min_dist=0.1`, and point sizes for the main-panel `2^13`, `2^14`, and `2^16` codebooks; allow the broader `2^13`-`2^17` neighborhood set in the detail view.
   - **E:** Kinase compositionality with the named 11 P-loop features (`F792`, `F10583`, `F1635`, `F3614`, `F10646`, `F278`, `F1013`, `F119`, `F4266`, `F4787`, `F6171`), CAMK1 region densities, the F3614-versus-F119 comparison across Src/RNase Z/fungal lipase, and four displayed family-specific examples selected from the seven analyzed kinase families.
   - **F:** Main Figure 4’s NMF feature-combination taxonomy distribution as a three-domain overlap graphic for Bacteria, Archaea, and Eukarya, preserving published counts and colors. Add the 81-proteome, phylogenetically ordered, per-topic-normalized heatmap only as an explicitly labeled Figure S37 extension.
   - **G:** Universal ATP synthase, lineage-specific immunoglobulin, and cross-lineage outer-membrane beta-barrel topic examples with exact feature weights, feature-colored structures, and taxonomy trees.

5. **Interactive Controls**
   - Selected feature and anchor feature.
   - Activation normalization and threshold.
   - Top-N feature count.
   - Decoder similarity cutoff.
   - Kinase family and protein.
   - Domain-overlap selection for Panel F and optional proteome/topic selection for the Figure S37 extension.
   - Structure visibility and top-feature count.
   - Cached artifact versus live API mode.

6. **GPU/MoLab Notebook**
   - Load `biohub/ESMC-6B` and `biohub/ESMC-6B-sae-k64-codebook16384`.
   - Use `device_map="auto"`, reduced precision, inference mode, and sparse outputs.
   - Recompute sequence-level panels C and E and decoder neighborhoods in the notebook.
   - Keep panels A, F, and G artifact-backed because their full source calculations are too large for an interactive MoLab cell.
   - Document Hugging Face authentication and GPU/storage requirements.

7. **Preparation and Documentation**
   - Add `scripts/prepare_figure4_data.py` for resumable downloads, checksum validation, sparse serialization, and compact fixtures.
   - Add a `data/figure4` artifact README with provenance, expected shapes, licenses, and source citations.
   - Update [README.md](README.md) with both notebook commands and setup instructions.
   - Update [.gitignore](.gitignore) for model caches, generated structures, and large Figure 4 intermediates.
   - Reuse error handling and embedding-fetch conventions from [scripts/fetch_embeddings.py](scripts/fetch_embeddings.py).

**Verification**

1. Validate manifest checksums, model identifiers, feature ranges, array shapes, and panel metadata.
2. Run `uvx marimo check` on both notebooks.
3. Run focused tests for sparse activation alignment, feature ranking, topic normalization, and deterministic UMAP settings.
4. In API mode, verify one sequence produces one activation row per residue and no more than 64 active features per residue.
5. In GPU mode, compare activation shapes, selected-feature maxima, and top-feature IDs against the API/cache contract within documented tolerance.
6. Check panel invariants: Panel A contains 16,384 points; Panel B contains eight categories with published counts; Panel C includes the 119/99 cohort and the 75-of-99, 25-of-32, and 17-of-21 summaries; Panel D uses the stated cutoff and UMAP settings; Panel E reports 531 kinase-universal and 254 family-specific features plus the 11 named P-loop IDs; main Panel F contains the three-domain overlap counts; optional Figure S37 contains 81 proteomes and 395 selected topics; Panel G contains the three named topic classes.
7. Manually compare layout, axes, labels, legends, colors, structure overlays, and taxonomy trees with the Figure 4 asset in the local PDF.

**Scope Decisions**

- “Full faithful” means reproducing the paper’s analyses from released or independently regenerated source artifacts. It does not include retraining ESMC or the SAEs.
- No synthetic data will appear in faithful mode.
- A small-subset educational NMF refit may exist only as an explicitly labeled diagnostic.
- Feature descriptions will be labeled as model-generated biological hypotheses, not definitive annotations.
- Residue numbering and sequence-to-structure alignment will be stored explicitly for every structure overlay.

The main technical risk is source-data availability for Panel A’s full specificity/granularity coordinates, the exact Panel B and G feature/asset selections, and the main Panel F overlap counts plus optional Figure S37 topic outputs. The first implementation handoff should therefore be the artifact audit and manifest, before notebook plotting begins.
