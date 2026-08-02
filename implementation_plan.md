# Interactive SAE Feature Neighborhood Explorer — anywidget

Convert the Plotly-based `figure4_improved.py` visualization into a premium, dark-mode, canvas-rendered anywidget that can be embedded in a marimo notebook, inspired by the **Biohub ESM Atlas** (scatter map + detail panel) and **eliebak J-Space Open** (dark theme, chip controls, monospace stats, canvas rendering, hover tooltips) designs.

## Design Vision

The widget will have a **two-panel layout**:

| Left Panel (~60%) | Right Panel (~40%) |
|---|---|
| Canvas-rendered UMAP scatter plot | Feature detail card |
| Chip-style filter controls (SAE dim toggles) | Sequence bar colored by activation |
| Legend + stats bar | Top neighbors table |
| Hover tooltip over canvas | 3D structure activation (small canvas) |

### Visual Language (merging both references)
- **Background**: `#111111` dark, matching j-space
- **Text**: `#ffffff` / `#bcbcbc` for secondary
- **Fonts**: `"Space Grotesk", "Helvetica Neue", sans-serif` for headings, monospace for stats/values
- **Chip controls**: dark border (`#3a3a3a`), white when active — j-space style
- **Scatter**: Canvas 2D (not Plotly), viridis-like colorscale for cosine similarity, dot sizes by SAE dim (paper convention)
- **Tooltips**: Dark floating card on hover with feature ID, SAE dim, cosine, label — like j-space CKA tooltip
- **Detail panel**: Right side, ESM Atlas-inspired — shows selected feature label, description, sequence bar, top activating proteins, category badge
- **Activation bar**: Horizontal colored strip under sequence (like ESM Atlas feature activation colorbar)

## Proposed Changes

### Widget Python Class

#### [NEW] [figure4_widget.py](file:///c:/Users/bigcr/scratch/protein-interpretability/figure4_widget.py)

A marimo notebook containing:

1. **Data loading cells** — reuse the data-fetching logic from `figure4_improved.py` (decoder weights, UMAP projection, neighbor computation)
2. **`FeatureNeighborhoodWidget(anywidget.AnyWidget)` class** with traits:
   - `neighbor_data` (JSON string): list of `{key, feature_id, feature_index, sae_dim, cosine, category, label, summary, description, is_anchor, x, y}` — same as `neighbor_rows`
   - `anchor_label` (Unicode): the anchor feature label
   - `selected_feature` (Unicode): synced back to Python on click, so downstream cells can react
3. **Widget display cell** wrapping with `mo.ui.anywidget()`

### Widget JavaScript (external ESM file)

#### [NEW] [figure4_widget.js](file:///c:/Users/bigcr/scratch/protein-interpretability/figure4_widget.js)

Canvas-based interactive visualization:

- **`render({ model, el })`**: builds the DOM structure (two-panel container, canvas, detail panel, controls)
- **Canvas scatter**: Draws all points sized by SAE dim, colored by cosine similarity (viridis), anchor as star
- **Hover detection**: On `mousemove`, find nearest point within radius, show floating tooltip card
- **Click selection**: On `click`, update `selected_feature` trait and render detail panel
- **Chip controls**: Toggle visibility of 2^13, 2^14, 2^16 features
- **Detail panel**: Shows selected feature's metadata (label, description, category, cosine, summary)
- **Activation bar**: Renders sequence positions colored by a gradient if activation data available
- **Responsive**: Resizes canvas on container resize via `ResizeObserver`

### Widget CSS (external file)

#### [NEW] [figure4_widget.css](file:///c:/Users/bigcr/scratch/protein-interpretability/figure4_widget.css)

Dark-mode-first styling with light-mode fallback:
- Container layout (CSS Grid, two panels)
- Chip/button styles matching j-space
- Tooltip positioning and animation
- Detail panel card styling
- Typography (Google Fonts: Space Grotesk)
- Scrollable feature list
- Activation bar styling
- Color legend strip

## Open Questions

> [!IMPORTANT]
> **Data scope**: The existing notebook loads ~80-170 MB of decoder weights to compute the UMAP. Should the widget:
> 1. **(Recommended)** Accept pre-computed `neighbor_rows` data from the existing notebook cells (widget is purely a visualization layer) — this is what I plan to do
> 2. Include data fetching inside the widget itself (heavy, defeats purpose of notebook integration)

> [!NOTE]
> **3D structure panel**: The ESM Atlas reference shows a 3D rotating protein. Implementing a full WebGL structure viewer inside the widget would be substantial. My plan is to:
> - Show a simplified 2D activation bar (sequence colored by activation) in the detail panel
> - Let the 3D structure remain as a separate Plotly cell in the notebook, driven by the `selected_feature` trait
> - This keeps the widget focused and performant

## Verification Plan

### Manual Verification
1. Run `uvx marimo check figure4_widget.py` to validate notebook structure
2. Run `uv run marimo edit figure4_widget.py` and confirm:
   - Dark-mode canvas scatter renders with correct dot sizes
   - Hover tooltips appear with feature details
   - Clicking a dot updates the detail panel and syncs `selected_feature` back to Python
   - SAE dim chip toggles filter points
   - Detail panel shows correct metadata for selected feature
