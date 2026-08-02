# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy>=2.0",
#     "matplotlib>=3.9",
#     "requests>=2.31",
#     "py3Dmol>=2.0",
#     "esm@git+https://github.com/Biohub/esm.git@main",
# ]
# ///

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🧬 ESMC & Top-K SAE Feature Interpretation Explorer

    [ESMC](https://biohub.ai/esm/protein) is a 6B parameter protein language model trained on billions of sequence records.
    Using **Sparse Autoencoders (SAEs)**, dense hidden embeddings are expanded into 16,384 sparse, interpretable biological features.

    This notebook connects directly to the **Biohub Forge API** to:
    1. Extract per-residue embeddings and **Top-K SAE feature representations** (`esmc-6b-2024-12-sae-layer60-k64-codebook16384`).
    2. Query the **Biohub Feature Metadata API** for human-readable feature descriptions and UniRef alignments.
    3. Rank and visualize localized (motif) vs broad (domain) features along the sequence and map them onto **3D PDB structures**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    model_choice = mo.ui.dropdown(
        options=["esmc-6b-2024-12", "esmc-300m-2024-12", "esmc-600m-2024-12"],
        value="esmc-6b-2024-12",
        label="ESMC model",
    )
    sae_model_choice = mo.ui.dropdown(
        options=["esmc-6b-2024-12-sae-layer60-k64-codebook16384", "None"],
        value="esmc-6b-2024-12-sae-layer60-k64-codebook16384",
        label="SAE model (Top-K=64)",
    )
    sequence_choice = mo.ui.dropdown(
        options={
            "ATP synthase alpha subunit (PDB 2XND Chain A)": (
                "2XND_A",
                "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQ"
                "CFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGH"
                "KLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSK"
                "DPNEKRDHMVLLEFVTAAGITHGMDELYK"
            ),
            "Hen Egg White Lysozyme (PDB 1LYZ Chain A)": (
                "1LYZ_A",
                "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTP"
                "GSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
            ),
            "Villin Headpiece HP35 (fast folder)": (
                "1YRF_A",
                "LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
            ),
        },
        value="ATP synthase alpha subunit (PDB 2XND Chain A)",
        label="Example Sequence / PDB",
    )
    layer_slider = mo.ui.slider(
        start=1, stop=60, step=1, value=60,
        label="Hidden layer for embeddings (1–60)",
    )
    mo.vstack([
        mo.md("## Controls"),
        mo.hstack([model_choice, sae_model_choice], gap=2),
        mo.hstack([sequence_choice, layer_slider], gap=2),
    ])
    return layer_slider, model_choice, sae_model_choice, sequence_choice


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Biohub API Authentication & Feature Extraction")
    return


@app.cell
def _(layer_slider, model_choice, mo, sae_model_choice, sequence_choice):
    import os
    from pathlib import Path
    import numpy as np

    def _resolve_token():
        for key in ("BIOHUB_API_KEY", "BIOHUB_TOKEN"):
            v = os.environ.get(key, "")
            if v:
                return v
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith(("BIOHUB_API_KEY=", "BIOHUB_TOKEN=")):
                    return line.split("=", 1)[1].strip().strip("'\"")
        return ""

    _token = _resolve_token()
    if not _token:
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    "**No API token found.**  \n"
                    "Set `BIOHUB_API_KEY` or `BIOHUB_TOKEN` in your environment or `.env` file.  \n"
                    "Create a token at https://biohub.ai/developer-console/api-keys"
                ),
                kind="warn",
            ),
        )

    try:
        from esm.sdk.forge import ESMCForgeInferenceClient
    except ImportError:
        from esm.sdk import esmc_client as ESMCForgeInferenceClient
    from esm.sdk.api import ESMProtein, LogitsConfig, SAEConfig

    pdb_tag, _sequence = sequence_choice.value
    _model = model_choice.value
    _sae_model = sae_model_choice.value
    _layer = layer_slider.value

    _client = ESMCForgeInferenceClient(model=_model, url="https://biohub.ai", token=_token)
    _protein = ESMProtein(sequence=_sequence)
    _protein_tensor = _client.encode(_protein)

    use_sae = (_sae_model != "None")
    if use_sae:
        _config = LogitsConfig(
            sequence=True,
            return_embeddings=True,
            return_hidden_states=True,
            ith_hidden_layer=_layer,
            sae_config=SAEConfig(model=_sae_model, normalize_features=True),
        )
    else:
        _config = LogitsConfig(
            sequence=True,
            return_embeddings=True,
            return_hidden_states=True,
            ith_hidden_layer=_layer,
        )

    _output = _client.logits(_protein_tensor, _config, return_bytes=False)

    embeddings = _output.embeddings[1:-1]
    hidden_states = _output.hidden_states[1:-1] if _output.hidden_states is not None else None
    
    sae_features = None
    if use_sae and hasattr(_output, "sae_outputs") and _sae_model in _output.sae_outputs:
        sae_features = _output.sae_outputs[_sae_model].to_dense().numpy()[1:-1]

    sequence_str = _sequence
    
    status_msg = f"✅ Fetched representations for **{sequence_choice.label}** (`{len(_sequence)} aa`).  \n"
    status_msg += f"Embedding shape: `{embeddings.shape}`"
    if sae_features is not None:
        status_msg += f" · SAE Feature matrix shape: `{sae_features.shape}`"

    mo.md(status_msg)
    return (
        ESMCForgeInferenceClient,
        ESMProtein,
        LogitsConfig,
        SAEConfig,
        embeddings,
        hidden_states,
        np,
        os,
        pdb_tag,
        sae_features,
        sequence_str,
        use_sae,
    )


@app.cell(hide_code=True)
def _(mo, sae_features):
    if sae_features is None:
        mo.stop(True, mo.md("*Select an SAE model to view Top-K feature rankings and metadata.*"))
    mo.md("## Top SAE Feature Interpretation & Biohub Metadata API")
    return


@app.cell
def _(mo, np, sae_features):
    from functools import lru_cache
    import requests

    @lru_cache(maxsize=16384)
    def fetch_feature_metadata(feature_idx: int) -> dict:
        url = f"https://biohub.ai/esm/protein/api/v1alpha1/features/{feature_idx}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {"description": f"SAE Feature {feature_idx}", "top_100_uniref_ids": ["N/A"]}

    max_activations = sae_features.max(axis=0)
    prevalence = (sae_features > 0.01).sum(axis=0)

    top_by_max = np.argsort(max_activations)[::-1][:10]

    rows = []
    for rank, feat_id in enumerate(top_by_max[:5], 1):
        meta = fetch_feature_metadata(int(feat_id))
        desc = meta.get("description", "No description available")
        top_uniref = meta.get("top_100_uniref_ids", ["N/A"])[0]
        rows.append({
            "Rank": rank,
            "Feature ID": int(feat_id),
            "Max Activation": f"{max_activations[feat_id]:.3f}",
            "Prevalence (Residues)": int(prevalence[feat_id]),
            "Biohub Description": desc,
            "Top UniRef Match": top_uniref,
        })

    md_table = "### 🏆 Top 5 SAE Features by Peak Activation\n\n"
    md_table += "| Rank | Feature ID | Max Act | Prevalence | Description | Top UniRef |\n"
    md_table += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for r in rows:
        md_table += f"| {r['Rank']} | **{r['Feature ID']}** | {r['Max Activation']} | {r['Prevalence (Residues)']} | {r['Biohub Description']} | `{r['Top UniRef Match']}` |\n"

    mo.md(md_table)
    return fetch_feature_metadata, max_activations, prevalence, top_by_max


@app.cell(hide_code=True)
def _(mo, sae_features):
    if sae_features is None:
        mo.stop(True)
    mo.md("## 📊 1D Feature Activation Profiles Along Sequence")
    return


@app.cell(hide_code=True)
def _(fetch_feature_metadata, mo, np, sae_features, sequence_str, top_by_max):
    import matplotlib.pyplot as plt

    top_3 = top_by_max[:3]
    fig, axes = plt.subplots(len(top_3), 1, figsize=(12, 2.5 * len(top_3)), sharex=True)
    if len(top_3) == 1:
        axes = [axes]

    positions = np.arange(1, len(sequence_str) + 1)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, (ax, feat_id) in enumerate(zip(axes, top_3)):
        acts = sae_features[:, feat_id]
        meta = fetch_feature_metadata(int(feat_id))
        desc = meta.get("description", "")
        desc_short = (desc[:60] + "...") if len(desc) > 60 else desc

        ax.bar(positions, acts, width=1.0, color=colors[idx % len(colors)], alpha=0.75)
        ax.set_ylabel("Activation", fontsize=9)
        ax.set_title(f"Feature {feat_id}: {desc_short}", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Residue Position", fontsize=10)
    fig.tight_layout()
    mo.center(fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 🧊 3D Structural Mapping of Top SAE Feature")
    return


@app.cell(hide_code=True)
def _(fetch_feature_metadata, mo, pdb_tag, sae_features, top_by_max):
    if sae_features is None:
        mo.stop(True)

    feat_id = int(top_by_max[0])
    acts = sae_features[:, feat_id]
    max_act = max(float(acts.max()), 1e-6)
    norm_acts = (acts / max_act).tolist()

    pdb_code = pdb_tag.split("_")[0].lower()
    chain_code = pdb_tag.split("_")[1] if "_" in pdb_tag else "A"
    meta = fetch_feature_metadata(feat_id)
    desc = meta.get("description", "Feature activation")

    html_code = f"""
    <div style="width: 100%; height: 450px; position: relative; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;">
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <div id="py3dmol_viewer" style="width: 100%; height: 100%;"></div>
        <div style="position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 8px 12px; border-radius: 4px; font-family: sans-serif; font-size: 12px;">
            <b>Feature {feat_id}</b>: {desc[:70]}<br/>
            <i>PDB: {pdb_code.upper()} Chain {chain_code} (Colored white → red by activation)</i>
        </div>
    </div>
    <script>
        (function() {{
            var viewer = $3Dmol.createViewer("py3dmol_viewer", {{backgroundColor: "white"}});
            var pdbUrl = "https://files.rcsb.org/download/{pdb_code}.pdb";
            var acts = {norm_acts};
            var chain = "{chain_code}";

            jQuery.get(pdbUrl, function(data) {{
                viewer.addModel(data, "pdb");
                viewer.setStyle({{}}, {{cartoon: {{hidden: true}}}});

                for (var i = 0; i < acts.length; i++) {{
                    var val = acts[i];
                    var r = 255;
                    var g = Math.round(255 * (1 - val));
                    var b = Math.round(255 * (1 - val));
                    var hexColor = "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
                    viewer.setStyle({{chain: chain, resi: i + 1}}, {{cartoon: {{color: hexColor}}}});
                }}
                viewer.zoomTo({{chain: chain}});
                viewer.render();
            }});
        }})();
    </script>
    """
    mo.html(html_code)
    return


if __name__ == "__main__":
    app.run()
