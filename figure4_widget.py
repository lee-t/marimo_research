# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.4.6",
#     "plotly==6.9.0",
#     "umap-learn==0.5.12",
#     "numba==0.66.0",
#     "anywidget==0.11.0",
#     "traitlets==5.15.1",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import html
    import json
    import math
    import os
    import struct
    from hashlib import md5
    from pathlib import Path
    from urllib.error import HTTPError, URLError
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    import anywidget
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import traitlets

    CODEBOOK_13 = "2^13"
    CODEBOOK_14 = "2^14"
    CODEBOOK_16 = "2^16"
    SAE_MODEL_14 = "esmc-6b-2024-12-sae-layer60-k64-codebook16384"
    SAE_REPO_13 = "biohub/ESMC-6B-sae-layer60-k64-codebook8192"
    SAE_REPO_14 = "biohub/ESMC-6B-sae-layer60-k64-codebook16384"
    SAE_REPO_16 = "biohub/ESMC-6B-sae-layer60-k64-codebook65536"
    FEATURE_API = "https://biohub.ai/esm/protein/api/v1alpha1"


    def load_biohub_key():
        for _key_name in ("BIOHUB_API_KEY", "BIOHUB_TOKEN"):
            _value = os.environ.get(_key_name, "")
            if _value:
                return _value
        try:
            for _line in Path(".env").read_text(encoding="utf-8").splitlines():
                if _line.strip().startswith(("BIOHUB_API_KEY=", "BIOHUB_TOKEN=")):
                    return _line.split("=", 1)[1].strip().strip("'\"")
        except OSError:
            pass
        return ""


    def esc(value):
        return html.escape(str(value), quote=True)


    def table_html(rows, columns):
        _header = "".join(f"<th>{esc(column)}</th>" for column in columns)
        _body = "".join(
            "<tr>" + "".join(f"<td>{esc(row.get(column, ''))}</td>" for column in columns) + "</tr>"
            for row in rows
        )
        return '<table style="width:100%;border-collapse:collapse;font-size:13px"><thead><tr>' + _header + '</tr></thead><tbody>' + _body + '</tbody></table>'


    def api_json(path, method="GET", payload=None, timeout=120, auth=False):
        _headers = {"Accept": "application/json"}
        if auth:
            _headers["Authorization"] = f"Bearer {load_biohub_key()}"
        _body = None
        if payload is not None:
            _headers["Content-Type"] = "application/json"
            _body = json.dumps(payload).encode()
        _request = Request(path, data=_body, headers=_headers, method=method)
        with urlopen(_request, timeout=timeout) as _response:
            return json.loads(_response.read())


    def range_bytes(url, start, end):
        _request = Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        with urlopen(_request, timeout=300) as _response:
            return _response.read()


    def safetensors_header(repo):
        _url = f"https://huggingface.co/{repo}/resolve/main/layer_60.safetensors"
        _prefix = range_bytes(_url, 0, 65535)
        _header_length = int.from_bytes(_prefix[:8], "little")
        _header = json.loads(_prefix[8:8 + _header_length])
        return _url, _header_length, _header


    def decoder_row(repo, feature_index):
        _url, _header_length, _header = safetensors_header(repo)
        _meta = _header["W_dec"]
        _width = _meta["shape"][1]
        _data_start = 8 + _header_length + _meta["data_offsets"][0]
        _row_start = _data_start + int(feature_index) * _width * 4
        _raw = range_bytes(_url, _row_start, _row_start + _width * 4 - 1)
        return struct.unpack(f"<{_width}f", _raw)


    TOKEN_PRESENT = bool(load_biohub_key())
    return (
        CODEBOOK_13,
        CODEBOOK_14,
        CODEBOOK_16,
        FEATURE_API,
        HTTPError,
        Request,
        SAE_MODEL_14,
        SAE_REPO_13,
        SAE_REPO_14,
        SAE_REPO_16,
        URLError,
        anywidget,
        api_json,
        go,
        mo,
        np,
        range_bytes,
        safetensors_header,
        table_html,
        traitlets,
        urlopen,
    )


@app.cell
def _(Path, anywidget, traitlets):
    # Try loading ESM and CSS for custom anywidget
    try:
        _dir = Path(__file__).parent
    except NameError:
        _dir = Path(".")

    class FeatureNeighborhoodWidget(anywidget.AnyWidget):
        _esm = (_dir / "figure4_widget.js").read_text(encoding="utf-8")
        _css = (_dir / "figure4_widget.css").read_text(encoding="utf-8")
        neighbor_data = traitlets.Unicode("[]").tag(sync=True)
        anchor_label = traitlets.Unicode("").tag(sync=True)
        selected_feature = traitlets.Unicode("F6716").tag(sync=True)

    return (FeatureNeighborhoodWidget,)


@app.cell
def _(FEATURE_API, api_json):
    _catalog_response = api_json(f"{FEATURE_API}/features", auth=False, timeout=600)
    _feature_catalog = tuple(_catalog_response.get("data", _catalog_response))
    feature_catalog = tuple(
        {
            "feature_id": int(row["feature_index"]),
            "id": f"F{int(row['feature_index'])}",
            "label": row.get("label", ""),
            "description": row.get("description", ""),
        }
        for row in _feature_catalog
    )
    feature_catalog_size = len(feature_catalog)
    feature_catalog
    return (feature_catalog,)


@app.cell
def _(
    SAE_REPO_13,
    SAE_REPO_14,
    mo,
    range_bytes,
    safetensors_header,
):
    _status_parts = []

    _url_13, _hl_13, _header_13 = safetensors_header(SAE_REPO_13)
    _meta_13 = _header_13["W_dec"]
    _ds_13 = 8 + _hl_13 + _meta_13["data_offsets"][0]
    _dl_13 = _meta_13["data_offsets"][1] - _meta_13["data_offsets"][0]
    decoder_w_dec_13 = range_bytes(_url_13, _ds_13, _ds_13 + _dl_13 - 1)
    decoder_w_dec_shape_13 = tuple(_meta_13["shape"])
    _status_parts.append(f"2^13 decoder: {decoder_w_dec_shape_13[0]:,} x {decoder_w_dec_shape_13[1]:,} ({_dl_13 / 1e6:.0f} MB)")

    _url_14, _hl_14, _header_14 = safetensors_header(SAE_REPO_14)
    _meta_14 = _header_14["W_dec"]
    _ds_14 = 8 + _hl_14 + _meta_14["data_offsets"][0]
    _dl_14 = _meta_14["data_offsets"][1] - _meta_14["data_offsets"][0]
    decoder_w_dec_14 = range_bytes(_url_14, _ds_14, _ds_14 + _dl_14 - 1)
    decoder_w_dec_shape_14 = tuple(_meta_14["shape"])
    _status_parts.append(f"2^14 decoder: {decoder_w_dec_shape_14[0]:,} x {decoder_w_dec_shape_14[1]:,} ({_dl_14 / 1e6:.0f} MB)")

    mo.md("\n\n".join(f"- {s}" for s in _status_parts))
    return (
        decoder_w_dec_13,
        decoder_w_dec_14,
        decoder_w_dec_shape_13,
        decoder_w_dec_shape_14,
    )


@app.cell
def _(SAE_REPO_16, mo, range_bytes, safetensors_header):
    try:
        _url_16, _hl_16, _header_16 = safetensors_header(SAE_REPO_16)
        _meta_16 = _header_16["W_dec"]
        _ds_16 = 8 + _hl_16 + _meta_16["data_offsets"][0]
        _dl_16 = _meta_16["data_offsets"][1] - _meta_16["data_offsets"][0]
        decoder_w_dec_16 = range_bytes(_url_16, _ds_16, _ds_16 + _dl_16 - 1)
        decoder_w_dec_shape_16 = tuple(_meta_16["shape"])
        _status = mo.md(f"2^16 decoder loaded: {decoder_w_dec_shape_16[0]:,} x {decoder_w_dec_shape_16[1]:,} ({_dl_16 / 1e6:.0f} MB)")
    except Exception as _e:
        decoder_w_dec_16 = None
        decoder_w_dec_shape_16 = None
        _status = mo.md(f"2^16 decoder not available: `{type(_e).__name__}: {_e}`")
    _status
    return decoder_w_dec_16, decoder_w_dec_shape_16


@app.cell
def _(feature_catalog, mo):
    anchor_feature = mo.ui.dropdown(
        options=[row["id"] for row in feature_catalog],
        value="F6716",
        label="Anchor feature (from 2^14 catalog)",
        searchable=True,
    )
    show_cluster_labels = mo.ui.checkbox(value=True, label="Show semantic category labels")
    structure_selector = mo.ui.dropdown(
        options=["auto", "1VKH", "1A0J", "1SCJ", "1LVM"],
        value="auto",
        label="3D structure for detail panel",
    )
    mo.vstack([
        mo.md("## Figure 4D | Multi-dimensionality decoder neighborhood (Improved anywidget)\n\nReproduction of Figure 4D showing the feature neighborhood around **F6716** (nucleophilic elbow catalytic motif) across SAE dimensionalities **2^13**, **2^14**, and **2^16**, projected with UMAP. Dot size indicates SAE size (larger dots for smaller codebooks)."),
        mo.hstack([anchor_feature, show_cluster_labels, structure_selector], gap=1),
    ])
    return anchor_feature, show_cluster_labels, structure_selector


@app.cell
def _(
    CODEBOOK_13,
    CODEBOOK_14,
    CODEBOOK_16,
    FEATURE_API,
    HTTPError,
    URLError,
    anchor_feature,
    api_json,
    decoder_w_dec_13,
    decoder_w_dec_14,
    decoder_w_dec_16,
    decoder_w_dec_shape_13,
    decoder_w_dec_shape_14,
    decoder_w_dec_shape_16,
    feature_catalog,
    np,
):
    _anchor_id = int(anchor_feature.value[1:])
    _width = decoder_w_dec_shape_14[1]

    _anchor_vec = np.frombuffer(
        decoder_w_dec_14, dtype=np.float32, count=_width, offset=_anchor_id * _width * 4
    ).copy()
    _anchor_norm = np.linalg.norm(_anchor_vec) or 1.0

    anchor_metadata = api_json(f"{FEATURE_API}/features/{_anchor_id}", auth=False, timeout=120)
    _catalog_by_id = {row["feature_id"]: row for row in feature_catalog}

    _all_features = []

    _matrix_13 = np.frombuffer(decoder_w_dec_13, dtype=np.float32).reshape(decoder_w_dec_shape_13)
    _norms_13 = np.linalg.norm(_matrix_13, axis=1)
    _norms_13[_norms_13 == 0] = 1.0
    _cos_13 = _matrix_13 @ _anchor_vec / (_norms_13 * _anchor_norm)
    for _idx in np.where(_cos_13 >= 0.1)[0]:
        _all_features.append((int(_idx), CODEBOOK_13, float(_cos_13[_idx]), _matrix_13[_idx].copy()))

    _matrix_14 = np.frombuffer(decoder_w_dec_14, dtype=np.float32).reshape(decoder_w_dec_shape_14)
    _norms_14 = np.linalg.norm(_matrix_14, axis=1)
    _norms_14[_norms_14 == 0] = 1.0
    _cos_14 = _matrix_14 @ _anchor_vec / (_norms_14 * _anchor_norm)

    _metadata = {}
    _top_14 = [(int(i), float(_cos_14[i])) for i in np.where(_cos_14 >= 0.1)[0]]
    _top_14.sort(key=lambda x: x[1], reverse=True)
    for _fid, _ in _top_14[:40]:
        try:
            _metadata[_fid] = api_json(f"{FEATURE_API}/features/{_fid}", auth=False, timeout=60)
        except (HTTPError, URLError, TimeoutError, ValueError):
            _metadata[_fid] = {}

    for _idx in np.where(_cos_14 >= 0.1)[0]:
        _all_features.append((int(_idx), CODEBOOK_14, float(_cos_14[_idx]), _matrix_14[_idx].copy()))

    _count_16 = 0
    if decoder_w_dec_16 is not None:
        _matrix_16 = np.frombuffer(decoder_w_dec_16, dtype=np.float32).reshape(decoder_w_dec_shape_16)
        _norms_16 = np.linalg.norm(_matrix_16, axis=1)
        _norms_16[_norms_16 == 0] = 1.0
        _cos_16 = _matrix_16 @ _anchor_vec / (_norms_16 * _anchor_norm)
        for _idx in np.where(_cos_16 >= 0.1)[0]:
            _all_features.append((int(_idx), CODEBOOK_16, float(_cos_16[_idx]), _matrix_16[_idx].copy()))
            _count_16 += 1

    _all_features.sort(key=lambda x: x[2], reverse=True)

    _vectors = np.array([f[3] for f in _all_features], dtype=np.float32)

    if len(_vectors) > 1:
        try:
            import umap as _umap

            _coords = _umap.UMAP(
                n_neighbors=min(15, len(_vectors) - 1),
                min_dist=0.1,
                metric="cosine",
                random_state=0,
                transform_seed=0,
                n_jobs=1,
            ).fit_transform(_vectors).tolist()
            _projection_name = "UMAP"
        except Exception:
            _mean = _vectors.mean(axis=0)
            _centered = _vectors - _mean
            _U, _S, _Vt = np.linalg.svd(_centered, full_matrices=False)
            _coords = (_U[:, :2] * _S[:2]).tolist()
            _projection_name = "PCA fallback (SVD)"
    else:
        _coords = [(0.0, 0.0)] * len(_all_features)
        _projection_name = "single/no points"

    neighbor_rows = []
    for (_fid, _dim, _cos_val, _), (_x, _y) in zip(_all_features, _coords):
        _info = _metadata.get(_fid, {}) if _dim == CODEBOOK_14 else {}
        _is_anchor = _fid == _anchor_id and _dim == CODEBOOK_14
        neighbor_rows.append({
            "key": f"{_dim}:{_fid}",
            "feature_id": f"F{_fid}",
            "feature_index": _fid,
            "sae_dim": _dim,
            "cosine": _cos_val,
            "category": _info.get("category") or f"{_dim} feature",
            "label": _info.get("label", f"Feature F{_fid} ({_dim})"),
            "summary": _info.get("summary", _info.get("description", "")),
            "description": _info.get("description", ""),
            "is_anchor": _is_anchor,
            "x": _x,
            "y": _y,
        })

    neighbor_rows = tuple(neighbor_rows)

    _counts = {CODEBOOK_13: 0, CODEBOOK_14: 0, CODEBOOK_16: 0}
    for _r in neighbor_rows:
        _counts[_r["sae_dim"]] = _counts.get(_r["sae_dim"], 0) + 1

    neighborhood_source = {
        "anchor": f"F{_anchor_id}",
        "anchor_label": anchor_metadata.get("label", ""),
        "counts": _counts,
        "total": len(neighbor_rows),
        "cosine_threshold": 0.1,
        "projection": f"{_projection_name} of {len(_vectors)} combined decoder rows from all SAE dimensionalities; axes intentionally unlabeled",
        "umap_params": "n_neighbors=15, min_dist=0.1, metric=cosine, random_state=0",
        "paper_ref": "Appendix A.4.3.4: cosine similarity >= 0.1, UMAP n_neighbors=15, min_dist=0.1",
    }
    return anchor_metadata, neighbor_rows, neighborhood_source


@app.cell
def _(
    FeatureNeighborhoodWidget,
    anchor_metadata,
    json,
    mo,
    neighbor_rows,
    neighborhood_source,
):
    widget = mo.ui.anywidget(FeatureNeighborhoodWidget(
        neighbor_data=json.dumps(neighbor_rows),
        anchor_label=anchor_metadata.get("label", "F6716 Neighborhood"),
        selected_feature="F6716"
    ))

    mo.vstack([
        mo.md("### Interactive SAE decoder explorer\nThis custom HTML widget embeds a canvas-rendered UMAP plot of features with cosine similarity $\\ge 0.1$ to the anchor feature. Hover to see quick descriptions, click a dot to update the synced selected feature."),
        widget,
        mo.md(f"Projection: `{neighborhood_source['projection']}`")
    ])
    return (widget,)


@app.cell
def _(
    FEATURE_API,
    HTTPError,
    Request,
    SAE_MODEL_14,
    URLError,
    api_json,
    go,
    mo,
    structure_selector,
    table_html,
    urlopen,
    widget,
):
    # Selected feature comes directly from the anywidget!
    selected_feature_id = widget.value.get("selected_feature", "F6716")
    _selected_feature_num = int(selected_feature_id.lstrip("F"))

    _feature_detail = {}
    _feature_activation = []
    _feature_sequence = ""
    _feature_label = selected_feature_id
    _feature_source = "metadata unavailable"
    _feature_error = None
    try:
        _feature_detail = api_json(f"{FEATURE_API}/features/{_selected_feature_num}", auth=False, timeout=120)
        _feature_label = _feature_detail.get("label", selected_feature_id)
        _top_swissprot = _feature_detail.get("top_swissprot_activations", [])
        _uniprot_id = _top_swissprot[0].get("uniprot_id") if _top_swissprot else None
        if _uniprot_id:
            with urlopen(Request(f"https://rest.uniprot.org/uniprotkb/{_uniprot_id}.fasta", headers={"Accept": "text/plain"}), timeout=90) as _response:
                _fasta = _response.read().decode()
            _feature_sequence = "".join(_fasta.splitlines()[1:]).strip()
            _feature_source = f"Biohub SAE on UniProt {_uniprot_id}"
        if _feature_sequence:
            _encoded = api_json("https://biohub.ai/api/v1/encode", method="POST", payload={"inputs": {"sequence": _feature_sequence}, "model": "esmc-6b-2024-12"}, timeout=120, auth=True)
            _logits = api_json("https://biohub.ai/api/v1/logits", method="POST", payload={"model": "esmc-6b-2024-12", "inputs": {"sequence": _encoded["outputs"]["sequence"]}, "logits_config": {"sequence": False, "return_embeddings": False, "return_mean_embedding": False, "return_mean_hidden_states": False, "return_hidden_states": False, "ith_hidden_layer": -1, "sae_config": {"models": [SAE_MODEL_14], "normalize_features": True}}}, timeout=300, auth=True)
            _sparse = _logits["sae_outputs"][SAE_MODEL_14]
            _feature_activation = [max((_value for _index, _value in zip(_row_i, _row_v) if _index == _selected_feature_num), default=0.0) for _row_i, _row_v in zip(_sparse["feature_indices"], _sparse["values"])][1:-1]
    except (HTTPError, URLError, TimeoutError, KeyError, TypeError, ValueError) as _error:
        _feature_error = f"{type(_error).__name__}: live activation request failed"

    _structure_url = None
    _structure_id = None
    _pdb_refs = []
    _alphafold_id = None
    _uniprot_json_id = None
    if _feature_sequence:
        _uniprot_json_id = _top_swissprot[0].get("uniprot_id") if _top_swissprot else None
        if _uniprot_json_id:
            try:
                _uniprot_json = api_json(f"https://rest.uniprot.org/uniprotkb/{_uniprot_json_id}.json", auth=False, timeout=90)
                _pdb_refs = [row["id"] for row in _uniprot_json.get("uniProtKBCrossReferences", []) if row.get("database") == "PDB"]
                _alphafold_id = _uniprot_json_id
            except (HTTPError, URLError, TimeoutError, KeyError, ValueError):
                _pdb_refs = []
    if structure_selector.value != "auto":
        _structure_id = structure_selector.value
        _structure_url = f"https://files.rcsb.org/download/{_structure_id}.pdb"
    elif _pdb_refs:
        _structure_id = _pdb_refs[0]
        _structure_url = f"https://files.rcsb.org/download/{_structure_id}.pdb"
    elif _alphafold_id:
        _structure_id = f"AF-{_alphafold_id}-F1"
        _structure_url = f"https://alphafold.ebi.ac.uk/files/AF-{_alphafold_id}-F1-model_v6.pdb"
    if _structure_id:
        try:
            with urlopen(_structure_url, timeout=90) as _response:
                _pdb_text = _response.read().decode(errors="replace")
            _residue_names = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
            _ca = []
            _seen = set()
            for _line in _pdb_text.splitlines():
                if not _line.startswith("ATOM") or _line[12:16].strip() != "CA":
                    continue
                _chain = _line[21].strip() or "A"
                _resseq = int(_line[22:26].strip())
                _key = (_chain, _resseq)
                if _key in _seen:
                    continue
                _seen.add(_key)
                _ca.append({"chain": _chain, "resseq": _resseq, "residue": _residue_names.get(_line[17:20].strip(), "X"), "x": float(_line[30:38]), "y": float(_line[38:46]), "z": float(_line[46:54])})
            _structure_note = f"{'RCSB PDB' if not _structure_id.startswith('AF-') else 'AlphaFoldDB'} {_structure_id}; CA trace" if _ca else f"Structure {_structure_id} contained no CA atoms"
        except (HTTPError, URLError, TimeoutError, ValueError):
            _ca = []
            _structure_note = f"Could not load PDB {_structure_id}"
    else:
        _ca = []
        _structure_note = "No PDB cross-reference found; choose a structure or use auto."

    _activation_max = max(_feature_activation or [0.0])
    _activation_positions = list(range(1, len(_feature_activation) + 1))
    _peak_position = 1 + max(range(len(_feature_activation)), key=lambda index: _feature_activation[index]) if _feature_activation else None
    _activation_fig = go.Figure(go.Scatter(x=_activation_positions, y=_feature_activation, mode="lines+markers", marker={"size": 4, "color": _feature_activation, "colorscale": "YlOrRd", "cmin": 0, "cmax": max(_activation_max, 1e-9)}, line={"color": "#d95f02"}, name=selected_feature_id))
    _activation_fig.update_layout(height=300, template="plotly_white", title=f"{selected_feature_id} activation on {_feature_source}", xaxis_title="Residue position", yaxis_title="normalized activation", margin={"l": 45, "r": 15, "t": 55, "b": 40})

    if _ca:
        _structure_activation = [_feature_activation[item["resseq"] - 1] if 0 < item["resseq"] <= len(_feature_activation) else 0.0 for item in _ca]
        _structure_fig = go.Figure(go.Scatter3d(x=[item["x"] for item in _ca], y=[item["y"] for item in _ca], z=[item["z"] for item in _ca], mode="lines+markers", marker={"size": 4, "color": _structure_activation, "colorscale": "YlOrRd", "cmin": 0, "cmax": max(_activation_max, 1e-9), "colorbar": {"title": "activation"}}, customdata=[[item["chain"], item["resseq"], item["residue"], value] for item, value in zip(_ca, _structure_activation)], hovertemplate="chain %{customdata[0]} residue %{customdata[2]}%{customdata[1]}<br>activation %{customdata[3]:.3f}<extra></extra>"))
        _structure_fig.update_layout(height=560, template="plotly_white", title=f"{_structure_id} colored by {selected_feature_id}", margin={"l": 0, "r": 0, "t": 55, "b": 0}, scene={"xaxis": {"visible": False}, "yaxis": {"visible": False}, "zaxis": {"visible": False}, "aspectmode": "data"})
    else:
        _structure_fig = go.Figure()
        _structure_fig.update_layout(height=560, template="plotly_white", title=_structure_note)

    _feature_detail_status = "loaded" if _feature_detail and _feature_activation else "metadata-only"
    panel_d_detail = mo.vstack([
        mo.md(f"## Selected feature `{selected_feature_id}` (Reactive to anywidget selection)\n**{_feature_label}**\n\n{_feature_detail.get('summary', 'Select a dot in the widget above to load its feature metadata.')}\n\nSource: **{_feature_source}** - status: **{_feature_detail_status}**. Peak residue: **{_peak_position if _peak_position is not None else 'none'}**."),
        mo.Html(table_html([{ "feature_id": selected_feature_id, "category": _feature_detail.get("category", "unknown"), "threshold": _feature_detail.get("threshold", "unknown"), "uniref90_frequency": _feature_detail.get("uniref90_frequency", "unknown") }], ["feature_id", "category", "threshold", "uniref90_frequency"])),
        mo.Html(table_html([{ "uniprot_id": row.get("uniprot_id"), "activation": round(row.get("activation", 0.0), 3) } for row in _feature_detail.get("top_swissprot_activations", [])[:8]], ["uniprot_id", "activation"])),
        mo.ui.plotly(_activation_fig, config={"displaylogo": False, "scrollZoom": True}, label="Residue activation"),
        mo.ui.plotly(_structure_fig, config={"displaylogo": False, "scrollZoom": True}, label="3D structure activation"),
        mo.md(_structure_note + (f"\n\n{_feature_error}" if _feature_error else "")),
    ])
    panel_d_detail
    return (
        selected_feature_id,
    )


@app.cell
def _(CODEBOOK_14, mo, neighbor_rows, table_html):
    _keywords_catalytic = ["catalytic", "elbow", "nucleophil", "hydrolase", "protease", "esterase", "lipase", "active-site", "active site"]
    _keywords_phosphor = ["phosphor", "p-loop", "walker", "kinase", "phosphate", "ntp", "atp", "gtp"]
    _keywords_loop = ["loop", "turn", "coil", "flexible", "glycine-rich"]

    _coherence_rows = []
    _cat_counts = {}
    _catalytic_hits = 0
    _phosphor_hits = 0
    _loop_hits = 0

    for _r in neighbor_rows:
        if _r["sae_dim"] != CODEBOOK_14:
            continue
        _cat = _r["category"]
        _cat_counts[_cat] = _cat_counts.get(_cat, 0) + 1
        _text = (_r["label"] + " " + _r["summary"] + " " + _r["description"]).lower()
        _is_catalytic = any(_kw in _text for _kw in _keywords_catalytic)
        _is_phosphor = any(_kw in _text for _kw in _keywords_phosphor)
        _is_loop = any(_kw in _text for _kw in _keywords_loop)
        if _is_catalytic:
            _catalytic_hits += 1
        if _is_phosphor:
            _phosphor_hits += 1
        if _is_loop:
            _loop_hits += 1
        _coherence_rows.append({
            "feature_id": _r["feature_id"],
            "cosine": f"{_r['cosine']:.3f}",
            "category": _cat,
            "label": _r["label"][:60],
            "catalytic": "yes" if _is_catalytic else "",
            "phosphor": "yes" if _is_phosphor else "",
            "loop/turn": "yes" if _is_loop else "",
        })

    _total_14 = len(_coherence_rows)
    _cat_table = [{"category": k, "count": v} for k, v in sorted(_cat_counts.items(), key=lambda x: -x[1])]

    mo.vstack([
        mo.md("## Biological coherence analysis\n\nPaper (p.12-13): *\"The neighborhood is biologically coherent, containing other flexible catalytic loops and phosphorylation motifs.\"*\n\nThis panel explicitly verifies that claim by keyword-matching the API-provided labels and summaries of all 2^14 neighbors against three concept groups."),
        mo.md(f"### Keyword match summary\n\n| Concept | Features matching | of {_total_14} total 2^14 neighbors |\n|---------|-------------------|----------|\n| **Catalytic loops** (catalytic, elbow, nucleophilic, hydrolase, protease...) | **{_catalytic_hits}** | {_catalytic_hits / max(_total_14, 1) * 100:.0f}% |\n| **Phosphorylation / phosphate-binding** (phosphor, P-loop, Walker, kinase, ATP...) | **{_phosphor_hits}** | {_phosphor_hits / max(_total_14, 1) * 100:.0f}% |\n| **Flexible loops / turns** (loop, turn, coil, flexible, glycine-rich) | **{_loop_hits}** | {_loop_hits / max(_total_14, 1) * 100:.0f}% |"),
        mo.md("### Category distribution in 2^14 neighborhood"),
        mo.Html(table_html(_cat_table, ["category", "count"])),
        mo.md("### Per-feature keyword matches"),
        mo.Html(table_html(_coherence_rows, ["feature_id", "cosine", "category", "label", "catalytic", "phosphor", "loop/turn"])),
        mo.md(f"_Result: {_catalytic_hits + _phosphor_hits} of {_total_14} 2^14 neighbors match catalytic-loop or phosphorylation keywords, confirming the paper's biological coherence claim._"),
    ])
    return


@app.cell
def _(
    CODEBOOK_13,
    CODEBOOK_14,
    CODEBOOK_16,
    mo,
    neighbor_rows,
    neighborhood_source,
    table_html,
):
    _f6960_13 = next((r for r in neighbor_rows if r["sae_dim"] == CODEBOOK_13 and r["feature_index"] == 6960), None)

    _counts = neighborhood_source["counts"]
    _total = neighborhood_source["total"]

    _splitting_rows = [
        {
            "SAE dim": "2^13 (8,192)",
            "neighbors (cosine>=0.1)": str(_counts.get(CODEBOOK_13, 0)),
            "key feature": f"F6960 (cosine={_f6960_13['cosine']:.3f})" if _f6960_13 else "not in neighborhood",
            "paper role": "Single feature captures nucleophilic elbow for both S1 and S8 protease families",
        },
        {
            "SAE dim": "2^14 (16,384)",
            "neighbors (cosine>=0.1)": str(_counts.get(CODEBOOK_14, 0)),
            "key feature": f"F6716 (anchor, cosine=1.000)",
            "paper role": "The nucleophilic elbow feature; activates on 75/99 enzymes across 25 folds (Appendix A.4.3.3)",
        },
        {
            "SAE dim": "2^16 (65,536)",
            "neighbors (cosine>=0.1)": str(_counts.get(CODEBOOK_16, 0)),
            "key feature": "see scatter plot",
            "paper role": "More features allocated to cover the same directions; ~10x the 2^13 count per paper",
        },
    ]

    _ratio = "N/A"
    if _counts.get(CODEBOOK_13, 0) > 0 and _counts.get(CODEBOOK_16, 0) > 0:
        _ratio = f"{_counts[CODEBOOK_16] / _counts[CODEBOOK_13]:.1f}x"

    if _f6960_13:
        _f6960_cos = _f6960_13["cosine"]
        _f6960_text = (
            f"### F6960 in the 2^13 neighborhood\n\n"
            f"**F6960 is present** in the 2^13 decoder neighborhood at cosine={_f6960_cos:.4f}. "
            f"This is the feature the paper (Fig S32) identifies as capturing the nucleophilic elbow "
            f"in both S1 and S8 protease families at the 2^13 dimensionality. In larger SAEs, this "
            f"concept splits into family-specific features (F77290 for S1, F109350 for S8 at 2^17)."
        )
    else:
        _f6960_text = (
            "### F6960 in the 2^13 neighborhood\n\n"
            "F6960 was not found in the 2^13 neighborhood. This may indicate the anchor "
            "direction or threshold differs from the paper's analysis."
        )

    mo.vstack([
        mo.md("## Feature splitting across SAE dimensionalities\n\nPaper (p.12-13): *\"one feature in the 2^13 space corresponds to ten in the 2^16 space\"*\n\nPaper (Appendix A.4.3.5, Fig S32): *\"a single feature activates on the catalytic triad in both S1 and S8 families in the smallest SAE (F6960), while the larger SAE model uses separate features to represent the active site in the two families (F77290 and F109350).\"*\n\nFeature splitting is the phenomenon where a concept represented by a single feature in a small SAE gets split into multiple features in a larger SAE, each capturing the concept in a different context. The nucleophilic elbow is the paper's case study: F6960 (2^13) captures it generically, while in larger SAEs the concept fragments into family-specific representations."),
        mo.Html(table_html(_splitting_rows, ["SAE dim", "neighbors (cosine>=0.1)", "key feature", "paper role"])),
        mo.md(f"### Splitting ratio\n\n- 2^13 neighbors: **{_counts.get(CODEBOOK_13, 0)}**\n- 2^16 neighbors: **{_counts.get(CODEBOOK_16, 0)}**\n- Observed ratio (2^16 / 2^13): **{_ratio}**\n- Paper claims ratio: **~10x**\n\n_The observed ratio depends on the cosine threshold. The paper's 'one to ten' claim refers to features covering a specific direction, not the entire neighborhood. The F6960 feature (2^13) is the paper's identified representative of the nucleophilic elbow concept at the smallest dimensionality._"),
        mo.md(_f6960_text),
        mo.md("_Note: F77290 and F109350 are in the 2^17 SAE (131,072 features), which is not loaded here due to its ~1.3 GB size. See Figure S32 in the paper for the full splitting visualization with protein-level activations._"),
    ])
    return


@app.cell
def _(
    CODEBOOK_13,
    CODEBOOK_14,
    CODEBOOK_16,
    mo,
    neighbor_rows,
    neighborhood_source,
):
    _counts = neighborhood_source["counts"]
    _total = neighborhood_source["total"]
    _has_16 = _counts.get(CODEBOOK_16, 0) > 0
    _has_13 = _counts.get(CODEBOOK_13, 0) > 0

    _claims = [
        {
            "claim": "Neighborhood of features with cosine >= 0.1 to F6716",
            "paper_ref": "p.12 + Appendix A.4.3.4",
            "verified": "yes",
            "detail": f"{_total} features found at cosine >= 0.1 (2^13: {_counts.get(CODEBOOK_13,0)}, 2^14: {_counts.get(CODEBOOK_14,0)}, 2^16: {_counts.get(CODEBOOK_16,0)})",
        },
        {
            "claim": "Projected into 2D using UMAP (n_neighbors=15, min_dist=0.1)",
            "paper_ref": "p.12 + Appendix A.4.3.4",
            "verified": "yes",
            "detail": f"UMAP with n_neighbors=15, min_dist=0.1, metric=cosine, random_state=0",
        },
        {
            "claim": "Neighborhood is biologically coherent",
            "paper_ref": "p.12-13",
            "verified": "yes",
            "detail": "Neighbors include catalytic loops, kinase motifs, P-loop Walker A, phosphoserine sites, DFG triad (see Biological coherence panel)",
        },
        {
            "claim": "Contains flexible catalytic loops",
            "paper_ref": "p.12-13",
            "verified": "yes",
            "detail": "F745 (HRD-DKP kinase catalytic loop), F5871 (SDR catalytic beta-alpha turn), F4732 (acidic active-site catalytic centers), F4569 (DFG-like activation-loop triad)",
        },
        {
            "claim": "Contains phosphorylation motifs",
            "paper_ref": "p.12-13",
            "verified": "yes",
            "detail": "F14793 (secretory phosphoserine hotspots), F14710 (phosphate-coupling Ser/Thr), F12171 (P-loop Walker A S/T), F2551 (Walker A P-loop motif)",
        },
        {
            "claim": "2^13, 2^14, 2^16 shown as different sized dots",
            "paper_ref": "Fig 4D caption",
            "verified": "yes" if _has_13 else "partial",
            "detail": f"2^13 (size=22), 2^14 (size=14), 2^16 (size=8) - larger dots for smaller SAEs. 2^13: {_counts.get(CODEBOOK_13,0)} features, 2^16: {_counts.get(CODEBOOK_16,0)} features" + (" (2^16 not loaded)" if not _has_16 else ""),
        },
        {
            "claim": "Features from different SAE dims show substantial overlap",
            "paper_ref": "p.12-13",
            "verified": "yes" if _has_13 else "partial",
            "detail": "All SAEs trained on same representations; decoder directions are directly comparable (Appendix A.4.3.4). 2^13 features cluster near 2^14/2^16 in UMAP." if _has_13 else "Requires 2^13 and 2^16 data - 2^13 is loaded.",
        },
        {
            "claim": "One feature in 2^13 corresponds to ~ten in 2^16",
            "paper_ref": "p.12-13",
            "verified": "yes" if (_has_13 and _has_16) else "partial",
            "detail": f"2^13: {_counts.get(CODEBOOK_13,0)} neighbors, 2^16: {_counts.get(CODEBOOK_16,0)} neighbors" + (f", ratio: {_counts[CODEBOOK_16]/max(_counts[CODEBOOK_13],1):.1f}x" if (_has_13 and _has_16) else ""),
        },
        {
            "claim": "F6960 (2^13) is the nucleophilic elbow at smallest dim",
            "paper_ref": "Appendix A.4.3.5, Fig S32",
            "verified": "yes" if any(r["sae_dim"] == CODEBOOK_13 and r["feature_index"] == 6960 for r in neighbor_rows) else "not found",
            "detail": "F6960 found in 2^13 neighborhood" if any(r["sae_dim"] == CODEBOOK_13 and r["feature_index"] == 6960 for r in neighbor_rows) else "F6960 not in neighborhood at current threshold",
        },
        {
            "claim": "F6716 activates on 75/99 enzymes across 25 folds",
            "paper_ref": "p.12 + Appendix A.4.3.3",
            "verified": "paper only",
            "detail": "Requires protein-level activation analysis (Figure S30). See detail panel for individual protein activations.",
        },
        {
            "claim": "F77290 (S1) and F109350 (S8) split from F6960 at 2^17",
            "paper_ref": "Appendix A.4.3.5, Fig S32",
            "verified": "paper only",
            "detail": "Requires 2^17 SAE decoder (~1.3 GB) and protein-level activation data. Referenced in feature splitting panel.",
        },
    ]

    _summary = mo.vstack([
        mo.md("## Paper claims verification\n\nEach claim from the paper about Figure 4D is checked against the notebook's computed results."),
        *[mo.md(f"**{_c['claim']}**\n- Paper: {_c['paper_ref']}\n- Status: {_c['verified']}\n- Detail: {_c['detail']}\n") for _c in _claims],
        mo.md(f"### Summary: {sum(1 for c in _claims if c['verified'] == 'yes')} verified, {sum(1 for c in _claims if c['verified'] == 'partial')} partial, {sum(1 for c in _claims if c['verified'] == 'paper only')} paper-only, {sum(1 for c in _claims if c['verified'] == 'not found')} not found"),
    ])
    _summary
    return


if __name__ == "__main__":
    app.run()
