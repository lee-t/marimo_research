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

    import marimo as mo
    import plotly.graph_objects as go

    CODEBOOK_14 = "2^14"
    CODEBOOK_16 = "2^16"
    SAE_MODEL_14 = "esmc-6b-2024-12-sae-layer60-k64-codebook16384"
    SAE_MODEL_16 = "esmc-6b-2024-12-sae-layer60-k64-codebook65536"
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
                    return _line.split("=", 1)[1].strip().strip("\\\'")
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
        return '<table style="width:100%;border-collapse:collapse"><thead><tr>' + _header + '</tr></thead><tbody>' + _body + '</tbody></table>'


    def api_json(path, method="GET", payload=None, timeout=120, auth=True):
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
        with urlopen(_request, timeout=120) as _response:
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


    def cosine(left, right):
        _dot = sum(a * b for a, b in zip(left, right))
        _norm_left = math.sqrt(sum(a * a for a in left)) or 1.0
        _norm_right = math.sqrt(sum(b * b for b in right)) or 1.0
        return _dot / (_norm_left * _norm_right)


    def pca_2d(vectors):
        if len(vectors) <= 1:
            return [(0.0, 0.0) for _ in vectors]
        _dimension = len(vectors[0])
        _mean = [sum(row[index] for row in vectors) / len(vectors) for index in range(_dimension)]
        _centered = [[value - _mean[index] for index, value in enumerate(row)] for row in vectors]
        _axes = []
        for _ in range(2):
            _axis = [math.sin(index * 1.731 + len(_axes)) for index in range(_dimension)]
            for _ in range(12):
                _projection = sum(sum(row[index] * _axis[index] for index in range(_dimension)) ** 2 for row in _centered)
                _next = [sum(sum(row[index] * _axis[index2] for index2 in range(_dimension)) * row[index] for row in _centered) for index in range(_dimension)]
                _norm = math.sqrt(sum(value * value for value in _next)) or 1.0
                _axis = [value / _norm for value in _next]
            _axes.append(_axis)
        return [(sum(row[index] * _axes[0][index] for index in range(_dimension)), sum(row[index] * _axes[1][index] for index in range(_dimension))) for row in _centered]


    TOKEN_PRESENT = bool(load_biohub_key())

    return (
        CODEBOOK_14,
        CODEBOOK_16,
        FEATURE_API,
        HTTPError,
        Request,
        SAE_MODEL_14,
        SAE_REPO_14,
        SAE_REPO_16,
        URLError,
        api_json,
        decoder_row,
        go,
        math,
        mo,
        range_bytes,
        safetensors_header,
        struct,
        table_html,
        urlopen,
    )


@app.cell
def feature_catalog(FEATURE_API, api_json):

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
def decoder_weights(SAE_REPO_14, range_bytes, safetensors_header):

    _decoder_url_14, _decoder_header_length_14, _decoder_header_14 = safetensors_header(SAE_REPO_14)
    _decoder_w_meta_14 = _decoder_header_14["W_dec"]
    _decoder_data_start_14 = 8 + _decoder_header_length_14 + _decoder_w_meta_14["data_offsets"][0]
    _decoder_bytes_length_14 = _decoder_w_meta_14["data_offsets"][1] - _decoder_w_meta_14["data_offsets"][0]
    decoder_w_dec_14 = range_bytes(
        _decoder_url_14,
        _decoder_data_start_14,
        _decoder_data_start_14 + _decoder_bytes_length_14 - 1,
    )
    decoder_w_dec_shape = tuple(_decoder_w_meta_14["shape"])
    decoder_w_dec_source = _decoder_url_14

    return decoder_w_dec_14, decoder_w_dec_shape


@app.cell
def controls(CODEBOOK_14, CODEBOOK_16, feature_catalog, mo):

    anchor_feature = mo.ui.dropdown(
        options=[row["id"] for row in feature_catalog],
        value="F6716",
        label="Anchor feature",
        searchable=True,
    )
    sae_dims_to_show = mo.ui.multiselect(
        options=[CODEBOOK_14, CODEBOOK_16],
        value=[CODEBOOK_14],
        label="SAE dimensions with decoder rows",
    )
    show_cluster_labels = mo.ui.checkbox(value=True, label="Show semantic labels")
    structure_selector = mo.ui.dropdown(
        options=["auto", "1VKH", "1A0J", "1SCJ", "1LVM"],
        value="auto",
        label="3D structure",
    )
    mo.vstack([
        mo.md("## Figure 4D | Interactive decoder-feature neighborhood"),
        mo.hstack([anchor_feature, sae_dims_to_show, show_cluster_labels], gap=1),
        structure_selector,
    ])

    return (
        anchor_feature,
        sae_dims_to_show,
        show_cluster_labels,
        structure_selector,
    )


@app.cell
def decoder_neighborhood(
    CODEBOOK_14,
    CODEBOOK_16,
    FEATURE_API,
    HTTPError,
    SAE_REPO_16,
    URLError,
    anchor_feature,
    api_json,
    decoder_row,
    decoder_w_dec_14,
    decoder_w_dec_shape,
    feature_catalog,
    math,
    sae_dims_to_show,
    struct,
):

    def _project(_vectors):
        _dimension = len(_vectors[0])
        _mean = [sum(row[index] for row in _vectors) / len(_vectors) for index in range(_dimension)]
        _centered = [[value - _mean[index] for index, value in enumerate(row)] for row in _vectors]
        _axes = []
        for _axis_index in range(2):
            _axis = [math.sin((index + 1) * (1.17 + _axis_index)) for index in range(_dimension)]
            for _ in range(12):
                _next = [sum(sum(row[j] * _axis[j] for j in range(_dimension)) * row[i] for row in _centered) for i in range(_dimension)]
                for _previous in _axes:
                    _dot = sum(a * b for a, b in zip(_next, _previous))
                    _next = [value - _dot * previous for value, previous in zip(_next, _previous)]
                _norm = math.sqrt(sum(value * value for value in _next)) or 1.0
                _axis = [value / _norm for value in _next]
            _axes.append(_axis)
        return [(sum(row[index] * _axes[0][index] for index in range(_dimension)), sum(row[index] * _axes[1][index] for index in range(_dimension))) for row in _centered]

    import numpy as _np
    import umap as _umap

    _anchor_id = int(anchor_feature.value[1:])
    _selected_dims = tuple(sae_dims_to_show.value or [CODEBOOK_14])
    _anchor_info = api_json(f"{FEATURE_API}/features/{_anchor_id}", auth=False, timeout=120)
    _catalog_by_id = {row["feature_id"]: row for row in feature_catalog}

    def _row_14(_feature_id):
        return struct.unpack_from("<2560f", decoder_w_dec_14, int(_feature_id) * 2560 * 4)

    _anchor_vector_14 = _row_14(_anchor_id)
    _anchor_norm_14 = math.sqrt(sum(value * value for value in _anchor_vector_14)) or 1.0
    _real_14 = []
    for _feature_id in range(decoder_w_dec_shape[0]):
        _row = _row_14(_feature_id)
        _row_norm = math.sqrt(sum(value * value for value in _row)) or 1.0
        _cosine = sum(left * right for left, right in zip(_anchor_vector_14, _row)) / (_anchor_norm_14 * _row_norm)
        if _cosine >= 0.1:
            _real_14.append((_feature_id, _row, _cosine))
    _real_14.sort(key=lambda item: item[2], reverse=True)
    _real_14 = _real_14[:300]

    _metadata_ids = [item[0] for item in _real_14[:40]]
    _metadata = {}
    for _feature_id in _metadata_ids:
        try:
            _metadata[_feature_id] = api_json(f"{FEATURE_API}/features/{_feature_id}", auth=False, timeout=60)
        except (HTTPError, URLError, TimeoutError, ValueError):
            _metadata[_feature_id] = {}

    _neighbor_rows = []
    for _feature_id, _row, _cosine in _real_14:
        _info = _metadata.get(_feature_id, _catalog_by_id.get(_feature_id, {}))
        _neighbor_rows.append({
            "key": f"{CODEBOOK_14}:{_feature_id}", "feature_id": f"F{_feature_id}", "feature_index": _feature_id,
            "sae_dim": CODEBOOK_14, "cosine": _cosine, "category": (_info.get("category") or "Catalog feature"),
            "label": _info.get("label", f"Feature F{_feature_id}"), "summary": _info.get("summary", _info.get("description", "")),
            "description": _info.get("description", ""), "vector": _row, "is_anchor": _feature_id == _anchor_id,
        })

    if CODEBOOK_16 in _selected_dims:
        _candidate_ids_16 = [row["feature_index"] for row in _neighbor_rows[:80]]
        _anchor_16 = decoder_row(SAE_REPO_16, _anchor_id)
        _anchor_norm_16 = math.sqrt(sum(value * value for value in _anchor_16)) or 1.0
        for _feature_id in _candidate_ids_16:
            try:
                _row = decoder_row(SAE_REPO_16, _feature_id)
                _row_norm = math.sqrt(sum(value * value for value in _row)) or 1.0
                _cosine = sum(left * right for left, right in zip(_anchor_16, _row)) / (_anchor_norm_16 * _row_norm)
                _info = _metadata.get(_feature_id, _catalog_by_id.get(_feature_id, {}))
                _neighbor_rows.append({
                    "key": f"{CODEBOOK_16}:{_feature_id}", "feature_id": f"F{_feature_id}", "feature_index": _feature_id,
                    "sae_dim": CODEBOOK_16, "cosine": _cosine, "category": (_info.get("category") or "Catalog feature"),
                    "label": _info.get("label", f"Feature F{_feature_id}"), "summary": _info.get("summary", ""),
                    "description": _info.get("description", ""), "vector": _row, "is_anchor": _feature_id == _anchor_id,
                })
            except (HTTPError, URLError, TimeoutError, struct.error):
                pass

    _projection_vectors = [row["vector"] for row in _neighbor_rows]
    if _projection_vectors:
        try:
            _coordinates = _umap.UMAP(n_neighbors=min(15, len(_projection_vectors) - 1), min_dist=0.1, metric="cosine", random_state=0, transform_seed=0, n_jobs=1).fit_transform(_np.asarray(_projection_vectors, dtype=_np.float32)).tolist()
            _projection_name = "UMAP"
        except Exception:
            _coordinates = _project(_projection_vectors)
            _projection_name = "deterministic PCA-like fallback"
    else:
        _coordinates = []
        _projection_name = "empty"
    for _row, (_x, _y) in zip(_neighbor_rows, _coordinates):
        _row["x"] = _x
        _row["y"] = _y
    neighbor_rows = tuple(_neighbor_rows)
    anchor_metadata = _anchor_info
    neighborhood_source = {
        "anchor": f"F{_anchor_id}", "candidate_count": len(_real_14), "dimensions": _selected_dims,
        "cosine_threshold": 0.1, "projection": f"{_projection_name} of real decoder rows; axes intentionally unlabeled",
        "real_matrix": decoder_w_dec_shape,
    }

    return neighbor_rows, neighborhood_source


@app.cell
def panel_d_scatter(
    CODEBOOK_14,
    anchor_feature,
    decoder_w_dec_shape,
    go,
    mo,
    neighbor_rows,
    neighborhood_source,
    sae_dims_to_show,
    show_cluster_labels,
):

    _category_colors = {
        "Catalytic function": "#d62728", "Functional site": "#d62728", "Structural motif": "#2ca02c",
        "Secondary structure": "#2ca02c", "Tertiary interaction": "#9467bd", "Domain / fold": "#ff7f0e",
        "Biochemical environment": "#bcbd22", "Localization / topology": "#17becf", "Catalog feature": "#64748b",
        "Uncategorized": "#64748b",
    }
    _visible_rows = [row for row in neighbor_rows if row["sae_dim"] in tuple(sae_dims_to_show.value or [CODEBOOK_14])]
    _non_anchor = [row for row in _visible_rows if not row["is_anchor"]]
    _anchor_rows = [row for row in _visible_rows if row["is_anchor"]]
    _scatter_fig = go.Figure()
    _scatter_fig.add_trace(go.Scattergl(
        x=[row["x"] for row in _non_anchor], y=[row["y"] for row in _non_anchor],
        mode="markers", name="features",
        marker={"size": [18 if row["sae_dim"] == CODEBOOK_14 else 10 for row in _non_anchor], "color": [_category_colors.get(row["category"], "#64748b") for row in _non_anchor], "opacity": 0.82, "line": {"width": 0.5, "color": "white"}},
        customdata=[[row["key"], row["feature_id"], row["sae_dim"], row["category"], row["label"], row["cosine"]] for row in _non_anchor],
        hovertemplate="<b>%{customdata[1]}</b><br>SAE: %{customdata[2]}<br>Category: %{customdata[3]}<br>Cosine: %{customdata[5]:.3f}<br>%{customdata[4]}<extra>Click for detail</extra>",
    ))
    if _anchor_rows:
        _anchor = _anchor_rows[0]
        _scatter_fig.add_trace(go.Scatter(
            x=[_anchor["x"]], y=[_anchor["y"]], mode="markers+text", name="anchor",
            text=[_anchor["feature_id"]], textposition="top center",
            marker={"symbol": "star", "size": 23, "color": "#111827", "line": {"width": 1, "color": "white"}},
            customdata=[[_anchor["key"], _anchor["feature_id"], _anchor["sae_dim"], _anchor["category"], _anchor["label"], _anchor["cosine"]]],
            hovertemplate="<b>%{customdata[1]}</b> anchor<br>SAE: %{customdata[2]}<br>Cosine: %{customdata[5]:.3f}<extra>Click for detail</extra>",
        ))
    if show_cluster_labels.value and _non_anchor:
        _categories = sorted(set(row["category"] for row in _non_anchor))
        for _category in _categories:
            _category_rows = [row for row in _non_anchor if row["category"] == _category]
            _scatter_fig.add_annotation(x=sum(row["x"] for row in _category_rows) / len(_category_rows), y=sum(row["y"] for row in _category_rows) / len(_category_rows), text=_category, showarrow=False, font={"size": 10, "color": _category_colors.get(_category, "#64748b")})
    _scatter_fig.update_layout(
        height=620, template="plotly_white", dragmode="select", clickmode="event+select",
        title=f"F{int(anchor_feature.value[1:])} decoder neighborhood | cosine ≥ 0.1 | {len(_visible_rows)} plotted rows",
        xaxis={"visible": False, "showgrid": False, "zeroline": False}, yaxis={"visible": False, "showgrid": False, "zeroline": False},
        legend={"orientation": "h"}, margin={"l": 15, "r": 15, "t": 55, "b": 15},
    )
    neighbor_plot = mo.ui.plotly(_scatter_fig, config={"displaylogo": False, "scrollZoom": True}, label="Click a decoder feature")
    panel_d_scatter = mo.vstack([
        mo.md(f"### Interactive decoder directions\nEach dot is an SAE feature, not a protein. Marker size encodes SAE dimensionality; color encodes feature category. Click a dot to load its metadata, residue activations, and structure."),
        neighbor_plot,
        mo.md(f"Real decoder matrix: `{decoder_w_dec_shape[0]:,} × {decoder_w_dec_shape[1]:,}` from Hugging Face. The 2^14 view is scanned exactly; {neighborhood_source["projection"]}."),
    ])
    panel_d_scatter

    return (neighbor_plot,)


@app.cell
def panel_d_detail(
    FEATURE_API,
    HTTPError,
    Request,
    SAE_MODEL_14,
    URLError,
    anchor_feature,
    api_json,
    go,
    mo,
    neighbor_plot,
    structure_selector,
    table_html,
    urlopen,
):

    _selected_points = neighbor_plot.points
    _selected_feature_id = int(anchor_feature.value[1:])
    if _selected_points:
        _point = _selected_points[0]
        _custom = _point.get("customdata", []) if isinstance(_point, dict) else []
        if _custom:
            _selected_feature_id = int(str(_custom[1]).lstrip("F"))
    selected_feature_id = f"F{_selected_feature_id}"

    _feature_detail = {}
    _feature_activation = []
    _feature_sequence = ""
    _feature_label = selected_feature_id
    _feature_source = "metadata unavailable"
    _feature_error = None
    try:
        _feature_detail = api_json(f"{FEATURE_API}/features/{_selected_feature_id}", auth=False, timeout=120)
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
            _feature_activation = [max((_value for _index, _value in zip(_row_i, _row_v) if _index == _selected_feature_id), default=0.0) for _row_i, _row_v in zip(_sparse["feature_indices"], _sparse["values"])][1:-1]
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
            _structure_note = f"{"RCSB PDB" if not _structure_id.startswith("AF-") else "AlphaFoldDB"} {_structure_id}; CA trace" if _ca else f"Structure {_structure_id} contained no CA atoms"
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
    panel_d_detail_status = _feature_detail_status
    panel_d_activation_peak = _activation_max
    panel_d_structure_id = _structure_id
    panel_d_structure_residues = len(_ca)
    panel_d_detail = mo.vstack([
        mo.md(f"## Selected feature `{selected_feature_id}`\n**{_feature_label}**\n\n{_feature_detail.get('summary', 'Select a dot to load its feature metadata.') }\n\nSource: **{_feature_source}** · status: **{_feature_detail_status}**. Peak residue: **{_peak_position if _peak_position is not None else 'none'}**."),
        mo.Html(table_html([{ "feature_id": selected_feature_id, "category": _feature_detail.get("category", "unknown"), "threshold": _feature_detail.get("threshold", "unknown"), "uniref90_frequency": _feature_detail.get("uniref90_frequency", "unknown") }], ["feature_id", "category", "threshold", "uniref90_frequency"])),
        mo.Html(table_html([{ "uniprot_id": row.get("uniprot_id"), "activation": round(row.get("activation", 0.0), 3) } for row in _feature_detail.get("top_swissprot_activations", [])[:8]], ["uniprot_id", "activation"])),
        mo.ui.plotly(_activation_fig, config={"displaylogo": False, "scrollZoom": True}, label="Residue activation"),
        mo.ui.plotly(_structure_fig, config={"displaylogo": False, "scrollZoom": True}, label="3D structure activation"),
        mo.md(_structure_note + (f"\n\n{_feature_error}" if _feature_error else "")),
    ])
    panel_d_detail

    return


@app.cell
def panel_d_splitting(CODEBOOK_14, mo, neighbor_rows, table_html):

    _splitting_rows = [
        {"SAE dimension": "2^13", "status": "not exposed by current Biohub API", "features": "F6960 (paper case study)", "interpretation": "coarse shared catalytic-triad concept"},
        {"SAE dimension": "2^14", "status": "real decoder scan", "features": f"{sum(row['sae_dim'] == CODEBOOK_14 for row in neighbor_rows)} neighbors at cosine ≥ 0.1", "interpretation": "current interactive view"},
        {"SAE dimension": "2^15", "status": "not exposed by current Biohub API", "features": "metadata unavailable", "interpretation": "intermediate split"},
        {"SAE dimension": "2^16", "status": "candidate rows only", "features": "same candidate IDs projected", "interpretation": "full scan requires 640 MB decoder slice"},
        {"SAE dimension": "2^17", "status": "not exposed by current Biohub API", "features": "F77290 / F109350 (paper case study)", "interpretation": "S1/S8-specific split"},
    ]
    panel_d_splitting = mo.vstack([
        mo.md("### Feature splitting across SAE dimensionalities\nThe 2^14 column is computed from the released `W_dec` matrix. Other rows are labeled honestly where the current API or artifact does not provide a complete decoder scan."),
        mo.Html(table_html(_splitting_rows, ["SAE dimension", "status", "features", "interpretation"])),
    ])
    panel_d_splitting

    return


if __name__ == "__main__":
    app.run()
