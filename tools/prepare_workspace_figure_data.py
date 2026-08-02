"""Prepare curated figure data for notebooks/jacobian_lens_workspace.py.

Reads the archived copy of https://transformer-circuits.pub/2026/workspace/
from downloads/workspace-global-workspace/source/2026/workspace/data/ and writes
a small, minified subset into notebooks/data/jacobian_lens_workspace/.

Run once when updating the committed data:

    uv run tools/prepare_workspace_figure_data.py

The output directory is committed so the notebook runs offline; it is NOT a
runtime dependency of this script. downloads/ is gitignored, so if the archive
is missing, re-create it with:

    uv run tools/archive_distill_article.py \
        https://transformer-circuits.pub/2026/workspace/ \
        downloads/workspace-global-workspace
"""

import json
import shutil
from pathlib import Path

SRC = Path("downloads/workspace-global-workspace/source/2026/workspace")
OUT = Path("notebooks/data/jacobian_lens_workspace")

# display-name -> (source data dir, source file)
PLAIN_COPIES = {
    "verbal_report": ("verbal-report", "data.json"),
    "verbal_introspection": ("verbal-introspection", "data.json"),
    "verbal_report_decomposition": ("verbal-report-decomposition-merged", "data.json"),
    "modulation_lines": ("modulation-lines", "lines.json"),
    "latent_patching": ("latent-patching", "data.json"),
    "multihop_swap_success": ("multihop-swap-success", "data.json"),
    "probe_swap": ("probe-swap", "data.json"),
    "flex_gen_example": ("flex-gen-example", "data.json"),
    "flex_gen_systematic": ("flex-generalization-systematic", "data.json"),
    "selectivity_language": ("selectivity-language", "swap.json"),
    "selectivity_linecount": ("selectivity-linecount", "data.json"),
    "ablation_strength": ("ablation-strength", "table.json"),
    "ablation_bars": ("ablation-bars", "bars.json"),
    "selfreport": ("selfreport", "data.json"),
    "layer_diagram_cka": ("layer-diagram", "cka.json"),
    "ignition": ("ignition", "data.json"),
    "capacity_occupancy": ("capacity-fve-occupancy", "data.json"),
    "capacity_lists": ("capacity-final-band", "data.json"),
    "mlp_gain": ("mlp-gain", "mlp_gain.json"),
    "attn_broadcast": ("attn-broadcast-reselect", "data.json"),
    "broadcast_ablation": ("broadcast-ablation", "data.json"),
    "jlens_rm_bias": ("jlens-rm-bias", "data.json"),
    "roleplay_lens": ("roleplay-lens", "data.json"),
    "pref_violation_lens": ("pref-violation-lens", "data.json"),
    "metacog_alarm": ("metacog-alarm", "data.json"),
    "reflection_fabrication": ("reflection-fabrication", "reflection.json"),
    "reflection_deception": ("reflection-deception", "reflection.json"),
}

# static paper figure (Figure 4, J-lens schematic)
PNG_COPY = ("png/img_1b62b10ab235e6e7.png", "fig4_jacobian_lens_schematic.png")


def trim_topk(rows, keep=5, minp=0.05):
    """Trim lens readout top-k lists, dropping near-zero-prob entries."""
    for row in rows:
        for lens in ("jacobian", "logit", "tuned"):
            if lens in row and "k" in row[lens]:
                row[lens]["k"] = [e for e in row[lens]["k"][:keep] if e["p"] > minp]
    return rows


def slim_lens_compare(data):
    """methods-qualitative: keep per-(position, layer) top-k readouts per lens."""
    panels = []
    for p in data["panels"]:
        panels.append(
            {
                "name": p["name"],
                "title": p["title"],
                "tokens": p["tokens"],
                "keyPos": p["keyPos"],
                "highlights": p["highlights"],
                "rows": trim_topk(p["rows"], keep=6, minp=0.1),
                "positions": {
                    pos: trim_topk(layers, keep=3, minp=0.1)
                    for pos, layers in p["positions"].items()
                },
            }
        )
    return {"methods": data["methods"], "panels": panels}


def slim_modulation_readout(data):
    """modulation-readout: trim per-position top-k lists."""
    panels = []
    for p in data["panels"]:
        positions = [
            [{**layer, "topk": [e for e in layer["topk"][:5] if e["p"] > 0.05]}
             for layer in layers]
            for layers in p["positions"]
        ]
        panels.append({**p, "positions": positions})
    return {"panels": panels}


def slim_misalign_lens(data):
    """Drop bulky per-prompt transcripts; keep aggregate stats and the example."""
    panels = []
    for panel in data["panels"]:
        cats = {
            cat: {
                model: {k: v for k, v in stats.items() if k != "prompts"}
                for model, stats in models.items()
            }
            for cat, models in panel["cats"].items()
        }
        panels.append({"title": panel["title"], "cats": cats})
    return {
        "models": data["models"],
        "categories": data["categories"],
        "panels": panels,
        "example": data["example"],
        "ylabel": data["ylabel"],
    }


def slim_blackmail_clamp(data):
    """Keep the aggregate rates and the two annotated excerpts; drop transcripts."""
    return {"agg": data["agg"], "excerpts": data["excerpts"]}


def slim_ablation_examples(data):
    """Drop raw logprob arrays; keep tokens and KL values."""
    return [
        {k: v for k, v in ex.items() if k != "raw"}
        for ex in data
    ]


CUSTOM = {
    "lens_compare": ("methods-qualitative", "qualitative.json", slim_lens_compare),
    "modulation_readout": ("modulation-readout", "modulation.json", slim_modulation_readout),
    "misalign_lens": ("misalign-lens", "data.json", slim_misalign_lens),
    "blackmail_clamp": ("blackmail-clamp", "data.json", slim_blackmail_clamp),
    "ablation_examples": ("ablation-examples", "chips.json", slim_ablation_examples),
}


def main():
    if not SRC.is_dir():
        raise SystemExit(f"Source archive missing: {SRC} (see docstring to re-create)")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    manifest = {"source": "https://transformer-circuits.pub/2026/workspace/", "files": {}}

    for name, (d, f) in PLAIN_COPIES.items():
        payload = json.loads((SRC / "data" / d / f).read_text(encoding="utf-8"))
        dest = OUT / f"{name}.json"
        dest.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8")
        manifest["files"][dest.name] = f"data/{d}/{f}"

    for name, (d, f, fn) in CUSTOM.items():
        payload = json.loads((SRC / "data" / d / f).read_text(encoding="utf-8"))
        slimmed = fn(payload)
        dest = OUT / f"{name}.json"
        dest.write_text(json.dumps(slimmed, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8")
        manifest["files"][dest.name] = f"data/{d}/{f}"

    src_png, dst_png = PNG_COPY
    shutil.copy2(SRC / src_png, OUT / dst_png)
    manifest["files"][dst_png] = src_png

    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"wrote {len(manifest['files'])} files to {OUT} ({total / 1024:.0f} KB)")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:45s} {p.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
