# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.0",
#     "esm@git+https://github.com/Biohub/esm.git@main",
# ]
# ///
"""Batch-fetch ESMC embeddings for sequences in a FASTA file.

Usage
-----
    uv run scripts/fetch_embeddings.py sequences.fasta --out embeddings.npz

Environment
-----------
    BIOHUB_TOKEN   API token from https://biohub.ai/developer-console/api-keys

Output
------
    .npz file with keys:
        names       (N,)          sequence identifiers
        sequences   (N,)          amino-acid strings
        embeddings  list[ndarray] per-sequence (L, D) float32 arrays
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Return list of (name, sequence) from a FASTA file."""
    entries: list[tuple[str, str]] = []
    name = ""
    seq_parts: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name:
                entries.append((name, "".join(seq_parts)))
            name = line[1:].split()[0]
            seq_parts = []
        else:
            seq_parts.append(line)
    if name:
        entries.append((name, "".join(seq_parts)))
    return entries


def _load_token(explicit_token: str | None = None) -> str:
    if explicit_token:
        return explicit_token
    for key in ("BIOHUB_API_KEY", "BIOHUB_TOKEN"):
        val = os.environ.get(key, "")
        if val:
            return val
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(("BIOHUB_API_KEY=", "BIOHUB_TOKEN=")):
                return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def _fetch_embeddings(
    entries: list[tuple[str, str]],
    model: str,
    token: str,
    *,
    layer: int | None = None,
    sae_model: str | None = None,
    normalize_features: bool = True,
) -> list:
    """Return list of (L, D) float32 numpy arrays, one per sequence."""
    try:
        from esm.sdk.forge import ESMCForgeInferenceClient
    except ImportError:
        from esm.sdk import esmc_client as ESMCForgeInferenceClient
    from esm.sdk.api import ESMProtein, LogitsConfig, SAEConfig

    client = ESMCForgeInferenceClient(model=model, url="https://biohub.ai", token=token)
    
    if sae_model:
        config = LogitsConfig(
            sae_config=SAEConfig(model=sae_model, normalize_features=normalize_features)
        )
    else:
        config = LogitsConfig(
            return_embeddings=True,
            return_hidden_states=(layer is not None),
            ith_hidden_layer=layer,
        )

    results = []
    for i, (name, seq) in enumerate(entries, 1):
        print(f"  [{i}/{len(entries)}] {name} ({len(seq)} aa) ...", file=sys.stderr)
        protein = ESMProtein(sequence=seq)
        tensor = client.encode(protein)
        out = client.logits(tensor, config, return_bytes=False)
        
        if sae_model:
            # Extract SAE features tensor and convert to numpy, stripping BOS/EOS tokens
            sae_out = out.sae_outputs[sae_model].to_dense().numpy()
            results.append(sae_out[1:-1])
        else:
            emb = out.hidden_states if layer is not None else out.embeddings
            results.append(emb[1:-1])  # strip BOS / EOS tokens
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Batch-fetch ESMC embeddings or SAE features for proteins in a FASTA file.",
    )
    parser.add_argument("fasta", type=Path, help="Input FASTA file")
    parser.add_argument(
        "--out", "-o", type=Path, default=Path("embeddings.npz"),
        help="Output .npz file (default: embeddings.npz)",
    )
    parser.add_argument(
        "--model", default="esmc-6b-2024-12",
        choices=["esmc-6b-2024-12", "esmc-300m-2024-12", "esmc-600m-2024-12"],
        help="ESMC model to use (default: esmc-6b-2024-12)",
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Return hidden states at this transformer layer (default: final layer)",
    )
    parser.add_argument(
        "--sae", action="store_true",
        help="Extract Top-K SAE features using esmc-6b-2024-12-sae-layer60-k64-codebook16384",
    )
    parser.add_argument(
        "--sae-model", default="esmc-6b-2024-12-sae-layer60-k64-codebook16384",
        help="SAE model identifier",
    )
    parser.add_argument(
        "--token", default=None,
        help="Biohub API token (default: reads $BIOHUB_API_KEY, $BIOHUB_TOKEN, or .env)",
    )
    args = parser.parse_args(argv)

    token = _load_token(args.token)
    if not token:
        print(
            "Error: Biohub API token not found.\n"
            "  Set $BIOHUB_API_KEY, $BIOHUB_TOKEN, create a .env file, or pass --token.",
            file=sys.stderr,
        )
        sys.exit(1)

    entries = _parse_fasta(args.fasta)
    if not entries:
        print(f"Error: no sequences found in {args.fasta}", file=sys.stderr)
        sys.exit(1)

    mode_str = f"SAE features ({args.sae_model})" if args.sae else f"embeddings ({args.model})"
    print(f"Fetching {mode_str} for {len(entries)} sequences ...", file=sys.stderr)
    
    sae_model_arg = args.sae_model if args.sae else None
    results = _fetch_embeddings(
        entries,
        args.model,
        token,
        layer=args.layer,
        sae_model=sae_model_arg,
    )

    import numpy as np
    names = [n for n, _ in entries]
    sequences = [s for _, s in entries]

    key_name = "features" if args.sae else "embeddings"
    save_kwargs = {
        "names": np.array(names, dtype=object),
        "sequences": np.array(sequences, dtype=object),
        key_name: np.array(results, dtype=object),
    }

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), **save_kwargs)
    print(f"Saved {len(entries)} sequences ({key_name}) → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
