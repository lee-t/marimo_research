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


def _fetch_embeddings(
    entries: list[tuple[str, str]],
    model: str,
    token: str,
    *,
    layer: int | None = None,
) -> list:
    """Return list of (L, D) float32 numpy arrays, one per sequence."""
    from esm.sdk import esmc_client
    from esm.sdk.api import ESMProtein, LogitsConfig

    client = esmc_client(model=model, url="https://biohub.ai", token=token)
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
        out = client.logits(tensor, config)
        emb = out.hidden_states if layer is not None else out.embeddings
        results.append(emb[1:-1])  # strip BOS / EOS tokens
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Batch-fetch ESMC embeddings for proteins in a FASTA file.",
    )
    parser.add_argument("fasta", type=Path, help="Input FASTA file")
    parser.add_argument(
        "--out", "-o", type=Path, default=Path("embeddings.npz"),
        help="Output .npz file (default: embeddings.npz)",
    )
    parser.add_argument(
        "--model", default="esmc-300m-2024-12",
        choices=["esmc-300m-2024-12", "esmc-600m-2024-12"],
        help="ESMC model to use",
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Return hidden states at this transformer layer (default: final layer)",
    )
    parser.add_argument(
        "--token", default=None,
        help="Biohub API token (default: $BIOHUB_TOKEN)",
    )
    args = parser.parse_args(argv)

    token = args.token or os.environ.get("BIOHUB_TOKEN", "")
    if not token:
        print(
            "Error: Biohub API token not found.\n"
            "  Set $BIOHUB_TOKEN or pass --token.",
            file=sys.stderr,
        )
        sys.exit(1)

    entries = _parse_fasta(args.fasta)
    if not entries:
        print(f"Error: no sequences found in {args.fasta}", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching embeddings for {len(entries)} sequences ...", file=sys.stderr)
    embeddings = _fetch_embeddings(entries, args.model, token, layer=args.layer)

    import numpy as np
    names = [n for n, _ in entries]
    sequences = [s for _, s in entries]

    np.savez(
        args.out,
        names=np.array(names, dtype=object),
        sequences=np.array(sequences, dtype=object),
        embeddings=np.array(embeddings, dtype=object),
    )
    print(f"Saved {len(entries)} embeddings → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
