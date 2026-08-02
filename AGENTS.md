# Repository Guide

## Notebooks

- `notebooks/*.py` are standalone marimo apps, not package modules. Their PEP 723 headers are the source of truth for Python and dependencies; the root `pyproject.toml`/`uv.lock` intentionally contain no runtime dependencies. Add a notebook dependency to its header, not the root project.
- Load `.agents/skills/marimo-notebook/SKILL.md` before changing a notebook. For paper-to-notebook work, also follow `.agents/skills/implement-paper-auto/SKILL.md`; keep code hidden and the narrative, controls, and outputs ahead of implementation cells.
- Preserve marimo's reactive cell structure: cell parameters and returns define dependencies. Do not convert notebooks into linear scripts or rely on source order.
- `pytorch_intro.html`, `papers/`, and `downloads/` are research inputs/artifacts, not runtime dependencies of the current self-contained notebooks. `papers/`, `downloads/`, and `__marimo__/` state are gitignored.
- Exception to self-containment: `notebooks/jacobian_lens_workspace.py` reads committed figure data from `notebooks/data/jacobian_lens_workspace/` (its `MANIFEST.json` maps each file to its source in the archived article). Regenerate it with `uv run tools/prepare_workspace_figure_data.py` after re-archiving the paper into `downloads/workspace-global-workspace/`. The companion `notebooks/jacobian_lens_live.py` is self-contained again (downloads models/lenses from HF at runtime) but has a heavy PEP 723 header (torch, transformers, `jlens` from git) by design.

## Verification

- Run one notebook non-interactively: `uv run notebooks/<name>.py`.
- Check one changed notebook: `uvx marimo check --strict notebooks/<name>.py`.
- Check all notebooks: `uvx marimo check --strict notebooks/*.py`.
- Run or edit interactively with the notebook's inline environment: `uvx marimo run --sandbox notebooks/<name>.py` or `uvx marimo edit --sandbox notebooks/<name>.py`. Do not assume `uv run marimo ...` works in a clean checkout because marimo is absent from the root environment.
- There is no repository-wide test, lint, typecheck, or CI configuration. For notebook changes, strict check followed by script execution is the focused automated verification.

## Archive Tool

- `tools/archive_distill_article.py <url> <output>` replaces the entire output directory before downloading. Use a dedicated path (normally under ignored `downloads/`), never a directory containing unrelated work.
