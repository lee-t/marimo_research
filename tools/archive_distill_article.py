#!/usr/bin/env python3
"""Archive a Distill-style article for retrieval-augmented LLM use.

The generated Markdown is the primary ingest artifact. The source tree keeps
the original interactive figures and their assets available for visual agents.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


USER_AGENT = "llm-paper-archive/1.0 (+https://transformer-circuits.pub/)"
ASSET_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".svg", ".json", ".csv", ".tsv", ".parquet", ".npy", ".npz", ".bin"}


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read()


def local_path(url: str, source_root: Path) -> Path:
    parsed = urlparse(url)
    path = parsed.path.lstrip("/") or "index.html"
    if parsed.query:
        path = f"{path}__{parsed.query.replace('/', '_')}"
    return source_root / path


def is_safe_download_url(url: str) -> bool:
    """Reject JavaScript string fragments accidentally matched as HTML URLs."""
    path = urlparse(url).path
    return path.isascii() and not any(character in path for character in '<>"\\|?*\r\n') and " " not in path


def write_download(url: str, source_root: Path) -> Path:
    destination = local_path(url, source_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(fetch(url))
    return destination


def text(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value)).strip()


def markdown_text(value: str) -> str:
    """Normalize HTML whitespace without collapsing Markdown block boundaries."""
    value = unescape(value)
    value = re.sub(r"[\t\r\f\v ]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


class ArticleParser(HTMLParser):
    """Extract a linear, citation-preserving representation from d-article."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_article = False
        self.skip_depth = 0
        self.current_heading: tuple[int, list[str]] | None = None
        self.current_caption: list[str] | None = None
        self.current_math: tuple[bool, list[str]] | None = None
        self.current_footnote: list[str] | None = None
        self.current_pre: list[str] | None = None
        self.figure: dict | None = None
        self.lines: list[str] = []
        self.figures: list[dict] = []
        self.footnotes: list[str] = []
        self.fig_refs: dict[str, int | str] = {}
        self._list_depth = 0
        self._fig_ref_depth = 0

    def emit(self, value: str) -> None:
        if value:
            self.lines.append(value)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "d-article":
            self.in_article = True
            return
        if not self.in_article:
            return
        if tag in {"style", "script", "nav", "d-title"}:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in {"h2", "h3", "h4"}:
            self.current_heading = ({"h2": 2, "h3": 3, "h4": 4}[tag], [])
        elif tag == "p":
            self.emit("\n\n")
        elif tag in {"ul", "ol"}:
            self._list_depth += 1
            self.emit("\n")
        elif tag == "li":
            self.emit("\n" + "  " * max(0, self._list_depth - 1) + "- ")
        elif tag == "br":
            self.emit("\n")
        elif tag == "d-cite":
            keys = [key.strip() for key in attrs_dict.get("key", "").split(",") if key.strip()]
            self.emit("[" + "; ".join(f"@{key}" for key in keys) + "]")
        elif tag == "d-footnote":
            self.current_footnote = []
        elif tag == "d-math":
            self.current_math = ("block" in attrs_dict, [])
        elif tag == "pre":
            self.current_pre = []
        elif tag == "figure":
            self.figure = {
                "id": attrs_dict.get("id"),
                "number": attrs_dict.get("data-fignum"),
                "class": attrs_dict.get("class", ""),
                "caption": "",
                "images": [],
                "interactive": False,
            }
        elif tag == "img" and self.figure is not None:
            src = attrs_dict.get("src")
            if src:
                self.figure["images"].append(src)
        elif tag == "figcaption" and self.figure is not None:
            self.current_caption = []
        elif tag == "div" and self.figure is not None:
            self.figure["interactive"] = True
        elif tag == "a" and "fig-ref" in attrs_dict.get("class", ""):
            reference = "{FIGREF:" + attrs_dict.get("data-ref", "unknown") + "}"
            if self.current_caption is not None:
                self.current_caption.append(reference)
            elif self.current_footnote is not None:
                self.current_footnote.append(reference)
            else:
                self.emit(reference)
            self._fig_ref_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "d-article":
            self.in_article = False
            return
        if not self.in_article:
            return
        if tag in {"style", "script", "nav", "d-title"} and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag in {"h2", "h3", "h4"} and self.current_heading:
            level, heading = self.current_heading
            self.emit("\n" + "#" * level + " " + text("".join(heading)) + "\n")
            self.current_heading = None
        elif tag in {"ul", "ol"}:
            self._list_depth = max(0, self._list_depth - 1)
            self.emit("\n")
        elif tag == "d-footnote" and self.current_footnote is not None:
            self.footnotes.append(text("".join(self.current_footnote)))
            self.emit(f"[^fn-{len(self.footnotes)}]")
            self.current_footnote = None
        elif tag == "d-math" and self.current_math is not None:
            is_block, math = self.current_math
            rendered = text("".join(math))
            if self.current_caption is not None:
                self.current_caption.append(f"${rendered}$")
            elif self.current_footnote is not None:
                self.current_footnote.append(f"${rendered}$")
            else:
                self.emit(f"\n$$\n{rendered}\n$$\n" if is_block else f"${rendered}$")
            self.current_math = None
        elif tag == "pre" and self.current_pre is not None:
            self.emit("\n```text\n" + "".join(self.current_pre).strip() + "\n```\n")
            self.current_pre = None
        elif tag == "figcaption" and self.current_caption is not None and self.figure is not None:
            self.figure["caption"] = text("".join(self.current_caption))
            self.current_caption = None
        elif tag == "figure" and self.figure is not None:
            figure = self.figure
            figure["kind"] = "interactive" if figure["interactive"] else "static"
            self.figures.append(figure)
            if figure["id"]:
                self.fig_refs[figure["id"]] = figure["number"] or figure["id"]
            label = f"Figure {figure['number']}" if figure["number"] else "Figure"
            caption = re.sub(r"^Figure\s+\d+:\s*", "", figure["caption"])
            anchor = f"<a id=\"figure-{figure['number']}\"></a>\n" if figure["number"] else ""
            if figure["images"]:
                self.emit(f"\n{anchor}![{label}: {caption}]({figure['images'][0]})\n")
            else:
                self.emit(
                    f"\n{anchor}> **{label} (interactive).** {caption} "
                    "The original interactive renderer and its available assets are retained in `source/`.\n"
                )
            self.figure = None
        elif tag == "a" and self._fig_ref_depth:
            self._fig_ref_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.in_article or self.skip_depth or self._fig_ref_depth:
            return
        if self.current_heading is not None:
            self.current_heading[1].append(data)
        elif self.current_math is not None:
            self.current_math[1].append(data)
        elif self.current_caption is not None:
            self.current_caption.append(data)
        elif self.current_footnote is not None:
            self.current_footnote.append(data)
        elif self.current_pre is not None:
            self.current_pre.append(data)
        elif self.figure is None:
            self.emit(data)

    def markdown(self, metadata: dict[str, object]) -> str:
        body = markdown_text("".join(self.lines))
        for figure_id, figure_number in self.fig_refs.items():
            body = body.replace(f"{{FIGREF:{figure_id}}}", f"[Figure {figure_number}](#figure-{figure_number})")
        body = re.sub(r"\{FIGREF:([^}]+)\}", r"[Reference: #\1]", body)

        front_matter = "\n".join(
            [
                "---",
                f"title: {json.dumps(metadata['title'])}",
                f"source_url: {json.dumps(metadata['source_url'])}",
                f"published: {json.dumps(metadata.get('published'))}",
                f"retrieved_at: {json.dumps(metadata['retrieved_at'])}",
                "citation_format: pandoc-bibtex-keys",
                "figure_policy: captions are transcribed; interactive renderers are retained in source/",
                "---",
            ]
        )
        resolved_notes: list[str] = []
        for note in self.footnotes:
            for figure_id, figure_number in self.fig_refs.items():
                note = note.replace(f"{{FIGREF:{figure_id}}}", f"Figure {figure_number}")
            resolved_notes.append(re.sub(r"\{FIGREF:([^}]+)\}", r"Reference #\1", note))
        notes = "\n".join(f"[^fn-{index}]: {note}" for index, note in enumerate(resolved_notes, 1))
        return f"{front_matter}\n\n# {metadata['title']}\n\n{body}\n\n## Footnotes\n\n{notes}\n"


def extract_metadata(html: str, source_url: str) -> dict[str, object]:
    title_match = re.search(r"<title>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
    authors = re.findall(r"<span class='author'>(.*?)</span>", html, re.DOTALL)
    published_match = re.search(r"<div>\s*(July \d+, 2026)\s*</div>", html)
    return {
        "title": text(title_match.group(1)) if title_match else "Untitled article",
        "authors": [text(re.sub(r"<.*?>", "", author)) for author in authors],
        "published": published_match.group(1) if published_match else None,
        "source_url": source_url,
        "retrieved_at": datetime.now(UTC).isoformat(),
    }


def urls_from_html(html: str, base_url: str) -> set[str]:
    values = re.findall(r"(?:src|href)=[\"']([^\"'#?]+)", html, re.IGNORECASE)
    return {urljoin(base_url, value) for value in values if not value.startswith(("mailto:", "javascript:"))}


def asset_urls_from_bundle(bundle: str, article_url: str) -> set[str]:
    """Find the data files loaded by the bundle's figure-initialization map."""
    marker = re.compile(r"// [^\n]* public/([^\n/]+(?:/[^\n/]+)*)/[^\n]+ ───")
    matches = list(marker.finditer(bundle))
    function_assets: dict[str, set[str]] = {}
    for index, match in enumerate(matches):
        section = bundle[match.end() : matches[index + 1].start() if index + 1 < len(matches) else len(bundle)]
        functions = re.findall(r"window\.(init[A-Za-z0-9]+)\s*=", section)
        candidates = re.findall(r"[\"']([^\"']+?(?:" + "|".join(re.escape(suffix) for suffix in ASSET_SUFFIXES) + r"))[\"']", section)
        for function in functions:
            function_assets[function] = {
                candidate
                for candidate in candidates
                if not candidate.startswith(("data:", "http:"))
                and re.fullmatch(r"[A-Za-z0-9_./-]+(?:" + "|".join(re.escape(suffix) for suffix in ASSET_SUFFIXES) + r")", candidate)
            }

    init_start = bundle.find("await Promise.all([")
    init_end = bundle.find("].map(async", init_start)
    init_block = bundle[init_start:init_end]
    specs: list[tuple[str, str, str]] = []
    for match in re.finditer(r"\[[^\[\]]*\]", init_block):
        item = match.group(0)[1:-1]
        function_match = re.search(r"window\.(init[A-Za-z0-9]+)", item)
        if not function_match:
            continue
        parts: list[str] = []
        start = 0
        depth = 0
        for index, character in enumerate(item):
            if character in "({":
                depth += 1
            elif character in ")}" and depth:
                depth -= 1
            elif character == "," and depth == 0:
                parts.append(item[start:index].strip())
                start = index + 1
        parts.append(item[start:].strip())
        if len(parts) < 2:
            continue
        directories = [part.strip().strip("'") for part in parts[1:]]
        if len(directories) == 1:
            specs.append((function_match.group(1), directories[0], ""))
        else:
            specs.append((function_match.group(1), directories[-2], directories[-1]))
    assets: set[str] = set()
    for function, mount_directory, data_directory in specs:
        for candidate in function_assets.get(function, set()):
            assets.add(urljoin(article_url, f"data/{data_directory or mount_directory}/{candidate}"))
    return assets


def archive(source_url: str, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    source_root = output_dir / "source"
    source_root.mkdir(parents=True)

    raw_html = fetch(source_url)
    html = raw_html.decode("utf-8", errors="replace")
    metadata = extract_metadata(html, source_url)
    parser = ArticleParser()
    parser.feed(html)
    for figure in parser.figures:
        caption = figure["caption"]
        for figure_id, figure_number in parser.fig_refs.items():
            caption = caption.replace(f"{{FIGREF:{figure_id}}}", f"Figure {figure_number}")
        figure["caption"] = re.sub(r"\{FIGREF:([^}]+)\}", r"Reference #\1", caption)

    article_path = urlparse(source_url).path.strip("/")
    source_prefix = f"source/{article_path}/" if article_path else "source/"
    markdown = parser.markdown(metadata).replace("](./png/", "](" + source_prefix + "png/")
    (output_dir / "article.md").write_text(markdown, encoding="utf-8")
    (output_dir / "figures.json").write_text(json.dumps(parser.figures, indent=2) + "\n", encoding="utf-8")
    (source_root / "index.html").write_bytes(raw_html)

    downloaded: list[str] = [source_url]
    failures: dict[str, str] = {}
    candidates = urls_from_html(html, source_url)
    bibliography_url = urljoin(source_url, "bibliography.bib")
    candidates.add(bibliography_url)
    bundle_url = urljoin(source_url, "public/bundle.js")
    candidates.add(bundle_url)

    for url in sorted(candidates):
        if urlparse(url).netloc != urlparse(source_url).netloc or not is_safe_download_url(url):
            continue
        try:
            destination = write_download(url, source_root)
            downloaded.append(url)
            if url == bibliography_url:
                shutil.copyfile(destination, output_dir / "references.bib")
        except (HTTPError, URLError, TimeoutError) as error:
            failures[url] = str(error)

    bundle_file = local_path(bundle_url, source_root)
    if bundle_file.exists():
        bundle_assets = asset_urls_from_bundle(bundle_file.read_text(encoding="utf-8", errors="replace"), source_url)
        for url in sorted(bundle_assets):
            if urlparse(url).netloc != urlparse(source_url).netloc or not is_safe_download_url(url):
                continue
            try:
                write_download(url, source_root)
                downloaded.append(url)
            except (HTTPError, URLError, TimeoutError) as error:
                failures[url] = str(error)

    readme = """# LLM Ingest Package

Use `article.md` as the primary text document and load `references.bib` to resolve
Pandoc-style citation keys. `figures.json` is the machine-readable figure index.
Static figures are linked from the Markdown. Interactive figures are intentionally
represented by their exact captions plus retained source assets in `source/`; they
should be rendered or inspected by a visual/browser-capable agent rather than
silently converted to invented text.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        **metadata,
        "article_file": "article.md",
        "bibliography_file": "references.bib",
        "figures_file": "figures.json",
        "figure_count": len(parser.figures),
        "interactive_figure_count": sum(figure["kind"] == "interactive" for figure in parser.figures),
        "footnote_count": len(parser.footnotes),
        "downloaded_urls": sorted(set(downloaded)),
        "failed_downloads": failures,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Canonical article URL")
    parser.add_argument("output", type=Path, help="Directory to create")
    args = parser.parse_args()
    manifest = archive(args.url, args.output)
    print(json.dumps({key: manifest[key] for key in ("title", "figure_count", "interactive_figure_count", "failed_downloads")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
