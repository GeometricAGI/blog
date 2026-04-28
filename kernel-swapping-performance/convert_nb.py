"""Convert the blog notebook to markdown.

- Cells tagged `remove`: input code stripped, outputs (plots, text) kept.
- Cells tagged `collapse`: code wrapped in a <details> expandable block.
- Stream outputs containing ANSI escape codes are converted to <pre> blocks
  with <strong> for bold sequences (color codes are dropped).
"""
import html
import re
from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import Preprocessor, TagRemovePreprocessor

NB_PATH = Path("kernel_swapping_blog.ipynb")
OUT_PATH = Path("kernel_swapping_blog.md")

COLLAPSE_OPEN = "<!--COLLAPSE_START-->"
COLLAPSE_CLOSE = "<!--COLLAPSE_END-->"


class CollapseMarkerPreprocessor(Preprocessor):
    """Wrap source of `collapse`-tagged code cells with marker comments."""

    def preprocess_cell(self, cell, resources, index):
        tags = cell.get("metadata", {}).get("tags", []) or []
        if cell.cell_type == "code" and "collapse" in tags:
            cell.source = f"{COLLAPSE_OPEN}\n{cell.source}\n{COLLAPSE_CLOSE}"
        return cell, resources


def _ansi_to_html(text: str) -> str:
    out = []
    bold_open = False
    i = 0
    while i < len(text):
        if text[i] == "\x1b" and text[i + 1 : i + 2] == "[":
            j = text.find("m", i + 2)
            if j != -1:
                code = text[i + 2 : j]
                if code == "1" and not bold_open:
                    out.append("<strong>")
                    bold_open = True
                elif code == "0" and bold_open:
                    out.append("</strong>")
                    bold_open = False
                # other codes (colors, etc.) are dropped
                i = j + 1
                continue
        out.append(html.escape(text[i]))
        i += 1
    if bold_open:
        out.append("</strong>")
    return "".join(out)


class AnsiToHtmlPreprocessor(Preprocessor):
    """Convert stream outputs containing ANSI escapes into raw HTML <pre> blocks."""

    def preprocess_cell(self, cell, resources, index):
        if cell.cell_type != "code":
            return cell, resources
        for out in cell.get("outputs", []):
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            if "\x1b[" not in text:
                continue
            html_body = _ansi_to_html(text)
            out["output_type"] = "display_data"
            out["data"] = {"text/html": f"<pre>{html_body}</pre>"}
            out["metadata"] = {}
            out.pop("text", None)
            out.pop("name", None)
        return cell, resources


def main():
    nb = nbformat.read(NB_PATH, as_version=4)

    exporter = MarkdownExporter()
    # Remove input of `remove`-tagged cells but keep outputs.
    exporter.register_preprocessor(
        TagRemovePreprocessor(remove_input_tags={"remove"}), enabled=True
    )
    exporter.register_preprocessor(CollapseMarkerPreprocessor, enabled=True)
    exporter.register_preprocessor(AnsiToHtmlPreprocessor, enabled=True)

    body, resources = exporter.from_notebook_node(nb)

    # Wrap marked code blocks in <details> elements.
    pattern = re.compile(
        rf"```python\n{re.escape(COLLAPSE_OPEN)}\n(.*?)\n{re.escape(COLLAPSE_CLOSE)}\n```",
        re.DOTALL,
    )

    def wrap(match):
        code = match.group(1)
        return (
            '<details markdown="1">\n'
            "<summary>Show code</summary>\n\n"
            f"```python\n{code}\n```\n\n"
            "</details>"
        )

    body = pattern.sub(wrap, body)

    OUT_PATH.write_text(body)

    # Save any output resources (e.g. embedded images) next to the md file.
    outputs = resources.get("outputs", {})
    for name, data in outputs.items():
        Path(name).write_bytes(data)
    if outputs:
        print(f"Wrote {len(outputs)} output assets: {list(outputs)}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
