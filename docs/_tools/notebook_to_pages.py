"""Write the Examples pages of the documentation from the tutorial notebook.

The notebook ``examples/ViSta_tutorial.ipynb`` is kept in the repository
without outputs.  Run it once (Colab, or a local Jupyter with ``VISTA_REPO``),
save the executed copy, and turn it into one page per section::

    python docs/_tools/notebook_to_pages.py executed.ipynb \\
        --replace /content/ViSta-HPC=/content/ViSta-HPC

The text of every page is the markdown of the notebook, the code its code
cells, the results its outputs (printed text and figures; warnings written to
stderr are left out).  ``--replace OLD=NEW`` rewrites paths in the outputs
(e.g. a local checkout as ``/content/ViSta-HPC``).  The markdown and the code
are taken from ``examples/ViSta_tutorial.ipynb`` and must match the executed
copy cell by cell (only the outputs are taken from it).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
REPO = os.path.dirname(DOCS)
OUT = os.path.join(DOCS, "examples")
FIG = os.path.join(OUT, "figures")

HEADING = re.compile(r"^##\s+(\d+)\.\s+(.*)$", re.M)


def slug(text, max_len=28):
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    out = ""
    for w in words:
        if out and len(out) + 1 + len(w) > max_len:
            break
        out = f"{out}_{w}" if out else w
    return out


def text_of(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def outputs_md(cell, number, counter, replace):
    parts = []
    for o in cell.get("outputs", []):
        kind = o.get("output_type")
        if kind == "stream":
            if o.get("name") == "stderr":
                continue
            txt = "".join(o["text"]) if isinstance(o["text"], list) else o["text"]
        elif kind in ("execute_result", "display_data"):
            data = o.get("data", {})
            if "image/png" in data:
                counter[0] += 1
                name = f"{number:02d}_{counter[0]:02d}.png"
                with open(os.path.join(FIG, name), "wb") as fh:
                    fh.write(base64.b64decode(data["image/png"]))
                parts.append(f"```{{image}} figures/{name}\n:width: 100%\n```")
                continue
            txt = "".join(data.get("text/plain", ""))
        elif kind == "error":
            raise RuntimeError(f"the notebook stopped with an error: {o.get('ename')}")
        else:
            continue
        for old, new in replace:
            txt = txt.replace(old, new)
        txt = "\n".join(l.rstrip() for l in txt.rstrip("\n").split("\n"))
        if txt.strip():
            parts.append("```text\n" + txt + "\n```")
    return parts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("executed", help="the executed copy of the tutorial notebook")
    ap.add_argument("--notebook", default=os.path.join(REPO, "examples", "ViSta_tutorial.ipynb"))
    ap.add_argument("--replace", action="append", default=[], metavar="OLD=NEW")
    a = ap.parse_args()
    replace = [tuple(r.split("=", 1)) for r in a.replace]

    clean = json.load(open(a.notebook))["cells"]
    done = json.load(open(a.executed))["cells"]
    if len(clean) != len(done):
        sys.exit("the executed notebook has a different number of cells")
    for i, (c, d) in enumerate(zip(clean, done)):
        if c["cell_type"] == "code" and text_of(c) != text_of(d):
            sys.exit(f"code cell {i} differs from the executed copy: run the notebook again")

    os.makedirs(FIG, exist_ok=True)
    for f in os.listdir(FIG):
        if f.endswith(".png"):
            os.remove(os.path.join(FIG, f))

    pages, intro, current = [], [], None
    for c, d in zip(clean, done):
        if c["cell_type"] == "markdown":
            m = HEADING.search(text_of(c))
            if m:
                current = {"number": int(m.group(1)), "title": m.group(2).strip(), "blocks": []}
                pages.append(current)
                body = HEADING.sub("", text_of(c), count=1).strip()
                if body:
                    current["blocks"].append(body)
                continue
            (current["blocks"] if current else intro).append(text_of(c).strip())
        else:
            if current is None:
                continue
            src = text_of(c).rstrip()
            if src.startswith("!"):                  # a shell command of the notebook
                current["blocks"].append("```bash\n" + src[1:] + "\n```")
            else:
                current["blocks"].append("```python\n" + src + "\n```")
            current.setdefault("cells", []).append(d)
            current["blocks"].extend(outputs_md(d, current["number"],
                                                current.setdefault("counter", [0]), replace))

    names = []
    for p in pages:
        name = f"{p['number']:02d}_{slug(p['title'])}"
        names.append(name)
        with open(os.path.join(OUT, name + ".md"), "w") as fh:
            fh.write(f"# {p['number']}. {p['title']}\n\n")
            fh.write("\n\n".join(p["blocks"]) + "\n")
    with open(os.path.join(OUT, "pages.txt"), "w") as fh:
        fh.write("\n".join(names) + "\n")
    print(f"{len(pages)} pages, {len(os.listdir(FIG))} figures in {OUT}")


if __name__ == "__main__":
    main()
