"""Build docs/gap_dipole_polarisability_models.tex (and .pdf when a LaTeX engine is on PATH) from the Markdown source.
Prose Unicode math symbols become macros so the file compiles under pdflatex or xelatex; math segments are left as written.
Usage: python docs/build_tex.py   (needs pandoc; PDF step uses tectonic or latexmk if found)."""
import re, shutil, subprocess, sys, pathlib
src = pathlib.Path("docs/gap_dipole_polarisability_models.md"); tex = src.with_suffix(".tex")
SUB = {"→": r"$\rightarrow$", "≈": r"$\approx$", "×": r"$\times$", "±": r"$\pm$", "≤": r"$\le$", "·": r"$\cdot$",
       "σ": r"$\sigma$", "μ": r"$\mu$", "α": r"$\alpha$", "φ": r"$\varphi$", "…": r"\ldots{}", "Å": r"\AA{}"}
def fix_text(t):
    for k, v in SUB.items(): t = t.replace(k, v)
    return t
md = src.read_text()
# substitute only outside $...$ / $$...$$ math
parts = re.split(r"(\$\$.*?\$\$|\$[^$\n]*?\$)", md, flags=re.S)
md = "".join(p if p.startswith("$") else fix_text(p) for p in parts)
header = r"""
\usepackage{amsmath,amssymb,booktabs,longtable,array,geometry,microtype}
\geometry{a4paper,margin=22mm}
\setlength{\LTleft}{0pt}\setlength{\LTright}{0pt}
\setcounter{secnumdepth}{0}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
"""
hdr = pathlib.Path("docs/_header.tex"); hdr.write_text(header)
subprocess.run(["pandoc", "-s", "-f", "markdown+pipe_tables+tex_math_dollars", "--columns=100",
                "-V", "fontsize=10pt", "-V", "colorlinks=true", "-H", str(hdr), "-o", str(tex), "-"], input=md, text=True, check=True)
hdr.unlink()
engine = shutil.which("tectonic") or shutil.which("latexmk")
if engine and "--pdf" in sys.argv:
    cmd = [engine, str(tex)] if "tectonic" in engine else [engine, "-xelatex", "-outdir=docs", str(tex)]
    subprocess.run(cmd, check=True)
print("wrote", tex)
