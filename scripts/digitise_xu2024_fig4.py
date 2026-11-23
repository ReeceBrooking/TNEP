"""Digitise the experimental and simulated water spectra of Xu et al. 2024 Fig. 4 (arXiv 2312.05233v2, page 7)
from the PDF's vector paths — the curves are embedded as line/Bézier paths, so this is exact to the figure's
resolution, not a raster trace. Brooker et al. 1989 and Morawietz et al. 2018 publish no numerical spectra;
these are their curves AS ADAPTED (reduced, integral-normalised, vertically offset) by Xu et al.

    python scripts/digitise_xu2024_fig4.py <paper.pdf>   (needs pymupdf; writes datasets/raman_water_xu2024_fig4/*.csv)

Axes (PDF points): x 354.3 → 0 cm⁻¹, 46.52 pt per 1000 cm⁻¹ (the break at 2500 cm⁻¹ keeps the scale); y spans 15 units per
panel. The right-hand panels of (b)/(c) are drawn ×25 / ×180 smaller; curves are stacked with a vertical offset, which
is removed here as the minimum of the curve's right-hand segment (2500–4000 cm⁻¹, where the spectra vanish).
Output columns: freq_cm, intensity (offset removed, right panel re-scaled, Xu et al.'s normalisation otherwise kept)."""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "datasets", "raman_water_xu2024_fig4")
X0, PT_PER_1000 = 354.3, 46.52
PANELS = {"ir": (337.3, 222.9, 1.0), "aniso": (451.7, 337.3, 25.0), "iso": (566.1, 451.7, 180.0)}   # (y of 0, y of 15, right-panel factor)
# drawing index → (name, panel) on page 7 of 2312.05233v2; L/R = left/right of the axis break
CURVES = {15: ("downing", "ir"), 16: ("max", "ir"), 17: ("tnep", "ir"), 18: ("dp", "ir"), 19: ("mbpol", "ir"), 20: ("tip3p", "ir"),
          50: ("brooker", "aniso"), 51: ("brooker", "aniso"), 52: ("morawietz", "aniso"), 53: ("morawietz", "aniso"),
          54: ("tnep", "aniso"), 55: ("tnep", "aniso"), 56: ("teann", "aniso"), 57: ("teann", "aniso"),
          58: ("dp", "aniso"), 59: ("dp", "aniso"), 60: ("mbpol", "aniso"), 61: ("mbpol", "aniso"),
          92: ("brooker", "iso"), 93: ("brooker", "iso"), 94: ("morawietz", "iso"), 95: ("morawietz", "iso"),
          96: ("tnep", "iso"), 97: ("tnep", "iso"), 98: ("teann", "iso"), 99: ("teann", "iso")}


def main(pdf):
    import pymupdf
    dr = pymupdf.open(pdf)[6].get_drawings()
    series = {}
    for i, (name, panel) in CURVES.items():
        pts = []
        for it in dr[i]["items"]:
            if it[0] == "l":
                pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
            elif it[0] == "c":
                pts += [(it[1].x, it[1].y), (it[4].x, it[4].y)]
        series.setdefault((name, panel), []).append(np.array(pts))
    os.makedirs(OUT, exist_ok=True)
    for (name, panel), segs in series.items():
        y0, y15, fac = PANELS[panel]
        P = np.concatenate(segs); P = P[np.argsort(P[:, 0])]
        f = (P[:, 0] - X0) / PT_PER_1000 * 1000.0; y = 15.0 * (y0 - P[:, 1]) / (y0 - y15)
        keep = (f >= 0.0) & (f <= 4000.0); f, y = f[keep], y[keep]       # paths run on under the clip box
        right = f > 2500.0
        base = y[right].min() if right.any() else y.min()
        y = y - base; y[right] *= fac
        f, idx = np.unique(np.round(f, 2), return_index=True); y = y[idx]
        np.savetxt(os.path.join(OUT, f"{name}_{panel}.csv"), np.column_stack([f, y]), delimiter=",", fmt="%.2f,%.5f",
                   header=f"{name} {panel} spectrum digitised from Xu et al. 2024 Fig. 4 vector paths (arXiv 2312.05233v2); "
                          f"offset {base:.3f} removed; x>2500 re-scaled by {fac:g}\nfreq_cm,intensity", comments="# ")
        print(f"{name:10s} {panel:6s} {len(f):5d} pts  {f.min():6.0f}–{f.max():4.0f} cm⁻¹  offset {base:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
