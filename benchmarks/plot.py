"""Render measured JSON results as standalone PNG and SVG figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot_report(report: dict, destination: str | Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    groups = list(dict.fromkeys((c["graph"]["family"], c["alpha"]) for c in report["cases"]))
    if not groups:
        raise ValueError("report contains no cases")
    ncols = min(2, len(groups))
    nrows = (len(groups)+ncols-1)//ncols
    colors = {"python": "#555555", "cython_t1": "#087f8c", "cython_t2": "#4169a3", "cython_t4": "#9b59b6",
              "cython_t8": "#854da8", "scipy": "#d9822b", "networkx": "#b53d46", "igraph": "#66812d",
              "direct": "#a06a50", "gmres": "#d2a321"}
    labels = {"python": "Plain Python", "cython_t1": "Cython (1 thread)", "cython_t2": "Cython (2 threads)",
              "cython_t4": "Cython (4 threads)", "cython_t8": "Cython (8 threads)",
              "scipy": "SciPy power", "networkx": "NetworkX API", "igraph": "igraph PRPACK",
              "direct": "Sparse direct", "gmres": "GMRES"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6*ncols, 4.4*nrows), squeeze=False)
    for ax, (family, alpha) in zip(axes.flat, groups):
        cases = sorted((c for c in report["cases"] if c["graph"]["family"] == family and c["alpha"] == alpha), key=lambda c:c["graph"]["n"])
        names = list(dict.fromkeys(name for c in cases for name in c["methods"]))
        for name in names:
            valid = [c for c in cases if c["methods"].get(name, {}).get("status") == "ok"]
            if not valid:
                continue
            x = np.array([c["graph"]["n"] for c in valid])
            summaries = [c["methods"][name]["summary"] for c in valid]
            median = np.array([s["median_seconds"]*1000 for s in summaries])
            lower = np.array([s["q25_seconds"]*1000 for s in summaries])
            upper = np.array([s["q75_seconds"]*1000 for s in summaries])
            ax.errorbar(x, median, yerr=[median-lower,upper-median], marker="o", capsize=3,
                        label=labels.get(name,name), color=colors.get(name), linewidth=1.8)
        ax.set(xscale="log", yscale="log", xlabel="Vertices", ylabel="Public API wall time (ms)", title=f"{family} | alpha = {alpha:g}")
        ax.grid(True, which="major", alpha=0.18)
        ax.legend(frameon=False, fontsize=8)
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    config = report["config"]
    fig.suptitle("PageRank: measured API timings", fontsize=18, x=0.04, ha="left")
    fig.text(0.04, 0.015,
        f"Float64 | target L1 bound <= {config['tol']:.0e} | {config['repeats']} repeats, {config['warmups']} warmups | median with IQR\n"
        "Prepared graph inputs. NetworkX's internal sparse conversion is included. Only residual-validated runs are shown.",
        fontsize=9, color="#444444", va="bottom")
    fig.tight_layout(rect=(0,0.075,1,0.95))
    paths = [output/"api_timings.png", output/"api_timings.svg"]
    for path in paths:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6*ncols, 3.7*nrows), squeeze=False)
    for ax, (family, alpha) in zip(axes.flat, groups):
        cases = sorted((c for c in report["cases"] if c["graph"]["family"] == family and c["alpha"] == alpha), key=lambda c:c["graph"]["n"])
        names = list(dict.fromkeys(name for c in cases for name in c["methods"] if name.startswith("cython")))
        for name in names:
            valid = [c for c in cases if c["methods"].get(name,{}).get("status") == "ok" and c["methods"].get("scipy",{}).get("status") == "ok"]
            if not valid:
                continue
            x = [c["graph"]["n"] for c in valid]
            ratio = [c["methods"]["scipy"]["summary"]["median_seconds"]/c["methods"][name]["summary"]["median_seconds"] for c in valid]
            ax.plot(x,ratio,marker="o",label=labels.get(name,name),color=colors.get(name),linewidth=1.8)
        ax.axhline(1,color="#888888",linestyle="--",linewidth=1)
        ax.set(xscale="log", xlabel="Vertices", ylabel="SciPy time / Cython time", title=f"{family} | alpha = {alpha:g}")
        ax.grid(True,alpha=.18)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False,fontsize=8)
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    fig.suptitle("Speed relative to SciPy power iteration",fontsize=18,x=.04,ha="left")
    fig.text(.04,.015,"Ratio of public API medians. A value below 1 means Cython was slower. This is not a confidence interval.",fontsize=9,color="#444444")
    fig.tight_layout(rect=(0,.05,1,.95))
    more = [output/"speedup_vs_scipy.png",output/"speedup_vs_scipy.svg"]
    for path in more:
        fig.savefig(path,dpi=180,bbox_inches="tight")
    plt.close(fig)
    return paths+more


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    report=json.loads(args.results.read_text(encoding="utf-8"))
    for path in plot_report(report,args.output or args.results.parent):
        print(path)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
