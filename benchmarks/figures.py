"""Create a compact article figure from the committed measured JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/figures"))
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11,
        "axes.spines.top":False, "axes.spines.right":False, "axes.spines.left":False})
    stanford = json.loads((args.root/"stanford/results.json").read_text())["cases"][0]
    slow = json.loads((args.root/"high_alpha/results.json").read_text())["cases"][-1]
    million = json.loads((args.root/"million/results.json").read_text())["cases"][0]
    groups = [
        (stanford, "Observed web graph", "Stanford | 281,903 vertices | alpha = 0.85",
         ["cython_t1","cython_t2","cython_t4","scipy","igraph","networkx"]),
        (slow, "A different solver can win", "8 blocks | 10,000 vertices | alpha = 0.99",
         ["cython_t1","cython_t4","scipy","gmres","igraph","networkx"]),
        (million, "Scaling to one million vertices", "Uniform | 15,999,877 edges | alpha = 0.85",
         ["cython_t1","cython_t2","cython_t4","cython_t8","scipy"]),
    ]
    names={"cython_t1":"Cython / 1 thread","cython_t2":"Cython / 2 threads",
        "cython_t4":"Cython / 4 threads","cython_t8":"Cython / 8 threads",
        "scipy":"SciPy power","networkx":"NetworkX API","igraph":"igraph PRPACK","gmres":"SciPy GMRES"}
    colors={"cython_t1":"#77a6ba","cython_t2":"#4387a0","cython_t4":"#096b85","cython_t8":"#034b62",
        "scipy":"#cf8c3f","networkx":"#8b939d","igraph":"#819558","gmres":"#815ca4"}
    fig,axes=plt.subplots(1,3,figsize=(16.8,5.1),gridspec_kw={"wspace":.8})
    for ax,(case,title,subtitle,methods) in zip(axes,groups):
        summaries=[case["methods"][k]["summary"] for k in methods]
        vals=np.array([s["median_seconds"]*1000 for s in summaries])
        lower=np.array([s["q25_seconds"]*1000 for s in summaries])
        upper=np.array([s["q75_seconds"]*1000 for s in summaries])
        y=np.arange(len(methods))
        ax.barh(y,vals,height=.55,color=[colors[m] for m in methods],zorder=2)
        ax.errorbar(vals,y,xerr=[vals-lower,upper-vals],fmt="none",ecolor="#23303d",capsize=3,zorder=3)
        ax.set_yticks(y,[names[m] for m in methods],fontsize=10)
        ax.invert_yaxis()
        for yi,v,u in zip(y,vals,upper):
            ax.text(u+max(upper)*.018,yi,f"{v:,.1f}",va="center",fontsize=9,color="#243543")
        ax.set_xlim(0,max(upper)*1.27)
        ax.set_xlabel("Public API time (ms)",fontsize=10,labelpad=9)
        ax.tick_params(axis="y",length=0,pad=8)
        ax.grid(axis="x",alpha=.15,zorder=0)
        ax.set_title(title,loc="left",fontsize=13,fontweight="bold",pad=31)
        ax.text(0,1.04,subtitle,transform=ax.transAxes,fontsize=9,color="#53616e")
    fig.suptitle("PageRank: implementation speed and solver choice",x=.02,ha="left",fontsize=20,fontweight="bold",y=.99)
    fig.text(.02,.035,"Measured on AMD EPYC 9V74 / GCC 13.3 / Cython 3.3.  Float64, target L1 residual bound <= 1e-10.\n"
        "Bars: median of 7 runs; whiskers: IQR. Prepared graph inputs; NetworkX internal sparse conversion is included. Lower is faster.",
        fontsize=9,color="#435361",linespacing=1.55)
    fig.subplots_adjust(top=.76,bottom=.22,left=.1,right=.99)
    args.output.mkdir(parents=True,exist_ok=True)
    for ext in ("png","svg"):
        path=args.output/f"benchmark_overview.{ext}"
        fig.savefig(path,dpi=180,bbox_inches="tight",facecolor="white")
        print(path)
    plt.close(fig)


if __name__=="__main__":
    main()
