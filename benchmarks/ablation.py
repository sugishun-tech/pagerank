"""Isolate equal-row versus balanced-work Cython partitioning.

Run from the checkout: python -m benchmarks.ablation --help
The recorded work units are nnz + rows, an estimator rather than measured cycles.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .run import (Config, _peak_rss, _timed, environment, independent_check,
                  make_adjacency, summarize_samples, write_report)


def run_ablation(config: Config, progress=None) -> dict:
    from pagerank_lab import Graph, pagerank

    config.validate()
    report = {
        "schema_version": 1, "experiment": "partition_ablation",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {**asdict(config), "partition_strategies": ["rows", "balanced"]},
        "environment": environment(), "cases": [],
        "timing_notes": [
            "Same Cython binary, graph, float64 arithmetic, initial vector and target; only contiguous destination-row partition boundaries differ.",
            "Public API wall time includes wrapper/allocation/final residual; internal solver time is reported separately.",
            "Graph, input arrays and every partition are prepared and separately timed before warmups.",
            "Work units estimate one unit per nonzero and one per destination row. They are not measured instruction counts.",
            "A single destination row is indivisible; one exceptionally large in-degree may still limit parallel balance.",
            "The one-thread run is a control: the kernel uses its serial path, so partition policy should not change the numeric workload.",
            "All warmup times are discarded; measured method order is randomized and stored; each result receives an independent residual check.",
            "preprocess_plus_median_seconds is a sum of separate measurements, not an independently timed cold run.",
            "RSS is the process-wide cumulative high-water mark, not method-specific memory.",
        ],
    }
    for family_index, family in enumerate(config.families):
        for n in config.sizes:
            seed = config.seed + family_index*1000003 + n
            adjacency, meta = make_adjacency(n,config.degree,family,seed,config.weighted)
            graph, graph_seconds = _timed(lambda: Graph.from_scipy(adjacency))
            _, prepare_seconds = _timed(lambda: graph.prepare("float64"))
            p = np.full(n,1.0/n)
            x0 = p.copy()
            costs = graph.incoming.indptr.astype(np.int64) + np.arange(n+1,dtype=np.int64)
            partitions = {}
            for strategy in ("rows","balanced"):
                for threads in config.threads:
                    cuts, seconds = _timed(lambda: graph.partition(threads,strategy))
                    work = np.diff(costs[cuts])
                    partitions[(strategy,threads)] = {
                        "prepare_seconds": seconds,"cuts": cuts.tolist(),
                        "estimated_work_units": work.tolist(),
                        "max_to_mean_work": float(work.max()/work.mean()),
                    }
            for alpha in config.alphas:
                if progress:
                    progress(f"ablation {family}: n={n:,} m={graph.m:,} alpha={alpha}")
                entries, runners = {}, {}
                for strategy in ("rows","balanced"):
                    for threads in config.threads:
                        name=f"{strategy}_t{threads}"
                        part=partitions[(strategy,threads)]
                        entries[name]={"method":"cython","partition":strategy,"threads":threads,
                            "samples":[],"partition_details":part,
                            "preprocess_seconds":graph_seconds+prepare_seconds+part["prepare_seconds"],
                            "scope":"Cython public API on prepared Graph and prepared partition"}
                        def call(strategy=strategy,threads=threads):
                            result=pagerank(graph,alpha=alpha,personalization=p,dangling=p,
                                x0=x0,tol=config.tol,max_iter=config.max_iter,method="cython",
                                dtype="float64",threads=threads,partition=strategy,history=False)
                            return result.scores,{"iterations":int(result.iterations),
                                "solver_seconds":float(result.elapsed_seconds),
                                "reported_converged":bool(result.converged)}
                        runners[name]=call
                disabled=set()
                for name,call in runners.items():
                    try:
                        for _ in range(config.warmups):
                            call()
                    except Exception as exc:
                        entries[name].update(status="failed",reason=f"{type(exc).__name__}: {exc}")
                        disabled.add(name)
                active=[name for name in runners if name not in disabled]
                order_rng=np.random.default_rng(seed+8051)
                orders=[]
                for repeat in range(config.repeats):
                    order=order_rng.permutation(active).tolist()
                    orders.append(order)
                    for name in order:
                        try:
                            (scores,details),wall=_timed(runners[name])
                            check=independent_check(graph,scores,alpha,p,p,config.tol)
                            sample={"repeat":repeat,"wall_seconds":wall,**details,**check}
                            if not details["reported_converged"]:
                                sample.update(valid=False,reason="solver reported nonconvergence")
                            entries[name]["samples"].append(sample)
                        except Exception as exc:
                            entries[name]["samples"].append({"repeat":repeat,"valid":False,"error":f"{type(exc).__name__}: {exc}"})
                ratios={}
                for name,item in entries.items():
                    if "status" not in item:
                        item["summary"]=summarize_samples(item["samples"])
                        item["status"]="ok" if item["summary"]["valid"] else "failed"
                        median=item["summary"].get("median_seconds")
                        if median is not None:
                            item["summary"]["preprocess_plus_median_seconds"]=item["preprocess_seconds"]+median
                    if progress:
                        median=item.get("summary",{}).get("median_seconds")
                        progress(f"  {name}: {item['status']}" + (f" {median*1000:.3f} ms" if median is not None else ""))
                for threads in config.threads:
                    row=entries[f"rows_t{threads}"]
                    balanced=entries[f"balanced_t{threads}"]
                    if row["status"]==balanced["status"]=="ok":
                        ratios[str(threads)]={
                            "api_rows_over_balanced":row["summary"]["median_seconds"]/balanced["summary"]["median_seconds"],
                            "solver_rows_over_balanced":row["summary"]["solver_median_seconds"]/balanced["summary"]["solver_median_seconds"],
                        }
                report["cases"].append({
                    "graph":{**meta,"dangling_count":int(graph.dangling.sum())},
                    "alpha":alpha,"tol_l1":config.tol,"dtype":"float64",
                    "personalization":"uniform","dangling_distribution":"personalization",
                    "preprocessing":{"pagerank_graph_from_canonical_csr_seconds":graph_seconds,
                        "cython_prepare_seconds":prepare_seconds},
                    "execution_orders":orders,"methods":entries,
                    "speedup_ratios":ratios,"process_peak_rss_bytes_cumulative":_peak_rss(),
                })
    report["finished_at_utc"]=datetime.now(timezone.utc).isoformat()
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes",nargs="+",type=int,default=[100000])
    parser.add_argument("--families",nargs="+",default=["uniform","skewed","block"])
    parser.add_argument("--threads",nargs="+",type=int,default=[1,4])
    parser.add_argument("--alphas",nargs="+",type=float,default=[.85])
    parser.add_argument("--degree",type=int,default=16)
    parser.add_argument("--repeats",type=int,default=7)
    parser.add_argument("--warmups",type=int,default=2)
    parser.add_argument("--seed",type=int,default=20261023)
    parser.add_argument("--tol",type=float,default=1e-10)
    parser.add_argument("--max-iter",type=int,default=10000)
    parser.add_argument("--weighted",action="store_true")
    parser.add_argument("--output",type=Path,default=Path("benchmark_results/ablation"))
    values=vars(parser.parse_args(argv)).copy()
    output=values.pop("output")
    for field in ("sizes","families","threads","alphas"):
        values[field]=tuple(values[field])
    try:
        report=run_ablation(Config(**values,methods=("cython",)),progress=lambda text:print(text,flush=True))
    except ValueError as exc:
        parser.error(str(exc))
    for path in write_report(report,output):
        print(path)
    return 1 if any(item["status"]=="failed" for case in report["cases"] for item in case["methods"].values()) else 0


if __name__=="__main__":
    raise SystemExit(main())
