"""Record OpenMP thread placement without running a timing benchmark.

One iteration on a four-node cycle starts the requested OpenMP team.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def affinity():
    return sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None


def thread_statuses():
    root=Path("/proc/self/task")
    if not root.is_dir():
        return {"available":False,"reason":"/proc/self/task is unavailable"}
    statuses=[]
    for task in sorted(root.iterdir(),key=lambda path:int(path.name)):
        try:
            lines=(task/"status").read_text().splitlines()
        except OSError:
            continue
        wanted={"Name","Pid","Tgid","Cpus_allowed_list","Mems_allowed_list"}
        fields={key:value.strip() for line in lines if ":" in line
                for key,value in [line.split(":",1)] if key in wanted}
        fields["task_directory_id"]=task.name
        statuses.append(fields)
    return {"available":True,"threads":statuses}


def openmp_places():
    library=ctypes.util.find_library("gomp")
    if library is None:
        return {"available":False,"reason":"libgomp not found; another OpenMP runtime may be in use"}
    try:
        runtime=ctypes.CDLL(library)
        runtime.omp_get_num_places.restype=ctypes.c_int
        runtime.omp_get_place_num_procs.argtypes=[ctypes.c_int]
        runtime.omp_get_place_num_procs.restype=ctypes.c_int
        runtime.omp_get_place_proc_ids.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_int)]
        runtime.omp_get_proc_bind.restype=ctypes.c_int
        count=runtime.omp_get_num_places()
        places=[]
        for index in range(count):
            length=runtime.omp_get_place_num_procs(index)
            buffer=(ctypes.c_int*length)()
            runtime.omp_get_place_proc_ids(index,buffer)
            places.append({"place":index,"processors":list(buffer)})
        return {"available":True,"runtime_library":library,"num_places":count,
                "proc_bind_enum":runtime.omp_get_proc_bind(),"places":places,
                "note":"OpenMP place processor IDs are reported by the runtime; outside a parallel region."}
    except (AttributeError,OSError) as exc:
        return {"available":False,"reason":f"{type(exc).__name__}: {exc}"}


def collect(threads=4):
    report={"recorded_at_utc":datetime.now(timezone.utc).isoformat(),"requested_threads":threads,
        "environment":{key:os.environ.get(key) for key in
            ("OMP_PROC_BIND","OMP_PLACES","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS")},
        "before_package_import_affinity":affinity(),
        "semantics":"sched_getaffinity(0) is the calling-thread mask, not a union of OpenMP worker masks. This diagnostic runs one tiny solve but measures no speed."}
    from pagerank_lab import Graph,pagerank,build_info
    report["after_package_import_affinity"]=affinity()
    report["build_info"]=build_info()
    report["after_build_info_affinity"]=affinity()
    report["openmp_places"]=openmp_places()
    graph=Graph.from_edges([0,1,2,3],[1,2,3,0],n=4)
    result=pagerank(graph,threads=threads,alpha=.85,tol=1e-10)
    report["tiny_solve"]={"n":graph.n,"m":graph.m,"iterations":result.iterations,
        "residual_l1":result.residual_l1,"converged":result.converged}
    report["after_parallel_solve_affinity"]=affinity()
    report["thread_statuses_after_solve"]=thread_statuses()
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--output",type=Path,default=Path("benchmark_results/thread_affinity.json"))
    args=parser.parse_args(argv)
    if args.threads<1:
        parser.error("threads must be positive")
    report=collect(args.threads)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    print(args.output)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
