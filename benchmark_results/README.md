# 実測データ

測定日: 2026年10月23日（JST）。すべて `float64`、絶対L1誤差目標 `1e-10`。各手法・条件でwarmup 2回、本計測7回を実施した。

`results.json` は条件・生の全試行・精度・前処理・環境・実行順を含む。`summary.csv` はその集計である。条件と検査済み試行数は [検証記録](../docs/validation.md)、数値の解釈は [Qiita記事](../docs/qiita_article.md)、測定方法は [benchmarking.md](../docs/benchmarking.md) にある。

| ディレクトリ | 条件 | 主な比較対象 |
|---|---|---|
| `main` | 1千 / 1万 / 10万頂点、4種類の合成グラフ、alpha=0.85 | Python、Cython 1/2/4、SciPy、NetworkX、igraph |
| `stanford` | 実Webグラフ、281,903頂点・2,312,497辺、alpha=0.85 | Cython 1/2/4、SciPy、NetworkX、igraph |
| `ablation` | 10万頂点、3種類、alpha=0.85 | 等行数分割と辺数を考慮した分割、1/4スレッド |
| `high_alpha` | 1千 / 1万頂点、block、alpha=0.99、先頭ブロックにteleportationを集中 | Cython 1/4、SciPy、direct、GMRES、NetworkX、igraph |
| `million` | 100万頂点・15,999,877辺、uniform、alpha=0.85 | Cython 1/2/4/8、SciPy |
| `weighted_personalized` | 1万頂点、dangling、lognormal重み、ランダムなpと独立したq | Cython 1/4、SciPy、NetworkX |

`thread_affinity.json` は速度測定とは別のOpenMP配置診断である。メインスレッドのaffinityだけからworker全体のCPU使用範囲を推定しない。

## 記録と同じ引数で再実行

プロジェクトのルートで実行する。先にREADMEのOpenMP + nativeビルド手順を済ませる。以下のコマンドは既存の同名測定結果を更新するため、元の記録を残す場合は `--output` を別のディレクトリへ変更する。

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

python -m benchmarks.run \
  --sizes 1000 10000 100000 --families uniform skewed block dangling \
  --methods python cython scipy networkx igraph --threads 1 2 4 \
  --alphas 0.85 --repeats 7 --warmups 2 --tol 1e-10 \
  --seed 20261023 --output benchmark_results/main

python -m benchmarks.stanford --download data/web-Stanford.txt.gz \
  --expect-sha256 49bfc0e366c3028dbd84e8557b09287cbd61051bde29d160c277688c727cb679 \
  --methods cython scipy networkx igraph --threads 1 2 4 \
  --alpha 0.85 --repeats 7 --warmups 2 --tol 1e-10 \
  --seed 20261023 --output benchmark_results/stanford

python -m benchmarks.ablation \
  --sizes 100000 --families uniform skewed block --threads 1 4 \
  --alphas 0.85 --repeats 7 --warmups 2 --tol 1e-10 \
  --seed 20261023 --output benchmark_results/ablation

python -m benchmarks.run \
  --sizes 1000 10000 --families block \
  --methods cython scipy direct gmres networkx igraph --threads 1 4 \
  --alphas 0.99 --personalization localized \
  --repeats 7 --warmups 2 --tol 1e-10 --seed 20261023 \
  --output benchmark_results/high_alpha

python -m benchmarks.run \
  --sizes 1000000 --families uniform --methods cython scipy --threads 1 2 4 8 \
  --alphas 0.85 --repeats 7 --warmups 2 --tol 1e-10 \
  --seed 20261023 --output benchmark_results/million

python -m benchmarks.run \
  --sizes 10000 --families dangling --weighted \
  --personalization random --dangling-distribution random \
  --methods cython scipy networkx igraph --threads 1 4 \
  --alphas 0.85 --repeats 7 --warmups 2 --tol 1e-10 \
  --seed 20261023 --output benchmark_results/weighted_personalized

python -m benchmarks.plot benchmark_results/main/results.json
python -m benchmarks.figures
```

## 図

- `figures/benchmark_overview.png` / `.svg`: 実Web・高alpha・100万頂点の比較
- `main/api_timings.png` / `.svg`: 4種類の合成グラフのサイズ別時間
- `main/speedup_vs_scipy.png` / `.svg`: SciPyに対する時間比

エラーバーはIQR。平均、信頼区間、最速試行の値ではない。数値と図は同梱JSONから生成する。外部データセット本体は同梱していない。
