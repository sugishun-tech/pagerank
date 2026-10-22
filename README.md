# PageRank Lab

CythonによるPageRank実装と、数値線形代数としての検証・性能比較をまとめた研究用Pythonプロジェクト。

**Qiita記事:** [ページランクに関する検証](docs/qiita_article.md)

## できること

- 重み付き有向グラフ、重複辺の合算、自己ループ、孤立点、danglingノード
- Personalized PageRank、独立したdangling分布、warm start、alpha sweep、複数personalization
- CythonのCSR pullカーネル、OpenMP、辺数を考慮した負荷分散、float32 / float64、int32 / int64
- SciPyべき乗法、疎LU + Sherman–Morrison、matrix-free GMRES
- L1残差と事後誤差上界、反復履歴、計算時間、非収束の明示
- Google行列の線形作用素、固有値、有理数厳密解、Neumann級数の経路長別寄与
- alpha・personalization・辺重みの感度解析、随伴法による全格納辺の勾配
- 同点を考慮した順位、Spearman相関、top-k一致率
- CSV / NPZ / Matrix Market入出力、CLI、合成グラフ、再現可能なベンチマーク

数学的仕様は [docs/mathematics.md](docs/mathematics.md)、API一覧は [docs/api.md](docs/api.md) を参照。

## インストール

Python 3.10以上とCコンパイラが必要。ZIPを展開し、`README.md` と `pyproject.toml` がある `pagerank-lab` ディレクトリへ移動して実行する。PyPIで公開済みのパッケージという前提にはしていない。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[all,igraph]'
```

Windowsでは仮想環境の有効化に `.venv\Scripts\Activate.ps1` を使う。標準ビルドはシリアルで、他のCPUへ配布しやすい設定になっている。

インストール後は、同梱の小さなグラフですぐに動作確認できる。

```bash
pagerank-lab run examples/toy.csv --output scores.json
python examples/basic.py
```

Linux / GCC・CPython 3.12で、実測と同じOpenMP・nativeビルドにする例を示す。`requirements-benchmark.txt` は記録した直接依存のバージョン指定であり、全OS・Python版に対応する完全な依存ロックではない。通常の利用では上のインストール手順を使える。

```bash
python -m pip install -r requirements-benchmark.txt
CC=gcc LDSHARED='gcc -shared' PAGERANK_OPENMP=1 PAGERANK_NATIVE=1 \
  python -m pip install -e . --no-build-isolation
python -c 'from pagerank_lab import build_info; print(build_info())'
```

`PAGERANK_NATIVE=1` はビルド機のCPU向け。生成したバイナリを異なるCPUへ配布する設定ではない。macOS、Windows、ビルドし直す場合の手順は [docs/build.md](docs/build.md) に記載した。

## Pythonから使う

```python
from pagerank_lab import Graph, pagerank

graph = Graph.from_edges(
    sources=[0, 0, 1, 2],
    targets=[1, 2, 2, 0],
    weights=[2, 1, 1, 1],
    n=4,  # 辺に現れないノード3も保持
)

result = pagerank(graph, alpha=0.85, tol=1e-10, history=True)
print(result.scores)
print(result.topk(3))
print(result.error_bound_l1, result.iterations)

# OpenMPを有効にしてビルドした場合だけ並列版も実行
from pagerank_lab import build_info
if build_info()["openmp"]:
    parallel = pagerank(graph, threads=4, partition="balanced")
```

`tol` は頂点数によらない**絶対L1誤差の目標**。返却候補の `||F(x)-x||₁/(1-alpha)` をfloat64で検査する。数式は厳密演算の上界であり、この評価自体は区間演算による証明付き計算ではない。

`float32` でも最後の検査は元のfloat64グラフに対して行う。達成できない精度を要求した場合は失敗する。例えば `dtype="float32", tol=1e-6` から検討できるが、すべてのグラフで達成できるという意味ではない。

```python
from pagerank_lab import exact_pagerank, alpha_derivative, edge_gradient

small = Graph.from_edges([0], [1], n=2)
print(exact_pagerank(small))  # Matrix([[20/57], [37/57]])
print(alpha_derivative(small))

# objective.T @ x の、生の辺重みに関する勾配
gradient = edge_gradient(graph, objective=[1, 0, 0, 0])
```

`Graph` は構築後に変更しない。辺を変更する場合は新しい `Graph` を作る。保存・再読み込みは数値ノードIDを使い、NetworkXから取り込んだ任意ラベルは自動保存しない。

## CLI

```bash
pagerank-lab generate graph.npz --kind skewed \
  --nodes 100000 --edges 1000000 --seed 42
pagerank-lab inspect graph.npz --components
pagerank-lab run graph.npz --threads 1 --tol 1e-10 \
  --history --output scores.json
pagerank-lab run examples/toy.csv --personalization examples/personalization.json \
  --method scipy --output scores.csv
```

OpenMPビルドでは `--threads 4 --partition balanced` のように並列実行できる。

べき乗法が未収束なら、CLIは診断と最終候補を保存して終了コード2を返す。direct / GMRES内部の計算失敗では例外を表示し、結果ファイルは作らない。入力の仕様と終了コードは [docs/format.md](docs/format.md) にある。

## 比較を再現する

以下はOpenMPを有効にしたビルドを前提とする。標準のシリアルビルドでは `--threads 1` として実行する。ベンチマークはインストール済みCLIのサブコマンドではなく、ソースディレクトリから実行するモジュールである。

```bash
OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
  python -m benchmarks.run \
  --sizes 1000 10000 100000 --families uniform skewed block dangling \
  --methods python cython scipy networkx igraph --threads 1 2 4 \
  --repeats 7 --warmups 2 --output benchmark_results/main

python -m benchmarks.plot benchmark_results/main/results.json
```

実Webグラフの測定:

```bash
OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
  python -m benchmarks.stanford --download data/web-Stanford.txt.gz \
  --methods cython scipy networkx igraph --threads 1 2 4 \
  --repeats 7 --warmups 2 --output benchmark_results/stanford
```

SNAPのデータ本体はこのプロジェクトに含めていない。ダウンロード元、元データと正規化前隣接行列のSHA-256、IDの再番号付けを記録する。

比較の基準は公開APIのwall time。自作カーネル内部時間、グラフ構築時間、出力の精度を別に記録する。NetworkXの公開APIには毎回の内部疎行列変換が含まれるため、対NetworkX倍率をカーネルだけの高速化率とは呼ばない。既定で7回の本計測と2回のwarmupを行い、生データ、中央値、IQR、MAD、実行順、環境を保存する。

測定方法と補助実験は [docs/benchmarking.md](docs/benchmarking.md)、実測結果の解説は記事に記載した。性能はCPU・グラフ・精度・スレッド数に依存する。

## テストと構成

```bash
python -m pytest
python examples/basic.py
python examples/research.py
```

| パス | 内容 |
|---|---|
| `src/pagerank_lab/` | Python API、Cythonカーネル、CLI |
| `tests/` | 独立解との照合、有限差分、疎行列入力検証、IO、ベンチマーク条件 |
| `benchmarks/` | 再実行用の測定・描画スクリプト |
| `benchmark_results/` | 実際に実行した計測のJSON / CSV / 図 |
| `docs/qiita_article.md` | Qiita用記事全文 |
| `docs/mathematics.md` | 数式・証明・感度解析の仕様 |
| `docs/validation.md` | 実行済みの検証と未実行のプラットフォーム |
| `.github/workflows/ci.yml` | GitHub Actions用テスト設定 |

GPU、分散計算、局所push法、動的グラフの増分更新、区間演算は実装範囲に含めていない。複数personalizationのAPIはグラフを再利用する逐次バッチであり、専用の並列SpMM実装ではない。

## ライセンス

コードはMIT。外部論文・データセットの権利は各提供者に帰属する。[参考資料](docs/references.md) を参照。
