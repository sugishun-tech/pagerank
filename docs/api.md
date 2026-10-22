# API一覧

## グラフ

`A[i,j]` をノード `i` から `j` への非負の生の辺重みとする。ノードIDは `0,...,n-1`。0頂点のグラフは扱わない。辺がない場合も `n>=1` を明示できる。

| API | 用途 |
|---|---|
| `Graph.from_edges(sources, targets, weights=None, n=None, index_dtype="auto")` | 辺リストから作成。重複辺は合算、重み0は除去 |
| `Graph.from_scipy(A, index_dtype="auto")` | 非負の正方疎行列・密行列から作成 |
| `Graph.from_networkx(G, nodelist=None, weight="weight")` | 指定された順のラベルを `graph.labels` に保持 |
| `graph.reverse()` | 辺の方向を反転して、改めて行正規化 |
| `graph.permute(order)` | ノード順序の置換。personalization等も同じ順に変更する |
| `graph.statistics(components=False)` | 頂点・辺・dangling・成分・疎配列の格納量 |
| `graph.prepare(dtype="float64")` | Cython用連続配列を準備し再利用 |
| `graph.partition(threads, strategy="balanced")` | 入次数と行の仕事量を考慮した分割を準備 |

`adjacency` は生の重み、`transition` はdangling行を0とした行正規化行列、`incoming` はその転置のCSR、`dangling` はbool配列。構造や配列を直接交換する操作は非対応。

## 計算

```python
pagerank(graph, alpha=0.85, personalization=None, dangling=None,
         tol=1e-10, max_iter=1000, method="cython", dtype="float64",
         threads=1, x0=None, history=False, raise_on_failure=True,
         partition="balanced")
```

全ての省略可能引数はキーワード引数。`0<=alpha<1`、`tol>0`、`max_iter>=1`、`threads>=1` を要求する。分布と初期値は有限・非負・正の総和を持つ長さnのベクトルで、内部で正規化する。`personalization=None` と `x0=None` は一様分布、`dangling=None` はpersonalizationと同じ分布を意味する。

`method` は `cython` / `scipy` / `direct` / `gmres`。`threads>1` はOpenMPを有効にしたCythonだけで使用可能。`dtype` はfloat32 / float64、directとGMRESはfloat64のみ。`partition="rows"` は等行数分割の比較用。

### 結果

`PageRankResult` の `scores` は読み取り専用。以下のフィールドがある。

| フィールド | 意味 |
|---|---|
| `scores` | PageRankベクトル |
| `iterations` | べき乗反復数。alpha=0・線形ソルバは0 |
| `converged` | 最終残差によるL1目標を満たし、成分が非負か |
| `residual_l1` | `||F(scores)-scores||₁` のfloat64評価 |
| `error_bound_l1` | `residual_l1/(1-alpha)` |
| `delta_l1` | 最後のべき乗反復差。alpha=0・線形ソルバは0 |
| `elapsed_seconds` | ソルバ内部時間。グラフ構築・配列準備・引数検証・最終残差評価を除外。directのLU分解はソルバ内部として含む |
| `history` | 要求した場合のL1反復差配列。通常はNone |
| `alpha`, `tol`, `method`, `dtype`, `threads`, `partition` | 計算条件 |

`topk(k)` は `(node_id, score)` のリストを返し、同点はID順。`as_dict(labels=None)` は辞書、`metadata()` はスコアと履歴以外の情報を返す。

べき乗法の未収束時は `ConvergenceError` が送出され、`.result` に診断を保持する。`raise_on_failure=False` なら最終候補を返す。direct / GMRESの失敗にはこの抑制を適用しない。

### 繰り返し利用

以下の `graph` はREADMEで作成した4頂点のグラフとする。

```python
from pagerank_lab import alpha_sweep, personalized_batch
results = alpha_sweep(graph, [0.5, 0.85, 0.95], warm_start=True)
queries = personalized_batch(graph, [[1, 0, 0, 0], [0, 1, 0, 0]])
```

`personalized_batch` は各行を一つの分布として解く。入力は `queries × n`。グラフは再利用するが、クエリは逐次処理する。

## 解析

以下は `pagerank_lab` から直接importできる。

| API | 結果・用途 |
|---|---|
| `google_operator(graph, ...)` | Google行列のSciPy `LinearOperator` |
| `residual(graph, scores, ...)` | 残差・上界・確率質量・非負性の辞書 |
| `solve_linear(graph, ..., method="direct")` | 疎LUまたはGMRESで解いたスコア配列 |
| `alpha_derivative(graph, ..., scores=None)` | `dx/dalpha` |
| `personalization_derivative(graph, direction, ...)` | 和が0の方向への確率分布の微分 |
| `edge_derivative(graph, source, target, ...)` | 生の集約辺重みへの微分 |
| `edge_gradient(graph, objective, ...)` | `objective.T @ x` の全格納辺への勾配をCSRで返す |
| `exact_pagerank(graph, alpha="17/20", ..., max_n=32)` | SymPyの有理数ベクトル |
| `spectrum(graph, ..., max_n=512)` | 全固有値・subdominant modulus・spectral gap |
| `neumann_series(graph, terms, ...)` | 経路長別寄与、部分和、尾部L1上界 |
| `iteration_bound(alpha, tol, initial_error_bound=2.0)` | 最悪反復数の事前見積もり |
| `rank_scores(scores, method="average")` | 同点処理を指定した順位 |
| `compare_scores(x, y, top_k=None)` | L1 / L∞ / cosine / Spearman / top-k一致率 |

解析系の数値計算はfloat64。有理数計算は保存された生の重みの十進表現を正確な有理数と解釈する。元データをfloat化した時点で失われた精度は復元しない。全固有値計算と有理数計算は小規模向け。感度解析の疎LUはfill-inで大きなメモリを使う場合がある。

詳しい数式と微分の境界条件は [mathematics.md](mathematics.md) に記載した。

## 入出力

入出力関数は `pagerank_lab.io` からimportする。

```python
from pagerank_lab import pagerank
from pagerank_lab.io import load_graph, save_graph, load_vector, save_result

graph = load_graph("examples/toy.csv")
save_graph(graph, "graph.npz")
p = load_vector("examples/personalization.json", n=graph.n)
result = pagerank(graph, personalization=p)
save_result(result, "scores.json")
```

`load_graph` / `save_graph` は辺リスト、SciPy sparse NPZ、Matrix Marketに対応する。`save_result` はJSON、NPZ、CSVを出力でき、CSVではメタデータ用の `.meta.json` も作る。詳細は [format.md](format.md) を参照。
