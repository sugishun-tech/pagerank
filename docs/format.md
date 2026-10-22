# 入出力形式とコマンド

頂点番号は `0, 1, ..., n-1` とする。隣接行列の行がリンク元、列がリンク先である。重複する辺の重みは合算する。自己ループは許す。重みは有限の非負実数で、重みゼロの辺は遷移確率に寄与しない。

## グラフ

### Sparse NPZ

SciPy の `scipy.sparse.save_npz` が作る正方行列を読み込める。保存するものは正規化前の重み付き隣接行列である。行列の形状により、入辺も出辺もない頂点を保持できる。NumPy の任意オブジェクトや pickle は使わない。

```bash
pagerank-lab generate graph.npz --nodes 10000 --edges 100000 --seed 42
pagerank-lab inspect graph.npz --components
pagerank-lab run graph.npz --output scores.json
```

### Matrix Market

拡張子 `.mtx` で自動判定する。座標形式を推奨する。行列は正方である必要がある。Matrix Market の添字は規格上 1 始まりだが、SciPy が 0 始まりへ変換するため、利用者が手作業で引く必要はない。

```bash
pagerank-lab run graph.mtx --alpha 0.95 --tol 1e-9 --output scores.npz
```

### 辺リスト

CSV の例を示す。

```csv
# n=5
source,target,weight
0,1,2.0
0,2,1.0
1,2,1.0
2,0,1.0
```

この例では頂点 3 と 4 が孤立点として残る。`# n=5` の代わりに `--nodes 5` を指定できる。両方指定した場合は一致が必要である。両方省略した場合は最大の頂点番号から頂点数を推定するため、末尾の孤立点は表現できない。

`source,target` の 2 列なら重みを 1 とする。ヘッダは省略できる。`.csv` 以外の拡張子は空白区切りを既定とする。`--delimiter ';'` のような指定もできる。空行と `#` で始まるコメント行は無視する。文字列ラベル、自動的な番号の付け替え、負の頂点番号は対応しない。文字列ラベルを使う場合は、番号との対応表を利用側で保持する。

```bash
pagerank-lab run edges.txt --nodes 100 --format edges --output scores.csv
```

テキストの読み込みは Python で検証しながら行う。大きなグラフを反復して実験する場合は、一度 NPZ に変換し、その読み込み時間とソルバーの計算時間を分けて測定する。

```python
from pagerank_lab.io import load_graph, save_graph

graph = load_graph("edges.txt", n=100)
save_graph(graph, "graph.npz")
```

## ベクトル

personalization、dangling、初期値 `x0` には長さ `n` の配列を渡す。

| 形式 | 内容 |
|---|---|
| JSON | 数値配列、または `{"scores": [...]}` |
| NPY | 1 次元の数値配列 |
| NPZ | `scores` キー、または `vector` キーの配列 |
| その他のテキスト | `numpy.loadtxt` で読める数値 1 列 |

全要素は有限である必要がある。確率分布としての非負性・総和の検証と正規化はソルバーが行う。NumPy の読み込みでは `allow_pickle=False` を使う。JSON は任意の Python コードを実行しない。

```bash
pagerank-lab run graph.npz \
  --personalization personalization.json \
  --dangling dangling.npy \
  --x0 previous-scores.npz \
  --history --output scores.json
```

`--dangling` の省略時は personalization と同じ分布を使う。両者が独立なモデルを検証する場合は、dangling を明示する。

## 結果

| 拡張子 | 出力 |
|---|---|
| `.json` | `scores` 配列と反復回数、収束フラグ、残差、誤差上界、実行時間など |
| `.npz` | `scores` 数値配列、`metadata` という JSON 文字列 |
| `.csv` | `node,score` の 2 列と、隣接する `.meta.json` |

CSV は頂点番号順であり、スコア順には並べ替えない。CLI の標準出力にはスコア上位の頂点を別途表示する。表示件数は `--top` で指定でき、同点なら小さい頂点番号を優先する。これは表示規則であり、同点順位に数学的な順序を追加するものではない。

`--history` を省略すると反復ごとの履歴を要求しない。べき乗法 (`cython` / `scipy`) で誤差目標を満たしていない結果は `converged=false` として扱い、CLI は終了コード 2 を返す。結果ファイルを指定していれば、未収束時も診断できるよう結果を保存する。`direct` / `gmres` が計算失敗や精度不足を検出した場合は例外を返すため、CLI はエラーメッセージと終了コード 2 を返し、結果ファイルは作らない。

`tol` の意味と浮動小数点計算における誤差上界の限界は README と記事を参照する。特に float32 で float64 と同じ厳しい許容誤差を機械的に指定しない。

## 合成グラフ

```bash
pagerank-lab generate uniform.npz --kind uniform --nodes 100000 --edges 1000000 --seed 42
pagerank-lab generate skewed.npz --kind skewed --nodes 100000 --edges 1000000 --exponent 1.2 --seed 42
pagerank-lab generate blocks.npz --kind block --nodes 100000 --edges 1000000 --blocks 8 --seed 42
```

`--edges` は重複を合算する前に抽出する辺の数である。実際の非零要素数はコマンドの出力に記録する。`uniform` は独立な端点抽出、`skewed` は Zipf 型のリンク先人気度、`block` は同一ブロック内または全体からのリンク先抽出である。`skewed` は preferential attachment の実装ではなく、実測 Web グラフの代用とも断定しない。

`--dangling-fraction` は出辺を持たせない末尾の頂点の割合である。他の頂点も辺の抽出結果次第では dangling になる。`--weighted` で各抽出辺に `[0.1, 2.0)` の一様乱数の重みを付けられる。

## ビルドオプション

既定ビルドは OpenMP を要求しない。C コンパイラは必要である。

```bash
python -m pip install ".[all]"
```

Linux で OpenMP を有効にする例を示す。

```bash
PAGERANK_OPENMP=1 python -m pip install --no-cache-dir --force-reinstall --no-deps .
pagerank-lab run graph.npz --threads 4
```

並列処理の分担は `--partition balanced` が既定である。入辺数と行数をもとに負荷を分担する。`--partition rows` は行数を均等に分ける比較用の指定で、次数が偏ったグラフでは負荷が偏る場合がある。どちらを使ったかは結果のメタデータに記録する。

ローカル CPU 向けのコード生成は `PAGERANK_NATIVE=1` で明示的に要求する。これは他の CPU へ配布する wheel には適さない。Windows/MSVC ではこのフラグを無視し警告する。既定では `fast-math` を使わない。

```bash
PAGERANK_OPENMP=1 PAGERANK_NATIVE=1 python -m pip install --no-cache-dir --force-reinstall --no-deps .
```

macOS で OpenMP を使う場合は別途 libomp が必要で、`PAGERANK_OPENMP_PREFIX` をそのインストール先に設定できる。Windows の OpenMP ビルドは MSVC の `/openmp` を利用する。OpenMP のスケジューリングやスレッド起動コストにより、スレッドを増やしても高速化しない場合がある。

`PAGERANK_ANNOTATE=1` を設定すると Cython の注釈付き HTML を生成する。

