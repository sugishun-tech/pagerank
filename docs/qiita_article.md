# ページランクに関する検証

PageRankをCythonで実装し、Python、SciPy、NetworkX、igraphのPRPACKと比較した。対象はCPU上の疎グラフである。重み付き辺、Personalized PageRank、danglingノードを扱い、実装を `pagerank-lab` というPythonパッケージとCLIにまとめた。

高速化では、Cythonへの移植に加えて、CSRによるpull型計算、更新と収束判定の融合、反復中のメモリ確保の除去、OpenMP、辺数を考慮した負荷分散まで実装した。比較は実測し、出力が同じL1誤差目標を満たすことを確認している。

先に主な結果を書く。Stanfordの実Webグラフでは、公開APIの中央値がCythonの4スレッド版で **235.729 ms**、SciPy版で **728.830 ms** となり、約 **3.09倍** の速度差が出た。一方、収束の遅い別の条件ではGMRESがCythonのべき乗法を上回った。実装の定数倍と、解法の選択は分けて考える必要がある。

## 1. 今回のPageRankの定義

辺の重みを $`W_{ij}\ge0`$ とし、行 $`i`$ から列 $`j`$ へ辺があるものとする。出ていく辺の重みの総和は

```math
s_i=\sum_j W_{ij}
```

である。行正規化した行列 $`H`$ を

```math
H_{ij}=\begin{cases}
W_{ij}/s_i & s_i>0,\\
0 & s_i=0
\end{cases}
```

とする。出ていく辺がないノードがdanglingノードで、その指示ベクトルを $`d`$ とする。

teleportationの分布を $`p`$、danglingからの再分配先を $`q`$ とする。どちらも非負で総和が1の列ベクトルである。既定では $`p`$ は一様分布、$`q=p`$ とするが、それぞれ独立に指定できる。

列ベクトルに作用する遷移行列は

```math
T=H^\mathsf{T}+qd^\mathsf{T}
```

となり、求めるPageRank $`x_*`$ は

```math
x_*=\alpha Tx_*+(1-\alpha)p,\qquad 0\le\alpha<1
```

を満たす。実装では $`qd^\mathsf{T}`$ を作らず、danglingノード上の質量 $`d^\mathsf{T}x`$ を求めて $`q`$ に分配する。密なGoogle行列も作らない。

PageRankの原型は [BrinとPageの論文](https://snap.stanford.edu/class/cs224w-readings/Brin98Anatomy.pdf) にある。Web以外も含む線形代数的な整理については [Gleich, PageRank Beyond the Web](https://www.cs.purdue.edu/homes/dgleich/publications/Gleich%202015%20-%20prbeyond.pdf) が参考になる。

### 一意性に必要な条件

固定点写像を $`F(x)=\alpha Tx+(1-\alpha)p`$ とする。$`T`$ は列確率行列なので、誘導L1ノルムが1であり、

```math
\|F(x)-F(y)\|_1\le\alpha\|x-y\|_1
```

が成り立つ。したがって $`\alpha<1`$ なら縮小写像になる。

この定義では、元のグラフが強連結であることも、$`p`$ の全成分が正であることも、一意性の必要条件ではない。孤立点や分離した成分を含むグラフも扱える。$`\alpha=1`$ はこの議論の対象外なので、APIでも受け付けない。

## 2. `tol`を同じ値にするだけでは比較にならない

PageRankのベンチマークでは、終了条件の違いが速度差に混ざりやすい。

例えば、[NetworkXの公開APIとソース](https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html) では、反復差のL1ノルムが `len(G) * tol` より小さくなると停止する。頂点数を変えると、同じ `tol` でも実際の条件が変わる。現在のNetworkXの標準実装は内部でSciPyを使うため、単純な「純Python対Cython」という比較にもならない。

本実装の `tol` は、頂点数によらない**PageRankベクトルの絶対L1誤差目標**とした。

### 反復差から求める上界

反復を $`x_{k+1}=F(x_k)`$ とし、$`\delta_k=\|x_{k+1}-x_k\|_1`$ とする。縮小性と幾何級数から、

```math
\|x_{k+1}-x_*\|_1
\le\sum_{j=1}^{\infty}\alpha^j\delta_k
=\frac{\alpha}{1-\alpha}\delta_k
```

となる。目標を $`\varepsilon`$ としたとき、$`\alpha>0`$ では

```math
\delta_k\le\varepsilon\frac{1-\alpha}{\alpha}
```

を停止条件にできる。NetworkXへ渡す値もこの条件に換算する。

```python
nx_tol = target_l1 * (1.0 - alpha) / (alpha * n)
```

$`\alpha=0`$ は $`x_*=p`$ なので別に処理する。

### 計算後にも残差を調べる

候補 $`\widehat{x}`$ の残差を $`r=F(\widehat{x})-\widehat{x}`$ とする。同じ縮小性から、

```math
\|\widehat{x}-x_*\|_1\le\frac{\|r\|_1}{1-\alpha}
```

が得られる。各実装が返した結果を共通のfloat64計算で評価し、この上界が目標以下であること、非負性、確率質量を確認した。PRPACKは公開APIから内部の停止精度を同じ形で指定できないため、返された結果が共通の検査を満たした場合だけ比較対象にする。実際に達成する精度まで各手法で完全に一致するわけではない。

以上は厳密演算における上界である。コードは残差を浮動小数点で評価しており、丸め誤差まで外向きに包む区間演算の保証ではない。`float32` で達成できない目標を渡した場合にも、勝手に許容誤差を緩めることはしない。

## 3. Cythonでどこを速くしたか

### 3.1 入ってくる辺をまとめて読む

元の辺の向きに沿って寄与を配るpush型では、複数スレッドが同じ出力ノードを書き換える場合がある。今回は $`H^\mathsf{T}`$ をCSRで保持し、出力ノードごとに入ってくる辺を集計するpull型にした。

```math
x_i^{\mathrm{new}}=
\alpha\sum_{j\to i}H_{ji}x_j
+\alpha q_i\sum_{j:d_j=1}x_j
+(1-\alpha)p_i
```

各出力要素の担当を一つに決められるため、PageRank値を書き込む際のatomic加算を使わずに済む。格納量と一反復の計算量は $`O(n+m)`$ である。ただし、このプロジェクトの `Graph` は解析用に生の隣接行列、正規化行列、その転置を保持するため、疎行列一本分だけのメモリという意味ではない。

### 3.2 Python処理とメモリ確保を反復の外へ出す

[Cythonのtyped memoryview](https://docs.cython.org/en/stable/src/userguide/memoryviews.html) で連続バッファを受け取り、内部のループはCの数値とポインタで動かす。辺の走査中にPythonオブジェクトを生成しない。スコア用の配列を二本確保して交換し、毎回の出力配列確保もなくした。

`boundscheck=False` などは、公開Python APIで形状、重み、添字、CSR構造を検証することを前提に使う。特に、疎行列をCSRへ変換する前の構造検査も必要になる。入力検査を無くした時間を、通常のAPIの性能として報告してはいない。

### 3.3 更新・収束判定・dangling質量を融合する

新しいスコアを書きながら、反復差と、次の反復で使うdangling質量を集計する。基本形は次になる。これは処理の構造を示す擬似コードであり、全コードは `_kernels.pyx` にある。

```cython
for i in range(start, stop):
    stored = alpha * row_sum(i, x) + teleport_term(i)
    y[i] = stored
    delta += fabs(stored - x[i])
    if dangling[i]:
        next_dangling_mass += stored
```

実装は、一様分布、$`p=q`$、独立した $`p,q`$ を分ける。一様分布ではスカラーの加算にでき、分布ベクトルの読み出しも減らせる。`float32` の場合も行の集計とノルムの集計はdoubleで行い、保存するスコアをfloat32にする。

### 3.4 OpenMPの負荷を辺数で分ける

[Cythonの `prange`](https://docs.cython.org/en/latest/src/userguide/parallelism.html) を使い、GILを解放して並列化する。ただし、ノード数を均等に分けても、入次数が偏っていれば仕事量は均等にならない。

そのため、累積非ゼロ要素数と行数の和を仕事量の近似として、連続した行区間を分割した。各スレッドの部分和は別の領域に書き、固定順に集計する。スレッド数1の場合はOpenMPの並列領域へ入らない。

比較用に、等行数分割も `partition="rows"` で選べる。辺数による分割でも巨大な一行は分割しないため、あらゆる次数分布で完全に負荷がそろうわけではない。

### 3.5 データ型とコンパイラ

スコアはfloat32 / float64、疎行列の添字はint32 / int64に対応した。収まるグラフではint32を使う。比較の主結果はすべてfloat64であり、精度を落とした結果を混ぜて高速化率を出してはいない。

ビルドは `-O3`、OpenMP、明示的なnativeオプションを使った。`-ffast-math` は使っていない。nativeビルドは測定したCPU向けであり、標準ビルドではこの最適化を有効にしない。

## 4. 測定方法

### 環境

測定日は2026年10月23日（JST）。一つのLinuxコンテナ環境で実行した。

| 項目 | 条件 |
|---|---|
| CPUモデルの表示 | AMD EPYC 9V74 80-Core Processor |
| コンテナに見えるCPU | 起動時affinityは9論理CPU、cgroup quotaは8 CPU相当 |
| メモリ上限 | 8 GiB |
| Python | 3.12.14 |
| Cython / GCC | 3.3.0 / 13.3.0 |
| NumPy / SciPy | 2.3.5 / 1.17.0 |
| NetworkX / igraph | 3.7 / 1.0.0 |
| 基本の減衰係数 | 0.85 |
| 目標L1誤差上界 | 1e-10 |
| 本計測 / warmup | 7回 / 2回 |
| 代表値 | 中央値。IQR、MAD、生試行も保存 |

`OPENBLAS_NUM_THREADS=1`、`OMP_PROC_BIND=close`、`OMP_PLACES=cores` とした。OpenMPが主スレッドをCPU 0へ固定するため、ライブラリ読み込み後の `sched_getaffinity(0)` は `[0]` になる。別の診断で4スレッドがCPU 0、1、2、3へ配置されることを確認した。CPUモデルに含まれる「80-Core」を、コンテナで使えるコア数として扱ってはいない。

各反復で実装の実行順をseed付きで無作為化し、warmupは集計から除外した。共有仮想化環境の測定であり、7試行から全マシンでの優劣を一般化するものではない。IQRは試行のばらつきで、信頼区間ではない。

### 比較した実装

| 名前 | 中身 |
|---|---|
| 純Python | 前処理済みリスト上のべき乗法。小規模だけで測定 |
| Cython | 今回の融合pullカーネル、1 / 2 / 4スレッドなど |
| SciPy | 同じCSR、初期値、停止条件による疎行列べき乗法 |
| NetworkX | `nx.pagerank` の公開API |
| igraph | `personalized_pagerank(implementation="prpack")` |
| direct / GMRES | 追加実験の線形方程式解法 |

[igraphの公式資料](https://igraph.org/c/html/develop/igraph-Structural.html) にあるように、PRPACKは線形方程式を用いる実装である。Cythonと同じ反復処理を、単に別言語で書いたものではない。

### 時間の区切り

主比較は、前処理済みの各ライブラリのグラフを渡す**公開APIのwall time**とした。自作APIには引数検査、作業配列の確保、最終残差検査も含む。NetworkXとigraphでは結果をNumPy配列へ揃える時間も含むが、共通の事後精度検査は主計測の外で行う。この点では自作側が追加検査の分だけ負担している。

NetworkXは公開APIの内部で疎行列を構築し直すため、その時間も含まれる。対NetworkXの倍率を「数値カーネルだけが何倍速い」とは解釈できない。

元のCSRから各グラフオブジェクトを作る時間、自作ソルバ内部の時間も別に保存した。前処理時間とAPI中央値を加えた列もあるが、それは別測定の和であり、cold end-to-endを直接測った値ではない。グラフ生成、ダウンロード、ファイルの展開は主計測に含めない。

## 5. 合成グラフでの結果

グラフは一様ランダム、Zipf型の終点分布、弱結合な8ブロック、約20%のdanglingノードを持つものを用意した。Zipf型はpreferential attachmentで生成しておらず、Barabási–Albertモデルではない。重複した辺は重みを足すため、標本辺数と非ゼロ辺数は異なる。

以下は10万頂点、$`\alpha=0.85`$ の結果。単位はmsで、すべて公開APIの中央値である。

| グラフ | 非ゼロ辺数 | Cython 1 | Cython 4 | SciPy | NetworkX | igraph PRPACK |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 1,599,864 | 34.252 | 16.414 | 39.727 | 1,684.741 | 327.604 |
| skewed | 1,379,937 | 43.173 | 25.706 | 43.454 | 1,174.905 | 216.391 |
| block | 1,599,091 | 164.838 | 66.332 | 166.615 | 1,586.698 | 297.578 |
| dangling | 1,279,924 | 33.760 | 13.812 | 33.827 | 1,344.058 | 139.465 |

Cythonの4スレッド版はSciPyに対して約1.69〜2.51倍の速度になった。一方、単一スレッドのCythonとSciPyでは、特にskewed、block、danglingの中央値が近い。Cythonに移しただけで桁違いに速くなったという結果ではない。SciPyも疎行列演算自体はコンパイルされた処理である。

小規模では並列化の費用が目立った。1,000頂点のdanglingグラフではCythonの1スレッドが0.256 ms、4スレッドが0.429 msで、並列版の方が遅い。1,000頂点のuniformでは純Pythonが13.386 ms、Cythonの1スレッドが0.406 msだった。Pythonループを除く効果と、すでにコンパイルされているSciPyに対する効果は、分けて読む必要がある。

1,000 / 10,000 / 100,000頂点の全結果、2スレッドの値、各試行の精度は `benchmark_results/main/` に保存した。

重みを対数正規分布で生成し、$`p`$ と $`q`$ を別々の非一様分布にした10,000頂点のdanglingグラフでも検査した。公開API中央値はCythonの4スレッドが1.689 ms、SciPyが3.977 msだった。独立した $`q`$ を同じ形で指定できないPRPACKは、このケースでは比較から外している。結果は `benchmark_results/weighted_personalized/` にある。

## 6. 実Webグラフでの結果

[SNAPのStanford web graph](https://snap.stanford.edu/data/web-Stanford.html) を使った。2002年に収集された281,903頂点、2,312,497辺の有向グラフである。現在のWebを代表するという意味ではないが、合成グラフだけの検証を避けるために加えた。

元のIDは1から始まるため、観測されたIDを昇順に0始まりへ再番号付けした。余分なノード0を孤立点として追加してはいない。元の圧縮ファイルとCSRのSHA-256も記録した。

| 実装 | API中央値 ms | IQR ms | 最大の事後L1誤差上界 |
|---|---:|---:|---:|
| Cython 1スレッド | 671.344 | 653.490〜701.179 | 9.14e-11未満 |
| Cython 2スレッド | 354.442 | 343.950〜370.825 | 9.14e-11未満 |
| Cython 4スレッド | 235.729 | 229.286〜245.338 | 9.14e-11未満 |
| SciPy | 728.830 | 707.782〜767.670 | 9.14e-11未満 |
| NetworkX | 3,322.724 | 3,209.247〜3,372.611 | 9.14e-11未満 |
| igraph PRPACK | 508.058 | 484.815〜547.192 | 8.05e-12未満 |

Cythonの4スレッド版はSciPyの約3.09倍、igraph PRPACKの約2.16倍の速度だった。単一スレッドのCythonはPRPACKより遅い。べき乗法の反復数はCythonとSciPyでいずれも129回だった。

元のCSRからの前処理は、自作Graphが約73.6 ms、NetworkXのグラフと分布辞書が約7.04秒、igraphのグラフと入力リストが約0.932秒だった。これらは主表の時間に加えていない。同じグラフを何回解くか、最初からどの形式でグラフを持っているかによって、利用時の費用は変わる。

## 7. 負荷分散だけを変えた比較

同じグラフ、精度、カーネルに対して、行の分け方だけを変えた。以下は10万頂点、4スレッドの公開API中央値である。

| グラフ | 等行数分割 ms | 辺数＋行数で分割 ms | 等行数 / 辺数分割 |
|---|---:|---:|---:|
| uniform | 16.294 | 17.104 | 0.95 |
| skewed | 31.222 | 24.304 | 1.28 |
| block | 69.226 | 61.453 | 1.13 |

入次数が偏ったskewedでは中央値が31.222 msから24.304 msへ小さくなり、推定仕事量の最大値／平均値も3.523から1.036へ近づいた。ただし、4スレッドのIQRは等行数で30.335〜33.272 ms、辺数分割で23.943〜34.656 msと重なる。uniformでは等行数分割の中央値が少し良かった。分割戦略を常に一方が勝つものとして扱うことはできない。

単一スレッドでは二つの設定が同じカーネル経路を通る。その対照でも、例えばblockは197.300 msと176.515 msにばらついた。したがって、blockの4スレッドでの約13%の差を、それだけで分割改善の確証とはしない。生試行とIQR、実際の分割境界、推定仕事量を `benchmark_results/ablation/` に残した。

## 8. 反復を速くするより、解法を変える方が速い場合

別の8ブロックグラフで $`\alpha=0.99`$ とし、teleportationを先頭の1ブロックへ集中させた。10,000頂点のケースでは、先頭の1,250頂点に一様に分配する。初期値は全頂点で一様とし、ブロック間に偏った解への収束も測る追加実験とした。

10,000頂点、159,024辺の結果は次のようになった。

| 解法・実装 | API中央値 ms |
|---|---:|
| Cython 1スレッド | 130.052 |
| Cython 4スレッド | 56.983 |
| SciPyべき乗法 | 147.248 |
| matrix-free GMRES | 21.866 |
| NetworkX | 256.728 |
| igraph PRPACK | 264.884 |

べき乗法は850回反復した。GMRESは共通の誤差目標を満たし、4スレッドのCythonより約2.61倍速かった。最大の事後L1誤差上界は、Cythonが約9.72e-11、GMRESが約2.60e-11、PRPACKが約8.33e-13であり、達成精度にも差がある。Cythonで一反復を高速化しても、反復数の多さを常に埋められるわけではない。

この実験ではPRPACKもGMRESを上回ってはいない。使用する線形方程式の解法、前処理、グラフ構造によって結果は変わる。GMRESのこの数値を全グラフに一般化するのも適切ではない。

疎LUによる直接法も実装したが、fill-inを考慮してベンチマークの既定上限を3,000頂点とした。1,000頂点の追加ケースでは直接法も計測し、その結果を保存した。10,000頂点の行を未測定のまま推定値で埋めることはしていない。

## 9. 100万頂点まで拡大する

uniformを1,000,000頂点、15,999,877辺まで拡大し、CythonとSciPyを測定した。精度、減衰係数、7回の本計測と2回のwarmupは主実験と同じである。

| 実装 | API中央値 ms | IQR ms |
|---|---:|---:|
| Cython 1スレッド | 765.312 | 702.437〜797.068 |
| Cython 2スレッド | 460.396 | 445.181〜468.506 |
| Cython 4スレッド | 261.737 | 250.397〜270.020 |
| Cython 8スレッド | 208.812 | 195.808〜219.287 |
| SciPy | 681.256 | 652.149〜760.204 |

8スレッド版はSciPyの約3.26倍の速度だった。一方、単一スレッドのCythonは中央値ではSciPyより遅い。4から8スレッドへ増やしても速度差は約1.25倍であり、スレッド数に比例した改善にはならなかった。

すべて17反復で、事後L1誤差上界は7.67e-12未満だった。プロセス全体の累積peak RSSは約972 MiBである。これにはグラフ生成や前処理も含むため、Cythonカーネルだけの必要メモリとは解釈しない。この規模でNetworkXやPRPACKを測った数値はない。

## 10. 数学的な検証に使える機能

速度を測るだけで終わらないよう、同じグラフに対して次を使えるようにした。

| 機能 | 用途 |
|---|---|
| `residual` | 任意の候補ベクトルの残差、誤差上界、総和、非負性 |
| `google_operator` | 密行列なしのGoogle行列の積、転置積、複数列の積 |
| `solve_linear` | 疎LU＋danglingのrank-one補正、またはGMRES |
| `alpha_derivative` | 減衰係数に対する感度 |
| `personalization_derivative` | 確率分布を動かす方向に対する感度 |
| `edge_derivative` | 一辺の生の重みを変えたときのスコア変化 |
| `edge_gradient` | 線形目的関数に対する全格納辺の勾配 |
| `exact_pagerank` | 小さいグラフの有理数解 |
| `spectrum` | 小さいGoogle行列の全固有値とspectral gap |
| `neumann_series` | 経路長ごとの寄与と打ち切り誤差 |
| `alpha_sweep` / `personalized_batch` | パラメータ掃引と複数の分布の計算 |
| `rank_scores` / `compare_scores` | 同点処理、L1距離、順位相関、top-k比較 |

### 感度解析で何を固定するか

$`T,p,q`$ を固定して $`\alpha`$ だけを変えると、

```math
(I-\alpha T)\frac{\partial x_*}{\partial\alpha}=Tx_*-p
```

を解けばよい。辺 $`W_{ij}`$ を動かす場合は、行正規化も微分する。$`s_i>0`$ のとき、

```math
(I-\alpha T)\frac{\partial x_*}{\partial W_{ij}}
=\frac{\alpha x_{*,i}}{s_i}
\left(e_j-H_{i,:}^{\mathsf{T}}\right)
```

となる。danglingノードに最初の辺を作る操作は規則の切り替わりを含むため、この微分APIでは受け付けない。

また、`dangling=None` の既定値は $`q=p`$ なので、personalizationを動かすとdanglingの分布も動く。明示的に固定した $`q`$ を渡す場合とは微分が異なる。この違いもAPIとテストで扱っている。

### 厳密解を小さい例の基準にする

```python
from pagerank_lab import Graph, pagerank, exact_pagerank

g = Graph.from_edges([0], [1], n=2)
print(exact_pagerank(g, alpha="17/20"))
# Matrix([[20/57], [37/57]])

r = pagerank(g, alpha=0.85, tol=1e-12)
print(r.scores)
print(r.error_bound_l1)
```

有理数解は既定で32頂点以下、全固有値は512頂点以下とした。グラフに保存されたfloat64の辺重みを十進有理数として解釈するため、floatへ変換する前に存在した分数を復元する機能ではない。集約後の重みがfloat64で正確に表現できる整数なら、その整数グラフの厳密解として使える。

Neumann級数は

```math
x_*=(1-\alpha)\sum_{k=0}^{\infty}\alpha^kT^kp
```

を経路長別に返す。先頭 $`K`$ 項までの部分和は正規化せず、残りのL1質量 $`\alpha^K`$ も返す。

## 11. ツールとして使う

ソースを展開し、`pyproject.toml` のあるディレクトリでインストールする。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[all,igraph]'

pagerank-lab run examples/toy.csv --method cython --output scores.json
```

標準ビルドは1スレッド用である。Linux / GCCでOpenMPを有効にし、今回の固定依存バージョンを使う場合は次のようにする。このバージョン集合はLinux / CPython 3.12で測定したもので、全OS・全Python向けのlockfileではない。

```bash
python -m pip install -r requirements-benchmark.txt
CC=gcc LDSHARED='gcc -shared' PAGERANK_OPENMP=1 PAGERANK_NATIVE=1 \
  python -m pip install -e . --no-build-isolation --force-reinstall --no-deps
```

ビルド済みのファイルが残る場合の再構築手順は `docs/build.md` にまとめた。作ったnativeバイナリはビルド機のCPU向けである。

```python
from pagerank_lab import Graph, pagerank

g = Graph.from_edges([0, 0, 1, 2], [1, 2, 2, 0], n=4)
r = pagerank(
    g,
    alpha=0.85,
    personalization=[1, 0, 0, 0],
    threads=4,
    tol=1e-10,
    history=True,
)
print(r.topk(4))
print(r.iterations, r.error_bound_l1)
```

CSVの辺リスト、SciPy NPZ、Matrix Marketを読み込める。結果はJSON、NPZ、CSVへ保存できる。ノード数を指定すれば、辺リストに現れない孤立点も保持する。

`Graph` の準備済み配列は再利用するため、構築後に書き換えない。辺を変更するときは新しい `Graph` を作る。複数personalizationのAPIは、グラフを再利用して各クエリを逐次処理するものであり、専用の並列SpMMカーネルではない。

## 12. 再現と検証

主な合成グラフの測定は次のコマンドで再実行できる。

```bash
OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
  python -m benchmarks.run \
  --sizes 1000 10000 100000 \
  --families uniform skewed block dangling \
  --methods python cython scipy networkx igraph \
  --threads 1 2 4 --alphas 0.85 \
  --repeats 7 --warmups 2 --output benchmark_results/main
```

実Webグラフも、ダウンロードを明示して同じハーネスで測定できる。

```bash
OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
  python -m benchmarks.stanford \
  --download data/web-Stanford.txt.gz \
  --expect-sha256 49bfc0e366c3028dbd84e8557b09287cbd61051bde29d160c277688c727cb679 \
  --methods cython scipy networkx igraph --threads 1 2 4 \
  --repeats 7 --warmups 2 --output benchmark_results/stanford
```

元データは配布物に同梱せず、取得先と測定したファイルのハッシュを残した。測定結果のJSON / CSVと、そこから作る図は `benchmark_results/` にある。

テストは独立した密行列解、有理数解、有限差分、随伴方程式の恒等式、ノード置換、行重みの尺度不変性、並列・逐次の一致、不正入力、CLIの入出力を含む。実行済み環境と件数は `docs/validation.md` に記録した。GitHub Actionsの設定はLinux / macOS / Windowsを対象とするが、ここで実際に実行した結果はLinuxのものだけである。

## 13. この検証で言えること

Pythonで辺を一つずつ処理する場合には、Cythonで数値ループをまとめる効果が大きかった。すでに疎行列演算がコンパイルされているSciPyに対しては、単一スレッドの差は小さく、スレッド数を増やした条件で差が大きかった。ループ融合も実装したが、その単独の寄与は今回の測定では切り分けていない。実Webグラフでも4スレッド版の改善を確認できた。

ただし、スレッド数を増やす効果はグラフサイズや次数分布に依存し、負荷分割の選択も万能ではなかった。収束が遅い場合には、Cythonのべき乗法よりGMRESが速かった。

今回の実装はCPU上の疎グラフを対象にしており、GPU、分散計算、局所push法、動的グラフの増分更新、厳密な区間演算との比較は行っていない。測っていない方法まで含めた最速性は主張しない。再利用可能なAPI、実際の入力、精度目標、前処理、試行のばらつきを揃えて、次の最適化を判断するための土台とした。
