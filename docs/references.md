# 一次資料と検証上の注意

確認日: 2026-10-23。`stable` / `latest` の文書は更新されるため、実験で実際に使った版は各 `results.json` の `environment.versions` を参照する。

## PageRank の定義

1. Sergey Brin and Lawrence Page, **The anatomy of a large-scale hypertextual Web search engine**, *Computer Networks and ISDN Systems* 30, 107–117 (1998), DOI: [10.1016/S0169-7552(98)00110-X](https://doi.org/10.1016/S0169-7552(98)00110-X)。[Stanford による PDF](https://snap.stanford.edu/class/cs224w-readings/Brin98Anatomy.pdf)。§2.1 がリンク構造、出次数による正規化、反復法、random surfer と personalization の基本概念を説明する。本プロジェクトは現在の検索サービスの内部実装を再現するものではない。
2. Lawrence Page, Sergey Brin, Rajeev Motwani and Terry Winograd, **The PageRank Citation Ranking: Bringing Order to the Web**, Stanford InfoLab Technical Report 1999-66 (1999)。[機関リポジトリの所在](http://ilpubs.stanford.edu:8090/422/)。この URL は環境によって取得できないことがある。本プロジェクトの定義は本文に明示しており、この所在の到達性には依存しない。

## 比較対象の実装

3. NetworkX, [`pagerank` API](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html) / [実装ソース](https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html)。公開 API は SciPy の疎行列を用いた反復法を呼び、停止条件は反復差の L1 ノルム `< len(G) * tol` である。公開呼び出しの中にグラフから疎行列への変換と行正規化が含まれる。したがって公開 API の所要時間を Cython の内部反復ループだけと直接比較してはならない。
4. igraph, [C library reference: PageRank](https://igraph.org/c/html/latest/igraph-Structural.html#igraph_pagerank)。PRPACK は線形方程式を解く方式であり、単純なべき乗法の別言語実装ではない。[PRPACK の公式リポジトリ](https://github.com/dgleich/prpack)。本ベンチマークでは Python の `personalized_pagerank(..., implementation="prpack")` を測定する。PRPACK の内部停止条件をこちらの値に置き換えられないため、得られたベクトルの残差を共通の式で評価する。

## Cython・疎行列

5. Cython, [Typed Memoryviews](https://docs.cython.org/en/stable/src/userguide/memoryviews.html)。型付き memoryview と連続性指定により NumPy 配列の数値要素を C の型として扱える。数値 memoryview の通常の添字アクセスは GIL を必要としない。
6. Cython, [Using Parallelism](https://cython.readthedocs.io/en/stable/src/userguide/parallelism.html)。`prange`、OpenMP、reduction、スケジューリングとビルドオプションを説明する。各頂点への入辺をまとめて読む gather/pull 方式では、出力を頂点ごとに別のスレッドが担当できる。ただしグラフの次数分布によって仕事量は偏る。ここでの方式選択は本実装の設計判断である。
7. SciPy, [`csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)。`data`, `indices`, `indptr` からなる CSR 形式と疎行列演算。転置した行正規化行列を CSR として保持し、入辺の走査に用いる。
8. SciPy, [`LinearOperator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html)。行列全体を明示的に作らず、行列ベクトル積で反復法を使用できる。本プロジェクトでは dangling 補正と teleportation の rank-one 項を密行列化しない。
9. SciPy, [`gmres`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html)。GMRES の API、残差停止条件、restart と callback の意味。GMRES の Euclid ノルムの停止条件と、本プロジェクトで報告する L1 残差境界は別物であり、最終的に後者も検査する。

## 本プロジェクトで明示的に導く関係

以下は資料の引用ではなく、文書中の定義から直接導く。

- $H$ を dangling 行がゼロの行正規化隣接行列、$d$ を dangling 指示ベクトル、$q$ をその再分配確率、$p$ を teleportation 確率とする。$S=H^T+qd^T$ は列確率行列である。
- $F(x)=\alpha Sx+(1-\alpha)p$ とすると $0\le\alpha<1$ で $\|F(x)-F(y)\|_1\le\alpha\|x-y\|_1$。したがって $p$ の全成分が正でなくても固定点は一意で、反復は収束する。不可約性をこの affine contraction の必要条件と取り違えない。
- 残差 $r=F(x)-x$ から $\|x-x_*\|_1\le\|r\|_1/(1-\alpha)$。
- 反復差 $\delta_k=\|x_{k+1}-x_k\|_1$ から、返す新しい反復点について $\|x_{k+1}-x_*\|_1\le\alpha\delta_k/(1-\alpha)$。
- よって同じ目標 $\varepsilon$ に対し NetworkX の `tol` は $\varepsilon(1-\alpha)/(\alpha n)$ にする。$\alpha=0$ は別処理する。

これらの不等式は厳密演算での定理である。float64 の残差計算には丸め誤差があり、ここでの値は区間演算による機械検証済み包含区間ではない。特に $\alpha\to1$、極端に小さい許容誤差、float32 の場合にはこの差が重要になる。

## ベンチマークで主張できる範囲

- 生データ、固定 seed、実際の非ゼロ辺数、隣接行列の SHA-256、依存バージョン、CPU affinity と cgroup 制限、全反復の実行順序を保存する。
- 主表は前処理済みオブジェクトに対する公開 API の実時間であり、Cython / SciPy の内部反復時間も別の列に残す。
- `preprocess_plus_median_seconds` は別々に測定した前処理と warm call 中央値の和である。独立した cold end-to-end 測定値とは呼ばない。
- 比較は float64 に固定する。float32 の速度を float64 の比較表に混ぜない。
- `process_peak_rss_bytes_cumulative` はプロセス全体の累積最高 RSS であり、手法別のメモリ使用量ではない。
- 合成グラフ上の結果から実 Web グラフの速度や全環境における最速性を断言しない。`skewed` は Zipf 型の宛先サンプリングであり、Barabási–Albert モデルではない。

## 実 Web グラフ

10. Stanford SNAP, [Stanford web graph](https://snap.stanford.edu/data/web-Stanford.html)。2002 年に収集した `stanford.edu` のページと有向ハイパーリンクを表し、公式統計は 281,903 頂点、2,312,497 辺。[公式の gzip 辺リスト](https://snap.stanford.edu/data/web-Stanford.txt.gz)を `python -m benchmarks.stanford --input PATH` で読み込む。ダウンロードには明示的な `--download PATH` を必要とし、生データはプロジェクトに同梱しない。観測 ID を昇順に連番へ写し、自己ループは保持し、頂点数・生の辺数・正規化前 CSR の非ゼロ数を検査する。測定 JSON に入力 gzip と CSR の SHA-256 を保存する。実在する一つの過去の Web グラフの測定であり、現代の Web 全体の性能を示すものではない。
