# ベンチマークの再現方法と読み方

## 何を比較するか

同じ有向重み付きグラフに対し、同じ $\alpha$、同じ teleportation 分布 $p$、同じ dangling 再分配分布 $q$、同じ一様初期ベクトルを与える。べき乗法は同じ目標 L1 誤差を使う。比較対象は次の通り。

| 名称 | 内容 | 入力オブジェクト |
| --- | --- | --- |
| `python` | Python の数値リストとループによる参照べき乗法 | 正規化済み入辺リスト |
| `cython_t1` | Cython の直列べき乗法 | 前処理済み `Graph` |
| `cython_t2`, `cython_t4` 等 | Cython / OpenMP のべき乗法 | 前処理済み `Graph` |
| `scipy` | SciPy 疎行列ベクトル積によるべき乗法 | 前処理済み `Graph` |
| `direct` | 疎 LU と dangling の rank-one 補正 | 前処理済み `Graph` |
| `gmres` | 行列を明示化しない GMRES | 前処理済み `Graph` |
| `networkx` | 公開 `networkx.pagerank` API | `DiGraph` |
| `igraph` | `personalized_pagerank` の PRPACK 実装 | igraph の有向グラフ |

NetworkX の公開 API には、呼び出しごとの疎行列作成と正規化が含まれる。対 NetworkX の倍率を、反復ループそのものの高速化率とは呼ばない。SciPy との比較が、同じ疎行列を使う反復処理の差を評価する上でより直接的である。

PRPACK は線形方程式を解くアルゴリズムであり、他の方法と反復列を揃える比較ではない。初期ベクトルやこちらの `tol` を指定できないため、計算後の残差が要求水準を満たした場合に限り速度の比較対象に含める。PRPACK は dangling に reset 分布を使うため、$q\ne p$ のケースは明示的にスキップする。

`direct` と `gmres` も、べき乗法と同じ最終残差条件に対する比較である。べき乗法の初期ベクトルや反復回数が同じという意味ではない。公開 API の `iterations=0` は反復不要だったという測定値ではなく、このカウントを提供しないことを示す。ベンチマークでは `null` に変換する。`direct` は fill-in を考慮して既定で $n\le3000$ のみを測り、`--direct-max-n` で変更できる。GMRES の `max_iter` は SciPy の restart cycle 数であり、べき乗法のステップ数とは単位が違う。

## コマンド

リポジトリのルートで実行する。インストール手順と OpenMP / native build の選択は README を参照する。

```bash
OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
python -m benchmarks.run \
  --sizes 1000 10000 100000 \
  --families uniform skewed block dangling \
  --methods python cython scipy networkx igraph \
  --threads 1 2 4 --alphas 0.85 \
  --repeats 7 --warmups 2 --tol 1e-10 \
  --seed 20261023 --output benchmark_results/main

python -m benchmarks.plot benchmark_results/main/results.json
```

`python` は既定では $n\le1000$ だけを測定する。`--python-max-n` で変更できる。スキップ理由は JSON に保存される。未インストールの NetworkX / igraph も、計測済みとして扱わずスキップする。Cython が使用不可、または計算結果が共通精度条件を満たさない場合には `failed` とし、終了コード 1 を返す。

線形ソルバも含める場合は `--methods python cython scipy direct gmres networkx igraph` を指定する。疎 LU は因子分解後に密に近づく場合があるため、大きなグラフで上限を引き上げる際には使用可能メモリを考慮する。

収束の遅い成分を意図的に励起する補助実験:

```bash
OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
python -m benchmarks.run \
  --sizes 1000 10000 --families block \
  --methods cython scipy networkx igraph --threads 1 2 4 \
  --alphas 0.99 --personalization localized \
  --repeats 7 --warmups 2 --output benchmark_results/slow_mixing
```

重みと非一様再分配の検証:

```bash
python -m benchmarks.run \
  --sizes 1000 10000 --families dangling --weighted \
  --personalization random --dangling-distribution random \
  --methods cython scipy networkx igraph --threads 1 2 \
  --repeats 7 --warmups 2 --output benchmark_results/weighted
```

実行時の CPU 制約とスレッド数を合わせる。`OMP_NUM_THREADS` は実行環境の記録にも残るが、Cython では `--threads` による明示的指定を使う。BLAS を含む他の計算が過剰にスレッドを作らないよう、再現時には `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` 等も記録する。ビルド時の環境変数と実行時の環境変数は同じとは限らないため、拡張モジュール自身の build metadata も保存する。

## 精度条件

$H$ を出次数・出重みで正規化した隣接行列、$d$ を dangling 指示ベクトルとすると、

$$
F(x)=\alpha(H^Tx+q(d^Tx))+(1-\alpha)p.
$$

全手法の出力に対し、独立した float64 / SciPy の行列ベクトル積で

$$
r_1=\|F(x)-x\|_1,\qquad B=\frac{r_1}{1-\alpha}
$$

を計算する。`B <= tol`、非負性、総和が 1 であることの数値誤差検査をすべて通過した結果だけが `ok` になる。残差を大きな係数で緩めて合格にする処理はない。

この境界は厳密演算では保証になるが、ここで報告する値は float64 による評価である。丸め誤差まで包含する区間演算ではない。`tol` を機械精度以下にしても実現可能とは限らない。

NetworkX の停止条件は $\delta<n\,\mathrm{tol}_{NX}$ である。他のべき乗法で返す反復点の境界 $\alpha\delta/(1-\alpha)$ と合わせるため、$\alpha>0$ では

$$
\mathrm{tol}_{NX}=\frac{\mathrm{tol}(1-\alpha)}{\alpha n}
$$

を使う。数値を同じにしただけの `tol=1e-10` は、同精度比較にはならない。浮動小数点の加算順序や `<` / `<=` の違いで反復回数が 1 回程度ずれることはあり、最終出力の独立残差検査を優先する。

## 測定区間

| JSON / CSV の項目 | 含めるもの | 含めないもの |
| --- | --- | --- |
| `median_seconds` 等 | 各手法の公開呼び出しと返り値の ndarray 化。自作 API では入力検査・必要な確保・最終残差検査も含む | 外部のグラフ作成、ハーネスが行う追加の独立残差検査 |
| `solver_median_seconds` | 自作べき乗法の内部反復処理。Python 参照法も内部反復だけを記録 | 公開ラッパー、最後の残差検査。NetworkX / igraph のこの値は取得せず `null` |
| `preprocess_seconds` | 共通の canonical CSR から、その手法が使うオブジェクトとベクトルを作る処理 | 合成グラフの乱数生成、ファイル I/O、solve |
| `preprocess_plus_median_seconds` | 前処理の単発測定値と、warm API 呼び出し中央値の和 | 独立した cold end-to-end の測定ではない |
| `process_peak_rss_bytes_cumulative` | プロセス全体の累積最高 RSS。全手法、Python、import、前処理を含む | 手法別のピークメモリには分離していない |

準備の共通始点は canonical CSR である。NetworkX の準備時間はそれを `DiGraph` に変換し、確率ベクトルを辞書にする時間。自作手法は `Graph` を作る時間と Cython 用の配列の準備時間である。これらのスコープが違うことを隠して単一の「純粋なカーネル速度」として報告しない。

Cython の `balanced` partition は各スレッド数について事前に作成し、その構築時間を `cython_partition_tN_seconds` と `preprocess_seconds` に計上する。warmup で暗黙に作成され、前処理時間から消えてしまう扱いにはしていない。

全手法を warmup した後、各反復で手法の順番を seed 付きで無作為化する。順序は JSON に残す。CPU キャッシュを明示的に flush する実験ではない。GC は有効のまま。中央値・最小値・最大値・四分位点・中央値絶対偏差 (MAD) と、生の全試行を保存する。7 回程度の測定から統計的に強い一般化は行わない。

## グラフの性質

| family | 生成方法 | 留意点 |
| --- | --- | --- |
| `uniform` | 始点・終点を独立に一様抽出 | 重複辺の重みを合算するため、非ゼロ辺数は `n*degree` より小さくなる |
| `skewed` | 始点は一様、終点は順位 $k$ に比例重み $k^{-1.1}$ | 固定された Zipf 型終点分布。優先的選択による成長モデルでも BA モデルでもない |
| `block` | 最大 8 群。98% の終点を同じ群の中から抽出 | 一様初期値が遅い固有成分をほとんど含まない場合、グラフが弱く結合していても速く収束して見える |
| `dangling` | 一様生成から `source % 5 == 0` の辺を除く | 頂点自体は残し、dangling と孤立点を検査できる |
| `cycle` | 有向サイクル | 一様初期値・一様 teleportation は既に解なので、一般的な収束速度の代表例に使わない |

`block` の補助実験では `--personalization localized` を使う。これは先頭の約 1/8 の頂点に teleportation の質量を集中させる。初期値は全手法で一様のままなので、群の間で確率質量を移す過程を測定できる。既定の一様ケースと混ぜず、別表として扱う。`--weighted` は独立な lognormal 重みを加え、重複辺では重みを足す。自己ループは保持する。

合成グラフは実 Web グラフではない。ノード次数、局所性、メモリ配置、dangling 比率、$\alpha$、要求精度によって優劣は変わる。最も都合のよいケースの倍率だけを代表値にしない。

## 出力の監査

`results.json` は生データと全条件、`summary.csv` は表計算で読める要約である。JSON には canonical CSR の `(indptr, indices, data)` を little-endian の int64 / int64 / float64 に揃えた SHA-256 を記録する。グラフの同一性を確認するためのものであり、ファイルの署名ではない。

環境情報は Python、OS、CPU model、論理 CPU 数、呼び出し元スレッドの CPU affinity、cgroup quota / memory limit、依存パッケージの版、コンパイラ・拡張モジュール情報、関連環境変数、時計の精度を含む。CPU の実効周波数・他プロセスの影響・マシン全体の負荷は制御したと仮定しない。

`environment.affinity` は `os.sched_getaffinity(0)` が返す呼び出し元スレッドのmaskである。`OMP_PROC_BIND` を指定した場合、拡張モジュールの読み込み時点でOpenMP runtimeがそのスレッドを1つのplaceへ固定し、例えば `[0]` になることがある。これはworker全体がCPU 0しか使えないという意味ではない。後続の測定では `affinity_semantics` と `calling_thread_affinity_before_build_info` も保存する。配置自体を確認するには次を実行する。

```bash
OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
python -m benchmarks.thread_affinity --threads 4 \
  --output benchmark_results/thread_affinity.json
```

この診断は4頂点のサイクルで1回の反復を行い、読み込み前後のmask、libgompのplace、`/proc/self/task/*/status` の各スレッドのmaskを記録する。速度は計測しない。

計測後のグラフは同じ JSON から作る。PNG は閲覧用、SVG は編集・論文図表等への利用向けである。エラーバーは試行の IQR であり、信頼区間ではない。異なる API の中央値の比にも信頼区間は付けていない。

## 行分割の ablation

```bash
OPENBLAS_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
python -m benchmarks.ablation \
  --sizes 100000 --families uniform skewed block --threads 1 4 \
  --alphas 0.85 --repeats 7 --warmups 2 --tol 1e-10 \
  --output benchmark_results/ablation
```

同じコンパイル済みバイナリ、データ、精度、初期値を用い、`partition="rows"` と `partition="balanced"` だけを切り替える。前者は頂点数で等分し、後者は累積 `nonzeros + rows` を仕事量の目安として連続した頂点範囲を分割する。CSR の 1 行は分割しないため、単一頂点に入辺が集中しすぎる場合には完全な均等化はできない。

準備した境界、各区間の推定仕事量、その最大値 / 平均値を JSON に残す。この推定値は実測の CPU cycle 数ではない。速度比 `api_rows_over_balanced` / `solver_rows_over_balanced` は 1 より大きいほど balanced の測定中央値が小さいことを意味する。1 スレッドは直列カーネルを使う対照条件であり、分割方法によって実質的な演算が変わらないことを確認する。
