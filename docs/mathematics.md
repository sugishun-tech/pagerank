# 数学的仕様と解析 API

この文書は、実装する PageRank の定義、誤差の評価方法、感度解析で固定する量を明示する。主な入口は `pagerank_lab.analysis` である。解析関数は `float64` を使い、`exact_pagerank` だけが SymPy の有理数演算を使う。

以下の不等式は厳密演算についての定理である。プログラムが通常の浮動小数点演算で計算する残差や誤差上界には丸め誤差がある。区間演算や有向丸めによる証明付き計算とは区別する。

## 1. グラフ、確率分布、行列の向き

頂点数を $n\ge 1$、重み付き隣接行列を $W\in\mathbb{R}_{\ge0}^{n\times n}$ とする。$W_{ij}$ は頂点 $i$ から頂点 $j$ に向かう辺の重みである。平行辺は重みを足し合わせ、重みゼロの辺は取り除く。自己ループと孤立点を許す。

出強度と行正規化行列を

$$
s_i=\sum_j W_{ij},\qquad
H_{ij}=\begin{cases}W_{ij}/s_i&s_i>0,\\0&s_i=0\end{cases}
$$

と定める。$d_i=1$ if $s_i=0$、それ以外は $d_i=0$ とする。$d$ は dangling node、つまり正の重みを持つ出辺がない頂点の指示ベクトルである。

列ベクトル $p,q$ は非負で総和が 1 の確率分布である。

- $p$ は teleportation の分布で、API の `personalization` に対応する。
- $q$ は dangling node からの再分配先で、API の `dangling` に対応する。
- `personalization=None` なら一様分布を使う。
- `dangling=None` なら $q=p$ とする。$p$ を変更するとデフォルトの $q$ も変わる。
- 配列で指定した分布は総和 1 に正規化する。負の値、非有限値、全成分ゼロは拒否する。

確率ベクトルを列で保持するため、遷移作用素は

$$
T=H^T+qd^T
$$

となる。$H$ は通常の行方向の隣接構造、$T$ は列方向の確率移動を表す。この向きの違いを混同しない。$T$ の各列和は 1 である。内部では $H^T$ を CSR として格納し、各出力頂点への入辺を連続して走査する。

PageRank は、$0\le\alpha<1$ に対する次の方程式の解 $x_*$ である。

$$
x_*=\alpha Tx_*+(1-\alpha)p,
\qquad
(I-\alpha T)x_*=(1-\alpha)p.
$$

グラフを作るときの行列の向きは次の例で確認できる。

```python
from pagerank_lab import Graph, pagerank

g = Graph.from_edges([0], [1], n=2)  # 0 -> 1、頂点 1 は dangling
r = pagerank(g, alpha=0.5)
print(r.scores)  # [0.4, 0.6]
```

## 2. 一意性、収束、正値性

$T$ は非負の列確率行列なので、誘導 L1 ノルムについて $\|T\|_1=1$ である。アフィン写像

$$
F(x)=\alpha Tx+(1-\alpha)p
$$

は

$$
\|F(x)-F(y)\|_1\le\alpha\|x-y\|_1
$$

を満たす。$\alpha<1$ なら縮小写像なので固定点は一意で、べき乗反復 $x_{k+1}=F(x_k)$ は収束する。確率単体は $F$ で不変だから、固定点も確率ベクトルである。

**この結論にグラフの強連結性、$T$ の不可約性、$p$ の全成分の正値性は必要ない。** 非負の $p$ が確率分布であればよい。一方、すべてのスコアが正であるとは限らない。$p_i>0$ なら $x_{*,i}\ge(1-\alpha)p_i>0$ だが、$p_i=0$ の頂点ではスコアが 0 の場合もある。

$\alpha=0$ なら $x_*=p$ である。$\alpha=1$ はこの縮小性を失い、定常分布の一意性や反復収束がグラフに依存するため、本 API では受け付けない。

## 3. 反復差と残差からの誤差評価

### 3.1 反復差

$\delta_k=\|x_{k+1}-x_k\|_1$ とする。以後の差は $\delta_{k+j}\le\alpha^j\delta_k$ なので、返す新しい点 $x_{k+1}$ の誤差は

$$
\begin{aligned}
\|x_*-x_{k+1}\|_1
&\le\sum_{j=1}^{\infty}\|x_{k+j+1}-x_{k+j}\|_1\\
&\le\frac{\alpha}{1-\alpha}\delta_k.
\end{aligned}
$$

したがって、目標誤差を $\varepsilon$ とすると停止基準は

$$
\delta_k\le\varepsilon\frac{1-\alpha}{\alpha}
$$

となる。$\alpha=0$ は別処理する。`tol=1e-10` を単に「前後の反復値の差が $10^{-10}$ 未満」とは解釈しない。

NetworkX の停止条件は反復差 `< n * tol` なので、同じ理論上の目標で比較するには NetworkX 側の `tol` を $\varepsilon(1-\alpha)/(\alpha n)$ にする必要がある。条件式の `<` と `<=` の違い、および演算順序の違いで反復回数が 1 回ずれることはあり得る。[NetworkX の実装](https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html)も参照。

### 3.2 残差

任意の候補 $\widehat{x}$ に対するアフィン残差を

$$
r=\alpha T\widehat{x}+(1-\alpha)p-\widehat{x}
$$

とする。Neumann 級数から

$$
(I-\alpha T)^{-1}=\sum_{k=0}^{\infty}\alpha^kT^k,
\qquad
\|(I-\alpha T)^{-1}\|_1\le\frac1{1-\alpha}
$$

であり、$x_*-\widehat{x}=(I-\alpha T)^{-1}r$ だから

$$
\boxed{\|x_*-\widehat{x}\|_1\le\frac{\|r\|_1}{1-\alpha}}.
$$

`residual(g, scores, ...)` は `residual_l1`、`error_bound_l1`、`mass`、`mass_error`、`min_score`、`nonnegative` を返す。候補の総和が厳密に 1 でなくても、アフィン残差の式は使える。

```python
from pagerank_lab.analysis import residual

report = residual(g, r.scores, alpha=0.5)
print(report["residual_l1"], report["error_bound_l1"])
```

この評価は異なるソルバー間の共通検査に使える。反復回数が同じかどうかよりも、同じ問題を解いたか、同じ残差条件を満たしたかを先に確認する。

### 3.3 浮動小数点演算での読み方

$\alpha$ が 1 に近いほど $1/(1-\alpha)$ が大きくなり、残差の丸め誤差も拡大される。有限精度では反復が停滞することがある。`float32` で過小な `tol` を指定しても精度は増えない。

本ツールの `error_bound_l1` は浮動小数点で評価した理論式の値であり、真の誤差を機械的に包含することを証明した区間ではない。厳密な小規模確認には有理数解を使う。大規模な証明付き計算を目的とする場合は、区間演算による残差評価などを別途実装する必要がある。

### 3.4 反復回数の事前見積もり

二つの確率ベクトルの L1 距離は高々 2 なので

$$
\|x_k-x_*\|_1\le 2\alpha^k.
$$

`iteration_bound(alpha, tol, initial_error_bound=2.0)` は、この式で目標を満たす反復数を返す。最悪の場合の見積もりであり、多くのグラフは早く収束する。より小さい初期誤差上界を独立に得ている場合だけ `initial_error_bound` を小さくする。

## 4. アフィン反復と Google 行列は異なるインターフェース

Google 行列を

$$
G=\alpha T+(1-\alpha)p\mathbf{1}^T
$$

とすると、線形写像としての積は

$$
Gv=\alpha Tv+(1-\alpha)p(\mathbf{1}^Tv)
$$

である。総和 1 の確率ベクトルについては $Gv=F(v)$ だが、一般の $v$ では一致しない。固有値計算や Krylov 法の `LinearOperator` に、定数 teleportation 項を足すアフィン写像をそのまま渡してはいけない。

`google_operator` は任意のベクトルに対して線形な `LinearOperator` を返す。`@`、転置演算、複数列への積を利用できる。逆方向の積は

$$
G^Tv=\alpha\{Hv+d(q^Tv)\}+(1-\alpha)\mathbf1(p^Tv)
$$

であり、実装ではこの式を使う。$qd^T$ と $p\mathbf1^T$ は密行列化しない。

```python
import numpy as np
from pagerank_lab.analysis import google_operator

G = google_operator(g, alpha=0.5)
v = np.array([2.0, -2.0])  # 総和 0 のベクトルも正しく扱う
print(G @ v)
print(G.T @ v)
```

## 5. 線形方程式として解く

`solve_linear(..., method="direct")` と `solve_linear(..., method="gmres")` は、同じ $(I-\alpha T)x=(1-\alpha)p$ を解く。結果は NumPy 配列である。

### 5.1 疎 LU と dangling 補正

$B=I-\alpha H^T$ とすると

$$
I-\alpha T=B-\alpha qd^T.
$$

$By=b$、$Bz=q$ を疎 LU で解き、Sherman–Morrison の式から

$$
(B-\alpha qd^T)^{-1}b
=y+z\frac{\alpha d^Ty}{1-\alpha d^Tz}
$$

を求める。dangling の個数が多くても、$qd^T$ を密行列として保存する必要はない。残差が十分小さくない場合は少数回の反復改良を行う。

ただし、**元のグラフが疎でも LU 因子が疎とは限らない。** fill-in によって時間・メモリが大幅に増えることがある。大規模なグラフに無条件で直接法を勧める仕様ではない。感度解析もこの疎 LU を利用するため、同じ制約がある。

### 5.2 GMRES

GMRES では $v\mapsto v-\alpha Tv$ を `LinearOperator` として使い、全行列を作らない。SciPy の収束判定は 2 ノルムなので、

$$
\|r\|_1\le\sqrt n\|r\|_2
$$

に基づき内部目標を設定し、最後に L1 残差を再評価する。`max_iter` は SciPy に合わせて restart cycle の回数であり、べき乗法の反復回数とは同じ意味ではない。`restart` は Krylov 部分空間のサイズを変え、収束性とメモリに影響する。詳細は [SciPy GMRES](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html) を参照。

要求した残差上界を満たさないときは `RuntimeError` を送出する。丸め誤差のため達成不能な目標に対して、無条件に成功を返さない。

## 6. 減衰係数に関する微分

$p,q,H$ を固定し、$\alpha$ だけを変える。方程式を微分すると

$$
(I-\alpha T)\frac{dx_*}{d\alpha}=Tx_*-p,
$$

したがって

$$
\frac{dx_*}{d\alpha}=(I-\alpha T)^{-1}(Tx_*-p).
$$

`alpha_derivative(g, alpha=..., ...)` はこのベクトルを返す。`dangling=None` の $q=p$ でも、$\alpha$ の微分では $p,q$ はともに固定する。成分の和は理論上 0 である。

有限差分でも確認できるが、差分幅を小さくし過ぎると桁落ちが起こる。付属テストは閉形式が分かる二頂点グラフと中心差分の双方で検査する。減衰係数依存性を研究した一次資料として [Boldi, Santini, Vigna (2005)](https://doi.org/10.1145/1060745.1060827) がある。

## 7. personalization に関する方向微分

正規化後の $p$ に対して、$p(t)=p+th$、$\mathbf1^Th=0$ とする。`personalization_derivative(g, direction=h, ...)` はこの方向微分を返す。入力の未正規化重みそのものに対する微分ではない。

### 7.1 $q$ を明示して固定した場合

$T$ は変化しないので

$$
(I-\alpha T)\dot x=(1-\alpha)h.
$$

### 7.2 `dangling=None` の場合

デフォルトでは $q(t)=p(t)$ も変化する。$\dot T=hd^T$ だから

$$
\boxed{(I-\alpha T)\dot x=
\bigl[(1-\alpha)+\alpha d^Tx_*\bigr]h.}
$$

この追加項を落とすと、dangling node があるグラフの微分が間違う。付属テストは両方の条件を個別に有限差分と比較する。

$p_i=0$ のような確率単体の境界では、両方向に動けるとは限らない。式はアフィン拡張の微分として意味を持つが、確率分布として解釈するには $p+th\ge0$ を満たす側で変化させる。

## 8. 辺の重みに関する微分

非 dangling の頂点 $i$ から頂点 $j$ への集約重み $W_{ij}$ を変え、その他の生の重みを固定する。行正規化も変わるため、$H_{ij}$ だけを増やす問題とは異なる。

$$
\frac{\partial H_{ik}}{\partial W_{ij}}
=\frac{\mathbf1_{k=j}-H_{ik}}{s_i}.
$$

したがって

$$
\boxed{(I-\alpha T)\frac{\partial x_*}{\partial W_{ij}}
=\frac{\alpha x_{*,i}}{s_i}
\left(e_j-H_{i,:}^T\right).}
$$

`edge_derivative(g, source=i, target=j, ...)` は全頂点スコアの微分を返す。存在しない辺については、重み 0 から正方向に加える右微分として利用できる。

出強度 $s_i=0$ の頂点に最初の正の辺を追加すると、遷移規則が $q$ からその辺の行き先へ切り替わる。一般には連続ですらないため、dangling の頂点を source に指定した微分は拒否する。平行辺は集約済みの重みに対する微分として扱う。

非常に小さい生の重みでは微分値が大きくなる。これは同じ遷移確率に対応する重みでも単位によって微分の尺度が変わるためである。`float64` の範囲を超える場合は例外を送出する。必要なら行の重みの単位を変える。

### 8.1 全辺の目的関数勾配を一度に求める

目的関数を $J=c^Tx_*$ とする。随伴方程式

$$
(I-\alpha T)^T\lambda=c
$$

を一度解けば、各辺について

$$
\boxed{\frac{\partial J}{\partial W_{ij}}
=\frac{\alpha x_{*,i}}{s_i}
\left[\lambda_j-(H\lambda)_i\right]}
$$

が得られる。`edge_gradient(g, objective=c, ...)` は、元の隣接行列の格納辺に対応する CSR を返す。各辺について別々に線形方程式を解く必要がない。

戻り値の格納されていない場所は「未評価」であり、「存在しない辺の微分が 0」という意味ではない。その辺を調べる場合は `edge_derivative` を使う。

行全体の重みを同じ正数倍しても遷移確率は変わらない。したがって

$$
\sum_j W_{ij}\frac{\partial J}{\partial W_{ij}}=0
$$

という整合性条件が得られる。テストでは随伴勾配と個別微分の一致に加えて、この尺度不変性も確認する。

## 9. 有理数による厳密解

`exact_pagerank(g, alpha="17/20", ..., max_n=32)` は SymPy の有理数行列を使う。オプション依存は次で入れられる。

```bash
python -m pip install '.[analysis]'
```

```python
from pagerank_lab.analysis import exact_pagerank

print(exact_pagerank(g, alpha="1/2"))
# Matrix([[2/5], [3/5]])
```

**何に対して厳密かを確認する。** グラフの重みは `Graph` 作成時点で `float64` になっている。厳密関数は各重みの `str(float(weight))` が表す十進有理数として再解釈する。集約後の重みがfloat64で正確に表現できる整数なら、その整数グラフの厳密解になる。大きい整数や元々の $1/3$ がfloatに丸められた後で、元の値を復元するわけではない。

一方、`alpha`、`personalization`、`dangling` に直接渡す値には `"1/3"` のような有理数文字列を使える。分布の正規化も有理数で行う。密な記号計算になるため、既定の頂点数上限は 32 とし、変更には明示的な `max_n` を要求する。

## 10. スペクトルと経路長分解

### 10.1 全固有値

`spectrum(g, ..., max_n=512)` は Google 行列を小規模な密行列として作り、全固有値を絶対値降順に返す。

- `eigenvalues`: 全固有値。
- `subdominant_modulus`: 絶対値で二番目の固有値の絶対値。
- `spectral_gap`: $1-\text{subdominant_modulus}$。

定常固有値は 1 で、その他の固有値の絶対値は $\alpha$ 以下になる。スペクトルは漸近的な収束を調べる情報だが、非正規行列では固有値だけで有限回の誤差を評価できない。Jordan 構造や固有ベクトルの条件にも注意する。メモリは $O(n^2)$、固有値計算は一般に $O(n^3)$ なので、大きいグラフ用の関数ではない。

### 10.2 Neumann 級数

PageRank は

$$
x_*=(1-\alpha)\sum_{k=0}^{\infty}\alpha^kT^kp
$$

と分解できる。$k$ 項目は、teleportation 分布から $k$ 回遷移した確率分布の寄与である。dangling の再分配も $T$ の遷移に含む。

`neumann_series(g, terms=K, ...)` は、$k=0,\ldots,K-1$ の各寄与を `contributions` の列に格納し、その和 `partial_sum` を返す。正規化はしない。打ち切り誤差は非負で、その L1 質量は厳密演算ならちょうど

$$
\left\|x_*-(1-\alpha)\sum_{k=0}^{K-1}\alpha^kT^kp\right\|_1=\alpha^K
$$

である。`tail_l1_bound` にこの値を返す。列をすべて保存するため、メモリは $O(nK)$ になる。

PageRank を線形方程式や拡散として捉える広い整理には [Gleich, *PageRank Beyond the Web* (2015)](https://www.cs.purdue.edu/homes/dgleich/publications/Gleich%202015%20-%20prbeyond.pdf) がある。

## 11. 順位とスコアの比較

`rank_scores(scores, method="average")` は大きいスコアを 1 位とする。`average`、`min`、`max`、`dense`、`ordinal` を選べる。同点の処理を実験の都合に合わせて明記する。

`compare_scores(x, y, top_k=...)` は L1 距離、L∞ 距離、cosine、Spearman 相関、top-k の共通率を返す。内部で入力を勝手に正規化しない。定数ベクトルの順位相関やゼロベクトルの cosine は未定義なので `NaN` を返す。top-k の境界に同点がある場合は頂点番号の昇順で決め、常にちょうど $k$ 頂点を選ぶ。

小さいスコア差の順位は、数値誤差だけでも入れ替わる。L1 誤差上界を $E$ とすると、候補ベクトルで $\widehat{x}_i-\widehat{x}_j>E$ なら、厳密演算での上界が有効な限り $x_{*,i}>x_{*,j}$ を保証できる。差が $E$ 以下だから順位が誤りだとは限らない。その評価だけでは確定できないという意味である。

## 12. API 一覧

| 関数 | 目的 | 主な制約 |
| --- | --- | --- |
| `google_operator` | $Gv$、$G^Tv$、複数列の積 | 密行列を作らない |
| `residual` | L1 残差・理論誤差上界・確率性の診断 | 浮動小数点での診断値 |
| `solve_linear` | 疎 LU または GMRES による解 | LU fill-in、GMRES の収束 |
| `alpha_derivative` | $\alpha$ による微分 | 疎 LU を使用 |
| `personalization_derivative` | 確率単体の方向微分 | 方向ベクトルの総和 0 |
| `edge_derivative` | 一辺の生の重みによる微分 | source が非 dangling |
| `edge_gradient` | 線形目的関数の全格納辺勾配 | 戻り値の支持は格納辺 |
| `exact_pagerank` | 十進有理数としての厳密解 | SymPy、既定 $n\le32$ |
| `spectrum` | Google 行列の全固有値 | 密行列、既定 $n\le512$ |
| `neumann_series` | 経路長ごとの寄与と尾部質量 | $O(nK)$ メモリ |
| `iteration_bound` | 最悪反復数の事前見積もり | 厳密演算での式 |
| `rank_scores` | 同点規則付き順位 | 浮動小数点の値そのものを比較 |
| `compare_scores` | 距離、順位相関、top-k 共通率 | 定義できない相関は `NaN` |
