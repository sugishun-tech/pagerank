# 実行済みの検証

検証日: 2026年10月23日（JST）。実際に実行したOSはLinux x86-64、Pythonは3.12.14、CコンパイラはGCC 13.3.0。依存パッケージの版は `requirements-benchmark.txt`、性能測定時の環境は各 `benchmark_results/*/results.json` に記録している。

## ビルドとテスト

| ビルド | 確認内容 | 結果 |
|---|---|---|
| OpenMP + native | Cython拡張をGCCでビルドし、公開API・数値解析・CLI・ベンチマーク条件の全テストを実行 | **140 passed**、失敗・skipなし |
| 標準ビルド | クリーンなソースからsdistを作り、そのsdistからwheelをビルド。OpenMPなし、CPU固有オプションなし | sdist / wheel生成成功 |
| 標準wheelの独立インストール | 別ディレクトリへwheelをインストールし、読み込み元と `build_info()` を確認して同じ全テストを実行 | **139 passed, 1 skipped**。skipはOpenMP並列テストのみ |

どちらのビルドも `fast_math=False`。標準wheelでは `openmp=False, native=False`、実測用ビルドでは `openmp=True, native=True` を確認した。テストのJUnit記録は `validation/native-pytest.xml` と `validation/portable-pytest.xml` にある。

`python examples/basic.py` と `python examples/research.py` も実行済み。主な検証内容は次の通り。

- 小規模グラフの独立した密行列解・有理数解との照合
- 重み付き辺、重複辺、自己ループ、孤立点、dangling、独立した再分配分布
- ノード置換による同変性、各行の重みを定数倍してもPageRankが変わらないこと
- Cython / SciPy / direct / GMRESの整合性と、並列・逐次の一致
- alpha・personalization・辺重みの導関数の有限差分検査、随伴勾配の恒等式
- float32 / float64、int32 / int64、精度不足・非収束・極端な数値の扱い
- CSR等の疎行列構造と不正な添字の検査。破損したCSC入力は別プロセスで検査し、ネイティブ変換前に例外となることを確認
- CSV / NPZ / Matrix Marketの読み書き、CLIの終了コード、ベンチマークの共通精度条件

GitHub Actionsの設定にはLinux / macOS / Windows、Python 3.10 / 3.12を用意した。**ここで実行済みとする結果はLinux / Python 3.12だけ**であり、他のOS・Python版のCI成功は確認していない。標準wheelの検証も同じLinux環境上で行ったもので、異なるCPUでの実測ではない。

## 性能測定結果の検査

すべて実際に実行した記録である。各手法・条件で2回のwarmup後に7回計測し、順序を無作為化した。以下の試行数はwarmupを含まない。

| 結果ディレクトリ | グラフ・alphaの条件数 | 計測した手法との組合せ数 | 本計測の試行数 | 共通精度検査 |
|---|---:|---:|---:|---|
| `main` | 12 | 76 | 532 | 全件合格 |
| `stanford` | 1 | 6 | 42 | 全件合格 |
| `ablation` | 3 | 12 | 84 | 全件合格 |
| `high_alpha` | 2 | 13 | 91 | 全件合格 |
| `million` | 1 | 5 | 35 | 全件合格 |
| `weighted_personalized` | 1 | 4 | 28 | 全件合格 |
| **合計** | **20** | **116** | **812** | **全件合格** |

共通精度検査は、元のfloat64グラフによる `||F(x)-x||₁/(1-alpha) <= 1e-10`、非負性、確率質量の数値誤差検査。全812試行で、この事後誤差上界の最大値は約 `9.7783e-11` だった。残差評価も浮動小数点であり、区間演算による丸め誤差込みの証明ではない。

意図してスキップした手法は試行数に含めない。例えば純Python参照実装の大規模ケース、疎LUの頂点数上限を超えたケース、igraphで独立したdangling分布を指定できないケースである。スキップ理由は生のJSONに残している。

機械可読な集計と各結果JSONのSHA-256は `validation/summary.json` にある。結果の保存先と再実行コマンドは [benchmark_results/README.md](../benchmark_results/README.md)、比較の精度・計時範囲・限界は [benchmarking.md](benchmarking.md) を参照。

## 解釈の範囲

一つのコンテナ環境での検証であり、全グラフ・全マシンに対する最速性は示していない。IQRは試行のばらつきで、信頼区間ではない。測定したRSSはプロセス全体の累積最高値であり、手法別メモリ使用量ではない。最終配布物にはテスト後の説明文書の追加を含むが、標準wheelの検証後に数値実装は変更していない。
