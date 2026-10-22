# ビルドと移植性

Python 3.10 以上と C コンパイラを使用する。`pip install .` はビルド用の隔離環境に Cython、NumPy、setuptools を用意して C 拡張を生成する。実行時の必須依存は NumPy と SciPy である。インターネットを使わない環境では、対応する wheel などの依存を事前に用意する。

## 標準ビルド

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[all]"
python -m pytest
```

Windows では仮想環境の有効化を使用するシェルに合わせる。例えば PowerShell なら `.venv\Scripts\Activate.ps1` となる。Windows のビルドには、使用する Python と互換性のある Microsoft C++ Build Tools が必要である。

既定ビルドは最適化を有効にするが、OpenMP と CPU 固有命令を要求しない。`fast-math` は設定しない。浮動小数点演算の並べ替えを無条件に許可して速度だけを比較する構成ではない。

## Linux / GCC

Python のビルド情報には、Python 自体を作ったコンパイラの名前が記録されている。そのコンパイラが実行環境にない場合は、環境変数で実際にインストールされているコンパイラを指定する。

この検証環境では Python が参照する `clang` コマンドを利用できず、以下のように GCC を指定した。

```bash
CC=gcc LDSHARED='gcc -shared' \
PAGERANK_OPENMP=1 PAGERANK_NATIVE=1 \
python -m pip install --no-cache-dir --force-reinstall --no-deps .
```

`CC` と `LDSHARED` はこの環境に対応する例であり、全 OS 共通の指定ではない。GCC が未導入なら、先に OS のパッケージ管理で C コンパイラと Python の開発ヘッダを用意する。

OpenMP のみを有効にした汎用 CPU 向けのビルドは、`PAGERANK_NATIVE=1` を外して行う。

```bash
CC=gcc LDSHARED='gcc -shared' PAGERANK_OPENMP=1 \
python -m pip install --no-cache-dir --force-reinstall --no-deps .
```

## macOS / Clang

既定の単一スレッドビルドは、通常は Xcode Command Line Tools の Clang を利用できる。OpenMP は別途 libomp が必要である。インストール先を `PAGERANK_OPENMP_PREFIX` で渡せる。

```bash
PAGERANK_OPENMP=1 PAGERANK_OPENMP_PREFIX=/path/to/libomp \
python -m pip install --no-cache-dir --force-reinstall --no-deps .
```

このときコンパイルに `-Xpreprocessor -fopenmp`、リンクに `-lomp` を使い、指定された prefix の `include` と `lib` を検索する。prefix は各環境の実際のインストール先に置き換える。

## Windows / MSVC

OpenMP を有効にする PowerShell の例を示す。

```powershell
$env:PAGERANK_OPENMP = "1"
python -m pip install --no-cache-dir --force-reinstall --no-deps .
```

MSVC では `/O2`、OpenMP を有効にした場合は `/openmp` を使う。`PAGERANK_NATIVE=1` は Windows では無視して警告する。Linux 向けの `-march=native` を MSVC へ流用しない。

## ビルド情報と再ビルド

```bash
python -c 'from pagerank_lab import build_info; print(build_info())'
```

OpenMP が無効なビルドでは、`threads > 1` を指定すると明示的なエラーになる。スレッド数の指定を無視して並列化したように見せる動作はしない。

環境変数を変更した後は拡張の再ビルドが必要である。`--no-cache-dir` は以前の wheel のキャッシュを避けるために指定している。開発中に同じビルドディレクトリを再利用する場合は、以下のように強制的に再コンパイルできる。このコマンドには、現在の環境への Cython、setuptools、NumPy のインストールが必要である。

```bash
PAGERANK_OPENMP=1 python setup.py build_ext --inplace --force
```

インストールのためには `pip install` を使う。上の `setup.py` コマンドは開発中の拡張の再コンパイル用である。

## CPU 固有ビルドの配布範囲

`PAGERANK_NATIVE=1` は Unix 系のコンパイラへ `-march=native` を渡す。コンパイルした CPU の命令を使えるため、その wheel や共有ライブラリを別の CPU へコピーすると実行できない場合がある。配布用 wheel は通常ビルドで作り、各利用者が自身のマシンで速度を測定する。ZIP にはビルド済みバイナリを含めず、ソースからビルドする。

OpenMP のランタイムやコンパイラも性能に影響する。ベンチマークの再現時は CPU、Python、NumPy、SciPy、Cython、コンパイラ、ビルドフラグ、スレッド数を合わせる。OS ごとの CI 定義は同梱しているが、定義しただけで全環境の実行確認が済んだことにはならない。

