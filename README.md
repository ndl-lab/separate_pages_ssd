# NDLOCR用ページ分割モジュール

本リポジトリには、見開きのページをのど元で分割するための学習プログラムと推論プログラムが含まれています。

本プログラムは、国立国会図書館が株式会社モルフォAIソリューションズに委託して作成したものです。

本プログラムは、[rykov8's repo](https://github.com/rykov8/ssd_keras)を改変して作成されたNDLラボが提供する次のリポジトリの、inference_divided.pyに対して必要な改変を加えたものです。

[ndl-lab/ssd_keras](https://github.com/ndl-lab/ssd_keras)
- 変更箇所: `inference_divided.py` ([ndl-lab/ssd_keras](https://github.com/ndl-lab/ssd_keras)でのファイル名は `inference_devided.py`)

# 環境設定
python3.7環境で

```
pip install -r requirements.txt 
```

を実行する。

# 使い方
## 推論

inference_inputディレクトリ(`-i` オプションで変更可能)にのど元を分割したい画像を入れ、`inference_divided.py`を実行する。inference_outputディレクトリ(-o オプションで変更可能)に分割後の画像が出力する。分割後の画像ファイル名は元画像ファイル名+`LEFT` or `RIGHT`(デフォルトでは `_01` と `_02`)となる。入力画像にノド元が検出されなかった場合、画像は分割されずに、元画像ファイル名+`SINGLE`(デフォルトでは `_00`)で出力する。

```
$ python3 inference_divided.py [-i INPUT] [-o OUTPUT] [-l LEFT] [-r RIGHT] [-s SINGLE] [-e EXT] [-q QUALITY]
```

optional arguments:
```
  -h, --help    ヘルプメッセージを表示して終了
  -i INPUT, --input INPUT
                入力画像または入力画像を格納したディレクトリのパス
                (default: inference_input)
  -o OUT, --out OUT
                出力画像を保存するディレクトリのパス (default: inference_output)
                また、"NO_DUMP"を指定した場合、ノド元で分割した画像を出力しない。
                後述のlogオプションと組み合わせることで画像出力を省略し、ノド元位置のみを取得できる。
  -l LEFT, --left LEFT
                左ページの出力画像のファイル名の末尾につけるフッター
                例）input image:  input.jpg, LEFT: _01(default)
                    output image: input_01.jpg
  -r RIGHT, --right RIGHT
                右ページの出力画像のファイル名の末尾につけるフッター
                例）input image:  input.jpg, RIGHT: _02(default)
                    output image: input_02.jpg
  -s SINGLE, --single SINGLE
                入力画像でノド元が検出されなかった場合に出力する画像ファイル名の末尾に着けるフッター
                例）input image:  input.jpg, SINGLE: _00(default)
                    output image: input_00.jpg
  -e EXT, --ext EXT     
                出力画像の拡張子。 (default: .jpg)
                ただし、"SAME"とした場合は入力画像と同じ拡張子を使用する。
  -q QUALITY, --quality QUALITY
                Jpeg画像出力時の画質。1~100の整数値で指定する。
                1が最低画質で最小ファイルサイズ、100が最高画質で最大ファイルサイズ。
                default: 100
  --short SHORT 出力画像の短辺の長さ。アスペクト比は維持したままリサイズする。
                指定しなかった場合オリジナルサイズで出力される。
  --lg LOG, --log LOG
                検出したノド元位置を記録するtsvファイルのパス。未指定の場合、出力しない。
                1行目に列名 image_name<tab>trimming_x
                2行目以降に入力画像のファイル名と検出したノド元位置を記録する。
                指定したtsvファイルが既に存在しているときは、入力ファイル名とノド元位置を追記する。
```

# 追加の学習手順

## 1. 学習ファイルの準備
学習させたい画像ファイルをtraining/imgに、
のど位置情報をtraining/image.tsvにそれぞれ用意しておく。

※例では　(ファイル名)\t(中心からのずれの割合)
としていますが、tsvの形式に応じてtraining/make_pkl_for_page.pyをカスタマイズしてください。

```
training/size_convertion.py
```

を実行し、画像のサイズを300*300に変換しておく

## 2. pklの生成

```
training/smake_pkl_for_page.py
```

を実行し、page_layout.pklを生成しておく。


## 3. 学習

```
training/train.py
```

を実行し、学習を開始する。
checkpointsディレクトリに学習済weightsファイルが生成される。
