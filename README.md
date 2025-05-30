# Litsea

Litsea is an extremely compact word segmentation software implemented in Rust, inspired by [TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/) and [TinySegmenterMaker](https://github.com/shogo82148/TinySegmenterMaker). Unlike traditional morphological analyzers such as [MeCab](https://taku910.github.io/mecab/) and [Lindera](https://github.com/lindera/lindera), Litsea does not rely on large-scale dictionaries but instead performs segmentation using a compact pre-trained model. It features a fast and safe Rust implementation along with a learner designed to be simple and highly extensible.

There is a small plant called Litsea cubeba (Aomoji) in the same camphoraceae family as Lindera (Kuromoji). This is the origin of the name Litsea.

## How to train models

Prepare a corpus with words separated by spaces in advance.

coupus.txt

```text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 、 Rust で 実装 さ れ た 極めて コンパクト な 単語 分割 ソフトウェア です 。

```

Extract the information and features from the corpus.

```shell
litsea extract ./resources/corpus.txt ./resources/features.txt
```

Train the features output by the above command using AdaBoost.
If the classification accuracy of the new weak classifier is 0.001 or less, and the number of repetitions is 10,000 or more, learning is terminated.

```shell
litsea train -t 0.001 -i 10000 ./resources/features.txt ./resources/model
```

The result of executing `train` command is as follows.

```text
finding instances...: 61 instances found

Iteration 9999 - margin: 0.16068839956263622
Result:
Accuracy: 100.00% (61 / 61)
Precision: 100.00% (24 / 24)
Recall: 100.00% (24 / 24)
Confusion Matrix: TP: 24, FP: 0, FN: 0, TN: 37
```

## How to segment sentences into words

Use the learned model to segment sentences into words.

```shell
echo "LitseaはTinySegmenterを参考に開発された、Rustで実装された極めてコンパクトな単語分割ソフトウェアです。" | litsea segment ./resources/model
```

The result of executing `segment` command is as follows.

'''text
Litsea は TinySegmenter を 参考 に 開発 さ れ た 、 Rust で 実装 さ れ た 極めて コンパクト な 単語 分割 ソフトウェア です 。
'''

## Pre-trained models

- JEITA\_Genpaku\_ChaSen\_IPAdic.model  
It is a model trained using the morphologically analyzed corpus published by the Japan Electronics and Information Technology Industries Association (JEITA).
We used the [Project Sugita Genpaku](http://www.genpaku.org/) analyzed with ChaSen+IPAdic.

- RWCP.model  
It is extracted from the original [TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/)
and contains only the model part.

## How to retrain existing models

You can resume learning from existing trained models and new corpora to improve performance.

```shell
litsea train -t 0.001 -i 10000 -m ./resources/model ./resources/new_features.txt ./resources/new_model
```
