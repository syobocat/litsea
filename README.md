# Litsea

Litsea is an extremely compact word segmentation software implemented in Rust, inspired by [TinySegmenter](http://chasen.org/~taku/software/TinySegmenter/) and [TinySegmenterMaker](https://github.com/shogo82148/TinySegmenterMaker). Unlike traditional morphological analyzers such as [MeCab](https://taku910.github.io/mecab/) and [Lindera](https://github.com/lindera/lindera), Litsea does not rely on large-scale dictionaries but instead performs segmentation using a compact pre-trained model. It features a fast and safe Rust implementation along with a learner designed to be simple and highly extensible.

There is a small plant called Litsea cubeba (Aomoji) in the same camphoraceae family as Lindera (Kuromoji). This is the origin of the name Litsea.

## How to learn

Prepare a corpus with words separated by spaces in advance.

```text
Litsea は TinySegmenter を 参考 に 開発 され た 、 Rust で 実装 され た 極め て コンパクト な 単語 分割 ソフトウェア です 。
```

Extract the information and features from the corpus.

```shell
litsea extract < ./resources/corpus.txt > ./resources/features.txt
```

Train the features output by the above command using AdaBoost.
If the classification accuracy of the new weak classifier is 0.001 or less, and the number of repetitions is 10,000 or more, learning is terminated.

```shell
litsea train -t 0.001 -n 10000 ./resources/features.txt ./resources/model
```

## How to segment sentences into words

Use the learned model to segment sentences into words.

```shell
echo "LitseaはTinySegmenterを参考に開発された、Rustで実装された極めてコンパクトな単語分割ソフトウェアです。" | litsea segment ./resources/model
```
