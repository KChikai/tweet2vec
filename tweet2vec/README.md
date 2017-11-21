# Tweet2vec

tweet2vecを自分用にいじったもの．
元実装は[ここ][tweet2vec]．


## How to use

コマンドはgpu,cpu選択可能．floatは64ではなく32．


1. __Encoder__ - 大量のツイートデータの分散行列を生成，タグの予測を行う．

    ```
    $ THEANO_FLAGS=device=gpu,floatX=float32 sh tweet2vec_encoder.sh
    ```

2. __Trainer__ - トレー二ングデータから学習を行う．

    ```
    $ THEANO_FLAGS=device=gpu,floatX=float32 sh tweet2vec_trainer.sh
    ```


3. __Tester__ - テストを行う．

    ```
    $ THEANO_FLAGS=device=gpu,floatX=float32 sh tweet2vec_tester.sh
    ```

4. __Predicter__ - 学習済みモデルから，ユーザが入力したデータのタグを予測するスクリプト

    ```
    $ python predict.py
    ```


[tweet2vec]: https://github.com/bdhingra/tweet2vec