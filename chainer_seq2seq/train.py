# coding: utf-8

import numpy as np
from seq2seq import Seq2Seq
from chainer import optimizer, optimizers, serializers, Variable
import random

VOCAB_SIZE = 1
EMBED_SIZE = 1
HIDDEN_SIZE = 1
BATCH_SIZE = 1
EPOCH_NUM = 1

def make_minibatch(minibatch):
    # enc_wordsの作成
    enc_words = [row[0] for row in minibatch]
    enc_max = np.max([len(row) for row in enc_words])
    enc_words = np.array([[-1]*(enc_max -len(row)) + row for row in enc_words], dtype='int32')
    enc_words = enc_words.T

    # dec_wordsの作成
    dec_words = [row[1] for row in minibatch]
    dec_max = np.max([len(row) for row in dec_words])
    dec_words = np.array([row + [-1]*(dec_max - len(row)) for row in dec_words], dtype='int32')
    dec_words = dec_words.T

    return enc_words, dec_words

def train():
    # モデル(Seq2Seq)のインスタンス化
    model = Seq2Seq(vocab_size=VOCAB_SIZE,
                    embed_size=EMBED_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    batch_size=BATCH_SIZE)
    # モデルの初期化
    model.reset()
    # GPU or CPU(今回はCPUのみ実装)
    ARR = np

    # 学習開始
    for epoch in range(EPOCH_NUM):
        # epochごとにoptimizerの初期化
        opt = optimizers.Adam()
        # modelをoptimizerにセット
        opt.setup(model)
        # 勾配調整
        opt.add_hook(optimizer.GradientClipping(5))

        # 学習データ読み込み
        data = # うまいこと読み込む
        # データのシャッフル
        random.shuffle(data)

        # バッチ学習スタート
        for num in range(len(data)//BATCH_SIZE):
            # minibatch作成
            minibatch = data[num*BATCH_SIZE: (num+1)*BATCH_SIZE]
            # 読み込み用のデータ作成
            enc_words, dec_words = make_minibatch(minibatch)
            # 順伝搬で損失計算
            total_loss = forward(enc_words=enc_words,
                                 dec_words=dec_words,
                                 model=model,
                                 ARR=ARR)
            # 誤差逆伝搬で勾配計算
            total_loss.backward()
            # 計算した勾配を使ってネットワークを更新
            opt.update()
            # 記録された勾配を初期化
            opt.zero_grads()
        # epochごとにモデル保存
        serializers.save_hdf5(outputpath, model)
