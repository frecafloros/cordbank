# coding: utf-8

import numpy as np
from chainer import Variable
import chainer.functions as F
from seq2seq import Seq2Seq

def forward(enc_words, dec_words, model, ARR):
    """
    順伝搬の計算
    enc_words:  入力文の単語を記録したリスト
    dec_words:  出力文の単語を記録したリスト
    model:      seq2seqのインスタンス
    ARR:        ここではnumpy
    return:     計算した損失の合計
    """
    # バッチサイズを記録
    batch_size = len(enc_words[0])
    # modelの勾配をリセット
    model.reset()
    # 発話リスト内の単語をVariable型に変更
    enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]

    # エンコードの計算(1)
    model.encode(enc_words)
    # 損失の初期化
    loss = Variable(ARR.zeros((), dtype='float32'))

    # <s>をデコーダーに読み込ませる(2)
    t = Variable(ARR.array([1 for _ in range(batch_size)], dtype='int32'))

    # デコーダーの計算
    for w in dec_words:
        # 1単語ずつデコードする(3)
        y = model.decode(t)
        # print("decode[y]:", y)
        # 正解単語をVariable型に変換
        t = Variable(ARR.array(w, dtype='int32'))
        # print("answer[t]:", t)
        # 正解単語と予測単語を照らし合わせて損失を計算(4)
        loss += F.softmax_cross_entropy(y,t)

    return loss
