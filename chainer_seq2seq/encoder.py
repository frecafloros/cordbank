# coding: utf-8

import numpy as np
from chainer import Chain
import chainer.functions as F
import chainer.links as L

class LSTM_Encoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        クラスの初期化と層の記述
        vocab_size:  語彙数
        embed_size:  単語をベクトル表現したときのベクトルのサイズ
        hidden_size: 隠れ層のサイズ
        """
        super(LSTM_Encoder).__init__()
        with self.init_scope():
            # word 2 vector(embedded) 層
            self.xe = L.EmbedID(vocab_size, embed_size, ignore_label=-1)
            # embeddedをhiddenの4倍のサイズに拡大する層
            self.eh = L.Linear(embed_size, 4*hidden_size)
            # hiddenを4倍のサイズにする層
            self.hh = L.Linear(hidden_size, 4*hidden_size)

    def __call__(self, x, c, h):
        """
        encoderの動作、層を配線する
        x: 入力vector
        c: 内部メモリ
        h: 隠れ層

        次代の内部メモリと隠れ層を返す
        """
        # xeで単語vectorに変換、tanhにかける
        e = F.tanh(self.xe(x))
        # 前の内部メモリ、単語vector*4 + hidden vector*4を入力
        return F.lstm(c, self.eh(e)+self.hh(h))
