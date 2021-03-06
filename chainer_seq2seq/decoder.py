# coding: utf-8

import numpy as np
from chainer import Chain
import chainer.functions as F
import chainer.links as L

class LSTM_Decoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        クラスの初期化と層の記述
        vocab_size:  語彙数
        embed_size:  単語をベクトル表現したときのベクトルのサイズ
        hidden_size: 隠れ層のサイズ
        """
        super(LSTM_Decoder, self).__init__(
        # with self.init_scope():
            # word 2 vector(embedded) 層
            ye = L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            # embeddedをhiddenの4倍のサイズに拡大する層
            eh = L.Linear(embed_size, 4*hidden_size),
            # hiddenを4倍のサイズにする層
            hh = L.Linear(hidden_size, 4*hidden_size),
            # 出力vectorを単語vectorのサイズに変換する層
            he = L.Linear(hidden_size, embed_size),
            # 単語vectorを語彙サイズのvectorに変換する層
            ey = L.Linear(embed_size, vocab_size)
        )

    def __call__(self, y, c, h):
        """
        encoderの動作、層を配線する
        y: hot な vector
        c: 内部メモリ
        h: 隠れ層

        予測単語、次代の内部メモリと隠れ層を返す
        """
        # yeで単語vectorに変換、tanhにかける
        # print(self.ye(y), self.ye(y).dtype, "3")
        e = F.tanh(self.ye(y))
        # 前の内部メモリ、単語vector*4 + hidden vector*4を入力
        c, h = F.lstm(c, self.eh(e)+self.hh(h))
        # 出力されたhidden vectorを単語vector -> 出力vectorのサイズにリサイズ
        t = self.ey(F.tanh(self.he(h)))
        # print(t.data.argmax(1)[0], "4")
        # t = t.data.argmax(1)[0]
        return t, c, h
