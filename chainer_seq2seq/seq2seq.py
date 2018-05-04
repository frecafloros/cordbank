# coding: utf-8

import numpy as np
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
from encoder import LSTM_Encoder
from decoder import LSTM_Decoder

class Seq2Seq(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size):
        """
        初期化
        vocab_size: 語彙数
        embed_size: 単語ベクトルのサイズ
        hidden_size:中間ベクトルのサイズ
        batch_size: ミニバッチのサイズ
        """
        super(Seq2Seq, self).__init__(
        # with self.init_scope:
            # encoder, decoderのインスタンス化
            encoder = LSTM_Encoder(vocab_size, embed_size, hidden_size),
            decoder = LSTM_Decoder(vocab_size, embed_size, hidden_size)
        )
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.ARR = np

    def encode(self, words):
        """
        Encoder計算部分
        words:  単語が記録されたリスト（入力）
        """
        # 内部メモリ、中間ベクトルの初期化
        c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

        #　エンコーダーに単語を順番に読み込ませる
        for w in words:
            c, h = self.encoder(w, c, h)

        # 中間ベクトルをインスタンス変数にして、デコーダーに引継ぎ
        self.h = h

        # 内部メモリは初期化
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

    def decode(self, w):
        """
        Decoder計算部分
        w:  単語
        return: 単語数サイズのベクトルを出力
        """
        # print(w, w.dtype, "2")
        t, self.c, self.h = self.decoder(w, self.c, self.h)
        return t

    def reset(self):
        """
        中間ベクトルh、内部メモリc、勾配の初期化
        """
        self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        self.zerograds()
