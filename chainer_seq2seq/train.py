# coding: utf-8

import numpy as np
from seq2seq import Seq2Seq
from forward import forward
from chainer import optimizer, optimizers, serializers, Variable
import random, sys, os, csv, collections

sd = os.path.dirname(__file__)
sys.path.append(sd)

argvs = sys.argv
argc = len(argvs)

print(argvs)
print(argc)
if(argc != 3):
    print("usage: python train.py [enc filename] [dec filename]")
    quit()

VOCAB_SIZE = 10000
EMBED_SIZE = 300
HIDDEN_SIZE = 150
BATCH_SIZE = 40
EPOCH_NUM = 2

enc_dict = collections.defaultdict(lambda: len(enc_dict))
dec_dict = collections.defaultdict(lambda: len(dec_dict))

def output_dict(outdict, filename):
    with open('%s.csv'%(filename), 'w') as fd:
        writer = csv.writer(fd)
        writer.writerows(outdict.items())

def make_data():
    data = []

    with open(argvs[1], 'r') as f_enc:
        enc_reader = csv.reader(f_enc)
        for enc_row in enc_reader:
            # data.append([enc_row[0].split()])
            data.append([[enc_dict[word.lower()] for word in enc_row[0].split()]])
    output_dict(dict(enc_dict), 'JEC_bs_ja_id')

    with open(argvs[2], 'r') as f_dec:
        dec_reader = csv.reader(f_dec)
        ct = 0
        for dec_row in dec_reader:
            # data[ct].append(dec_row[0].split())
            data[ct].append([dec_dict[word] for word in dec_row[0].split()])
            ct += 1
    output_dict(dict(dec_dict), 'JEC_bs_en_id')

    return data

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
        data = make_data()
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
            opt.use_cleargrads(use=False)
        print('epoch %s is ended' %(epoch+1))
        # epochごとにモデル保存
        outputpath = 'ja2en_EMB%s_H%s_B%s_EP%s.weights'%(EMBED_SIZE, HIDDEN_SIZE, BATCH_SIZE, epoch+1)
        serializers.save_hdf5(outputpath, model)

print('start')
train()
print('end')
