# coding:utf-8

import numpy as np
from seq2seq import Seq2Seq
from chainer import Variable, serializers
import sys, os, csv

sd = os.path.dirname(__file__)
sys.path.append(sd)

argvs = sys.argv
argc = len(argvs)

print(argvs)
print(argc)
if(argc != 4):
    print("usage: python train.py [model] [enc_dict] [dec_dict]")
    quit()


def inport_dict(filename):
    indict = {}
    with open('%s'%(filename), 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            indict.update({row[0]:int(row[1])})
    return indict

def translate(enc_words, model, ARR, batch_size):
    # modelの勾配をリセット
    model.reset()
    # 発話リスト内の単語をVariable型に変更
    # enc_words = Variable(ARR.array([[-1]*(256-len(enc_words)) + enc_words], dtype='int32'))
    enc_words = [Variable(ARR.array([word], dtype='int32')) for word in enc_words]
    # print(enc_words)
    # エンコードの計算
    model.encode(enc_words)
    # <eos>をデコーダーに読み込ませる?
    # eos = Variable(ARR.array([1 for _ in range(batch_size)], dtype='int32'))
    eos = Variable(ARR.array([1], dtype='int32'))

    y1 = eos
    dec_words = []
    while len(dec_words)<50:
        # 1単語ずつデコードする
        # print(y1, "1")
        y2 = model.decode(y1)
        # print(y2, "5")
        y2 = y2.data.argmax(1)[0]
        dec_words.append(y2)
        # print(y2, "6")
        if(y2 == 2):
            break
        # y1 = Variable(ARR.array([y2 for _ in range(batch_size)], dtype='int32'))
        y1 = Variable(ARR.array([y2], dtype='int32'))

    return dec_words


# 辞書の読み込み
enc_dict = inport_dict(argvs[2])
dec_dict = inport_dict(argvs[3])
out_dict = dict([(v, k) for k, v in dec_dict.items()])


input_words = '0'
while(1):
    input_words = raw_input('入力 > ')

    if(input_words == 'q'):
        quit()

    # 分かち書き
    input_words = input_words.split()
    # 単語id変換
    enc_words = []
    for item in input_words:
        enc_words.append(enc_dict[item])

    # print(enc_words)

    # こいつらの意味があるかはちとわからん
    VOCAB_SIZE = 8192
    EMBED_SIZE = 300
    HIDDEN_SIZE = 150
    BATCH_SIZE = 1

    # モデルの読み込み
    model = Seq2Seq(vocab_size=VOCAB_SIZE,
                    embed_size=EMBED_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    batch_size=BATCH_SIZE)
    serializers.load_npz(argvs[1], model)

    dec_words = translate(enc_words, model, np, BATCH_SIZE)
    sys.stdout.write('出力 > ')
    outputs = [unicode(out_dict[a], "utf-8") for a in dec_words]
    for a in range(0, len(outputs)):
        sys.stdout.write(outputs[a].encode('utf-8') + ' ')
    print('')
