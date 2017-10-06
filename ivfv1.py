# -*- coding:utf-8 -*-
"""Identify Vowels from Formants Version 1
   母音推定器、コマンドラインでwavファイル指定、エラー処理書いてない
"""
import sys
import wave
import numpy as np
import scipy.fftpack
import scipy.signal
from pylab import *

def wavread(filename):
    """音声データの情報を取得する
    """
    wf = wave.open(filename, "r")
    fs = wf.getframerate()
    x = wf.readframes(wf.getnframes())
    x = np.frombuffer(x, dtype="int16")/32768   # (-1, 1)に正規化
    wf.close()
    return x, fs

def autocorr(x, nlags=None):
    """自己相関関数を求める
    x:      信号
    nlags:  自己相関関数のサイズ（lag=0からnlags-1まで）
            引数がなければlag=0からlen(x)-1まですべて
    """
    N = len(x)
    if nlags == None: nlags = N
    r = np.zeros(nlags)
    for lag in range(nlags):
        for n in range(N - lag):
            r[lag] += x[n] * x[n + lag]
    return r

def LevinsonDurbin(r, lpcOrder):
    """Levinson-Durbinのアルゴリズム
    k次のLPC係数からk+1次のLPC係数を再帰的に計算
    """
    # a[0]が1固定なのでlpcOrder個の係数を得るには+1
    a = np.zeros(lpcOrder + 1)
    e = np.zeros(lpcOrder + 1)

    # when k = 1
    a[0] = 1.0
    a[1] = -r[1] / r[0]
    e[1] = r[0] + r[1] * a[1]

    # kの場合からk+1の場合を再帰的に求める
    for k in range(1, lpcOrder):
        # lambda更新
        lam = 0.0
        for j in range(k+1):
            lam -= a[j] * r[k+1-j]
        lam /= e[k]

        # UとVからaを更新
        U = [1]
        U.extend([a[i] for i in range(1, k+1)])
        U.append(0)

        V = [0]
        V.extend([a[i] for i in range(k, 0, -1)])
        V.append(1)

        a = np.array(U) + lam * np.array(V)

        # eを更新
        e[k+1] = e[k] * (1 - lam * lam)

    return a, e[-1]

def eucd2(x1, x2, y1, y2):
    """ユークリッド距離の2乗を計算する
    """
    return (x1-y1)*(x1-y1) + (x2-y2)*(x2-y2)

def vowel(f1, f2):
    """母音を推定する
    """
    d = []
    ave_formant_freq = [[240, 2400],    # i
                        [235, 2100],    # y
                        [390, 2300],    # e
                        [370, 1900],    # ø
                        [610, 1900],    # ɛ
                        [585, 1710],    # œ
                        [850, 1610],    # a(front)
                        [820, 1530],    # ɶ
                        [750, 940],     # ɑ(back)
                        [700, 760],     # ɒ
                        [600, 1170],    # ʌ
                        [500, 700],     # ɔ
                        [460, 1310],    # ɤ
                        [360, 640],     # o
                        [300, 1390],    # ɯ
                        [250, 595]]     # u
    for avef1, avef2 in ave_formant_freq:
        d.append(eucd2(f1, f2, avef1, avef2))

    min_vowel = np.argmin(d)
    if min_vowel == 0:
        return "[i]"
    elif min_vowel == 1:
        return "[y]"
    elif min_vowel == 2:
        return "[e]"
    elif min_vowel == 3:  # ø
        return "[ø]"
    elif min_vowel == 4:  # ɛ
        return "[ɛ]"
    elif min_vowel == 5:  # œ
        return "[œ]"
    elif min_vowel == 6:  # a(front)
        return "[a]"
    elif min_vowel == 7:  # ɶ
        return "[ɶ]"
    elif min_vowel == 8:  # ɑ(back)
        return "[ɑ]"
    elif min_vowel == 9:  # ɒ
        return "[ɒ]"
    elif min_vowel == 10:  # ʌ
        return "[ʌ]"
    elif min_vowel == 11:  # ɔ
        return "[ɔ]"
    elif min_vowel == 12:  # ɤ
        return "[ɤ]"
    elif min_vowel == 13:
        return "[o]"
    elif min_vowel == 14:  # ɯ
        return "[ɯ]"
    elif min_vowel == 15:
        return "[u]"
    else:
        return "-"

if __name__ == "__main__":
    # waveファイル読み込み(そのうちリアルタイムにしたい)
    sys.path.append('/path/to/dir')
    speech = sys.argv[1]

    # 音声データの情報取得
    wav, fs = wavread(speech)

    # ハミング窓かけて周期波形に
    start = 310000       # サンプリング開始位置座標
    N = 8192         # FFTサンプル数
    SHIFT = 1024     # 窓関数をずらすサンプル数

    freqList = scipy.fftpack.fftfreq(N, d=1/fs)[:int(N/2)]     # 周波数軸の値を計算
    """このへん不合理"""
    x = np.hamming(N) * wav[start:start+N]        # ハミング窓をかける

    # LPC係数計算
    lpcOrder = 64
    r = autocorr(x, lpcOrder + 1)
    a, e = LevinsonDurbin(r, lpcOrder)

    # LPCスペクトル包絡線を求める
    w, h = scipy.signal.freqz(np.sqrt(e), a, N, "whole")
    h = np.abs(h)

    # フォルマント（F1,F2）特定
    peak = scipy.signal.argrelmax(h, order=10)[0]
    f1 = freqList[peak[0]]
    f2 = freqList[peak[1]]

    # 母音推定
    print(vowel(f1, f2), "(f1:", f1, ", f2:", f2, ")")
    
    
