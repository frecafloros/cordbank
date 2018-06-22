from nltk import CFG

"""
V：動詞
St：動詞語幹
DVo：派生態動詞
VoA：能動系
VoF：強勢・使役系
VoF1：強勢・使役系１
VoF2：強勢・使役系２
VO：原動詞
VS：動詞文
T：終止形
"""

grammar = nltk.CFG.fromstring("""
V -> St DVo | St VO
DVo -> VoA VO | VoA VoF | VoF
VoA -> VoA1 | VoA2 | VoA3
VoA1 -> 'e' | 're'
VoA2 -> 'ar' | 'rar'
VoA3 -> 'are' | 'rare'
VoF -> VoF1 VO | VoF1 VoF2
VoF1 -> VoF11 | VoF12
VoF11 -> 'as' | 'sas'
VoF12 -> 'ase' | 'sase'
VoF2 -> VoA1 VO | VoA2 VO | VoA3 VO
VO -> VS
VS -> T
T -> 'u' | 'ru'
St -> '飲m'
""")

verb = ['飲m', 'ase', 'rare', 'ru']
