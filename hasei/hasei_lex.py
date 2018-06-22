# coding: utf-8
import ply.lex as lex

tokens = (
    # 態詞
    'CAUSATIVE',    # 使役態
    'PASSIVE',      # 受動態
    'PERMISSIVE',   # 許容態
    # 動詞文派生
    'TERMINAL',     # 終止形
    'ASSUMPTION',   # 仮定条件
    'MEIREI',       # 命令
    'PERFECT',       # 完了
    'FORMAL',       # 敬体
    # 否定
    'NEGATION',     # 否定辞
    'FORMALNEG',    # 否定敬体
    # 助動詞文派生
    'DESCRIPTIVE',  # 形容詞終止（描写詞）
    'SUPPOSITION',  # 推量
    # 断定派生
)

# token rule
t_CAUSATIVE = r's?as'
t_PASSIVE = r'r?a(r|t)'
t_PERMISSIVE = r'r?e(?!(n$|ba))'
t_TERMINAL = r'r?u'
t_ASSUMPTION = r'r?eba'
t_MEIREI = r'(e|ro)$'
t_PERFECT = r'(t|d)a'
t_FORMAL = r'i?mas'
t_NEGATION = r'a?nak?'
t_FORMALNEG = r'en$'
t_DESCRIPTIVE = r'i'
t_SUPPOSITION = r'y?oo'

# error handling
def t_error(t):
    print("illegal character '%s'" % t.value[0])

# construct lexer
lexer = lex.lex()

# debug
if __name__ == '__main__':
    affix = 'sasenai'
    print(affix)
    lexer.input(affix)

    while True:
        tok = lexer.token()
        if not tok:
            # no tokens
            break
        print(tok)
