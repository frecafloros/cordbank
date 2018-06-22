# coding: utf-8
import ply.yacc as yacc
from hasei_lex import tokens

def p_expression_verb(p):
    '''expression : CAUSATIVE expression
                  | PASSIVE expression
                  | PERMISSIVE expression
                  | TERMINAL expression
                  | ASSUMPTION expression
                  | PERFECT expression
                  | FORMAL expression
                  | NEGATION expression
                  | DESCRIPTIVE expression
                  | SUPPOSITION expression
    '''
    p[0] = p[1] + ' ' +p[2]

def p_expression_verb_terminal(p):
    '''expression : MEIREI
                  | FORMALNEG
                  | empty
    '''
    p[0] = p[1]

def p_empty(p):
    'empty :'
    p[0] = ' '

# syntax error
def p_error(p):
    print("syntax error in input")

# construct parser
parser = yacc.yacc()

if __name__ == '__main__':
    while True:
        try:
            s = input('verb > ')
        except EOFError:
            break
        if not s:
            continue
        result = parser.parse(s)
        print(result.split())
