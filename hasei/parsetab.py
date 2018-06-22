
# parsetab.py
# This file is automatically generated. Do not edit.
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'CAUSATIVE PASSIVE PERMISSIVE TERMINAL ASSUMPTION MEIREI PERFECT FORMAL NEGATION FORMALNEG DESCRIPTIVE SUPPOSITIONexpression : CAUSATIVE expression\n                  | PASSIVE expression\n                  | PERMISSIVE expression\n                  | TERMINAL expression\n                  | ASSUMPTION expression\n                  | PERFECT expression\n                  | FORMAL expression\n                  | NEGATION expression\n                  | DESCRIPTIVE expression\n                  | SUPPOSITION expression\n    expression : MEIREI\n                  | FORMALNEG\n                  | empty\n    empty :'
    
_lr_action_items = {'FORMALNEG':([0,3,4,6,8,9,10,11,12,13,14,],[2,2,2,2,2,2,2,2,2,2,2,]),'NEGATION':([0,3,4,6,8,9,10,11,12,13,14,],[3,3,3,3,3,3,3,3,3,3,3,]),'DESCRIPTIVE':([0,3,4,6,8,9,10,11,12,13,14,],[4,4,4,4,4,4,4,4,4,4,4,]),'$end':([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,],[-14,-13,-12,-14,-14,-11,-14,0,-14,-14,-14,-14,-14,-14,-14,-8,-9,-10,-3,-7,-6,-5,-1,-2,-4,]),'MEIREI':([0,3,4,6,8,9,10,11,12,13,14,],[5,5,5,5,5,5,5,5,5,5,5,]),'SUPPOSITION':([0,3,4,6,8,9,10,11,12,13,14,],[6,6,6,6,6,6,6,6,6,6,6,]),'FORMAL':([0,3,4,6,8,9,10,11,12,13,14,],[9,9,9,9,9,9,9,9,9,9,9,]),'PERMISSIVE':([0,3,4,6,8,9,10,11,12,13,14,],[8,8,8,8,8,8,8,8,8,8,8,]),'PERFECT':([0,3,4,6,8,9,10,11,12,13,14,],[10,10,10,10,10,10,10,10,10,10,10,]),'ASSUMPTION':([0,3,4,6,8,9,10,11,12,13,14,],[11,11,11,11,11,11,11,11,11,11,11,]),'CAUSATIVE':([0,3,4,6,8,9,10,11,12,13,14,],[12,12,12,12,12,12,12,12,12,12,12,]),'PASSIVE':([0,3,4,6,8,9,10,11,12,13,14,],[13,13,13,13,13,13,13,13,13,13,13,]),'TERMINAL':([0,3,4,6,8,9,10,11,12,13,14,],[14,14,14,14,14,14,14,14,14,14,14,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'empty':([0,3,4,6,8,9,10,11,12,13,14,],[1,1,1,1,1,1,1,1,1,1,1,]),'expression':([0,3,4,6,8,9,10,11,12,13,14,],[7,15,16,17,18,19,20,21,22,23,24,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> expression","S'",1,None,None,None),
  ('expression -> CAUSATIVE expression','expression',2,'p_expression_verb','hasei_yacc.py',6),
  ('expression -> PASSIVE expression','expression',2,'p_expression_verb','hasei_yacc.py',7),
  ('expression -> PERMISSIVE expression','expression',2,'p_expression_verb','hasei_yacc.py',8),
  ('expression -> TERMINAL expression','expression',2,'p_expression_verb','hasei_yacc.py',9),
  ('expression -> ASSUMPTION expression','expression',2,'p_expression_verb','hasei_yacc.py',10),
  ('expression -> PERFECT expression','expression',2,'p_expression_verb','hasei_yacc.py',11),
  ('expression -> FORMAL expression','expression',2,'p_expression_verb','hasei_yacc.py',12),
  ('expression -> NEGATION expression','expression',2,'p_expression_verb','hasei_yacc.py',13),
  ('expression -> DESCRIPTIVE expression','expression',2,'p_expression_verb','hasei_yacc.py',14),
  ('expression -> SUPPOSITION expression','expression',2,'p_expression_verb','hasei_yacc.py',15),
  ('expression -> MEIREI','expression',1,'p_expression_verb_terminal','hasei_yacc.py',20),
  ('expression -> FORMALNEG','expression',1,'p_expression_verb_terminal','hasei_yacc.py',21),
  ('expression -> empty','expression',1,'p_expression_verb_terminal','hasei_yacc.py',22),
  ('empty -> <empty>','empty',0,'p_empty','hasei_yacc.py',27),
]
