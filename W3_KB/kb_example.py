from logic import *

wumpus_kb = PropKB()

P11, P12, P21, P22, P31, B11,B21 = expr('P11, P12, P21, P22, P31, B11, B21')
wumpus_kb.tell(P11)
wumpus_kb.tell(~B11)
wumpus_kb.tell(B21)
wumpus_kb.tell(B11 | '<=>' | (P12 | P21))
wumpus_kb.tell(B21 | '<=>' | (P11 | P22 | P31))


print(wumpus_kb.clauses)
print(' P21: ', wumpus_kb.ask_if_true(P21))
print(' P22: ', wumpus_kb.ask_if_true(P22))
print('~P21: ', wumpus_kb.ask_if_true(~P21))
print('~P22: ', wumpus_kb.ask_if_true(~P22))
print(' B21: ', wumpus_kb.ask_if_true(B21))
print(' P11: ', wumpus_kb.ask_if_true(P11))
