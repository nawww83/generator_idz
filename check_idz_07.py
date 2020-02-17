# Модуль для проверки ответов к Заданию № 07
from openpyxl import load_workbook
from openpyxl.styles import Font
import sys
import error_rate as er
import source_codes as sc
import linear_codes as lc
from pprint import pprint as pp
import numpy as np
from random import random

mjr = sys.version_info.major
mnr = sys.version_info.minor
if (mjr == 3 and mnr < 7) or mjr < 3:
    print('Требуется Python версии 3.7 и выше!')
    exit()

student = 'IvanovAA'
task_code = '07'
group = '1B6'

# Наименьшая вероятность
p_min = 0.05
# Наибольшая вероятность
p_max = 0.95

hf = Font(name = 'Calibri', bold = True)

fname = f'{student}_{task_code}_{group}.xlsx'

wb = load_workbook(fname)

ws = wb['Main']
params = []
for row in ws.iter_rows(min_row = 1, max_col = 4, max_row = 4, values_only = True):
    row = list(filter(None.__ne__, row)) # Убирает ненужные None
    n_row = len(row)
    if n_row > 0:
        params.append(row)

_, sym_, P_ = params
p1, p2, p3 = P_

assert(p1 >= p_min)
assert(p1 <= p_max)
assert(p2 >= p_min)
assert(p2 <= p_max)
assert(p3 >= p_min)
assert(p3 <= p_max)

al_ = dict(zip(sym_, P_))
print(f'Алфавит, {{X, P}}: {al_}')
print(f'Норма: {sum(al_.values())}')


wsC = wb['Check']
params = []
len_code_limit = 2 * int(np.log2(1. / min(P_)) + 0.5) + 2 # Учет двойного символа XX
for row in wsC.iter_rows(min_row = 1, max_col = len_code_limit + 2, max_row = 18, values_only = True):
    row = list(filter(None.__ne__, row)) # Убирает ненужные None
    n_row = len(row)
    if n_row > 0:
        params.append(row)

_, _, _, code_0_, code_1_, code_2_, *_ = params

sym = code_0_.pop(0), code_1_.pop(0), code_2_.pop(0)
code_ = dict(zip(sym, [code_0_, code_1_, code_2_]))

print('Введенный код Хаффмана для символа X:')
print(code_)

code = sc.make_huffman_table(al_)

print('Найденный код Хаффмана (один из множества вариантов):')
print(code)

# Проверка правильности введенного ответа
# По равенству длин кодовых слов
l_ = {}
for k, v in code_.items():
    l_[k] = len(v)
l = {}
for k, v in code.items():
    l[k] = len(v)

assert(l_ == l)

# По свойству префикса
cv = code_.values()
is_prefix = False
for k, v in code_.items():
    for c in cv:
        if len(v) < len(c):
            is_prefix = (v == c[: len(v)])
            if is_prefix:
                break
    if is_prefix:
        break

assert(not is_prefix)

params = params[-9:]

sym = [i.pop(0) for i in params]
code_ = params # pop(0)
code_ = dict(zip(sym, code_))

print('Введенный код Хаффмана для символа XX:')
print(code_)

alal = {}
sym_match = {}
for k1, v1 in al_.items():
    for k2, v2 in al_.items():
        sym_match[k1 + k2 * 3] = str(k2) + str(k1)
        alal[k1 + k2 * 3] = v1 * v2

code_xx = sc.make_huffman_table(alal)
code_xx = dict(sorted(code_xx.items()))

code = {}
for k, v in code_xx.items():
    code[sym_match[k]] = v

print('Найденный код Хаффмана (один из множества вариантов):')
print(code)

# Проверка правильности введенного ответа
# По равенству длин кодовых слов
l_ = {}
for k, v in code_.items():
    l_[k] = len(v)
l = {}
for k, v in code.items():
    l[k] = len(v)

assert(l_ == l)

# По свойству префикса
cv = code_.values()
is_prefix = False
for k, v in code_.items():
    for c in cv:
        if len(v) < len(c):
            is_prefix = (v == c[: len(v)])
            if is_prefix:
                break
    if is_prefix:
        break

assert(not is_prefix)

print('All Ok')
