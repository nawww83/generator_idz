# Модуль для проверки ответов к Заданию № 01
from openpyxl import load_workbook
from openpyxl.styles import Font
import linear_codes as lc
import sys
import operator
from functools import reduce
from pprint import pprint as pp

# Проверяет состоит ли вектор только из 0 и 1
def is_bits_vector(v):
    return all(map(lambda x: (x == 0) or (x == 1), v))

# Проверяет состоит ли вектор только из чисел типа int
def is_int_vector(v):
    return all(map(lambda x: isinstance(x, int), v))

mjr = sys.version_info.major
mnr = sys.version_info.minor
if (mjr == 3 and mnr < 7) or mjr < 3:
    print('Требуется Python версии 3.7 и выше!')
    exit()

student = 'IvanovAA'
task_code = '01'
group = '1B6'

# Ограничения на параметры (n, k) кода
min_n = 6
max_n = 15
min_k = 3
max_k = 5
min_r = 2

assert(min_k < min_n)
assert(max_k < max_n)

fname = f'{student}_{task_code}_{group}.xlsx'

wb = load_workbook(fname)

ws = wb['Main']
G = []
parameters = []
for row in ws.iter_rows(min_row = 1, max_col = max_n, max_row = 16 + max_k, values_only = True):
    row = list(filter(None.__ne__, row)) # Убирает ненужные None
    g_Ok = is_bits_vector(row)
    n = len(row)
    if g_Ok and n >= min_n and n <= max_n:
        G.append(row) # Читаем порождающую матрицу G кода
    if n == 1 and isinstance(row[0], int):
        parameters.append(row[0])

print('Порождающая матрица кода')
pp(G)

k, n, ok = lc.check_matrix(G)
r = n - k

assert((len(parameters) == 6) and ok)

n_, k_, r_, d_, qo_, qi_ = parameters

CS = lc.gen_code(G)

C, Wsp, d = CS
qo = d - 1
qi = (d - 1) // 2

print('Правильные ответы:')
pp(f'n = {n}')
pp(f'k = {k}')
pp(f'r = {r}')
pp(f'd = {d}')
pp(f'qo = {qo}')
pp(f'qi = {qi}')

print('Введенные ответы:')
pp(f'n = {n_}')
pp(f'k = {k_}')
pp(f'r = {r_}')
pp(f'd = {d_}')
pp(f'qo = {qo_}')
pp(f'qi = {qi_}')

assert(n_ == n)
assert(k_ == k)
assert(r_ == r)
assert(d_ == d)
assert(qo_ == qo)
assert(qi_ == qi)

wsC = wb['Code']
C_ = []
for row in wsC.iter_rows(min_row = 1, max_col = max_n, max_row = 1 + 2**max_k, values_only = True):
    row = list(filter(None.__ne__, row))
    g_Ok = is_bits_vector(row)
    n = len(row)
    if g_Ok and n >= min_n and n <= max_n:
        C_.append(row) # Читаем множество кодовых векторов

print('Правильный код')
pp(C)

print('Введенный код')
pp(C_)

ok = True
for c in C:
    ok = ok and (c in C_)

pp(ok)
assert(len(C) == len(C_) and ok)

wsSp = wb['CodeSpectrum']
tmp_ = []
for row in wsSp.iter_rows(min_row = 1, max_col = max_n + 1, max_row = 5, values_only = True):
    row = list(filter(None.__ne__, row))
    g_Ok = is_int_vector(row)
    n = len(row)
    if g_Ok and n <= max_n + 1:
        tmp_.append(row)

assert(len(tmp_) == 2)

spC_ = dict((k, v) for k, v in zip(tmp_[0], tmp_[1]) if v > 0)

print('Правильный спектр кода')
pp(Wsp)

print('Введенный спектр кода')
pp(spC_)

assert(spC_ == Wsp)

pp('All Ok')
