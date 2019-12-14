# Модуль для проверки ответов к Заданию № 02
from openpyxl import load_workbook
from openpyxl.styles import Font
import linear_codes as lc
import sys
import operator
from functools import reduce
from pprint import pprint as pp
from copy import deepcopy

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
task_code = '02'
group = '1B6'

# Ограничения на параметры (n, k) кода
min_n = 6
max_n = 15
min_k = 3
max_k = 5

assert(min_k < min_n)
assert(max_k < max_n)

fname = f'{student}_{task_code}_{group}.xlsx'

wb = load_workbook(fname)

ws = wb['Main']
G = []
H_ = []
parameters = []
i = 0
G_fulled = False
H_fulled = False
for row in ws.iter_rows(min_row = 1, max_col = max_n, max_row = 6 + max_n, values_only = True):
    row = list(filter(None.__ne__, row)) # Убирает ненужные None
    g_Ok = is_bits_vector(row)
    n = len(row)
    if g_Ok and n >= min_n and n <= max_n:
        if not G_fulled:
            G.append(row) # Читаем матрицы кода
        else:
            H_.append(row)
        i += 1
    elif n == 1 and isinstance(row[0], int):
        parameters.append(row[0])
    elif not G_fulled and G:
        G_fulled = True
    elif G_fulled and not H_fulled and H_:
        H_fulled = True

p = lc.check_matrix(G)
k = p[0]
n = p[1]
assert(p[2])

r = n - k

pp(G)
pp(parameters)

assert((len(parameters) == 1))

d_ = parameters[0]

d = lc.gen_code(G)[2]

H = lc.get_check_matrix(G)

print('Правильные ответы:')
pp(f'd = {d}')

print('Введенные ответы:')
pp(f'd = {d_}')

assert(d_ == d)

print('Правильная проверочная матрица кода')
pp(H)

print('Введенная проверочная матрица кода')
pp(H_)

zero = sum(map(sum, lc.multM(G, lc.transpose(H))))
zero_ = sum(map(sum, lc.multM(G, lc.transpose(H_))))

print('Контроль нуля ')
pp(zero)
pp(zero_)

ok = (zero == zero_)

assert(ok)

pp('All Ok')

