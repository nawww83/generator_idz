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
min_r = 2

assert(min_k < min_n)
assert(max_k < max_n)

fname = f'{student}_{task_code}_{group}.xlsx'

wb = load_workbook(fname)

ws = wb['Main']
G = []
for row in ws.iter_rows(min_row = 1, max_col = max_n, max_row = 2 + max_k, values_only = True):
    row = list(filter(None.__ne__, row)) # Убирает ненужные None
    g_Ok = is_bits_vector(row)
    n = len(row)
    if g_Ok and n >= min_n and n <= max_n:
        G.append(row) # Читаем матрицу кода

k, n, ok = lc.check_matrix(G)
r = n - k
assert(ok)

print('Порождающая матрица кода')
pp(G)

parameters = []
H_ = []
wsC = wb['Check']
for row in wsC.iter_rows(min_row = 1, max_col = max_n, max_row = 5 + max_n - min_k, values_only = True):
    row = list(filter(None.__ne__, row)) # Убирает ненужные None
    h_Ok = is_bits_vector(row)
    n = len(row)
    if h_Ok and n >= min_n and n <= max_n:
        H_.append(row) # Читаем матрицу кода
    elif n == 1 and isinstance(row[0], int):
        parameters.append(row[0])

r, n, ok = lc.check_matrix(H_)
k = n - r
assert(ok)

assert((len(parameters) == 1))

d_ = parameters[0]
*_, d = lc.gen_code(G)
print('Подождите идет подбор матрицы H...')
H = lc.get_check_matrix(G)
d_alter = lc.get_code_distance(H)

print('Правильные ответы:')
pp(f'd = {d}, альтернативный метод d = {d_alter}')

assert(d == d_alter)

print('Введенные ответы:')
pp(f'd = {d_}')

assert(d_ == d)

print('Правильная проверочная матрица кода')
pp(H)

print('Введенная проверочная матрица кода')
pp(H_)

zero = sum(map(sum, lc.mult_M(G, lc.transpose(H))))
zero_ = sum(map(sum, lc.mult_M(G, lc.transpose(H_))))

print('Контроль правильности проверочной матрицы')
ok = (zero == zero_ == 0)
pp(ok)
assert(ok)

pp('All Ok')

