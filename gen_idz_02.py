# Модуль генерирует файл с Заданием № 02
from openpyxl import Workbook
from openpyxl.styles import Font
import linear_codes as lc
import sys
from pprint import pprint as pp

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

hf = Font(name = 'Calibri', bold = True)

# Генерация случайных (n, k) так, что k < n
while True:
    n = lc.randint(min_n, max_n)
    k = lc.randint(min_k, max_k)
    if k < n:
        break

r = n - k

print(f'(n,k) = ({n},{k})')
G = lc.gen_matrix(n, k)
Gsh = lc.shuffle_matrix(G, n, True)

print('Порождающая матрица G в систематической форме')
pp(G)

print('Матрица G после тасовки')
pp(Gsh)

wb = Workbook()
ws = wb.active
ws.title = 'Main'

ws['A1'].font = hf
ws['A1'] = 'Порождающая матрица G'
for g_r in Gsh:
    ws.append(g_r)

ws.append(['Введите ответы:'])
ws.append(['Проверочная матрица H:'])
for _ in range(r):
    ws.append(lc.get_rand_bits(n))
ws.append(['Кодовое расстояние кода dк:'])
ws.append([0])

wb.save(f'{student}_{task_code}_{group}.xlsx')

