# Модуль генерирует файл с Заданием № 01
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
task_code = '01'
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

print(f'(n,k) = ({n},{k})')
G = lc.gen_matrix(n, k)
Gsh = lc.shuffle_matrix(G, n, True)

print('Порождающая матрица G в систематической форме')
pp(G)

print('Матрица G после тасовки')
pp(Gsh)

print('Код C1 по систематической матрице G')
C = lc.gen_code(G)
pp(C[0])
print('Спектр кода Wsp(C1)')
pp(C[1])

print('Код C2 по тасованной матрице G')
Csh = lc.gen_code(Gsh)
pp(Csh[0])
print('Спектр кода Wsp(C2)')
pp(Csh[1])

wb = Workbook()
ws = wb.active
ws.title = 'Main'

ws['A1'].font = hf
ws['A1'] = 'Порождающая матрица G'
for g_r in Gsh:
    ws.append(g_r)

ws.append(['Введите ответы:'])
ws.append(['Длина кода n:'])
ws.append([0])
ws.append(['Число информационных символов k:'])
ws.append([0])
ws.append(['Число проверочных символов r:'])
ws.append([0])
ws.append(['Кодовое расстояние кода dк:'])
ws.append([0])
ws.append(['Кратность гарантированного обнаружения qо:'])
ws.append([0])
ws.append(['Кратность гарантированного исправления qи:'])
ws.append([0])

code_name = 'Code'
spectr_name = 'CodeSpectrum'

ws.append([f'Не забудьте заполнить листы {code_name} и {spectr_name}!'])
ws.cell(row = ws.max_row, column = 1).font = hf

wsC = wb.create_sheet(code_name)
wsC['A1'].font = hf
wsC['A1'] = 'Введите кодовые слова кода C в произвольном порядке'

for _ in range(2**k):
    wsC.append(lc.get_rand_bits(n))

wsWC = wb.create_sheet(spectr_name)
wsWC['A1'].font = hf
wsWC['A1'] = 'Введите спектр Wsp(C) кода (в отсортированном порядке)'
wsWC.append(['Веса w:'])
wsWC.append([i for i in range(n + 1)])
wsWC.append(['Количество весов Nw:'])
wsWC.append([0] * (n + 1))

wb.save(f'{student}_{task_code}_{group}.xlsx')
