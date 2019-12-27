# Модуль генерирует файл с Заданием № 05 на тему "Вероятность ошибки при 
# оптимальном приеме цифрового сигнала"
from openpyxl import Workbook
from openpyxl.styles import Font
import sys
from random import choice
from random import random
from pprint import pprint as pp

mjr = sys.version_info.major
mnr = sys.version_info.minor
if (mjr == 3 and mnr < 7) or mjr < 3:
    print('Требуется Python версии 3.7 и выше!')
    exit()

student = 'IvanovAA'
task_code = '05'
group = '1B6'

# Ограничения на вид модуляции
m_min = 0
m_max = 8

modulations = {0: 'АМн когер.', 1: 'АМн некогер.', 2: 'ЧМн когер.', \
3: 'ЧМн некогер.', 4: 'ФМн когер.', 5: 'ФМн част. когер.', \
6: 'ФМн-4/КАМ-4 когер.', 7: 'ФМн-4/КАМ-4 част. когер', 8: 'ФМн-8 когер.'}

# Ограничения на параметры (n, k) кода
min_n = 31
max_n = 1023
# Ограничение на вероятность битовой ошибки на выходе декодера
p_min = 1.e-08
p_max = 1.e-02

hf = Font(name = 'Calibri', bold = True)

# Вид модуляции
m = round( random() * (m_max - m_min) + m_min )

print(f'Вид модуляции: {m}')

# Генерация вероятности ошибки в канале
p = random() * (p_max - p_min) + p_min

print(f'Вероятность ошибки в канале: {p}')

wb = Workbook()
ws = wb.active
ws.title = 'Main'

ws.append(['Вид модуляции m:'])
ws.append([m])
ws.append(['Требуемая вероятность битовой ошибки P бит. дек.:'])
ws.append([p])

wsC = wb.create_sheet('Check')
wsC.append(['Введите ответы:'])
wsC.cell(row = wsC.max_row, column = 1).font = hf
wsC.append(['Скорость кодирования R:'])
wsC.append([0.])
wsC.append(['Требуемое отношение сигнал-шум на один бит, Eb/N0, дБ'])
wsC.append([0.])
wsC.append(['Внимание! Ответы давать с точностью не хуже 1%'])
wsC.cell(row = wsC.max_row, column = 1).font = hf

wb.save(f'{student}_{task_code}_{group}.xlsx')

