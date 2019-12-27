# Модуль для проверки ответов к Заданию № 05
from openpyxl import load_workbook
from openpyxl.styles import Font
import sys
import error_rate as er
import linear_codes as lc
from pprint import pprint as pp
import numpy as np

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

def error_func(i):
    switcher = {
        0: er.err_ask,
        1: er.err_ask,
        2: er.err_fsk,
        3: er.err_fsk,
        4: er.err_bpsk,
        5: er.err_bpsk,
        6: er.err_qpsk,
        7: er.err_qpsk,
        8: er.err_psk_8
    }
    return switcher.get(i, '')

coherencies = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0}

# Число бит на символ
bits_per_symbol = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 3}

# Ограничения на параметры (n, k) кода
min_n = 31
max_n = 1023
# Ограничение на вероятность битовой ошибки на выходе декодера
p_min = 1.e-08
p_max = 1.e-02

hf = Font(name = 'Calibri', bold = True)

fname = f'{student}_{task_code}_{group}.xlsx'

wb = load_workbook(fname)

ws = wb['Main']
params = []
for row in ws.iter_rows(min_row = 1, max_col = 1, max_row = 5, values_only = True):
    row = list(filter(None.__ne__, row)) # Убирает ненужные None
    n_row = len(row)
    if n_row == 1 and (isinstance(row[0], int) or isinstance(row[0], float)):
        params.append(row[0])

m_, p_ = params
assert(m_ >= m_min)
assert(m_ <= m_max)
assert(p_ <= p_max)
assert(p_ >= p_min)

print(f'Требуемая вероятность битовой ошибки P бит. дек.: {p_}')
print(f'Вид модуляции m: {m_}: {modulations[m_]}')

wsC = wb['Check']
params = []
for row in wsC.iter_rows(min_row = 1, max_col = 1, max_row = 6, values_only = True):
    row = list(filter(None.__ne__, row)) # Убирает ненужные None
    n_row = len(row)
    if n_row == 1 and isinstance(row[0], float) or isinstance(row[0], int):
        params.append(row[0])

R_, EbN0_dB_ = params

assert(R_ > 0.)
assert(R_ <= 1.)

#print(f'Скорость кодирования R: {R_}')
#print(f'Отношение сигнал-шум на один бит Eb/N0: {EbN0_dB_}, дБ')

EbN0_ = er.dB2pow(EbN0_dB_)
M_ = bits_per_symbol[m_]
EsN0_ = M_ * EbN0_ * R_

p_sym_err, p_bit_err = error_func(m_)(EsN0_, coherencies[m_])

# Решение задачи оптимизации
p_bit_dec = p_ # Целевая вероятность
n = 127
m = m_
M = bits_per_symbol[m]
R = 0.55
k = int(round(R * n))
qi = lc.resolve_hamming_constrain(n, k)
d = 2 * qi + 1
rel_error = 1.
EbN0 = 0.1
dE = 0.0006
EsN0 = er.EsN0(EbN0, R, M)
while rel_error > 0.01:
    _, p_bit_err = error_func(m)(EsN0, coherencies[m])
    _, p_bit_err_2 = error_func(m)(EsN0 + dE, coherencies[m])
    diff = n - qi
    p_err = 0.
    p_err_2 = 0.
    if diff > qi:
        for q in range(qi + 1):
            p_err += lc.probability_bsc(q, n, p_bit_err)
            p_err_2 += lc.probability_bsc(q, n, p_bit_err_2)
        p_err = 1. - p_err
        p_err_2 = 1. - p_err_2
    else:
        for q in range(qi + 1, n + 1):
            p_err += lc.probability_bsc(q, n, p_bit_err)
            p_err_2 += lc.probability_bsc(q, n, p_bit_err_2)

    p_bit = p_err * d / n
    p_bit_2 = p_err_2 * d / n
    rel_error = np.fabs(p_bit - p_bit_dec) / p_bit_dec
    if p_bit_2 > p_bit:
        EsN0 -= np.sign(p_bit - p_bit_dec)* dE
    else:
        EsN0 += np.sign(p_bit - p_bit_dec)* dE
    # print(f'... EbN0: {er.pow2dB(er.EbN0(EsN0, R, M))} дБ, P бит. дек. {p_bit}')
print(f'Скорость кодирования R: {R}')
print(f'EbN0: {er.pow2dB(er.EbN0(EsN0, R, M)):.2f} дБ')
#
print(f'P бит. дек.:{p_bit}')
print(f'Длина кода n: {n}')

print('All Ok')

