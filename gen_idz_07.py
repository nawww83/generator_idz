# Модуль генерирует файл с Заданием № 07 на тему "Коды Хаффмана и Шеннона-Фано"
from openpyxl import Workbook
from openpyxl.styles import Font
import sys
from pprint import pprint as pp
import pytils.translit
import re
import numpy as np
import linear_codes as lc
import source_codes as sc

def has_numbers(s):
    return re.search('\d', s)

def generator(group, student, task_code):
    # Вероятности троичного символа, p1, p2, p3
    P = [0.] * m_alphabet
    while not sc.all_in_range_incl(P, min_p, max_p):
        P = sc.get_probabilities_vector(m_alphabet)
        

    al = dict(zip(range(m_alphabet), P))
    print(f'Алфавит: {al}, норма: {sum(P)}')

    wb = Workbook()
    ws = wb.active
    ws.title = 'Main'

    hf = Font(name = 'Calibri', bold = True)

    ws.append(['Алфавит, X:'])
    ws.append(list(al.keys()))
    ws.append(list(al.values()))

    wsC = wb.create_sheet('Check')
    wsC.append(['Введите ответы:'])
    wsC.cell(row = wsC.max_row, column = 1).font = hf
    wsC.append(['Код Хаффмана для символа X:'])
    wsC.append(['Символ', 'Код'])
    for k, v in al.items():
        len_code_limit = int(np.log2(1. / v) + 0.5) + 1
        code = lc.get_rand_bits(len_code_limit)
        wsC.append([k] + code)
    wsC.append(['Код Хаффмана для символа XX:'])
    wsC.append(['Символ', 'Код'])
    for k1, v1 in al.items():
        for k2, v2 in al.items():
            len_code_limit = int(np.log2(1. / v1 / v2) + 0.5) + 1
            code = lc.get_rand_bits(len_code_limit)
            wsC.append([str(k1) + str(k2)] + code)

    wb.save(f'{student}_{task_code}_{group}.xlsx')

if __name__ == "__main__":
    mjr = sys.version_info.major
    mnr = sys.version_info.minor
    if (mjr == 3 and mnr < 7) or mjr < 3:
        print('Требуется Python версии 3.7 и выше!')
        exit()

    # Объем алфавита X
    m_alphabet = 3
    # Наименьшая вероятность
    min_p = 0.05
    # Наибольшая вероятность
    max_p = 0.95
    
    fn = 'list_magister_titpi_2020.txt'
    task_code = '07'

    students_file = open(fn, 'r', encoding = 'utf-8')
    students = students_file.readlines()
    group = ''
    student = ''
    for s in students:
        s = s.strip()
        if s:
            s_translit = pytils.translit.translify(s)
            print(s_translit)
            if has_numbers(s_translit):
                group = s_translit
            else:
                student = s_translit
                generator(group, student, task_code)
