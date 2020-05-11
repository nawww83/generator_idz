# Модуль "Кодирование источника" (экономное кодирование)
# Тест
# Алфавит X:
#{0: 0.0069,
# 1: 0.0489,
# 2: 0.0678,
# 3: 0.0569,
# 4: 0.0904,
# 5: 0.0510,
# 6: 0.1369,
# 7: 0.0668,
# 8: 0.1185,
# 9: 0.0381,
# 10: 0.1533,
# 11: 0.1645}
# Энтропия
# H = 3.3342767... бит/символ
# Код Хаффмана
# {11: [1, 1, 1],
# 10: [1, 1, 0],
# 6: [1, 0, 1],
# 8: [0, 1, 1],
# 4: [0, 0, 0],
# 2: [1, 0, 0, 1],
# 7: [1, 0, 0, 0],
# 3: [0, 1, 0, 1],
# 5: [0, 1, 0, 0],
# 1: [0, 0, 1, 1],
# 9: [0, 0, 1, 0, 1],
# 0: [0, 0, 1, 0, 0]}
# Средняя длина кода
# L = 3.3814 бит > H
# Код Шеннона-Фано
# {11: [1, 1],
# 10: [1, 0, 1],
# 6: [1, 0, 0],
# 8: [0, 1, 1],
# 4: [0, 1, 0, 1],
# 2: [0, 1, 0, 0],
# 7: [0, 0, 1, 1],
# 3: [0, 0, 1, 0],
# 5: [0, 0, 0, 1],
# 1: [0, 0, 0, 0, 1],
# 9: [0, 0, 0, 0, 0, 1],
# 0: [0, 0, 0, 0, 0, 0]}
# Средняя длина
# L = 3.4012 бит > H

from functools import reduce
from random import random
import numpy as np

# Проверяет принадлежит ли отрезку [a, b] скаляр x
def in_range_incl(x, a, b):
    return (x >= a) and (x <= b)

# Проверяет все ли элементы вектора X принадлежат отрезку [a, b]
def all_in_range_incl(X, a, b):
    return reduce(lambda x, y: x and y, \
        map(in_range_incl, X, [a]*len(X), [b]*len(X)))

# Возвращает вектор n вероятностей, так, что их сумма равна единице
def get_probabilities_vector(n):
    P = [random() for _ in range(n)]
    norma = sum(P)
    n_of_digits = int(np.round(np.log10(norma / min(P)) + 0.5) + 1)
    P = [round(p / norma, n_of_digits) for p in P]
    epsilon = np.power(10., -n_of_digits)
    norma = sum(P)
    i = P.index(max(P))
    j = P.index(min(P))
    while norma != 1.:
        if norma > 1.:
            P[i] -= epsilon
        if norma < 1.:
            P[j] += epsilon
        norma = round(sum(P), n_of_digits + 1)

    return [round(p, n_of_digits) for p in P]

# Возвращает алфавит объемом m символов
def get_alphabet(m):
    return dict(zip(range(m), get_probabilities_vector(m)))

# Возвращает шенноновскую энтропию алфавита (источника) al, бит/символ
def entropy(al):
    return sum( [-v * np.log2(v) for k, v in al.items()] )

# Возвращает среднюю длину кода code для алфавита al
def ave_length(al, code):
    return sum( [ al[k] * len(v) for k, v in code.items()] )

# Проверяет свойство префикса кода code = {symbol: code_word}
def check_prefix(code):
    ok = True
    for k, v in code.items():
        cv = list(code.values())
        iv = cv.index(v)
        cv.pop(iv) # Чтобы не было пересечения вектора с самим собой
        for c in cv:
            if len(v) <= len(c):
                ok = (v != c[: len(v)])
                if not ok:
                    break
        if not ok:
            break

    return ok

# Генерирует таблицу кода Хаффмана по заданному алфавиту 
# al = {symbol: probability}, где symbol - символы 0, 1, ..., m - 1
# probability - вероятности символов. Численные "огрехи" вероятностей здесь
# не корректируются (расчет как есть)
def make_huffman_table(al):
    n = len(al)
    al_sorted = {}
    code_tree = {}
    trace = {}
    i = 0
    # Сортировка алфавита в порядке убывания вероятностей
    for k in sorted(al, key = al.get, reverse = True):
        al_sorted[i] = al[k]
        code_tree[k] = []
        trace[i] = [k]
        i += 1

    while len(al_sorted) > 1:
        last = len(al_sorted) - 1
        # Самый нижний символ объединяется с рядом стоящим 
        #(вероятности складываются) и удаляется
        probability = al_sorted.pop(last)
        last -= 1
        al_sorted[last] += probability
        # trace указывает в какие векторы добавить по 1 и 0 (дерево)
        for t in trace[last]: # Верх, 1
            code_tree[t].append(1)
        for t in trace[last + 1]: # Низ, 0
            code_tree[t].append(0)
        # Т.к. нижний символ удален, то его (значение) следует запомнить в
        # истории trace текущего символа
        trace[last] += trace.pop(last + 1)
        if last > 1:
            ii = last - 1
            # Нижний символ всплывает вверх до своего уровня (по вероятности)
            while al_sorted[ii] < al_sorted[ii + 1]:
                al_sorted[ii], al_sorted[ii + 1] = \
                    al_sorted[ii + 1], al_sorted[ii]
                trace[ii], trace[ii + 1] = trace[ii + 1], trace[ii]
                ii -= 1
                if ii < 0:
                    break
        elif last == 1: # Осталось два элемента
            if al_sorted[0] < al_sorted[1]:
                al_sorted[0], al_sorted[1] = al_sorted[1], al_sorted[0]
                trace[0], trace[1] = trace[1], trace[0]

    for i in range(n):
        code_tree[i].reverse() # Идем от корня дерева к листьям

    return code_tree

# Необходима коррекция индекса вниз? (есть ли перебор по накопленной сумме?)
def need_down_correct(val, weight, threshold):
    need = False
    delta = np.fabs(weight - threshold)
    if weight != threshold:
        delta_2 = np.fabs(weight - val - threshold)
        if delta_2 < delta:
            need = True

    return need

# По заданному алфавиту al = {symbol: probability} генерирует кодовую таблицу
# кода Шеннона-Фано
def make_shannon_fano_table(al):
    n = len(al)
    al_sorted = {}
    trace = []
    i = 0
    # Сортировка алфавита в порядке убывания вероятностей
    for k in sorted(al, key = al.get, reverse = True):
        al_sorted[i] = al[k]
        trace.append(k)
        i += 1
    # Поиск подгрупп с равными суммарными вероятностями
    boundaries = {} # Требуется Python >= 3.7 для гарантии 
    # сохранения порядка вставки (keeps insertion order)
    level = 0 # Первоначальное состояние level = 0, 
    # при построении кода игнорируется
    boundaries.setdefault(level, [])
    boundaries[level].append(n - 1)
    # Уровни level отображают этапы формирования кодовой таблицы
    # level = 1: первое деление на подгруппы и присвоение 1 и 0
    # level = 2: второе ... до тех пор пока возможно деление
    while len(boundaries[level]) < n:
        bnd = boundaries[level]
        start = 0
        level += 1
        boundaries.setdefault(level, [])
        # Если boundaries[level] содержит все числа от 0 до n - 1, то
        # дальнейшее деление на подгруппы невозможно: код сформирован
        boundaries[level] += boundaries[level - 1]
        for b in bnd:
            stop = b
            threshold = 0. # Пороговая вероятность
            if stop == start: # Подгруппа уже сформирована
                start = stop + 1 # Переход к следующей подгруппе
                continue
            for k in range(start, stop + 1):
                threshold += al_sorted[k]
            threshold *= 0.5
            weight = 0.
            i = start
            while weight < threshold:
                weight += al_sorted[i]
                i += 1
                if i == stop + 1 or i == n:
                    break
            i = i - 1
            # Проверяем не перебрали ли?
            need = need_down_correct(al_sorted[i], weight, threshold)
            if need:
                i -= 1
            # Нашли границу в подгруппе
            boundaries[level].append(i)
            start = stop + 1
        boundaries[level].sort()
    # Формирование кодовой таблицы по найденным границам boundaries
    code_tree = {}
    used = {n - 1} # Использованные средние точки (границы)
    for k, lv in boundaries.items():
        if k == 0: # Игнорируем level = 0
            continue 
        # Текущий набор средних точек (границ)
        middle_points = list(set(lv) - used)
        middle_points.sort()
        for mp in middle_points:
            start = 0
            stop = n - 1
            start = mp - 1
            while start not in lv:
                if start < 0:
                    break
                start -= 1
            start += 1
            stop = mp + 1
            while stop not in lv:
                if stop == n - 1:
                    break
                stop += 1
            i = start
            bit = int(i <= mp) # Верхней подгруппе бит 1
            # Верхняя подгруппа - до средней точки
            used.add(mp)
            while True:
                code_tree.setdefault(trace[i], [])
                code_tree[trace[i]].append(bit)
                i += 1
                bit = int(i <= mp)
                if i > stop:
                    break

    return code_tree

