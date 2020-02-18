# Модуль "Кодирование источника" (экономное кодирование)
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

# Проверяет свойство префикса кода code = {symbol: code_word}
def check_prefix(code):
    is_prefix = False
    for k, v in code.items():
        cv = list(code.values())
        iv = cv.index(v)
        cv.pop(iv)
        for c in cv:
            if len(v) <= len(c):
                is_prefix = (v == c[: len(v)])
                if is_prefix:
                    break
        if is_prefix:
            break

    return is_prefix

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
        trace[last] += trace[last + 1]
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
    i = 0
    # Сортировка алфавита в порядке убывания вероятностей
    for k in sorted(al, key = al.get, reverse = True):
        al_sorted[i] = al[k]
        i += 1
    
    # Поиск подгрупп с равными суммарными вероятностями
    boundaries = {}
    level = 0 # Первоначальное состояние, 
    # при построении кода level = 0 игнорируется
    boundaries.setdefault(level, [])
    threshold = 0.5
    boundaries[level].append(n - 1)

    while len(boundaries[level]) < n:
        #print(boundaries)
        bnd = boundaries[level]
        start = 0
        level += 1
        boundaries.setdefault(level, [])
        boundaries[level] += boundaries[level - 1]
        for b in bnd:
            stop = b
            threshold = 0.
            if stop == start: # Подгруппа сформирована
                start = stop + 1
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
            need = need_down_correct(al_sorted[i], weight, threshold)
            if need:
                i -= 1
            boundaries[level].append(i)
            start = stop + 1
        boundaries[level].sort()
    # TODO: доделать формирование кодовой таблице по boundaries
    return boundaries