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
    perm = {}
    code_tree = {}
    trace = {}
    i = 0
    for k in sorted(al, key = al.get, reverse = True):
        al_sorted[i] = al[k]
        code_tree[k] = []
        trace[i] = [k]
        i += 1

    while len(al_sorted) > 1:
        #print(al_sorted)
        last = len(al_sorted) - 1
        probability = al_sorted.pop(last)
        last -= 1
        al_sorted[last] += probability
        for t in trace[last]:
            code_tree[t].append(1)
        for t in trace[last + 1]:
            code_tree[t].append(0)
        trace[last] += trace[last + 1]
        if last > 1:
            ii = last - 1
            while al_sorted[ii] < al_sorted[ii + 1]:
                al_sorted[ii], al_sorted[ii + 1] = \
                    al_sorted[ii + 1], al_sorted[ii]
                trace[ii], trace[ii + 1] = trace[ii + 1], trace[ii]
                ii -= 1
                if ii < 0:
                    break
        elif last == 1:
            if al_sorted[0] < al_sorted[1]:
                al_sorted[0], al_sorted[1] = al_sorted[1], al_sorted[0]
                trace[0], trace[1] = trace[1], trace[0]

    for i in range(n):
        code_tree[i].reverse()

    return code_tree

