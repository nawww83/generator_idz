# Модуль для работы с Линейными Блочными (n,k) Кодами
from random import randint
import operator
from functools import reduce
from itertools import product
from copy import deepcopy

# Проверяет вложенный список на соответствие матрице размером (rows, cols)
# Возвращает кортеж в виде (размеры матрицы, флаг проверки)
def check_matrix(M):
    rows = len(M)
    assert(rows > 0)
    nn = len(reduce(operator.iconcat, M, []))
    cols = nn // rows
    ok = True
    for r in range(rows):
        ok = ok and (len(M[r]) == cols)
    return (rows, cols, ok)

# Возвращает пару различных случайных целых чисел из отрезка [0, n-1]
def get_random_pair(n):
    assert(n > 0)
    i1 = 0
    i2 = 0
    while i1 == i2:
        i1 = randint(1, n) - 1
        i2 = randint(1, n) - 1
    return (i1, i2)

# Объединяет две матрицы с одинаковым числом строк в одну
def augment(I, Q):
    k = len(I)
    k2 = len(Q)
    assert(k == k2)
    R = []
    for row in range(k):
        R.append(I[row] + Q[row])
    return R

# Возвращает единичную матрицу
def identity(k):
    z = [0] * k
    I = []
    for row in range(k):
        I.append(z[:])
        I[row][row] = 1
    return I

# Возвращает сумму (по модулю два) двух векторов
def xor(x, y):
    assert(len(x) == len(y))
    mod2 = [2] * len(x)
    z = list(map(operator.add, x, y))
    return list(map(operator.mod, z, mod2))

# Возвращает произведение вектора на матрицу
def mult(v, M):
    k = len(v)
    params = check_matrix(M)
    k_ = params[0]
    n = params[1]
    assert(k == k_)
    assert(params[2] == True)
    result = [0] * n
    for r in range(k):
        if v[r] == 1:
            result = xor(result, M[r])
    return result

# Возвращает случайный двоичный вектор
def get_rand_bits(n):
    return [randint(0, 1) for _ in range(n)]

# Возвращает случайную порождающую матрицу линейного (n, k) кода
# в систематической форме G = [I, Q]
def gen_matrix(n, k):
    assert(n > 1)
    assert(k < n)
    G = []
    I = identity(k)
    Q = []
    r = n - k
    for _ in range(k):
        q = get_rand_bits(r)
        Q.append(q)
    G = augment(I, Q)
    return G

# По порождающей матрице G возвращает множество кодовых векторов, а также
# спектр кода и кодовое расстояние
def gen_code(G):
    k = len(G)
    it = product([0,1], repeat = k)
    code = []
    ws = {}
    for a in it:
        s = mult(a, G)
        w = sum(s)
        if w in ws:
            ws[w] += 1
        else:
            ws[w] = 1
        code.append(s)
    ws = dict(sorted(ws.items()))
    ws_ = deepcopy(ws)
    del ws_[0]
    d = list(ws_.keys())
    v = list(ws_.values())
    dmin = min(d)
    return (code, ws, dmin)

# Возвращает перемешанную матрицу
# В перемешивание входит суммирование случайных пар строк и запись результата
# в одну из этих строк - эта процедура выполняется nsh раз. После опционально,
# если with_columns = True, делается перестановка столбцов nsh раз
def shuffle_matrix(M, nsh, with_columns):
    k = len(M)
    result = deepcopy(M)
    for i in range(nsh):
        (i1, i2) = get_random_pair(k)
        g1 = result[i1]
        g2 = result[i2]
        b = randint(0, 1)
        result[i1 * b + i2 * (1 - b)] = xor(g1, g2)
    if with_columns:
        params = check_matrix(result)
        n = params[1]
        assert(params[2])
        for i in range(nsh):
            (i1, i2) = get_random_pair(n)
            for j in range(k):
                result[j][i1], result[j][i2] = result[j][i2], result[j][i1]
    return result