# Модуль для работы с двоичными Линейными Блочными (n,k) Кодами
# Для хранения матриц и векторов использует список list() из чисел 0 и 1
from random import randint
import operator
from functools import reduce
from itertools import product
from copy import deepcopy

# Возвращает кортеж в виде (размеры матрицы, флаг проверки)
# Проверяет вложенный список на соответствие матрице размером (rows, cols)
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

# Возвращает сумму по модулю два двух векторов
def xor(x, y):
    assert(len(x) == len(y))
    mod2 = [2] * len(x)
    z = list(map(operator.add, x, y))
    return list(map(operator.mod, z, mod2))

# Возврашает сумму по модулю два элементов вектора v
def xor1(v):
    return reduce(lambda x, y: (x + y) % 2, v)

# Возвращает произведение вектора на матрицу
def mult_v(v, M):
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

# Возвращает произведение двух матриц
def mult_M(A, B):
    paramsA = check_matrix(A)
    paramsB = check_matrix(B)
    kA = paramsA[0]
    nA = paramsA[1]
    kB = paramsB[0]
    nB = paramsB[1]
    assert(nA == kB)
    assert(paramsA[2] == True)
    assert(paramsB[2] == True)
    result = [[0] * nB for _ in range(kA)]
    Bt = transpose(B)
    for r in range(kA):
        for c in range(nB):
            result[r][c] = xor1( map(operator.mul, A[r], Bt[c]) )
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
        s = mult_v(a, G)
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
# 1. Суммируется случайная пара строк, результат записывается в одну из 
# этих строк. Процедура выполняется n_sh раз. Затем, если with_columns = True, 
# делается перестановка столбцов n_sh раз
def shuffle_matrix(M, n_sh, with_columns):
    rows = len(M)
    result = deepcopy(M)
    for i in range(n_sh):
        (i1, i2) = get_random_pair(rows)
        g1 = result[i1]
        g2 = result[i2]
        b = randint(0, 1)
        result[i1 * b + i2 * (1 - b)] = xor(g1, g2)
    if with_columns:
        params = check_matrix(result)
        cols = params[1]
        assert(params[2])
        for i in range(n_sh):
            (i1, i2) = get_random_pair(cols)
            for j in range(rows):
                result[j][i1], result[j][i2] = result[j][i2], result[j][i1]
    return result

# Возвращает матрицу, полученную путем перестановки столбцов матрицы M по 
# правилу p, в котором указаны индексы куда переставлять столбцы
def permute_columns(M, p):
    params = check_matrix(M)
    rows = params[0]
    cols = params[1]
    assert(params[2])
    assert(len(p) == cols)
    resultT = transpose(M)
    resultT_tmp = deepcopy(resultT)
    for i in range(cols):
        i_to = p[i]
        if i != i_to:
            resultT_tmp[i_to] = resultT[i]
    return transpose(resultT_tmp)

# Возвращает индексы (в порядке возрастания) всех единичных столбцов матрицы M
# Единичный столбец - это столбец с единственным ненулевым элементом равным 
# единице. Определяется по сумме элементов (для двоичных кодов это справедливо).
def find_unity_columns(M):
    MT = list(map(sum, zip(*M)))
    iloc = [i for i, e in enumerate(MT) if e == 1]
    return iloc

# Возвращает индексы m уникальных столбцов если их возможно найти 
# (в порядке iloc), в противном случае возвращает пустой список.
# Требует предварительного определения индексов iloc всех единичных столбцов
# матрицы M.
def tune_uniq_unity_columns(iloc, m, M):
    MT = transpose(M)
    d = set()
    iloc_ = {}
    for i in iloc:
        index = MT[i].index(1)
        if index not in d:
            iloc_[index] = i
        d.add(index)
    if len(d) >= m:
        return list(iloc_.values())
    else:
        return []

# Возвращает транспонированную матрицу, т.е. Y = X^T
def transpose(X):
    return list(map(list, zip(*X)))

# Возвращает проверочную матрицу H линейного кода по его порождающей матрице G
def get_check_matrix(G):
    params = check_matrix(G)
    k = params[0]
    n = params[1]
    assert(params[2])
    pi = list(range(n)) # Вектор перестановок
    Gsh = G
    iloc = find_unity_columns(Gsh)
    iloc = tune_uniq_unity_columns(iloc, k, Gsh)
    # Перетасовываем матрицу G до тех пор, пока не получим матрицу, содержащую
    # k уникальных единичных столбцов - базис
    while not iloc:
        Gsh = shuffle_matrix(G, n, False)
        iloc = find_unity_columns(Gsh)
        iloc = tune_uniq_unity_columns(iloc, k, Gsh)
    assert(len(iloc) == k)
    # Индексы остальных столбцов - не базис
    niloc = list(set(range(n)) - set(iloc))
    niloc.sort()
    Gsht = transpose(Gsh)
    for c in iloc:
        i = Gsht[c].index(1) # Позиция единицы в столбце
        # Указываем размещение единичных столбцов Gsh в порядке 
        # единичной матрицы
        pi[i] = c
    # Для остальных столбцов Gsh указываем размещение после единичной матрицы 
    # в порядке возрастания их индексов
    pi[k:] = niloc
    # Из небазисных столбцов формируем матрицу Q^T
    Qt = [Gsht[i] for i in niloc]
    # Формируем проверочную матрицу H в систематической форме
    H = augment(Qt, identity(n - k))
    # Переставляем столбцы H для приведения матрицы в соответствие с кодом,
    # образованным порождающей матрицей G
    Hp = permute_columns(H, pi)
    return Hp

