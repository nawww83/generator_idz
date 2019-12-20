# Модуль для работы с двоичными Линейными Блочными (n,k) Кодами
# Для хранения матриц и векторов использует список list() из чисел 0 и 1
from random import randint
from random import choice
from random import random
import operator
from functools import reduce
from itertools import product
from itertools import combinations
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

def hamming_weight(v):
    return len(list(filter(lambda x: x != 0 , v)))

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

# Возвращает случайный двоичный вектор ошибки с вероятностью ошибки p
def get_error_vector(n, p):
    return [int(random() < p) for _ in range(n)]

# Возвращает случайную порождающую матрицу линейного (n, k) кода
# в систематической форме G = [I, Q] с кодовым расстоянием не ниже d_low
def gen_matrix(n, k, d_low):
    assert(n > 1)
    assert(k < n)
    I = identity(k)
    r = n - k
    while True:
        G = []
        Q = []
        w_min = r
        for _ in range(k):
            q = get_rand_bits(r)
            w = hamming_weight(q)
            w_min = min(w, w_min)
            Q.append(q)
        d_high = w_min + 1 # Оценка кодового расстояния сверху
        if d_high < d_low:
            continue
        G = augment(I, Q)
        d = gen_code(G)[2] # Вычисление кодового расстояния
        if d < d_low:
            continue
        else:
            break
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

# Возвращает True, если в матрице M найдется ровно m линейно зависимых строк
def exists_linear_dependence(M, m, rows):
    assert(m <= rows)
    result = False
    it = combinations(range(rows), m)
    for index in it:
        w = [0] * rows
        for i in index:
            w[i] = 1
        l = mult_v(w, M) # Линейная комбинация строк с весами w
        result = (sum(l) == 0)
        if result:
            break
    return result

# Возвращает кодовое расстояние (n, k)-кода по его проверочной матрице
# Последовательно начиная с m = 1 ищет первую попавшуюся группу из m линейно
# зависимых столбцов матрицы H. При этом кодовое расстояние равно d = m.
def get_code_distance(H):
    params = check_matrix(H)
    r = params[0]
    n = params[1]
    assert(params[2])
    Ht = transpose(H)
    d = 1
    while True:
        res = exists_linear_dependence(Ht, d, n)
        if res:
            break
        else:
            d += 1
            print(f'... at distance {d} estimation...', flush = True)
    return d

# Возвращает перемешанную матрицу.
# Суммируется случайная пара строк, результат записывается в одну из 
# этих строк. Процедура выполняется n_sh раз. При этом строки с индексами
# exclude_rows считаются блокированными так, что они не "испортят" других 
# строк операцией xor, при этом сами могут быть "испорчеными", но только 
# свободными строками, т.е. не принадлежащими списку exclude_rows.
# Затем, если with_columns = True, делается перестановка столбцов n_sh раз.
def shuffle_matrix(M, n_sh, with_columns, exclude_rows):
    rows = len(M)
    result = deepcopy(M)
    locked_rows = set(exclude_rows)
    if rows > 1:
        for i in range(n_sh):
            i1, i2 = get_random_pair(rows)
            while i1 in locked_rows and i2 in locked_rows:
                i1, i2 = get_random_pair(rows)
            g1 = result[i1]
            g2 = result[i2]
            if i1 in locked_rows:
                result[i1] = xor(g1, g2)
            elif i2 in locked_rows:
                result[i2] = xor(g1, g2)
            else:
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

# Возвращает индексы i уникальных единичных столбцов в порядке iloc и позиции p 
# единиц в этих столбцах в виде словаря {p: i}. Требует предварительного 
# определения индексов iloc единичных столбцов матрицы M.
def filter_uniq_unity_columns(iloc, M):
    MT = transpose(M)
    d = set()
    iloc_ = {}
    for i in iloc:
        p = MT[i].index(1)
        if p not in d:
            iloc_[p] = i
        d.add(p)
    return iloc_

# Возвращает транспонированную матрицу, т.е. Y = X^T
def transpose(X):
    return list(map(list, zip(*X)))

# Возвращает матрицу, эквивалентную M, такую, что она содержит базисные столбцы
# Также возвращает индексы базисных строк iloc и небазисных niloc
# Необходимо передавать верное число строк rows и столбцов cols, а также
# правильную матрицу M, т.к. функция не контролирует правильность.
def reduce_to_basis(M, rows, cols):
    Msh = M
    iloc = find_unity_columns(Msh)
    iloc = filter_uniq_unity_columns(iloc, Msh)
    # Перетасовываем матрицу M до тех пор, пока не получим матрицу, содержащую
    # rows уникальных единичных столбцов - базис
    while not iloc:
        Msh = shuffle_matrix(M, cols, False)
        iloc = find_unity_columns(Msh)
        iloc = tune_uniq_unity_columns(iloc, Msh)
    assert(len(iloc) == rows)
    # Индексы остальных столбцов - не базис
    niloc = list(set(range(cols)) - set(iloc))
    niloc.sort()
    return (Msh, iloc, niloc)

# Возвращает индексы столбцов M, которые могут быть базисными
def find_basis_candidates(M, rows, cols):
    assert(rows <= cols)
    MT = transpose(M)
    nonBasis = True
    iloc_basis = set() # Индексы столбцов M под базис
    while nonBasis: # Пока не найден базис
        Msq, iloc_basis = get_random_square_submatrix(MT, rows, cols)
        ir = 1
        _nb = True
        while ir <= rows:
            _nb = exists_linear_dependence(Msq, ir, rows)
            if _nb or ir == rows:
                break
            ir += 1
        if ir == rows and _nb == False: # Если нигде не было линейной 
        # зависимости, то нашли базис (линейно независимые столбцы M)
            nonBasis = False
    return iloc_basis

# Версия 2. Алгоритмически ускорен расчет
# (rows, cols) - размеры матрицы M
def reduce_to_basis_2(M, rows, cols):
    # Индексы, закрепленные под базис
    iloc_basis = find_basis_candidates(M, rows, cols)
    Msh = M
    rows_locked = [] # Индексы строк для блокировки
    while True:
        # Ищем все единичные столбцы
        where_unity = set(find_unity_columns(Msh))
        # Отбираем те индексы, которые пересекаются с закрепленными под базис.
        # Сортируем по возрастанию.
        active_unity = sorted(list(iloc_basis.intersection(where_unity)))
        # Отфильтровываем возможные повторяющиеся столбцы и одновременно
        # определяем позиции единиц
        uniq_cols = filter_uniq_unity_columns(active_unity, Msh)
        # Индексы строк для блокировки уже сформированных единичных столбцов
        rows_locked = uniq_cols.keys()
        if len(rows_locked) == rows: # Все rows столбцов - единичные и разные
            break
        # Тасуем матрицу путем xor двух случайных строк с учетом того, что
        # нельзя "портить" уже сформированные единичные базисные столбцы. 
        # Указываются индексы строк, в которых стоит единица соответствующего 
        # единичного столбца.
        Msh = shuffle_matrix(Msh, 1, False, rows_locked)
    # Определяем индексы остальных столбцов - небазисных
    niloc = list(set(range(cols)) - set(iloc_basis))
    niloc.sort() # Сортируем
    return Msh, iloc_basis, niloc

# Возвращает квадратную подматрицу размером (rows, rows) выбором случайных 
# строк матрицы M, содержащей cols строк
def get_random_square_submatrix(M, rows, cols):
    assert(rows <= cols)
    Sq = []
    used = set()
    free = list(range(cols))
    while len(used) < rows:
        i = choice(free)
        if i not in used:
            Sq.append(M[i])
            used.add(i)
            free.remove(i)
    return Sq, used

# Возвращает проверочную матрицу H линейного кода по его порождающей матрице G
def get_check_matrix(G):
    params = check_matrix(G)
    k = params[0]
    n = params[1]
    assert(params[2])
    pi = list(range(n)) # Вектор перестановок
    # BS = reduce_to_basis(G, k, n)
    BS = reduce_to_basis_2(G, k, n)
    Gsh = BS[0]
    iloc = BS[1]
    niloc = BS[2]
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

# Возвращает декодированный кодовый вектор s по принятому вектору v и 
# проверочной матрице H
def correct(v, H):
    
    return s

# Возвращает информационный вектор a по кодовому вектору s и 
# порождающей матрице G
def decode(s, G):
    return a