# Модуль для работы с двоичными Линейными Блочными (n,k) Кодами
# Для хранения матриц и векторов использует список list() из чисел 0 и 1
from random import randint
from random import choices
from random import choice
from random import random
import operator
from functools import reduce
from itertools import product
from itertools import combinations
from copy import deepcopy
from scipy.special import comb
import numpy as np
from progress.bar import Bar
from progress.bar import ChargingBar

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
    z = map(operator.add, x, y)
    return list(map(operator.mod, z, mod2))

# Возврашает сумму по модулю два элементов вектора v
def xor1(v):
    return reduce(lambda x, y: x ^ y, v)

# Возврашает НЕ от логического ИЛИ элементов вектора v
def nor1(v):
    return 1 - reduce(lambda x, y: x | y, v)

def hamming_weight(v):
    return sum(filter(lambda x: x != 0 , v))

# Возвращает произведение вектора на матрицу
def mult_v(v, M):
    k = len(v)
    k_, n, ok = check_matrix(M)
    assert(k == k_)
    assert(ok)
    result = [0] * n
    for r in range(k):
        if v[r] == 1:
            result = xor(result, M[r])
    return result

# Возвращает произведение двух матриц
def mult_M(A, B):
    kA, nA, okA = check_matrix(A)
    kB, nB, okB = check_matrix(B)
    assert(nA == kB)
    assert(okA)
    assert(okB)
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
    Ik = identity(k)
    r = n - k
    defect = ((1 // r) or (1 // k))
    assert(d_low > 1)
    assert(d_low <= r + defect)
    Ir = identity(r)
    Q = []
    lq = len(Q)
    max_pops = n * (k * (2 ** r)) // r
    pops = 0
    eldl = exists_linear_dependence_level
    iterations = 1
    bar = Bar('Processing', max = k)
    while len(Q) < k:
        w = -1
        lq = len(Q)
        while w < d_low - 1:
            q = choices([0, 1], \
            weights = [r + defect + 1 - d_low, lq + d_low], k = r)
            w = hamming_weight(q)
        failed = False
        for i in range(1, min(d_low - 1, lq + 1) + 1):
            failed = eldl(Q, i, d_low - 1 - i, lq, q)
            if failed:
                break
        Q.append(q)
        lq += 1
        if failed:
            Q.pop()
            lq -= 1
            pops += 1
            if pops > max_pops:
                print(f'\n... failed, formed {lq} rows, pops = {pops}, \
iterations = {iterations}', flush = True)
                pops = 0
                Q = []
                iterations += 1
                bar = Bar('Processing', max = k)
        else:
            bar.next()
            if lq == k - 1:
                #print(f'... last row searching', flush = True)
                it = product([0, 1], repeat = r)
                for q in it:
                    q = list(q)
                    w = hamming_weight(q)
                    if w < d_low - 1:
                        continue
                    lq = len(Q)
                    failed = False
                    for i in range(1, min(d_low - 1, lq + 1) + 1):
                        failed = eldl(Q, i, d_low - 1 - i, lq, q)
                        if failed:
                            break
                    Q.append(q)
                    lq += 1
                    if failed:
                        Q.pop()
                    else:
                        bar.next()
                        break
                if len(Q) < k:
                    pops = 0
                    Q = []
                    iterations += 1
                    print(f'\n... failed {k}th (last) row searching', flush = True)
                    bar = Bar('Processing', max = k)
    bar.finish()
    print(f'\nfinished, formed {len(Q)} rows, {iterations} iterations', flush = True)
    G = augment(Ik, Q)
    return G, iterations

# По порождающей матрице G возвращает спектр кода
def gen_spectrum(G):
    k = len(G)
    it = product([0, 1], repeat = k)
    ws = {}
    bar = ChargingBar('Processing', max = 2**k // k)
    i = 0
    for a in it:
        s = mult_v(list(a), G)
        w = sum(s)
        ws[w] = ws.get(w, 0) + 1
        i += 1
        if not (i % k):
            bar.next()
    ws = dict(sorted(ws.items()))
    bar.finish()
    return ws

# Возвращает кодовое (минимальное) расстояние d по спектру кода ws.
# Спектр ws = {w: v} не должен содержать нулевых значений v.
def spectrum_to_code_distance(ws):
    ws_ = deepcopy(ws)
    del ws_[0]
    d = list(ws_.keys())
    return min(d)

# Возвращает True, если в матрице M найдется ровно m линейно зависимых строк
def exists_linear_dependence(M, m, rows):
    assert(m <= rows)
    result = False
    it = combinations(range(rows), m)
    for index in it:
        w = [int(i in index) for i in range(rows)]
        l = mult_v(w, M) # Линейная комбинация строк с весами w
        result = nor1(l)
        if result:
            break
    return result

# Возвращает True, если в матрице M найдется ровно m строк, линейная комбинация
# которых дает строку с весом не выше lev. Если в функцию передана.строка 
# include, то она обязана войти в линейную комбинацию и из матрицы М будет 
# взята m-1 строка.
def exists_linear_dependence_level(M, m, lev, rows, include):
    assert(m - int(bool(include)) <= rows)
    result = False
    if not include:
        it = combinations(range(rows), m)
        for index in it:
            w = [int(i in index) for i in range(rows)]
            l = mult_v(w, M) # Линейная комбинация строк с весами w
            result = (sum(l) <= lev)
            if result:
                break
    else:
        if m == 1:
            result = (sum(include) <= lev)
        else:
            it = combinations(range(rows), m - 1)
            for index in it:
                w = [int(i in index) for i in range(rows)]
                l = mult_v(w, M) # Линейная комбинация строк с весами w
                l = xor(l, include)
                result = (sum(l) <= lev)
                if result:
                    break
    return result

# Возвращает кодовое расстояние (n, k)-кода по его проверочной матрице
# Последовательно начиная с m = 1 ищет первую попавшуюся группу из m линейно
# зависимых столбцов матрицы H. При этом кодовое расстояние равно d = m.
# Если silence = True, то текущая оценка кодового расстояния на экран 
# не выводится.
def get_code_distance(H, silence):
    r, n, ok = check_matrix(H)
    assert(ok)
    Ht = transpose(H)
    d = 1
    while True:
        res = exists_linear_dependence(Ht, d, n)
        if res:
            break
        else:
            d += 1
            if not silence:
                print(f'... at distance {d} estimation...', flush = True)
    return d

# Возвращает перемешанную матрицу, а также список соответствующих пар 
# индексов (i1, i2), где i1 - индекс строки, которая перезаписала с помощью xor
# строку с индексом i2. При перестановке столбцов i1 и i2 - индексы 
# переставляемых столбцов.
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
    shuffled_rows = []
    shuffled_cols = []
    if rows > 1:
        for i in range(n_sh):
            i1, i2 = get_random_pair(rows)
            while i1 in locked_rows and i2 in locked_rows:
                i1, i2 = get_random_pair(rows)
            g1 = result[i1]
            g2 = result[i2]
            if i1 in locked_rows:
                result[i1] = xor(g1, g2)
                shuffled_rows.append((i2, i1))
            elif i2 in locked_rows:
                result[i2] = xor(g1, g2)
                shuffled_rows.append((i1, i2))
            else:
                b = randint(0, 1)
                to_ = i1 * b + i2 * (1 - b)
                from_ = i1 * (1 - b) + i2 * b
                result[to_] = xor(g1, g2)
                shuffled_rows.append((from_, to_))
    if with_columns:
        params = check_matrix(result)
        cols = params[1]
        assert(params[2])
        for i in range(n_sh):
            (i1, i2) = get_random_pair(cols)
            shuffled_cols.append((i1, i2))
            for j in range(rows):
                result[j][i1], result[j][i2] = result[j][i2], result[j][i1]
    return result, shuffled_rows, shuffled_cols

# Возвращает матрицу, полученную путем перестановки столбцов матрицы M по 
# правилу p, в котором указаны индексы куда переставлять столбцы
def permute_columns(M, p):
    rows, cols, ok = check_matrix(M)
    assert(ok)
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
        Msh, *_ = shuffle_matrix(M, cols, False)
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
    eld = exists_linear_dependence
    grss = get_random_square_submatrix
    iloc_basis = set() # Индексы столбцов M под базис
    while nonBasis: # Пока не найден базис
        Msq, iloc_basis = grss(MT, rows, cols)
        ir = 1
        _nb = True
        while ir <= rows:
            _nb = eld(Msq, ir, rows)
            if _nb or ir == rows:
                break
            ir += 1
        if ir == rows and _nb == False: # Если нигде не было линейной 
        # зависимости, то нашли базис (линейно независимые столбцы M)
            nonBasis = False
    return iloc_basis

# Версия 2
def find_basis_candidates_2(M, rows, cols):
    assert(rows <= cols)
    MT = transpose(M)
    eldl = exists_linear_dependence_level
    iter = 1
    used = set()
    free = set(range(cols))
    Sq = []
    iloc_basis = set() # Индексы столбцов M под базис
    l = len(Sq)
    max_iter = 2 * cols # Максимальное число выборок строки при неудачах
    while l < rows:
        i = choice(tuple(free))
        ir = 1
        is_good = ir > l+1
        while not eldl(Sq, ir, 0, l, MT[i]):
            ir += 1
            is_good = ir > l+1
            if is_good:
                break
        if is_good:
            iloc_basis.add(i)
            Sq.append(MT[i])
            l += 1
            used.add(i)
            free.remove(i)
        else:
            iter += 1
        if iter > max_iter: # Сброс набора из-за его неудачности в плане базиса
            iter = 1
            used = set()
            free = set(range(cols))
            used = set()
            Sq = []
            iloc_basis = set() # Индексы столбцов M под базис
            l = len(Sq)
    return iloc_basis

# Версия 2. Алгоритмически ускорен расчет
# (rows, cols) - размеры матрицы M
def reduce_to_basis_2(M, rows, cols):
    # Индексы, закрепленные под базис
    print(f'... search basis ...', flush = True)
    iloc_basis = find_basis_candidates_2(M, rows, cols)
    print(f'    basis is found: {iloc_basis}', flush = True)
    Msh = M
    rows_locked = [] # Индексы строк для блокировки
    locked = len(rows_locked)
    fuc = find_unity_columns
    fuuc = filter_uniq_unity_columns
    shm = shuffle_matrix
    while True:
        # Ищем все единичные столбцы
        where_unity = set(fuc(Msh))
        # Отбираем те индексы, которые пересекаются с закрепленными под базис.
        # Сортируем по возрастанию.
        active_unity = sorted(list(iloc_basis.intersection(where_unity)))
        # Отфильтровываем возможные повторяющиеся столбцы и одновременно
        # определяем позиции единиц
        uniq_cols = fuuc(active_unity, Msh)
        # Индексы строк для блокировки уже сформированных единичных столбцов
        rows_locked = uniq_cols.keys()
        tmp = len(rows_locked)
        if tmp > locked:
            locked = tmp
            print(f'... locked {locked} columns from {rows}...', flush = True)
        if locked == rows: # Все rows столбцов - единичные и разные
            break
        # Тасуем матрицу путем xor двух случайных строк с учетом того, что
        # нельзя "портить" уже сформированные единичные базисные столбцы. 
        # Указываются индексы строк, в которых стоит единица соответствующего 
        # единичного столбца.
        Msh, *_ = shm(Msh, 1, False, rows_locked)
    # Определяем индексы остальных столбцов - небазисных
    niloc = list(set(range(cols)) - set(iloc_basis))
    niloc.sort() # Сортируем
    return Msh, iloc_basis, niloc

# Возвращает квадратную подматрицу размером (rows, rows) выбором случайных 
# строк матрицы M, содержащей cols строк
def get_random_square_submatrix(M, rows, cols):
    assert(rows <= cols)
    assert(len(M[0]) == rows)
    Sq = []
    used = set()
    free = set(range(cols))
    while len(used) < rows:
        i = choice(tuple(free))
        Sq.append(M[i])
        used.add(i)
        free.remove(i)
    return Sq, used

# Возвращает проверочную матрицу H линейного кода по его порождающей матрице G
def get_check_matrix(G):
    k, n, ok = check_matrix(G)
    assert(ok)
    pi = list(range(n)) # Вектор перестановок
    # BS = reduce_to_basis(G, k, n)
    Gsh, iloc, niloc = reduce_to_basis_2(G, k, n)
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

# Возвращает классы смежности {c: [e]} по проверочной матрице H кода.
# Здесь c - вектор синдрома c = eH^T, [e] - набор соответствующих векторов 
# ошибок
def get_adjacent_classes(H):
    r, n, ok = check_matrix(H)
    assert(ok)
    ac = {}
    it = product([0, 1], repeat = r)
    for c in it:
        address = tuple(c)
        ac[address] = []
    HT = transpose(H)
    it = product([0, 1], repeat = n)
    for e in it:
        e = list(e)
        c = mult_v(e, HT)
        address = tuple(c)
        ac[address].append(e)
    return ac

# Возвращает минимальный класс смежности {c: e} по проверочной матрице H кода.
# Здесь c - вектор синдрома c = eH^T, e - соответствующий вектор ошибки 
# минимальной кратности
def get_min_adjacent_classes(H):
    r, n, ok = check_matrix(H)
    assert(ok)
    ac = {}
    HT = transpose(H)
    ones = [1] * n
    it = product([0, 1], repeat = n)
    for e in it:
        e = list(e)
        address = tuple( mult_v(e, HT) )
        w = hamming_weight(e)
        e_present = ac.get(address, ones)
        w_present = hamming_weight(e_present)
        if w < w_present:
            ac[address] = e
    return ac

# Версия 2: исключен перебор 2^n векторов ошибок => ускорен расчет
def get_min_adjacent_classes_2(H):
    r, n, ok = check_matrix(H)
    assert(ok)
    ac = {}
    HT = transpose(H)
    ones = [1] * n
    Nc = 2 ** r
    q = 0
    while True:
        it = combinations(range(n), q)
        overwrite = False
        for index in it:
            e = [int(i in index) for i in range(n)]
            address = tuple( mult_v(e, HT) )
            #w = hamming_weight(e)
            #assert(w == q)
            e_present = ac.get(address, ones)
            w_present = hamming_weight(e_present)
            presented = (q >= w_present)
            if not presented:
                ac[address] = e
        if len(ac) == Nc:
            break
        q += 1
    return ac

# Возвращает кортеж в виде декодированного кодового вектора s, вектора ошибки
# e и синдрома c. Принятый вектор v декодируется на основании проверочной 
# матрицы H кода и предварительно найденных с помощью get_adjacent_class() 
# классов смежности ac. Стратегия декодирования основана на случайном выборе 
# вектора ошибки из набора всех возможных векторов с наименьшим весом.
def correct(v, H, ac):
    c = mult_v(v, transpose(H))
    address = tuple(c)
    es = ac[address]
    ws = []
    for e in es:
        ws.append(hamming_weight(e))
    min_w = min(ws)
    es_min_w = []
    for e in es:
        if hamming_weight(e) == min_w:
            es_min_w.append(e)
    ec = choice(es_min_w) # Случайный выбор вектора ошибки с наименьшим весом
    return xor(v, ec), list(ec), c

# Возвращает вектор w, являющийся решением системы линейных уравнений v = w * M
# Матрица M должна быть квадратной
def solve_le(v, M):
    rows, cols, ok = check_matrix(M)
    assert(ok)
    assert(rows == cols)
    Msh = transpose(M)
    rows_locked = [] # Индексы строк для блокировки
    while True:
        where_unity = find_unity_columns(Msh)
        active_unity = sorted(where_unity)
        uniq_cols = filter_uniq_unity_columns(active_unity, Msh)
        # Индексы строк для блокировки уже сформированных единичных столбцов
        rows_locked = uniq_cols.keys()
        if len(rows_locked) == rows: # Все rows столбцов - единичные и разные
            break
        Msh, sh_rows, sh_cols = shuffle_matrix(Msh, 1, False, rows_locked)
        for row in sh_rows:
            v[row[1]] = (v[row[0]] + v[row[1]]) % 2
    Msh = transpose(Msh)
    where_unity = find_unity_columns(Msh)
    uniq_cols = filter_uniq_unity_columns(active_unity, Msh)
    w = [0] * rows
    for key, val in uniq_cols.items():
        w[key] = v[val]
    return w

# Возвращает информационный вектор a по кодовому вектору s и порождающей 
# матрице G
def decode(s, G):
    k, n, ok = check_matrix(G)
    assert(ok)
    iloc_basis = find_basis_candidates_2(G, k, n)
    GT = transpose(G)
    s_cut = []
    GT_cut = []
    for i in iloc_basis:
        s_cut.append(s[i])
        GT_cut.append(GT[i])
    G_cut = transpose(GT_cut)
    a = solve_le(s_cut, G_cut)
    return a

# По параметрам линейного (n, k)-кода разрешает неравенство Хемминга и 
# возвращает наибольшую кратность исправляемой ошибки qi
def resolve_hamming_constrain(n, k):
    assert(k < n)
    r = n - k
    N_cyndromes = np.power(2., r)
    qi = r // 4
    N_errors = 0
    for i in range(qi + 1):
        N_errors += comb(n, i)
    overhead = (N_errors > N_cyndromes)
    step = False
    while not step:
        if overhead:
            N_errors -= comb(n, qi)
            qi -= 1
        else:
            qi += 1
            N_errors += comb(n, qi)
        step = overhead ^ (N_errors > N_cyndromes)
        overhead = (N_errors > N_cyndromes)
    if overhead:
        N_errors -= comb(n, qi)
        qi -= 1
    return qi

# Возвращает вероятность q-кратной ошибки в слове из n битов при его передаче
# через BSC-канал, при этом вероятность ошибки в одном бите равна p.
# BSC - Binary Symmetric Channel - Двоичный симметричный канал с 
# независимыми ошибками.
def probability_bsc(q, n, p):
    return comb(n, q) * np.power(float(p), q) * np.power(1. - p, n - q)

# Возвращает вероятность того, что при передаче слова из n битов через 
# BSC-канал произойдет ошибка кратности выше q_low. Также возвращает 
# вероятность противоположного события.
# BSC - Binary Symmetric Channel - Двоичный симметричный канал с независимыми
# ошибками.
def probability_bsc_more(q_low, n, p):
    tmp = 0.
    if n * p > q_low:
        for q in range(0, q_low + 1):
            tmp += probability_bsc(q, n, p)
        p_err, p_compl = 1. - tmp, tmp
    else:
        for q in range(q_low + 1, n + 1):
            tmp += probability_bsc(q, n, p)
        p_err, p_compl = tmp, 1. - tmp
    return p_err, p_compl

# Возвращает рекомендуемое кодовое расстояние. 
# Альтернатива неравенству Хемминга.
def get_recomend_code_distance(n, k):
    if n - k == 1: # Код с проверкой на четность
        return 2
    elif k == 1:
        return n   # Код с повторением
    else:
        return max(2, int(2. * (n - k) / np.log2(n + 1.) + 1.))

