# Скрипт для тестирования модуля linear_codes.py
import linear_codes as lc
import time
from pprint import pprint as pp

# Ограничения на параметры (n, k) кода
min_n = 3
max_n = 17
min_k = 1
max_k = 15
min_r = 2

assert(min_k < min_n)
assert(max_k < max_n)

while True:
    # Генерация случайных (n, k) так, что k < n
    while True:
        n = lc.randint(min_n, max_n)
        k = lc.randint(min_k, max_k)
        if k < n and (n - k) >= min_r:
            break

    print(f'*******************')
    r = n - k
    d_max = r + 1
    d_low = max( d_max // 2, 2)
    print(f'n = {n}, k = {k}, r = {r}', flush = True)
    print(f'Generate a generator matrix G...', flush = True)
    t0 = time.perf_counter()
    G = lc.gen_matrix(n, k, d_low)
    t1 = time.perf_counter()
    print(f'Elapsed {t1 - t0} s', flush = True)
    print(f'Shuffling G matrix...', flush = True)
    t0 = time.perf_counter()
    Gsh = lc.shuffle_matrix(G, n, True, [])
    t1 = time.perf_counter()
    # pp(f'Generator matrix: {Gsh}')
    print(f'Elapsed {t1 - t0} s', flush = True)
    print(f'Find the parity check matrix H...', flush = True)
    t0 = time.perf_counter()
    Hsh = lc.get_check_matrix(Gsh)
    t1 = time.perf_counter()
    # pp(f'Check matrix: {Hsh}')
    print(f'Elapsed {t1 - t0} s', flush = True)
    print(f'Find code distance by check matrix...', flush = True)
    t0 = time.perf_counter()
    d = lc.get_code_distance(Hsh)
    t1 = time.perf_counter()
    print(f'Elapsed {t1 - t0} s', flush = True)
    print(f'Generation code and calc spectrum and distance...', flush = True)
    t0 = time.perf_counter()
    Code, Wsp, dist = lc.gen_code(Gsh)
    t1 = time.perf_counter()
    print(f'Elapsed {t1 - t0} s', flush = True)
    print(f'd_max = {d_max}, d_low = {d_low}, d = {d}, d_dist = {dist}', flush = True)
    assert(d == dist)
    assert(d >= d_low)
    assert(dist >= d_low)
    assert(Wsp[0] == 1)
    assert(sum(Wsp.values()) == 2**k)
    assert(len(Code) == 2**k)

