# Скрипт для тестирования модуля linear_codes.py
import linear_codes as lc
import time

# Ограничения на параметры (n, k) кода
min_n = 6
max_n = 20
min_k = 3
max_k = 10
min_r = 3

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
    t0 = time.perf_counter()
    G = lc.gen_matrix(n, k, d_low)
    t1 = time.perf_counter()
    print(f'Generation matrix {t1 - t0}', flush = True)
    t0 = time.perf_counter()
    Gsh = lc.shuffle_matrix(G, n, True)
    t1 = time.perf_counter()
    print(f'Shuffle G matrix {t1 - t0}', flush = True)
    #t0 = time.perf_counter()
    #Hsh = lc.get_check_matrix(Gsh)
    #t1 = time.perf_counter()
    #print(f'Find check H matrix {t1 - t0}', flush = True)
    #t0 = time.perf_counter()
    #d = lc.get_code_distance(Hsh)
    #t1 = time.perf_counter()
    #print(f'Calc code distance by H matrix {t1 - t0}', flush = True)
    t0 = time.perf_counter()
    Code, Wsp, dist = lc.gen_code(Gsh)
    t1 = time.perf_counter()
    print(f'Calc code, spectrum and distance by G matrix {t1 - t0}', flush = True)
    print(f'd_max = {d_max}, d_low = {d_low}, d = {dist}', flush = True)
    #assert(d == dist)
    #assert(d >= d_low)
    assert(dist >= d_low)
    assert(Wsp[0] == 1)
    assert(sum(Wsp.values()) == 2**k)
    assert(len(Code) == 2**k)

