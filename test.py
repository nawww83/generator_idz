# Скрипт для тестирования модуля linear_codes.py
import linear_codes as lc
import time
from pprint import pprint as pp
from random import choice
import numpy as np

# Ограничения на параметры (n, k) кода
min_n = 3
max_n = 18
min_k = 1
max_k = 15
min_r = 1

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
    d_max = lc.get_recomend_code_distance(n, k)
    d_low = d_max
    print(f'n = {n}, k = {k}, r = {r}', flush = True)
    print(f'Generate a generator matrix G...', flush = True)
    t0 = time.perf_counter()
    G, _ = lc.gen_matrix(n, k, d_low)
    t1 = time.perf_counter()
    print(f'Elapsed {t1 - t0} s', flush = True)
    print(f'Shuffling G matrix...', flush = True)
    t0 = time.perf_counter()
    Gsh, *_ = lc.shuffle_matrix(G, n, True, [])
    t1 = time.perf_counter()
    # pp(f'Generator matrix: {Gsh}')
    print(f'Elapsed {t1 - t0} s', flush = True)
    print(f'Find the parity check matrix H...', flush = True)
    t0 = time.perf_counter()
    Hsh = lc.get_check_matrix(Gsh)
    t1 = time.perf_counter()
    # pp(f'Check matrix: {Hsh}')
    print(f'Elapsed {t1 - t0} s', flush = True)
    print(f'Check zero property GH^T = HG^T = 0', flush = True)
    t0 = time.perf_counter()
    zero = sum(map(sum, lc.mult_M(Gsh, lc.transpose(Hsh))))
    t1 = time.perf_counter()
    ok = (zero == 0)
    print(f'Ok = {ok}', flush = True)
    assert(ok)
    print(f'Elapsed {t1 - t0} s', flush = True)
    print(f'Find code distance by check matrix...', flush = True)
    t0 = time.perf_counter()
    d = lc.get_code_distance(Hsh, False)
    t1 = time.perf_counter()
    print(f'Elapsed {t1 - t0} s', flush = True)
    print(f'Find code distance by check matrix (method 2)...', flush = True)
    t0 = time.perf_counter()
    d2 = lc.get_code_distance_2(Hsh, False)
    t1 = time.perf_counter()
    print(f'Elapsed {t1 - t0} s', flush = True)
    print(f'Generation code and calc spectrum and distance...', flush = True)
    t0 = time.perf_counter()
    Wsp = lc.gen_spectrum(Gsh)
    dist = lc.spectrum_to_code_distance(Wsp)
    t1 = time.perf_counter()
    print(f'Elapsed {t1 - t0} s', flush = True)
    print(f'd_max = {d_max}, d_low = {d_low}, d = {d}, d2 = {d2}, \
        d_dist = {dist}', flush = True)
    assert(d == dist)
    assert(d2 == dist)
    assert(d >= d_low)
    assert(Wsp[0] == 1)
    assert(sum(Wsp.values()) == 2**k)
    a = lc.get_rand_bits(k)
    s = lc.mult_v(a, Gsh)
    print(f'Transmitted code vector s = {s}')
    qi = (d - 1) // 2 # Целевая кратность ошибки - кратность исправления
    p = 1. * qi / n # Средняя кратность случайной величины q = np
    e = lc.get_error_vector(n, p)
    q = lc.hamming_weight(e) # Получившаяся кратность ошибки
    print(f'Error vector e = {e} with weight {q}')
    v = lc.xor(s, e)
    print(f'Received code vector v = {v}')
    print(f'Generation min adjacent classes...', flush = True)
    t0 = time.perf_counter()
    ac = lc.get_min_adjacent_classes(Hsh)
    t1 = time.perf_counter()
    print(f'Elapsed {t1 - t0} s', flush = True)
    print(f'Correction received vector...', flush = True)
    t0 = time.perf_counter()
    c_ = lc.mult_v(v, lc.transpose(Hsh))
    e_ = ac[tuple(c_)]
    s_est = lc.xor(v, e_)
    t1 = time.perf_counter()
    print(f'Elapsed {t1 - t0} s', flush = True)
    print(f'Corrected code vector s_est = {s_est}')
    print(f' by error vector e_ = {e_} and cyndrome c = {c_}')
    print(f'Decoding corrected vector...', flush = True)
    t0 = time.perf_counter()
    a_est = lc.decode(s_est, Gsh)
    t1 = time.perf_counter()
    print(f'Elapsed {t1 - t0} s', flush = True)
    print(f'Decoded vector a_est = {a_est}')
    print(f'Check decoding result...', flush = True)
    t0 = time.perf_counter()
    s_ = lc.mult_v(a_est, Gsh)
    t1 = time.perf_counter()
    print(f'Elapsed {t1 - t0} s', flush = True)
    assert(s_est == s_)
    ok_corrected = (s == s_est)
    assert( ((q <= qi) and ok_corrected) or (q > qi) )
    if ok_corrected:
        print(f'The error was corrected!')
    else:
        print(f'The error was not corrected(:')

