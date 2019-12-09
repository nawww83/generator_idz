from random import randint
import operator
from functools import reduce
from itertools import product
from copy import deepcopy

def check_matrix(M):
    rows = len(M)
    nn = len(reduce(operator.iconcat, M, []))
    cols = nn // rows
    ok = True
    for r in range(rows):
        ok = ok and (len(M[r]) == cols)
    return (rows, cols, ok)

def get_random_pair(n):
    i1 = 0
    i2 = 0
    while i1 == i2:
        i1 = randint(1, n) - 1
        i2 = randint(1, n) - 1
    return (i1, i2)

def augment(I, Q):
    k = len(I)
    k2 = len(Q)
    assert(k == k2)
    R = []
    for row in range(k):
        R.append(I[row] + Q[row])
    return R

def identity(k):
    z = [0] * k
    I = []
    for row in range(k):
        I.append(z[:])
        I[row][row] = 1
    return I

def xor(x, y):
    assert(len(x) == len(y))
    mod2 = [2] * len(x)
    z = list(map(operator.add, x, y))
    return list(map(operator.mod, z, mod2))

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

def get_rand_bits(n):
    return [randint(0, 1) for _ in range(n)]

def add_row_with_sys(G_sys, g):
    G_sys.append(g[:])
    params = check_matrix(G_sys)
    k = params[0]
    n = params[1]
    assert(params[2] == True)
    g_tmp = G_sys[-1]
    result = False
    if k == 1:
        result = bool(g_tmp[0])
    else:
        # Append row with reduce matrix to Upper Triangular
        for c in range(k-1):
            if g_tmp[c] == 1:
                g_tmp = xor(g_tmp, G_sys[c])
        G_sys[-1] = g_tmp
        result = bool(g_tmp[k-1])
    if not result:
        G_sys.pop()
    else:
    # convert Upper Triangular matrix to Diagonal
        for r in range(k - 1):
            g_tmp = G_sys[r]
            for c in range(r + 1, k):
                if g_tmp[c] == 1:
                    g_tmp = xor(g_tmp, G_sys[c])
            G_sys[r] = g_tmp
    return result

def gen_matrix_tmp(n, k):
    assert(n > 1)
    assert(k < n)
    G = []
    G_sys = []
    i_gen = 0
    for _ in range(k):
        basisOk = False
        while not basisOk:
            g = get_rand_bits(n)
            i_gen += 1
            basisOk = add_row_with_sys(G_sys, g)
        G.append( g )
    return (G, G_sys, i_gen)

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
    dave = sum(list(map(operator.mul, d, v))) / sum(v)
    return (code, ws, dmin, dave)

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