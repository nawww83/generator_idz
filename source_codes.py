# Модуль "Кодирование источника" (экономное кодирование)

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

