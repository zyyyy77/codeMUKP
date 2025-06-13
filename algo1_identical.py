import time
import numpy as np
import pandas as pd
import math

demandData = pd.read_excel("data.xlsx", sheet_name='demandData')
w = demandData['profit']
d = demandData['demand']

I = len(w)
M = 20
K = 300
r = 0.9

def binomial(m, n):
    aa = bb = result = 1
    minNI = min(n, m-n)
    for j in range(0, minNI):
        aa = aa*(m-j)
        bb = bb*(minNI-j)
        result = aa/bb
    return result


# w_i(q_i)
def margFunc(kk, mm, dd, ww):
    c = 0
    for x in range(1, M+1, 1):
        low = max(0, x-1-M+mm)
        high = min(min(mm - 1, dd - x * kk + x - 1), x-1)
        for a in range(low, high + 1, 1):
            b = x-1-a
            c = c + binomial(mm-1, a) * binomial(M-mm, b) * (r**x) * ((1-r)**(M-x));
    return ww * c

start = time.time()
e = 0
S = []
k = np.ones(I)
m = np.ones(I)
q = np.zeros(I)
pi = np.zeros(I)

gross = 0

i_optimal = 0
i_max = 0

while e < M * K:

    pi[i_optimal] = margFunc(int(k[i_optimal]), int(m[i_optimal]), int(d[i_optimal]), w[i_optimal])
    if i_max + 1 < I:
        pi[i_max+1] = margFunc(int(k[i_max+1]), int(m[i_max+1]), int(d[i_max+1]), w[i_max+1])

    gross = gross + np.max(pi)
    i_optimal = np.argmax(pi)
    S.append(i_optimal)

    q[i_optimal] = q[i_optimal] + 1
    i_max = max(i_max, i_optimal)
    k[i_optimal] = math.ceil((q[i_optimal]+1)/M)
    m[i_optimal] = (q[i_optimal]+1) % M
    if m[i_optimal] == 0:
        m[i_optimal] = M

    e = e + 1

S.sort()
place = np.array(S).reshape(K, M) + 1
print("Final placement result:")
print(place)

print("Expected profit:")
print(np.round(gross, 2))

end = time.time()
print("Program runtime:")
print(end - start)
print(np.round(end - start, 3), 's')

