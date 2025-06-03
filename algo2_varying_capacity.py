import time
import numpy as np
import pandas as pd
import math
from collections import Counter
import itertools


demandData = pd.read_excel("data.xlsx", sheet_name='demandData')
w = demandData['profit']
d = demandData['demand']

I = len(w)
MM = [4, 6]
M = sum(MM)
K1 = [60, 40]
KK = []
for l in range(len(MM)):
    for m in range(MM[l]):
        KK.append(K1[l])

K = max(KK)
r = 0.9
totalCapacity = 1 * sum(KK)

def binomial(m, n):
    aa = bb = result = 1
    minNI = min(n, m-n)
    for j in range(0, minNI):
        aa = aa*(m-j)
        bb = bb*(minNI-j)
        result = aa/bb
    return result

def margFuncAllFensan(i, kk, mm):
    c = 0
    for x in range(1, M+1, 1):
        xia = int(max(0, x-1-M+mm))
        shang = int(min(min(mm - 1, d[i] - x * kk + x - 1), x-1))
        for a in range(xia, shang + 1):
            b = x-1-a
            c = c + binomial(mm-1, a) * binomial(M-mm, b) * (r**x) * ((1-r)**(M-x));
    return c * w[i]

def margFuncPartialFensan(i, k1, m1, k2list, m2list):
    M_2 = sum(m2list)
    M_1 = M - M_2
    margin_i = 0
    for x1 in range(1, M_1 + 1):
        for x2 in range(M_2 + 1):
            for alpha1 in range(int(m1 + 1)):
                beta1 = x1 - 1 - alpha1
                if beta1 < 0 or beta1 > M_1 - m1 - 1:
                    continue
                for alpha2_list in itertools.product(*[range(m2+1) for m2 in m2list]):
                    if sum(alpha2_list) != x2:
                        continue
                    if alpha1 * k1 + beta1 * (k1 - 1) + k1 + sum([alpha2_list[a] * k2list[a] for a in range(len(k2list))]) <= d[i]:
                        a_xi = math.comb(int(m1), int(alpha1)) * math.comb(int(M_1 - m1 - 1), int(beta1))
                        for a in range(len(m2list)):
                            a_xi *= math.comb(m2list[a], alpha2_list[a])
                        margin_i += a_xi * (r ** (x1 + x2) * (1 - r) ** (M - x1 - x2))
    return margin_i * w[i]


def get_lessMachineList(qimm, ii, para_mNonEmpty):
    lessM = []
    minQuantity = max(KK)
    for m in para_mNonEmpty:
        if qimm[ii][m] < minQuantity:
            minQuantity = qimm[ii][m]
    for m in para_mNonEmpty:
        if qimm[ii][m] == minQuantity:
            lessM.append(m)
    return lessM

def get_mOptiIndex(para_remainCapacity, para_i, lessMachineList):
    if len(lessMachineList) == 1:
        return min(lessMachineList)
    else:
        qm = []
        for m in lessMachineList[para_i]:
            qm.append(para_remainCapacity[m])
            # global maxQ
            maxQ = max(qm)
        mmMax = []
        for m in lessMachineList[para_i]:
            if para_remainCapacity[m] == maxQ:
                mmMax.append(m)

        mIndex = list(set(lessMachineList[para_i]).intersection(set(mmMax)))
        return min(mIndex)

start = time.time()
e = 0
k1 = np.ones(I)
k2_list = {i: [] for i in range(I)}
m1 = np.zeros(I)
m2_list = {i: [] for i in range(I)}
pi = np.zeros(I)
remainCapacity = KK * 1
totalRevenue = 0

group1 = 1 * list(range(M))
group2 = []

qim = np.zeros((I, M))
qi = np.zeros(I)
i_max = 0
mIndex = np.zeros(I)
lessMachineList = {i: [] for i in range(I)}
m_optiIndex = np.zeros(I)
flag = False

while e < totalCapacity:

    if group2 == []:
        for i in range(min(i_max + 2, I + 1)):
            if qi[i] < d[i]:
                pi[i] = r * w[i]
            else:
                k1[i] = max(qim[i])
                if k1[i] == min(qim[i]):
                    k1[i] += 1
                m1[i] = list(qim[i]).count(k1[i]) + 1
                pi[i] = margFuncAllFensan(i, k1[i], m1[i])
    else:
        for i in range(min(i_max + 2, I + 1)):
            if qi[i] < d[i]:
                pi[i] = r * w[i]
            else:
                num_i_InGroup1 = [qim[i][int(m)] for m in group1]
                k1[i] = max(num_i_InGroup1)
                if k1[i] == min(qim[i][0:len(group1)]):
                    k1[i] += 1
                m1[i] = num_i_InGroup1.count(k1[i])

                if flag == True:
                    num_i_InGroup2 = [list(qim[i])[int(m)] for m in group2]
                    class_i_InGroup2 = Counter(num_i_InGroup2)
                    k2_list[i] = list(set(num_i_InGroup2))
                    m2_list[i] = [class_i_InGroup2[k2_list[i][m]] for m in range(len(k2_list[i]))]
                pi[i] = margFuncPartialFensan(i, k1[i], m1[i], k2_list[i], m2_list[i])

    totalRevenue += np.max(pi)
    i_optimal = np.argmax(pi)
    i_max = max(i_max, i_optimal)

    lessMachineList[i_optimal] = get_lessMachineList(qim, i_optimal, group1)
    mIndex[i_optimal] = get_mOptiIndex(remainCapacity, i_optimal, lessMachineList)

    qim[i_optimal][int(mIndex[i_optimal])] += 1
    qi[i_optimal] += 1
    remainCapacity[int(mIndex[i_optimal])] -= 1
    flag = False
    if remainCapacity[int(mIndex[i_optimal])] == 0:
        flag = True
        group1.remove(mIndex[i_optimal])
        if group1 == []:
            break
        group2.append(mIndex[i_optimal])

    e += 1

end = time.time()
print("Quantity of each product in each machine:")
print(qim)

print("Expected total profit:")
print(totalRevenue)

print("Program runtime:")
print(str(end - start) + 's')
