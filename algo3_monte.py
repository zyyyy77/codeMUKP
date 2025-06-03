from gurobipy import *
import time
import numpy as np
import pandas as pd
import itertools


demandData = pd.read_excel("data.xlsx", sheet_name='demandData')
w = demandData['profit']
d = demandData['demand']

I = len(w)
MM = [4, 6]
M = sum(MM)
K1 = 60
K2 = 40
KK = []
for m in range(0, MM[0]):
    KK.append(K1)
for m in range(MM[0], M):
    KK.append(K2)

machine_r = [0.85, 0.85, 0.85, 0.85, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
print(machine_r)

ttlist = list(itertools.product([1, 0], repeat=M))

probabilities = []
for combo in ttlist:
    prob = 1
    for i, value in enumerate(combo):
        if value == 1:
            prob *= machine_r[i]
        else:
            prob *= (1 - machine_r[i])
    probabilities.append(prob)

def gurobi_solve(tlist, probability, num):
    model = Model('myModel')

    q = model.addVars(I, M, vtype=GRB.INTEGER, name='q')
    s = model.addVars(num, I, vtype=GRB.CONTINUOUS)

    model.update()

    model.setObjective(
        quicksum(w[i] * probability[tlist.index(t)] * s[tlist.index(t), i] for t in tlist for i in range(I)),
        GRB.MAXIMIZE)

    model.addConstrs((q.sum('*', m) == KK[m] for m in range(M)), 'yueshu1_')
    model.addConstrs(
        (s[tlist.index(t), i] <= quicksum((t[m] * q[i, m]) for m in range(M)) for t in tlist for i in range(I)),
        'yueshu2_')
    model.addConstrs((s[tlist.index(t), i] <= d[i] for t in tlist for i in range(I)), name='yueshu3_')

    model.optimize()

    if model.status == GRB.OPTIMAL:

        qVal = [q[i, m].x for i in range(I) for m in range(M)]
        qim_star = np.round(np.array(qVal).reshape(I, M))

    elif model.status == GRB.INF_OR_UNBD:
        print('Model is infeasible or unbounded')
        sys.exit(0)
    elif model.status == GRB.INFEASIBLE:
        print('Model is infeasible')
        sys.exit(0)
    elif model.status == GRB.UNBOUNDED:
        print('Model is unbounded')
        sys.exit(0)
    else:
        print('Optimization ended with status %d' % model.status)
        sys.exit(0)

    return model.objVal

start = time.time()

num_samples = 20
N = 500
obj_mt = np.zeros(N)

for j in range(N):
    samples = np.array([
        np.random.choice([1, 0], size=num_samples, p=[p, 1 - p])
        for p in machine_r
    ]).T
    tlist = [tuple(row) for row in samples]
    probability_mtkl = [1 / num_samples] * num_samples
    obj_mt[j] = gurobi_solve(tlist, probability_mtkl, num_samples)

df = pd.DataFrame(obj_mt, columns=[f'n'])

obj_mt1 = np.mean(obj_mt)
print(f"Total expected revenue with {num_samples} Monte Carlo samples: {obj_mt1},  time: {time.time() - start:.2f}s")

