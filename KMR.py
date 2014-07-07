# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mc_tools import mc_compute_stationary, mc_sample_path
from discrete_rv import DiscreteRV
import random
import fractions
from scipy.stats import binom

pay = np.array([[[4, 4], [0, 3]], [[3, 0], [2, 2]]])
n = 10
t = 10000
epsilon = 0.4
x0 = 0

pay0 = np.array([[pay[0][0][0], pay[0][1][0]], [pay[1][0][0], pay[1][1][0]]])
cur_x = x0
psi = (1, 0)
# d = DiscreteRV(psi)

P = np.zeros([n+1, n+1])  # ここから先はマルコフ連鎖の遷移行列Ｐの設定
for i in range(n+1):
    num = fractions.Fraction(i, n)
    pr = np.array([1-num, num])
    exp = np.dot(pay0, pr)
    if exp[0] > exp[1]:
        P[i][i] = num*epsilon*0.5+(1-num)*(1-epsilon*0.5)
    elif exp[0] < exp[1]:
        P[i][i] = num*(1-epsilon*0.5)+(1-num)*epsilon*0.5
    else:
        P[i][i] = 0.5
for i in range(1, n+1):
    num = fractions.Fraction(i, n)
    pr = np.array([1-num, num])
    exp = np.dot(pay0, pr)
    if exp[0] > exp[1]:
        P[i][i-1] = num*(1-epsilon*0.5)
    elif exp[0] < exp[1]:
        P[i][i-1] = num*epsilon*0.5
    else:
        P[i][i-1] = num*0.5
for i in range(n):
    num = fractions.Fraction(i, n)
    pr = np.array([1-num, num])
    exp = np.dot(pay0, pr)
    if exp[0] > exp[1]:
        P[i][i+1] = (1-num)*epsilon*0.5
    elif exp[0] < exp[1]:
        P[i][i+1] = (1-num)*(1-epsilon*0.5)
    else:
        P[i][i+1] = (1-num)*0.5

X = mc_sample_path(P, psi, t)

fig, ax = plt.subplots()
ax.plot(X)
plt.show()
