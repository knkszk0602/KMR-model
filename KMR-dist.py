# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mc_tools import mc_compute_stationary, mc_sample_path
from discrete_rv import DiscreteRV
import random
import fractions
from scipy.stats import binom
from mpl_toolkits.axes_grid.axislines import SubplotZero

pay = np.array([[[4, 4], [0, 3]], [[3, 0], [2, 2]]])
n = 10
t = 10000
epsilon = {1:0.8, 2:0.6, 3:0.4, 4:0.2}
pay0 = np.array([[pay[0][0][0], pay[0][1][0]], [pay[1][0][0], pay[1][1][0]]])
psi = (1, 0)

def make_matrix(pay0, n, epsilon):
    P = np.zeros([n+1, n+1])
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
    return P

fig = plt.figure()
for val in epsilon:
    P = make_matrix(pay0, n, epsilon[val])
    stationary_dist = mc_compute_stationary(P)
    ax = plt.subplot(2,2,val)
    fig.add_subplot(ax)
    ax.hist(stationary_dist)
    plt.title('epsilon='+str(epsilon[val]))
    #plt.legend(str(epsilon[val]), loc = 1, prop={'size' : 8})
#plt.savefig('X_t_hist.png')
plt.show()
