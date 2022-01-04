# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 09:02:20 2018

@author: Johannes
"""

######################################
#Agrawal et al. Implementation
#Version 1.1 
#July 23, 2018
######################################
#Run several permutations

######################################

#Preparation
print("\033[H\033[J")   #Clear Console
import numpy as np
import scipy as sc
from scipy.optimize import linprog
from gurobipy import *
import math
from Functions import Gurobi, Gurobi_integral, simulate, permute, One_Time_Learning, Dynamic_Learning

#Initialization
print("Welcome to the Ticket Case Simulation")
print("Please initialize the problem instance in the following")
n = 0
m = 0
m = int(input("Total number of resources m: "))
b = np.zeros(m, dtype = np.int)
b_con = np.zeros(m, dtype = np.int)     #Copy of b which remains unchanged
i = 0
for i in range(m):
    str1 = "Total capacity of resource " + str(i+1) + ": "
    b[i] = int(input(str1))
b_con = b[:]                            #Copy by values, do not copy reference
eps = float(input("Fraction epsilon: "))

lb_one = math.floor(eps*math.exp((min(b_con)*math.pow(eps, 3))/(6*m)))
lb_dyn = math.floor(eps*math.exp((min(b_con)*math.pow(eps, 2))/(10*m)))

n = int(input("Total number of customers n. Please choose a number less or equal to " + str(lb_dyn) + " to satisfy the right-hand-side condition of the Dynamic Learning Algorithm: "))
count = int(input("Number of permutations to be tested: "))

#Check right-hand side conditions
ind_one = "not"
ind_dyn = "not"

if min(b) >= (6 * float(m) * math.log(float(n)/eps)) / (pow(eps, 3)):
        ind_one = ""
        
if min(b_con) >= (10 * float(m) * math.log(float(n)/eps)) / (pow(eps,2)):
        ind_dyn = ""

######################################
#Simulate incoming customers
sim = simulate(n, m)
p = sim[0]
a = sim[1]

k = 0
ex = 0
ex_i = 0
one = np.zeros(count)
dyn = np.zeros(count)

one_win = 0     #Number of times where One-Time Learning Algorithm performs better
dyn_win = 0     #Number of times where Dynamic Learning Algorithm performs better

ex = Gurobi(n, m, p, a, b_con)              #Ex-post optimum remains unchanged over permutations
ex_i = Gurobi_integral(n, m, p, a, b_con)   #Ex-post optimum remains unchanged over permutations

for k in range(count):
    
    ######################################
    #Choose random permutation
    perm = permute(p, a)
    p = perm[0]
    a = perm[1]
    
    ######################################
    #One-time Learning Algorithm
    one[k] = One_Time_Learning(n, m, p, a, b_con, eps)[0]
    
    ######################################
    #Dynamic pricing algorithm      
    dyn[k] = Dynamic_Learning(n, m, p, a, b_con, eps)[0]
    
    if dyn[k] > one[k]:
        dyn_win = dyn_win + 1
    if dyn[k] < one[k]:
        one_win = one_win + 1

avg_one = np.sum(one) / count
avg_dyn = np.sum(dyn) / count

perc_one = (avg_one / ex[0]) * 100
perc_dyn = (avg_dyn / ex[0]) * 100

print("The One-Time Learning Algorithm performed better in " + str(one_win) + " out of " + str(count) + " permutations.")
print("The Dynamic Learning Algorithm performed better in " + str(dyn_win) + " out of " + str(count) + " permutations.")
print("Equal performance in " + str(count - dyn_win - one_win) + " permutations.")
print("The ex-post optimum was: " + str(round(ex[0], 2)))
print("The average performance of the One-Time Learning Algorithm was: " + str(round(avg_one, 2)) + ". That is " + str(round(perc_one, 2)) + "% of the ex-post optimum.")
print("The average performance of the Dynamic Learning Algorithm was: " + str(round(avg_dyn, 2)) + ". That is " + str(round(perc_dyn, 2)) + "% of the ex-post optimum.")
print("The right-hand side condition was " + ind_one + " satisfied for the One-Time Learning Algorithm.")
print("The right-hand side condition was " + ind_dyn + " satisfied for the Dynamic Time Learning Algorithm.")
print("Done")
