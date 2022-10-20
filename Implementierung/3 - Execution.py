# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 09:02:20 2018

@author: Johannes, Philipp
"""

######################################
#Agrawal et al. Implementation
#Version 1.1 
#July 23, 2018
######################################
#This file is only for execution of Agrawal using predefined functions

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
print("Note: There is a ticket limit: A maximum of 5 tickets per category per request permitted")
n = 0
m = 0
max_a = 5   #Maximum number of tickets permitted per customer and category. CHANGE HERE IF NECESSARY AND IN ACCORDANCE WITH "FUNCTIONS"

n = int(input("Total number of customers n: "))
m = int(input("Total number of resources m: "))
b = np.zeros(m, dtype = np.int)
b_con = np.zeros(m, dtype = np.int)     #Copy of b which remains unchanged
i = 0
for i in range(m):
    str1 = "Total capacity of resource " + str(i+1) + ": "
    b[i] = int(input(str1))
eps = float(input("Fraction epsilon: "))
b_con = b[:]                            #Copy by values, do not copy reference

######################################
#Simulate incoming customers
sim = simulate(n, m)
p = sim[0]
a = sim[1]

######################################
#Choose random permutation
perm = permute(p, a)
p = perm[0]
a = perm[1]

######################################
#Calculate fractional ex-post optimum
ex = Gurobi(n, m, p, a, b_con)
#Calculate ex-post integral optimum
ex_i = Gurobi_integral(n, m, p, a, b_con)

######################################
#One-time Learning Algorithm
one = One_Time_Learning(n, m, p, a, b_con, eps)

######################################
#Dynamic pricing algorithm      
dyn = Dynamic_Learning(n, m, p, a, b_con, eps)

print("Done")
