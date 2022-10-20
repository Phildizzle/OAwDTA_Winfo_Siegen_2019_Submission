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
#Changelog:
#   -   Defined Gurobi() and Gurobi_integral() functions
#   -   Defined One_Time_Learning() and Dynamic_Learning() functions

######################################

#Preparation
print("\033[H\033[J")   #Clear Console
import numpy as np
import scipy as sc
from scipy.optimize import linprog
from gurobipy import *
import math

def Gurobi(n, m, p, a, b):  #Fractional optimization
    model = Model("Model")
    customers = range(n)
    categories = range(m)
    v = model.addVars(customers, name = "decision", lb = 0, ub = 1)  #Decision variables and variable bounds
    model.addConstrs(    
       quicksum(a[i][j]*v[j] for j in range(n)) <= b[i] for i in range(m)   #Capacity constraint
       )
    model.setObjective(quicksum(p[j]*v[j] for j in range(n)), GRB.MAXIMIZE)      #Objective Function

    #Method = 0
    res = model.optimize()
    
    status = model.status
    if status == GRB.Status.UNBOUNDED:
        print('The model cannot be solved because it is unbounded')
    if status == GRB.Status.OPTIMAL:
        print('The optimal objective is %g' % model.objVal)
    if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % status)
    
    obj = model.getObjective()
    obj = obj.getValue()
    
    sol = np.zeros(n, dtype = float)
    i = 0
    for v in model.getVars():
        sol[i] = round(v.X,2)
        i = i + 1
        
    sp = np.asarray(model.getAttr("Pi", model.getConstrs()))    #Retrieve shadow prices
    sp = sp.transpose()
    
    return (obj, sol, sp)

def Gurobi_integral(n, m, p, a, b): #Integral optimization
    model = Model("Model")
    customers = range(n)
    categories = range(m)
    v = model.addVars(customers, name = "decision", vtype=GRB.BINARY)  #Decision variables and variable bounds
    model.addConstrs(    
       quicksum(a[i][j]*v[j] for j in range(n)) <= b[i] for i in range(m)   #Capacity constraint
       )
    model.setObjective(quicksum(p[j]*v[j] for j in range(n)), GRB.MAXIMIZE)      #Objective Function

    #Method = 0
    res = model.optimize()
    
    status = model.status
    if status == GRB.Status.UNBOUNDED:
        print('The model cannot be solved because it is unbounded')
    if status == GRB.Status.OPTIMAL:
        print('The optimal objective is %g' % model.objVal)
    if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % status)
    
    obj = model.getObjective()
    obj = obj.getValue()
    
    sol = np.zeros(n, dtype = float)
    i = 0
    for v in model.getVars():
        sol[i] = round(v.X,2)
        i = i + 1
        
    #Note: Shadow prices "Pi" only available for continuous models: http://www.gurobi.com/documentation/8.0/refman/pi.html#attr:Pi
    
    return (obj, sol)

#Initialization
print("Welcome to the Ticket Case Simulation")
print("Please initialize the problem instance in the following")
n = 0
m = 0
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
j = 0
p = np.zeros(n, dtype = np.int)
a = np.zeros((m, n), dtype = np.int)

for j in range(n):
    p[j] = int(round(np.random.normal(60, 5)))      #Objective function coefficient
    for i in range(m):
        a[i][j] = np.random.poisson(lam = 0.75)     #Capacity consumption of customer j for resource i

######################################
#Choose random permutation
perm = np.random.permutation(n)
p1 = np.zeros(n, dtype = np.int)
a1 = np.zeros((m, n), dtype = np.int)
for j in range(n):
    p1[j] = p[perm[j]]
    for i in range(m):
        a1[i][j] = a[i][perm[j]]
p = p1
a = a1
#Later: retrieve all permutations of p via "from sympy.utilities.iterables import multiset_permutations"

######################################
#Calculate fractional ex-post optimum
ex = Gurobi(n, m, p, a, b_con)
print("Ex-post fractional optimum: " + str(round(ex[0])))
input("Hit enter to continue ")
print("Ex-post fractional solution: " + str(ex[1]))
input("Hit enter to continue ")

#Calculate ex-post integral optimum
ex_i = Gurobi_integral(n, m, p, a, b_con)
print("Ex-post fractional optimum: " + str(round(ex[0])))
print("Ex-post integral optimum: " + str(round(ex_i[0])))
input("Hit enter to continue ")
print("Ex-post integral solution: " + str(ex_i[1]))
input("Hit enter to continue ")

######################################
#One-time Learning Algorithm
def One_Time_Learning(n, m, p, a, b, eps):
    rev_one = 0     #Total revenue
    
    #Check right-hand side condition
    if min(b) >= (6 * float(m) * math.log(float(n)/eps)) / (pow(eps, 3)):
        input("Right-hand-side condition for one-time learning: TRUE")
    else:
        input("Right-hand-side condition for one-time learning: FALSE")
    
    #Step (i)
    s = math.ceil(n*eps)
    sol_one = np.zeros(n, dtype = np.int)   #Solution vector
    for j in range(s):
        sol_one[j] = 0
    
    #Gurobi
    one = Gurobi(s, m, p, a, b*(1-eps)*(s/n))
    sp_one_trans = one[2]   #Shadow prices
    
    #Algorithm
    for j in range(s, n): #Remember: indices start at 0, so we need indices from s to n-1 in order to get customers s to n
        x_hat = 0
        
        #Equation 12
        res_con = np.zeros(m, dtype = np.int)           #Resource consumption vector of customer j
        for i in range(m):
            res_con[i] = a[i][j]                        #Retrieve resource consumption of customer j
        if p[j] <= np.dot(sp_one_trans, res_con):       #Matrix product; Check if bid is less than resource consumption weighted with dual price
            x_hat = 0
        else:
            x_hat = 1
        
        #Step (ii)
        k = 0
        total_con = np.zeros(m, dtype = np.int) #Total resource consumption so far
        for k in range(s, j):   #Total consumption only requires customers from s to j (first s customers will always be zero)
            for i in range(m):
                total_con[i] = total_con[i] + (a[i][k] * sol_one[k])
        indicator = 1   #Indicates whether constraint would be violated in case of acceptance
        for i in range(m):
            if a[i][j] * x_hat > b[i] - total_con[i]: 
                indicator = 0   #Change indicator if any constraint would be violated
        if indicator == 1:
            sol_one[j] = x_hat  #If feasbible, use x_hat 
            rev_one = rev_one + (p[j] * sol_one[j])   #Add revenue
            #for i in range(m):
            #    b[i] = b[i] - a[i][j] * sol_one[j]  #Adjust capacity
            
        else:
            sol_one[j] = 0
    
    obj_one = rev_one
    return (obj_one, sol_one)

one = One_Time_Learning(n, m, p, a, b_con, eps)
print("Ex-post fractional optimum: " + str(round(ex[0])))
print("Ex-post integral optimum: " + str(round(ex_i[0])))
print("One-time learning result: " + str(round(one[0])))
input("Hit enter to continue ")
print("One-time learning solution: " + str(one[1]))
input("Hit enter to continue ")

######################################
#Dynamic pricing algorithm      
def Dynamic_Learning(n, m, p, a, b, eps):
    rev_dyn = 0     #Total revenue
    l = 0
    l_old = 0
    #b = b_con[:]       #Reset capacity
    
    #Check right-hand side condition
    if min(b) >= (10 * float(m) * math.log(float(n)/eps)) / (pow(eps,2)):
        input("Right-hand-side condition for dynamic learning: TRUE")
    else:
        input("Right-hand-side condition for dynamic learning: FALSE")
    
    #Step (i)
    t0 = math.ceil(n * eps)
    sol_dyn = np.zeros(n, dtype = np.int)   #Solution vector
    t1 = 0
    for t1 in range(t0):
        sol_dyn[t1] = 0
        
    #Step (ii)
    for t in range(t0, n):
        l_old = l
        r = 0
        while math.ceil(n * eps * math.pow(2,r)) < t + 1:
            if math.ceil(n * eps * math.pow(2,r+1)) >= t + 1:
                break
            r = r + 1
        l = math.ceil(n * eps * math.pow(2,r))
        if l != l_old and l != 0:
            #Gurobi
            dyn = Gurobi(l, m, p, a, b*(l/n)*(1-eps*math.pow(n/l,0.5)))
            sp_dyn_trans = dyn[2]
            
        x_hat = 0 
        # (a)
        res_con = np.zeros(m, dtype = np.int) #Resource consumption vector of customer j
        for i in range(m):
            res_con[i] = a[i][t]    #Retrieve resource consumption of customer j
        if p[t] <= np.dot(sp_dyn_trans, res_con): #Check if bid is less than resource consumption weighted with dual price
            x_hat = 0
        else:
            x_hat = 1
        
        # (b)
        k = 0
        total_con = np.zeros(m, dtype = np.int) #Total resource consumption so far
        for k in range(t0, t):   #Total consumption only requires customers from t0 to j (first t0 customers will always be zero)
            for i in range(m):
                total_con[i] = total_con[i] + (a[i][k] * sol_dyn[k])
        indicator = 1   #Indicates whether constraint would be violated in case of acceptance
        for i in range(m):
            if a[i][t] * x_hat > b[i] - total_con[i]: 
                indicator = 0   #Change indicator if any constraint would be violated
        if indicator == 1:
            sol_dyn[t] = x_hat  #If feasbible, use x_hat 
            rev_dyn = rev_dyn + (p[t] * sol_dyn[t])   #Add revenue
            #for i in range(m):
            #    b[i] = b[i] - a[i][t] * sol_dyn[t]  #Adjust capacity
            
        else:
            sol_dyn[t] = 0
    
    obj_dyn = rev_dyn
    return (obj_dyn, sol_dyn)

dyn = Dynamic_Learning(n, m, p, a, b_con, eps)    
print("Ex-post fractional optimum: " + str(round(ex[0])))
print("Ex-post integral optimum: " + str(round(ex_i[0])))
print("One-time learning result: " + str(round(one[0])))
print("Dynamic learning result: " + str(round(dyn[0])))
input("Hit enter to continue ")
print("Dynamic learning solution: " + str(dyn[1]))
input("Hit enter to continue ")

print("Done")
