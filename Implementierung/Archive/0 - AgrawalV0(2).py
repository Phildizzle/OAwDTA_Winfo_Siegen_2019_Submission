# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 09:02:20 2018

@author: Johannes
"""

######################################
#Agrawal et al. Implementation
#Version 1.0 
#July 20, 2018
######################################

#To Do: Implementierung des Omega, Über alle Permutationen laufen lassen, 
#Prototyp mit eigener Eingabe eines jeden Kunden, Check wieso dynamic < one-time
#Gurobi optimizer als Funktion implementieren => Übersichtlicher

#Problem mit b und b_con: https://stackoverflow.com/questions/40382487/copy-a-list-of-list-by-value-and-not-reference/40382592

#Wieso bei kleinen eps hinten raus keine Annahmen mehr im one-time Fall? Keine Kapazitäten mehr?

######################################

#Preparation
print("\033[H\033[J")   #Clear Console
import numpy as np
import scipy as sc
from scipy.optimize import linprog
from gurobipy import *
import math

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
#Later: retrieve all permutations of p via from sympy.utilities.iterables import multiset_permutations

######################################
#Calculate fractional ex-post optimum
bnd = (0,1)

#Built-in Python function - not needed, use Gurobi
#res = linprog(c = -p, A_ub = a, b_ub = b, bounds = bnd, options = {"disp": True})
#print(res)

#Gurobi
m_ex = Model("Model_Ex")
customers = range(n)
categories = range(m) 


v = m_ex.addVars(customers, name = "decision", lb = 0, ub = 1)  #Decision variables and variable bounds
m_ex.addConstrs(    
       quicksum(a[i][j]*v[j] for j in range(n)) <= b[i] for i in range(m)   #Capacity constraint
       )
m_ex.setObjective(quicksum(p[j]*v[j] for j in range(n)), GRB.MAXIMIZE)      #Objective Function

#Method = 0
res_ex = m_ex.optimize()

status = m_ex.status
if status == GRB.Status.UNBOUNDED:
    print('The model cannot be solved because it is unbounded')
if status == GRB.Status.OPTIMAL:
    print('The optimal objective is %g' % m_ex.objVal)
if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)


obj_ex = m_ex.getObjective()
obj_ex = obj_ex.getValue()
print("Ex-post fractional optimum: " + str(round(obj_ex)))
input("Hit enter to continue ")
sol_ex = np.zeros(n, dtype = float)
i = 0
for v in m_ex.getVars():
    sol_ex[i] = round(v.X,2)
    i = i + 1
print("Ex-post fractional solution: " + str(sol_ex))
input("Hit enter to continue ")

#Calculate ex-post integral optimum
m_ex_i = Model("Model_Ex_i")
customers = range(n)
categories = range(m)


v = m_ex_i.addVars(customers, name = "decision", vtype=GRB.BINARY)  #Decision variables and variable bounds
m_ex_i.addConstrs(    
       quicksum(a[i][j]*v[j] for j in range(n)) <= b[i] for i in range(m)   #Capacity constraint
       )
m_ex_i.setObjective(quicksum(p[j]*v[j] for j in range(n)), GRB.MAXIMIZE)      #Objective Function

res_ex_i = m_ex_i.optimize()

status = m_ex_i.status
if status == GRB.Status.UNBOUNDED:
    print('The model cannot be solved because it is unbounded')
if status == GRB.Status.OPTIMAL:
    print('The optimal objective is %g' % m_ex_i.objVal)
if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)


obj_ex_i = m_ex_i.getObjective()
obj_ex_i = obj_ex_i.getValue()
print("Ex-post fractional optimum: " + str(round(obj_ex)))
print("Ex-post integral optimum: " + str(round(obj_ex_i)))
input("Hit enter to continue ")
sol_ex_i = np.zeros(n, dtype = float)
i = 0
for v in m_ex_i.getVars():
    sol_ex_i[i] = round(v.X,2)
    i = i + 1
print("Ex-post integral solution: " + str(sol_ex_i))
input("Hit enter to continue ")

######################################
#One-time Learning Algorithm
rev_one = 0     #Total revenue

#Check right-hand side condition
if min(b_con) >= (6 * float(m) * math.log(float(n)/eps)) / (pow(eps, 3)):
    input("Right-hand-side condition for one-time learning: TRUE")
else:
    input("Right-hand-side condition for one-time learning: FALSE")

#Step (i)
s = math.ceil(n*eps)
sol_one = np.zeros(n, dtype = np.int)   #Solution vector
for j in range(s):
    sol_one[j] = 0

#Gurobi
m_one = Model("Model_One")
customers = range(s)
categories = range(m)


v = m_one.addVars(customers, name = "decision", lb = 0, ub = 1)  #Decision variables and variable bounds
m_one.addConstrs(    
       (quicksum(a[i][j]*v[j] for j in range(s)) <= b_con[i]*(1-eps)*(s/n) for i in range(m)),   #Capacity constraint
       name = "c")
m_one.setObjective(quicksum(p[j]*v[j] for j in range(s)), GRB.MAXIMIZE)      #Objective Function
res_one = m_one.optimize()

status = m_one.status
if status == GRB.Status.UNBOUNDED:
    print('The model cannot be solved because it is unbounded')
if status == GRB.Status.OPTIMAL:
    print('The optimal objective is %g' % m_one.objVal)
if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)

sp_one = np.asarray(m_one.getAttr("Pi", m_one.getConstrs()))    #Retrieve shadow prices
sp_one_trans = sp_one.transpose()

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
        if a[i][j] * x_hat > b_con[i] - total_con[i]: 
            indicator = 0   #Change indicator if any constraint would be violated
    if indicator == 1:
        sol_one[j] = x_hat  #If feasbible, use x_hat 
        rev_one = rev_one + (p[j] * sol_one[j])   #Add revenue
        #for i in range(m):
        #    b[i] = b[i] - a[i][j] * sol_one[j]  #Adjust capacity
        
    else:
        sol_one[j] = 0

obj_one = rev_one
print("Ex-post fractional optimum: " + str(round(obj_ex)))
print("Ex-post integral optimum: " + str(round(obj_ex_i)))
print("One-time learning result: " + str(round(obj_one)))
input("Hit enter to continue ")
print("One-time learning solution: " + str(sol_one))
input("Hit enter to continue ")

######################################
#Dynamic pricing algorithm      
rev_dyn = 0     #Total revenue
l = 0
l_old = 0
#b = b_con[:]       #Reset capacity

#Check right-hand side condition
if min(b_con) >= (10 * float(m) * math.log(float(n)/eps)) / (pow(eps,2)):
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
        m_dyn = Model("Model_Dyn")
        customers = range(l)
        categories = range(m)
        
        
        v = m_dyn.addVars(customers, name = "decision", lb = 0, ub = 1)  #Decision variables and variable bounds
        m_dyn.addConstrs(    
               (quicksum(a[i][j]*v[j] for j in range(l)) <= b_con[i]*(l/n)*(1-eps*math.pow(n/l,0.5)) for i in range(m)),   #Capacity constraint
               name = "c")
        m_dyn.setObjective(quicksum(p[j]*v[j] for j in range(l)), GRB.MAXIMIZE)      #Objective Function
        res_dyn = m_dyn.optimize()
        
        status = m_dyn.status
        if status == GRB.Status.UNBOUNDED:
            print('The model cannot be solved because it is unbounded')
        if status == GRB.Status.OPTIMAL:
            print('The optimal objective is %g' % m_dyn.objVal)
        if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
            print('Optimization was stopped with status %d' % status)
        
        sp_dyn = np.asarray(m_dyn.getAttr("Pi", m_dyn.getConstrs()))    #Retrieve shadow prices
        sp_dyn_trans = sp_dyn.transpose()
         
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
        if a[i][t] * x_hat > b_con[i] - total_con[i]: 
            indicator = 0   #Change indicator if any constraint would be violated
    if indicator == 1:
        sol_dyn[t] = x_hat  #If feasbible, use x_hat 
        rev_dyn = rev_dyn + (p[t] * sol_dyn[t])   #Add revenue
        #for i in range(m):
        #    b[i] = b[i] - a[i][t] * sol_dyn[t]  #Adjust capacity
        
    else:
        sol_dyn[t] = 0

obj_dyn = rev_dyn
print("Ex-post fractional optimum: " + str(round(obj_ex)))
print("Ex-post integral optimum: " + str(round(obj_ex_i)))
print("One-time learning result: " + str(round(obj_one)))
print("Dynamic learning result: " + str(round(obj_dyn)))
input("Hit enter to continue ")
print("Dynamic learning solution: " + str(sol_dyn))
input("Hit enter to continue ")

print("Done")