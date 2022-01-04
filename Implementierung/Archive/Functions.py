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
#This document contains the functions accessed by other scripts

######################################

#Preparation
print("\033[H\033[J")   #Clear Console
import numpy as np
import scipy as sc
from scipy.optimize import linprog
from gurobipy import *
import math
import time
"""import cProfile
import re"""

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
    """if status == GRB.Status.UNBOUNDED:
        print('The model cannot be solved because it is unbounded')
    if status == GRB.Status.OPTIMAL:
        print('The optimal objective is %g' % model.objVal)
    if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % status)"""
    
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
    """if status == GRB.Status.UNBOUNDED:
        print('The model cannot be solved because it is unbounded')
    if status == GRB.Status.OPTIMAL:
        print('The optimal objective is %g' % model.objVal)
    if status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % status)"""
    
    obj = model.getObjective()
    obj = obj.getValue()
    
    sol = np.zeros(n, dtype = float)
    i = 0
    for v in model.getVars():
        sol[i] = round(v.X,2)
        i = i + 1
        
    #Note: Shadow prices "Pi" only available for continuous models: http://www.gurobi.com/documentation/8.0/refman/pi.html#attr:Pi
    
    return (obj, sol)

"""def clear():
    print("\033[H\033[J")   #Clear Console
    import numpy as np
    import scipy as sc
    from scipy.optimize import linprog
    from gurobipy import *
    import math
    n = 0
    m = 0
    b = np.zeros(m, dtype = np.int)
    b_con = np.zeros(m, dtype = np.int)     #Copy of b which remains unchanged
    eps = 0
    p = np.zeros(n, dtype = np.int)
    a = np.zeros((m, n), dtype = np.int)
    j = 0

def initialize():
    print("Welcome to the Ticket Case Simulation")
    print("Please initialize the problem instance in the following")
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
    return (n, m, b_con, eps)"""
    
def simulate(n, m, max_a):
    j = 0
    p = np.zeros(n, dtype = np.int)
    a = np.zeros((m, n), dtype = np.int)

    for j in range(n):
        p[j] = 0
        #p[j] = int(round(np.random.normal(60, 5)))      #Objective function coefficient
        for i in range(m):
            if m != 1:
                a[i][j] = int(round(max(np.random.normal(0.4*(1 + i/(2*(m-1))), 0.3*(i + 1)),0))) #max(np.random.normal(2*(1 + i/(2*(m-1))), 1*(m - i)),0) #np.random.poisson(lam = 0.4*(1 + i/(2*(m-1))))     #Capacity consumption of customer j for resource i: CHANGE HERE IF OTHER DISTRIBUTION DESIRED
            else: a[i][j] = int(round(max(np.random.normal(0.4, 0.5),0))) #max(np.random.normal(2, 1),0) #np.random.poisson(lam = 0.4)
            if a[i][j] > max_a:
                a[i][j] = max_a                             #Maximum of max_a tickets per category allowed: CHANGE HERE IF OTHER LIMIT DESIRED
            if i == m-1 and a[i][j] == 0 and sum(a[i][j] for i in range(m-1)) == 0:
                a[i][j] = 1     #Every customer must request at least one ticket of worst category
            if m != 1:
                p[j] = max(p[j] + a[i][j]*int(round(np.random.normal(100*(1 - i/(2*(m-1))), 10*(m - i)))),0)  #5  #Objective function coefficient: CHANGE HERE IF OTHER DISTRIBUTION DESIRED
            else: p[j] = max(p[j] + a[i][j]*int(round(np.random.normal(100, 10))),0) #5
    return (p, a)

def permute(p, a):
    n = len(p)
    m = len(a)
    perm = np.random.permutation(n)
    p1 = np.zeros(n, dtype = np.int)
    a1 = np.zeros((m, n), dtype = np.int)
    for j in range(n):
        p1[j] = p[perm[j]]
        for i in range(m):
            a1[i][j] = a[i][perm[j]]
    p = p1
    a = a1
    return (p, a)
    #Later: retrieve all permutations of p via "from sympy.utilities.iterables import multiset_permutations"

def One_Time_Learning(n, m, p, a, b, eps):
    start_time = time.time()
    rev_one = 0     #Total revenue
    one_time_cust = []  #Time one-time learning algorithm needs for each customer
    resource_consumption = np.zeros(m, dtype = np.int)  #Resources consumed through algorithm
    total_con = np.zeros(m, dtype = np.int) #Total resource consumption so far
    
    #Check right-hand side condition
    if min(b) >= (6 * float(m) * math.log(float(n)/eps)) / (pow(eps, 3)):
        print("Right-hand-side condition for one-time learning: TRUE")
    else:
        print("Right-hand-side condition for one-time learning: FALSE")
    
    #Step (i)
    s = math.ceil(n * eps)
    sol_one = np.zeros(n, dtype = np.int)   #Solution vector
    for j in range(s):
        cust_start_time = time.time()
        sol_one[j] = 0
        one_time_cust.append(time.time()-cust_start_time)
    
    #Gurobi
    one = Gurobi(s, m, p[0:s], [i[0:s] for i in a], b*(1-eps)*(s/n))
    sp_one_trans = one[2]   #Shadow prices
    
    #Algorithm
    for j in range(s, n): #Remember: indices start at 0, so we need indices from s to n-1 in order to get customers s to n
        cust_start_time = time.time()
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
        for i in range(m):
            total_con[i] = total_con[i] + (a[i][j-1] * sol_one[j-1]) #Add consumption of last customer to total consumption
        indicator = 1   #Indicates whether constraint would be violated in case of acceptance
        for i in range(m):
            if a[i][j] * x_hat > b[i] - total_con[i]: 
                indicator = 0   #Change indicator if any constraint would be violated
        if indicator == 1:
            sol_one[j] = x_hat  #If feasbible, use x_hat 
            rev_one = rev_one + (p[j] * sol_one[j])   #Add revenue
            for i in range(m):
                resource_consumption[i] = resource_consumption[i] + a[i][j]*sol_one[j] #Add resource consumption
            
        else:
            sol_one[j] = 0
        
        one_time_cust.append(time.time()-cust_start_time)

    obj_one = rev_one
    
    tot_time = time.time() - start_time
    
    return (obj_one, sol_one, tot_time, one_time_cust, sp_one_trans, resource_consumption)

def Dynamic_Learning(n, m, p, a, b, eps):
    start_time = time.time()
    rev_dyn = 0     #Total revenue
    l = 0
    l_old = 0
    dyn_time_cust = []  #Time dynamic learning algorithm needs for each customer
    sp_dyn_prices = []  #Array of arrays with shadow prices
    resource_consumption = np.zeros(m, dtype = np.int)  #Resources consumed through algorithm
    total_con = np.zeros(m, dtype = np.int) #Total resource consumption so far
    
    #Check right-hand side condition
    if min(b) >= (10 * float(m) * math.log(float(n)/eps)) / (pow(eps,2)):
        print("Right-hand-side condition for dynamic learning: TRUE")
    else:
        print("Right-hand-side condition for dynamic learning: FALSE")
    
    #Step (i)
    t0 = math.ceil(n * eps)
    sol_dyn = np.zeros(n, dtype = np.int)   #Solution vector
    t1 = 0
    for t1 in range(t0):
        cust_start_time = time.time()
        sol_dyn[t1] = 0
        dyn_time_cust.append(time.time()-cust_start_time)
        
    #Step (ii)
    for t in range(t0, n):
        cust_start_time = time.time()
        l_old = l
        r = 0
        while math.ceil(n * eps * math.pow(2,r)) < (t + 1):
            if math.ceil(n * eps * math.pow(2,r+1)) >= (t + 1):
                break
            else:
                r = r + 1
        l = math.ceil(n * eps * math.pow(2,r))
        if l != l_old and l != 0:
            #Gurobi
            dyn = Gurobi(l, m, p[0:l], [i[0:l] for i in a], b*(l/n)*(1-eps*math.pow(n/l,0.5)))
            sp_dyn_trans = dyn[2]
            sp_dyn_prices.append(sp_dyn_trans)
            
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
        for i in range(m):
            total_con[i] = total_con[i] + (a[i][t-1] * sol_dyn[t-1]) #Add consumption of last customer to total consumption
        indicator = 1   #Indicates whether constraint would be violated in case of acceptance
        for i in range(m):
            if a[i][t] * x_hat > b[i] - total_con[i]: 
                indicator = 0   #Change indicator if any constraint would be violated
        if indicator == 1:
            sol_dyn[t] = x_hat  #If feasbible, use x_hat 
            rev_dyn = rev_dyn + (p[t] * sol_dyn[t])   #Add revenue
            for i in range(m):
                resource_consumption[i] = resource_consumption[i] + a[i][t]*sol_dyn[t] #Add resource consumption
           
        else:
            sol_dyn[t] = 0
        
        dyn_time_cust.append(time.time()-cust_start_time)
    
    obj_dyn = rev_dyn
    
    tot_time = time.time() - start_time
    
    return (obj_dyn, sol_dyn, tot_time, dyn_time_cust, sp_dyn_prices, resource_consumption)