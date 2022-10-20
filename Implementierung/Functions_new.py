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
#This document contains the functions accessed by Run_Simulation and AutoSim

######################################
#Readme:
#Contains functions:
# 
#Main functions
#initialize (28), Gurobi (168), Gurobi_integral (195), simulate (221), permute (242), One_time_Learning (257), Dynamic_Learning (322)
#Benchmarks:
#Greedy (402), Interval_Learner (420), One_Time_Relaxed (490), Dynamic_Relaxed (550), WTP_Learner (622), Amazon_Learner (696)
# 



#Import
print("\033[H\033[J")   #Clear Console
import numpy as np
from gurobipy import *
import math
import time
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def initialize():
    """ Initializes the ORA problem with a set of parameters n, m, max_a, b, 
    b_con, eps, numb, count, bench, ind_one, ind_dyn. Returns the same parameters."""
    
    n = 0
    m = 0
    max_a = 4   #Maximum number of tickets permitted per customer and category. CHANGE HERE IF NECESSARY AND IN ACCORDANCE WITH "FUNCTIONS"
    numb = 0
    count = 0
    bench = 0
    ind_exit = 0
    ind_exit = 0
    
    while True:
        try:
            m = input("Total number of resources m: ")
            if m == "Exit" or m == "exit":
                ind_exit = 1
                break
            m = int(m)
            if m > 0:
                break
        except:
            print("\nPlease enter an integer number")
    if ind_exit == 1:
        print("Execution will be stopped")
        sys.exit(0)
    
    b = np.zeros(m, dtype = np.int)
    b_con = np.zeros(m, dtype = np.int)     #Copy of b which remains unchanged
    i = 0
    
    for i in range(m):
        while True:
            try:
                str1 = "Total capacity of resource " + str(i+1) + ": "
                inp = input(str1)
                if inp == "Exit" or inp == "exit":
                    ind_exit = 1
                    break
                b[i] = int(inp)
                if b[i] >= 0:
                    break
            except:
               print("\nPlease enter an integer number") 
        if ind_exit == 1:
            print("Execution will be stopped")
            sys.exit(0)
            
    b_con = b[:]                            #Copy by values, do not copy reference
    
    while True:
        try:
            eps = input("Fraction epsilon: ")
            if eps == "Exit" or eps == "exit":
                    ind_exit = 1
                    break
            eps = float(eps)
            if eps > 0 and eps <= 1:
                break
        except:
            print("\nPlease enter a number between 0 and 1: ")
    if ind_exit == 1:
        print("Execution will be stopped")
        sys.exit(0) 
    
    lb_one = math.floor(eps*math.exp((min(b_con/max_a)*math.pow(eps, 3))/(6*m)))    #Divide b_con by max_a since this reflects the maximum number of tickets permitted per customer and category. This requires an adjustment of right-hand side condition (see Remark 1.1 in Agrawal et al.)
    lb_dyn = math.floor(eps*math.exp((min(b_con/max_a)*math.pow(eps, 2))/(10*m)))   #Divide b_con by max_a since max_a reflects the maximum number of tickets permitted per customer and category. This requires an adjustment of right-hand side condition (see Remark 1.1 in Agrawal et al.)
    
    while True:
        try:
            n = input("Total number of customers n. Please choose a number less or equal to " + str(lb_dyn) + " to satisfy the right-hand-side condition of the Dynamic Learning Algorithm: ")
            if n == "Exit" or n == "exit":
                ind_exit = 1
                break
            n = int(n)
            if n > 0:
                break
        except:
            print("\nPlease enter an integer number")
    if ind_exit == 1:
        print("Execution will be stopped")
        sys.exit(0)
    
    while True:
        try:
            numb = input("Number of simulations to be drawn: ")
            if numb == "Exit" or numb == "exit":
                ind_exit = 1
                break
            numb = int(numb)
            if numb > 0:
                break
        except:
            print("\nPlease enter an integer number")
    if ind_exit == 1:
        print("Execution will be stopped")
        sys.exit(0)
    
    while True:
        try:
            count = input("Number of permutations to be tested for each simulation run: ")
            if count == "Exit" or count == "exit":
                ind_exit = 1
                break
            count = int(count)
            if count > 0:
                break
        except:
            print("\nPlease enter an integer number")
    if ind_exit == 1:
        print("Execution will be stopped")
        sys.exit(0)
    
    bench = 0
    
    while True:
        try:
            ans = input("Would you like to execute all benchmarks as well? ")
            if ans == "Exit" or ans == "exit":
                ind_exit = 1
                break
            if ans == "Yes" or ans == "yes" or ans == "y" or ans == 1:
                bench = 1
                break
            if ans == "No" or ans == "no" or ans == "n" or ans == 0:
                bench = 0
                break
        except:
            print("\nPlease enter \"yes\" or \"no\"")
    if ind_exit == 1:
        print("Execution will be stopped")
        sys.exit(0)
    
    #Check right-hand side conditions
    ind_one = "not"
    ind_dyn = "not"
    
    if min(b_con/max_a) >= (6 * float(m) * math.log(float(n)/eps)) / (pow(eps, 3)):
            ind_one = ""
            
    if min(b_con/max_a) >= (10 * float(m) * math.log(float(n)/eps)) / (pow(eps,2)):
            ind_dyn = ""
    return n, m, max_a, b, b_con, eps, numb, count, bench, ind_one, ind_dyn

def Gurobi(n, m, p, a, b):
    """ Calculates the fractional optimum by calling Gurobi. Input parameters 
    are n, m, p, a, b. Return parameters are obj, sol, sp."""
    
    model = Model("Model")
    customers = range(n)
    v = model.addVars(customers, name = "decision", lb = 0, ub = 1)         #Decision variables and variable bounds
    model.addConstrs(    
       quicksum(a[i][j]*v[j] for j in range(n)) <= b[i] for i in range(m)   #Capacity constraint
       )
    model.setObjective(quicksum(p[j]*v[j] for j in range(n)), GRB.MAXIMIZE) #Objective Function
    res = model.optimize()
    obj = model.getObjective()
    obj = obj.getValue()
    
    sol = np.zeros(n, dtype = float)
    i = 0
    for v in model.getVars():
        sol[i] = round(v.X,2)
        i = i + 1
        
    sp = np.asarray(model.getAttr("Pi", model.getConstrs()))    #Retrieve shadow prices
    sp = sp.transpose()
    
    return obj, sol, sp

def Gurobi_integral(n, m, p, a, b):
    """ Calculates the integer optimum by calling Gurobi. Input parameters 
    re n, m, p, a, b. Return parameters are obj, sol."""
    
    model = Model("Model")
    customers = range(n)
    v = model.addVars(customers, name = "decision", vtype=GRB.BINARY)       #Decision variables and variable bounds
    model.addConstrs(    
       quicksum(a[i][j]*v[j] for j in range(n)) <= b[i] for i in range(m)   #Capacity constraint
       )
    model.setObjective(quicksum(p[j]*v[j] for j in range(n)), GRB.MAXIMIZE) #Objective Function
    res = model.optimize()
    obj = model.getObjective()
    obj = obj.getValue()
    
    sol = np.zeros(n, dtype = float)
    i = 0
    for v in model.getVars():
        sol[i] = round(v.X,2)
        i = i + 1   
        
    return obj, sol

def simulate(n, m, max_a):
    """ Simulates willingnesses to pay and demands of incoming customers. 
    Takes in n, m, max_a. Returns simulated p's and a's."""
    
    j = 0
    p = np.zeros(n, dtype = np.int)
    a = np.zeros((m, n), dtype = np.int)

    for j in range(n):
        p[j] = 0
        for i in range(m):
            if m != 1:
                a[i][j] = int(round(max(np.random.normal(0.4*(1 + i/(2*(m-1))), 0.3*(i + 1)),0))) #Other specification: max(np.random.normal(2*(1 + i/(2*(m-1))), 1*(m - i)),0) #np.random.poisson(lam = 0.4*(1 + i/(2*(m-1))))     #Capacity consumption of customer j for resource i: CHANGE HERE IF OTHER DISTRIBUTION DESIRED
            else: a[i][j] = int(round(max(np.random.normal(0.4, 0.5),0))) #Other specification: max(np.random.normal(2, 1),0) #np.random.poisson(lam = 0.4)
            if a[i][j] > max_a:
                a[i][j] = max_a                            
            if i == m-1 and a[i][j] == 0 and sum(a[i][j] for i in range(m-1)) == 0:
                a[i][j] = 1     #Every customer must request at least one ticket of worst category
            if m != 1:
                p[j] = max(p[j] + a[i][j]*int(round(np.random.normal(100*(1 - i/(2*(m-1))), 10*(m - i)))),0)  #5  #Objective function coefficient: CHANGE HERE IF OTHER DISTRIBUTION DESIRED
            else: p[j] = max(p[j] + a[i][j]*int(round(np.random.normal(100, 10))),0) #5
    
    return p, a

def permute(p, a):
    """ Permutes the WTP and demands of incoming customers. 
    Takes in p and a and returns both as well."""
    
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
    
    return p, a
    #Later: retrieve all permutations of p via "from sympy.utilities.iterables import multiset_permutations"

def One_Time_Learning(n, m, p, a, b, eps):
    """ Calculates the One Time Learner by calculating the solution of the dual 
    prices at eps*n and subsequently answers customer requests with these
    threshold prices. Takes as input parameters n, m, p, a, b, eps and returns 
    obj_one, sol_one, tot_time, one_time_cust, sp_one_trans, resource_consumption."""
    
    start_time = time.time()
    rev_one = 0                                         #Total revenue
    one_time_cust = []                                  #Time one-time learning algorithm needs for each customer
    resource_consumption = np.zeros(m, dtype = np.int)  #Resources consumed through algorithm
    total_con = np.zeros(m, dtype = np.int)             #Total resource consumption so far
    
    #Check right-hand side condition
    if min(b) >= (6 * float(m) * math.log(float(n)/eps)) / (pow(eps, 3)):
        print("Right-hand-side condition for one-time learning: TRUE")
    else:
        print("Right-hand-side condition for one-time learning: FALSE")
    
    #Step (i)
    s = math.ceil(n * eps)
    sol_one = np.zeros(n, dtype = np.int)               #Solution vector
    
    for j in range(s):
        cust_start_time = time.time()
        sol_one[j] = 0
        one_time_cust.append(time.time()-cust_start_time)
    
    #Gurobi call
    one = Gurobi(s, m, p[0:s], [i[0:s] for i in a], b*(1-eps)*(s/n))
    sp_one_trans = one[2]   #Shadow prices
    
    #Algorithm
    for j in range(s, n):                               #Note to ourselves: indices start at 0, so we need indices from s to n-1 in order to get customers s to n
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
            total_con[i] = total_con[i] + (a[i][j-1] * sol_one[j-1])#Add consumption of last customer to total consumption
        if x_hat == 1:
            indicator = 1                                           #Indicates whether constraint would be violated in case of acceptance
            for i in range(m):
                if a[i][j] * x_hat > b[i] - total_con[i]: 
                    indicator = 0                                   #Change indicator if any constraint would be violated
            if indicator == 1:
                sol_one[j] = x_hat                                  #If feasbible, use x_hat 
                rev_one = rev_one + (p[j] * sol_one[j])             #Add revenue
                for i in range(m):
                    resource_consumption[i] = resource_consumption[i] + a[i][j]*sol_one[j] #Add resource consumption        
            else:
                sol_one[j] = 0
        else:
            sol_one[j] = 0
            
        one_time_cust.append(time.time()-cust_start_time)

    obj_one = rev_one
    tot_time = time.time() - start_time
    
    return obj_one, sol_one, tot_time, one_time_cust, sp_one_trans, resource_consumption

def Dynamic_Learning(n, m, p, a, b, eps):
    """ Calculates the Dynamic Learner by calculating the solution of the dual 
    prices at l=eps*n*2^r and subsequently answers customer requests with these
    threshold prices for each period s.t. l < t. Takes as input parameters 
    n, m, p, a, b, eps and returns obj_dyn, sol_dyn, tot_time, dyn_time_cust, 
    sp_dyn_prices, resource_consumption."""
    
    start_time = time.time()
    rev_dyn = 0                                         #Total revenue
    l = 0
    l_old = 0
    dyn_time_cust = []                                  #Time dynamic learning algorithm needs for each customer
    sp_dyn_prices = []                                  #Array of arrays with shadow prices
    resource_consumption = np.zeros(m, dtype = np.int)  #Resources consumed through algorithm
    total_con = np.zeros(m, dtype = np.int)             #Total resource consumption so far
    
    #Check right-hand side condition
    if min(b) >= (10 * float(m) * math.log(float(n)/eps)) / (pow(eps,2)):
        print("Right-hand-side condition for dynamic learning: TRUE")
    else:
        print("Right-hand-side condition for dynamic learning: FALSE")
    
    #Step (i)
    t0 = math.ceil(n * eps)
    sol_dyn = np.zeros(n, dtype = np.int)               #Solution vector
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
        #(a)
        res_con = np.zeros(m, dtype = np.int)           #Resource consumption vector of customer j
        for i in range(m):
            res_con[i] = a[i][t]                        #Retrieve resource consumption of customer j
        if p[t] <= np.dot(sp_dyn_trans, res_con):       #Check if bid is less than resource consumption weighted with dual price
            x_hat = 0
        else:
            x_hat = 1
        
        #(b)
        for i in range(m):
            total_con[i] = total_con[i] + (a[i][t-1] * sol_dyn[t-1]) #Add consumption of last customer to total consumption
        if x_hat == 1:
            indicator = 1                               #Indicates whether constraint would be violated in case of acceptance
            for i in range(m):
                if a[i][t] * x_hat > b[i] - total_con[i]: 
                    indicator = 0                       #Change indicator if any constraint would be violated
            if indicator == 1:
                sol_dyn[t] = x_hat                      #If feasbible, use x_hat 
                rev_dyn = rev_dyn + (p[t] * sol_dyn[t]) #Add revenue
                for i in range(m):
                    resource_consumption[i] = resource_consumption[i] + a[i][t]*sol_dyn[t] #Add resource consumption       
            else:
                sol_dyn[t] = 0
        else:
            sol_dyn[t] = 0
        
        dyn_time_cust.append(time.time()-cust_start_time)
    
    obj_dyn = rev_dyn
    tot_time = time.time() - start_time
    
    return obj_dyn, sol_dyn, tot_time, dyn_time_cust, sp_dyn_prices, resource_consumption

def Greedy(n, m, p, a, b):
    """ Calculates a greedy algorithm which accepts every incoming inquiry as 
    long as no capacity constraint is violated. Takes as input n, m, p, a, b, 
    eps and returns rev, x, used_cap. Starts accepting customers at t=0."""
    
    used_cap = np.zeros(m, dtype = np.int)
    x = np.zeros(n, dtype = np.int)
    rev = 0

    for j in range(n):          #Do not exclude calibration period! Due to reasons of fairness!
        indicator = 0           #Check whether resource limit would be exceeded
        for i in range(m):
            if used_cap[i] + a[i][j] > b[i]:    
                indicator = 1   #One resource capacity is violated
                break
        if indicator == 0:      #If resource capacity is not violated, then accept offer
            rev = rev + p[j]
            for i in range(m):
                used_cap[i] = used_cap[i] + a[i][j]
            x[j] = 1
            
    return rev, x, used_cap

def Interval_Learner(n, m, p, a, b):
    """ Calculates an interval learner which learns new threshold prices every 
    10% of n. Takes as input n, m, p, a, b and returns obj_dyn, sol_dyn, 
    tot_time, dyn_time_cust, sp_dyn_prices, resource_consumption."""
    
    start_time = time.time()
    rev_dyn = 0                             #Total revenue
    l = 0
    l_old = 0
    dyn_time_cust = []                      #Time dynamic learning algorithm needs for each customer
    sp_dyn_prices = []                      #Nested list of shadow prices
    resource_consumption = np.zeros(m, dtype = np.int)  #Resources consumed through algorithm
    total_con = np.zeros(m, dtype = np.int) #Total resource consumption so far
    
    #Step (i)
    t0 = math.ceil(n * 0.1)
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
        r = 1
        while math.ceil(n * 0.1 * r) < (t + 1):
            if math.ceil(n * 0.1 * (r+1)) >= (t + 1):
                break
            else:
                r = r + 1
        l = math.ceil(n * 0.1 * r)
        if l != l_old and l != 0:
            #Gurobi Call
            dyn = Gurobi(l, m, p[0:l], [i[0:l] for i in a], b*(l/n))    #Use naive fraction of b, namely l/n
            sp_dyn_trans = dyn[2]
            sp_dyn_prices.append(sp_dyn_trans)
        x_hat = 0 
        
        #(a)
        res_con = np.zeros(m, dtype = np.int)       #Resource consumption vector of customer j
        
        for i in range(m):
            res_con[i] = a[i][t]                    #Retrieve resource consumption of customer j
        if p[t] <= np.dot(sp_dyn_trans, res_con):   #Check if bid is less than resource consumption weighted with dual price
            x_hat = 0
        else:
            x_hat = 1
        
        #(b)
        for i in range(m):
            total_con[i] = total_con[i] + (a[i][t-1] * sol_dyn[t-1]) #Add consumption of last customer to total consumption
        indicator = 1                                                #Indicates whether constraint would be violated in case of acceptance
        for i in range(m):
            if a[i][t] * x_hat > b[i] - total_con[i]: 
                indicator = 0                                        #Change indicator if any constraint would be violated
        if indicator == 1:
            sol_dyn[t] = x_hat                                       #If feasbible, use x_hat 
            rev_dyn = rev_dyn + (p[t] * sol_dyn[t])                  #Add revenue
            for i in range(m):
                resource_consumption[i] = resource_consumption[i] + a[i][t]*sol_dyn[t] #Add resource consumption
        else:
            sol_dyn[t] = 0
        
        dyn_time_cust.append(time.time()-cust_start_time)
    
    obj_dyn = rev_dyn
    tot_time = time.time() - start_time
    
    return obj_dyn, sol_dyn, tot_time, dyn_time_cust, sp_dyn_prices, resource_consumption  

def One_Time_Relaxed(n, m, p, a, b, eps): #Do NOT use modifying factor on right-hand side
    """ Calculates the One Time Learner by calculating the solution of the dual 
    prices at eps*n and subsequently answers customer requests with these
    threshold prices. The new restriction does not include the forward looking 
    correction term (1-eps). Takes as input parameters n, m, p, a, b, eps and 
    returns obj_one, sol_one, tot_time, one_time_cust, sp_one_trans, resource_consumption."""
    
    start_time = time.time()
    rev_one_r = 0                                           #Total revenue
    one_time_cust_r = []                                    #Time one-time learning algorithm needs for each customer
    resource_consumption_r = np.zeros(m, dtype = np.int)    #Resources consumed through algorithm
    total_con = np.zeros(m, dtype = np.int)                 #Total resource consumption so far
    
    #Step (i)
    s = math.ceil(n * eps)
    sol_one_r = np.zeros(n, dtype = np.int)                 #Solution vector
    for j in range(s):
        cust_start_time = time.time()
        sol_one_r[j] = 0
        one_time_cust_r.append(time.time()-cust_start_time)
    
    #Gurobi Call
    with suppress_stdout():
        one = Gurobi(s, m, p[0:s], [i[0:s] for i in a], b*(s/n))#*(1-eps)))
    sp_one_trans_r = one[2]                                 #Shadow prices
    
    #Algorithm
    for j in range(s, n):                                   #Note to ourselves: indices start at 0, so we need indices from s to n-1 in order to get customers s to n
        cust_start_time = time.time()
        x_hat = 0
        
        #Equation 12
        res_con = np.zeros(m, dtype = np.int)                #Resource consumption vector of customer j
        for i in range(m):
            res_con[i] = a[i][j]                             #Retrieve resource consumption of customer j
        if p[j] <= np.dot(sp_one_trans_r, res_con):          #Matrix product; Check if bid is less than resource consumption weighted with dual price
            x_hat = 0
        else:
            x_hat = 1
        
        #Step (ii)
        for i in range(m):
            total_con[i] = total_con[i] + (a[i][j-1] * sol_one_r[j-1]) #Add consumption of last customer to total consumption
        indicator = 1                                       #Indicates whether constraint would be violated in case of acceptance
        for i in range(m):
            if a[i][j] * x_hat > b[i] - total_con[i]: 
                indicator = 0                               #Change indicator if any constraint would be violated
        if indicator == 1:
            sol_one_r[j] = x_hat                            #If feasbible, use x_hat 
            rev_one_r = rev_one_r + (p[j] * sol_one_r[j])   #Add revenue
            for i in range(m):
                resource_consumption_r[i] = resource_consumption_r[i] + a[i][j]*sol_one_r[j] #Add resource consumption
            #for i in range(m):
            #    b[i] = b[i] - a[i][j] * sol_one[j]  #Adjust capacity
        else:
            sol_one_r[j] = 0
        
        one_time_cust_r.append(time.time()-cust_start_time)
        
    obj_one_r = rev_one_r
    tot_time_r = time.time() - start_time
    
    return obj_one_r, sol_one_r, tot_time_r, one_time_cust_r, sp_one_trans_r, resource_consumption_r

def Dynamic_Relaxed(n, m, p, a, b, eps):
    """ Calculates the Dynamic Learner by calculating the solution of the dual 
    prices at l=eps*n*2^r and subsequently answers customer requests with these
    threshold prices for each period s.t. l < t. Does not include the forward-
    looking restriction correction term (1-eps*math.pow(n/l,0.5))) 
    Takes as input parameters n, m, p, a, b, eps and returns obj_dyn_r, 
    sol_dyn_r, tot_time_r, dyn_time_cust_r, sp_dyn_prices_r, resource_consumption_r."""

    start_time = time.time()
    rev_dyn_r = 0                   #Total revenue
    l = 0
    l_old = 0
    dyn_time_cust_r = []            #Time dynamic learning algorithm needs for each customer
    sp_dyn_prices_r = []            #Array of arrays with shadow prices
    resource_consumption_r = np.zeros(m, dtype = np.int)  #Resources consumed through algorithm
    total_con = np.zeros(m, dtype = np.int) #Total resource consumption so far
       
    #Step (i)
    t0 = math.ceil(n * eps)
    sol_dyn_r = np.zeros(n, dtype = np.int)   #Solution vector
    t1 = 0
    for t1 in range(t0):
        cust_start_time = time.time()
        sol_dyn_r[t1] = 0
        dyn_time_cust_r.append(time.time()-cust_start_time)
        
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
            with suppress_stdout():
                dyn = Gurobi(l, m, p[0:l], [i[0:l] for i in a], b*(l/n))#*(1-eps*math.pow(n/l,0.5)))
            sp_dyn_trans = dyn[2]
            sp_dyn_prices_r.append(sp_dyn_trans)
            
        x_hat = 0 
        
        #(a)
        res_con = np.zeros(m, dtype = np.int)        #Resource consumption vector of customer j
        for i in range(m):
            res_con[i] = a[i][t]                     #Retrieve resource consumption of customer j
        if p[t] <= np.dot(sp_dyn_trans, res_con):    #Check if bid is less than resource consumption weighted with dual price
            x_hat = 0
        else:
            x_hat = 1
        
        #(b)
        for i in range(m):
            total_con[i] = total_con[i] + (a[i][t-1] * sol_dyn_r[t-1]) #Add consumption of last customer to total consumption
        indicator = 1                               #Indicates whether constraint would be violated in case of acceptance
        for i in range(m):
            if a[i][t] * x_hat > b[i] - total_con[i]: 
                indicator = 0                       #Change indicator if any constraint would be violated
        if indicator == 1:
            sol_dyn_r[t] = x_hat                    #If feasbible, use x_hat 
            rev_dyn_r = rev_dyn_r + (p[t] * sol_dyn_r[t])   #Add revenue
            for i in range(m):
                resource_consumption_r[i] = resource_consumption_r[i] + a[i][t]*sol_dyn_r[t] #Add resource consumption
           
        else:
            sol_dyn_r[t] = 0
        
        dyn_time_cust_r.append(time.time()-cust_start_time)
    
    obj_dyn_r = rev_dyn_r
    tot_time_r = time.time() - start_time
    
    return (obj_dyn_r, sol_dyn_r, tot_time_r, dyn_time_cust_r, sp_dyn_prices_r, resource_consumption_r)

    
def WTP_Learner(n, m, p, a, b):
    """ Updates dual prices whenever the average WTP per ticket changes up or 
    down by 5%. This includes average over all ticket categories, since WTP 
    for one bundle cannot be split up into the single tickets. Takes as input 
    parameters n, m, p, a, b and returns obj_dyn_w, sol_dyn_w, tot_time_w, 
    dyn_time_cust_w, sp_dyn_prices_w, resource_consumption_w."""
    
    start_time = time.time()
    rev_dyn_w = 0                                           #Total revenue
    p_avg = 0                                               #Average WTP
    dyn_time_cust_w = []                                    #Time dynamic learning algorithm needs for each customer
    sp_dyn_prices_w = []                                    #Array of arrays with shadow prices
    resource_consumption_w = np.zeros(m, dtype = np.int)    #Resources consumed through algorithm
    total_con = np.zeros(m, dtype = np.int)                 #Total resource consumption so far
    sum_p = 0
    sum_a = 0
    
    #Step (i)
    t0 = math.ceil(n * 0.1)
    sol_dyn_w = np.zeros(n, dtype = np.int)   #Solution vector
    t1 = 0
    
    for t1 in range(t0):
        cust_start_time = time.time()
        sol_dyn_w[t1] = 0
        dyn_time_cust_w.append(time.time()-cust_start_time)
        
    p_avg = (sum(p[0:t0]))/(sum(sum([i[0:t0] for i in a])))          #Average WTP for one ticket
    #Gurobi Call
    with suppress_stdout():
        dyn = Gurobi(t0, m, p[0:t0], [i[0:t0] for i in a], b*(t0/n)) #Use naive fraction of b, namely l/n
    sp_dyn_trans = dyn[2]
    sp_dyn_prices_w.append(sp_dyn_trans)
    
    #Step (ii)
    for t in range(t0, n):
        cust_start_time = time.time()
        sum_p = sum_p + p[t]
        sum_a = sum_a + sum([i[t] for i in a])
        p_avg_new = sum_p / sum_a
        #p_avg_new = (sum(p[0:t]))/(sum(sum([i[0:t] for i in a])))      #Current average WTP for one ticket
        if p_avg_new <= 0.95 * p_avg or p_avg_new >= 1.05 * p_avg:      #If average WTP changes by 5% in either direction
            p_avg = p_avg_new                                           #Update average WTP
            #Gurobi Call
            with suppress_stdout():
                dyn = Gurobi(t, m, p[0:t], [i[0:t] for i in a], b*(t/n))#Use naive fraction of b, namely l/n
            sp_dyn_trans = dyn[2]
            sp_dyn_prices_w.append(sp_dyn_trans)
        x_hat = 0
        
        #(a)
        res_con = np.zeros(m, dtype = np.int)                           #Resource consumption vector of customer j
        for i in range(m):
            res_con[i] = a[i][t]                                        #Retrieve resource consumption of customer j
        if p[t] <= np.dot(sp_dyn_trans, res_con):                       #Check if bid is less than resource consumption weighted with dual price
            x_hat = 0
        else:
            x_hat = 1
        
        #(b)
        for i in range(m):
            total_con[i] = total_con[i] + (a[i][t-1] * sol_dyn_w[t-1])  #Add consumption of last customer to total consumption
        indicator = 1                                                   #Indicates whether constraint would be violated in case of acceptance
        for i in range(m):
            if a[i][t] * x_hat > b[i] - total_con[i]: 
                indicator = 0                                           #Change indicator if any constraint would be violated
        if indicator == 1:
            sol_dyn_w[t] = x_hat                                        #If feasbible, use x_hat 
            rev_dyn_w = rev_dyn_w + (p[t] * sol_dyn_w[t])               #Add revenue
            for i in range(m):
                resource_consumption_w[i] = resource_consumption_w[i] + a[i][t]*sol_dyn_w[t] #Add resource consumption
        else:
            sol_dyn_w[t] = 0
        dyn_time_cust_w.append(time.time()-cust_start_time)
    
    obj_dyn_w = rev_dyn_w
    tot_time_w = time.time() - start_time
    
    return obj_dyn_w, sol_dyn_w, tot_time_w, dyn_time_cust_w, sp_dyn_prices_w, resource_consumption_w

def Amazon_Learner(n, m, p, a, b, eps):
    """ Calculates a learning algorithm which is similar to the one associated 
    with the one from Amazon Co. Learns prices at l=eps*n*2^r. Highly demanded categories 
    receive a price discount of 10% whereas lower demand categories receive 
    a surcharge of 10% on the price. The function takes as input 
    parameters n, m, p, a, b, eps and returns."""
    
    start_time = time.time()
    rev_dyn_a = 0     #Total revenue
    l = 0
    l_old = 0
    dyn_time_cust_a = []  #Time dynamic learning algorithm needs for each customer
    sp_dyn_prices_a = []  #Array of arrays with shadow prices
    resource_consumption_a = np.zeros(m, dtype = np.int)  #Resources consumed through algorithm
    total_con = np.zeros(m, dtype = np.int) #Total resource consumption so far
    
    #Check right-hand side condition
    if min(b) >= (10 * float(m) * math.log(float(n)/eps)) / (pow(eps,2)):
        print("Right-hand-side condition for dynamic learning: TRUE")
    else:
        print("Right-hand-side condition for dynamic learning: FALSE")
    
    #Step (i)
    t0 = math.ceil(n * eps)
    sol_dyn_a = np.zeros(n, dtype = np.int)   #Solution vector
    t1 = 0
    
    for t1 in range(t0):
        cust_start_time = time.time()
        sol_dyn_a[t1] = 0
        dyn_time_cust_a.append(time.time()-cust_start_time)
        
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
            #Now: adjust dual prices: reduce for cheap categories, increase for expensive categories
            if m % 2 == 1: #uneven m
                for i in range(int(((m + 1) / 2) - 1)):
                    sp_dyn_trans[i] = sp_dyn_trans[i] + sp_dyn_trans[i] * (0.1 / (((m + 1) / 2) - 1)) * (((m + 1) / 2) - (i + 1))   #Tuning parameter: 10% (surcharge)
                for i in range(int(((m + 1) / 2)), m):
                    sp_dyn_trans[i] = sp_dyn_trans[i] + sp_dyn_trans[i] * (0.1 / (((m + 1) / 2) - 1)) * (((m + 1) / 2) - (i + 1))   #Tuning parameter: 10% (discount)
            if m % 2 == 0:
                for i in range(int(m / 2)):
                    sp_dyn_trans[i] = sp_dyn_trans[i] + sp_dyn_trans[i] * (0.1 / (m / 2)) * ((m / 2) + 1 - (i + 1))                 #Tuning parameter: 10% (Zuschlag)
                for i in range(int(m / 2), m):
                    sp_dyn_trans[i] = sp_dyn_trans[i] + sp_dyn_trans[i] * (0.1 / (m / 2)) * ((m / 2) - (i + 1))                     #Tuning parameter: 10% (Abschlag)
            sp_dyn_prices_a.append(sp_dyn_trans)
            
        x_hat = 0 
        
        #(a)
        res_con = np.zeros(m, dtype = np.int)     #Resource consumption vector of customer j
        
        for i in range(m):
            res_con[i] = a[i][t]                  #Retrieve resource consumption of customer j
        if p[t] <= np.dot(sp_dyn_trans, res_con): #Check if bid is less than resource consumption weighted with dual price
            x_hat = 0
        else:
            x_hat = 1
        
        #(b)
        for i in range(m):
            total_con[i] = total_con[i] + (a[i][t-1] * sol_dyn_a[t-1]) #Add consumption of last customer to total consumption
        
        indicator = 1                             #Indicates whether constraint would be violated in case of acceptance
        
        for i in range(m):
            if a[i][t] * x_hat > b[i] - total_con[i]: 
                indicator = 0                     #Change indicator if any constraint would be violated
        
        if indicator == 1:
            sol_dyn_a[t] = x_hat                  #If feasbible, use x_hat 
            rev_dyn_a = rev_dyn_a + (p[t] * sol_dyn_a[t])   #Add revenue
            for i in range(m):
                resource_consumption_a[i] = resource_consumption_a[i] + a[i][t]*sol_dyn_a[t] #Add resource consumption
           
        else:
            sol_dyn_a[t] = 0
        
        dyn_time_cust_a.append(time.time()-cust_start_time)
    
    obj_dyn_a = rev_dyn_a
    tot_time_a = time.time() - start_time
    
    return obj_dyn_a, sol_dyn_a, tot_time_a, dyn_time_cust_a, sp_dyn_prices_a, resource_consumption_a
