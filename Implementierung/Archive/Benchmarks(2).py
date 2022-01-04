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
#Run several simulations

######################################

#Preparation
print("\033[H\033[J")   #Clear Console
import numpy as np
import scipy as sc
import sys, os
from scipy.optimize import linprog
from gurobipy import *
import math
from Functions import Gurobi, Gurobi_integral, simulate, permute, One_Time_Learning, Dynamic_Learning
import time
import matplotlib.pyplot as plt
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

#Initialization
def Greedy(n, m, p, a, b, eps):  #Accept every incoming inquiry as long as no capacity constraint is violated
    used_cap = np.zeros(m, dtype = np.int)
    x = np.zeros(n, dtype = np.int)
    rev = 0
    for j in range(int(np.ceil(n*eps)), n): #Exclude calibration period! Due to reasons of fairness!
        indicator = 0   #Check whether resource limit would be exceeded
        for i in range(m):
            if used_cap[i] + a[i][j] > b[i]:    
                indicator = 1   #One resource capacity is violated
                break
        if indicator == 0:      #If resource capacity is not violated, then accept offer
            rev = rev + p[j]
            for i in range(m):
                used_cap[i] = used_cap[i] + a[i][j]
            x[j] = 1
        #if a resource capacity is violated, then do nothing
    return(rev, x, used_cap)
    
def Interval_Learner(n, m, p, a, b):
    start_time = time.time()
    rev_dyn = 0     #Total revenue
    l = 0
    l_old = 0
    dyn_time_cust = []  #Time dynamic learning algorithm needs for each customer
    sp_dyn_prices = []  #Array of arrays with shadow prices
    resource_consumption = np.zeros(m, dtype = np.int)  #Resources consumed through algorithm
     
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
            #Gurobi
            dyn = Gurobi(l, m, p[0:l], [i[0:l] for i in a], b*(l/n))    #Use naive fraction of b, namely l/n
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
            for i in range(m):
                resource_consumption[i] = resource_consumption[i] + a[i][t]*sol_dyn[t] #Add resource consumption
            
        else:
            sol_dyn[t] = 0
        
        dyn_time_cust.append(time.time()-cust_start_time)
    
    obj_dyn = rev_dyn
    
    tot_time = time.time() - start_time
    
    return (obj_dyn, sol_dyn, tot_time, dyn_time_cust, sp_dyn_prices, resource_consumption)   

def One_Time_Relaxed(n, m, p, a, b, eps): #Do NOT use modifying factor on right-hand side
    start_time = time.time()
    rev_one = 0     #Total revenue
    one_time_cust = []  #Time one-time learning algorithm needs for each customer
    resource_consumption = np.zeros(m, dtype = np.int)  #Resources consumed through algorithm
      
    #Step (i)
    s = math.ceil(n * eps)
    sol_one = np.zeros(n, dtype = np.int)   #Solution vector
    for j in range(s):
        cust_start_time = time.time()
        sol_one[j] = 0
        one_time_cust.append(time.time()-cust_start_time)
    
    #Gurobi
    with suppress_stdout():
        one = Gurobi(s, m, p[0:s], [i[0:s] for i in a], b*(1-eps))#*(s/n))
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
            for i in range(m):
                resource_consumption[i] = resource_consumption[i] + a[i][j]*sol_one[j] #Add resource consumption
            #for i in range(m):
            #    b[i] = b[i] - a[i][j] * sol_one[j]  #Adjust capacity
            
        else:
            sol_one[j] = 0
        
        one_time_cust.append(time.time()-cust_start_time)
        
    obj_one = rev_one
    
    tot_time = time.time() - start_time
    
    return (obj_one, sol_one, tot_time, one_time_cust, sp_one_trans, resource_consumption)

def Dynamic_Relaxed(n, m, p, a, b, eps):
    start_time = time.time()
    rev_dyn = 0     #Total revenue
    l = 0
    l_old = 0
    dyn_time_cust = []  #Time dynamic learning algorithm needs for each customer
    sp_dyn_prices = []  #Array of arrays with shadow prices
    resource_consumption = np.zeros(m, dtype = np.int)  #Resources consumed through algorithm
    #b = b_con[:]       #Reset capacity
       
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
            with suppress_stdout():
                dyn = Gurobi(l, m, p[0:l], [i[0:l] for i in a], b*(l/n))#*(1-eps*math.pow(n/l,0.5)))
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
            for i in range(m):
                resource_consumption[i] = resource_consumption[i] + a[i][t]*sol_dyn[t] #Add resource consumption
           
        else:
            sol_dyn[t] = 0
        
        dyn_time_cust.append(time.time()-cust_start_time)
    
    obj_dyn = rev_dyn
    
    tot_time = time.time() - start_time
    
    return (obj_dyn, sol_dyn, tot_time, dyn_time_cust, sp_dyn_prices, resource_consumption)

    
def WTP_Learner(n, m, p, a, b):     #Updates dual prices whenever the average WTP per ticket changes by 10%. This includes average over all ticket categories, since WTP for one bundle cannot be split up into the single tickets!
    start_time = time.time()
    rev_dyn = 0     #Total revenue
    p_avg = 0       #Average WTP
    dyn_time_cust = []  #Time dynamic learning algorithm needs for each customer
    sp_dyn_prices = []  #Array of arrays with shadow prices
    resource_consumption = np.zeros(m, dtype = np.int)  #Resources consumed through algorithm
     
    #Step (i)
    t0 = math.ceil(n * 0.1)
    sol_dyn = np.zeros(n, dtype = np.int)   #Solution vector
    t1 = 0
    for t1 in range(t0):
        cust_start_time = time.time()
        sol_dyn[t1] = 0
        dyn_time_cust.append(time.time()-cust_start_time)
    p_avg = (sum(p[0:t0]))/(sum(sum([i[0:t0] for i in a])))    #Average WTP for one ticket
    #Gurobi: initial dual prices
    with suppress_stdout():
        dyn = Gurobi(t0, m, p[0:t0], [i[0:t0] for i in a], b*(t0/n))        #Use naive fraction of b, namely l/n
    sp_dyn_trans = dyn[2]
    sp_dyn_prices.append(sp_dyn_trans)
    
    #Step (ii)
    for t in range(t0, n):
        cust_start_time = time.time()
        p_avg_new = (sum(p[0:t]))/(sum(sum([i[0:t] for i in a])))   #Current average WTP for one ticket
        if p_avg_new <= 0.95 * p_avg or p_avg_new >= 1.05 * p_avg:    #If average WTP changes by 10% in either direction
            p_avg = p_avg_new       #Update average WTP
            #Gurobi
            with suppress_stdout():
                dyn = Gurobi(t, m, p[0:t], [i[0:t] for i in a], b*(t/n))        #Use naive fraction of b, namely l/n
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
            for i in range(m):
                resource_consumption[i] = resource_consumption[i] + a[i][t]*sol_dyn[t] #Add resource consumption
            
        else:
            sol_dyn[t] = 0
        
        dyn_time_cust.append(time.time()-cust_start_time)
    
    obj_dyn = rev_dyn
    
    tot_time = time.time() - start_time
    
    return (obj_dyn, sol_dyn, tot_time, dyn_time_cust, sp_dyn_prices, resource_consumption) 

"""def Posted_Price(n, m, p, a, b, eps):     #Use dual prices as posted prices
    start_time = time.time()
    rev_dyn = 0     #Total revenue
    l = 0
    l_old = 0
    dyn_time_cust = []  #Time dynamic learning algorithm needs for each customer
    sp_dyn_prices = []  #Array of arrays with shadow prices
    resource_consumption = np.zeros(m, dtype = np.int)  #Resources consumed through algorithm

    #Step (i)
    t0 = math.ceil(n * eps)
    sol_dyn = np.zeros(n, dtype = np.int)   #Solution vector
    t1 = 0
    for t1 in range(t0):
        cust_start_time = time.time()
        sol_dyn[t1] = 0
        dyn_time_cust.append(time.time()-cust_start_time)
    #!!Use first eps customers for calibration; they provide their WTP denoted as p!!
    with suppress_stdout():
        dyn = Gurobi(t0, m, p[0:t0], [i[0:t0] for i in a], b*(t0/n)*(1-eps*math.pow(n/t0,0.5)))
    
    sp_dyn_trans = dyn[2]
    sp_dyn_prices.append(sp_dyn_trans)
    input("sp: " + str(sp_dyn_trans))
    
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
        l=t
        if l != l_old and l != 0:   #!!Update dual prices, but now with posted prices in objective function instead of WTP!!
            with suppress_stdout():
                #Gurobi
                model = Model("Model")
                customers = range(l)
                v = model.addVars(customers, name = "decision", lb = 0, ub = 1)  #Decision variables and variable bounds
                model.addConstrs(    
                   quicksum(a[i][j]*v[j] for j in range(l)) <= b[i]*(l/n)*(1-eps*math.pow(n/l,0.5)) for i in range(m)   #Capacity constraint
                   )
                model.setObjective(quicksum(quicksum(sp_dyn_trans[i]*a[i][j]*v[j] for i in range(m)) for j in range(l)), GRB.MAXIMIZE)      #Objective Function
            
                #Method = 0
                res = model.optimize()
                sp = np.asarray(model.getAttr("Pi", model.getConstrs()))    #Retrieve shadow prices
            sp_dyn_trans = sp.transpose()   #Update dual prices
            print("l: " + str(l))
            input("sp_it: " + str(sp_dyn_trans))
            sp_dyn_prices.append(sp_dyn_trans)
            
        x_hat = 0 
        # (a)
        res_con = np.zeros(m, dtype = np.int) #Resource consumption vector of customer j
        for i in range(m):
            res_con[i] = a[i][t]    #Retrieve resource consumption of customer j
        if p[t] < np.dot(sp_dyn_trans, res_con): #Check if bid is less than resource consumption weighted with dual price
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
            rev_dyn = rev_dyn + sum(sp_dyn_trans[i]*a[i][t]*sol_dyn[t] for i in range(m))   #Add revenue
            for i in range(m):
                resource_consumption[i] = resource_consumption[i] + a[i][t]*sol_dyn[t] #Add resource consumption
           
        else:
            sol_dyn[t] = 0
        
        dyn_time_cust.append(time.time()-cust_start_time)
    
    obj_dyn = rev_dyn
    
    tot_time = time.time() - start_time
    
    return (obj_dyn, sol_dyn, tot_time, dyn_time_cust, sp_dyn_prices, resource_consumption)"""
