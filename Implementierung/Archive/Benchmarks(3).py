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
def Greedy(n, m, p, a, b):  #Accept every incoming inquiry as long as no capacity constraint is violated
    used_cap = np.zeros(m, dtype = np.int)
    x = np.zeros(n, dtype = np.int)
    rev = 0
    for j in range(n):
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
        print("Kunde: " + str(t+1))
        print("Index: " + str(t))
        print("r old: " + str(r))
        while math.ceil(n * 0.1 * r) < (t + 1):
            if math.ceil(n * 0.1 * (r+1)) >= (t + 1):
                break
            else:
                r = r + 1
        l = math.ceil(n * 0.1 * r)
        print("r: " + str(r))
        input("l: " + str(l))
        if l != l_old and l != 0:
            #Gurobi
            dyn = Gurobi(l, m, p[0:l], [i[0:l] for i in a], b*(l/n))
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
    
  def WTP_Learner(n, m, p, a, b):
    start_time = time.time()
    rev_dyn = 0     #Total revenue
    #l = 0
    #l_old = 0
    p_avg = 0
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
        print("Kunde: " + str(t+1))
        print("Index: " + str(t))
        print("r old: " + str(r))
        while math.ceil(n * 0.1 * r) < (t + 1):
            if math.ceil(n * 0.1 * (r+1)) >= (t + 1):
                break
            else:
                r = r + 1
        l = math.ceil(n * 0.1 * r)
        print("r: " + str(r))
        input("l: " + str(l))
        if l != l_old and l != 0:
            #Gurobi
            dyn = Gurobi(l, m, p[0:l], [i[0:l] for i in a], b*(l/n))
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
    
    
print("Done")