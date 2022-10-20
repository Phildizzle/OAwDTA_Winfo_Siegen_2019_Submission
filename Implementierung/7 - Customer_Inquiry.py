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
#This file should mirror the real-time application

######################################

#Preparation
print("\033[H\033[J")   #Clear Console
import numpy as np
import scipy as sc
import sys
from scipy.optimize import linprog
from gurobipy import *
import math
from Functions import Gurobi, Gurobi_integral, simulate, permute, One_Time_Learning, Dynamic_Learning
from Analysis_Customer_Inquiry import Analysis_Customer_Inquiry
from contextlib import contextmanager
import sys, os
import matplotlib.pyplot as plt

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
max_a = 5   #Maximum number of tickets permitted per customer and category. CHANGE HERE IF NECESSARY AND IN ACCORDANCE WITH "FUNCTIONS"

print("Welcome to the Ticket Case Simulation")
print("Please initialize the problem instance in the following")
print("Note: There is a ticket limit: A maximum of " + str(max_a) + " tickets per category per request permitted")
print("\nNote: In case you want to quit the simulations, simply enter \"exit\" as input")
n = 0
m = 0

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

lb_one = math.floor(eps*math.exp((min(b_con/max_a)*math.pow(eps, 3))/(6*m)))
lb_dyn = math.floor(eps*math.exp((min(b_con/max_a)*math.pow(eps, 2))/(10*m)))

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

j = 0
i = 0
p = np.zeros(n, dtype = np.int)
a = np.zeros((m, n), dtype = np.int)

#One-Time Learning Algorithm
rev_one = 0
s = math.ceil(n * eps)
sol_one = np.zeros(n, dtype = np.int) 

#Dynamic Learning Algorithm
rev_dyn = 0     #Total revenue
l = 0
l_old = 0
t0 = math.ceil(n * eps)
sol_dyn = np.zeros(n, dtype = np.int)   #Solution vector
t1 = 0

for j in range(n):
    print("\n\n----------------------------------------------------\nHello Customer " + str(j + 1) + "!")
    for i in range(m):
        while True:
            try:
                inp = input("How many tickets of category " + str(i + 1) + " would you like to purchase: ")
                if inp == "Exit" or inp == "exit":
                    ind_exit = 1
                    break
                a[i][j] = int(inp)
                if a[i][j] > max_a:
                    print("\nSorry, we currently have a ticket limit of " + str(max_a) + " tickets per category and customer.\nPlease enter an integer number between 0 and 5")
                if a[i][j] >= 0 and a[i][j] <= max_a:
                    break
            except:
                print("\nPlease enter an integer number between 0 and 5")
        if ind_exit == 1:
            print("Execution will be stopped")
            sys.exit(0)
    
    while True:
        try:
            inp = input("How much are you willing to pay in total for all tickets: EUR ")
            if inp == "Exit" or inp == "exit":
                ind_exit = 1
                break
            p[j] = float(inp)
            if p[j] > 0:
                break
        except:
            print("\nPlease enter a positive number")
    if ind_exit == 1:
        print("Execution will be stopped")
        sys.exit(0)
        
    #One-Time Learning Algorithm
    print("\n\nYour request will be processed with the One-Time Learning Algorithm...")
    #Step (i)
    if j < s - 1: #First s-1 customers rejected
        sol_one[j] = 0
        print("Sorry, your ticket request cannot be satisfied.")
    if j == s - 1: #s-th customer rejected and computation of shadow prices
        sol_one[j] = 0
        print("Sorry, your ticket request cannot be satisfied.")
        with suppress_stdout():
            one = Gurobi(s, m, p, a, b*(1-eps)*(s/n))
        sp_one_trans = one[2]   #Shadow prices
    if j > s - 1: #Compare with shadow prices
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
        if sol_one[j] == 1:
            print("Congratulations, your ticket request has been granted!")
        else:
            print("Sorry, your ticket request cannot be satisfied.")

    #Dynamic Learning Algorithm
    print("\n\nYour request will be processed with the Dynamic Learning Algorithm...")
    #Step (i)
    if j <= t0 - 1: #First t0 customers rejected
        sol_dyn[j] = 0
        print("Sorry, your ticket request cannot be satisfied.")
    #Step (ii)
    if j > t0 - 1:
        l_old = l
        r = 0
        while math.ceil(n * eps * math.pow(2,r)) < (j + 1):
            if math.ceil(n * eps * math.pow(2,r+1)) >= (j + 1):
                break
            else:
                r = r + 1
        l = math.ceil(n * eps * math.pow(2,r))
        if l != l_old and l != 0:
            #Gurobi
            with suppress_stdout():
                dyn = Gurobi(l, m, p, a, b*(l/n)*(1-eps*math.pow(n/l,0.5)))
            sp_dyn_trans = dyn[2]
            
        x_hat = 0 
        # (a)
        res_con = np.zeros(m, dtype = np.int) #Resource consumption vector of customer j
        for i in range(m):
            res_con[i] = a[i][j]    #Retrieve resource consumption of customer j
        if p[j] <= np.dot(sp_dyn_trans, res_con): #Check if bid is less than resource consumption weighted with dual price
            x_hat = 0
        else:
            x_hat = 1
        
        # (b)
        k = 0
        total_con = np.zeros(m, dtype = np.int) #Total resource consumption so far
        for k in range(t0, j):   #Total consumption only requires customers from t0 to j (first t0 customers will always be zero)
            for i in range(m):
                total_con[i] = total_con[i] + (a[i][k] * sol_dyn[k])
        indicator = 1   #Indicates whether constraint would be violated in case of acceptance
        for i in range(m):
            if a[i][j] * x_hat > b[i] - total_con[i]: 
                indicator = 0   #Change indicator if any constraint would be violated
        if indicator == 1:
            sol_dyn[j] = x_hat  #If feasbible, use x_hat 
            rev_dyn = rev_dyn + (p[j] * sol_dyn[j])   #Add revenue
        else:
            sol_dyn[j] = 0
            
        if sol_dyn[j] == 1:
            print("Congratulations, your ticket request has been granted!")
        else:
            print("Sorry, your ticket request cannot be satisfied.")
    
obj_one = rev_one
obj_dyn = rev_dyn

print("\n\n---------------------------------------------\n")
print("Total revenue under One-Time Learning Algorithm: " + str(round(obj_one, 2)))
print("Total revenue under Dynamic Learning Algorithm: " + str(round(obj_dyn, 2)))

with suppress_stdout():
    #Calculate fractional ex-post optimum
    ex = Gurobi(n, m, p, a, b_con)
    #Calculate ex-post integral optimum
    ex_i = Gurobi_integral(n, m, p, a, b_con)
print("Optimal ex-post fractional revenue: " + str(round(ex[0], 2)))
print("Optimal ex-post integral revenue: " + str(round(ex_i[0], 2)))
print("\n\n---------------------------------------------\n")
print("\nThe right-hand side condition was " + ind_one + " satisfied for the One-Time Learning Algorithm.")
print("The right-hand side condition was " + ind_dyn + " satisfied for the Dynamic Time Learning Algorithm.")
print("\nFor further analysis: Run \"Analysis_Customer_Inquiry\" ")
Analysis_Customer_Inquiry(n, m, p, a, b_con, eps, bench, max_a)
print("\n\nSimulation finished.\n\n")
