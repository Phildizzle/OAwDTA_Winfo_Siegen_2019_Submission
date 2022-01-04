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
import sys
from scipy.optimize import linprog
from gurobipy import *
import math
from Functions import Gurobi, Gurobi_integral, simulate, permute, One_Time_Learning, Dynamic_Learning
import time
import matplotlib.pyplot as plt

#Initialization
print("Welcome to the Ticket Case Simulation")
print("Please initialize the problem instance in the following")
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

lb_one = math.floor(eps*math.exp((min(b_con)*math.pow(eps, 3))/(6*m)))
lb_dyn = math.floor(eps*math.exp((min(b_con)*math.pow(eps, 2))/(10*m)))

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
    
#Check right-hand side conditions
ind_one = "not"
ind_dyn = "not"

if min(b_con) >= (6 * float(m) * math.log(float(n)/eps)) / (pow(eps, 3)):
        ind_one = ""
        
if min(b_con) >= (10 * float(m) * math.log(float(n)/eps)) / (pow(eps,2)):
        ind_dyn = ""


ex = np.zeros(numb)
ex_i = np.zeros(numb)
avg_one = np.zeros(numb)
avg_dyn = np.zeros(numb)
perc_one = np.zeros(numb)
perc_dyn = np.zeros(numb)

one_win = np.zeros(numb)     #Number of times where One-Time Learning Algorithm performs better
dyn_win = np.zeros(numb)     #Number of times where Dynamic Learning Algorithm performs better
avg_perm_time = np.zeros(numb)
avg_one_time = np.zeros(numb)
avg_dyn_time = np.zeros(numb)  

sim_start_time = np.zeros(numb)
sim_end_time = np.zeros(numb)

one_res = []        #Array of array with results of one-time learning algorithm
dyn_res = []        #Array of array with results of dynamic learning algorithm

one_time_res = []   #Array of array with runtimes of one-time learning algorithm
dyn_time_res = []   #Array of array with runtimes of dynamic learning algorithm
perm_time_res =  [] #Array with average runtime of each permutation

if count == 1:
    one_time_cust = []   #In case of one permutation: record time one-time learning algorithm needs for each customer
    dyn_time_cust = []   #In case of one permutation: record time dynamic learning algorithm needs for each customer
    wtp = []             #In case of one permutation: all incoming willingness to pay
    wtp_one_ac = []      #In case of one permutation: accepted willingness to pay under one-time learning algorithm
    wtp_dyn_ac = []      #In case of one permutation: accepted willingness to pay under dynamic algorithm
    sp_one = []          #In case of one permutation: Shadow prices of one-time-learning algorithm
    sp_dyn = []          #In case of one permutation: Shadow prices of dynamic learning algorithm

for f in range(numb):
    sim_start_time[f] = time.time()
    #Simulate incoming customers
    sim = simulate(n, m)
    p = sim[0]
    a = sim[1]
    
    k = 0
   
    ex[f] = Gurobi(n, m, p, a, b_con)[0]              #Ex-post optimum remains unchanged over permutations
    ex_i[f] = Gurobi_integral(n, m, p, a, b_con)[0]   #Ex-post optimum remains unchanged over permutations
    
    one = np.zeros(count)
    dyn = np.zeros(count)
    
    perm_start_time = np.zeros(count)
    perm_end_time = np.zeros(count)
    
    one_time = np.zeros(count)  
    dyn_time = np.zeros(count)  
    
    for k in range(count):
        
        perm_start_time[k] = time.time()
        
        ######################################
        #Choose random permutation
        perm = permute(p, a)
        p = perm[0]
        a = perm[1]
        if count == 1:
            wtp.append(p)
            
        ######################################
        #One-time Learning Algorithm
        o = One_Time_Learning(n, m, p, a, b_con, eps)
        one[k] = o[0]
        one_time[k] = o[2] 
        if count == 1:
            one_time_cust.append(o[3])  #Time per customer
            w = []
            for j in range(n):
                if o[1][j] == 1:
                    w.append(p[j])
            wtp_one_ac.append(w)    #Accepted willingess to pay
            sp_one.append(o[4]) #Shadow prices
        
        ######################################
        #Dynamic pricing algorithm      
        d = Dynamic_Learning(n, m, p, a, b_con, eps)
        dyn[k] = d[0]
        dyn_time[k] = d[2]
        if count == 1:
            dyn_time_cust.append(d[3])  #Time per customer
            w = []
            for j in range(n):
                if d[1][j] == 1:
                    w.append(p[j])
            wtp_dyn_ac.append(w)    #Accepted willingess to pay
            sp_dyn.append(d[4]) #Shadow prices
        
        if dyn[k] > one[k]:
            dyn_win[f] = dyn_win[f] + 1
        if dyn[k] < one[k]:
            one_win[f] = one_win[f] + 1
            
        perm_end_time[k] = time.time()
    
    avg_one[f] = np.sum(one) / count
    avg_dyn[f] = np.sum(dyn) / count
    
    perc_one[f] = (avg_one[f] / ex[f]) * 100
    perc_dyn[f] = (avg_dyn[f] / ex[f]) * 100
    
    avg_perm_time[f] = np.sum(perm_end_time - perm_start_time) / count
    avg_one_time[f] = np.sum(one_time) / count
    avg_dyn_time[f] = np.sum(dyn_time) / count
    
    one_res.append(one)
    dyn_res.append(dyn)
    
    one_time_res.append(one_time)
    dyn_time_res.append(dyn_time)
    perm_time_res.append(perm_end_time - perm_start_time)
    
    sim_end_time[f] = time.time()

for f in range(numb):
    print("\n\n--------------------------------------------------\nSimulation No. " + str(f + 1) + ": ")
    print("The One-Time Learning Algorithm performed better in " + str(one_win[f]) + " out of " + str(count) + " permutations.")
    print("The Dynamic Learning Algorithm performed better in " + str(dyn_win[f]) + " out of " + str(count) + " permutations.")
    print("Equal performance in " + str(count - dyn_win[f] - one_win[f]) + " permutations.")
    print("The ex-post optimum was: " + str(round(ex[f], 2)))
    print("The average performance of the One-Time Learning Algorithm was: " + str(round(avg_one[f], 2)) + ". That is " + str(round(perc_one[f], 2)) + "% of the ex-post optimum.")
    print("The average performance of the Dynamic Learning Algorithm was: " + str(round(avg_dyn[f], 2)) + ". That is " + str(round(perc_dyn[f], 2)) + "% of the ex-post optimum.")
    print("Average runtime of One-Time Learning Algorithm: --- %s seconds ---" % round(avg_one_time[f], 5))
    print("Average runtime of Dynamic Algorithm: --- %s seconds ---" % round(avg_dyn_time[f], 5))
    print("Average runtime of permutations: --- %s seconds ---" % round(avg_perm_time[f], 5))
    print("Total runtime of simulation: --- %s seconds ---" % round(sim_end_time[f] - sim_start_time[f], 5))
    
    if count > 1:
        x_ax = np.arange(count) + 1
        fig, ax = plt.subplots()
        ax.plot(x_ax, 100*one_res[f]/ex[f], 'b', label = 'One-Time')
        ax.plot(x_ax, 100*dyn_res[f]/ex[f], 'g', label = 'Dynamic')
        ax.plot(x_ax, 100*one_res[f]/ex[f], 'bs', label = 'One-Time')
        ax.plot(x_ax, 100*dyn_res[f]/ex[f], 'g^', label = 'Dynamic')
        plt.xlabel('Permutation')
        plt.ylabel('Performance in percent of ex-post optimum')
        plt.title('Simulation ' + str(f + 1) + ': Performance of One-Time vs. Dynamic')
        plt.axis([0, count+1, 0, 100])
        plt.xticks(np.arange(0, count + 1, step=1))
        legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(1.75, 0.5))
        plt.show()
        
        fig, ax = plt.subplots()
        ax.plot(x_ax, one_time_res[f], 'b', label='One-Time')
        ax.plot(x_ax, dyn_time_res[f], 'g', label='Dynamic')
        ax.plot(x_ax, one_time_res[f], 'bs', label='One-Time')
        ax.plot(x_ax, dyn_time_res[f], 'g^', label='Dynamic')
        plt.xlabel('Permutation')
        plt.ylabel('Runtime of algorithm in seconds')
        plt.title('Runtime of Algorithms for each Permutation')
        plt.axis([0, count+1, 0, max(dyn_time_res[f])*1.25])
        plt.xticks(np.arange(0, count + 1, step=1))
        legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(1.75, 0.5))
        plt.show()
    
        fig, ax = plt.subplots()
        ax.plot(x_ax, perm_time_res[f], 'b', label='Runtime of Total Permutation')
        ax.plot(x_ax, perm_time_res[f], 'bs', label='Runtime of Total Permutation')
        plt.xlabel('Permutation')
        plt.ylabel('Runtime of total permutation in seconds')
        plt.title('Runtime of Total Permutation')
        plt.axis([0, count+1, 0, max(perm_time_res[f])*1.25])
        plt.xticks(np.arange(0, count + 1, step=1))
        legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(2, 0.5))
        plt.show()
    
    if count == 1:
        x_ax = np.arange(n) + 1
        fig, ax = plt.subplots()
        ax.plot(x_ax, one_time_cust[f], 'b', label='Runtime of One-Time Learning Algorithm')
        ax.plot(x_ax, one_time_cust[f], 'bs', label='Runtime of One-Time Learning Algorithm')
        ax.plot(x_ax, dyn_time_cust[f], 'g', label='Runtime of Dynamic Learning Algorithm')
        ax.plot(x_ax, dyn_time_cust[f], 'g^', label='Runtime of Dynamic Learning Algorithm')
        plt.xlabel('Customer')
        plt.ylabel('Runtime for customer in seconds')
        plt.title('Runtime for Each Customer in the Single Permutation')
        plt.axis([0, n+1, 0, max(dyn_time_cust[f])*1.25])
        plt.xticks(np.arange(0, count + 1, step=1))
        legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(2, 0.5))
        plt.show()    
        
        kwargs = dict(histtype='stepfilled', alpha=0.3, bins=10)
        plt.hist(wtp[f], **kwargs, label='All Incoming Willingness-To-Pay\nunder One-Time Learning Algorithm')
        plt.hist(wtp_one_ac[f], **kwargs, label='Accepted Willingness-To-Pay\nunder One-Time Learning Algorithm')
        plt.hist(wtp_dyn_ac[f], **kwargs, label='Accepted Willingness-To-Pay\nunder Dynamic Learning Algorithm')
        plt.xlabel('Willingness-to-Pay')
        plt.ylabel('Frequency')
        plt.title('Histogram of Willingness-To-Pay in the Single Permutation')
        legend = plt.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(2, 0.5))
        plt.show()
        
        """#Separate diagram for dynamic learning
        kwargs = dict(histtype='stepfilled', alpha=0.3, bins=10)
        plt.hist(wtp[f], **kwargs, label='All Incoming Willingness-To-Pay\nunder Dynamic Learning Algorithm')
        plt.hist(wtp_dyn_ac[f], **kwargs, label='Accepted Willingness-To-Pay\nunder Dynamic Learning Algorithm')
        plt.xlabel('Willingness-to-Pay')
        plt.ylabel('Frequency')
        plt.title('Histogram of Willingness-To-Pay in the Single Permutation\nunder Dynamic Learning Algorithm')
        legend = plt.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(2, 0.5))
        plt.show()"""
        
        if m <= 5:  #Only plot if not too many resources (otherwise too many plots)
            for i in range(m):
                fig, ax = plt.subplots()
                ax.plot([row[i] for row in sp_dyn[f]], 'b', label='Dynamic Learning Shadow Prices')
                ax.plot([row[i] for row in sp_dyn[f]], 'bs', label='Dynamic Learning Shadow Prices')
                plt.axhline(y=sp_one[f][i], color = 'g', label = 'One-Time Learning Shadow Price')
                plt.xlabel('Shadow Price Iteration')
                plt.ylabel('Shadow Price')
                plt.title('Shadow Price for Resource ' + str(i + 1) + " in the Single Permutation")
                plt.ylim(ymin = 0)
                legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(2, 0.5))
                plt.show()

print("\n\n--------------------------------------------------\nThe right-hand side condition was " + ind_one + " satisfied for the One-Time Learning Algorithm.")
print("The right-hand side condition was " + ind_dyn + " satisfied for the Dynamic Learning Algorithm.")

if numb > 1:
    x_ax = np.arange(numb) + 1
    fig, ax = plt.subplots()
    ax.plot(x_ax, perc_one, 'b', label='One-Time')
    ax.plot(x_ax, perc_dyn, 'g', label='Dynamic')
    ax.plot(x_ax, perc_one, 'bs', label='One-Time')
    ax.plot(x_ax, perc_dyn, 'g^', label='Dynamic')
    plt.xlabel('Simulation')
    plt.ylabel('Average Performance in percent of ex-post optimum')
    plt.title('Average Performance: One-Time vs. Dynamic')
    plt.axis([0, numb+1, 0, 100])
    plt.xticks(np.arange(0, numb + 1, step=1))
    legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(1.75, 0.5))
    plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(x_ax, sim_end_time - sim_start_time, 'b', label='Total Runtime')
    ax.plot(x_ax, sim_end_time - sim_start_time, 'bs', label='Total Runtime')
    plt.xlabel('Simulation')
    plt.ylabel('Runtime of simulation in seconds')
    plt.title('Runtime of Total Simulation')
    plt.axis([0, numb+1, 0, max(sim_end_time - sim_start_time)*1.25])
    plt.xticks(np.arange(0, numb + 1, step=1))
    legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(1.75, 0.5))
    plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(x_ax, avg_one_time, 'b', label='Average runtime of\nOne-Time Learning Algorithm in seconds')
    ax.plot(x_ax, avg_one_time, 'bs', label='Average runtime of\nOne-Time Learning Algorithm in seconds')
    ax.plot(x_ax, avg_dyn_time, 'g', label='Average runtime of\nDynamic Learning Algorithm in seconds')
    ax.plot(x_ax, avg_dyn_time, 'g^', label='Average runtime of\nDynamic Learning Algorithm in seconds')
    plt.xlabel('Simulation')
    plt.ylabel('Average runtime of algorithms in seconds')
    plt.title('Average Runtime of Algorithms over all Permutations')
    plt.axis([0, numb+1, 0, max(avg_dyn_time)*1.25])
    plt.xticks(np.arange(0, numb + 1, step=1))
    legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(2.5, 0.5))
    plt.show()
    
    fig, ax = plt.subplots()
    ax.plot(x_ax, avg_perm_time, 'b', label='Average runtime of a permutation in seconds')
    ax.plot(x_ax, avg_perm_time, 'bs', label='Average runtime of a permutation in seconds')
    plt.xlabel('Simulation')
    plt.ylabel('Average runtime of a permutation in seconds')
    plt.title('Average Runtime of Total Permutation')
    plt.axis([0, numb+1, 0, max(avg_perm_time)*1.25])
    plt.xticks(np.arange(0, numb + 1, step=1))
    legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(2, 0.5))
    plt.show()


print("Done")