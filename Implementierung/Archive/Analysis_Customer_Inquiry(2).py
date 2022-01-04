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
#Analyse customer inquiry

######################################

#Preparation
import numpy as np
import scipy as sc
import sys
from scipy.optimize import linprog
from gurobipy import *
import math
from Functions import Gurobi, Gurobi_integral, simulate, permute, One_Time_Learning, Dynamic_Learning
import time
import matplotlib.pyplot as plt
from contextlib import contextmanager


#Initialization
def Analysis_Customer_Inquiry(n, m, p, a, b_con, eps):
        
    max_a = 5   #Maximum number of tickets permitted per customer and category. CHANGE HERE IF NECESSARY AND IN ACCORDANCE WITH "FUNCTIONS"
    
    @contextmanager
    def suppress_stdout():
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:  
                yield
            finally:
                sys.stdout = old_stdout
            
    lb_one = math.floor(eps*math.exp((min(b_con/max_a)*math.pow(eps, 3))/(6*m)))
    lb_dyn = math.floor(eps*math.exp((min(b_con/max_a)*math.pow(eps, 2))/(10*m)))
        
    #Check right-hand side conditions
    ind_one = "not"
    ind_dyn = "not"
    
    if min(b_con/max_a) >= (6 * float(m) * math.log(float(n)/eps)) / (pow(eps, 3)):
            ind_one = ""
            
    if min(b_con/max_a) >= (10 * float(m) * math.log(float(n)/eps)) / (pow(eps,2)):
            ind_dyn = ""
    
    numb = 1
    count = 1
    
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
    
    one_resource = []   #Array of array with resource consumption through one-time learning algorithm
    dyn_resource = []   #Array of array with resource consumption through dynamic learning algorithm
    ex_resource = []    #Array of array with resource consumption according to ex-post optimum

    if count == 1:
        one_time_cust = []   #In case of one permutation: record time one-time learning algorithm needs for each customer
        dyn_time_cust = []   #In case of one permutation: record time dynamic learning algorithm needs for each customer
        wtp = []             #In case of one permutation: all incoming willingness to pay
        wtp_one_ac = []      #In case of one permutation: accepted willingness to pay under one-time learning algorithm
        wtp_dyn_ac = []      #In case of one permutation: accepted willingness to pay under dynamic algorithm
        sp_one = []          #In case of one permutation: Shadow prices of one-time-learning algorithm
        sp_dyn = []          #In case of one permutation: Shadow prices of dynamic learning algorithm
        acc_dem = []         #In case of one permutation: Array of arrays with accumulated demand

    for f in range(numb):
        sim_start_time[f] = time.time()  
        with suppress_stdout():
            ex_solution = Gurobi(n, m, p, a, b_con)              #Ex-post optimum remains unchanged over permutations
            ex[f] = ex_solution[0]
            for i in range(m):
                ex_resource.append(sum(ex_solution[1]*a[i]))
            ex_i[f] = Gurobi_integral(n, m, p, a, b_con)[0] 
            
        acc_dem_f = []
        for i in range(m):
            acc_dem_f.append(sum(a[i]))
        acc_dem.append(acc_dem_f)    
    
        one = np.zeros(count)
        dyn = np.zeros(count)
        
        perm_start_time = np.zeros(count)
        perm_end_time = np.zeros(count)
        
        one_time = np.zeros(count)  
        dyn_time = np.zeros(count)  
        
        for k in range(count):
            
            perm_start_time[k] = time.time()
            
            ######################################
            if count == 1:
                wtp.append(p/sum(a[i] for i in range(m)))
                
            ######################################
            #One-time Learning Algorithm
            with suppress_stdout():
                o = One_Time_Learning(n, m, p, a, b_con, eps)
            one[k] = o[0]
            one_time[k] = o[2] 
            one_resource.append(o[5])   #Resource consumption
            if count == 1:
                one_time_cust.append(o[3])  #Time per customer
                w = []
                for j in range(n):
                    if o[1][j] == 1:
                        w.append(p[j]/sum(a[i][j] for i in range(m)))
                wtp_one_ac.append(w)    #Accepted willingess to pay
                sp_one.append(o[4]) #Shadow prices
            
            ######################################
            #Dynamic pricing algorithm      
            with suppress_stdout():
                d = Dynamic_Learning(n, m, p, a, b_con, eps)
            dyn[k] = d[0]
            dyn_time[k] = d[2]
            dyn_resource.append(d[5])   #Resource consumption
            if count == 1:
                dyn_time_cust.append(d[3])  #Time per customer
                w = []
                for j in range(n):
                    if d[1][j] == 1:
                        w.append(p[j]/sum(a[i][j] for i in range(m)))
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
        print("\n\n--------------------------------------------------\n")
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
        
        if count == 1:
            x_ax = np.arange(n) + 1
            fig, ax = plt.subplots()
            ax.plot(x_ax, one_time_cust[f], 'b')
            ax.plot(x_ax, one_time_cust[f], 'bs', label='Runtime of One-Time Learning Algorithm')
            ax.plot(x_ax, dyn_time_cust[f], 'g')
            ax.plot(x_ax, dyn_time_cust[f], 'g^', label='Runtime of Dynamic Learning Algorithm')
            plt.xlabel('Customer')
            plt.ylabel('Runtime for customer in seconds')
            plt.title('Runtime for Each Customer in the Single Permutation')
            plt.axis([0, n+1, 0, max(dyn_time_cust[f])*1.25])
            #plt.xticks(np.arange(0, count + 1, step=1))
            legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(2, 0.5))
            plt.show()    
            
            kwargs = dict(histtype='stepfilled', alpha=0.3, bins=10)
            plt.hist(wtp[f], **kwargs, label='All Incoming Willingness-To-Pay per Ticket')
            plt.hist(wtp_one_ac[f], **kwargs, label='Accepted Willingness-To-Pay per Ticket\nunder One-Time Learning Algorithm')
            plt.hist(wtp_dyn_ac[f], **kwargs, label='Accepted Willingness-To-Pay per Ticket\nunder Dynamic Learning Algorithm')
            plt.xlabel('Willingness-to-Pay per Ticket')
            plt.ylabel('Frequency')
            plt.title('Histogram of Willingness-To-Pay per Ticket in the Single Permutation')
            legend = plt.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(2.25, 0.5))
            plt.show()
            
            if m <= 5:  #Only plot if not too many resources (otherwise too many plots)
                for i in range(m):
                    fig, ax = plt.subplots()
                    ax.plot([row[i] for row in sp_dyn[f]], 'g')
                    ax.plot([row[i] for row in sp_dyn[f]], 'g^', label='Dynamic Learning Shadow Prices')
                    plt.axhline(y=sp_one[f][i], color = 'b', label = 'One-Time Learning Shadow Price')
                    plt.xlabel('Shadow Price Iteration')
                    plt.ylabel('Shadow Price')
                    plt.title('Shadow Price for Resource ' + str(i + 1) + " in the Single Permutation")
                    plt.ylim(ymin = 0)
                    legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(2, 0.5))
                    plt.show()
                    
            fig, ax = plt.subplots()
            ax.bar(np.arange(m)+0.8, ex_resource[0 + f*m:m + f*m], width = 0.2, color = 'y', align = 'center', label = 'Resource Consumption of\nEx-Post Optimal Allocation')
            ax.bar(np.arange(m)+1, one_resource[f], width = 0.2, color = 'b', align = 'center', label = 'Resource Consumption of\nOne-Time Learning Algorithm')
            ax.bar(np.arange(m)+1.2, dyn_resource[f], width = 0.2, color = 'g', align = 'center', label = 'Resource Consumption of\nDynamic Learning Algorithm')
            ax.step(np.arange(m+1)+0.5, [0]+list(b_con), color = 'r', label = 'Maxmimum Capacity')
            ax.step(np.arange(m+1)+0.5, [0]+list(acc_dem[f]), color = 'c', label = 'Accumulated Demand')
            plt.xlabel('Resource')
            plt.ylabel('Resource consumption in units')
            plt.title('Resource Consumption')
            plt.xticks(np.arange(1, m + 1, step=1))
            legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(2, 0.5))
            plt.show()
    
    print("\n\n--------------------------------------------------\nThe right-hand side condition was " + ind_one + " satisfied for the One-Time Learning Algorithm.")
    print("The right-hand side condition was " + ind_dyn + " satisfied for the Dynamic Learning Algorithm.")