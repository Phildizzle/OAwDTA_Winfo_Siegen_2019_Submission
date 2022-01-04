# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 09:02:20 2018

@author: Johannes & Philipp
"""

######################################
#Agrawal et al. Implementation
#Version 1.1 
#July 23, 2018
######################################
#Run automated simulations

######################################

#Preparation
print("\033[H\033[J")   #Clear Console
import numpy as np
import sys, os
from gurobipy import *
import math
from Functions_new import initialize, Gurobi, Gurobi_integral, simulate, permute, One_Time_Learning, Dynamic_Learning, Greedy, Interval_Learner, WTP_Learner, One_Time_Relaxed, Dynamic_Relaxed, Amazon_Learner
import time
import matplotlib.pyplot as plt
from contextlib import contextmanager
import csv

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def readcsv(filename):	
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=";")

    rownum = 0	
    a = []

    for row in reader:
        a.append (row)
        rownum += 1
    
    ifile.close()
    return a

print("Auto-Simulation\n")

inp = readcsv("VaryingM.csv")
m_array = np.asarray(inp[0])
b_array = np.asarray(inp[1])
max_a_array = np.asarray(inp[2])
n_array = np.asarray(inp[3])
eps_array = np.asarray(inp[4])
numb_array = np.asarray(inp[5])
count_array = np.asarray(inp[6])

myData = []

size = len(m_array)
df = 0
for m_count in range(size):
    if not m_array[m_count]:
        break
    for b_count in range(size):
        if not b_array[b_count]:
            break
        for max_a_count in range(size):
            if not max_a_array[max_a_count]:
                break
            for n_count in range(size):
                if not n_array[n_count]:
                    break
                for eps_count in range(size):
                    if not eps_array[eps_count]:
                        break
                    for numb_count in range(size):
                        if not numb_array[numb_count]:
                            break
                        for count_count in range(size):
                            if not count_array[count_count]:
                                break
                            df = df + 1
print("Total combinations: " + str(df))
df = 0
for m_count in range(size):
    if not m_array[m_count]:
        break
    for b_count in range(size):
        if not b_array[b_count]:
            break
        for max_a_count in range(size):
            if not max_a_array[max_a_count]:
                break
            for n_count in range(size):
                if not n_array[n_count]:
                    break
                for eps_count in range(size):
                    if not eps_array[eps_count]:
                        break
                    for numb_count in range(size):
                        if not numb_array[numb_count]:
                            break
                        for count_count in range(size):
                            if not count_array[count_count]:
                                break
                            df = df + 1
                            print("\n\nCombination " + str(df) + "...:")
                            #Initialization
                            max_a = int(max_a_array[max_a_count])   #Maximum number of tickets permitted per customer and category. CHANGE HERE IF NECESSARY AND IN ACCORDANCE WITH "FUNCTIONS"
                            
                            print("\nNote: There is a ticket limit: A maximum of " + str(max_a) + " tickets per category per request permitted")
                            n = int(n_array[n_count])
                            m = int(m_array[m_count])
                            b = np.zeros(m, dtype = np.int)
                            b_con = np.zeros(m, dtype = np.int)     #Copy of b which remains unchanged
                            for i in range(m):
                                b[i] = int(b_array[b_count])
                            b_con = b[:] 
                            eps = float(eps_array[eps_count])
                    
                            lb_one = math.floor(eps*math.exp((min(b_con/max_a)*math.pow(eps, 3))/(6*m)))    #Divide b_con by max_a since this reflects the maximum number of tickets permitted per customer and category. This requires an adjustment of right-hand side condition (see Remark 1.1 in Agrawal et al.)
                            lb_dyn = math.floor(eps*math.exp((min(b_con/max_a)*math.pow(eps, 2))/(10*m)))   #Divide b_con by max_a since max_a reflects the maximum number of tickets permitted per customer and category. This requires an adjustment of right-hand side condition (see Remark 1.1 in Agrawal et al.)
                            
                            numb = int(numb_array[numb_count])
                            count = int(count_array[count_count])
                                                        
                            bench = 1
                            
                            #Check right-hand side conditions
                            ind_one = "not"
                            ind_dyn = "not"
                            
                            if min(b_con/max_a) >= (6 * float(m) * math.log(float(n)/eps)) / (pow(eps, 3)):
                                    ind_one = ""
                                    
                            if min(b_con/max_a) >= (10 * float(m) * math.log(float(n)/eps)) / (pow(eps,2)):
                                    ind_dyn = ""
                            
                            ex = np.zeros(numb)         #Solutions for ex-post algo
                            ex_i = np.zeros(numb)       #Solutions for ex-post integer algo
                            avg_one = np.zeros(numb)    #Avg over all solutions for OTL
                            avg_dyn = np.zeros(numb)    #Avg over all solutions for DL
                            perc_one = np.zeros(numb)   #Pctg of OTL solutions as a fraction of ex-post optimum
                            perc_dyn = np.zeros(numb)   #Pctg of DL solutions as a fraction of ex-post optimum
                            
                            one_res = []                    #Nested list with results of OTL
                            dyn_res = []                    #Nested list with results of DL
                            
                            if bench == 1:
                                greedy_res = []             #Nested list with results of greedy benchmark
                                interval_res = []           #Nested list with results of interval benchmark
                                one_relaxed_res = []        #Nested list with results of OTL-relaxed benchmark
                                dyn_relaxed_res = []        #Nested list with results of DDL-relaxed benchmark
                                wtp_learner_res = []        #Nested list with results of WTP learner benchmark
                                amazon_learner_res = []     #Nested list with results of Amazon learner benchmark
                            
                            one_time_res = []               #Nested list with runtimes of OTL
                            dyn_time_res = []               #Nested list with runtimes of DL
                            perm_time_res =  []             #List with average runtime of each permutation
                            one_resource = []               #Nested list with resource consumption through OTL
                            dyn_resource = []               #Nested list with resource consumption through DL
                            ex_resource = []                #Nested list with resource consumption according to ex-post optimum
                            
                            ###########################################
                            #2 Runtime analysis
                            func_sim_time = np.zeros(numb)      #Time which the simulate function takes for all simulations
                            ex_post_time = np.zeros(numb)       #Time which the Gurobi function takes for all simulations
                            ex_post_time_int = np.zeros(numb)   #Time which the Gurobi_integral function takes for all simulations 
                            
                            if bench == 1:                  #Time which the individual benchmark algorithms take
                                greedy_time = []
                                interval_time = []
                                one_relaxed_time = []
                                dyn_relaxed_time = []
                                wtp_learner_time = []
                                amazon_learner_time = []
                             
                            if bench == 1:              #Pctgs of Benchmarks as fraction of ex-post optimum
                                perc_greedy = np.zeros(numb)
                                perc_interval = np.zeros(numb)
                                perc_one_relaxed = np.zeros(numb)
                                perc_dyn_relaxed = np.zeros(numb)
                                perc_wtp_learner = np.zeros(numb)
                                perc_amazon_learner = np.zeros(numb)
                            
                            one_win = np.zeros(numb)        #Number of times where OTL performs better
                            dyn_win = np.zeros(numb)        #Number of times where DL performs better
                            avg_perm_time = np.zeros(numb)  #Avg permutation time for all permutations
                            avg_one_time = np.zeros(numb)   #Avg time of OTL for all simulations
                            avg_dyn_time = np.zeros(numb)   #Avg time of DL for all simulations
                            
                            sim_start_time = np.zeros(numb) #Simulation start timer
                            sim_end_time = np.zeros(numb)   #Simulation end timer
                            func_perm_time = []             #List of running time of permutations of each simulation
                            
                            
                            
                            ###########################################
                            # One permutation initialization
                            
                            if count == 1:                  
                                one_time_cust = []          #Time which OTL needs for each customer
                                dyn_time_cust = []          #Time which DL needs for each customer
                                wtp = []                    #All incoming willingness to pay
                                wtp_one_ac = []             #Accepted willingness to pay under OTL
                                wtp_dyn_ac = []             #Accepted willingness to pay under DL
                                sp_one = []                 #Shadow prices of OTL
                                sp_dyn = []                 #Shadow prices of DL
                                acc_dem = []                #Nested list with accumulated demands
                            
                            #3 Start simulation    
                            for f in range(numb):           #Simulations loop
                                sim_start_time[f] = time.time()
                                print("\nRun simulation " + str(f + 1) + "...")
                                #Simulate incoming customers
                                func_sim_time[f] = time.time()
                                p, a = simulate(n, m, max_a)
                                func_sim_time[f] = time.time() - func_sim_time[f]
                                
                                if count == 1:
                                    acc_dem_f = []          #List of accumulated demand of one permutation
                                    for i in range(m):
                                        acc_dem_f.append(sum(a[i]))
                                    acc_dem.append(acc_dem_f)
                                
                                with suppress_stdout():                               #Calculates ex-post and ex-post integer Optimum
                                    ex_post_time[f] = time.time()
                                    ex_solution = Gurobi(n, m, p, a, b_con)           #Ex-post optimum remains unchanged over permutations
                                    ex[f] = ex_solution[0]
                                    for i in range(m):
                                        ex_resource.append(sum(ex_solution[1]*a[i]))
                                    ex_post_time[f] = time.time() - ex_post_time[f]
                                    ex_post_time_int[f] = time.time()
                                    ex_i[f] = Gurobi_integral(n, m, p, a, b_con)[0]   #Ex-post optimum remains unchanged over permutations
                                    ex_post_time_int[f] = time.time() - ex_post_time_int[f]
                                    
                                one = np.zeros(count)                                 #np array of OTL solutions
                                dyn = np.zeros(count)                                 #np array of DL solutions
                                
                                if bench == 1:                                        #np array of benchmark solutionss
                                    greedy = np.zeros(count)
                                    interval = np.zeros(count)
                                    one_relaxed = np.zeros(count)
                                    dyn_relaxed = np.zeros(count)
                                    wtp_learner = np.zeros(count)
                                    amazon_learner = np.zeros(count)
                                
                                perm_start_time = np.zeros(count)                     #np array which measures a permuttion's start time
                                perm_end_time = np.zeros(count)                       #np array which measures a permuttion's end time
                                
                                one_time = np.zeros(count)  
                                dyn_time = np.zeros(count)  
                                
                                ###########################################
                                #Runtime analysis
                                perm_time = np.zeros(count)
                                if bench == 1:
                                    gr_time = np.zeros(count) 
                                    inter_time = np.zeros(count) 
                                    one_rel_time = np.zeros(count) 
                                    dyn_rel_time = np.zeros(count) 
                                    wtp_time = np.zeros(count)
                                    am_time = np.zeros(count)
                                    
                                ###########################################
                                for k in range(count):                              #Permutations loop
                                    perm_start_time[k] = time.time()
                                    print("Run permutation " + str(k + 1) + " of simulation " + str(f + 1) + "...")
                                    
                                    ######################################
                                    #Choose random permutation
                                    perm_time[k] = time.time()
                                    p, a = permute(p, a)
                                    perm_time[k] = time.time() - perm_time[k]
                                    if count == 1:
                                        wtp.append(p/sum(a[i] for i in range(m)))   #Willingness to pay per ticket
                                    
                                    ######################################
                                    #One-time Learning Algorithm
                                    with suppress_stdout():                         #Calculates OTL
                                        o = One_Time_Learning(n, m, p, a, b_con, eps)
                                    one[k] = o[0]
                                    one_time[k] = o[2] 
                                    one_resource.append(o[5])       #Resource consumption
                                    if count == 1:
                                        one_time_cust.append(o[3])  #Time per customer
                                        w = []
                                        for j in range(n):
                                            if o[1][j] == 1:
                                                w.append(p[j]/sum(a[i][j] for i in range(m)))
                                        wtp_one_ac.append(w)        #Accepted willingess to pay per ticket
                                        sp_one.append(o[4])         #Shadow prices
                                    
                                    ######################################
                                    #Dynamic pricing algorithm      
                                    with suppress_stdout():         #Calculates DL
                                        d = Dynamic_Learning(n, m, p, a, b_con, eps)
                                    dyn[k] = d[0]
                                    dyn_time[k] = d[2]
                                    dyn_resource.append(d[5])       #Resource consumption
                                    if count == 1:
                                        dyn_time_cust.append(d[3])  #Time per customer
                                        w = []
                                        for j in range(n):
                                            if d[1][j] == 1:
                                                w.append(p[j]/sum(a[i][j] for i in range(m)))
                                        wtp_dyn_ac.append(w)        #Accepted willingess to pay per ticket
                                        sp_dyn.append(d[4])         #Shadow prices
                                    
                                    if dyn[k] > one[k]:             #Counter for which algorithm performs better
                                        dyn_win[f] = dyn_win[f] + 1
                                    elif dyn[k] < one[k]:
                                        one_win[f] = one_win[f] + 1
                                    
                                    ######################################
                                    #Benchmarks
                                    if bench == 1:
                                        with suppress_stdout():     #Calculates benchmark algorithms
                                            time_start = time.time()
                                            greedy[k] = Greedy(n, m, p, a, b_con)[0]
                                            gr_time[k] = time.time() - time_start
                                            
                                            time_start = time.time()
                                            interval[k] = Interval_Learner(n, m, p, a, b_con)[0]
                                            inter_time[k] = time.time() - time_start
                                            
                                            time_start = time.time()
                                            one_relaxed[k] = One_Time_Relaxed(n, m, p, a, b_con, eps)[0]
                                            one_rel_time[k] = time.time() - time_start
                                            
                                            time_start = time.time()
                                            dyn_relaxed[k] = Dynamic_Relaxed(n, m, p, a, b_con, eps)[0]
                                            dyn_rel_time[k] = time.time() - time_start
                                            
                                            time_start = time.time()
                                            wtp_learner[k] = WTP_Learner(n, m, p, a, b_con)[0]
                                            wtp_time[k] = time.time() - time_start
                                            
                                            time_start = time.time()
                                            amazon_learner[k] = Amazon_Learner(n, m, p, a, b_con, eps)[0]
                                            am_time[k] = time.time() - time_start
                                            
                                    perm_end_time[k] = time.time()
                                            
                                avg_one[f] = np.sum(one) / count
                                avg_dyn[f] = np.sum(dyn) / count
                                   
                                avg_perm_time[f] = np.sum(perm_end_time - perm_start_time) / count
                                avg_one_time[f] = np.sum(one_time) / count
                                avg_dyn_time[f] = np.sum(dyn_time) / count
                                
                                one_res.append(one)
                                dyn_res.append(dyn)
                                
                                if bench == 1:  	                   #Append results for benchmarks
                                    greedy_res.append(greedy)
                                    interval_res.append(interval)
                                    one_relaxed_res.append(one_relaxed)
                                    dyn_relaxed_res.append(dyn_relaxed)
                                    wtp_learner_res.append(wtp_learner)
                                    amazon_learner_res.append(amazon_learner)
                                    
                                perc_one[f] = (avg_one[f] / ex[f]) * 100
                                perc_dyn[f] = (avg_dyn[f] / ex[f]) * 100
                                
                                if bench == 1:                      #Calculates performance percentages of benchmarks
                                    perc_greedy[f] = (np.mean(greedy_res[f]/ex[f])) * 100
                                    perc_interval[f] = (np.mean(interval_res[f]/ex[f])) * 100
                                    perc_one_relaxed[f] = (np.mean(one_relaxed_res[f]/ex[f])) * 100
                                    perc_dyn_relaxed[f] = (np.mean(dyn_relaxed_res[f]/ex[f])) * 100
                                    perc_wtp_learner[f] = (np.mean(wtp_learner_res[f]/ex[f])) * 100
                                    perc_amazon_learner[f] = (np.mean(amazon_learner_res[f]/ex[f])) * 100
                                 
                                one_time_res.append(one_time)
                                dyn_time_res.append(dyn_time)
                                perm_time_res.append(perm_end_time - perm_start_time)
                                
                                ###########################################
                                #Runtime analysis
                                func_perm_time.append(perm_time)
                                if bench == 1:
                                    greedy_time.append(gr_time)
                                    interval_time.append(inter_time)
                                    one_relaxed_time.append(one_rel_time)
                                    dyn_relaxed_time.append(dyn_rel_time)
                                    wtp_learner_time.append(wtp_time)
                                    amazon_learner_time.append(am_time) 
                                    ###########################################
                                
                                sim_end_time[f] = time.time()
                            
                            
                            #4 Plotting and Runtime outputs
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
                                    ax.plot(x_ax, 100*one_res[f]/ex[f], 'b')
                                    ax.plot(x_ax, 100*dyn_res[f]/ex[f], 'g')
                                    ax.plot(x_ax, 100*one_res[f]/ex[f], 'bs', label = 'One-Time')
                                    ax.plot(x_ax, 100*dyn_res[f]/ex[f], 'g^', label = 'Dynamic')
                                    plt.xlabel('Permutation')
                                    plt.ylabel('Performance in percent of ex-post optimum')
                                    plt.title('Simulation ' + str(f + 1) + ': Performance of One-Time vs. Dynamic')
                                    plt.axis([0, count+1, 0, 100])
                                    #plt.xticks(np.arange(0, count + 1, step=1))
                                    legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(1.75, 0.5))
                                    plt.show()
                                    
                                    fig, ax = plt.subplots()
                                    ax.plot(x_ax, one_time_res[f], 'b')
                                    ax.plot(x_ax, dyn_time_res[f], 'g')
                                    ax.plot(x_ax, one_time_res[f], 'bs', label='One-Time')
                                    ax.plot(x_ax, dyn_time_res[f], 'g^', label='Dynamic')
                                    plt.xlabel('Permutation')
                                    plt.ylabel('Runtime of algorithm in seconds')
                                    plt.title('Runtime of Algorithms for each Permutation')
                                    plt.axis([0, count+1, 0, max(dyn_time_res[f])*1.25])
                                    #plt.xticks(np.arange(0, count + 1, step=1))
                                    legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(1.75, 0.5))
                                    plt.show()
                                
                                    fig, ax = plt.subplots()
                                    ax.plot(x_ax, perm_time_res[f], 'b')
                                    ax.plot(x_ax, perm_time_res[f], 'bs', label='Runtime of Total Permutation')
                                    plt.xlabel('Permutation')
                                    plt.ylabel('Runtime of total permutation in seconds')
                                    plt.title('Runtime of Total Permutation')
                                    plt.axis([0, count+1, 0, max(perm_time_res[f])*1.25])
                                    #plt.xticks(np.arange(0, count + 1, step=1))
                                    legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(2, 0.5))
                                    plt.show()
                                    
                                    if m <= 5:  #Only plot if not too many resources (otherwise too many plots)
                                        for i in range(m):
                                            fig, ax = plt.subplots()
                                            ax.plot(x_ax, [row[i] for row in one_resource][0 + f*count:count + f*count], 'b')
                                            ax.plot(x_ax, [row[i] for row in one_resource][0 + f*count:count + f*count], 'bs', label='Resource Consumption of\nOne-Time Learning Algorithm')
                                            ax.plot(x_ax, [row[i] for row in dyn_resource][0 + f*count:count + f*count], 'g')
                                            ax.plot(x_ax, [row[i] for row in dyn_resource][0 + f*count:count + f*count], 'g^', label='Resource Consumption of\nDynamic Learning Algorithm')
                                            plt.axhline(y=ex_resource[i + f*m], color = 'y', label = 'Resource Consumption of\nEx-Post Optimal Allocation')
                                            plt.axhline(y=b_con[i], color = 'r', label = 'Maxmimum Capacity')
                                            plt.xlabel('Permutation')
                                            plt.ylabel('Resource consumption in units')
                                            plt.title('Resource Consumption of Resource ' + str(i + 1) + '\nfor Every Permutation')
                                            #plt.axis([0, count+1, 0, b_co*1.25])
                                            #plt.xticks(np.arange(0, count + 1, step=1))
                                            legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(2, 0.5))
                                            plt.show()
                                    
                                    if bench == 1:
                                        fig, ax = plt.subplots()
                                        ax.plot(x_ax, 100*one_res[f]/ex[f], 'b')
                                        ax.plot(x_ax, 100*dyn_res[f]/ex[f], 'g')
                                        ax.plot(x_ax, 100*one_res[f]/ex[f], 'b', marker = 's', label = 'One-Time')
                                        ax.plot(x_ax, 100*dyn_res[f]/ex[f], 'g', marker = '^', label = 'Dynamic')
                                        ax.plot(x_ax, 100*greedy_res[f]/ex[f], 'r')
                                        ax.plot(x_ax, 100*greedy_res[f]/ex[f], 'r', marker = 'o', label = 'Greedy')
                                        ax.plot(x_ax, 100*interval_res[f]/ex[f], 'darkorange')
                                        ax.plot(x_ax, 100*interval_res[f]/ex[f], 'darkorange', marker = 'o', label = 'Interval')
                                        ax.plot(x_ax, 100*one_relaxed_res[f]/ex[f], 'darkviolet')
                                        ax.plot(x_ax, 100*one_relaxed_res[f]/ex[f], 'darkviolet', marker = 'o', label = 'Relaxed One-Time')
                                        ax.plot(x_ax, 100*dyn_relaxed_res[f]/ex[f], 'y')
                                        ax.plot(x_ax, 100*dyn_relaxed_res[f]/ex[f], 'y', marker = 'o', label = 'Relaxed Dynamic')
                                        ax.plot(x_ax, 100*wtp_learner_res[f]/ex[f], 'magenta')
                                        ax.plot(x_ax, 100*wtp_learner_res[f]/ex[f], 'magenta', marker = 'o', label = 'WTP-Learner')
                                        ax.plot(x_ax, 100*amazon_learner_res[f]/ex[f], 'brown')
                                        ax.plot(x_ax, 100*amazon_learner_res[f]/ex[f], 'brown', marker = 'o', label = 'Amazon Learner')
                                        plt.xlabel('Permutation')
                                        plt.ylabel('Performance in percent of ex-post optimum')
                                        plt.title('Simulation ' + str(f + 1) + ': Performance of All Benchmarks')
                                        plt.axis([0, count+1, 0, 100])
                                        legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(1.75, 0.5))
                                        plt.show() 
                                    
                                        fig, ax = plt.subplots()
                                        ax.bar(1, 100*np.mean(one_res[f]/ex[f]), width = 0.8, color = 'b', align = 'center', label = 'One-Time')
                                        ax.bar(2, 100*np.mean(dyn_res[f]/ex[f]), width = 0.8, color = 'g', align = 'center', label = 'Dynamic')
                                        ax.bar(3, 100*np.mean(greedy_res[f]/ex[f]), width = 0.8, color = 'r', align = 'center', label = 'Greedy')
                                        ax.bar(4, 100*np.mean(interval_res[f]/ex[f]), width = 0.8, color = 'darkorange', align = 'center', label = 'Interval')
                                        ax.bar(5, 100*np.mean(one_relaxed_res[f]/ex[f]), width = 0.8, color = 'darkviolet', align = 'center', label = 'Relaxed One-Time')
                                        ax.bar(6, 100*np.mean(dyn_relaxed_res[f]/ex[f]), width = 0.8, color = 'y', align = 'center', label = 'Relaxed Dynamic')
                                        ax.bar(7, 100*np.mean(wtp_learner_res[f]/ex[f]), width = 0.8, color = 'magenta', align = 'center', label = 'WTP-Learner')
                                        ax.bar(8, 100*np.mean(amazon_learner_res[f]/ex[f]), width = 0.8, color = 'brown', align = 'center', label = 'Amazon Learner')
                                        plt.ylabel('Average performance in percent of ex-post optimum')
                                        plt.title('Average performance of all benchmarks over all permutations')
                                        #plt.xticks(np.arange(1, m + 1, step=1))
                                        legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(2, 0.5))
                                        plt.show()
                                        
                                    ###########################################
                                    #Runtime analysis    
                                    print("\n\nRuntime Breakdown:")
                                    print("\nSimulation: --- %s seconds ---" % round(sim_end_time[f] - sim_start_time[f], 5))
                                    print("Average of permutations: --- %s seconds ---" % round(avg_perm_time[f], 5))
                                    print("Simulating the input parameters: --- %s seconds ---" % round(func_sim_time[f], 5))
                                    print("Calculating the fractional ex-post solution: --- %s seconds ---" % round(ex_post_time[f], 5))
                                    print("Calculating the integral ex-post solution: --- %s seconds ---" % round(ex_post_time_int[f], 5))
                                    print("Average creating of permutations: --- %s seconds ---" % round(np.mean(func_perm_time[f]), 5))
                                    print("Average One-Time Learning Algorithm: --- %s seconds ---" % round(np.mean(one_time_res[f]), 5))
                                    print("Average Dynamic Learning Algorithm: --- %s seconds ---" % round(np.mean(dyn_time_res[f]), 5))
                                    if bench == 1:
                                        print("Average Greedy Algorithm: --- %s seconds ---" % round(np.mean(greedy_time[f]), 5))
                                        print("Average Interval Learner: --- %s seconds ---" % round(np.mean(interval_time[f]), 5))
                                        print("Average Relaxed One-Time Learning Algorithm: --- %s seconds ---" % round(np.mean(one_relaxed_time[f]), 5))
                                        print("Average Relaxed Dynamic Learning Algorithm: --- %s seconds ---" % round(np.mean(dyn_relaxed_time[f]), 5))
                                        print("Average WTP-Learner: --- %s seconds ---" % round(np.mean(wtp_learner_time[f]), 5))
                                        print("Average Amazon Learner: --- %s seconds ---" % round(np.mean(amazon_learner_time[f]), 5))
                                    ###########################################
                                
                                
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
                                    
                                    if bench == 1:
                                        fig, ax = plt.subplots()
                                        ax.bar(1, 100*one_res[f]/ex[f], width = 0.8, color = 'b', align = 'center', label = 'One-Time')
                                        ax.bar(2, 100*dyn_res[f]/ex[f], width = 0.8, color = 'g', align = 'center', label = 'Dynamic')
                                        ax.bar(3, 100*greedy_res[f]/ex[f], width = 0.8, color = 'r', align = 'center', label = 'Greedy')
                                        ax.bar(4, 100*interval_res[f]/ex[f], width = 0.8, color = 'darkorange', align = 'center', label = 'Interval')
                                        ax.bar(5, 100*one_relaxed_res[f]/ex[f], width = 0.8, color = 'darkviolet', align = 'center', label = 'Relaxed One-Time')
                                        ax.bar(6, 100*dyn_relaxed_res[f]/ex[f], width = 0.8, color = 'y', align = 'center', label = 'Relaxed Dynamic')
                                        ax.bar(7, 100*wtp_learner_res[f]/ex[f], width = 0.8, color = 'magenta', align = 'center', label = 'WTP-Learner')
                                        ax.bar(8, 100*amazon_learner_res[f]/ex[f], width = 0.8, color = 'brown', align = 'center', label = 'Amazon Learner')
                                        plt.ylabel('Average performance in percent of ex-post optimum')
                                        plt.title('Performance of all benchmarks in the Single Permutation')
                                        legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(2, 0.5))
                                        plt.show()
                                    
                                    ###########################################
                                    #Runtime analysis    
                                    print("\n\nRuntime Breakdown:")
                                    print("\nSimulation: --- %s seconds ---" % round(sim_end_time[f] - sim_start_time[f], 5))
                                    print("Single permutation: --- %s seconds ---" % round(avg_perm_time[f], 5))
                                    print("Simulating the input parameters: --- %s seconds ---" % round(func_sim_time[f], 5))
                                    print("Calculating the fractional ex-post solution: --- %s seconds ---" % round(ex_post_time[f], 5))
                                    print("Calculating the integral ex-post solution: --- %s seconds ---" % round(ex_post_time_int[f], 5))
                                    print("Creating the single permutation: --- %s seconds ---" % round(np.sum(func_perm_time[f]), 5))
                                    print("One-Time Learning Algorithm: --- %s seconds ---" % round(np.sum(one_time_res[f]), 5))
                                    print("Dynamic Learning Algorithm: --- %s seconds ---" % round(np.sum(dyn_time_res[f]), 5))
                                    if bench == 1:
                                        print("Greedy Algorithm: --- %s seconds ---" % round(np.sum(greedy_time[f]), 5))
                                        print("Interval Learner: --- %s seconds ---" % round(np.sum(interval_time[f]), 5))
                                        print("Relaxed One-Time Learning Algorithm: --- %s seconds ---" % round(np.sum(one_relaxed_time[f]), 5))
                                        print("Relaxed Dynamic Learning Algorithm: --- %s seconds ---" % round(np.sum(dyn_relaxed_time[f]), 5))
                                        print("WTP-Learner: --- %s seconds ---" % round(np.sum(wtp_learner_time[f]), 5))
                                        print("Amazon Learner: --- %s seconds ---" % round(np.sum(amazon_learner_time[f]), 5))
                                    ###########################################
                            
                            
                            print("\n\n--------------------------------------------------\nThe right-hand side condition was " + ind_one + " satisfied for the One-Time Learning Algorithm.")
                            print("The right-hand side condition was " + ind_dyn + " satisfied for the Dynamic Learning Algorithm.")
                            
                            if numb > 1:
                                x_ax = np.arange(numb) + 1
                                fig, ax = plt.subplots()
                                ax.plot(x_ax, perc_one, 'b')
                                ax.plot(x_ax, perc_dyn, 'g')
                                ax.plot(x_ax, perc_one, 'bs', label='One-Time')
                                ax.plot(x_ax, perc_dyn, 'g^', label='Dynamic')
                                plt.xlabel('Simulation')
                                plt.ylabel('Average Performance in percent of ex-post optimum')
                                plt.title('Average Performance: One-Time vs. Dynamic')
                                plt.axis([0, numb+1, 0, 100])
                                #plt.xticks(np.arange(0, numb + 1, step=1))
                                legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(1.75, 0.5))
                                plt.show()
                                
                                fig, ax = plt.subplots()
                                ax.plot(x_ax, sim_end_time - sim_start_time, 'b')
                                ax.plot(x_ax, sim_end_time - sim_start_time, 'bs', label='Total Runtime')
                                plt.xlabel('Simulation')
                                plt.ylabel('Runtime of simulation in seconds')
                                plt.title('Runtime of Total Simulation')
                                plt.axis([0, numb+1, 0, max(sim_end_time - sim_start_time)*1.25])
                                #plt.xticks(np.arange(0, numb + 1, step=1))
                                legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(1.75, 0.5))
                                plt.show()
                                
                                fig, ax = plt.subplots()
                                ax.plot(x_ax, avg_one_time, 'b')
                                ax.plot(x_ax, avg_one_time, 'bs', label='Average runtime of\nOne-Time Learning Algorithm in seconds')
                                ax.plot(x_ax, avg_dyn_time, 'g')
                                ax.plot(x_ax, avg_dyn_time, 'g^', label='Average runtime of\nDynamic Learning Algorithm in seconds')
                                plt.xlabel('Simulation')
                                plt.ylabel('Average runtime of algorithms in seconds')
                                plt.title('Average Runtime of Algorithms over all Permutations')
                                plt.axis([0, numb+1, 0, max(avg_dyn_time)*1.25])
                                #plt.xticks(np.arange(0, numb + 1, step=1))
                                legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(2.5, 0.5))
                                plt.show()
                                
                                fig, ax = plt.subplots()
                                ax.plot(x_ax, avg_perm_time, 'b')
                                ax.plot(x_ax, avg_perm_time, 'bs', label='Average runtime of a permutation in seconds')
                                plt.xlabel('Simulation')
                                plt.ylabel('Average runtime of a permutation in seconds')
                                plt.title('Average Runtime of Total Permutation')
                                plt.axis([0, numb+1, 0, max(avg_perm_time)*1.25])
                                #plt.xticks(np.arange(0, numb + 1, step=1))
                                legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(2.5, 0.5))
                                plt.show()
                                
                                if bench == 1:
                                    x_ax = np.arange(numb) + 1
                                    fig, ax = plt.subplots()
                                    ax.plot(x_ax, perc_one, 'b')
                                    ax.plot(x_ax, perc_one, 'b', marker = 's', label='One-Time')
                                    ax.plot(x_ax, perc_dyn, 'g')
                                    ax.plot(x_ax, perc_dyn, 'g', marker = '^', label='Dynamic')
                                    ax.plot(x_ax, perc_greedy, 'r')
                                    ax.plot(x_ax, perc_greedy, 'r', marker = 'o', label='Greedy')
                                    ax.plot(x_ax, perc_interval, 'darkorange')
                                    ax.plot(x_ax, perc_interval, 'darkorange', marker = 'o', label='Interval')
                                    ax.plot(x_ax, perc_one_relaxed, 'darkviolet')
                                    ax.plot(x_ax, perc_one_relaxed, 'darkviolet', marker = 'o', label='Relaxed One-Time')
                                    ax.plot(x_ax, perc_dyn_relaxed, 'y')
                                    ax.plot(x_ax, perc_dyn_relaxed, 'y', marker = 'o', label='Relaxed Dynamic')
                                    ax.plot(x_ax, perc_wtp_learner, 'magenta')
                                    ax.plot(x_ax, perc_wtp_learner, 'magenta', marker = 'o', label='WTP-Learner')
                                    ax.plot(x_ax, perc_amazon_learner, 'brown')
                                    ax.plot(x_ax, perc_amazon_learner, 'brown', marker = 'o', label='Amazon Learner')
                                    plt.xlabel('Simulation')
                                    plt.ylabel('Average Performance in percent of ex-post optimum')
                                    plt.title('Average Performance of All Benchmarks')
                                    plt.axis([0, numb+1, 0, 100])
                                    #plt.xticks(np.arange(0, numb + 1, step=1))
                                    legend = ax.legend(loc='right', shadow=True, fontsize='x-large', bbox_to_anchor=(1.75, 0.5))
                                    plt.show()

                            
                            for f in range(numb):
                                myData.append([n, m, max_a, b[0], eps, numb, count, ex[f], np.mean(one_res[f]), np.mean(dyn_res[f]), np.mean(greedy_res[f]), np.mean(interval_res[f]), np.mean(one_relaxed_res[f]), np.mean(dyn_relaxed_res[f]), np.mean(wtp_learner_res[f]), np.mean(amazon_learner_res[f]), (sim_end_time-sim_start_time)[f], np.mean(perm_end_time - perm_start_time), func_sim_time[f], ex_post_time[f], ex_post_time_int[f], np.mean(func_perm_time[f]), np.mean(one_time_res[f]), np.mean(dyn_time_res[f]), np.mean(greedy_time[f]), np.mean(interval_time[f]), np.mean(one_relaxed_time[f]), np.mean(dyn_relaxed_time[f]), np.mean(wtp_learner_time[f]), np.mean(amazon_learner_time[f])])
                                #Not included: a, p, sp_dyn, b[1], b[2], ex_resource, one_resource, dyn_resource
myFile = open('VaryingM2.csv', 'w')
csv.register_dialect('myDialect', delimiter=';', quoting=csv.QUOTE_NONE)
with myFile:  
   writer = csv.writer(myFile, dialect='myDialect')
   writer.writerows(myData) 
print("\n\nAuto-Simulation finished.\n\n")