# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 08:28:42 2018

@author: Johannes
"""

import numpy as np
import math

epsilon = np.arange(0, 0.167, 0.001) + 0.001 
min_b = np.arange(5000, 80000, 5000) #80000
max_a = np.arange(0, 10, 1) + 1  #10
m_res = np.arange(0, 10, 1) + 1  #10
count = 1

for eps in range(len(epsilon)):
    #print("\n\n---------------------------------------\nRun epsilon = " + str(epsilon[eps]))
    for b in range(len(min_b)):
        #print("Run min_b = " + str(min_b[b]))
        for a in range(len(max_a)):
            #print("Run a = " + str(max_a[a]))
            for m in range(len(m_res)):
                #print("m: " + str(m_res[m]))                
                max_n_one = math.floor(epsilon[eps]*math.exp(((min_b[b]/max_a[a])*math.pow(epsilon[eps], 3))/(6*m_res[m])))    
                max_n_dyn = math.floor(epsilon[eps]*math.exp(((min_b[b]/max_a[a])*math.pow(epsilon[eps], 2))/(10*m_res[m])))   
                if max_n_dyn > m_res[m] * min_b[b] and max_n_one > m_res[m] * min_b[b]:
                    if m_res[m] * min_b[b] < 100000:
                        if m_res[m] > 1:
                            if max_a[a] > 0:
                                print("\nKandidat No. " + str(count))
                                print("Epsilon: " + str(epsilon[eps]))
                                print("Min_b: " + str(min_b[b]))
                                print("Max_a: " + str(max_a[a]))
                                print("Resources m: " + str(m_res[m]))
                                print("Max. customers n one-time: " + str(max_n_one))
                                print("Max. customers n dynamic: " + str(max_n_dyn))
                                count = count + 1
if count == 1:
    print("\nNo candidates found.")
else: print("\nTotal number of candidates: " + str(count-1))