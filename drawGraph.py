#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 20:27:08 2020

@author: kubota
"""

import matplotlib.pyplot as plt 
import numpy as np

# line 1 points 
x1 = ['2012-01-08','2012-01-15','2012-01-22', '2012-02-12', '2012-02-19', '2012-04-29', '2012-06-15', '2013-01-10', '2013-04-05'] 

y1 = [0.3590, 0.3079, 0.3367, 0.2760, 0.3064, 0.2796, 0.3184, 0.3996, 0.3912] 
# plotting the line 1 points 
plt.plot(x1, y1, label = "TQE") 

plt.xlabel('Datasets') 
# naming the y axis 
plt.ylabel('TQE') 
plt.ylim(0, 1.0)
# giving a title to my graph 
plt.title('Total Quantization Error (TQE)') 

plt.xticks(rotation=90)

# show a legend on the plot 
plt.legend() 

# function to show the plot 
plt.show() 

# line 2 points 
x2 = ['2012-01-08','2012-01-15','2012-01-22', '2012-02-12', '2012-02-19', '2012-04-29', '2012-06-15', '2013-01-10', '2013-04-05'] 
y2 = [204070/754, 242310/837, 189590/821, 186310/723, 193570/662, 93760/440, 118060/841, 149750/425, 36770/555] 
# plotting the line 2 points 
plt.plot(x2, y2, label = "CR") 

# naming the x axis 
plt.xlabel('Datasets') 
# naming the y axis 
plt.ylabel('CR') 
plt.ylim(0, 500)
# giving a title to my graph 
plt.title('Compression Ratio (CR)') 

plt.xticks(rotation=90)

# show a legend on the plot 
plt.legend() 

# function to show the plot 
plt.show() 

#NCLT
# line 1 points 
x1 = ['2012-01-08','2012-01-15','2012-01-22', '2012-02-12', '2012-02-19', '2012-04-29', '2012-06-15', '2013-01-10', '2013-04-05'] 

y1 = [0.9692, 0.9360, 0.9066, 0.9658, 0.8606, 0.9575, 0.9136, 0.8830,  0.9608] 
# plotting the line 1 points 
plt.plot(x1, y1, label = "Accuracy") 

# line 2 points 
x2 = ['2012-01-08','2012-01-15','2012-01-22', '2012-02-12', '2012-02-19', '2012-04-29', '2012-06-15', '2013-01-10', '2013-04-05'] 

y2 = [0.3590, 0.3079, 0.3367, 0.2760, 0.3064, 0.2796, 0.3184, 0.3996, 0.3912] 
# plotting the line 1 points 
plt.plot(x2, y2, label = "TQE") 

plt.xlabel('Datasets') 
# naming the y axis 
plt.ylabel('TQE and Accuracy') 
#plt.ylim(0, 100)
# giving a title to my graph 
plt.title('TQE and Accuracy') 

plt.xticks(rotation=90)

# show a legend on the plot 
plt.legend() 

# function to show the plot 
plt.show() 

#TMU TQE
# line 1 points 
x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 

y1 = [0.2507, 0.1733, 0.1556, 0.1680, 0.1729, 0.1752, 0.1578, 0.1509, 0.1537, 0.1630] 
# plotting the line 1 points 
plt.plot(x1, y1, label = "Episodic Memory Network") 

# line 2 points 
x2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 

y2 = [0.2741, 0.1651, 0.1537, 0.1646, 0.1721, 0.1438, 0.1426, 0.1733, 0.1786, 0.1750] 
# plotting the line 1 points 
plt.plot(x2, y2, label = "Semantic Memory Network") 

plt.xlabel('Datasets') 
# naming the y axis 
plt.ylabel('TQE') 
plt.ylim(0, 0.5)
plt.xticks(np.arange(0, 11, step=1))
#plt.ylim(0, 100)
# giving a title to my graph 
plt.title('TQE') 

#plt.xticks(rotation=90)

# show a legend on the plot 
plt.legend() 

# function to show the plot 
plt.show() 

# TMU Accuracy
# line 1 points 
x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 

y1 = [0.9046*100, 0.91733*100, 0.91556*100, 0.91680*100, 0.91729*100, 0.92752*100, 0.93578*100, 0.90509*100, 0.8937*100, 0.91630*100] 
# plotting the line 1 points 
plt.plot(x1, y1, label = "MC-GWR") 

# line 2 points 
x2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 

y2 = [0.6498*100, 0.6519*100, 0.6567*100, 0.6449*100,  0.6525*100,  0.6434*100, 0.6551*100, 0.6434*100, 0.6472*100,  0.6411*100] 
# plotting the line 1 points 
plt.plot(x2, y2, label = "Gamma-GWR") 

plt.xlabel('Datasets') 
# naming the y axis 
plt.ylabel('Accuracy %') 
plt.ylim(50, 100)
plt.xticks(np.arange(0, 11, step=1))
#plt.ylim(0, 100)
# giving a title to my graph 
plt.title('Accuracy of Semantic Memory Network') 

#plt.xticks(rotation=90)

# show a legend on the plot 
plt.legend() 

# function to show the plot 
plt.show() 
