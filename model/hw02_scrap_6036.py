#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

6.036 HW2
Created on Fri Sep 11 21:29:29 2020

@author: gliu
"""



import numpy as np



# Perceptron algorithm

def perceptron_origin(dims,tau,data):
    
    theta = np.zeros(dims)
    thetaall  = []
    mistakes = 0
    for t in range(tau):
        changed = False
        
        for i in range(len(data)):
            x,y = data[i] # Get Datapoint and label
            if theta.T@x*y <= 0 :
                theta = theta + y*x
                thetaall.append(theta)
                changed = True
                mistakes += 1
            
        if changed is False:
            break
        
    return theta, mistakes,thetaall



dims = 2
tau  = 999


# Question 2.2
points = np.array([[1,-1],[0,1],[-10,-1]])
labels = np.array([1,-1,1])
data = np.array(list(zip(points,labels)))
th,mstk,thall= perceptron_origin(dims,tau,data)
print(mstk)
print(thall)


# Question 2.3
points = np.array([[0,1],[-10,-1],[1,-1]])
labels = np.array([-1,1,1])
data = np.array(list(zip(points,labels)))
th,mstk,thall= perceptron_origin(dims,tau,data)
print(mstk)
print(thall)


# Question 7 ------- MY ANSWER

data = np.array([[2, 3, 9, 12],[5, 2, 6, 5]])
labels = np.array([[1, -1, 1, -1]])

import matplotlib.pyplot as plt

def hook2d(th,th0,data,labels,mistakes):
    
    
    
    fig,ax=plt.subplots(1,1,figsize=(4,3))
    
    # Idx positive and negative labels
    idpos = np.where(labels[0,:]>0)
    idneg = np.where(labels[0,:]<=0)    
    
    # Plot points
    ax.scatter(data[0,idpos],data[1,idpos],color='red',label="+1")
    ax.scatter(data[0,idneg],data[1,idneg],color='blue',label="-1")
    
    
    # Plot Classifier
    m = -th[0]/th[1]
    b = -th0/th[1]
    
    plotx = np.arange(np.min(data[0,:])-5,np.max(data[0,:])+5+1,1)
    h = m*plotx-b
    
    ax.plot(plotx,h,color='k',label=r"$\theta= %s$,  $\theta_{0}=%s$" %(str(th),str(th0)))
    ax.legend(fontsize=8)
    ax.set_title("Mistakes = %i"% (mistakes))
    
    

def perceptron(data,labels,params={},hook=None):
    
    T = params.get('T',100)
    
    d   = len(data) # Dimensionality of theta
    n   = len(labels[0,:]) # Number of points
    
    # Initiallize Classifier
    th = np.zeros(d) 
    th0 = 0 # 
    mistakes = 0
    for t in range(T): # Loop for T iterations
    
        changed = False
    
        for i in range(n):# Loop through each pair
            
            x = data[:,i] # Get Pair
            y = labels[0,i] # Get label
            
            if y*(th.T@x+th0) <= 0:
                
                th  += y*x
                th0 += y
                
                changed = True
                mistakes +=1
                print(mistakes)
                
                if hook is not None:
                    hook2d(th,th0,data,labels,mistakes)
                
        if changed is False:
            break
        
    return np.array([th]).T,np.array([[th0,]])
            

perceptron(data,labels,{'T':100},hook=True)
    

# %%QUESTION 7 (COURSE ANSWER)
# x is dimension d by 1
# th is dimension d by 1
# th0 is dimension 1 by 1
# return 1 by 1 matrix of +1, 0, -1
def positive(x, th, th0):
   return np.sign(th.T@x + th0)

# Perceptron algorithm with offset.
# data is dimension d by n
# labels is dimension 1 by n
# T is a positive integer number of steps to run
# Perceptron algorithm with offset.
# data is dimension d by n
# labels is dimension 1 by n
# T is a positive integer number of steps to run
def perceptron(data, labels, params = {}, hook = None):
    # if T not in params, default to 100
    T = params.get('T', 100)
    (d, n) = data.shape

    theta = np.zeros((d, 1))
    theta_0 = np.zeros((1, 1))
    
    
    for t in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if y * positive(x, theta, theta_0) <= 0.0:
                theta = theta + y * x
                theta_0 = theta_0 + y
                if hook:
                    hook((theta, theta_0))
    return theta, theta_0
# Note, the solution doesn't have to be pretty; it's far better that it is understandable.



#%%  Question 9 
def eval_classifier(learner, data_train, labels_train, data_test, labels_test):

    th,th0 = learner(data_train,labels_train,{'T':100})
    sc = score(data_test,labels_test,th,th0)
    sc = sc/data_test.shape[1]

    # Your implementation here
    return sc
ans=eval_classifier(perceptron, np.hstack([data1,data2]), np.hstack([labels1,labels2]), data2, labels2)




#%% Question 9
data = np.tile(np.arange(0,71,1)[:,None],2).T



k = 12
d,n = data.shape
labels = np.ones((1,n))

D_i = np.array_split(data,k,axis=1)
L_i = np.array_split(labels,k,axis=1)

totalscore = 0
for i in range(k-1): #

    # Separate test and training data
    data_train   = D_i[0:i] + D_i[i+1:]
    data_test    = D_i[i]
    labels_train = L_i[0:i] + L_i[i+1:]
    labels_test  = L_i[i]
    
    
    # Score Learning Algorithm
    totalscore += eval_classifier(learner, data_train, labels_train, data_test, labels_test)
return totalscore/k
    
    



D_i = dsplit[1:k]
D_j = dsplit[-1]


label_i = lsplit[1:k]
label_j = lsplit[-1]