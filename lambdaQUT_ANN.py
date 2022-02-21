#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from numpy import *
from sys import argv

p1_grid = [16,32,64,128,256,512]
p2_grid = [16,32,64,128,256,512]


# In[3]:


# generate synthetic data

n=100#shape(load('inputX-'+str(p1_grid[0])+'.npy'))[0]            # number of samples
eta = 1           # sample noise (std)

p1=10              # number of the first layer neurons (=number of input features)
p2=10             # number of the second layer neurons

h=2               # hidden number of features?
mu = 10.0         # final additive constant
xi = 0.1          # observatio noise (std)
eta = 1           # noise of the input vector (std)


# In[4]:


def activation(x):
    M = 20
    return (1/M)*(tf.nn.softplus(M*x) - log(2))

# In[5]:


def lambdaQUT(inputx,nSample=1000,miniBatchSize=1,alpha=0.05,option='quantile'):
    
    print('scanning p1='+str(p1)+', p2='+str(p2), flush=True)
    
    # option can be 'quantile' which returns a single real value of he quantile (with the specified alpha)
    # or it can be 'full' which returns the whole list of the maximum absolute gradients
    
    if mod(nSample, miniBatchSize)==0:
        offset=0
    else:
        offset=1
    
    fullList = zeros((miniBatchSize*(nSample//miniBatchSize+offset),))
    normlist = []
    w1 = tf.Variable(zeros((p1,p2)), trainable = False)
    b1 = zeros((1,p2))
    
    for index in range(nSample//miniBatchSize+offset):

        # loc and scale could be anything since the statistics is pivotal
        ySample = random.normal(loc=0., scale=eta, size=(n, miniBatchSize))  
        
        s = zeros(p1)
        for i in range(n):
            s += (ySample[i] - mean(ySample))*inputx[i]
        lam = (1/(2*linalg.norm(ySample - mean(ySample), 2)))*linalg.norm(s, inf)
        normlist.append(lam)
        
        w2 = random.normal(loc=0., scale=eta, size=(p2,miniBatchSize))
        w2_L2Norm = sqrt(sum(w2*w2, axis=0))
        
        # b2 is not defined in this function
        # instead, it was replaced by the mean of ySample

        with tf.GradientTape() as g:

            g.watch(w1)
            yhat = mean(ySample, axis=0) + tf.matmul(activation(tf.matmul(inputx, w1)+b1),w2/w2_L2Norm)
            cost = tf.sqrt(tf.reduce_sum(tf.square(yhat-ySample), axis=0))

        gradients = reshape(g.jacobian(cost,w1).numpy(), [miniBatchSize, p1*p2])
        
        stat = amax(abs(gradients), axis=1)         

        fullList[index*miniBatchSize:(index+1)*miniBatchSize]=stat[:]
        
        
 
    if option=='full':
        import matplotlib as plt
        plt.pyplot.hist(fullList, bins = 20)
        plt.pyplot.hist(normlist, bins = 20)
        return fullList, normlist
    elif option=='quantile':
        return quantile(fullList, 1-alpha)
    else:
        pass

x = random.normal(size = (n, p1))
tflistn, nlist = lambdaQUT(x, nSample = 1000, miniBatchSize = 1, alpha = 0.05, option = 'full')
print(quantile(tflistn, 0.95))
print(quantile(nlist, 0.95))

# In[6]:

'''
import time 

data = zeros((len(p1_grid)*len(p2_grid),2+100000))

counter = -1

for ii in range(len(p1_grid)):
    for jj in range(len(p2_grid)):
        
        p1 = p1_grid[ii]
        p2 = p2_grid[jj]
        
        counter += 1
        
        # input matrix, n\times p1
        X = load('inputOutputFiles/inputX-'+str(p1)+'-'+argv[1]+'.npy')
        
        start_time = time.time()
        
        data[counter,0] = p1
        data[counter,1] = p2
        data[counter,2:] = lambdaQUT(X,nSample=100000,option='full')
                         
        print("--- %s seconds ---" % (time.time() - start_time), flush=True)
                
        save('inputOutputFiles/LambdaQUT-'+argv[1], data)

'''



