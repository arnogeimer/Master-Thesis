import tensorflow as tf
from numpy import *
import numpy as np
import random
import matplotlib as plt
import DataCreate as dc

#### MODEL FUNCTIONS

def relu(x):
    return tf.nn.relu(x)

def leakyrelu(x):
    return tf.nn.leaky_relu(x)

def softplus(x):
    M = 20
    return (1/M)*(tf.nn.softplus(M*x) - np.log(2))

def LASSO(Yhat, Y, weight, lamb):
    return 0.5*tf.sqrt(tf.reduce_sum(tf.square(Yhat - Y), axis=0)) + lamb * tf.reduce_sum(tf.abs(weight), axis=0)

def extracttheta(w1):
    p1 = w1.shape[0]
    thetahat = np.zeros(p1)
    for i in range(p1):
        thetahat[i] = (w1[i,2*i] - w1[i,2*i+1])/2
    return thetahat

#### MODEL PARAMETERS

def soft_tresh(x, thresh):
    
    if x <= -thresh:
        return x + thresh
    if x >= thresh:
        return x - thresh
    else:
        return 0

def hard_tresh(x, thresh):
    
    if x <= -thresh:
        return x
    if x >= thresh:
        return x
    else:
        return 0
    
def check_entries(xtrue, y):
    
    if len(xtrue) != len(y):
        raise ValueError('Not the same lengths')
    
    result = True
    
    for i in range(len(xtrue)):
        if xtrue [i] != 0 and y[i] != 0:
            pass
        if xtrue [i] != 0 and y[i] == 0:
            result = False
    return result

iterations = 100
maxIterN = 1000

# Defining the TF model

def TFmodel(Xsample,Ysample,activation,lamb,learningRate,thetavalue,sparsity_indexes,oracle,ifTrain=True):
    
    n, p1 = Xsample.shape
    p2 = 2*p1
    
    if ifTrain:
        loss = np.zeros(maxIterN)
    else:
        loss = None
    
    Ysample = Ysample.reshape(n,1)
        
    b1_ini = np.random.normal(loc=0., scale=1, size=(1, p2))
    w2_ini = np.random.normal(loc=0., scale=0.05, size=(p2,1))
    b2_ini = np.random.normal(loc=0., scale=1, size=(1,1))
    
    if not oracle:

        # if not Oracle, we shall regenerate the parameters randomly
        w1_ini = np.random.uniform(0., 10, size=(p1,p2))
        
    if oracle:
        w1_ini = np.zeros((p1,p2))
        for i in sparsity_indexes:
            w1_ini[i, 2*i] = thetavalue
            w1_ini[i, 2*i+1] = -thetavalue
            
    w1 = tf.Variable(w1_ini ,shape=(p1,p2), trainable=True)
    b1 = tf.Variable(b1_ini ,shape=(1,p2), trainable=True)
    w2 = tf.Variable(w2_ini ,shape=(p2,1), trainable=True)
    b2 = tf.Variable(b2_ini ,shape=(1,1), trainable=True)
    
    
    def evaluate(examx, examy, ifUpdate=True, ifISTA=False):

        # new structure in TF2.0 to track the parts of computation that will need backprop
        with tf.GradientTape(persistent=True) as g:

            # these are the parameters of which we will need the gradients
            g.watch([w1,b1,w2,b2])

            # the structure is used muptiple times, we should just compute once X \cdot w_1
            Xw1 = tf.matmul(examx, w1)
            
            w2_L2Norm = tf.sqrt(tf.reduce_sum(w2*w2, axis=0))
            
            yhat = b2 + tf.matmul(activation(Xw1+b1), w2/w2_L2Norm)
            
            bareCost = tf.sqrt(tf.reduce_sum(tf.square(yhat-examy), axis=0)) # sqrt Lasso
            #bareCost = tf.reduce_sum(tf.square(yhat-examy), axis=0)
            cost = bareCost + lamb*tf.reduce_sum(tf.abs(w1))

        # computing cost gradients
        w1_gra          = g.gradient(cost, w1)
        b1_gra          = g.gradient(cost, b1)
        w2_gra          = g.gradient(cost, w2)
        b2_gra          = g.gradient(cost, b2)

        bare_w1_gra     = g.gradient(bareCost, w1)
        bare_b1_gra     = g.gradient(bareCost, b1)
        bare_w2_gra     = g.gradient(bareCost, w2)
        bare_b2_gra     = g.gradient(bareCost, b2)

        del g

        if (ifUpdate):

            if (ifISTA):

                backtrackingFlag = True
                counter = 0
                shrinkageFactor = 0.95

                toSaveFlag = True

                while backtrackingFlag:

                    effLearningRate = learningRate*(shrinkageFactor**counter)

                    argGreater = w1 - bare_w1_gra*learningRate > lamb*learningRate
                    argSmaller = w1 - bare_w1_gra*learningRate < -lamb*learningRate


                    # update w1
                    proposed_w1 = tf.where(argGreater, w1 - bare_w1_gra*learningRate - lamb*learningRate, 0)
                    proposed_w1 = tf.where(argSmaller, w1 - bare_w1_gra*learningRate + lamb*learningRate, proposed_w1)

                    # update b1 with constraints
                    # first we compute the bounding envolope

                    maxXw1 = tf.reduce_max(Xw1, axis=0)
                    minXw1 = tf.reduce_min(Xw1, axis=0)

                    # proposition
                    proposed_b1 = b1 - effLearningRate*b1_gra

                    # Booleans that determines if outside the bounds
                    ifBelow = (proposed_b1 < minXw1)
                    ifAbove = (proposed_b1 > maxXw1)

                    # If outside, take the bounds
                    proposed_b1 = tf.where( ifBelow, minXw1, proposed_b1)
                    proposed_b1 = tf.where( ifAbove, maxXw1, proposed_b1)

                    # update w2
                    proposed_w2 = w2 - effLearningRate*w2_gra

                    # update b2
                    proposed_b2 = b2 - effLearningRate*b2_gra

                    # extract information for backtracking step
                    w1_atProposition = proposed_w1.numpy()
                    b1_atProposition = proposed_b1.numpy()
                    w2_atProposition = proposed_w2.numpy()
                    b2_atProposition = proposed_b2.numpy()

                    diff_w1 = (w1.numpy()-w1_atProposition)/effLearningRate
                    diff_b1 = (b1.numpy()-b1_atProposition)/effLearningRate
                    diff_w2 = (w2.numpy()-w2_atProposition)/effLearningRate
                    diff_b2 = (b2.numpy()-b2_atProposition)/effLearningRate

                    w2_L2Norm_atProposition = sqrt(sum(w2_atProposition*w2_atProposition, axis=0))
                    w2_normalized_atProposition = w2_atProposition/w2_L2Norm_atProposition

                    w1_gra_v = bare_w1_gra.numpy()
                    b1_gra_v = bare_b1_gra.numpy()
                    w2_gra_v = bare_w2_gra.numpy()
                    b2_gra_v = bare_b2_gra.numpy()

                    yhat_atProposition = matmul( activation( matmul(examx, w1_atProposition) + b1_atProposition ) , w2_normalized_atProposition ) + b2_atProposition

                    bareCost_atProposition = sqrt(sum(square(yhat_atProposition-examy), axis=0))

                    quadraticApproximation = bareCost.numpy() - effLearningRate*(sum(w1_gra_v*diff_w1) + sum(w2_gra_v*diff_w2) + sum(b1_gra_v*diff_b1) + sum(b2_gra_v*diff_b2)) + effLearningRate/2*( sum(diff_w1*diff_w1) +  sum(diff_w2*diff_w2) + sum(diff_b1*diff_b1) + sum(diff_b2*diff_b2) )

                    backtrackingFlag = False #bareCost_atProposition > quadraticApproximation

                w1.assign(proposed_w1)
                b1.assign(proposed_b1)
                w2.assign(proposed_w2)
                b2.assign(proposed_b2)

            else:

                w1.assign_add(-learningRate*w1_gra.numpy())

                # update b1 with constraints
                # first we compute the bounding envolope

                maxXw1 = tf.reduce_max(Xw1, axis=0)
                minXw1 = tf.reduce_min(Xw1, axis=0)

                # proposition
                proposed_b1 = b1 - learningRate*b1_gra

                # Booleans that determines if outside the bounds
                ifBelow = (proposed_b1 < minXw1)
                ifAbove = (proposed_b1 > maxXw1)

                # If outside, take the bounds
                proposed_b1 = tf.where( ifBelow, minXw1, proposed_b1)
                proposed_b1 = tf.where( ifAbove, maxXw1, proposed_b1)

                #update b1
                b1.assign(proposed_b1)

                # update w2
                w2.assign_add(-learningRate*w2_gra)

                # update b2
                b2.assign_add(-learningRate*b2_gra)

        # return a list of gradients
        #return cost, [w1_gra,b1_gra,w2_gra,b2_gra]
        return cost

    if (ifTrain):
        
        print('>>> Starting Training')
        
        # iteratively update the modell parameters
        for epoch in range(iterations):      # steepest descent part
        
            if epoch % 50 == 0 and epoch != 0:
                print('         At epoch', epoch,'cost =', cost.numpy())

            cost = evaluate(Xsample, Ysample, ifUpdate=True, ifISTA=False)
            #loss[epoch] = cost
        
        print('>>> ISTA')
        
        for epoch in range(iterations, maxIterN):    # ISTA part
            
            if epoch % 50 == 0:
                print('         At epoch', epoch,'cost =', cost.numpy())

            cost = evaluate(Xsample, Ysample, ifUpdate=True, ifISTA=True)
            #loss[epoch] = cost

    # we return only the numpy values

    thetahat = extracttheta(w1.numpy())
    return thetahat
   
