import sys
import numpy as np
import random
import matplotlib as plt #pip install matplotlib
import statsmodels.api as sm #pip install statsmodels
import scipy.stats as ss
import sklearn.preprocessing #pip install sklearn
from sklearn import linear_model
import pickle
#import Main
import cvxopt

import time

### Parameters

path = 'C:/Users/Master'
thetavalue = 10
noise_variance = 1
alpha = 0.05
noise = True

### Functions

threshhold = 1e-6
def threshold(array, thresh):
    return [0 if abs(a)<thresh else a for a in array]

def TPR(theta, thetahat):  #True Positive Ratio
    
    if len(theta) != len(thetahat):
        raise ValueError('Not the same lengths')
    thetahat = threshold(thetahat, threshhold)
    PR = 0
    positives = 0
    for i in range(len(theta)):
        if theta[i] != 0 :
             positives += 1
             if thetahat[i] > 0:
                 PR += 1
    if positives == 0:
        return 1
    return PR/positives

def FPR(theta, thetahat):  #False Positive Ratio
    
    if len(theta) != len(thetahat):
        raise ValueError('Not the same lengths')
    thetahat = threshold(thetahat, threshhold)
    NR = 0
    positives = 0
    for i in range(len(theta)):
        if thetahat[i] > 0 :
             positives += 1
             if theta[i] == 0:
                 NR += 1
    if positives == 0:
        return 0
    return NR/positives

def TNR(theta, thetahat):  #True Negative Ratio
    
    if len(theta) != len(thetahat):
        raise ValueError('Not the same lengths')
    thetahat = threshold(thetahat, threshhold)
    NR = 0
    negatives = 0
    for i in range(len(theta)):
        if theta[i] == 0 :
             negatives += 1
             if thetahat[i] == 0:
                 NR += 1
    if negatives == 0:
        return 1
    return NR/negatives

def FNR(theta, thetahat):  #False Negative Ratio
    
    if len(theta) != len(thetahat):
        raise ValueError('Not the same lengths')
    thetahat = threshold(thetahat, threshhold)
    NR = 0
    negatives = 0
    for i in range(len(theta)):
        if theta[i] != 0 :
             negatives += 1
             if thetahat[i] == 0:
                 NR += 1
    if negatives == 0:
        return NR
    return NR/negatives

def ESR(theta, thetahat):  #exact support recovery
    
    if len(theta) != len(thetahat):
        raise ValueError('Not the same lengths')
    thetahat = threshold(thetahat, threshhold)
    esr = True
    i = 0
    while esr and i < len(thetahat):
        if theta[i] == 0 and thetahat[i] != 0:
            esr = False
        elif theta[i] != 0 and thetahat[i] == 0:
            esr = False
        i += 1
    return esr

def sparse_multi(X, v): #sparse multiplication for faster computation
    n, p1 = X.shape
    if p1 != len(v):
        raise ValueError('Dimension mismatch')
        
    indices = np.nonzero(v)
    X_ = np.matrix([X[:,ind] for ind in indices][0])
    v_ = np.array([v[ind] for ind in indices][0])
    return X_@v_

#Model

class Model:
    
    def __init__(self, X, s, method): # methods = ['sqrtLASSO', 'LASSO', 'ANN']
        
        self.X = X
        self.s = s
        self.method = method
        self.n, self.p1 = X.shape
        
        self.normalize
        
        self.Ys = []
        self.thetas = []
        self.sparseindexes = []
        
        self.thetahats = []
        self.thetahats_pred = []
        self.simulcount = 0
        self.lamdaqut = None
        
        self.learnt = False
        
    @property
    
    def normalize(self):
        self.X = sklearn.preprocessing.normalize(self.X, norm='l2', axis=0, copy=True, return_norm=False)
    
    def simulate(self, count): #Simulate /count instances of theta and corresponding Y for X, s
        for i in range(count):
            sparsity_indexes = random.sample(range(1, self.p1), self.s)
            theta = np.array([thetavalue*np.int(i in sparsity_indexes) for i in range(0, self.p1)])
            Y = self.X.dot(theta) + noise*np.random.normal(0, noise_variance, size=(self.n))
            self.Ys.append(Y)
            self.thetas.append(theta)
            self.sparseindexes.append(sparsity_indexes)
            self.simulcount += 1
            
    def approxlam0(self): #Simulate /sims H0's and calculate corresponding lamda0, then lamdaqut = (1-alpha)quantile
        
        lamda0s = []
        sims = 500
        
        if self.method == 'sqrtLASSO':
            for i in range(sims):
                Y = np.random.normal(0, noise_variance, size=(self.n))
                Ym = Y - 1/np.sqrt(self.n)*np.mean(Y)
                lam0 = np.sqrt(self.n)*np.linalg.norm(self.X.T.dot(Ym), np.inf)/np.linalg.norm(Ym, 2)
                lamda0s.append(lam0)
                
        if self.method == 'LASSO':
             for i in range(sims):
                Y = np.random.normal(0, noise_variance, size=(self.n))
                Ym = Y - np.mean(Y)
                lam0 = (1/self.n)*np.linalg.norm(self.X.T.dot(Ym), np.inf)
                lamda0s.append(lam0)
                
        if self.method == 'ANN':
            for i in range(sims):
                Y = np.random.normal(0, noise_variance, size=(self.n))
                lam0 = np.linalg.norm(self.X.T.dot(Y), np.inf)/np.linalg.norm(Y, 2)
                lamda0s.append(lam0)
                
        self.lamdaqut = np.quantile(lamda0s, 1-alpha)
        #print('\u03BB =', self.lamdaqut)
            
    def learn(self): #Loops over the simuated data and calculates method prediction (saved @thetahats)
             
        if self.method == 'sqrtLASSO':
            if self.lamdaqut == None:
                self.approxlam0()
            for y in self.Ys:
                thetahat = sm.regression.linear_model.OLS(y, self.X).fit_regularized(method  = 'sqrt_lasso', alpha = self.lamdaqut).params
                self.thetahats.append(thetahat)
            self.learnt = True
            
        if self.method == 'LASSO':
            if self.lamdaqut == None:
                self.approxlam0()
            for y in self.Ys:
                clf = linear_model.Lasso(alpha=self.lamdaqut, fit_intercept = True, positive = False)
                clf.fit(self.X, y)
                thetahat = clf.coef_
                self.thetahats.append(thetahat)
            self.learnt = True
            
        if self.method == 'ANN':
            if self.lamdaqut == None:
                self.approxlam0()
            i = 0
            for y in self.Ys:
                thetahat = Main.TFmodel(self.X,y,Main.softplus,self.lamdaqut,0.25,thetavalue,self.sparseindexes[i],True,ifTrain=True)
                self.thetahats.append(thetahat)
                i += 1
            self.learnt = True
            
    def bestlamdapred(self):
        
        if not self.learnt:
            self.approxlam0()
            if len(self.Ys) == 0:
                self.simulate(1)
        
        lampreds = []
        
        lamrange = np.linspace(self.lamdaqut, 0, 100)[:-1]
        Xtest = np.random.normal(size = (self.n**2, self.p1)) 
        self.best_lamda_preds = []
        self.pred_errors = [[] for i in range(self.simulcount)]
        
        if self.method == 'LASSO':
            
            maxiter = 1250
            if self.s == 0:
                maxiter = 1500

            for i in range(self.simulcount): #loop over thetas/ys
                pred_err_max = np.infty
                y_pred = Xtest.dot(self.thetas[i])
                for lamda in lamrange:
                    clf = linear_model.Lasso(alpha=lamda, fit_intercept = True, positive = False, warm_start = True, max_iter = maxiter)
                    clf.fit(self.X, self.Ys[i])
                    thetahat = clf.coef_
                    error = np.linalg.norm(clf.intercept_ + sparse_multi(Xtest,thetahat) - y_pred)
                    if error < pred_err_max:
                        pred_err_max = error
                        best_lamda_pred = lamda
                    self.pred_errors[i].append(error)
                self.best_lamda_preds.append(best_lamda_pred)
                
        
    def learn_predictive(self):
        
        self.bestlamdapred()
        
        self.intercepts_pred = []
        for i in range(self.simulcount):
            if self.method == 'LASSO':
                clf = linear_model.Lasso(alpha=self.best_lamda_preds[i], fit_intercept = True, positive = False)
                clf.fit(self.X, self.Ys[i])
                self.intercepts_pred.append(clf.intercept_)
                thetahat = clf.coef_
                self.thetahats_pred.append(thetahat)
                 
    def save(self):
        if self.learnt:
            with open(path+'{}/n = {}/p = {} s = {}.pkl'.format(self.method, self.n, self.p1, self.s), 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)  
        else:
            with open(path+'{}/n = {}/p = {} s = {}.pkl'.format(self.n, self.p1, self.s), 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)  

x = Model(np.random.normal(size = (100, 150)), s = 20, method = 'LASSO')
x.simulate(100)
x.learn_predictive()

def LASSOPATH(x, j):
    lamrange = np.linspace(x.lamdaqut, 0, 100)[:-1]
    thetahats = []
    for i in range(len(lamrange)):
        clf = linear_model.Lasso(alpha=lamrange[i], fit_intercept = True, warm_start = True)
        clf.fit(x.X, x.Ys[j])
        thetahat = clf.coef_
        thetahats.append(thetahat)
    for entry in np.matrix(thetahats).T:
        plt.pyplot.plot(lamrange, entry.T)
        
n = 100
p1range = np.flip((1000/np.linspace(1, 10, 20)).astype(int))
srange = np.linspace(0, 99, 50).astype(int)
methods = ['sqrtLASSO', 'LASSO', 'ANN']


'''
def createData(n, p1range, srange): 
    xvals = [np.random.normal(0, 1, size = (n, p1)) for p1 in p1range]
    for x in xvals:
        for s in srange:
            ct = Model(x, s, None)
            ct.simulate(50)
            ct.save()
        
def SimulateThread(s, method):
    for i in range(len(p1range)):
        with open(path+'Data/n = {}/p = {} s = {}.pkl'.format(n, p1range[i], s), 'rb') as f:
            ct = pickle.load(f)
            ct.method = method
        ct.learn()
        ct.save()
        print('Learnt p1 = {} with s = {} \n'.format(p1range[i], s))

for s in srange:
    SimulateThread(s, method)
    print('>>>Calculated' , str(s), '\n')
'''

def getRates(function,method,n,p1range,srange):
    sr = len(srange)
    pr = len(p1range)
    rates = np.zeros((sr, pr))
    for i in range(sr):
        for j in range(pr):
            with open(path+'{}/n = {}/p = {} s = {}.pkl'.format(method,n, p1range[j], srange[i]), 'rb') as input:
                rate = 0
                ctnew = pickle.load(input)
                for k in range(ctnew.simulcount):
                    rate += function(ctnew.thetas[k], ctnew.thetahats[k])/ctnew.simulcount
                rates[i][j] = rate
    print(rates)
    return rates

def getpredictiveRates(function,method,n,p1range,srange):
    sr = len(srange)
    pr = len(p1range)
    rates = np.zeros((sr, pr))
    for i in range(sr):
        for j in range(pr):
            print('S = {} p = {}'.format(2*i, j))
            with open(path+'{}/n = {}/p = {} s = {}.pkl'.format(method,n, p1range[j], srange[i]), 'rb') as input:
                rate = 0
                ctnew = pickle.load(input)
                #ctnew.learn()
                #ctnew.pred_errors = []
                #ctnew.best_lamda_pred = None
                #ctnew.thetahats_pred = []
                #ctnew.learn_predictive()
                
                ctnew.save()
                for k in range(ctnew.simulcount):
                    rate += function(ctnew.thetas[k], ctnew.thetahats_pred[k])/ctnew.simulcount
                rates[i][j] = rate
    print(rates)
    return rates
    
def pickleopen(n, p, s, method):
    with open(path+'{}/n = {}/p = {} s = {}.pkl'.format(method,n, p, s), 'rb') as f:
        x = pickle.load(f)
    return x


method = 'LASSO'
function = FPR
#trus = getpredictiveRates(function,method,n,p1range,srange)
trus[-2][-2] = 1
mini, maxi = round(np.min(trus), 3), round(np.max(trus), 3)
plt.pyplot.pcolormesh(n/np.array(p1range), srange/n, trus)
#plt.pyplot.pcolormesh(trus)
plt.pyplot.title('{} with min {} and max {} for {}'.format(function.__code__.co_name, mini, maxi, method))
plt.pyplot.xlabel('n/p1')
plt.pyplot.ylabel('s/n')
