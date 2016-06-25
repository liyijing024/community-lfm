"""
Predict community membership by presence or absence of a feature.

Use a given feature matrix as well as the average of features of neighbors.
"""

import numpy as np
import pandas as pd
#import numpy.linalg as la
#import random
import cf_logit as cf
import feature_logit as fl
import sys
#import os
import pickle
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse import csc_matrix
from scipy.sparse import linalg

adj = sys.argv[1]

BASE = ''

# feature matrix with neighbor averages
N_FILE           = {'ego0':BASE+'data/0_hometown_NF.pck',
                    'full9core-protection':BASE+'data/allwave1_H1CO3-dense-9core_NF.pck',
                    'full9core-risky':BASE+'data/allwave1_H1CO3-dense-9core_NF.pck'}
# data set partitions (training, validation, test)
PCK_FILE         = {'ego0':BASE+'data/0_partitions.pck',
                    'ego3059':BASE+'data/3059_partitions.pck',
                    'full9core-protection':BASE+'data/allwave1_H1CO3-dense-9core_partitions.pck',
                    'full9core-risky':BASE+'data/allwave1_H1CO3-dense-9core_flip_partitions.pck'}
# learned coefficients
PCK_THETA_FILE   = {'ego0':BASE+'results/ego0/0_hometown_neighbor_theta.pck',
                    'ego3059':BASE+'results/ego3059/3059_hometown_neighbor_theta.pck',
                    'full9core-protection':BASE+'results/full9core-protection/allwave1_H1CO3-dense-9core_neighbor_theta.pck',
                    'full9core-risky':BASE+'results/full9core-risky/allwave1_H1CO3-dense-9core_neighbor_theta.pck'}
# output
OUT_FILE         = {'ego0':base+'results/ego0/0_hometown_neighbor.out',
                    'ego3059':base+'results/ego3059/3059_hometown_neighbor.out',
                    'full9core-protection':base+'results/full9core-protection/allwave1_H1CO3-dense-9core_neighbor.out',
                    'full9core-risky':base+'results/full9core-risky/allwave1_H1CO3-dense-9core_neighbor.out'}
# figures
FIG_FILE         = {'ego0':base+'results/ego0/0_hometown_neighbor',
                    'ego3059':base+'results/ego3059/3059_hometown_neighbor',
                    'full9core-protection':base+'results/full9core-protection/allwave1_H1CO3-dense-9core_neighbor'}
                    'full9core-risky':base+'results/full9core-risky/allwave1_H1CO3-dense-9core_neighbor'}

print '\n\nloading and preparing data...'
print '=================================================='

# load averages of neighbor features
nf_pck = open(N_FILE[adj], 'rb')
X = pickle.load(nf_pck)
nf_pck.close()


###################################################################################
###      Network-Enhanced Community Detection -- Average of neighbor values     ###
###################################################################################

out = open(OUT_FILE[adj], 'w')
out.write('Network enhanced community detection with average of neighbor values')
out.write('\n==================================================')
out.write('\n==================================================')
out.write('\n\nData matrix: '+str(n)+' x '+str(f))

"""
Prepare data
"""

# predictive feature
y = np.array([row[-1] for row in F])

# partition sets
pck = open(PCK_FILE[adj], 'rb')
pos_train = pickle.load(pck)
neg_train = pickle.load(pck)
pos_val = pickle.load(pck)
neg_val = pickle.load(pck)
pos_test = pickle.load(pck)
neg_test = pickle.load(pck)
pck.close()

# number of positive instances
numpos = np.count_nonzero(y)
# number of negative instances
numneg = (y==0).sum()
# indices of positive instances
I = np.nonzero(y)
# indices of negative instances
zI = np.where(y==0)

out.write('\nnumber of positive examples:'+str(numpos))

# build sets
X_train = np.take(X, [I[0][x] for x in pos_train], axis=0)
X_train = np.concatenate((X_train, np.take(X, [zI[0][x] for x in neg_train], axis=0)), axis=0)
np.random.shuffle(X_train)
y_train = np.array([row[-1] for row in X_train])
X_train = X_train[:, :-1]

X_val = np.take(X, [I[0][x] for x in pos_val], axis=0)
X_val = np.concatenate((X_val, np.take(X, [zI[0][x] for x in neg_val], axis=0)), axis=0)
np.random.shuffle(X_val)
y_val = np.array([row[-1] for row in X_val])
X_val = X_val[:, :-1]

X_test = np.take(X, [I[0][x] for x in pos_test], axis=0)
X_test = np.concatenate((X_test, np.take(X, [zI[0][x] for x in neg_test], axis=0)), axis=0)
np.random.shuffle(X_test)
y_test = np.array([row[-1] for row in X_test])
X_test = X_test[:, :-1]

"""
Train and validate model
"""

print '\ntraining regression model...'

errs = []
costs = []
accs = []
acc_opt = 0 
lam_opt = 0
theta_opt = 0
#regularizer = [0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
regularizer = [0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]
for lam in regularizer:
    print 'training on lambda=', lam
    # learn parameters from training set
    theta = fl.train_model(X_train, y_train, lam)
    
    # evaluate parameters on training set
    err_train, acc_train, _, _ = fl.test_model(X_train, theta, y_train)
    cost_train = fl.logLikelihood(theta, X_train, y_train, lam)

    # evaluate parameters on validation set
    err_val, acc_val, _, _ = fl.test_model(X_val, theta, y_val)
    cost_val = fl.logLikelihood(theta, X_val, y_val, lam)

    errs.append([lam, err_train, err_val])
    costs.append([lam, cost_train, cost_val])
    accs.append([lam, acc_train, acc_val])

    # maybe use different criterion?
    if acc_val > acc_opt:
        acc_opt = acc_val
        lam_opt = lam
        theta_opt = theta

## retrain on entire training+validation
#X_fulltrain = np.concatenate((X_train, X_val), axis=0)
#y_fulltrain = np.concatenate((y_train, y_val))
#
#theta_opt = fl.train_model(X_fulltrain, y_fulltrain, lam_opt)

# save coefficients
print 'saving...'
t = open(PCK_THETA_FILE[adj], 'wb')
pickle.dump(theta_opt, t)
t.close()

"""
Test model
"""
print '=================================================='

print '\noptimal regularization factor:', lam_opt
#print 'evaluate model on full training set...'
#err, acc, percent_pred_pos, percent_actual_pos = fl.test_model(X_fulltrain, theta_opt, y_fulltrain)
#cost = fl.logLikelihood(theta_opt, X_fulltrain, y_fulltrain, lam_opt)
print 'evaluate model on training set...'
err, acc, percent_pred_pos, percent_actual_pos = fl.test_model(X_train, theta_opt, y_train)
cost = fl.logLikelihood(theta_opt, X_train, y_train, lam_opt)
print '\ttraining RMSE:', err
print '\ttraining Loglikelihood:', cost
print '\ttraining accuracy:', acc
print '\n\tpercent predicted positive:', percent_pred_pos
print '\tpercent actual positive:', percent_actual_pos
print 'evaluate model on test set...'
errX, accX, percent_pred_posX, percent_actual_posX = fl.test_model(X_test, theta_opt, y_test)
costX = fl.logLikelihood(theta_opt, X_test, y_test, lam_opt)
print '\ttest RMSE:', errX
print '\ttest Loglikelihood:', costX
print '\ttest accuracy:', accX
print '\n\tpercent predicted positive:', percent_pred_posX
print '\tpercent actual positive:', percent_actual_posX

out.write('\n\nlambda:\t\t' + str(lam_opt))
out.write('\ntraining error:\t\t' + str(err))
out.write('\ntraining ll:\t\t' + str(cost))
out.write('\ntraining accuracy:\t' + str(acc))
out.write('\n\n\tpercent predicted positive:' + str(percent_pred_pos))
out.write('\n\tpercent actual positive:' + str(percent_actual_pos))

out.write('\n\ntest error:\t\t' + str(errX))
out.write('\ntest ll:\t\t' + str(costX))
out.write('\ntest accuracy:\t' + str(accX))
out.write('\n\n\tpercent predicted positive:' + str(percent_pred_posX))
out.write('\n\tpercent actual positive:' + str(percent_actual_posX))

out.write('\n\nerror:')
out.write('\nlambda \t training \t validation')
for row in errs:
    out.write('\n'+str(row[0])+'\t'+str(row[1])+'\t'+str(row[2]))

out.write('\n\nLoglikelihood:')
out.write('\nlambda \t training \t validation')
for row in costs:
    out.write('\n'+str(row[0])+'\t'+str(row[1])+'\t'+str(row[2]))

out.write('\n\naccuracy:')
out.write('\nlambda \t training \t validation')
for row in accs:
    out.write('\n'+str(row[0])+'\t'+str(row[1])+'\t'+str(row[2]))

y_pred_raw = fl.sigmoid(np.dot(X_test, theta_opt))
t = 0.5
y_pred = [1 if i >= t else 0 for i in y_pred_raw]

# compute percent misclassification error
y_pred, y_test = np.array(y_pred), np.array(y_test)
# confusion matrix
TP, FP, FN, TN = 0,0,0,0
for i in range(len(y_pred)):
    # actual positive
    if y_test[i] == 1:
        # true positive
        if y_pred[i] == 1:
            TP += 1
        # false negative
        else:
            FN += 1
    # actual negative
    if y_test[i] == 0:
        # true negative 
        if y_pred[i] == 0:
            TN += 1
        # false positive
        else:
            FP += 1
print '\nperformance on test set for threshold', t, ':'
try:
    precision = TP/(TP+FP*1.0)
    print '\tprecision:\t', precision
except(ZeroDivisionError):
    precision = None
try:
    recall = TP/(TP+FN*1.0)
    print '\trecall:\t\t', recall
except(ZeroDivisionError):
    recall = None
try:
    if recall and precision:
        F1 = (2*precision*recall)/(precision+recall)
        print '\tF1 score:\t', F1
except(ZeroDivisionError):
    pass

#for i in range(10):
#    print df.columns[i], '\t', theta_opt[i]

#out.write('\n\ncoefficients:\n')
#for i in range(len(theta_opt)):
#    out.write('\n'+str(theta_opt[i]))

out.close()


## plot training error and validation error against lambda
#plt.figure()
#plt.semilogx([row[0] for row in errs], [row[1] for row in errs])
#plt.semilogx([row[0] for row in errs], [row[2] for row in errs])
#plt.legend(('training error', 'validation error'), loc='best')
#plt.xlabel('lambda')
#plt.ylabel('RMSE')
##plt.show()
#plt.savefig(FIG_FILE[adj]+'_error.png')
#
#plt.figure()
#plt.semilogx([row[0] for row in costs], [row[1] for row in costs])
#plt.semilogx([row[0] for row in costs], [row[2] for row in costs])
#plt.legend(('training Loglikelihood', 'validation Loglikelihood'), loc='best')
#plt.xlabel('lambda')
#plt.ylabel('Loglikelihood')
##plt.show()
#plt.savefig(FIG_FILE[adj]+'_cost.png')
#
#plt.figure()
#plt.semilogx([row[0] for row in accs], [row[1] for row in accs])
#plt.semilogx([row[0] for row in accs], [row[2] for row in accs])
#plt.legend(('training accuracy', 'validation accuracy'), loc='best')
#plt.xlabel('lambda')
#plt.ylabel('accuracy')
##plt.show()
#plt.savefig(FIG_FILE[adj]+'_accuracy.png')
#
