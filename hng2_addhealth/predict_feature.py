"""
Predict community membership by presence or absence of a feature.

This is the "regression on F" method.
"""


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import numpy.linalg as la
import random
import cf_logit as cf
#import alladj_logit as adj
import feature_logit as fl
import os
import pickle
import scipy.linalg
import sys

ds = sys.argv[1]

BASE = ''

# feature matrix
DATA_FILE        = {'ego0':BASE+'data/0_hometown.features',
                    'ego3059':BASE+'data/3059_hometown.features',
                    'full9core-protection':BASE+'data/allwave1_H1CO3-dense-9core.csv',
                    'full9core-risky':BASE+'data/allwave1_H1CO3-dense-9core_flip.csv'}
# data set partitions (training, validation, test)
PCK_FILE         = {'ego0':BASE+'data/0_partitions.pck',
                    'ego3059':BASE+'data/3059_partitions.pck',
                    'full9core-protection':BASE+'data/allwave1_H1CO3-dense-9core_partitions.pck',
                    'full9core-risky':BASE+'data/allwave1_H1CO3-dense-9core_flip_partitions.pck'}
# learned coefficients
PCK_THETA_FILE   = {'ego0':BASE+'results/ego0/0_hometown_regression_theta.pck',
                    'ego3059':BASE+'results/ego3059/3059_hometown_regression_theta.pck',
                    'full9core-protection':BASE+'results/full9core-protection/allwave1_H1CO3-dense-9core_regression_theta.pck',
                    'full9core-risky':BASE+'results/full9core-risky/allwave1_H1CO3-dense-9core_regression_theta.pck'}
# output
OUT_FILE         = {'ego0':base+'results/ego0/0_hometown_regression.out',
                    'ego3059':base+'results/ego3059/3059_hometown_regression.out',
                    'full9core-protection':base+'results/full9core-protection/allwave1_H1CO3-dense-9core_regression.out',
                    'full9core-risky':base+'results/full9core-risky/allwave1_H1CO3-dense-9core_regression.out'}
# figures
FIG_FILE         = {'ego0':base+'results/ego0/0_hometown',
                    'ego3059':base+'results/ego3059/3059_hometown',
                    'full9core-protection':base+'results/full9core-protection/allwave1_H1CO3-dense-9core',
                    'full9core-risky':base+'results/full9core-risky/allwave1_H1CO3-dense-9core'}

print '\n\nloading and preparing data...'
print '=================================================='

if ds[:3] == 'ego':
    F = np.genfromtxt(DATA_FILE[ds] , delimiter=',', comments='#')
    n = F.shape[0]
    f = F.shape[1]
    F = [np.insert(row,0,1.0) for row in F]
    F = np.reshape(F, (n, f+1))

else:
    df = pd.read_csv(DATA_FILE[ds])
    df.drop('AID', axis=1, inplace=True)
    F = df.values

n = F.shape[0]
f = F.shape[1]

############################################################
###      Community Detection using only Features         ###
############################################################

out = open(OUT_FILE[ds], 'w')
out.write('Regression with imported data')
out.write('\n==================================================')
out.write('\n==================================================')
out.write('\n\nData matrix: '+str(n)+' x '+str(f))

"""
Prepare data
"""

# predictive feature
y = np.array([row[-1] for row in F])

## add intercept
#F = [np.insert(row,0,1.0) for row in F]
#F = np.reshape(F, (n, f+1))

# partition sets
pck = open(PCK_FILE[ds], 'rb')
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

## set sizes of sets
#train_size = int(n*0.6)
#pos_to_train = int(numpos*0.6)
#neg_to_train = train_size - pos_to_train
#
#val_size = int(n*0.2)
#pos_to_val = int(numpos*0.2)
#neg_to_val = val_size - pos_to_val
#
#test_size = n - (train_size + val_size)
#pos_to_test = numpos - (pos_to_train + pos_to_val)
#neg_to_test = test_size - pos_to_test
#
#pos_inds = range(numpos)
#np.random.shuffle(pos_inds)
#pos_train = pos_inds[:pos_to_train]
#pos_val = pos_inds[pos_to_train:(pos_to_train+pos_to_val)]
#pos_test = pos_inds[-pos_to_test:]
#
#neg_inds = range(numneg)
#np.random.shuffle(neg_inds)
#neg_train = neg_inds[:neg_to_train]
#neg_val = neg_inds[neg_to_train:(neg_to_train+neg_to_val)]
#neg_test = neg_inds[-neg_to_test:]

# build sets
F_train = np.take(F, [I[0][x] for x in pos_train], axis=0)
F_train = np.concatenate((F_train, np.take(F, [zI[0][x] for x in neg_train], axis=0)), axis=0)
#np.random.shuffle(F_train)
y_train = np.array([row[-1] for row in F_train])
F_train = F_train[:, :-1]

F_val = np.take(F, [I[0][x] for x in pos_val], axis=0)
F_val = np.concatenate((F_val, np.take(F, [zI[0][x] for x in neg_val], axis=0)), axis=0)
#np.random.shuffle(F_val)
y_val = np.array([row[-1] for row in F_val])
F_val = F_val[:, :-1]

F_test = np.take(F, [I[0][x] for x in pos_test], axis=0)
F_test = np.concatenate((F_test, np.take(F, [zI[0][x] for x in neg_test], axis=0)), axis=0)
#np.random.shuffle(F_test)
y_test = np.array([row[-1] for row in F_test])
F_test = F_test[:, :-1]

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
    theta = fl.train_model(F_train, y_train, lam)
    
    # evaluate parameters on training set
    err_train, acc_train, _, _ = fl.test_model(F_train, theta, y_train)
    cost_train = fl.logLikelihood(theta, F_train, y_train, lam)

    # evaluate parameters on validation set
    err_val, acc_val, _, _ = fl.test_model(F_val, theta, y_val)
    cost_val = fl.logLikelihood(theta, F_val, y_val, lam)

    errs.append([lam, err_train, err_val])
    costs.append([lam, cost_train, cost_val])
    accs.append([lam, acc_train, acc_val])

    # maybe use different criterion?
    if acc_val > acc_opt:
        acc_opt = acc_val
        lam_opt = lam
        theta_opt = theta

## retrain on entire training+validation
#F_fulltrain = np.concatenate((F_train, F_val), axis=0)
#y_fulltrain = np.concatenate((y_train, y_val))
#
#theta_opt = fl.train_model(F_fulltrain, y_fulltrain, lam_opt)

# save coefficients
print 'saving...'
t = open(PCK_THETA_FILE[ds], 'wb')
pickle.dump(theta_opt, t)
t.close()

"""
Test model
"""
print '=================================================='

print '\noptimal regularization factor:', lam_opt
#print 'evaluate model on full training set...'
#err, acc, percent_pred_pos, percent_actual_pos = fl.test_model(F_fulltrain, theta_opt, y_fulltrain)
#cost = fl.logLikelihood(theta_opt, F_fulltrain, y_fulltrain, lam_opt)
print 'evaluate model on training set...'
err, acc, percent_pred_pos, percent_actual_pos = fl.test_model(F_train, theta_opt, y_train)
cost = fl.logLikelihood(theta_opt, F_train, y_train, lam_opt)
print '\ttraining RMSE:', err
print '\ttraining Loglikelihood:', cost
print '\ttraining accuracy:', acc
print '\n\tpercent predicted positive:', percent_pred_pos
print '\tpercent actual positive:', percent_actual_pos
print 'evaluate model on test set...'
errF, accF, percent_pred_posF, percent_actual_posF = fl.test_model(F_test, theta_opt, y_test)
costF = fl.logLikelihood(theta_opt, F_test, y_test, lam_opt)
print '\ttest RMSE:', errF
print '\ttest Loglikelihood:', costF
print '\ttest accuracy:', accF
print '\n\tpercent predicted positive:', percent_pred_posF
print '\tpercent actual positive:', percent_actual_posF

out.write('\n\nlambda:\t' + str(lam_opt))
out.write('\ntraining error:\t\t' + str(err))
out.write('\ntraining ll:\t\t' + str(cost))
out.write('\ntraining accuracy:\t' + str(acc))
out.write('\n\n\tpercent predicted positive:\t' + str(percent_pred_pos))
out.write('\n\tpercent actual positive:\t' + str(percent_actual_pos))

out.write('\n\ntest error:\t\t' + str(errF))
out.write('\ntest ll:\t\t' + str(costF))
out.write('\ntest accuracy:\t\t' + str(accF))
out.write('\n\n\tpercent predicted positive:\t' + str(percent_pred_posF))
out.write('\n\tpercent actual positive:\t' + str(percent_actual_posF))

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

y_pred_raw = fl.sigmoid(np.dot(F_test, theta_opt))
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
#
#out.write('\n\ncoefficients:\n')
#for i in range(len(theta_opt)):
#    out.write('\n'+str(df.columns[i])+'\t'+str(theta_opt[i]))

out.close()


# plot training error and validation error against lambda
plt.figure()
plt.semilogx([row[0] for row in errs], [row[1] for row in errs])
plt.semilogx([row[0] for row in errs], [row[2] for row in errs])
plt.legend(('training error', 'validation error'), loc='best')
plt.xlabel('lambda')
plt.ylabel('RMSE')
#plt.show()
plt.savefig(FIG_FILE[ds]+'_error.png')

plt.figure()
plt.semilogx([row[0] for row in costs], [row[1] for row in costs])
plt.semilogx([row[0] for row in costs], [row[2] for row in costs])
plt.legend(('training Loglikelihood', 'validation Loglikelihood'), loc='best')
plt.xlabel('lambda')
plt.ylabel('Loglikelihood')
#plt.show()
plt.savefig(FIG_FILE[ds]+'_cost.png')

plt.figure()
plt.semilogx([row[0] for row in accs], [row[1] for row in accs])
plt.semilogx([row[0] for row in accs], [row[2] for row in accs])
plt.legend(('training accuracy', 'validation accuracy'), loc='best')
plt.xlabel('lambda')
plt.ylabel('accuracy')
#plt.show()
plt.savefig(FIG_FILE[ds]+'_accuracy.png')

