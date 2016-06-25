import numpy as np
import pandas as pd
import feature_logit as fl
import sys
import pickle
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse import csc_matrix
from scipy.sparse import linalg

adj = sys.argv[1]

#BASE = '/home/addhealth/osimpson/'
BASE = ''

# feature matrix
DATA_FILE        = {'ego0':BASE+'data/0_hometown.features',
                    'ego3059':BASE+'data/3059_hometown.features',
                    'full9core-protection':BASE+'data/allwave1_H1CO3-dense-9core.csv',
                    'full9core-risky':BASE+'data/allwave1_H1CO3-dense-9core_flip.csv'}
# feature matrix with neighbor averages
N_FILE           = {'ego0':BASE+'data/0_hometown_NF.pck',
                    'full9core-protection':BASE+'data/allwave1_H1CO3-dense-9core_NF.pck',
                    'full9core-risky':BASE+'data/allwave1_H1CO3-dense-9core_NF.pck'}
# data set partitions (training, validation, test)
PCK_FILE         = {'ego0':BASE+'data/0_partitions.pck',
                    'ego3059':BASE+'data/3059_partitions.pck',
                    'full9core-protection':BASE+'data/allwave1_H1CO3-dense-9core_partitions.pck',
                    'full9core-risky':BASE+'data/allwave1_H1CO3-dense-9core_flip_partitions.pck'}
# social network degree sequence
DEG_FILE         = {'ego0':BASE+'data/0_hometown_indeg.pck',
                    'ego3059':BASE+'data/3059_hometown_indeg.pck',
                    'full9core-protection':BASE+'data/allwave1_H1CO3-dense-9core_indeg.pck',
                    'full9core-risky':BASE+'data/allwave1_H1CO3-dense-9core_indeg.pck'}
# data quality per node
DATA_DEG_FILE    = {'full9core-protection':BASE+'data/allwave1_H1CO3-dense-9core_data_deg.pck',
                    'full9core-risky':BASE+'data/allwave1_H1CO3-dense-9core_data_deg.pck'}
# latent factors
PCK_LFM_FILE     = {'ego0':BASE+'results/ego0/0_latent_factors.pck',
                    'ego3059':BASE+'results/ego3059/3059_latent_factors.pck',
                    'full9core-protection':BASE+'results/full9core-protection/allwave1_H1CO3-dense-9core_latent_factors.pck',
                    'full9core-risky':BASE+'results/full9core-risky/allwave1_H1CO3-dense-9core_latent_factors.pck'}
# learned coefficients with regression
F_THETA_FILE     = {'ego0':BASE+'results/ego0/0_hometown_regression_theta.pck',
                    'ego3059':BASE+'results/ego3059/3059_hometown_regression_theta.pck',
                    'full9core-protection':BASE+'results/full9core-protection/allwave1_H1CO3-dense-9core_regression_theta.pck',
                    'full9core-risky':BASE+'results/full9core-risky/allwave1_H1CO3-dense-9core_regression_theta.pck'}
# learned coefficients with neighbor features
N_THETA_FILE     = {'ego0':BASE+'results/ego0/0_hometown_neighbor_theta.pck',
                    'ego3059':BASE+'results/ego3059/3059_hometown_neighbor_theta.pck',
                    'full9core-protection':BASE+'results/full9core-protection/allwave1_H1CO3-dense-9core_neighbor_theta.pck',
                    'full9core-risky':BASE+'results/full9core-risky/allwave1_H1CO3-dense-9core_neighbor_theta.pck'}
# learned coefficients with LFM
LFM_THETA_FILE   = {'ego0':BASE+'results/ego0/0_hometown_lfm_theta.pck',
                    'ego3059':BASE+'results/ego3059/3059_hometown_lfm_theta.pck',
                    'full9core-protection':BASE+'results/full9core-protection/allwave1_H1CO3-dense-9core_lfm_theta.pck',
                    'full9core-risky':BASE+'results/full9core-risky/allwave1_H1CO3-dense-9core_lfm_theta.pck'}


print '\n\nloading and preparing data...'
print '=================================================='


# load feature matrix
if adj[:3] == 'ego':
    F = np.genfromtxt(DATA_FILE[adj] , delimiter=',', comments='#')
    n = F.shape[0]
    f = F.shape[1]
    F = [np.insert(row,0,1.0) for row in F]
    F = np.reshape(F, (n, f+1))

else:
    df = pd.read_csv(DATA_FILE[adj])
    _examples = df['AID']
    examples = _examples.values
    df.drop('AID', axis=1, inplace=True)
    F = df.values

n = F.shape[0]
f = F.shape[1]

# load averages of neighbor features
nf_pck = open(N_FILE[adj], 'rb')
N = pickle.load(nf_pck)
nf_pck.close()


"""
Prepare data
"""

# predictive feature
y = np.array([row[-1] for row in F])

pos_examples = sum(y)
print '\npositive examples:', pos_examples
print 'negative examples:', len(y)-pos_examples
print 'percent positive:', (pos_examples*100.0)/len(y)

# get latent factor matrices 
lfm_f = open(PCK_LFM_FILE[adj], 'rb')
U = pickle.load(lfm_f)
V = pickle.load(lfm_f)
beta = pickle.load(lfm_f)
alpha = pickle.load(lfm_f)
lfm_f.close()

# add latent factors
X = np.concatenate((U, V, F), axis=1)

# partition sets
pck = open(PCK_FILE[adj], 'rb')
pos_train = pickle.load(pck)
neg_train = pickle.load(pck)
pos_val = pickle.load(pck)
neg_val = pickle.load(pck)
pos_test = pickle.load(pck)
neg_test = pickle.load(pck)
pck.close()

# indices of positive instances
I = np.nonzero(y)
# indices of negative instances
zI = np.where(y==0)

# build sets
X_train = np.take(X, [I[0][x] for x in pos_train], axis=0)
X_train = np.concatenate((X_train, np.take(X, [zI[0][x] for x in neg_train], axis=0)), axis=0)
#np.random.shuffle(X_train)
y_train = np.array([row[-1] for row in X_train])
X_train = X_train[:, :-1]

X_val = np.take(X, [I[0][x] for x in pos_val], axis=0)
X_val = np.concatenate((X_val, np.take(X, [zI[0][x] for x in neg_val], axis=0)), axis=0)
#np.random.shuffle(X_val)
y_val = np.array([row[-1] for row in X_val])
X_val = X_val[:, :-1]

X_test = np.take(X, [I[0][x] for x in pos_test], axis=0)
X_test = np.concatenate((X_test, np.take(X, [zI[0][x] for x in neg_test], axis=0)), axis=0)
#np.random.shuffle(X_test)
y_test = np.array([row[-1] for row in X_test])
X_test = X_test[:, :-1]

# get averages of neighbor features
nf_pck = open(N_FILE[adj], 'rb')
N = pickle.load(nf_pck)
nf_pck.close()

# build sets
N_train = np.take(N, [I[0][x] for x in pos_train], axis=0)
N_train = np.concatenate((N_train, np.take(N, [zI[0][x] for x in neg_train], axis=0)), axis=0)
N_train = N_train[:, :-1]

N_val = np.take(N, [I[0][x] for x in pos_val], axis=0)
N_val = np.concatenate((N_val, np.take(N, [zI[0][x] for x in neg_val], axis=0)), axis=0)
N_val = N_val[:, :-1]

N_test = np.take(N, [I[0][x] for x in pos_test], axis=0)
N_test = np.concatenate((N_test, np.take(N, [zI[0][x] for x in neg_test], axis=0)), axis=0)
N_test = N_test[:, :-1]

# get degree sequence
deg = pickle.load(open(DEG_FILE[adj], 'rb'))
in_deg = np.reshape(deg, (1, deg.size))
# get data quality per node
if adj[:3] == 'ego':
    data_deg = np.zeros_like(in_deg)
else:
    data_deg = np.array(pickle.load(open(DATA_DEG_FILE[adj], 'rb')))
    data_deg = np.reshape(data_deg, in_deg.shape)

F = np.concatenate((np.transpose(in_deg), np.transpose(data_deg), F), axis=1)

# build sets
F_train = np.take(F, [I[0][x] for x in pos_train], axis=0)
F_train = np.concatenate((F_train, np.take(F, [zI[0][x] for x in neg_train], axis=0)), axis=0)
#np.random.shuffle(F_train)
in_deg_train = np.array([row[0] for row in F_train])
data_deg_train = np.array([row[1] for row in F_train])
F_train = F_train[:, 2:-1]

F_val = np.take(F, [I[0][x] for x in pos_val], axis=0)
F_val = np.concatenate((F_val, np.take(F, [zI[0][x] for x in neg_val], axis=0)), axis=0)
#np.random.shuffle(F_val)
in_deg_val = np.array([row[0] for row in F_val])
data_deg_val = np.array([row[1] for row in F_val])
F_val = F_val[:, 2:-1]

F_test = np.take(F, [I[0][x] for x in pos_test], axis=0)
F_test = np.concatenate((F_test, np.take(F, [zI[0][x] for x in neg_test], axis=0)), axis=0)
#np.random.shuffle(F_test)
in_deg_test = np.array([row[0] for row in F_test])
data_deg_test = np.array([row[1] for row in F_test])
F_test = F_test[:, 2:-1]

pos_examples = sum(y_test)
print '\npositive test examples:', pos_examples
print 'negative test examples:', len(y_test)-pos_examples
print 'percent positive:', (pos_examples*100.0)/len(y_test)

X = X[:, :-1]
N = N[:, :-1]
F = F[:, 2:-1]

# get regression coefficents
theta_f = pickle.load(open(F_THETA_FILE[adj], 'rb'))

# get neighbor feature regression coefficents
theta_n = pickle.load(open(N_THETA_FILE[adj], 'rb'))

# get lfm coefficents
theta_lfm = pickle.load(open(LFM_THETA_FILE[adj], 'rb'))

"""
Make predictions
"""

# threshold
t = 0.5

## Regression ##
# get predictions on test data
y_pred_raw_f_test = fl.sigmoid(np.dot(F_test, theta_f))
y_pred_f_test = np.array([1 if i >= t else 0 for i in y_pred_raw_f_test])

# decompose error
diff_f_test = np.abs(y_pred_raw_f_test - y_test)
acc_f_test = (y_pred_f_test == y_test)

# get predictions on full data
y_pred_raw_f_full = fl.sigmoid(np.dot(F, theta_f))
y_pred_f_full = np.array([1 if i >= t else 0 for i in y_pred_raw_f_full])

# decompose error
diff_f_full = np.abs(y_pred_raw_f_full - y)
acc_f_full = (y_pred_f_full == y)

## Neighbor features ##
# get predictions on test data
y_pred_raw_n_test = fl.sigmoid(np.dot(N_test, theta_n))
y_pred_n_test = np.array([1 if i >= t else 0 for i in y_pred_raw_n_test])

# decompose error
diff_n_test = np.abs(y_pred_raw_n_test - y_test)
acc_n_test = (y_pred_n_test == y_test)

# get predictions on full data
y_pred_raw_n_full = fl.sigmoid(np.dot(N, theta_n))
y_pred_n_full = np.array([1 if i >= t else 0 for i in y_pred_raw_n_full])

# decompose error
diff_n_full = np.abs(y_pred_raw_n_full - y)
acc_n_full = (y_pred_n_full == y)

## LFM ##
# get predictions on test data
y_pred_raw_lfm_test = fl.sigmoid(np.dot(X_test, theta_lfm))
y_pred_lfm_test = [1 if i >= t else 0 for i in y_pred_raw_lfm_test]

# decompose error
diff_lfm_test = np.abs(y_pred_raw_lfm_test - y_test)
acc_lfm_test = (y_pred_lfm_test == y_test)

# get predictions on full data
y_pred_raw_lfm_full = fl.sigmoid(np.dot(X, theta_lfm))
y_pred_lfm_full = [1 if i >= t else 0 for i in y_pred_raw_lfm_full]

# decompose error
diff_lfm_full = np.abs(y_pred_raw_lfm_full - y)
acc_lfm_full = (y_pred_lfm_full == y)


"""
Analyze results
"""

#### Global measures ####

def global_measures(y_pred, y_actual):
    TP, FP, FN, TN = 0,0,0,0
    for i in range(len(y_pred)):
        # actual positive
        if y_test[i] == 1:
            # true positive
            if y_pred[i] == 1:
                TP += 1.0
            # false negative
            else:
                FN += 1.0
        # actual negative
        if y_test[i] == 0:
            # true negative 
            if y_pred[i] == 0:
                TN += 1.0
            # false positive
            else:
                FP += 1.0
    print '\nperformance on test set for threshold', t, ':'
    try:
        precision = TP/(TP+FP)
        print '\tprecision:\t', precision
    except(ZeroDivisionError):
        if TP == 0 and FP == 0:
            precision = 0.0
            print '\tprecision:\t', precision
        else:
            print 'unknown issue'
            print 'TP, FP, FN:', TP, FP, FN
    #try:
    #    precision = TP/(TP+FP*1.0)
    #    print '\tprecision:\t', precision
    #except(ZeroDivisionError):
    #    precision = None
    try:
        recall = TP/(TP+FN)
        print '\trecall:\t\t', recall
    except(ZeroDivisionError):
        if TP == 0 and FN == 0:
            recall = 0.0
            print '\trecall:\t\t', recall
        else:
            print 'unknown issue'
            print 'TP, FP, FN:', TP, FP, FN
    try:
        F1 = (2*precision*recall)/(precision+recall)
        print '\tF1 score:\t', F1
    except(ZeroDivisionError):
        if precision == 0 and recall == 0:
            F1 = 0.0
            print '\tF1 score:\t', F1
        else:
            print 'unknown issue'        
    try:
        fp = FP/(TN+FP)
    except(ZeroDivisionError):
        if FP == 0 and TN == 0:
            fp = 0.0
        else:
            print 'unknown issue'
            print 'TP, FP, TN, FN:', TP, FP, TN, FN
    try:
        fn = FN/(FN+TP)
    except(ZeroDivisionError):
        if FN == 0 and TP == 0:
            fn = 0.0
        else:
            print 'unknown issue'
            print 'TP, FP, TN, FN:', TP, FP, TN, FN
    ber = 0.5*(fp+fn)
    print '\tbalanced error rate:\t', ber 
    try:
        tp = TP/(TP+FN)
    except(ZeroDivisionError):
        if TP == 0 and FN == 0:
            tp = 0.0
        else:
            print 'unknown issue'
            print 'TP, FP, TN, FN:', TP, FP, TN, FN
    try:
        tn = TN/(TN+FP)
    except(ZeroDivisionError):
        if TN == 0 and FP == 0:
            tn = 0.0
        else:
            print 'unknown issue'
            print 'TP, FP, TN, FN:', TP, FP, TN, FN
    bcr = 0.5*(tp+tn)
    print '\tbalanced classification rate:\t', bcr

## Regression ##

print '\nglobal measures for regression on F'
global_measures(y_pred_f_test, y_test)

print '\nglobal measures using neighbor features'
global_measures(y_pred_n_test, y_test)

print '\nglobal measures for regression on X'
global_measures(y_pred_lfm_test, y_test)
