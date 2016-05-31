"""
This module computes a low-rank approximation of the adjacency matrix
for learning community membership.

Given an adjacency matrix A, compute nXk matrices U, V, such that
A(i,j) ~ \alpha + \beta_i + \beta_j + U_iV_j^T.

We use a simple validation to compute the best value of k.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.sparse
from scipy.sparse import csc_matrix
from scipy.sparse import linalg
import random
import cf_logit as cf
import sys
import pickle

adj = sys.argv[1]

base = ''

ADJ_FILE     = {'ego0':base+'data/0.adj',
                'ego12800':base+'data/12800.adj',
                'ego3059':base+'data/3059.adj',
                'full-9core':base+'data/allwave1_H1CO3-dense-9core_adjmat.csc.npz'}
PCK_FILE     = {'ego0':base+'data/0_lfm_partitions.pck',
                'ego12800':base+'data/12800_lfm_partitions.pck',
                'ego3059':base+'data/3059_lfm_partitions.pck',
                'full-9core':base+'data/allwave1_H1CO3-dense-9core_lfm_partitions.pck'}
MAT_FILE     = {'ego0':base+'data/0_lfm_mat_partitions_',
                'ego12800':base+'data/12800_lfm_mat_partitions_',
                'ego3059':base+'data/3059_lfm_mat_partitions_',
                'full-9core':base+'data/allwave1_H1CO3-dense-9core_lfm_mat_partitions_'}
PCK_LFM_FILE = {'ego0':base+'results/ego0/0_latent_factors.pck',
                'ego12800':base+'results/ego12800/12800_latent_factors.pck',
                'ego3059':base+'results/ego3059/3059_latent_factors.pck',
                'full-9core':base+'results/allwave1_H1CO3-dense-9core_latent_factors.pck'}
RESULTS_FILE = {'ego0':base+'results/ego0/0_latent_factors.txt',
                'ego12800':base+'results/ego12800/12800_latent_factors.txt',
                'ego3059':base+'results/ego3059/3059_latent_factors.txt',
                'full-9core':base+'results/allwave1_H1CO3-dense-9core_latent_factors.txt'}

print '\n\nloading and preparing data...'
print '=================================================='

if adj[:3] == 'ego':
    A = np.loadtxt(ADJ_FILE[adj])
    #A = csc_matrix(A)

else:
    temp = np.load(ADJ_FILE[adj])
    A = csc_matrix((temp['data'],temp['indices'],temp['indptr']), shape=temp['shape'])
    #A = A.toarray()

n = A.shape[0]

"""
Build training, validation, and test sets
"""

#f = open(PCK_FILE[adj], 'rb')
#pos_train = pickle.load(f)
#pos_val = pickle.load(f)
#pos_test = pickle.load(f)
#neg_train = pickle.load(f)
#neg_val = pickle.load(f)
#neg_test = pickle.load(f)
#f.close()
#
## build sets
#
## indices of positive instances
##I, J = np.nonzero(A)
#I, J = A.nonzero()
## indices of negative instances
#zI, zJ = np.where(A==0) 
## number of positive instances
##nnz = np.count_nonzero(A)
#nnz = A.nnz
## number of negative instances
##nz = (A==0).sum()
#nz = len(zI)
#
### full matrices
##R_train = np.zeros_like(A)
##for l in pos_train:
##    R_train[I[l], J[l]] = 1
##for l in neg_train:
##    R_train[zI[l], zJ[l]] = 1
##
##R_val = np.zeros_like(A)
##for l in pos_val:
##    R_val[I[l], J[l]] = 1
##for l in neg_val:
##    R_val[zI[l], zJ[l]] = 1
##
##R_test = np.zeros_like(A)
##for l in pos_test:
##    R_test[I[l], J[l]] = 1
##for l in neg_test:
##    R_test[zI[l], zJ[l]] = 1
#
##A = csc_matrix(A)
#
## sparse matrices
#row_ind_train = [I[l] for l in pos_train]
#row_ind_train.extend([zI[l] for l in neg_train])
#col_ind_train = [J[l] for l in pos_train]
#col_ind_train.extend([zJ[l] for l in neg_train])
#R_train = csc_matrix(([1]*len(row_ind_train), (row_ind_train, col_ind_train)), shape=A.shape)
#
#row_ind_val = [I[l] for l in pos_val]
#row_ind_val.extend([zI[l] for l in neg_val])
#col_ind_val = [J[l] for l in pos_val]
#col_ind_val.extend([zJ[l] for l in neg_val])
#R_val = csc_matrix(([1]*len(row_ind_val), (row_ind_val, col_ind_val)), shape=A.shape)
#
#row_ind_test = [I[l] for l in pos_test]
#row_ind_test.extend([zI[l] for l in neg_test])
#col_ind_test = [J[l] for l in pos_test]
#col_ind_test.extend([zJ[l] for l in neg_test])
#R_test = csc_matrix(([1]*len(row_ind_test), (row_ind_test, col_ind_test)), shape=A.shape)
#
##print '\nnumber of training examples:', np.count_nonzero(R_train)
##print 'number of validation examples:', np.count_nonzero(R_val)
##print 'number of test examples:', np.count_nonzero(R_test)

temp = np.load(MAT_FILE[adj]+'R_train.npz')
R_train = csc_matrix((temp['data'],temp['indices'],temp['indptr']), shape=temp['shape'])
temp = np.load(MAT_FILE[adj]+'R_val.npz')
R_val = csc_matrix((temp['data'],temp['indices'],temp['indptr']), shape=temp['shape'])
temp = np.load(MAT_FILE[adj]+'R_test.npz')
R_test = csc_matrix((temp['data'],temp['indices'],temp['indptr']), shape=temp['shape'])

del temp

print '\nnumber of training examples:', R_train.nnz
print 'number of validation examples:', R_val.nnz
print 'number of test examples:', R_test.nnz

R_train = R_train.toarray()
R_val = R_val.toarray()
R_test = R_test.toarray()

#k = 5
#U = (np.random.rand(n,k)-0.5)*0.1
#V = (np.random.rand(n,k)-0.5)*0.1
#beta = (np.random.rand(n,1)-0.5)*0.1
#alpha = (random.random()-0.5)*0.1
#Uflat = np.reshape(U, (1,U.size))
#Vflat = np.reshape(V, (1,V.size))
#X = np.concatenate((Uflat, Vflat, np.transpose(beta), np.array([[alpha]])), axis=1)[0]
#
#lam = 1.0
#
#print 'computing cost...'
#cost_ls = cf.cost_logit_lowspace(X, A, R_train, lam, n, k)
#print 'cost_lowspace =', cost
#print 'computing sample gradient...'
#dX = cf.grad_logit_lowspace(X, A, R_train, lam, n, k)
#print 'computing new cost...'
#new_cost = cf.cost_logit_lowspace(dX, A, R_train, lam, n, k)
#print 'new cost =', new_cost
#
#U, V, beta, alpha = cf.get_reps(A, R_train, lam, k)
#sys.exit()

"""
Train and validate
"""

# log likelihood
ll_train = []
ll_val = []
ll_test = []
# rmse
errs_train = []
errs_val = []
errs_test = []
# accuracy
acc_train = []
acc_val = []
acc_test = []
# optimal parameters
lam = 0.2
rmse_opt = 1000000
acc_opt = 0
k_opt = 0
lam_opt = 0
params_opt = (0,0,0,0)
for k, lam in [(k, lam) for k in range(5, 11) for lam in [0.01, 0.04, 0.16, 0.64, 2.56, 10.24]]:
    print '===================================================='
    print '\ntesting for k =', k, 'lam =', lam

    # get params from training set
    # only testing one value for k, use whole set
    #R_train = R_train + R_val
    U, V, beta, alpha = cf.get_reps(A, R_train, lam, k)
    #lfm = open(PCK_LFM_FILE[adj], 'wb')
    #pickle.dump(U, lfm)
    #pickle.dump(V, lfm)
    #pickle.dump(beta, lfm)
    #pickle.dump(alpha, lfm)
    #lfm.close()
    Uflat = np.reshape(U, (1,U.size))
    Vflat = np.reshape(V, (1,V.size))
    X = np.concatenate((Uflat, Vflat, np.transpose(beta), np.array([[alpha]])), axis=1)
    X = X[0]
 
    # evaluate params on training set
    print '\n\nevaluate model on training set\n'
    #likelihood_train = cf.cost_logit_lowspace(X, A, R_train, lam, n, k)
    likelihood_train = cf.cost_logit(X, A, R_train, lam, n, k)
    rmse_train, accuracy_train, _, _ = cf.test_model(A, R_train, U, V, beta, alpha, lam)
    ll_train.append((k, likelihood_train))
    errs_train.append((k, rmse_train))
    acc_train.append((k, accuracy_train))

    # evaluate params on validation set
    print '\n\nevaluate model on validation set\n'
    #likelihood_val = cf.cost_logit_lowspace(X, A, R_val, lam, n, k)
    likelihood_val = cf.cost_logit(X, A, R_val, lam, n, k)
    rmse_val, accuracy_val, _, _ = cf.test_model(A, R_val, U, V, beta, alpha, lam)
    ll_val.append((k, likelihood_val))
    errs_val.append((k, rmse_val))
    acc_val.append((k, accuracy_val))

    #if rmse < rmse_opt:
    #    k_opt = k
    #    rmse_opt = rmse
    #    params_opt = (U, V, beta, alpha)
    if accuracy_val > acc_opt:
        acc_opt = accuracy_val
        k_opt = k
        lam_opt = lam
        params_opt = (U, V, beta, alpha)
    
    # evaluate params on test set
    print '\n\nevaluate model on test set\n'
    #likelihood_test = cf.cost_logit_lowspace(X, A, R_test, lam, n, k)
    likelihood_test = cf.cost_logit(X, A, R_test, lam, n, k)
    rmse_test, accuracy_test, _, _ = cf.test_model(A, R_test, U, V, beta, alpha, lam)
    ll_test.append((k, likelihood_test))
    errs_test.append((k, rmse_test))
    acc_test.append((k, accuracy_test))


print 'saving...'
lfm = open(PCK_LFM_FILE[adj], 'wb')
pickle.dump(params_opt[0], lfm)
pickle.dump(params_opt[1], lfm)
pickle.dump(params_opt[2], lfm)
pickle.dump(params_opt[3], lfm)
lfm.close()

print '\n=================================================='
print '=================================================='

print '\noptimal k:', k_opt
print '\noptimal lam:', lam_opt
rmse_test, accuracy_test, pred_nnz, actual_nnz = cf.test_model(A, R_test, 
                                                               params_opt[0], 
                                                               params_opt[1], 
                                                               params_opt[2], 
                                                               params_opt[3],
                                                               lam)
f = open(RESULTS_FILE[adj], 'w')
f.write('optimal k:'+str(k_opt))
f.write('\noptimal lam:'+str(lam_opt))
f.write('\n\nerror: '+str(rmse_test))
f.write('\naccuracy: '+str(accuracy_test))
f.write('\npercent predicted positive: '+str(pred_nnz))
f.write('\npercent actual positive: '+str(actual_nnz))
f.close()

#plt.figure()
#plt.plot([entry[0] for entry in ll_train], [entry[1] for entry in ll_train], 'r') 
#plt.plot([entry[0] for entry in ll_val], [entry[1] for entry in ll_val], 'b') 
#plt.plot([entry[0] for entry in ll_test], [entry[1] for entry in ll_test], 'g') 
#plt.legend(('training log-likelihood', 'validation log-likelihood', 'test log-likelihood'), loc='best')
#plt.xlabel('k')
#plt.ylabel('log-likelihood')
#plt.save('results/allwave1_LFM_cost.png')
#
#plt.figure()
#plt.plot([entry[0] for entry in errs_train], [entry[1] for entry in errs_train], 'r') 
#plt.plot([entry[0] for entry in errs_val], [entry[1] for entry in errs_val], 'b') 
#plt.plot([entry[0] for entry in errs_test], [entry[1] for entry in errs_test], 'g') 
#plt.legend(('training error', 'validation error', 'test error'), loc='best')
#plt.xlabel('k')
#plt.ylabel('RMSE')
#plt.save('results/allwave1_LFM_error.png')
#
#plt.figure()
#plt.plot([entry[0] for entry in acc_train], [entry[1] for entry in acc_train], 'r') 
#plt.plot([entry[0] for entry in acc_val], [entry[1] for entry in acc_val], 'b') 
#plt.plot([entry[0] for entry in acc_test], [entry[1] for entry in acc_test], 'g') 
#plt.legend(('training accuracy', 'validation accuracy', 'test accuracy'), loc='best')
#plt.xlabel('k')
#plt.ylabel('Accuracy')
#plt.save('results/allwave1_LFM_acc.png')
