"""
Partition the adjacency matrix into training, validation, and test sets.
"""

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

ADJ_FILE = {'ego0':base+'data/0.adj',
            'ego12800':base+'data/12800.adj',
            'ego3059':base+'data/3059.adj',
            'full-9core':base+'data/allwave1_H1CO3-dense-9core_adjmat.csc.npz'}
PCK_FILE = {'ego0':base+'data/0_lfm_partitions.pck',
            'ego12800':base+'data/12800_lfm_partitions.pck',
            'ego3059':base+'data/3059_lfm_partitions.pck',
            'full-9core':base+'data/allwave1_H1CO3-dense-9core_lfm_partitions.pck'}
MAT_FILE = {'ego0':base+'data/0_lfm_mat_partitions_',
            'ego12800':base+'data/12800_lfm_mat_partitions_',
            'ego3059':base+'data/3059_lfm_mat_partitions_',
            'full-9core':base+'data/allwave1_H1CO3-dense-9core_lfm_mat_partitions_'}

print '\n\nloading and preparing data...'
print '=================================================='

if adj[:3] == 'ego':
    A = np.loadtxt(ADJ_FILE[adj])
    A = csc_matrix(A)

else:
    temp = np.load(ADJ_FILE[adj])
    A = csc_matrix((temp['data'],temp['indices'],temp['indptr']), shape=temp['shape'])
    #A = A.toarray()

n = A.shape[0]

"""
Build training, validation, and test sets
"""

# partition known A values into train, validation, and test

# use positive instances for training
# some of the positive instances (Aij = 1) will be sent to the
# training set (corresponding to '1' entries), validation set
# (corresponding to '2' entries), and test set (corresponding to
# '3' entries in 'R')

# indices of positive instances
#I, J = np.nonzero(A)
I, J = A.nonzero()
# indices of negative instances
#zI, zJ = np.where(A==0) 
zI, zJ = zip(*list(set([(i,j) for i in range(n) for j in range(n)]) - set(zip(I, J))))
# number of positive instances
#nnz = np.count_nonzero(A)
nnz = A.nnz
# number of negative instances
#nz = (A==0).sum()
nz = n**2 - nnz

print '\nadjacency matrix:'
print '\tnnz:\t', nnz
print '\tnz:\t', nz

# set sizes of sets
pos_to_train = int(0.5*nnz)
neg_to_train = 2*pos_to_train
train_size = pos_to_train + neg_to_train

print '\ntraining set:'
print '\tsize:\t', train_size
print '\tnnz:\t', pos_to_train
print '\tpercent nonzeros:\t', pos_to_train*1.0/train_size
print '\tnz:\t', neg_to_train

pos_to_val = int(0.25*nnz)
neg_to_val = 2*pos_to_val
val_size = pos_to_val + neg_to_val
 
print '\nvalidation set:'
print '\tsize:\t', val_size
print '\tnnz:\t', pos_to_val
print '\tpercent nonzeros:\t', pos_to_val*1.0/val_size
print '\tnz:\t', neg_to_val

pos_to_test = nnz - (pos_to_train + pos_to_val)
neg_to_test = 2*pos_to_test
test_size = pos_to_test + neg_to_test

print '\ntest set:'
print '\tsize:\t', test_size
print '\tnnz:\t', pos_to_test
print '\tpercent nonzeros:\t', pos_to_test*1.0/test_size
print '\tnz:\t', neg_to_test
 
#partition = [1]*train_size
#partition.extend([2]*val_size)
#partition.extend([3]*test_size)
#partition.extend([0]*(A.size-train_size-val_size-test_size))
#partition = np.random.permutation(partition)
#R = np.reshape(partition, A.shape)
#R_train = (R == 1)
#R_val = (R == 2)
#R_test = (R == 3)

# positive instances for training, val, and test
pos_inds = range(nnz)
np.random.shuffle(pos_inds)
pos_train = pos_inds[:pos_to_train]
pos_val = pos_inds[pos_to_train:(pos_to_train+pos_to_val)]
pos_test = pos_inds[-pos_to_test:] 

print '\nnumber of positive examples to training set:', len(pos_train) 
print 'number of positive examples to validation set:', len(pos_val) 
print 'number of positive examples to test set:', len(pos_test) 

# negative instances for training and test
neg_inds = range(nz)
np.random.shuffle(neg_inds)
neg_train = neg_inds[:neg_to_train]
neg_val = neg_inds[neg_to_train:(neg_to_train+neg_to_val)]
neg_test = neg_inds[-neg_to_test:] 

print '\nnumber of negative examples to training set:', len(neg_train) 
print 'number of negative examples to validation set:', len(neg_val) 
print 'number of negative examples to test set:', len(neg_test) 

# sparse matrices
row_ind_train = [I[l] for l in pos_train]
row_ind_train.extend([zI[l] for l in neg_train])
col_ind_train = [J[l] for l in pos_train]
col_ind_train.extend([zJ[l] for l in neg_train])
R_train = csc_matrix(([1]*len(row_ind_train), (row_ind_train, col_ind_train)), shape=A.shape)

row_ind_val = [I[l] for l in pos_val]
row_ind_val.extend([zI[l] for l in neg_val])
col_ind_val = [J[l] for l in pos_val]
col_ind_val.extend([zJ[l] for l in neg_val])
R_val = csc_matrix(([1]*len(row_ind_val), (row_ind_val, col_ind_val)), shape=A.shape)

row_ind_test = [I[l] for l in pos_test]
row_ind_test.extend([zI[l] for l in neg_test])
col_ind_test = [J[l] for l in pos_test]
col_ind_test.extend([zJ[l] for l in neg_test])
R_test = csc_matrix(([1]*len(row_ind_test), (row_ind_test, col_ind_test)), shape=A.shape)

#print '\nnumber of training examples:', np.count_nonzero(R_train)
#print 'number of validation examples:', np.count_nonzero(R_val)
#print 'number of test examples:', np.count_nonzero(R_test)

print '\nnumber of training examples:', R_train.nnz
print 'number of validation examples:', R_val.nnz
print 'number of test examples:', R_test.nnz

print '\npercent positive training instances:', (1.0*A.multiply(R_train).nnz)/R_train.nnz
print 'percent positive validation instances:', (1.0*A.multiply(R_val).nnz)/R_val.nnz
print 'percent positive test instances:', (1.0*A.multiply(R_test).nnz)/R_test.nnz

print '\nsaving...'
f = open(PCK_FILE[adj], 'wb')
pickle.dump(pos_train, f)
pickle.dump(pos_val, f)
pickle.dump(pos_test, f)
pickle.dump(neg_train, f)
pickle.dump(neg_val, f)
pickle.dump(neg_test, f)
f.close()

np.savez(MAT_FILE[adj]+'R_train', data=R_train.data, indices=R_train.indices, indptr=R_train.indptr, shape=R_train.shape)
np.savez(MAT_FILE[adj]+'R_val', data=R_val.data, indices=R_val.indices, indptr=R_val.indptr, shape=R_val.shape)
np.savez(MAT_FILE[adj]+'R_test', data=R_test.data, indices=R_test.indices, indptr=R_test.indptr, shape=R_test.shape)
