import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import csc_matrix
from scipy.sparse import linalg
import pickle
import sys


adj = sys.argv[1]

BASE = ''

# feature matrix
DATA_FILE      = {'ego0':BASE+'data/0_hometown.features',
                  'ego3059':BASE+'data/3059_hometown.features',
                  'full9core-protection':BASE+'data/allwave1_H1CO3-dense-9core.csv',
                  'full9core-risky':BASE+'data/allwave1_H1CO3-dense-9core_flip.csv'}
# social network adjacency matrix
ADJ_FILE       = {'ego0':BASE+'data/0.adj',
                  'ego3059':BASE+'data/3059.adj',
                  'full9core-protection':BASE+'data/allwave1_H1CO3-dense-9core_adjmat.csc.npz',
                  'full9core-risky':BASE+'data/allwave1_H1CO3-dense-9core_adjmat.csc.npz'}
# feature matrix with neighbor averages
NF_FILE        = {'ego0':BASE+'data/0_hometown_NF.pck',
                  'full9core-protection':BASE+'data/allwave1_H1CO3-dense-9core_NF.pck',
                  'full9core-risky':BASE+'data/allwave1_H1CO3-dense-9core_NF.pck'}


print '\n\nloading and preparing data...'
print '=================================================='

# load adjacency matrix
if adj[:3] == 'ego':
    A = np.loadtxt(ADJ_FILE[adj])
    #A = csc_matrix(A)

else:
    temp = np.load(ADJ_FILE[adj])
    A = csc_matrix((temp['data'],temp['indices'],temp['indptr']), shape=temp['shape'])
    A = A.toarray()

n = A.shape[0]

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

f = F.shape[1]

if F.shape[0] != n:
    sys.exit('misaligned number of instances, break')


# get averages of neighbor features
_F = np.zeros((n,1))
for feature in range(f):
    avgs = np.zeros((n,1))
    for ego in range(n):
        avgs[ego] = np.mean(A[:,ego]*F[:,feature])
    _F = np.concatenate((_F, avgs), axis=1)

# add averages of neighbor features
X = np.concatenate((_F[:,1:], F), axis=1)

nf_pck = open(NF_FILE[adj], 'wb')
pickle.dump(X, nf_pck)
nf_pck.close()

