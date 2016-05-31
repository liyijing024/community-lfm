import numpy as np
import random
import sys
import os
import pickle

base = 'data/'

ego = sys.argv[1]
pred_feature = 'hometown'

F = np.genfromtxt(base+ego+'_'+pred_feature+'.features', delimiter=',', comments
= '#')
# number of features
n = F.shape[0]
f = F.shape[1]

# predictive feature
y = np.array([row[-1] for row in F])

# partition sets

# number of positive instances
numpos = np.count_nonzero(y)
# number of negative instances
numneg = (y==0).sum()
# indices of positive instances
I = np.nonzero(y)
# indices of negative instances
zI = np.where(y==0)

# set sizes of sets
train_size = int(n*0.6)
pos_to_train = int(numpos*0.6)
neg_to_train = train_size - pos_to_train

val_size = int(n*0.2)
pos_to_val = int(numpos*0.2)
neg_to_val = val_size - pos_to_val

test_size = n - (train_size + val_size)
pos_to_test = numpos - (pos_to_train + pos_to_val)
neg_to_test = test_size - pos_to_test

pos_inds = range(numpos)
np.random.shuffle(pos_inds)
pos_train = pos_inds[:pos_to_train]
pos_val = pos_inds[pos_to_train:(pos_to_train+pos_to_val)]
pos_test = pos_inds[-pos_to_test:]

neg_inds = range(numneg)
np.random.shuffle(neg_inds)
neg_train = neg_inds[:neg_to_train]
neg_val = neg_inds[neg_to_train:(neg_to_train+neg_to_val)]
neg_test = neg_inds[-neg_to_test:]

# pickle indices
f = open(ego+'_partitions.pck', 'w')
pickle.dump(pos_train, f)
pickle.dump(neg_train, f)
pickle.dump(pos_val, f)
pickle.dump(neg_val, f)
pickle.dump(pos_test, f)
pickle.dump(neg_test, f)

f.close()
