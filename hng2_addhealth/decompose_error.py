import numpy as np
import pandas as pd
import feature_logit as fl
import sys
import pickle
import matplotlib.pyplot as plt
import scipy.stats
import scipy.sparse
from scipy.sparse import csc_matrix
from scipy.sparse import linalg

adj = sys.argv[1]
test = sys.argv[2]
method = 'UV'

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
# output
OUT_FILE         = {'ego0':BASE+'results/ego0/decomposed-error/0_hometown_decomposed_error',
                    'ego3059':BASE+'results/ego3059/decomposed-error/3059_hometown_decomposed_error',
                    'full9core-protection':BASE+'results/full9core-protection/decomposed-error/allwave1_H1CO3-dense-9core_decomposed_error',
                    'full9core-risky':BASE+'results/full9core-risky/decomposed-error/allwave1_H1CO3-dense-9core_decomposed_error'}
# figures
FIG_FILE         = {'ego0':BASE+'results/ego0/decomposed-error/0_hometown_decomposed_error',
                    'ego3059':BASE+'results/ego3059/decomposed-error/3059_hometown_decomposed_error',
                    'full9core-protection':BASE+'results/full9core-protection/decomposed-error/allwave1_H1CO3-dense-9core_decomposed_error',
                    'full9core-risky':BASE+'results/full9core-risky/decomposed-error/allwave1_H1CO3-dense-9core_decomposed_error'}

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

# concatenate test and val
X_test = np.concatenate((X_val, X_test), axis=0)
N_test = np.concatenate((N_val, N_test), axis=0)
F_test = np.concatenate((F_val, F_test), axis=0)
y_test = np.concatenate((y_val, y_test))
in_deg_test = np.concatenate((in_deg_val, in_deg_test))
data_deg_test = np.concatenate((data_deg_val, data_deg_test))

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

## get predictions with labels for gephi
#df9core = pd.read_csv('data/allwave1_H1CO3-dense-9core.csv')
#examples = df9core['AID'].astype(str).tolist()
#
#id_actual = []
#for i in range(len(examples)):
#    id_actual.append([examples[i], y[i]])
#
#f = open('9core_actual.csv', 'w')
#for row in id_actual:
#    f.write(row[0]+','+str(row[1])+'\n')
#f.close()
#
#id_lfm_pred = []
#for i in range(len(examples)):
#    id_lfm_pred.append([examples[i], y_pred_lfm_full[i]])
#
#f = open('9core_lfm_pred.csv', 'w')
#for row in id_lfm_pred:
#    f.write(row[0]+','+str(row[1])+'\n')
#f.close()
#
#id_lfm_confusion = []
#for i in range(len(examples)):
#    if y[i] == 1:
#        # true positive
#        if y_pred_lfm_full[i] == 1:
#            id_lfm_confusion.append([examples[i], 'TP'])
#        # false negative
#        elif y_pred_lfm_full[i] == 0:
#            id_lfm_confusion.append([examples[i], 'FN'])
#        else:
#            print 'unrecognized prediction value'
#    elif y[i] == 0:
#        # true negative
#        if y_pred_lfm_full[i] == 0:
#            id_lfm_confusion.append([examples[i], 'TN'])
#        # false positive
#        elif y_pred_lfm_full[i] == 1:
#            id_lfm_confusion.append([examples[i], 'FP'])
#        else:
#            print 'unrecognized prediction value'
#    else:
#        print 'unrecognized actual value'
#        
#f = open('9core_lfm_confusion.csv', 'w')
#for row in id_lfm_confusion:
#    f.write(row[0]+','+str(row[1])+'\n')
#f.close()
#
##id_f_pred = []
##for i in range(len(examples)):
##    id_f_pred.append([examples[i], y_pred_f_full[i]])
##
##f = open('9core_f_pred.csv', 'w')
##for row in id_f_pred:
##    f.write(row[0]+','+str(row[1])+'\n')
##f.close()
#
#sys.exit()


"""
Analyze results
"""

#### Average error per degree ####

'''
error = |confidence - actual value|
'''

def error_per_degree(data):
    err = []
    d_stats = {}
    for row in data:
        if row[0] not in d_stats:
           d_stats[row[0]] = [row[1]]
        else:
           d_stats[row[0]].append(row[1])
    for d in d_stats:
        err.append([d, np.mean(d_stats[d]), scipy.stats.sem(d_stats[d])])
    return err


if test == 'perplexity':
    print '\ncomputing average perplexity per degree...'

    ## Regression ##

    # test data
    # create table whose columns are: degree, confidence error
    res_f_test = np.column_stack((in_deg_test, diff_f_test))
    # save
    f = open(OUT_FILE[adj]+'_f_degree_confidence_test.txt', 'w')
    f.write('Confidence error of predictions on test data set with parameters learned via feature regression\n\n')
    f.write('degree\tconfidence error\n')
    for line in res_f_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average confidence error per degree
    err_f_test = error_per_degree(res_f_test)
    # save
    f = open(OUT_FILE[adj]+'_f_perplexity_test.txt', 'w')
    f.write('Average confidence error per degree\n')
    f.write('Test data set with parameters learned via feature regression\n\n')
    f.write('degree\taverage confidence error\tstandard error\n')
    for line in err_f_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\t\t\t'+str(line[2])+'\n')
    f.close()

    # full data
    # create table whose columns are: degree, confidence error
    res_f_full = np.column_stack((np.reshape(in_deg,diff_f_full.shape), diff_f_full))
    # save
    f = open(OUT_FILE[adj]+'_f_degree_confidence_full.txt', 'w')
    f.write('Confidence error of predictions on full data set with parameters learned via feature regression\n\n')
    f.write('degree\tconfidence error\n')
    for line in res_f_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average confidence error per degree
    err_f_full = error_per_degree(res_f_full)
    # save
    f = open(OUT_FILE[adj]+'_f_perplexity_full.txt', 'w')
    f.write('Average confidence error per degree\n')
    f.write('Full data set with parameters learned via feature regression\n\n')
    f.write('degree\taverage confidence error\tstandard error\n')
    for line in err_f_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\t\t\t'+str(line[2])+'\n')
    f.close()

    ## Neighbor features ##

    # test data
    # create table whose columns are: degree, confidence error
    res_n_test = np.column_stack((in_deg_test, diff_n_test))
    # save
    f = open(OUT_FILE[adj]+'_n_degree_confidence_test.txt', 'w')
    f.write('Confidence error of predictions on test data set with parameters learned with neighbor features\n\n')
    f.write('degree\tconfidence error\n')
    for line in res_n_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average confidence error per degree
    err_n_test = error_per_degree(res_n_test)
    # save
    f = open(OUT_FILE[adj]+'_n_perplexity_test.txt', 'w')
    f.write('Average confidence error per degree\n')
    f.write('Test data set with parameters learned with neighbor features\n\n')
    f.write('degree\taverage confidence error\tstandard error\n')
    for line in err_n_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\t\t\t'+str(line[2])+'\n')
    f.close()

    # full data
    # create table whose columns are: degree, confidence error
    res_n_full = np.column_stack((np.reshape(in_deg,diff_n_full.shape), diff_n_full))
    # save
    f = open(OUT_FILE[adj]+'_n_degree_confidence_full.txt', 'w')
    f.write('Confidence error of predictions on full data set with parameters learned with neighbor features\n\n')
    f.write('degree\tconfidence error\n')
    for line in res_n_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average confidence error per degree
    err_n_full = error_per_degree(res_n_full)
    # save
    f = open(OUT_FILE[adj]+'_n_perplexity_full.txt', 'w')
    f.write('Average confidence error per degree\n')
    f.write('Full data set with parameters learned with neighbor features\n\n')
    f.write('degree\taverage confidence error\tstandard error\n')
    for line in err_n_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\t\t\t'+str(line[2])+'\n')
    f.close()

    ## LFM ##

    # test data
    # create table whose columns are: degree, confidence error
    res_lfm_test = np.column_stack((in_deg_test, diff_lfm_test))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_degree_confidence_test.txt', 'w')
    f.write('Confidence error of predictions on test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tconfidence error\n')
    for line in res_lfm_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average confidence error per degree
    err_lfm_test = error_per_degree(res_lfm_test)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_perplexity_test.txt', 'w')
    f.write('Average confidence error per degree\n')
    f.write('Test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\taverage confidence error\tstandard error\n')
    for line in err_lfm_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\t\t\t'+str(line[2])+'\n')
    f.close()

    # full data
    # create table whose columns are: degree, confidence error
    res_lfm_full = np.column_stack((np.reshape(in_deg,diff_lfm_full.shape), diff_lfm_full))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_degree_confidence_full.txt', 'w')
    f.write('Confidence error of predictions on full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tconfidence error\n')
    for line in res_lfm_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average confidence error per degree
    err_lfm_full = error_per_degree(res_lfm_full)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_perplexity_full.txt', 'w')
    f.write('Average confidence error per degree\n')
    f.write('Full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\taverage confidence error\tstandard error\n')
    for line in err_lfm_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\t\t\t'+str(line[2])+'\n')
    f.close()
    
    # plot average confidence error per degree
    # full data
    plt.figure()
    plt.errorbar([x[0] for x in err_f_full], [x[1] for x in err_f_full], yerr=[x[2] for x in err_f_full], fmt='ys' )
    plt.errorbar([x[0] for x in err_n_full], [x[1] for x in err_n_full], yerr=[x[2] for x in err_n_full], fmt='r^' )
    plt.errorbar([x[0] for x in err_lfm_full], [x[1] for x in err_lfm_full], yerr=[x[2] for x in err_lfm_full], fmt='bo' )
    plt.legend(('regression on F', 'regression on N', 'regression on X'), loc='best')
    plt.xlabel('node degree')
    plt.ylabel('perplexity')
    plt.title('Average perplexity per degree over full data set')
    plt.show()
    #plt.savefig(FIG_FILE[adj]+'_f_lfm_'+method+'_perplexity_full.png')

    # test data
    plt.figure()
    plt.errorbar([x[0] for x in err_f_test], [x[1] for x in err_f_test], yerr=[x[2] for x in err_f_test], fmt='ys' )
    plt.errorbar([x[0] for x in err_n_test], [x[1] for x in err_n_test], yerr=[x[2] for x in err_n_test], fmt='r^' )
    plt.errorbar([x[0] for x in err_lfm_test], [x[1] for x in err_lfm_test], yerr=[x[2] for x in err_lfm_test], fmt='bo' )
    plt.legend(('regression on F', 'regression on N', 'regression on X'), loc='best')
    plt.xlabel('node degree')
    plt.ylabel('perplexity')
    plt.title('Average perplexity per degree')
    #plt.show()
    plt.savefig(FIG_FILE[adj]+'_f_n_lfm_'+method+'_perplexity_test.png')

    print 'done.\n'


#### Average error per data quality ####

'''
error = |confidence - actual value|
'''

if test == 'perplexity-data':
    print '\ncomputing average perplexity per data quality...'

    ## Regression ##

    # test data
    # create table whose columns are: data degree, confidence error
    res_f_test = np.column_stack((data_deg_test, diff_f_test))
    # save
    f = open(OUT_FILE[adj]+'_f_degree_confidence_per_data_test.txt', 'w')
    f.write('Confidence error of predictions on test data set with parameters learned via feature regression\n\n')
    f.write('degree\tconfidence error\n')
    for line in res_f_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average confidence error per degree
    err_f_test = error_per_degree(res_f_test)
    # save
    f = open(OUT_FILE[adj]+'_f_confidence_error_per_data_test.txt', 'w')
    f.write('Average confidence error per data quality\n')
    f.write('Test data set with parameters learned via feature regression\n\n')
    f.write('degree\taverage confidence error\tstandard error\n')
    for line in err_f_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\t\t\t'+str(line[2])+'\n')
    f.close()

    # full data
    # create table whose columns are: degree, confidence error
    res_f_full = np.column_stack((np.reshape(data_deg,diff_f_full.shape), diff_f_full))
    # save
    f = open(OUT_FILE[adj]+'_f_degree_confidence_per_data_full.txt', 'w')
    f.write('Confidence error of predictions on full data set with parameters learned via feature regression\n\n')
    f.write('degree\tconfidence error\n')
    for line in res_f_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average confidence error per degree
    err_f_full = error_per_degree(res_f_full)
    # save
    f = open(OUT_FILE[adj]+'_f_confidence_error_per_data_full.txt', 'w')
    f.write('Average confidence error per data quality\n')
    f.write('Full data set with parameters learned via feature regression\n\n')
    f.write('degree\taverage confidence error\tstandard error\n')
    for line in err_f_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\t\t\t'+str(line[2])+'\n')
    f.close()

    ## LFM ##

    # test data
    # create table whose columns are: degree, confidence error
    res_lfm_test = np.column_stack((data_deg_test, diff_lfm_test))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_degree_confidence_per_data_test.txt', 'w')
    f.write('Confidence error of predictions on test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tconfidence error\n')
    for line in res_lfm_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average confidence error per degree
    err_lfm_test = error_per_degree(res_lfm_test)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_confidence_error_per_data_test.txt', 'w')
    f.write('Average confidence error per data quality\n')
    f.write('Test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\taverage confidence error\tstandard error\n')
    for line in err_lfm_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\t\t\t'+str(line[2])+'\n')
    f.close()

    # full data
    # create table whose columns are: degree, confidence error
    res_lfm_full = np.column_stack((np.reshape(data_deg,diff_lfm_full.shape), diff_lfm_full))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_degree_confidence_per_data_full.txt', 'w')
    f.write('Confidence error of predictions on full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tconfidence error\n')
    for line in res_lfm_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average confidence error per degree
    err_lfm_full = error_per_degree(res_lfm_full)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_confidence_error_per_data_full.txt', 'w')
    f.write('Average confidence error per data quality\n')
    f.write('Full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\taverage confidence error\tstandard error\n')
    for line in err_lfm_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\t\t\t'+str(line[2])+'\n')
    f.close()
    
    # plot average confidence error per degree
    # full data
    plt.figure()
    plt.errorbar([x[0] for x in err_f_full], [x[1] for x in err_f_full], yerr=[x[2] for x in err_f_full], fmt='ys' )
    plt.errorbar([x[0] for x in err_lfm_full], [x[1] for x in err_lfm_full], yerr=[x[2] for x in err_lfm_full], fmt='bo' )
    plt.legend(('regression on F', 'regression on X'), loc='best')
    plt.xlabel('data degree')
    plt.ylabel('Confidence error')
    plt.title('Average confidence error per data quality over full data set')
    plt.show()
    #plt.savefig(FIG_FILE[adj]+'_f_lfm_'+method+'_confidence_error_per_data_full.png')

    # test data
    plt.figure()
    plt.errorbar([x[0] for x in err_f_test], [x[1] for x in err_f_test], yerr=[x[2] for x in err_f_test], fmt='ys' )
    plt.errorbar([x[0] for x in err_lfm_test], [x[1] for x in err_lfm_test], yerr=[x[2] for x in err_lfm_test], fmt='bo' )
    plt.legend(('regression on F', 'regression on X'), loc='best')
    plt.xlabel('data degree')
    plt.ylabel('Confidence error')
    plt.title('Average confidence error per data quality over test data set')
    #plt.show()
    plt.savefig(FIG_FILE[adj]+'_f_lfm_'+method+'_confidence_error_per_data_test.png')

    print 'done.\n'


#### Binary classification accuracy per degree ####

'''
Percentage correct predictions
'''

def acc_per_degree(data):
    acc = []
    d_stats = {}
    for row in data:
        if row[0] not in d_stats:
           d_stats[row[0]] = [row[1]]
        else:
           d_stats[row[0]].append(row[1])
    for d in d_stats:
        acc.append([d, np.mean(d_stats[d]), scipy.stats.sem(d_stats[d])])
    return acc

if test == 'accuracy':
    print '\ncomputing average binary classification accuracy per degree...'

    ## Regression ##

    # test data
    # create table whose columns are: degree, correct prediction
    res_f_test = np.column_stack((in_deg_test, acc_f_test))
    # save
    f = open(OUT_FILE[adj]+'_f_degree_accuracy_test.txt', 'w')
    f.write('Accuracy of predictions on test data set with parameters learned via feature regression\n\n')
    f.write('degree\tcorrect prediction\n')
    for line in res_f_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average binary classfication per degree
    acc_f_test = acc_per_degree(res_f_test)
    # save
    f = open(OUT_FILE[adj]+'_f_accuracy_test.txt', 'w')
    f.write('Average accuracy per degree\n')
    f.write('Test data set with parameters learned via feature regression\n\n')
    f.write('degree\taverage accuracy\tstandard error\n')
    for line in acc_f_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # full data
    # create table whose columns are: degree, correct prediction
    res_f_full = np.column_stack((np.reshape(in_deg,acc_f_full.shape), acc_f_full))
    # save
    f = open(OUT_FILE[adj]+'_f_degree_accuracy_full.txt', 'w')
    f.write('Accuracy of predictions on full data set with parameters learned via feature regression\n\n')
    f.write('degree\tcorrect prediction\n')
    for line in res_f_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average binary classfication per degree
    acc_f_full = acc_per_degree(res_f_full)
    # save
    f = open(OUT_FILE[adj]+'_f_accuracy_full.txt', 'w')
    f.write('Average accuracy per degree\n')
    f.write('Full data set with parameters learned via feature regression\n\n')
    f.write('degree\taverage accuracy\n')
    for line in acc_f_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    ## Neighbor features ##

    # test data
    # create table whose columns are: degree, correct prediction
    res_n_test = np.column_stack((in_deg_test, acc_n_test))
    # save
    f = open(OUT_FILE[adj]+'_n_degree_accuracy_test.txt', 'w')
    f.write('Accuracy of predictions on test data set with parameters learned with neighbor features\n\n')
    f.write('degree\tcorrect prediction\n')
    for line in res_n_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average binary classfication per degree
    acc_n_test = acc_per_degree(res_n_test)
    # save
    f = open(OUT_FILE[adj]+'_n_accuracy_test.txt', 'w')
    f.write('Average accuracy per degree\n')
    f.write('Test data set with parameters learned with neighbor features\n\n')
    f.write('degree\taverage accuracy\tstandard error\n')
    for line in acc_n_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # full data
    # create table whose columns are: degree, correct prediction
    res_n_full = np.column_stack((np.reshape(in_deg,acc_n_full.shape), acc_n_full))
    # save
    f = open(OUT_FILE[adj]+'_n_degree_accuracy_full.txt', 'w')
    f.write('Accuracy of predictions on full data set with parameters learned with neighbor features\n\n')
    f.write('degree\tcorrect prediction\n')
    for line in res_n_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average binary classfication per degree
    acc_n_full = acc_per_degree(res_n_full)
    # save
    f = open(OUT_FILE[adj]+'_n_accuracy_full.txt', 'w')
    f.write('Average accuracy per degree\n')
    f.write('Full data set with parameters learned with neighbor features\n\n')
    f.write('degree\taverage accuracy\n')
    for line in acc_n_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    ## LFM ##

    # test data
    # create table whose columns are: degree, correct prediction
    res_lfm_test = np.column_stack((in_deg_test, acc_lfm_test))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_degree_accuracy_test.txt', 'w')
    f.write('Accuracy of predictions on test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tcorrect prediction\n')
    for line in res_lfm_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average binary classfication per degree
    acc_lfm_test = acc_per_degree(res_lfm_test)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_accuracy_test.txt', 'w')
    f.write('Average accuracy per degree\n')
    f.write('Test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\taverage accuracy\tstandard error\n')
    for line in acc_lfm_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\n')
    f.close()

    # full data
    # create table whose columns are: degree, correct prediction
    res_lfm_full = np.column_stack((np.reshape(in_deg, acc_lfm_full.shape), acc_lfm_full))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_degree_accuracy_full.txt', 'w')
    f.write('Accuracy of predictions on full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tcorrect prediction\n')
    for line in res_lfm_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average binary classfication per degree
    acc_lfm_full = acc_per_degree(res_lfm_full)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_accuracy_full.txt', 'w')
    f.write('Average accuracy per degree\n')
    f.write('Full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\taverage accuracy\n')
    for line in acc_lfm_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # plot average binary classification per degree
    # full data
    plt.figure()
    #plt.plot([x[0] for x in acc_f_full], [x[1] for x in acc_f_full], 'ks' )
    #plt.plot([x[0] for x in acc_lfm_full], [x[1] for x in acc_lfm_full], 'ro' )
    plt.errorbar([x[0] for x in acc_f_full], [x[1] for x in acc_f_full], yerr=[x[2] for x in acc_f_full], fmt='ys' )
    plt.errorbar([x[0] for x in acc_n_full], [x[1] for x in acc_n_full], yerr=[x[2] for x in acc_n_full], fmt='r^' )
    plt.errorbar([x[0] for x in acc_lfm_full], [x[1] for x in acc_lfm_full], yerr=[x[2] for x in acc_lfm_full], fmt='bo' )
    plt.legend(('regression on F', 'regression on N', 'regression on X'), loc='best')
    plt.xlabel('node degree')
    plt.ylabel('Binary classification accuracy')
    plt.title('Accuracy per degree over full data set')
    plt.show()
    #plt.savefig(FIG_FILE[adj]+'_f_lfm_'+method+'_accuracy_full.png')

    # test data
    plt.figure()
    #plt.plot([x[0] for x in acc_f_test], [x[1] for x in acc_f_test], 'ks' )
    #plt.plot([x[0] for x in acc_lfm_test], [x[1] for x in acc_lfm_test], 'ro' )
    plt.errorbar([x[0] for x in acc_f_test], [x[1] for x in acc_f_test], yerr=[x[2] for x in acc_f_test], fmt='ys' )
    plt.errorbar([x[0] for x in acc_n_test], [x[1] for x in acc_n_test], yerr=[x[2] for x in acc_n_test], fmt='r^' )
    plt.errorbar([x[0] for x in acc_lfm_test], [x[1] for x in acc_lfm_test], yerr=[x[2] for x in acc_lfm_test], fmt='bo' )
    plt.legend(('regression on F', 'regression on N', 'regression on X'), loc='best')
    plt.xlabel('node degree')
    plt.ylabel('Binary classification accuracy')
    plt.title('Average accuracy per degree')
    #plt.show()
    plt.savefig(FIG_FILE[adj]+'_f_n_lfm_'+method+'_accuracy_test.png')

    print 'done.\n'


#### Binary classification accuracy per data quality ####

'''
Percentage correct predictions
'''

if test == 'accuracy-data':
    print '\ncomputing average binary classification accuracy per data quality...'

    ## Regression ##

    # test data
    # create table whose columns are: data degree, correct prediction
    res_f_test = np.column_stack((data_deg_test, acc_f_test))
    # save
    f = open(OUT_FILE[adj]+'_f_data_accuracy_test.txt', 'w')
    f.write('Accuracy of predictions on test data set with parameters learned via feature regression\n\n')
    f.write('degree\tcorrect prediction\n')
    for line in res_f_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average binary classfication per data quality
    acc_f_test = acc_per_degree(res_f_test)
    # save
    f = open(OUT_FILE[adj]+'_f_accuracy_per_data_test.txt', 'w')
    f.write('Average accuracy per data quality\n')
    f.write('Test data set with parameters learned via feature regression\n\n')
    f.write('degree\taverage accuracy\n')
    for line in acc_f_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # full data
    # create table whose columns are: data degree, correct prediction
    res_f_full = np.column_stack((np.reshape(data_deg,acc_f_full.shape), acc_f_full))
    # save
    f = open(OUT_FILE[adj]+'_f_data_accuracy_full.txt', 'w')
    f.write('Accuracy of predictions on full data set with parameters learned via feature regression\n\n')
    f.write('degree\tcorrect prediction\n')
    for line in res_f_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average binary classfication per data quality
    acc_f_full = acc_per_degree(res_f_full)
    # save
    f = open(OUT_FILE[adj]+'_f_accuracy_per_data_full.txt', 'w')
    f.write('Average accuracy per data quality\n')
    f.write('Full data set with parameters learned via feature regression\n\n')
    f.write('degree\taverage accuracy\n')
    for line in acc_f_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    ## LFM ##

    # test data
    # create table whose columns are: data degree, correct prediction
    res_lfm_test = np.column_stack((data_deg_test, acc_lfm_test))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_data_accuracy_test.txt', 'w')
    f.write('Accuracy of predictions on test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tcorrect prediction\n')
    for line in res_lfm_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average binary classfication per data quality
    acc_lfm_test = acc_per_degree(res_lfm_test)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_accuracy_per_data_test.txt', 'w')
    f.write('Average accuracy per data quality\n')
    f.write('Test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\taverage accuracy\n')
    for line in acc_lfm_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # full data
    # create table whose columns are: data degree, correct prediction
    res_lfm_full = np.column_stack((np.reshape(data_deg, acc_lfm_full.shape), acc_lfm_full))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_data_accuracy_full.txt', 'w')
    f.write('Accuracy of predictions on full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tcorrect prediction\n')
    for line in res_lfm_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\n')
    f.close()

    # get average binary classification per degree
    acc_lfm_full = acc_per_degree(res_lfm_full)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_accuracy_per_data_full.txt', 'w')
    f.write('Average accuracy per data quality\n')
    f.write('Full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\taverage accuracy\n')
    for line in acc_lfm_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # plot average binary classification per degree
    # full data
    plt.figure()
    plt.errorbar([x[0] for x in acc_f_full], [x[1] for x in acc_f_full], yerr=[x[2] for x in acc_f_full], fmt='ys' )
    plt.errorbar([x[0] for x in acc_lfm_full], [x[1] for x in acc_lfm_full], yerr=[x[2] for x in acc_lfm_full], fmt='bo' )
    plt.legend(('regression on F', 'regression on X'), loc='best')
    plt.xlabel('node degree')
    plt.ylabel('Binary classfication accuracy')
    plt.title('Accuracy per data degree over full data set')
    plt.show()
    #plt.savefig(FIG_FILE[adj]+'_f_lfm_'+method+'_BCR_per_data_full.png')

    # test data
    plt.figure()
    #plt.plot([x[0] for x in acc_f_test], [x[1] for x in acc_f_test], 'ks' )
    #plt.plot([x[0] for x in acc_lfm_test], [x[1] for x in acc_lfm_test], 'ro' )
    plt.errorbar([x[0] for x in acc_f_test], [x[1] for x in acc_f_test], yerr=[x[2] for x in acc_f_test], fmt='ys' )
    plt.errorbar([x[0] for x in acc_lfm_test], [x[1] for x in acc_lfm_test], yerr=[x[2] for x in acc_lfm_test], fmt='bo' )
    plt.legend(('regression on F', 'regression on X'), loc='best')
    plt.xlabel('node degree')
    plt.ylabel('Binary classfication accuracy')
    plt.title('Accuracy per data degree over test data set')
    #plt.show()
    plt.savefig(FIG_FILE[adj]+'_f_lfm_'+method+'_accuracy_per_data_test.png')

    print 'done.\n'


#### Balanced classification rate per degree ####

'''
BCR = 0.5*(TP/(TP+FN) + TN/(TN+FP))
'''

def bcr_per_degree(data):
    pos = sum(data[:,2])
    neg = data.shape[0] - pos
    bcr = []
    for d in range(int(max(data[:,0]))):
        TP, FP, TN, FN = 0, 0, 0, 0
        for row in data:
            if row[0] == d: # check degree
                # predicted positive
                if row[1] == 1:
                    # actual positive
                    if row[2] == 1:
                        TP += 1.0
                    # actual negative
                    else:
                        FP += 1.0
                # predicted negative
                else:
                    # actual positive
                    if row[2] == 1:
                        FN += 1.0
                    # actual negative
                    else:
                        TN += 1.0
        tp = TP/pos
        tn = TN/neg
        bcr_d = 0.5*(tp+tn)
        bcr.append([d, bcr_d])
    return bcr

if test == 'balanced-classification':
    print '\ncomputing balanced classification rate per degree...'

    ## Regression ##

    # test data
    # create table whose columns are: degree, prediction, actual value
    res_f_test = np.column_stack((in_deg_test, y_pred_f_test, y_test))
    # save
    f = open(OUT_FILE[adj]+'_f_degree_prediction_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned via feature regression\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_f_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BCR per degree
    bcr_f_test = bcr_per_degree(res_f_test)
    # save
    f = open(OUT_FILE[adj]+'_f_bcr_test.txt', 'w')
    f.write('Balanced classification rate per degree\n')
    f.write('Test data set with parameters learned via feature regression\n\n')
    f.write('degree\tBCR\n')
    for line in bcr_f_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # full data
    # create table whose columns are: degree, prediction, actual value
    res_f_full = np.column_stack((np.reshape(in_deg, y.shape), y_pred_f_full, y))
    # save
    f = open(OUT_FILE[adj]+'_f_degree_prediction_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned via feature regression\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_f_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BCR per degree
    bcr_f_full = bcr_per_degree(res_f_full)
    # save
    f = open(OUT_FILE[adj]+'_f_bcr_full.txt', 'w')
    f.write('Balanced classification rate per degree\n')
    f.write('Full data set with parameters learned via feature regression\n\n')
    f.write('degree\tBCR\n')
    for line in bcr_f_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    ## Neighbor features ##

    # test data
    # create table whose columns are: degree, prediction, actual value
    res_n_test = np.column_stack((in_deg_test, y_pred_n_test, y_test))
    # save
    f = open(OUT_FILE[adj]+'_n_degree_prediction_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned with neighbor features\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_n_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BCR per degree
    bcr_n_test = bcr_per_degree(res_n_test)
    # save
    f = open(OUT_FILE[adj]+'_n_bcr_test.txt', 'w')
    f.write('Balanced classification rate per degree\n')
    f.write('Test data set with parameters learned with neighbor features\n\n')
    f.write('degree\tBCR\n')
    for line in bcr_n_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # full data
    # create table whose columns are: degree, prediction, actual value
    res_n_full = np.column_stack((np.reshape(in_deg, y.shape), y_pred_n_full, y))
    # save
    f = open(OUT_FILE[adj]+'_n_degree_prediction_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned with neighbor features\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_n_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BCR per degree
    bcr_n_full = bcr_per_degree(res_n_full)
    # save
    f = open(OUT_FILE[adj]+'_n_bcr_full.txt', 'w')
    f.write('Balanced classification rate per degree\n')
    f.write('Full data set with parameters learned with neighbor features\n\n')
    f.write('degree\tBCR\n')
    for line in bcr_n_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    ## LFM ##

    # test data
    # create table whose columns are: degree, prediction, actual value
    res_lfm_test = np.column_stack((in_deg_test, y_pred_lfm_test, y_test))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_degree_prediction_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_lfm_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BCR per degree
    bcr_lfm_test = bcr_per_degree(res_lfm_test)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_bcr_test.txt', 'w')
    f.write('Balanced classification rate per degree\n')
    f.write('Test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tBCR\n')
    for line in bcr_lfm_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # full data
    # create table whose columns are: degree, prediction, actual value
    res_lfm_full = np.column_stack((np.reshape(in_deg, y.shape), y_pred_lfm_full, y))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_degree_prediction_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_lfm_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BCR per degree
    bcr_lfm_full = bcr_per_degree(res_lfm_full)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_bcr_full.txt', 'w')
    f.write('Balanced classification rate per degree\n')
    f.write('Full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tBER\n')
    for line in bcr_lfm_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # plot BCR per degree
    # full data
    plt.figure()
    plt.plot([x[0] for x in bcr_f_full], [x[1] for x in bcr_f_full], 'ys-' )
    plt.plot([x[0] for x in bcr_n_full], [x[1] for x in bcr_n_full], 'r^-' )
    plt.plot([x[0] for x in bcr_lfm_full], [x[1] for x in bcr_lfm_full], 'bo-' )
    plt.legend(('regression on F', 'regression on N', 'regression on X'), loc='best')
    plt.xlabel('node degree')
    plt.ylabel('Balanced classification rate')
    plt.title('Balanced classification rate per degree over full data set')
    plt.show()
    #plt.savefig(FIG_FILE[adj]+'_f_lfm_'+method+'_ber_full.png')

    # test data
    plt.figure()
    plt.plot([x[0] for x in bcr_f_test], [x[1] for x in bcr_f_test], 'ys-' )
    plt.plot([x[0] for x in bcr_n_test], [x[1] for x in bcr_n_test], 'r^-' )
    plt.plot([x[0] for x in bcr_lfm_test], [x[1] for x in bcr_lfm_test], 'bo-' )
    plt.legend(('regression on F', 'regression on N', 'regression on X'), loc='best')
    plt.xlabel('node degree')
    plt.ylabel('Balanced classification rate')
    plt.title('Balanced classification rate per degree over test data set')
    #plt.show()
    plt.savefig(FIG_FILE[adj]+'_f_n_lfm_'+method+'_bcr_test.png')

    print 'done.\n'


#### Balanced error rate per data quality ####

'''
BCR = 0.5*(FP/(TN+FP) + FN/(FN+TP))
'''

if test == 'balanced-classification-data':
    print '\ncomputing balanced classification rate per data quality...'

    ## Regression ##

    # test data
    # create table whose columns are: data degree, prediction, actual value
    res_f_test = np.column_stack((data_deg_test, y_pred_f_test, y_test))
    # save
    f = open(OUT_FILE[adj]+'_f_data_prediction_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned via feature regression\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_f_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BCR per degree
    bcr_f_test = bcr_per_degree(res_f_test)
    # save
    f = open(OUT_FILE[adj]+'_f_bcr_per_data_test.txt', 'w')
    f.write('Balanced classification rate per data quality\n')
    f.write('Test data set with parameters learned via feature regression\n\n')
    f.write('degree\tBER\n')
    for line in bcr_f_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # full data
    # create table whose columns are: data degree, prediction, actual value
    res_f_full = np.column_stack((np.reshape(data_deg, y.shape), y_pred_f_full, y))
    # save
    f = open(OUT_FILE[adj]+'_f_data_prediction_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned via feature regression\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_f_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BCR per degree
    bcr_f_full = bcr_per_degree(res_f_full)
    # save
    f = open(OUT_FILE[adj]+'_f_bcr_per_data_full.txt', 'w')
    f.write('Balanced classification rate per data quality\n')
    f.write('Full data set with parameters learned via feature regression\n\n')
    f.write('degree\tBCR\n')
    for line in bcr_f_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    ## LFM ##

    # test data
    # create table whose columns are: data degree, prediction, actual value
    res_lfm_test = np.column_stack((data_deg_test, y_pred_lfm_test, y_test))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_data_prediction_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_lfm_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BCR per degree
    bcr_lfm_test = bcr_per_degree(res_lfm_test)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_bcr_per_data_test.txt', 'w')
    f.write('Balanced classification rate per data quality\n')
    f.write('Test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tBCR\n')
    for line in bcr_lfm_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # full data
    # create table whose columns are: data degree, prediction, actual value
    res_lfm_full = np.column_stack((np.reshape(data_deg, y.shape), y_pred_lfm_full, y))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_data_prediction_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_lfm_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BCR per degree
    bcr_lfm_full = bcr_per_degree(res_lfm_full)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_bcr_per_data_full.txt', 'w')
    f.write('Balanced classification rate per data quality\n')
    f.write('Full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tBCR\n')
    for line in bcr_lfm_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # plot BCR per degree
    # full data
    plt.figure()
    plt.plot([x[0] for x in bcr_f_full], [x[1] for x in bcr_f_full], 'ys-' )
    plt.plot([x[0] for x in bcr_lfm_full], [x[1] for x in bcr_lfm_full], 'bo-' )
    plt.legend(('regression on F', 'regression on X'), loc='best')
    plt.xlabel('Data deficiency')
    plt.ylabel('Balanced error rate')
    plt.title('Balanced error rate per data deficiency over full data set')
    plt.show()
    #plt.savefig(FIG_FILE[adj]+'_f_lfm_'+method+'_bcr_per_data_full.png')

    # test data
    plt.figure()
    plt.plot([x[0] for x in bcr_f_test], [x[1] for x in bcr_f_test], 'ys-' )
    plt.plot([x[0] for x in bcr_lfm_test], [x[1] for x in bcr_lfm_test], 'bo-' )
    plt.legend(('regression on F', 'regression on X'), loc='best')
    plt.xlabel('Data deficiency')
    plt.ylabel('Balanced error rate')
    plt.title('Balanced error rate per data deficiency over test data set')
    #plt.show()
    plt.savefig(FIG_FILE[adj]+'_f_lfm_'+method+'_bcr_per_data_test.png')

    print 'done.\n'


#### Balanced error rate per degree ####

'''
BER = 0.5*(FP/(TN+FP) + FN/(FN+TP))
'''

def ber_per_degree(data):
    pos = sum(data[:,2])
    neg = data.shape[0] - pos
    ber = []
    for d in range(int(max(data[:,0]))):
        TP, FP, TN, FN = 0, 0, 0, 0
        for row in data:
            if row[0] == d: # check degree
                # predicted positive
                if row[1] == 1:
                    # actual positive
                    if row[2] == 1:
                        TP += 1.0
                    # actual negative
                    else:
                        FP += 1.0
                # predicted negative
                else:
                    # actual positive
                    if row[2] == 1:
                        FN += 1.0
                    # actual negative
                    else:
                        TN += 1.0
        #try:
        #    fp = FP/(TN+FP)
        #except(ZeroDivisionError):
        #    if FP == 0 and TN == 0:
        #        fp = 0.0
        #    else:
        #        print 'unknown issue'
        #        print 'TP, FP, TN, FN:', TP, FP, TN, FN
        fp = FP/neg
        #try:
        #    fn = FN/(FN+TP)
        #except(ZeroDivisionError):
        #    if FN == 0 and TP == 0:
        #        fn = 0.0
        #    else:
        #        print 'unknown issue'
        #        print 'TP, FP, TN, FN:', TP, FP, TN, FN
        fn = FN/pos
        ber_d = 0.5*(fp+fn) 
        ber.append([d, ber_d])
    return ber

if test == 'balanced-error':
    print '\ncomputing balanced error per degree...'

    ## Regression ##

    # test data
    # create table whose columns are: degree, prediction, actual value
    res_f_test = np.column_stack((in_deg_test, y_pred_f_test, y_test))
    # save
    f = open(OUT_FILE[adj]+'_f_degree_prediction_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned via feature regression\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_f_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BER per degree
    ber_f_test = ber_per_degree(res_f_test)
    # save
    f = open(OUT_FILE[adj]+'_f_ber_test.txt', 'w')
    f.write('Balanced error rate per degree\n')
    f.write('Test data set with parameters learned via feature regression\n\n')
    f.write('degree\tBER\n')
    for line in ber_f_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # full data
    # create table whose columns are: degree, prediction, actual value
    res_f_full = np.column_stack((np.reshape(in_deg, y.shape), y_pred_f_full, y))
    # save
    f = open(OUT_FILE[adj]+'_f_degree_prediction_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned via feature regression\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_f_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BER per degree
    ber_f_full = ber_per_degree(res_f_full)
    # save
    f = open(OUT_FILE[adj]+'_f_ber_full.txt', 'w')
    f.write('Balanced error rate per degree\n')
    f.write('Full data set with parameters learned via feature regression\n\n')
    f.write('degree\tBER\n')
    for line in ber_f_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    ## Neighbor features ##

    # test data
    # create table whose columns are: degree, prediction, actual value
    res_n_test = np.column_stack((in_deg_test, y_pred_n_test, y_test))
    # save
    f = open(OUT_FILE[adj]+'_n_degree_prediction_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned with neighbor features\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_n_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BER per degree
    ber_n_test = ber_per_degree(res_n_test)
    # save
    f = open(OUT_FILE[adj]+'_n_ber_test.txt', 'w')
    f.write('Balanced error rate per degree\n')
    f.write('Test data set with parameters learned with neighbor features\n\n')
    f.write('degree\tBER\n')
    for line in ber_n_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # full data
    # create table whose columns are: degree, prediction, actual value
    res_n_full = np.column_stack((np.reshape(in_deg, y.shape), y_pred_n_full, y))
    # save
    f = open(OUT_FILE[adj]+'_n_degree_prediction_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned with neighbor features\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_n_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BER per degree
    ber_n_full = ber_per_degree(res_n_full)
    # save
    f = open(OUT_FILE[adj]+'_n_ber_full.txt', 'w')
    f.write('Balanced error rate per degree\n')
    f.write('Full data set with parameters learned with neighbor features\n\n')
    f.write('degree\tBER\n')
    for line in ber_n_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    ## LFM ##

    # test data
    # create table whose columns are: degree, prediction, actual value
    res_lfm_test = np.column_stack((in_deg_test, y_pred_lfm_test, y_test))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_degree_prediction_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_lfm_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BER per degree
    ber_lfm_test = ber_per_degree(res_lfm_test)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_ber_test.txt', 'w')
    f.write('Balanced error rate per degree\n')
    f.write('Test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tBER\n')
    for line in ber_lfm_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # full data
    # create table whose columns are: degree, prediction, actual value
    res_lfm_full = np.column_stack((np.reshape(in_deg, y.shape), y_pred_lfm_full, y))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_degree_prediction_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_lfm_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BER per degree
    ber_lfm_full = ber_per_degree(res_lfm_full)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_ber_full.txt', 'w')
    f.write('Balanced error rate per degree\n')
    f.write('Full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tBER\n')
    for line in ber_lfm_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # plot BER per degree
    # full data
    plt.figure()
    plt.plot([x[0] for x in ber_f_full], [x[1] for x in ber_f_full], 'ys-' )
    plt.plot([x[0] for x in ber_n_full], [x[1] for x in ber_n_full], 'r^-' )
    plt.plot([x[0] for x in ber_lfm_full], [x[1] for x in ber_lfm_full], 'bo-' )
    plt.legend(('regression on F', 'regression on N', 'regression on X'), loc='best')
    plt.xlabel('node degree')
    plt.ylabel('Balanced error rate')
    plt.title('Balanced error rate per degree over full data set')
    plt.show()
    #plt.savefig(FIG_FILE[adj]+'_f_lfm_'+method+'_ber_full.png')

    # test data
    plt.figure()
    plt.plot([x[0] for x in ber_f_test], [x[1] for x in ber_f_test], 'ys-' )
    plt.plot([x[0] for x in ber_n_test], [x[1] for x in ber_n_test], 'r^-' )
    plt.plot([x[0] for x in ber_lfm_test], [x[1] for x in ber_lfm_test], 'bo-' )
    plt.legend(('regression on F', 'regression on N', 'regression on X'), loc='best')
    plt.xlabel('node degree')
    plt.ylabel('Balanced error rate')
    plt.title('Balanced error rate per degree over test data set')
    #plt.show()
    plt.savefig(FIG_FILE[adj]+'_f_n_lfm_'+method+'_ber_test.png')

    print 'done.\n'


#### Balanced error rate per data quality ####

'''
BER = 0.5*(FP/(TN+FP) + FN/(FN+TP))
'''

if test == 'balanced-error-data':
    print '\ncomputing balanced error per data quality...'

    ## Regression ##

    # test data
    # create table whose columns are: data degree, prediction, actual value
    res_f_test = np.column_stack((data_deg_test, y_pred_f_test, y_test))
    # save
    f = open(OUT_FILE[adj]+'_f_data_prediction_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned via feature regression\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_f_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BER per degree
    ber_f_test = ber_per_degree(res_f_test)
    # save
    f = open(OUT_FILE[adj]+'_f_ber_per_data_test.txt', 'w')
    f.write('Balanced error rate per data quality\n')
    f.write('Test data set with parameters learned via feature regression\n\n')
    f.write('degree\tBER\n')
    for line in ber_f_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # full data
    # create table whose columns are: data degree, prediction, actual value
    res_f_full = np.column_stack((np.reshape(data_deg, y.shape), y_pred_f_full, y))
    # save
    f = open(OUT_FILE[adj]+'_f_data_prediction_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned via feature regression\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_f_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BER per degree
    ber_f_full = ber_per_degree(res_f_full)
    # save
    f = open(OUT_FILE[adj]+'_f_ber_per_data_full.txt', 'w')
    f.write('Balanced error rate per data quality\n')
    f.write('Full data set with parameters learned via feature regression\n\n')
    f.write('degree\tBER\n')
    for line in ber_f_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    ## LFM ##

    # test data
    # create table whose columns are: data degree, prediction, actual value
    res_lfm_test = np.column_stack((data_deg_test, y_pred_lfm_test, y_test))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_data_prediction_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_lfm_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BER per degree
    ber_lfm_test = ber_per_degree(res_lfm_test)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_ber_per_data_test.txt', 'w')
    f.write('Balanced error rate per data quality\n')
    f.write('Test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tBER\n')
    for line in ber_lfm_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # full data
    # create table whose columns are: data degree, prediction, actual value
    res_lfm_full = np.column_stack((np.reshape(data_deg, y.shape), y_pred_lfm_full, y))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_data_prediction_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_lfm_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get BER per degree
    ber_lfm_full = ber_per_degree(res_lfm_full)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_ber_per_data_full.txt', 'w')
    f.write('Balanced error rate per data quality\n')
    f.write('Full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tBER\n')
    for line in ber_lfm_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # plot BER per degree
    # full data
    plt.figure()
    plt.plot([x[0] for x in ber_f_full], [x[1] for x in ber_f_full], 'ys-' )
    plt.plot([x[0] for x in ber_lfm_full], [x[1] for x in ber_lfm_full], 'bo-' )
    plt.legend(('regression on F', 'regression on X'), loc='best')
    plt.xlabel('Data deficiency')
    plt.ylabel('Balanced error rate')
    plt.title('Balanced error rate per data deficiency over full data set')
    plt.show()
    #plt.savefig(FIG_FILE[adj]+'_f_lfm_'+method+'_ber_per_data_full.png')

    # test data
    plt.figure()
    plt.plot([x[0] for x in ber_f_test], [x[1] for x in ber_f_test], 'ys-' )
    plt.plot([x[0] for x in ber_lfm_test], [x[1] for x in ber_lfm_test], 'bo-' )
    plt.legend(('regression on F', 'regression on X'), loc='best')
    plt.xlabel('Data deficiency')
    plt.ylabel('Balanced error rate')
    plt.title('Balanced error rate per data deficiency over test data set')
    #plt.show()
    plt.savefig(FIG_FILE[adj]+'_f_lfm_'+method+'_ber_per_data_test.png')

    print 'done.\n'


#### F1 per degree ####

'''
F1 = 2*precision*recall/(precision+recall)
'''

def F1_per_degree(data):
    pos = sum(data[:,2])
    F1 = []
    for d in range(int(max(data[:,0]))):
        TP, FP, TN, FN = 0, 0, 0, 0
        for row in data:
            if row[0] == d: # check degree
                # predicted positive
                if row[1] == 1:
                    # actual positive
                    if row[2] == 1:
                        TP += 1.0
                    # actual negative
                    else:
                        FP += 1.0
                # predicted negative
                else:
                    # actual positive
                    if row[2] == 1:
                        FN += 1.0
                    # actual negative
                    else:
                        TN += 1.0
        try:
            precision = TP/(TP+FP)
        except(ZeroDivisionError):
            if TP == 0 and FP == 0:
                precision = 0.0
            else:
                print 'unknown issue'
                print 'TP, FP, FN:', TP, FP, FN
        #try:
        #    recall = TP/(TP+FN)
        #except(ZeroDivisionError):
        #    if TP == 0 and FN == 0:
        #        recall = 0.0
        #    else:
        #        print 'unknown issue'
        #        print 'TP, FP, FN:', TP, FP, FN
        recall = TP/pos
        try:
            F1_d = 2*precision*recall/(precision+recall) 
        except(ZeroDivisionError):
            if precision == 0 and recall == 0:
                F1_d = 0.0
            else:
                print 'unknown issue'
                print 'precision, recall:', precision, recall
        F1.append([d, F1_d, precision, recall])
    return F1

if test == 'F1':
    print '\ncomputing F1 per degree...'

    ## Regression ##

    # test data
    # create table whose columns are: degree, prediction, actual value
    res_f_test = np.column_stack((in_deg_test, y_pred_f_test, y_test))
    # save
    f = open(OUT_FILE[adj]+'_f_degree_prediction_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned via feature regression\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_f_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get F1 per degree
    F1_f_test = F1_per_degree(res_f_test)
    # save
    f = open(OUT_FILE[adj]+'_f_F1_test.txt', 'w')
    f.write('F1 score per degree\n')
    f.write('Test data set with parameters learned via feature regression\n\n')
    f.write('degree\tF1\n')
    for line in F1_f_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\t'+str(line[3])+'\n')
    f.close()

    # full data
    # create table whose columns are: degree, prediction, actual value
    res_f_full = np.column_stack((np.reshape(in_deg, y.shape), y_pred_f_full, y))
    # save
    f = open(OUT_FILE[adj]+'_f_degree_prediction_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned via feature regression\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_f_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get F1 per degree
    F1_f_full = F1_per_degree(res_f_full)
    # save
    f = open(OUT_FILE[adj]+'_f_F1_full.txt', 'w')
    f.write('F1 score per degree\n')
    f.write('Full data set with parameters learned via feature regression\n\n')
    f.write('degree\tF1\n')
    for line in F1_f_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    ## Neighbor features ##

    # test data
    # create table whose columns are: degree, prediction, actual value
    res_n_test = np.column_stack((in_deg_test, y_pred_n_test, y_test))
    # save
    f = open(OUT_FILE[adj]+'_n_degree_prediction_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned with neighbor features\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_n_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get F1 per degree
    F1_n_test = F1_per_degree(res_n_test)
    # save
    f = open(OUT_FILE[adj]+'_n_F1_test.txt', 'w')
    f.write('F1 score per degree\n')
    f.write('Test data set with parameters learned with neighbor features\n\n')
    f.write('degree\tF1\n')
    for line in F1_n_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\t'+str(line[3])+'\n')
    f.close()

    # full data
    # create table whose columns are: degree, prediction, actual value
    res_n_full = np.column_stack((np.reshape(in_deg, y.shape), y_pred_n_full, y))
    # save
    f = open(OUT_FILE[adj]+'_n_degree_prediction_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned with neighbor features\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_n_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get F1 per degree
    F1_n_full = F1_per_degree(res_n_full)
    # save
    f = open(OUT_FILE[adj]+'_n_F1_full.txt', 'w')
    f.write('F1 score per degree\n')
    f.write('Full data set with parameters learned with neighbor features\n\n')
    f.write('degree\tF1\n')
    for line in F1_n_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    ## LFM ##

    # test data
    # create table whose columns are: degree, prediction, actual value
    res_lfm_test = np.column_stack((in_deg_test, y_pred_lfm_test, y_test))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_degree_prediction_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_lfm_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get F1 per degree
    F1_lfm_test = F1_per_degree(res_lfm_test)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_F1_test.txt', 'w')
    f.write('F1 score per degree\n')
    f.write('Test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tF1\n')
    for line in F1_lfm_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # full data
    # create table whose columns are: degree, prediction, actual value
    res_lfm_full = np.column_stack((np.reshape(in_deg, y.shape), y_pred_lfm_full, y))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_degree_prediction_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_lfm_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get F1 per degree
    F1_lfm_full = F1_per_degree(res_lfm_full)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_F1_full.txt', 'w')
    f.write('F1 score per degree\n')
    f.write('Full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tF1\n')
    for line in F1_lfm_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # plot F1 per degree
    # full data
    plt.figure()
    plt.plot([x[0] for x in F1_f_full], [x[1] for x in F1_f_full], 'ys-' )
    plt.plot([x[0] for x in F1_n_full], [x[1] for x in F1_n_full], 'r^-' )
    plt.plot([x[0] for x in F1_lfm_full], [x[1] for x in F1_lfm_full], 'bo-' )
    plt.legend(('regression on F', 'regression on N', 'regression on X'), loc='best')
    plt.xlabel('node degree')
    plt.ylabel('F1 score')
    plt.title('F1 score per degree over full data set')
    plt.show()
    #plt.savefig(FIG_FILE[adj]+'_f_lfm_'+method+'_f1_full.png')

    # test data
    plt.figure()
    plt.plot([x[0] for x in F1_f_test], [x[1] for x in F1_f_test], 'ys-' )
    plt.plot([x[0] for x in F1_n_test], [x[1] for x in F1_n_test], 'r^-' )
    plt.plot([x[0] for x in F1_lfm_test], [x[1] for x in F1_lfm_test], 'bo-' )
    plt.legend(('regression on F', 'regression on N', 'regression on X'), loc='best')
    plt.xlabel('node degree')
    plt.ylabel('F1 score')
    plt.title('F1 score per degree over test data set')
    #plt.show()
    plt.savefig(FIG_FILE[adj]+'_f_n_lfm_'+method+'_f1_test.png')

    print 'done.\n'


#### F1 per data quality ####

'''
F1 = 2*precision*recall/(precision+recall)
'''

if test == 'F1-data':
    print '\ncomputing F1 per data quality...'

    ## Regression ##

    # test data
    # create table whose columns are: data degree, prediction, actual value
    res_f_test = np.column_stack((data_deg_test, y_pred_f_test, y_test))
    # save
    f = open(OUT_FILE[adj]+'_f_data_prediction_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned via feature regression\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_f_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get F1 per data quality
    F1_f_test = F1_per_degree(res_f_test)
    # save
    f = open(OUT_FILE[adj]+'_f_F1_per_data_test.txt', 'w')
    f.write('F1 score per data quality\n')
    f.write('Test data set with parameters learned via feature regression\n\n')
    f.write('degree\tF1\n')
    for line in F1_f_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # full data
    # create table whose columns are: data degree, prediction, actual value
    res_f_full = np.column_stack((np.reshape(data_deg, y.shape), y_pred_f_full, y))
    # save
    f = open(OUT_FILE[adj]+'_f_data_prediction_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned via feature regression\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_f_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get F1 per data quality
    F1_f_full = F1_per_degree(res_f_full)
    # save
    f = open(OUT_FILE[adj]+'_f_F1_per_data_full.txt', 'w')
    f.write('F1 score per data quality\n')
    f.write('Full data set with parameters learned via feature regression\n\n')
    f.write('degree\tF1\n')
    for line in F1_f_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    ## LFM ##

    # test data
    # create table whose columns are: data degree, prediction, actual value
    res_lfm_test = np.column_stack((data_deg_test, y_pred_lfm_test, y_test))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_data_prediction_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_lfm_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get F1 per data quality
    F1_lfm_test = F1_per_degree(res_lfm_test)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_F1_per_data_test.txt', 'w')
    f.write('F1 score per data quality\n')
    f.write('Test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tF1\n')
    for line in F1_lfm_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # full data
    # create table whose columns are: data degree, prediction, actual value
    res_lfm_full = np.column_stack((np.reshape(data_deg, y.shape), y_pred_lfm_full, y))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_data_prediction_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tprediction\tactual value\n')
    for line in res_lfm_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get F1 per data quality
    F1_lfm_full = F1_per_degree(res_lfm_full)
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_F1_per_data_full.txt', 'w')
    f.write('F1 score per data quality\n')
    f.write('Full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('degree\tF1\n')
    for line in F1_lfm_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\n')
    f.close()

    # plot F1 per degree
    # full data
    plt.figure()
    plt.plot([x[0] for x in F1_f_full], [x[1] for x in F1_f_full], 'ys-' )
    plt.plot([x[0] for x in F1_lfm_full], [x[1] for x in F1_lfm_full], 'bo-' )
    plt.legend(('regression on F', 'regression on X'), loc='best')
    plt.xlabel('Data Deficiency')
    plt.ylabel('F1 score')
    plt.title('F1 score per data deficiency over full data set')
    plt.show()
    #plt.savefig(FIG_FILE[adj]+'_f_lfm_'+method+'_f1_per_data_full.png')

    # test data
    plt.figure()
    plt.plot([x[0] for x in F1_f_test], [x[1] for x in F1_f_test], 'ys-' )
    plt.plot([x[0] for x in F1_lfm_test], [x[1] for x in F1_lfm_test], 'bo-' )
    plt.legend(('regression on F', 'regression on X'), loc='best')
    plt.xlabel('Data Deficiency')
    plt.ylabel('F1 score')
    plt.title('F1 score per data deficiency over test data set')
    #plt.show()
    plt.savefig(FIG_FILE[adj]+'_f_lfm_'+method+'_f1_per_data_test.png')

    print 'done.\n'

#### Precision Recall Curve ####

'''
Precision/recall of top k examples as ranked by |confidence - 0.5|
'''

def precision_recall_atk(data, k, interval=1):
    # count positive examples
    pos = sum(data[:,2])
    pr_curve = []
    for i in range(1,k,interval): # @k
        TP, FP, FN = 0, 0, 0
        for j in range(i+1): # j in top k
            row = data[j, :]
            # predicted positive
            if row[1] == 1:
                # actual positive
                if row[2] == 1:
                    TP += 1.0
                # actual negative
                else:
                    FP += 1.0
            ## predicted negative
            #else:
            #    # actual positive
            #    if row[2] == 1:
            #        FN += 1.0
        try:
            precision = TP/(TP+FP)
        except(ZeroDivisionError):
            if TP == 0 and FP == 0:
                precision = 0.0
            else:
                print 'unknown issue'
                print 'TP, FP, FN:', TP, FP, FN
        #try:
        #    recall = TP/(TP+FN)
        #except(ZeroDivisionError):
        #    if TP == 0 and FN == 0:
        #        recall = 0.0
        #    else:
        #        print 'unknown issue'
        #        print 'TP, FP, FN:', TP, FP, FN
        recall = TP/pos
        pr_curve.append([i+1, recall, precision])
    return pr_curve

if test == 'precision-recall':
    print '\ncreating precision/recall curves...'

    ## Regression ##

    # test data
    # create table whose columns are: |confidence-0.5|, prediction, actual value
    res_f_test = np.column_stack((np.absolute(y_pred_raw_f_test-0.5), y_pred_f_test, y_test))
    # sort by decreasing |confidence-0.5|
    res_f_sorted_test = res_f_test[res_f_test[:,0].argsort()[::-1]]
    # save
    f = open(OUT_FILE[adj]+'_f_prediction_table_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned via feature regression\n\n')
    f.write('|confidence - 0.5|\tprediction\tactual value\n')
    for line in res_f_sorted_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()
   
    # get precision/recall for top k
    pr_curve_f_test = np.array(precision_recall_atk(res_f_sorted_test, len(y_test), 1))
    # save
    f = open(OUT_FILE[adj]+'_f_recall_precision_test.txt', 'w')
    f.write('Recall/Precision for top k points per |confidence-0.5|\n')
    f.write('Test data set with parameters learned via feature regression\n\n')
    f.write('k\trecall\tprecision\n')
    for line in pr_curve_f_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\n')
    f.close()

    # full data
    # create table whose columns are: |confidence-0.5|, prediction, actual value
    res_f_full = np.column_stack((np.absolute(y_pred_raw_f_full-0.5), y_pred_f_full, y))
    # sort by decreasing |confidence-0.5|
    res_f_sorted_full = res_f_full[res_f_full[:,0].argsort()[::-1]]
    # save
    f = open(OUT_FILE[adj]+'_f_prediction_table_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned via feature regression\n\n')
    f.write('|confidence - 0.5|\tprediction\tactual value\n')
    for line in res_f_sorted_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()
   
    # get precision/recall for top k
    pr_curve_f_full = np.array(precision_recall_atk(res_f_sorted_full, len(y), 1))
    # save
    f = open(OUT_FILE[adj]+'_f_recall_precision_full.txt', 'w')
    f.write('Recall/Precision for top k points per |confidence-0.5|\n')
    f.write('Full data set with parameters learned via feature regression\n\n')
    f.write('k\trecall\tprecision\n')
    for line in pr_curve_f_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\n')
    f.close()
   
    ## Neighbor features ##

    # test data
    # create table whose columns are: |confidence-0.5|, prediction, actual value
    res_n_test = np.column_stack((np.absolute(y_pred_raw_n_test-0.5), y_pred_n_test, y_test))
    # sort by decreasing |confidence-0.5|
    res_n_sorted_test = res_n_test[res_n_test[:,0].argsort()[::-1]]
    # save
    f = open(OUT_FILE[adj]+'_n_prediction_table_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned with neighbor features\n\n')
    f.write('|confidence - 0.5|\tprediction\tactual value\n')
    for line in res_n_sorted_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()
   
    # get precision/recall for top k
    pr_curve_n_test = np.array(precision_recall_atk(res_n_sorted_test, len(y_test), 1))
    # save
    f = open(OUT_FILE[adj]+'_n_recall_precision_test.txt', 'w')
    f.write('Recall/Precision for top k points per |confidence-0.5|\n')
    f.write('Test data set with parameters learned with neighbor features\n\n')
    f.write('k\trecall\tprecision\n')
    for line in pr_curve_n_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\n')
    f.close()

    # full data
    # create table whose columns are: |confidence-0.5|, prediction, actual value
    res_n_full = np.column_stack((np.absolute(y_pred_raw_n_full-0.5), y_pred_n_full, y))
    # sort by decreasing |confidence-0.5|
    res_n_sorted_full = res_n_full[res_n_full[:,0].argsort()[::-1]]
    # save
    f = open(OUT_FILE[adj]+'_n_prediction_table_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned with neighbor features\n\n')
    f.write('|confidence - 0.5|\tprediction\tactual value\n')
    for line in res_n_sorted_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()
   
    # get precision/recall for top k
    pr_curve_n_full = np.array(precision_recall_atk(res_n_sorted_full, len(y), 1))
    # save
    f = open(OUT_FILE[adj]+'_n_recall_precision_full.txt', 'w')
    f.write('Recall/Precision for top k points per |confidence-0.5|\n')
    f.write('Full data set with parameters learned with neighbor features\n\n')
    f.write('k\trecall\tprecision\n')
    for line in pr_curve_n_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\n')
    f.close()
   
    ## LFM ##
 
    # test data
    # create table whose columns are: confidence, prediction, actual value
    res_lfm_test = np.column_stack((np.absolute(y_pred_raw_lfm_test-0.5), y_pred_lfm_test, y_test))
    # sort by confidence
    res_lfm_sorted_test = res_lfm_test[res_lfm_test[:,0].argsort()[::-1]]
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_prediction_table_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('|confidence - 0.5|\tprediction\tactual value\n')
    for line in res_lfm_sorted_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()
    
    # get precision/recall
    pr_curve_lfm_test = np.array(precision_recall_atk(res_lfm_sorted_test, len(y_test), 1))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_recall_precision_test.txt', 'w')
    f.write('Recall/Precision for top k points per |confidence-0.5|\n')
    f.write('Test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('k\trecall\tprecision\n')
    for line in pr_curve_lfm_test:
        f.write(str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\n')
    f.close()
    
    # full data
    # create table whose columns are: confidence, prediction, actual value
    res_lfm_full = np.column_stack((np.absolute(y_pred_raw_lfm_full-0.5), y_pred_lfm_full, y))     
    # sort by confidence
    res_lfm_sorted_full = res_lfm_full[res_lfm_full[:,0].argsort()[::-1]]
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_prediction_table_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('|confidence - 0.5|\tprediction\tactual value\n')
    for line in res_lfm_sorted_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()
    
    # get precision/recall
    pr_curve_lfm_full = np.array(precision_recall_atk(res_lfm_sorted_full, len(y), 1))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_recall_precision_full.txt', 'w')
    f.write('Recall/Precision for top k points per |confidence-0.5|\n')
    f.write('Full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('k\trecall\tprecision\n')
    for line in pr_curve_lfm_full:
        f.write(str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\n')
    f.close()
    
    # plot precision-recall curves
    # full data
    plt.figure()
    plt.plot([x[1] for x in pr_curve_f_full], [x[2] for x in pr_curve_f_full], 'ys-' )
    plt.plot([x[1] for x in pr_curve_n_full], [x[2] for x in pr_curve_n_full], 'r^-' )
    plt.plot([x[1] for x in pr_curve_lfm_full], [x[2] for x in pr_curve_lfm_full], 'bo-' )
    plt.legend(('regression on F', 'regression on N', 'regression on X'), loc='best')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Precision/Recall curve per top |confidence-0.5| over full data set')
    plt.show()
    #plt.savefig(FIG_FILE[adj]+'_f_lfm_'+method+'_recall_precision_full.png')

    # test data
    plt.figure()
    plt.plot([x[1] for x in pr_curve_f_test], [x[2] for x in pr_curve_f_test], 'ys-' )
    plt.plot([x[1] for x in pr_curve_n_test], [x[2] for x in pr_curve_n_test], 'r^-' )
    plt.plot([x[1] for x in pr_curve_lfm_test], [x[2] for x in pr_curve_lfm_test], 'bo-' )
    plt.legend(('regression on F', 'regression on N', 'regression on X'), loc='best')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Precision/Recall curve per top |confidence-0.5|')
    #plt.show()
    plt.savefig(FIG_FILE[adj]+'_f_n_lfm_'+method+'_recall_precision_test.png')

    f_area_test = np.trapz([x[2] for x in pr_curve_f_test], [x[1] for x in pr_curve_f_test])
    n_area_test = np.trapz([x[2] for x in pr_curve_n_test], [x[1] for x in pr_curve_n_test])
    lfm_area_test = np.trapz([x[2] for x in pr_curve_lfm_test], [x[1] for x in pr_curve_lfm_test])
    print 'area under f curve:', f_area_test
    print 'area under n curve:', n_area_test
    print 'area under lfm curve:', lfm_area_test


    print 'done.\n'

#### Accuracy @k ####

'''
Accuracy of kth percentile of confidence
'''

def accuracy_atk(data, k, interval=1):
    acc_atk_curve = []
    for i in range(1,k+1,interval): # @kth percentile
        # compute kth percentile
        point = np.percentile(data[:,0],i)
        num_ex = 0
        num_correct = 0
        for row in data:
            if row[0] >= point: # j in kth percentile
                num_ex += 1.0
                # correct prediction
                if row[1] == row[2]:
                    num_correct += 1.0
        accuracy = num_correct/num_ex
        acc_atk_curve.append([i, accuracy, num_ex]) 
    return acc_atk_curve

if test == 'accuracy-atk':
    print '\ncreating accuracy @k curves...'

    ## Regression ##

    # test data
    # create table whose columns are: confidence, prediction, actual value
    res_f_test = np.column_stack((y_pred_raw_f_test, y_pred_f_test, y_test))
    # sort by decreasing confidence
    res_f_sorted_test = res_f_test[res_f_test[:,0].argsort()[::-1]]
    # save
    f = open(OUT_FILE[adj]+'_f_confidence_prediction_table_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned via feature regression\n\n')
    f.write('confidence\tprediction\tactual value\n')
    for line in res_f_sorted_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get accuracy @kth percentile
    acc_atk_curve_f_test = np.array(accuracy_atk(res_f_sorted_test, 90))
    # save
    f = open(OUT_FILE[adj]+'_f_accuracy_atk_test.txt', 'w')
    f.write('Accuracy @kth percentile\n')
    f.write('Test data set with parameters learned via feature regression\n\n')
    f.write('percentile\taccuracy\tnumber of examples included\n')
    for line in acc_atk_curve_f_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # full data
    # create table whose columns are: confidence, prediction, actual value
    res_f_full = np.column_stack((y_pred_raw_f_full, y_pred_f_full, y))
    # sort by decreasing confidence
    res_f_sorted_full = res_f_full[res_f_full[:,0].argsort()[::-1]]
    # save
    f = open(OUT_FILE[adj]+'_f_confidence_prediction_table_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned via feature regression\n\n')
    f.write('confidence\tprediction\tactual value\n')
    for line in res_f_sorted_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get accuracy @kth percentile
    acc_atk_curve_f_full = np.array(accuracy_atk(res_f_sorted_full, 90))
    # save
    f = open(OUT_FILE[adj]+'_f_accuracy_atk_full.txt', 'w')
    f.write('Accuracy @kth percentile\n')
    f.write('Full data set with parameters learned via feature regression\n\n')
    f.write('percentile\taccuracy\tnumber of examples included\n')
    for line in acc_atk_curve_f_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    ## Neighbor features ##

    # test data
    # create table whose columns are: confidence, prediction, actual value
    res_n_test = np.column_stack((y_pred_raw_n_test, y_pred_n_test, y_test))
    # sort by decreasing confidence
    res_n_sorted_test = res_n_test[res_n_test[:,0].argsort()[::-1]]
    # save
    f = open(OUT_FILE[adj]+'_n_confidence_prediction_table_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned with neighbor features\n\n')
    f.write('confidence\tprediction\tactual value\n')
    for line in res_n_sorted_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get accuracy @kth percentile
    acc_atk_curve_n_test = np.array(accuracy_atk(res_n_sorted_test, 90))
    # save
    f = open(OUT_FILE[adj]+'_n_accuracy_atk_test.txt', 'w')
    f.write('Accuracy @kth percentile\n')
    f.write('Test data set with parameters learned with neighbor features\n\n')
    f.write('percentile\taccuracy\tnumber of examples included\n')
    for line in acc_atk_curve_n_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # full data
    # create table whose columns are: confidence, prediction, actual value
    res_n_full = np.column_stack((y_pred_raw_n_full, y_pred_n_full, y))
    # sort by decreasing confidence
    res_n_sorted_full = res_n_full[res_n_full[:,0].argsort()[::-1]]
    # save
    f = open(OUT_FILE[adj]+'_n_confidence_prediction_table_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned via feature regression\n\n')
    f.write('confidence\tprediction\tactual value\n')
    for line in res_n_sorted_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get accuracy @kth percentile
    acc_atk_curve_n_full = np.array(accuracy_atk(res_n_sorted_full, 90))
    # save
    f = open(OUT_FILE[adj]+'_n_accuracy_atk_full.txt', 'w')
    f.write('Accuracy @kth percentile\n')
    f.write('Full data set with parameters learned with neighbor features\n\n')
    f.write('percentile\taccuracy\tnumber of examples included\n')
    for line in acc_atk_curve_n_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    ## LFM ##

    # test data
    # create table whose columns are: confidence, prediction, actual value
    res_lfm_test = np.column_stack((y_pred_raw_lfm_test, y_pred_lfm_test, y_test))
    # sort by decreasing confidence
    res_lfm_sorted_test = res_lfm_test[res_lfm_test[:,0].argsort()[::-1]]
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_confidence_prediction_table_test.txt', 'w')
    f.write('Predictions on test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('confidence\tprediction\tactual value\n')
    for line in res_lfm_sorted_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get accuracy @kth percentile
    acc_atk_curve_lfm_test = np.array(accuracy_atk(res_lfm_sorted_test, 90))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_accuracy_atk_test.txt', 'w')
    f.write('Accuracy @kth percentile\n')
    f.write('Test data set with parameters learned via feature regression including latent factors\n\n')
    f.write('percentile\taccuracy\tnumber of examples included\n')
    for line in acc_atk_curve_lfm_test:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # full data
    # create table whose columns are: confidence, prediction, actual value
    res_lfm_full = np.column_stack((y_pred_raw_lfm_full, y_pred_lfm_full, y))     
    # sort by decreasing confidence
    res_lfm_sorted_full = res_lfm_full[res_lfm_full[:,0].argsort()[::-1]]
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_confidence_prediction_table_full.txt', 'w')
    f.write('Predictions on full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('confidence\tprediction\tactual value\n')
    for line in res_lfm_sorted_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # get accuracy @kth percentile
    acc_atk_curve_lfm_full = np.array(accuracy_atk(res_lfm_sorted_full, 90))
    # save
    f = open(OUT_FILE[adj]+'_lfm_'+method+'_accuracy_atk_full.txt', 'w')
    f.write('Accuracy @kth percentile\n')
    f.write('Full data set with parameters learned via feature regression including latent factors\n\n')
    f.write('percentile\taccuracy\tnumber of examples included\n')
    for line in acc_atk_curve_lfm_full:
        f.write(str(line[0])+'\t\t'+str(line[1])+'\t\t'+str(line[2])+'\n')
    f.close()

    # plot accuracy @kth percentile curves

    # full data
    plt.figure()
    plt.plot([x[0] for x in acc_atk_curve_f_full], [x[1] for x in acc_atk_curve_f_full], 'ys-' )
    plt.plot([x[0] for x in acc_atk_curve_n_full], [x[1] for x in acc_atk_curve_n_full], 'r^-' )
    plt.plot([x[0] for x in acc_atk_curve_lfm_full], [x[1] for x in acc_atk_curve_lfm_full], 'bo-' )
    plt.gca().invert_xaxis()
    plt.legend(('regression on F', 'regression on N', 'regression on X'), loc='best')
    plt.xlabel('confidence (percentile rank)')
    plt.ylabel('accuracy@k')
    plt.title('Accuracy @kth percentile over full data set')
    plt.show()
    #plt.savefig(FIG_FILE[adj]+'_f_lfm_'+method+'_accuracy_atk_full.png')

    # test data
    plt.figure()
    plt.plot([x[0] for x in acc_atk_curve_f_test], [x[1] for x in acc_atk_curve_f_test], 'ys-' )
    plt.plot([x[0] for x in acc_atk_curve_n_test], [x[1] for x in acc_atk_curve_n_test], 'r^-' )
    plt.plot([x[0] for x in acc_atk_curve_lfm_test], [x[1] for x in acc_atk_curve_lfm_test], 'bo-' )
    plt.gca().invert_xaxis()
    plt.legend(('regression on F', 'regression on N', 'regression on X'), loc='best')
    plt.xlabel('confidence (percentile rank)')
    plt.ylabel('accuracy@k')
    plt.title('Accuracy @kth percentile')
    #plt.show()
    plt.savefig(FIG_FILE[adj]+'_f_n_lfm_'+method+'_accuracy_atk_test.png')

    print 'done.\n'

