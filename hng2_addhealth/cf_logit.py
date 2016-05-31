"""
Collaborative filtering based community detection - simultaneously learn the U
link creation preference matrix, the V link acceptance preference matrix, the
beta individual preference vector, and the universal alpha bias.
"""

import numpy as np
import scipy.optimize as opt
import scipy.sparse as sparse
import random
import itertools
import cProfile
import multiprocessing as mp
from multiprocessing import Pool
import sys


NUM_PROCESSES = mp.cpu_count()

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def cost_logit(X, A, R, lam, n, k):
    '''
    The cost function

    n is the number of examples
    k is the feature dimension
    R is the matrix indicating which entries of A are known.
    '''
    # get the matrices
    # U, V, beta, alpha
    U = X[:n*k]
    U = np.reshape(U, (n,k))
    V = X[n*k:2*n*k]
    V = np.reshape(V, (n,k))
    beta = X[2*n*k:2*n*k+n]
    beta = np.reshape(beta, (n,1))
    alpha = X[-1]
    num_knowns = np.count_nonzero(R)
    num_edges = np.count_nonzero(np.multiply(A, R))
    num_nonedges = num_knowns - num_edges
    h = alpha + np.dot(U, np.transpose(V))
    # add beta to every row, column
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            h[i,j] += beta[i]+beta[j]
    sigH = sigmoid(h)
    J = ((-A/(2*num_edges))*np.log(sigH)) - (((1-A)/(2*num_nonedges))*np.log(1-sigH))
    J = J*R
    # regularizer
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            J[i,j] += lam*( np.abs(beta[i])**2 + np.abs(beta[j])**2 + np.linalg.norm(U[i,:])**2 + np.linalg.norm(V[j,:])**2 )
    # sum over known values
    cost = sum(sum(J))
    return cost


def grad_logit(X, A, R, lam, n, k):
    '''
    gradient functions

    n is the number of examples
    k is the feature dimension
    R is the matrix indicating which entries of A are known.
    '''
    # get the matrices
    # U, V, beta, alpha
    U = X[:n*k]
    U = np.reshape(U, (n,k))
    V = X[n*k:2*n*k]
    V = np.reshape(V, (n,k))
    beta = X[2*n*k:2*n*k+n]
    beta = np.reshape(beta, (n,1))
    alpha = X[-1]
    # construct the function h
    h = alpha + np.dot(U, np.transpose(V))
    # add beta to every row, column
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            h[i,j] += beta[i]+beta[j]
    # count number of known edges and non-edges
    num_knowns = np.count_nonzero(R)
    num_edges = np.count_nonzero(np.multiply(A, R))
    num_nonedges = num_knowns - num_edges
    # compute derivatives
    #sigHminA = sigmoid(h) - A
    sigH = sigmoid(h)
    sigHminA = (1.0/(2*num_nonedges))*sigH - A*( (1.0/(2*num_nonedges))*sigH + (1.0/(2*num_edges))*(1.0 / (1 + np.exp(h))) )
    dU = n*2*lam*U + np.dot((sigHminA)*R,V)
    dV = n*2*lam*V + np.dot(np.transpose((sigHminA)*R),U)
    dbeta = n*4*lam*beta + np.reshape(np.sum( (sigHminA)*R, axis=1 ),beta.shape) + np.reshape(np.sum( (sigHminA)*R, axis=0 ), beta.shape)
    dalpha = sum(sum( (sigHminA)*R ))
    # concatenate derivatives
    dUflat = np.reshape(dU, (1,U.size))
    dVflat = np.reshape(dV, (1,V.size))
    return np.concatenate((dUflat, dVflat, np.transpose(dbeta), np.array([[dalpha]])), axis=1)[0]


def cost_logit_lowspace(X, A, R, lam, n, k):
    '''
    The cost function without matrix algebra

    n is the number of examples
    k is the feature dimension
    R is the matrix indicating which entries of A are known.

    A and R are csc_sparse matrices
    '''
    # get the matrices
    # U, V, beta, alpha
    U = X[:n*k]
    U = np.reshape(U, (n,k))
    V = X[n*k:2*n*k]
    V = np.reshape(V, (n,k))
    beta = X[2*n*k:2*n*k+n]
    beta = np.reshape(beta, (n,1))
    alpha = X[-1]
    num_knowns = R.nnz 
    num_edges = R.multiply(A).nnz
    num_nonedges = num_knowns - num_edges
    # compute the cost
    #I, J = np.nonzero(R) # indices of known examples
    I, J = R.nonzero() # indices of known examples
    cost = 0
    for i, j in zip(I,J):
        h = (alpha+beta[i]+beta[j]+np.dot(U[i,:], V[j,:]))[0]
        cost += (-A[i,j]/(2*num_edges))*np.log(sigmoid(h)) - ((1-A[i,j])/(2*num_nonedges))*np.log(1-sigmoid(h))
    # regularizer
    regularizer = lam*(n*sum(np.linalg.norm(U, axis=1)**2) + n*sum(np.linalg.norm(V, axis=1)**2) + 2*n*sum(beta**2))
    cost += regularizer
    #for i in range(n):
    #    for j in range(n):
    #        cost += lam*( np.abs(beta[i])**2 + np.abs(beta[j])**2 + np.linalg.norm(U[i,:])**2 + np.linalg.norm(V[j,:])**2 )
    return cost[0]


def grad_logit_lowspace(X, A, R, lam, n, k):
    '''
    gradient functions

    n is the number of examples
    k is the feature dimension
    R is the matrix indicating which entries of A are known.
    '''
    # get the matrices
    # U, V, beta, alpha
    U = X[:n*k]
    U = np.reshape(U, (n,k))
    V = X[n*k:2*n*k]
    V = np.reshape(V, (n,k))
    beta = X[2*n*k:2*n*k+n]
    beta = np.reshape(beta, (n,1))
    alpha = X[-1]
    # count number of known edges and non-edges
    num_knowns = R.nnz 
    num_edges = R.multiply(A).nnz
    num_nonedges = num_knowns - num_edges
    #num_knowns = np.count_nonzero(R)
    #num_edges = np.count_nonzero(np.multiply(A, R))
    #num_nonedges = num_knowns - num_edges

    I, J = R.nonzero() # indices of known examples
    known_entries = zip(I,J)

    # compute derivatives

    # dU, dV
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    idxiter = iter([(i,l) for i in range(n) for l in range(k)] )
    for i,l in idxiter:
        dU[i,l] += n*2*lam*U[i,l]
        Js = [v for (u, v) in known_entries if u == i]
        for j in Js:
            h = (alpha+beta[i]+beta[j]+np.dot(U[i,:], V[j,:]))[0]
            dU[i,l] += ((1.0/(2*num_nonedges))*sigmoid(h) - A[i,j]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h)))))*V[j,l]
        # now i is j
        dV[i,l] += n*2*lam*V[i,l]
        Js = [u for (u, v) in known_entries if v == i]
        for j in Js:
            h = (alpha+beta[i]+beta[j]+np.dot(U[j,:], V[i,:]))[0]
            dV[i,l] += ((1.0/(2*num_nonedges))*sigmoid(h) - A[j,i]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h)))))*U[j,l]
        
    # dbeta
    dbeta = np.zeros_like(beta)
    dalpha = 0
    for i in xrange(n):
        dbeta[i] += 4*n*lam*beta[i]
        Js = [v for (u, v) in known_entries if u == i]
        for j in Js:
            h = (alpha+beta[i]+beta[j]+np.dot(U[i,:], V[j,:]))[0]
            dbeta[i] += (1.0/(2*num_nonedges))*sigmoid(h) - A[i,j]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h))))
            dalpha += (1.0/(2*num_nonedges))*sigmoid(h) - A[i,j]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h))))
        Js = [u for (u, v) in known_entries if v == i]
        for j in Js:
            h = (alpha+beta[i]+beta[j]+np.dot(U[j,:], V[i,:]))[0]
            dbeta[i] += (1.0/(2*num_nonedges))*sigmoid(h) - A[j,i]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h))))

    # concatenate derivatives
    dUflat = np.reshape(dU, (1,U.size))
    dVflat = np.reshape(dV, (1,V.size))
    return np.concatenate((dUflat, dVflat, np.transpose(dbeta), np.array([[dalpha]])), axis=1)[0]
    

Neval = 1

def get_reps(A, R_train, lam, k):
    '''
    Perform gradient descent to learn the parameters U, V, beta, alpha
    '''
    # initialize U, V, beta, alpha
    n = A.shape[0]
    U0 = (np.random.rand(n,k)-0.5)*0.1
    V0 = (np.random.rand(n,k)-0.5)*0.1
    beta0 = (np.random.rand(n,1)-0.5)*0.1
    alpha0 = (random.random()-0.5)*0.1
    U0flat = np.reshape(U0, (1,U0.size))
    V0flat = np.reshape(V0, (1,V0.size))
    X0 = np.concatenate((U0flat, V0flat, np.transpose(beta0), np.array([[alpha0]])), axis=1)[0]
    args=(A, R_train, lam, n, k)

    def callback_logit(Xk):
        global Neval
        #print '{0:4d}  {1: 3.6f}'.format(Neval, cost_logit_lowspace(Xk, args[0], args[1], args[2], args[3], args[4]))
        print '{0:4d}  {1: 3.6f}'.format(Neval, cost_logit(Xk, args[0], args[1], args[2], args[3], args[4]))
        Neval += 1    

    print '\tminimizing with BFGS...'
    #Xopt = opt.fmin_bfgs(cost_logit_lowspace, X0, fprime=grad_logit_lowspace, args=args, callback=callback_logit)
    Xopt = opt.fmin_bfgs(cost_logit, X0, fprime=grad_logit, args=args)
    #print '\tminimizing with CG...'
    #Xopt = opt.fmin_cg(cost_logit_lowspace, X0, fprime=grad_logit_lowspace, args=args, maxiter=10, callback=callback_logit)
    print '\tdone.'
    U = Xopt[:n*k]
    U = np.reshape(U, (n,k))
    V = Xopt[n*k:2*n*k]
    V = np.reshape(V, (n,k))
    beta = Xopt[2*n*k:2*n*k+n]
    beta = np.reshape(beta, (n,1))
    alpha = Xopt[-1]
    return U, V, beta, alpha


def pred(a, t):
    if a >= t:
        return 1.0
    else:
        return 0.0

vpred = np.vectorize(pred)

def build_param_mat(U, V, beta, alpha):
    # construct the function h
    h = alpha + np.dot(U, np.transpose(V))
    # add beta to every row, column
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            h[i,j] += beta[i]+beta[j]
    return h
    

def test_model(A, R_test, U, V, beta, alpha, lam, verbose=True):
    # construct the function h
    h = build_param_mat(U, V, beta, alpha)

    A_pred_raw = sigmoid(h)
    t = 0.50
    A_pred = vpred(A_pred_raw, t)


    accuracy = 0
    num_ex = 0
    rmse = 0
    pred_num0 = 0
    pred_num1 = 0
    actual_num0 = 0
    actual_num1 = 0
    TP, FP, FN, TN = 0,0,0,0

    #I, J = R_test.nonzero()
    I, J = np.nonzero(R_test)
    known_entries = zip(I,J)
    for i,j in known_entries:
    #for i in range(A.shape[0]):
    #    for j in range(A.shape[1]):
    #        if R_test[i,j] == 1:
        num_ex += 1
        rmse += (A[i,j]-A_pred_raw[i,j])**2
        if A[i,j] == A_pred[i,j]:
            accuracy += 1
        if A_pred[i,j] == 0:
            pred_num0 += 1
        if A_pred[i,j] == 1:
            pred_num1 += 1
        if A[i,j] == 0:
            actual_num0 += 1
        if A[i,j] == 1:
            actual_num1 += 1
        # actual positive
        if A[i,j] == 1:
            # true positive
            if A_pred[i,j] == 1:
                TP += 1
            # false negative
            else:
                FN += 1
        # actual negative
        if A[i,j] == 0:
            # true negative
            if A_pred[i,j] == 0:
                TN += 1
            # false positive
            else:
                FP += 1
  
    Uflat = np.reshape(U, (1,U.size))
    Vflat = np.reshape(V, (1,V.size))
    X = np.concatenate((Uflat, Vflat, np.transpose(beta), np.array([[alpha]])), axis=1)
    X = X[0]
    (n, k) = U.shape
 
    #ll = cost_logit_lowspace(X, A, R_test, lam, n, k) 
    ll = cost_logit(X, A, R_test, lam, n, k) 
    rmse = (1.0/num_ex)*rmse
    rmse = np.sqrt(rmse)
    accuracy = accuracy/(1.0*num_ex)

    percent_pred_nnz = pred_num1*1.0/num_ex
    percent_actual_nnz = actual_num1*1.0/num_ex

    if verbose:
        print 'log likelihood:\t', ll 
        print 'RMSE:\t\t', rmse
        print '\nperformance for threshold', t, ':'
        print '\taccuracy:\t', accuracy
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

        print '\npercent predicted non-zeros:\t', percent_pred_nnz 
        print 'percent actual non-zeros:\t', percent_actual_nnz 

        ## compare to predicting all 0's
        #zero_pred = np.zeros_like(A)

        #z_accuracy = 0
        #z_TP, z_FP, z_FN, z_TN = 0,0,0,0
        #for i in range(A.shape[0]):
        #    for j in range(A.shape[1]):
        #        if R_test[i,j] == 1:
        #            if A[i,j] == zero_pred[i,j]:
        #                z_accuracy += 1
        #            # actual positive
        #            if A[i,j] == 1:
        #                # true positive
        #                if zero_pred[i,j] == 1:
        #                    z_TP += 1
        #                # false negative
        #                else:
        #                    z_FN += 1
        #            # actual negative
        #            if A[i,j] == 0:
        #                # true negative
        #                if zero_pred[i,j] == 0:
        #                    z_TN += 1
        #                # false positive
        #                else:
        #                    z_FP += 1
  
        #z_accuracy = z_accuracy/(1.0*num_ex)

        #print '\ncompare to predicting all 0s'
        #print 'performance for threshold', t, ':'
        #print '\taccuracy:\t', z_accuracy
        #try:
        #    precision = z_TP/(z_TP+z_FP*1.0)
        #    print '\tprecision:\t', precision
        #except(ZeroDivisionError):
        #    precision = None
        #try:
        #    recall = z_TP/(z_TP+z_FN*1.0)
        #    print '\trecall:\t\t', recall
        #except(ZeroDivisionError):
        #    recall = None
        #try:
        #    if recall and precision:
        #        F1 = (2*precision*recall)/(precision+recall)
        #        print '\tF1 score:\t', F1
        #except(ZeroDivisionError):
        #    pass

    return rmse, accuracy, percent_pred_nnz, percent_actual_nnz

    
"""
Lab
"""

def offset(param, ind, eps):
    p_plus = param.copy()
    p_plus[ind] = p_plus[ind] + eps
    p_minus = param.copy()
    p_minus[ind] = p_minus[ind] - eps
    return p_plus, p_minus




#def cost_logit_mp(X, A, R, lam, n, k):
#    '''
#    The cost function without matrix algebra with multiprocessing to compute
#    cost of entries in parallel
#
#    n is the number of examples
#    k is the feature dimension
#    R is the matrix indicating which entries of A are known.
#
#    A, R are csc_sparse matrices
#    '''
#    # get the matrices
#    # U, V, beta, alpha
#    U = X[:n*k]
#    U = np.reshape(U, (n,k))
#    V = X[n*k:2*n*k]
#    V = np.reshape(V, (n,k))
#    beta = X[2*n*k:2*n*k+n]
#    beta = np.reshape(beta, (n,1))
#    alpha = X[-1]
#    # count number of known edges and non-edges
#    num_knowns = R.nnz 
#    num_edges = R.multiply(A).nnz
#    num_nonedges = num_knowns - num_edges
#    #num_knowns = np.count_nonzero(R)
#    #num_edges = np.count_nonzero(np.multiply(A, R))
#    #num_nonedges = num_knowns - num_edges
#
#    # split up the computation over all processors and collect in a queue
#    collect_costs = mp.Queue()
#    # compute the cost
#    def compute_costs(collect_costs, idx):
#        print '\tstarting process...'
#        cost = 0
#        for i,j in idx:
#            h = (alpha+beta[i]+beta[j]+np.dot(U[i,:], V[j,:]))[0]
#            cost += (-A[i,j]/(2*num_edges))*np.log(sigmoid(h)) - ((1-A[i,j])/(2*num_nonedges))*np.log(1-sigmoid(h))
#        print '\tending process...'
#        collect_costs.put(cost)
#        print '\tdone...'
#
#    # set up a list of processes
#    #I, J = np.nonzero(R) # indices of known examples
#    I, J = R.nonzero() # indices of known examples
#    known_entries = zip(I,J)
#    # divide computation
#    #chunk_size = len(known_entries)/(NUM_PROCESSES-1)
#    #chunks = [known_entries[i:i+chunk_size] 
#    #          for i in xrange(0, len(known_entries), chunk_size)]
#    div = range(0, len(known_entries), len(known_entries)/5)
#    chunks = [iter(known_entries[div[i]:div[i+1]]) for i in range(len(div)-1)]
#    chunks.append(iter(known_entries[div[-1]:]))
#    del known_entries
#    processes = [mp.Process(target=compute_costs, args=(collect_costs, idx)) 
#                 for idx in chunks]
#
#    # run processes
#    print 'starting processes'
#    for p in processes:
#        p.start()
#    # exit completed processes
#    print 'ending processes'
#    for p in processes:
#        p.join()
#    print 'done'
#
#    # get process results
#    print 'getting process results'
#    all_costs = [collect_costs.get() for p in processes]
#    print 'computing cost'
#    cost = sum(all_costs)
#
#    ## split up regularizer computation over all processors and collect in a queue
#    #collect_reg = mp.Queue()
#    #def compute_reg(collect_reg, idxiter):
#    #    print '\tstarting process...'
#    #    # regularizer
#    #    reg = 0
#    #    for i,j in idxiter:
#    #        reg += lam*( np.abs(beta[i])**2 + np.abs(beta[j])**2 + np.linalg.norm(U[i,:])**2 + np.linalg.norm(V[j,:])**2 )
#    #    print '\tending process...'
#    #    collect_reg.put(reg)
#    #    print '\tdone.'
#
#    ## set up a list of processes
#    #all_entries = [(x,y) for x in range(n) for y in range(n)]
#    ## divide computation
#    ##chunk_size = n**2/(NUM_PROCESSES-1)
#    ##chunks = [iter(all_entries[i:i+chunk_size]) 
#    ##          for i in xrange(0, len(all_entries), chunk_size)]
#    #div = range(0, len(all_entries), len(all_entries)/5)
#    #chunks = [iter(all_entries[div[i]:div[i+1]]) for i in range(len(div)-1)]
#    #chunks.append(iter(all_entries[div[-1]:]))
#    #del all_entries
#    #processes = [mp.Process(target=compute_reg, args=(collect_reg, idxiter)) 
#    #             for idxiter in chunks]
#
#    ## run processes
#    #print 'regularizer starting processes'
#    #for p in processes:
#    #    p.start()
#    ## exit completed processes
#    #print 'regularizer ending processes'
#    #for p in processes:
#    #    p.join()
#    #print 'done.'
#
#    ## get process results
#    #print 'getting process results'
#    #all_regs = [collect_reg.get() for p in processes]
#    #print 'computing regularization factor'
#    #regularizer = sum(all_regs)[0]
#
#    # regularizer
#    regularizer = lam*(n*sum(np.linalg.norm(U, axis=1)**2) + n*sum(np.linalg.norm(V, axis=1)**2) + 2*n*sum(beta**2))
#    cost += regularizer
#
#    #print '\t\tcost at iteration:', cost
#    return cost


#def cost_logit_SM(X, A, R, lam, n, k):
#    '''
#    The cost function using sparse matrix representations
#
#    n is the number of examples
#    k is the feature dimension
#    R is the matrix indicating which entries of A are known.
#    '''
#    # get the matrices
#    # U, V, beta, alpha
#    U = X[:n*k]
#    U = np.reshape(U, (n,k))
#    V = X[n*k:2*n*k]
#    V = np.reshape(V, (n,k))
#    beta = X[2*n*k:2*n*k+n]
#    beta = np.reshape(beta, (n,1))
#    alpha = X[-1]
#    h = alpha + np.dot(U, np.transpose(V))
#    # add beta to every row, column
#    for i in range(h.shape[0]):
#        for j in range(h.shape[1]):
#            h[i,j] += beta[i]+beta[j]
#    sigH = sigmoid(h)
#    one = np.ones_like(A)
#    A = sparse.csr_matrix(A)
#    R = sparse.csr_matrix(R)
#    # count number of known edges and non-edges
#    num_knowns = R.nnz 
#    num_edges = R.multiply(A).nnz
#    num_nonedges = num_knowns - num_edges
#    # compute the cost
#    #J = -A.multiply(np.log(sigH)) - sparse.csr_matrix(one-A).multiply(np.log(1-sigH))
#    J = (1.0/(2*num_edges))*(-A.multiply(np.log(sigH))) - (1.0/(2*num_nonedges))*(sparse.csr_matrix(one-A).multiply(np.log(1-sigH)))
#    J = R.multiply(J)
#    cost = J.sum()
#    # regularizer
#    for i in range(n):
#        for j in range(n):
#            cost += lam*( np.abs(beta[i])**2 + np.abs(beta[j])**2 + np.linalg.norm(U[i,:])**2 + np.linalg.norm(V[j,:])**2 )
#    return cost[0]


#def cost_logit_NV(X, A, R, lam, n, k):
#    '''
#    non-vectorized cost for verification
#
#    n is the number of examples
#    k is the feature dimension
#    R is the matrix indicating which entries of A are known.
#    '''
#    # get the matrices
#    # U, V, beta, alpha
#    U = X[:n*k]
#    U = np.reshape(U, (n,k))
#    V = X[n*k:2*n*k]
#    V = np.reshape(V, (n,k))
#    beta = X[2*n*k:2*n*k+n]
#    beta = np.reshape(beta, (n,1))
#    alpha = X[-1]
#    h = alpha + np.dot(U, np.transpose(V))
#    # add beta to every row, column
#    for i in range(h.shape[0]):
#        for j in range(h.shape[1]):
#            h[i,j] += beta[i]+beta[j]
#    cost = 0
#    for i in range(n):
#        for j in range(n):
#            if R[i,j] == 1:
#                cost += -A[i,j]*np.log(sigmoid(h[i,j])) - (1-A[i,j])*np.log(1-sigmoid(h[i,j]))
#    # regularizer
#    for i in range(n):
#        for j in range(n):
#            cost += lam*( np.abs(beta[i])**2 + np.abs(beta[j])**2 + np.linalg.norm(U[i,:])**2 + np.linalg.norm(V[j,:])**2 )
#    return cost[0]


#def grad_logit_ls(X, A, R, lam, n, k):
#    '''
#    gradient functions
#
#    n is the number of examples
#    k is the feature dimension
#    R is the matrix indicating which entries of A are known.
#    '''
#    # get the matrices
#    # U, V, beta, alpha
#    U = X[:n*k]
#    U = np.reshape(U, (n,k))
#    V = X[n*k:2*n*k]
#    V = np.reshape(V, (n,k))
#    beta = X[2*n*k:2*n*k+n]
#    beta = np.reshape(beta, (n,1))
#    alpha = X[-1]
#    # count number of known edges and non-edges
#    num_knowns = R.nnz 
#    num_edges = R.multiply(A).nnz
#    num_nonedges = num_knowns - num_edges
#    #num_knowns = np.count_nonzero(R)
#    #num_edges = np.count_nonzero(np.multiply(A, R))
#    #num_nonedges = num_knowns - num_edges
#
#    I, J = R.nonzero() # indices of known examples
#    known_entries = zip(I,J)
#
#    # compute derivatives
#
#    # dU
#    dU = np.zeros_like(U)
#    #idxiter = iter([(i,l) for i in range(n) for l in range(k)] )
#    #for i,l in idxiter:
#    #    dU[i,l] += n*2*lam*U[i,l]
#    #    Js = [v for (u, v) in known_entries if u == i]
#    #    for j in Js:
#    #        h = (alpha+beta[i]+beta[j]+np.dot(U[i,:], V[j,:]))[0]
#    #        dU[i,l] += ((1.0/(2*num_nonedges))*sigmoid(h) - A[i,j]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h)))))*V[j,l]
#    for i in range(n):
#        for l in range(k):
#            for j in range(n):
#                if R[i,j]:
#                    h = (alpha+beta[i]+beta[j]+np.dot(U[i,:], V[j,:]))[0]
#                    dU[i,l] += ((1.0/(2*num_nonedges))*sigmoid(h) - A[i,j]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h)))))*V[j,l]
#                dU[i,l] += 2*lam*U[i,l]
#
#    # dV
#    dV = np.zeros_like(V)
#    ##idxiter = iter([(i,l) for i in range(n) for l in range(k)] )
#    #for j,l in idxiter:
#    #    dV[j,l] += n*2*lam*U[j,l]
#    #    Is = [u for (u, v) in known_entries if v == j]
#    #    for j in Is:
#    #        h = (alpha+beta[i]+beta[j]+np.dot(U[i,:], V[j,:]))[0]
#    #        dV[j,l] += ((1.0/(2*num_nonedges))*sigmoid(h) - A[i,j]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h)))))*U[j,l]
#    for j in range(n):
#        for l in range(k):
#            for i in range(n):
#                if R[i,j]:
#                    h = (alpha+beta[i]+beta[j]+np.dot(U[i,:], V[j,:]))[0]
#                    dV[j,l] += ((1.0/(2*num_nonedges))*sigmoid(h) - A[i,j]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h)))))*U[j,l]
#                dV[j,l] += 2*lam*V[j,l]
#
#    # dbeta
#    dbeta = np.zeros_like(beta)
#    #for i in xrange(n):
#    #    dbeta[i] += 4*n*lam*beta[i]
#    #    Js = [v for (u, v) in known_entries if u == i]
#    #    for j in Js:
#    #        h = (alpha+beta[i]+beta[j]+np.dot(U[i,:], V[j,:]))[0]
#    #        dbeta[i] += (1.0/(2*num_nonedges))*sigmoid(h) - A[i,j]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h))))
#    #        h = (alpha+beta[i]+beta[j]+np.dot(U[j,:], V[i,:]))[0]
#    #        dbeta[i] += (1.0/(2*num_nonedges))*sigmoid(h) - A[j,i]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h))))
#    for i in range(n):
#        for j in range(n):
#            if R[i,j]:
#                h = (alpha+beta[i]+beta[j]+np.dot(U[i,:], V[j,:]))[0]
#                dbeta[i] += (1.0/(2*num_nonedges))*sigmoid(h) - A[i,j]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h))))
#            dbeta[i] += 2*lam*beta[i] 
#        for j in range(n):
#            if R[i,j]:
#                h = (alpha+beta[i]+beta[j]+np.dot(U[j,:], V[i,:]))[0]
#                dbeta[i] += (1.0/(2*num_nonedges))*sigmoid(h) - A[j,i]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h))))
#            dbeta[i] += 2*lam*beta[i] 
#
#    # dalpha
#    dalpha = 0
#    #for i in xrange(n):
#    #    Js = [v for (u, v) in known_entries if u == i]
#    #    for j in Js:
#    #        h = (alpha+beta[i]+beta[j]+np.dot(U[i,:], V[j,:]))[0]
#    #        dalpha += (1.0/(2*num_nonedges))*sigmoid(h) - A[i,j]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h))))
#    for i in range(n):
#        for j in range(n):
#            if R[i,j]:
#                h = (alpha+beta[i]+beta[j]+np.dot(U[i,:], V[j,:]))[0]
#                dalpha += (1.0/(2*num_nonedges))*sigmoid(h) - A[i,j]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h))))
#
#    # concatenate derivatives
#    dUflat = np.reshape(dU, (1,U.size))
#    dVflat = np.reshape(dV, (1,V.size))
#    return np.concatenate((dUflat, dVflat, np.transpose(dbeta), np.array([[dalpha]])), axis=1)[0]


#def grad_logit_mp(X, A, R, lam, n, k):
#    '''
#    gradient functions using multiprocessing
#
#    n is the number of examples
#    k is the feature dimension
#    R is the matrix indicating which entries of A are known.
#
#    A, R are csc_sparse matrices
#    '''
#    # get the matrices
#    # U, V, beta, alpha
#    U = X[:n*k]
#    U = np.reshape(U, (n,k))
#    V = X[n*k:2*n*k]
#    V = np.reshape(V, (n,k))
#    beta = X[2*n*k:2*n*k+n]
#    beta = np.reshape(beta, (n,1))
#    alpha = X[-1]
#    # count number of known edges and non-edges
#    num_knowns = R.nnz 
#    num_edges = R.multiply(A).nnz
#    num_nonedges = num_knowns - num_edges
#    #num_knowns = np.count_nonzero(R)
#    #num_edges = np.count_nonzero(np.multiply(A, R))
#    #num_nonedges = num_knowns - num_edges
#
#    I, J = R.nonzero() # indices of known examples
#    known_entries = zip(I,J)
#
#    # compute derivatives
#
#    ## dU ##
#    print '\t\tcomputing dU...'
#    # split up derivative computations over all processors
#    collect_dU = mp.Queue()
#    def compute_dU(collect_dU, idxiter):
#        print '\t\t\tstarting job...'
#        dU_piece = np.zeros_like(U)
#        for i,l in idxiter:
#            dU_piece[i,l] += n*2*lam*U[i,l]
#            Js = [v for (u, v) in known_entries if u == i]
#            #for j in range(U.shape[0]):
#            #    dU_piece[i,l] += 2*lam*U[i,l]
#            #    if R[i,j]:
#            for j in Js:
#                    h = (alpha+beta[i]+beta[j]+np.dot(U[i,:], V[j,:]))[0]
#                    dU_piece[i,l] += ((1.0/(2*num_nonedges))*sigmoid(h) - A[i,j]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h)))))*V[j,l]
#        print '\t\t\tending job...'
#        collect_dU.put(dU_piece)
#        print '\t\t\tdone.'
#
#    # set up a list of processes
#    all_entries = [(i,l) for i in range(n) for l in range(k)] 
#    # divide computation
#    #chunk_size = len(all_entries)/2
#    #chunks = [iter(all_entries[i:i+chunk_size])
#    #          for i in xrange(0, len(all_entries), chunk_size+1)]
#    div = range(0, len(all_entries), len(all_entries)/6)
#    chunks = [iter(all_entries[div[i]:div[i+1]]) for i in range(len(div)-1)]
#    chunks.append(iter(all_entries[div[-1]:]))
#    del all_entries
#    processes = [mp.Process(target=compute_dU, args=(collect_dU, idxiter)) 
#                 for idxiter in chunks]
#    
#    # run processes
#    print '\tstarting processes'
#    for p in processes:
#        p.start()
#    # exit completed processes
#    print '\tending processes'
#    for p in processes:
#        p.join()
#
#    # get process results from output queue
#    print '\tgetting process results'
#    all_dU = [collect_dU.get() for p in processes]
#    print '\tcomputing dU'
#    dU = sum(all_dU)
#    #dU = dU + 2*n*lam*U
#
#    return dU
#
#    ### dV ##
#    #print '\t\tcomputing dV...'
#    ## split up derivative computations over all processors
#    #collect_dV = mp.Queue()
#    #def compute_dV(collect_dV, idxiter):
#    #    print '\t\t\tstarting job...'
#    #    dV_piece = np.zeros_like(V)
#    #    for j,l in idxiter:
#    #        for i in range(V.shape[0]):
#    #            if R[i,j]:
#    #                h = (alpha+beta[i]+beta[j]+np.dot(U[i,:], V[j,:]))[0]
#    #                dV_piece[j,l] += ((1.0/(2*num_nonedges))*sigmoid(h) - A[i,j]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h)))))*U[j,l]
#    #            dV_piece[j,l] += 2*lam*V[j,l]
#    #    print '\t\t\tending job...'
#    #    collect_dV.put(dV_piece)
#    #    print '\t\t\tdone.'
#
#    ## set up a list of processes
#    #all_entries = [(j,l) for j in range(n) for l in range(k)] 
#    ## divide computation
#    ##chunk_size = len(all_entries)/2
#    ##chunks = [iter(all_entries[i:i+chunk_size])
#    ##          for i in xrange(0, len(all_entries), chunk_size+1)]
#    #div = range(0, len(all_entries), len(all_entries)/4)
#    #chunks = [iter(all_entries[div[i]:div[i+1]]) for i in range(len(div)-1)]
#    #chunks.append(iter(all_entries[div[-1]:]))
#    #del all_entries
#    #processes = [mp.Process(target=compute_dV, args=(collect_dV, idxiter)) 
#    #             for idxiter in chunks]
#    #
#    ## run processes
#    #for p in processes:
#    #    p.start()
#    ## exit completed processes
#    #for p in processes:
#    #    p.join()
#
#    ## get process results from output queue
#    #all_dV = [collect_dV.get() for p in processes]
#    #dV = sum(all_dV)
#
#    ### dbeta ##
#    #print '\t\tcomputing dbeta...'
#    ## split up derivative computations over all processors
#    #collect_dbeta = mp.Queue()
#    #def compute_dbeta(collect_dbeta, idxiter):
#    #    print '\t\t\tstarting job...'
#    #    dbeta_piece = np.zeros_like(beta)
#    #    for i in idxiter:
#    #        for j in range(beta.shape[0]):
#    #            if R[i,j]:
#    #                h = (alpha+beta[i]+beta[j]+np.dot(U[i,:], V[j,:]))[0]
#    #                dbeta_piece[i] += (1.0/(2*num_nonedges))*sigmoid(h) - A[i,j]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h))))
#    #                h = (alpha+beta[i]+beta[j]+np.dot(U[j,:], V[i,:]))[0]
#    #                dbeta_piece[i] += (1.0/(2*num_nonedges))*sigmoid(h) - A[j,i]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h))))
#    #            dbeta_piece[i] += 4*lam*beta[i] 
#    #    print '\t\t\tending job...'
#    #    collect_dbeta.put(dbeta_piece)
#    #    print '\t\t\tdone.'
#
#    ## set up a list of processes
#    #all_entries = range(n) 
#    ## divide computation
#    ##chunk_size = len(all_entries)/8
#    ##chunks = [iter(all_entries[i:i+chunk_size])
#    ##          for i in xrange(0, len(all_entries), chunk_size+1)]
#    #div = range(0, len(all_entries), len(all_entries)/5)
#    #chunks = [iter(all_entries[div[i]:div[i+1]]) for i in range(len(div)-1)]
#    #chunks.append(iter(all_entries[div[-1]:]))
#    #del all_entries
#    #processes = [mp.Process(target=compute_dbeta, args=(collect_dbeta, idxiter)) 
#    #             for idxiter in chunks]
#    #
#    ## run processes
#    #for p in processes:
#    #    p.start()
#    ## exit completed processes
#    #for p in processes:
#    #    p.join()
#
#    ## get process results from output queue
#    #all_dbeta = [collect_dbeta.get() for p in processes]
#    #dbeta = sum(all_dbeta)
#
#    ### dalpha ##
#    #print '\t\tcomputing dalpha...'
#    ## split up derivative computations over all processors
#    #collect_dalpha = mp.Queue()
#    #def compute_dalpha(collect_dalpha, idxiter):
#    #    print '\t\t\tstarting job...'
#    #    dalpha_piece = 0
#    #    for i in idxiter:
#    #        for j in range(beta.shape[0]):
#    #            if R[i,j]:
#    #                h = (alpha+beta[i]+beta[j]+np.dot(U[i,:], V[j,:]))[0]
#    #                dalpha_piece += (1.0/(2*num_nonedges))*sigmoid(h) - A[i,j]*((1.0/(2*num_nonedges))*sigmoid(h) + (1.0/(2*num_edges))*(1.0/(1+np.exp(h))))
#    #    print '\t\t\tending job...'
#    #    collect_dalpha.put(dalpha_piece)
#    #    print '\t\t\tdone.'
#
#    ## set up a list of processes
#    #all_entries = range(n) 
#    ## divide computation
#    ##chunk_size = len(all_entries)/8
#    ##chunks = [iter(all_entries[i:i+chunk_size])
#    ##          for i in xrange(0, len(all_entries), chunk_size+1)]
#    #div = range(0, len(all_entries), len(all_entries)/4)
#    #chunks = [iter(all_entries[div[i]:div[i+1]]) for i in range(len(div)-1)]
#    #chunks.append(iter(all_entries[div[-1]:]))
#    #del all_entries
#    #processes = [mp.Process(target=compute_dalpha, args=(collect_dalpha, idxiter)) 
#    #             for idxiter in chunks]
#    #
#    ## run processes
#    #for p in processes:
#    #    p.start()
#    ## exit completed processes
#    #for p in processes:
#    #    p.join()
#
#    ## get process results from output queue
#    #all_dalpha = [collect_dalpha.get() for p in processes]
#    #dalpha = sum(all_dalpha)
#    #
#    ## concatenate derivatives
#    #dUflat = np.reshape(dU, (1,U.size))
#    #dVflat = np.reshape(dV, (1,V.size))
#    #return np.concatenate((dUflat, dVflat, np.transpose(dbeta), np.array([[dalpha]])), axis=1)[0]
    
   
 
#def grad_logit_NV(X, A, R, lam, n, k):
#    '''
#    nonvectorized gradient for verification
#
#    n is the number of examples
#    k is the feature dimension
#    R is the matrix indicating which entries of A are known.
#    '''
#    # get the matrices
#    # U, V, beta, alpha
#    U = X[:n*k]
#    U = np.reshape(U, (n,k))
#    V = X[n*k:2*n*k]
#    V = np.reshape(V, (n,k))
#    beta = X[2*n*k:2*n*k+n]
#    beta = np.reshape(beta, (n,1))
#    alpha = X[-1]
#    # construct the function h
#    h = alpha + np.dot(U, np.transpose(V))
#    # add beta to every row, column
#    for i in range(h.shape[0]):
#        for j in range(h.shape[1]):
#            h[i,j] += beta[i]+beta[j]
#    dU = np.zeros_like(U)
#    for i in range(n):
#        for l in range(k):
#            for j in range(n):
#                dU[i,l] += 2*lam*U[i,l]
#                if R[i,j] == 1:
#                    dU[i,l] += (sigmoid(h[i,j]) - A[i,j])*V[j,l]
#    dV = np.zeros_like(V)
#    for j in range(n):
#        for l in range(k):
#            for i in range(n):
#                dV[j,l] += 2*lam*V[j,l]
#                if R[i,j] == 1:
#                    dV[j,l] += (sigmoid(h[i,j]) - A[i,j])*U[i,l]
#    dbeta = np.zeros_like(beta)
#    for i in range(n):
#        for j in range(n):
#            dbeta[i] += 4*lam*beta[i]
#            if R[i,j] == 1:
#                dbeta[i] += sigmoid(h[i,j]) - A[i,j]
#            if R[j,i] == 1:
#                dbeta[i] += sigmoid(h[j,i]) - A[j,i]
#    dalpha = 0
#    for i in range(n):
#        for j in range(n):
#            if R[i,j] == 1:
#                dalpha += sigmoid(h[i,j]) - A[i,j]
#    # concatenate derivatives
#    dUflat = np.reshape(dU, (1,U.size))
#    dVflat = np.reshape(dV, (1,V.size))
#    #return dU, dV, dbeta, dalpha
#    return np.concatenate((dUflat, dVflat, np.transpose(dbeta), np.array([[dalpha]])), axis=1)[0]

