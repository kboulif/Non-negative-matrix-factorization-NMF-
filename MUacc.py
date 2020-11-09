import numpy as np
import scipy as sp


def D_distance (H1, H2):

# This function computes the 'L'-distance between the two set of vectors collected in the rows of H1 and H2. In our paper notation, this is $\mathscr{L}(H_1, H_2)$.

    n1 = H1.shape[0]
    n2 = H2.shape[0]
    D = 0
    for i in range (0,n1):
        d = (np.linalg.norm(H1[i,:] - H2[0,:]))**2
        for j in range (1,n2):
            d = min(d, (np.linalg.norm(H1[i,:] - H2[j,:])**2))
        D = D+d
    return D
def l2distance( x, U, x0 ):

# This function computes <x-x0, (U^T*U)*(x-x0)>.

    lx = np.linalg.norm(x-x0)**2
    lpx = np.linalg.norm(np.dot(U,x-x0))**2
    return (lx-lpx)


def initH(X, r):
    # This function computes 'r' initial archetypes given rows of 'X' as the data points. The method used here is the successive projections method explained in the paper.

    n = X.shape[0]
    d = X.shape[1]
    H = np.zeros((r, d))
    maxd = np.linalg.norm(X[0, :])
    imax = 0
    for i in range(1, n):
        newd = np.linalg.norm(X[i, :])
        if (newd > maxd):
            imax = i
            maxd = newd
    H[0, :] = X[imax, :]
    maxd = np.linalg.norm(X[0, :] - H[0, :])
    imax = 0
    for i in range(1, n):
        newd = np.linalg.norm(X[i, :] - H[0, :])
        if (newd > maxd):
            imax = i
            maxd = newd
    H[1, :] = X[imax, :]

    for k in range(2, r):
        M = H[1:k, :] - np.outer(np.ones(k - 1), H[0, :])
        [U, s, V] = np.linalg.svd(M, full_matrices=False)
        maxd = l2distance(X[0, :], V, H[0, :])
        imax = 0
        for i in range(1, n):
            newd = l2distance(X[i, :], V, H[0, :])
            if (newd > maxd):
                imax = i
                maxd = newd
        H[k, :] = X[imax, :]
    return H


def MUacc(M,U,V,alpha = 1, delta = 0.1, maxiter = 10000,H0=[]):
    nM = np.linalg.norm(M)**2
    (n,m) = M.shape
    (m,r) = U.shape
    if sp.sparse.issparse(M):
        K = np.sum(M)
    else:
        K = m*n
    rhoU = 1 + (K + n * r) / (m * (r + 1))
    rhoV = 1 + (K + m * r) / (n * (r + 1))
    a = 0
    e = []
    t = []
    iter = 0
    j = 1
    L = []
    while iter <= maxiter:
        A = np.dot(M,np.transpose(V))
        B = np.dot(V,np.transpose(V))
        eps = 1
        eps0 = 1
        while j <= 1+rhoU*alpha and eps >= delta*eps0:
            U0 = U
            U = np.maximum(1e-16*np.ones((1000,4)), U * (np.divide(A, np.dot(U,B))))
            if j == 1:
                eps0 = np.linalg.norm(U0 - U)
            eps = np.linalg.norm(U0 - U)
            j = j + 1

        A = np.dot(np.transpose(U),M)
        B = np.dot(np.transpose(U),U)
        eps = 1
        eps0 = 1
        j = 1
        while j <= (1 + rhoV * alpha) and eps >= delta*eps0 :
            V0 = V
            V = np.maximum(1e-16 * np.ones((4, 87)), V * (np.divide(A, np.dot(B, V))))
            if j == 1:
                eps0 = np.linalg.norm(V0 - V)
            eps = np.linalg.norm(V0 - V)
            j = j + 1

        #e.append(np.sqrt((nM-2*np.sum(np.sum(np.dot(V,A)))+ np.sum(np.sum(np.dot(B,(np.dot(V,np.transpose(V)))))))))
        e.append(np.linalg.norm(M - np.dot(U,V))**2)
        L.append(np.sqrt(D_distance(H0, V)))
        print(e)

        iter += 1


    return U,V,L,e

