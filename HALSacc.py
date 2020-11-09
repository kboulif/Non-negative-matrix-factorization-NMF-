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

def HALSacc(M,U,V,alpha = 2, delta = 0.1, maxiter = 100, H0=[]):
    nM = np.linalg.norm(M)**2
    (n,m) = M.shape
    (m,r) = U.shape
    a = 0
    e = []
    t = []
    iter = 0
    A = np.dot(M, np.transpose(V))
    B = np.dot(V, np.transpose(V))
    j=0
    scaling = np.sum(np.sum(A*U)/np.sum(B*(np.dot(np.transpose(U),U))))
    U = scaling*U
    L = []
    while iter <= maxiter:
        print('iter'+str(iter))
        if j==1:
            A = np.dot(M, np.transpose(V))
            B = np.dot(V, np.transpose(V))
        j = 1
        eps = 1
        esp0 = 1
        U = HALSupdt(np.transpose(U),np.transpose(B),np.transpose(A),alpha,delta)
        U = np.transpose(U)

        #update of V
        A = np.dot(np.transpose(U),M)
        B = np.dot(np.transpose(U),U)
        eps = 1
        esp0 = 1
        V=HALSupdt(V,B,A,alpha,delta)
        iter+=1
        j=1
        #evaluation de l'erreur à l'instant t
        e.append(np.linalg.norm(M - np.dot(U, V)) ** 2)
        L.append(np.sqrt(D_distance(H0, V)))
    return U,V,L,e



def HALSupdt(V_init,UtU,UtM,alpha,delta):
    V = V_init
    (r,n)=V.shape
    cnt=1
    eps=1
    eps0=1
    while cnt==1 or eps >= (delta)**2*eps0:
        nodelta = 0
        for k in range(r):
            deltaV = np.maximum(((1/UtU[k,k])*(UtM[k,:]-np.dot(UtU[k,:],V))), -V[k,:])
            V[k,:]=V[k,:]+deltaV
            nodelta = nodelta + np.dot(deltaV, np.transpose(deltaV))
            if np.all(V[k,:]==0):
                V[k, :]=10**(-16)* np.max(V)
        if cnt==1:
            eps0 = nodelta
        eps = nodelta
        cnt=0

    return V






























