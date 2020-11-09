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


def PGLINacc(V,Winit,Hinit,alpha,delta,maxiter,H0=[]):
    nM = np.linalg.norm(V) ** 2
    (m,n) = V.shape
    (m,r) = Winit.shape

    if sp.sparse.issparse(V):
        K = np.sum(V)
    else:
        K = m * n

    rhoW = 1 + (K + n * r) / (m * (r + 1))
    rhoH = 1 + (K + m * r) / (n * (r + 1))
    W=Winit
    H=Hinit
    t=[]
    e=[]
    iter=0
    L = []
    while iter <= maxiter:
        W,gradW,iterW,HHt,VtH = nlssubprob(np.transpose(V),np.transpose(H),np.transpose(W),1+alpha*rhoW,delta)

        W = np.transpose(W)

        H,gradH,iterH,WtW,WtV = nlssubprob(V,W,H,1+alpha*rhoH,delta)
        L.append(np.sqrt(D_distance(H0, H)))
        e.append(np.linalg.norm(V - np.dot(W, H)) ** 2)
        iter+=1
    return W,H,L,e


def nlssubprob(V,W,Hinit,maxiter,delta):
    H=Hinit
    WtV=np.dot(np.transpose(W),V)
    WtW=np.dot(np.transpose(W),W)
    alpha=1
    beta=0.1
    iter=1
    eps=1
    eps0=1
    while iter<=maxiter and eps >= delta*eps0:
        grad=np.dot(WtW,H)-WtV
        # projgrad = np.linalg.norm(grad(grad(<0 and H>0)))
        H0 = H
        for inner_iter in range(20):
            Hn=np.maximum(H-alpha*grad,0)
            d=Hn-H
            gradd=np.sum(np.sum(grad*d))
            dQd=np.sum(np.sum(np.dot(WtW,d)*d))
            suff_decr = 0.99*gradd+0.5*dQd<0
            if inner_iter == 0:
                decr_alpha = not suff_decr
                Hp = H
            if decr_alpha:
                if suff_decr:
                    H=Hn
                    break
                else:
                    alpha = alpha * beta
            else:
                if not suff_decr or np.array_equal(Hp,Hn):
                    H = Hp
                    break
                else:
                    alpha=alpha/beta
                    Hp=Hn
        if iter ==1:
            eps0 = np.linalg.norm(H-H0)
        eps=np.linalg.norm(H-H0)
        iter=iter+1

    return H,grad,iter,WtW,WtV







