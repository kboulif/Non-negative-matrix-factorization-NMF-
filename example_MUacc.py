import numpy as np
import matplotlib.pyplot as plt
import time
import NMF as nmf
from MUacc import *
from HALSacc import *
from PGLINacc import *
plt.style.use('ggplot')
import time

start_time = time.time()

n = 1000 		# Number of Data points
d = 87			# Dimension of the Data
r = 4			# Rank of the model
n_f = 20		# Number of data points on the faces
deg_prob = np.array((0, 0.6, 0.4))		# Distribution of the support size of the weights for data points on the faces. 

alpha = 1			# Dirichlet Parameter
sigma = 0.001			# Noise Magnitude
W0 = nmf.generate_weights(n, r, alpha, n_f, deg_prob)		# Generating Weights

def read_spectral_data():

# Reading spectral data used as archetypes.

    c_data_file = open("CAFFEINE.txt","r")
    C = []
    for line in c_data_file:
        l = line.split()
        del l[0]
        l = [float(x) for x in l]
        l = np.asarray(l)
        C = np.append(C,l)

    C = C[184:271]
    C = C/np.sum(C)
    print(C)

    s_data_file = open("Sucrose.txt","r")
    S = []
    for line in s_data_file:
        l = line.split()
        del l[0]
        l = [float(x) for x in l]
        l = np.asarray(l)
        S = np.append(S,l)

    S = S[783:1131]
    S = np.reshape(S,(len(S)//4,4))
    S = 2 - np.log10(100*S[:,0])
    S = S/np.sum(S)

    l_data_file = open("Lactose.txt","r")
    L = []
    for line in l_data_file:
        l = line.split()
        del l[0]
        # l = map(float,l)
        l = [float(x) for x in l]
        l = np.asarray(l)
        L = np.append(L,l)

    L = L[2463:2811]
    L = L[::-1]
    L = np.reshape(L,(len(L)//4,4))
    L = 2 - np.log10(100*L[:,0])
    L = L/np.sum(L)

    t_data_file = open("Trioctanoin.txt","r")
    T = []
    for line in t_data_file:
        l = line.split()
        del l[0]
        l = [float(x) for x in l]
        l = np.asarray(l)
        T = np.append(T,l)

    T = T[656:917]
    T = T[::-1]
    T = np.reshape(T,(len(T)//3,3))
    T = 2- np.log10(100*T[:,0])
    T = T/np.sum(T)
    
    return [C, S, L, T]


[C, S, L, T] = read_spectral_data()

H0 = C				# Setting archetypes
H0 = np.vstack([H0, S])
H0 = np.vstack([H0, L])
H0 = np.vstack([H0, T])

X0 = np.dot(W0,H0)
print(W0)
print(C)
print(H0)
print(X0.shape)# Generating data points
X =  X0 + sigma*np.random.normal(0,1,(n,d))
print(X.shape)
print('PASSED')

#W, H, L, Err = palm_nmf(X, r=4, l=0.001,  maxiter=5000, epsilon=1e-6, threshold=1e-8, c1 = 1, c2 = 1, verbose =False, proj_low_dim = False, oracle = True, H0=H0)



W_acc_PALM, H_acc_PALM, L_acc_PALM, Err_acc_PALM = nmf.acc_palm_nmf(X, r=4, maxiter=500, delta = 1e-5, epsilon=1e-6, threshold=1e-8, c1 = 1, c2 = 1, verbose = False, oracle = True, H0=H0)

W_acc_MU, H_acc_MU, L_acc_MU, Err_acc_MU = MUacc(X, U = np.ones((n, r))/r,V = initH(X, r),alpha =1, delta = 0.1, maxiter = 500, H0=H0)

W_acc_HALS, H_acc_HALS, L_acc_HALS, Err_acc_HALS = HALSacc(X, U = np.ones((n, r))/r,V = initH(X, r),alpha = 1, delta = 0.1, maxiter = 500, H0=H0)

W_acc_PGL, H_acc_PGL, L_acc_PGL, Err_acc_PGL = PGLINacc(X ,Winit= np.ones((n, r))/r,Hinit= initH(X, r),alpha =1,delta = 0.1,maxiter = 500,H0=H0)

# Plotting Results and ground truth.

order_H_acc_palm = [i for i in range(0,r)]
for i in range (0,r):
    d = np.linalg.norm(H0[i,:] - H_acc_PALM[0,:])
    for j in range (1,r):
        dj = np.linalg.norm(H0[i,:] - H_acc_PALM[j,:])
        if (dj <= d):
            order_H_acc_palm[i] = j
            d = dj

H_acc_PALM = H_acc_PALM[order_H_acc_palm,:]


order_H_acc_hals = [i for i in range(0,r)]
for i in range (0,r):
    d = np.linalg.norm(H0[i,:] - H_acc_HALS[0,:])
    for j in range (1,r):
        dj = np.linalg.norm(H0[i,:] - H_acc_HALS[j,:])
        if (dj <= d):
            order_H_acc_hals[i] = j
            d = dj

H_acc_HALS = H_acc_HALS[order_H_acc_hals,:]


order_H_acc_PGL = [i for i in range(0,r)]
for i in range (0,r):
    d = np.linalg.norm(H0[i,:] - H_acc_PGL[0,:])
    for j in range (1,r):
        dj = np.linalg.norm(H0[i,:] - H_acc_PGL[j,:])
        if (dj <= d):
            order_H_acc_PGL[i] = j
            d = dj

H_acc_PGL = H_acc_PGL[order_H_acc_PGL,:]


order_H_acc_MU = [i for i in range(0,r)]
for i in range (0,r):
    d = np.linalg.norm(H0[i,:] - H_acc_MU[0,:])
    for j in range (1,r):
        dj = np.linalg.norm(H0[i,:] - H_acc_MU[j,:])
        if (dj <= d):
            order_H_acc_MU[i] = j
            d = dj

H_acc_MU = H_acc_MU[order_H_acc_MU,:]


















############# figures de PALM


fig_h0_palm = plt.figure()
#fig_h0_palm.suptitle("PALM method", fontsize=20)
plt.subplot(4,4,1)
plt.title('PALM')
plt.plot(H0[0,:],"b--")
plt.plot(H_acc_PALM[0,:])
plt.ylabel('Caffeine', fontsize=8)


#fig_h1_palm = plt.figure()
plt.subplot(4,4,5)
plt.plot(H0[1,:],"b--")
plt.plot(H_acc_PALM[1,:])
plt.ylabel('Sucrose', fontsize=8)


#fig_h2_palm = plt.figure()
plt.subplot(4,4,9)
plt.plot(H0[2,:],"b--")
plt.plot(H_acc_PALM[2,:])
plt.ylabel('Lactose', fontsize=8)


#fig_h3_palm = plt.figure()
plt.subplot(4,4,13)
plt.plot(H0[3,:],"b--")
plt.plot(H_acc_PALM[3,:])
plt.ylabel('Trioctanoin', fontsize=8)

plt.show()
plt.xlabel('Wave number index', fontsize=8)



############# figures de HALS



plt.subplot(4,4,2)
plt.title('HALS')
plt.plot(H0[0,:],"b--")
plt.plot(H_acc_HALS[0,:])
# plt.ylabel('Caffeine', fontsize=8)

#fig_h1_palm = plt.figure()
plt.subplot(4,4,6)
plt.plot(H0[1,:],"b--")
plt.plot(H_acc_HALS[1,:])
# plt.ylabel('Sucrose', fontsize=8)

#fig_h2_palm = plt.figure()
plt.subplot(4,4,10)
plt.plot(H0[2,:],"b--")
plt.plot(H_acc_HALS[2,:])
# plt.ylabel('Lactose', fontsize=8)

#fig_h3_palm = plt.figure()
plt.subplot(4,4,14)
plt.plot(H0[3,:],"b--")
plt.plot(H_acc_HALS[3,:])
# plt.ylabel('Trioctanoin', fontsize=8)
plt.show()

plt.xlabel('Wave number index', fontsize=8)


############# figures de PGL


# fig_h0_pglin = plt.figure()
# fig_h0_pglin.suptitle("PGLIN method", fontsize=20)
plt.subplot(4,4,3)
plt.title('PGLIN')
plt.plot(H0[0,:],"b--")
plt.plot(H_acc_PGL[0,:])
# plt.ylabel('Caffeine', fontsize=8)

#fig_h1_palm = plt.figure()
plt.subplot(4,4,7)
plt.plot(H0[1,:],"b--")
plt.plot(H_acc_PGL[1,:])
# plt.ylabel('Sucrose', fontsize=8)

#fig_h2_palm = plt.figure()
plt.subplot(4,4,11)
plt.plot(H0[2,:],"b--")
plt.plot(H_acc_PGL[2,:])
# plt.ylabel('Lactose', fontsize=8)

#fig_h3_palm = plt.figure()
plt.subplot(4,4,15)
plt.plot(H0[3,:],"b--")
plt.plot(H_acc_PGL[3,:])
# plt.ylabel('Trioctanoin', fontsize=8)
plt.xlabel('Wave number index', fontsize=8)
plt.show()



############# figures de MU


#fig_h0_palm = plt.figure()
# fig_h0_mu = plt.figure()
# fig_h0_mu.suptitle("MU method", fontsize=20)
plt.subplot(4,4,4)
plt.title('MU')
plt.plot(H0[0,:],"b--")
plt.plot(H_acc_MU[0,:])
# plt.ylabel('Caffeine', fontsize=8)

#fig_h1_palm = plt.figure()
plt.subplot(4,4,8)
plt.plot(H0[1,:],"b--")
plt.plot(H_acc_MU[1,:])
# plt.ylabel('Sucrose', fontsize=8)

#fig_h2_palm = plt.figure()
plt.subplot(4,4,12)
plt.plot(H0[2,:],"b--")
plt.plot(H_acc_MU[2,:])
# plt.ylabel('Lactose', fontsize=8)

#fig_h3_palm = plt.figure()
plt.subplot(4,4,16)
plt.plot(H0[3,:],"b--")
plt.plot(H_acc_MU[3,:])
# plt.ylabel('Trioctanoin', fontsize=8)
plt.xlabel('Wave number index', fontsize=8)
plt.show()





fig_err = plt.figure()
plt.plot(Err_acc_MU,"-r",label = 'MU_acc')
plt.plot(Err_acc_PALM,'-m',label = 'PALM_acc')
plt.plot(Err_acc_HALS, '-g',label = 'HALS_acc')
plt.plot(Err_acc_PGL,"-b",label = 'PGL_acc')
plt.legend(loc="upper right")
fig_err.suptitle('error', fontsize=20)
#plt.yscale('log')
plt.xlabel('iteration', fontsize=18)
plt.ylabel('Reconstruction Error vs iteration', fontsize=16)
plt.show()


fig_loss = plt.figure()
plt.plot(L_acc_MU,"-r",label = 'MU_acc')
plt.plot(L_acc_PALM,'-m',label = 'PALM_acc')
plt.plot(L_acc_HALS, '-g',label = 'HALS_acc')
plt.plot(L_acc_PGL,"-b",label = 'PGL_acc')
plt.legend(loc="upper right")
#plt.yscale('log')
fig_loss.suptitle('Loss', fontsize=20)
plt.xlabel('iteration', fontsize=18)
plt.ylabel('L(H,H_true)', fontsize=16)
plt.show()



fig_orig = plt.figure()

plt.subplot(4,1,1)
plt.title('original spectra')
plt.plot(H0[0,:],"b--")
plt.ylabel('Caffeine', fontsize=8)
plt.subplot(4,1,2)
plt.plot(H0[1,:],"b--")
plt.ylabel('Sucrose', fontsize=8)
plt.subplot(4,1,3)
plt.plot(H0[2,:],"b--")
plt.ylabel('Lactose', fontsize=8)
plt.subplot(4,1,4)
plt.plot(H0[3,:],"b--")
plt.ylabel('Trioctanoin', fontsize=8)
plt.xlabel('Wave number index', fontsize=8)

print("Temps d execution : %s secondes ---" % (time.time() - start_time))