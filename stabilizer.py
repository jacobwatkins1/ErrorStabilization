# Error stabilizing algorithm for solving noisy generalized eigenvector problem
#
# N matrices are sampled using Metropolis algorithm. 
#
# Adapted for python from Dean Lee's Matlab code, May 2020

import numpy as np

N = 100                 # Dimension of Hilbert space
order = 5               # Max order of eigenvector continuation
lowest_order_ratio = 3  # Lowest order where convergence test is applied
convergence_ratio = 0.6 # Standard for convergence rate
EC_step_coupling = 0.1  # Increments in EC parameter
target_coupling = 1.0   
errsize_nmat = 0.001    # Error sizes of N and H matrices
errsize_hmat = 0.01
ntrials_nmat = 1000
ntrials_hmat = 1000
metro_step = 0.1
nmat_delta = 0.0001


np.random.seed(3)

# The following section 
H0 = 2*np.random.rand(N,N)-1
H1 = 2*np.random.rand(N,N)-1
H0 = (H0+H0.T)/2
H1 = (H1+H1.T)/2

np.random.seed(13)
# Extract ground states for small coupling, put them in matrix vv
v = np.zeros((N,order))

for nc in range(order):
    dd,vv = np.linalg.eig(H0+nc*EC_step_coupling*H1)
    v[:,nc] = vv[:,0]

# Target Hamiltonian and desired ground state
Ht = H0 + target_coupling*H1
E0 = np.min(np.linalg.eigvals(Ht))

# Exact N matrix and Hamiltonian in EC subspace
nmat_exact = np.dot(v.T,v)
hmat_exact = np.dot(v.T,np.dot(Ht,v))

# Generate noisy N matrix 
nmat_err = np.random.rand(order,order)
nmat_err = (nmat_err + nmat_err.T)/2*np.sqrt(2)
nmat_start = nmat_exact+errsize_nmat*nmat_err

# Generate noisy H matrix 
hmat_err = np.random.rand(order,order)
hmat_err = (hmat_err + hmat_err.T)/2*np.sqrt(2)
hmat_start = hmat_exact + errsize_hmat*hmat_err

# Solve eigenvalues/vectors of H exactly, in subspace
vsmall_exact = np.zeros((order,order))
for k in range(1,order):
    dtemp,vtemp = np.linalg.eig(np.dot(nmat_exact[:k,:k],np.linalg.inv(hmat_exact[:k,:k])))
    vsmall_exact[:k,k]=vtemp[:k,np.argmin(dtemp)]

# List of ground states at each order k of EC, exact and with noise
e_exact = np.array([min(np.linalg.eigvals(np.dot(nmat_exact[:k,:k],np.linalg.inv(hmat_exact[:k,:k])))) for k in range(1,order+1)])
e_start = np.array([min(np.linalg.eigvals(np.dot(nmat_start[:k,:k],np.linalg.inv(hmat_start[:k,:k])))) for k in range(1,order+1)])

print("Exact gs energy at each order:", e_exact)
print("With noise:", e_start)

# Method for computing overlaps. Inputs are vectors u and v, and norm matrix N
# sandwiching them
def overlap(u,v,N):
    dotp = np.dot(np.dot(u.T,N),v)
    normu = np.sqrt(np.abs(np.dot(np.dot(u.T,N),u)))
    normv = np.sqrt(np.abs(np.dot(np.dot(v.T,N),v)))
    return np.abs(dotp)/(normu*normv)

overlap_start = np.zeros(order)
max_order_overlap_start = np.zeros(order)
max_order_overlap_exact = np.zeros(order)
vsmall_start = np.zeros((order,order))
for k in range(1,order+1):
    dtemp,vtemp = np.linalg.eig(np.dot(nmat_start[:k,:k],np.linalg.inv(hmat_start[:k,:k])))
    vsmall_start[:k,k-1] = vtemp[:k, np.argmin(dtemp)]
    overlap_start[k-1]= \
            np.abs(np.dot(np.dot(vsmall_exact[:k,k-1].T,nmat_exact[:k,:k]),vsmall_start[:k,k-1]))/\
            np.sqrt(np.abs(np.dot(np.dot(vsmall_exact[:k,k-1].T,nmat_exact[:k,:k]),vsmall_exact[:k,k-1])*\
            np.abs(np.dot(np.dot(vsmall_start[:k,k-1].T,nmat_exact[:k,:k]),vsmall_start[:k,k-1]))))
    max_order_overlap_start[k-1] = \
            np.abs(np.dot(np.dot(vsmall_exact[:order,order-1].T,nmat_exact[:order,:k]),vsmall_start[:k,k-1]))/\
            np.sqrt(np.abs(np.dot(np.dot(vsmall_exact[:order,order-1].T,nmat_exact[:order,:order]),vsmall_exact[:order,order-1])*\
            np.abs(np.dot(np.dot(vsmall_start[:k,k-1].T,nmat_exact[:k,:k]),vsmall_start[:k,k-1]))))
    max_order_overlap_exact[k-1] = \
            np.abs(np.dot(np.dot(vsmall_exact[:order,order-1].T,nmat_exact[:order,:k]),vsmall_exact[:k,k-1]))/\
            np.sqrt(np.abs(np.dot(np.dot(vsmall_exact[:order,order-1].T,nmat_exact[:order,:order]),vsmall_exact[:order,order-1])*\
            np.abs(np.dot(np.dot(vsmall_exact[:k,k-1].T,nmat_exact[:k,:k]),vsmall_exact[:k,k-1]))))


print(overlap_start)
print(max_order_overlap_start)
print(max_order_overlap_exact)

# The following section is independent of the above (save some initial variable copying)
nmat = nmat_start.copy()
mineig = np.min(np.linalg.eigvals(nmat))
guide = 1/(np.exp(-mineig/nmat_delta)+1)
gauss = np.exp(-sum(sum((nmat-nmat_start)**2))/(2*errsize_nmat**2))

e_list = np.zeros(order)
accept_nmat = 0
accept_hmat = 0
e_list = np.zeros((0,order))
overlap_list = np.zeros((0,order))
max_order_overlap_list = np.zeros((0,order))

for ii in range(ntrials_nmat):

    nmat_eps = metro_step*errsize_nmat*np.random.rand(order,order)
    nmat_eps = (nmat_eps + nmat_eps.T)/2*np.sqrt(2)
    nmat_new = nmat+nmat_eps
    mineig_new = np.min(np.linalg.eigvals(nmat_new))
    guide_new = 1/(np.exp(-mineig_new/nmat_delta)+1)
    gauss_new = np.exp(-sum(sum((nmat_new-nmat_start)**2))/(2*errsize_nmat**2))

    if (np.random.rand() < (guide_new*gauss_new)/(guide*gauss)):
        nmat = nmat_new
        guide = guide_new
        gauss = gauss_new
        accept_nmat = accept_nmat + 1
        mineig = mineig_new

    # If N is positive definite...
    if (mineig)>0:
        fraction = ntrials_hmat/guide - np.floor(ntrials_hmat/guide)
        if np.random.rand()<fraction:
            iterations_hmat = np.floor(ntrials_hmat/guide)+1
        else:
            iterations_hmat = np.floor(ntrials_hmat/guide)
        
        for jj in range(int(iterations_hmat)):

            hmat_err = np.random.rand(order,order)
            hmat_err = (hmat_err+hmat_err.T)/2*np.sqrt(2)
            hmat = hmat_start + errsize_hmat*hmat_err
            e = []
            for k in range (1,order+1):
                e.append(np.min(np.linalg.eigvals(np.dot(nmat[:k,:k],hmat[:k,:k]))))
            concave = 1
            for k in range(lowest_order_ratio,order):
                if (np.abs(e[k-1]-e[k]) > convergence_ratio*np.abs(e[k-2]-e[k-1])):
                    concave = 0
            if concave == 1:
                accept_hmat = accept_hmat + 1
                e_list = np.vstack((e_list,e))
                if (e_list.shape[0] > 1):
                    print(e_exact,e_start,np.mean(e_list,0),np.std(e_list,0))

                overlap_temp = []
                max_order_overlap_temp = []
                for k in range(order):
                    vtemp,dtemp = np.linalg.eig(np.dot(nmat[:k+1,:k],np.linalg.inv(hmat[:k+1,:k+1])))
                    vsmall[:k+1,k] = vtemp[:k+1, np.argmin(dtemp)]
                    overlap_temp[k] = \
                        np.abs(np.dot(np.dot(vsmall_exact[:k+1,k].T,nmat_exact[:k+1,:k+1]),vsmall[:k+1,k]))/\
                        np.sqrt(np.abs(np.dot(np.dot(vsmall_exact[:k+1,k].T,nmat_exact[:k+1,:k+1]),vsmall_exact[:k+1,k])*\
                        np.abs(np.dot(np.dot(vsmall[:k+1,k].T,nmat_exact[:k+1,:k+1]),vsmall[:k+1,k]))))
                    max_order_overlap_temp[k] = \
                        np.abs(np.dot(np.dot(vsmall_exact[:order,order].T,nmat_exact[:order,:k+1]),vsmall[:k+1,k]))/\
                        np.sqrt(np.abs(np.dot(np.dot(vsmall_exact[:order,order].T,nmat_exact[:order,:order]),vsmall_exact[:order,order])*\
                        np.abs(np.dot(np.dot(vsmall[:k+1,k].T,nmat_exact[:k+1,:k]),vsmall[:k+1,k]))))
                overlap_list = np.vstack((overlap_list,overlap_temp))
                max_order_overlap_list = np.vstack((max_order_overlap_list,overlap_temp))
                
                if e_list.shape[1] > 1:
                    print(np.ones((1,order)),overlap_start,np.mean(overlap_start,0),np.std(overlap_start,0))
                    print(max_order_overlap_exact,max_order_overlap_start,np.mean(max_order_overlap_list,0),np.std(max_order_overlap_list,0))




