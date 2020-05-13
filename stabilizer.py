# Error stabilizing algorithm for solving noisy generalized eigenvector problem
#
# N matrices are sampled using Metropolis algorithm. 
#
# Adapted for python from Dean Lee's Matlab code, May 2020

import numpy as np

N = 100                 # Dimension of Hilbert space
order = 5               # Max order of eigenvector continuation
lowest_order_ratio = 3  # Lowest order where convergence test is applied
convergence_ratio = .6 # Standard for convergence rate
EC_step_coupling = 0.1  # Increments in EC parameter
target_coupling = 1.0   
errsize_nmat = 0.01   # Error sizes of N and H matrices
errsize_hmat = 0.01
ntrials_nmat = 10000
ntrials_hmat = 10000
metro_step = 0.1
nmat_delta = 0.0001


np.random.seed(4) #If set to 4, converged instantly. When set to 3, did not converge

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

# Methods for computing overlaps. Inputs are vectors u and v, and norm matrix N
# sandwiched between them
def inprod(u, v, N):
    return np.dot(np.dot(u.T,N),v)

#Overlap is cosine of angle between vectors, absolute value. 
def overlap(u, v, N):
    usize = u.size
    vsize = v.size
    dotp = inprod(u, v, N[:usize,:vsize])
    if dotp ==0: return 0
    return np.abs(dotp)/(np.sqrt(np.abs(inprod(u,u,N[:usize,:usize])*inprod(v,v,N[:vsize,:vsize]))))

overlap_start = np.zeros(order)
max_order_overlap_start = np.zeros(order)
max_order_overlap_exact = np.zeros(order)
vsmall_start = np.zeros((order,order))

#Compute overlaps (i.e. cosine theta) at each order of EC
for k in range(1,order+1):
    dtemp,vtemp = np.linalg.eig(np.dot(nmat_start[:k,:k],np.linalg.inv(hmat_start[:k,:k])))
    vsmall_start[:k,k-1] = vtemp[:k, np.argmin(dtemp)]
    
    overlap_start[k-1] = overlap(vsmall_exact[:k,k-1],vsmall_start[:k,k-1],nmat_exact[:k,:k])
    max_order_overlap_start[k-1] = overlap(vsmall_exact[:order,order-1],vsmall_start[:k,k-1],nmat_exact[:order,:order])
    max_order_overlap_exact[k-1] = overlap(vsmall_exact[:order,order-1],vsmall_exact[:k,k-1],nmat_exact[:order,:order])

print("Overlap of...")
print("noisy vs exact EC at each order:", overlap_start)
print("noisy kth order with with exact max order:", max_order_overlap_start)
print("kth exact with exact max order:", max_order_overlap_exact)

# The following section is essentially independent of the above (save some initial variable copying)

nmat = nmat_start.copy()                  # Initial N for Metropolis walk
mineig = np.min(np.linalg.eigvals(nmat))  # Minium eigenvalue of N
guide = 1/(np.exp(-mineig/nmat_delta)+1)  # Logistic function for filtering ill-behaved N matrices
gauss = np.exp(-sum(sum((nmat-nmat_start)**2))/(2*errsize_nmat**2)) 

e_list = np.zeros(order) # 
accept_nmat = 0          # Counts number of accepted N matrices
accept_hmat = 0          # Counts number of accepted H matrices
e_list = np.zeros((0,order))
overlap_list = np.zeros((0,order))
max_order_overlap_list = np.zeros((0,order))

# Loop over N sampling
for ii in range(ntrials_nmat):

    #Step to new N matrix
    nmat_eps = metro_step*errsize_nmat*np.random.rand(order,order)
    nmat_eps = (nmat_eps + nmat_eps.T)/2*np.sqrt(2)
    nmat_new = nmat+nmat_eps

    # Test for acceptance. We ask nmat_new be close to the old, and mineig_new be positive
    mineig_new = np.min(np.linalg.eigvals(nmat_new))
    guide_new = 1/(np.exp(-mineig_new/nmat_delta)+1)
    gauss_new = np.exp(-sum(sum((nmat_new-nmat_start)**2))/(2*errsize_nmat**2))
    if (np.random.rand() < (guide_new*gauss_new)/(guide*gauss)):
        #Commit to step
        nmat = nmat_new
        guide = guide_new
        gauss = gauss_new
        accept_nmat = accept_nmat + 1
        mineig = mineig_new

    #Check n_mat positive definite...
    if (mineig)>0:
        print('accepted N matrix',flush=True)
        fraction = ntrials_hmat/guide - np.floor(ntrials_hmat/guide)
        if np.random.rand()<fraction:
            iterations_hmat = np.floor(ntrials_hmat/guide)+1
        else:
            iterations_hmat = np.floor(ntrials_hmat/guide)

        # Sample H matrices using simple Gaussian sampling (likely to be changed in future version)
        for jj in range(int(iterations_hmat)):

            # Generate noisy H
            hmat_err = np.random.rand(order,order)
            hmat_err = (hmat_err+hmat_err.T)/2*np.sqrt(2)
            hmat = hmat_start + errsize_hmat*hmat_err
            
            e = [] # list of gs energies at each order (ie each submatrix)
            for k in range(1,order+1):
                e.append(np.min(np.linalg.eigvals(np.dot(nmat[:k,:k],hmat[:k,:k]))))

            concave = True
            #print(np.abs(e[lowest_order_ratio-2]-e[lowest_order_ratio-1]))

            # Test for concavity/convergence, characterized by convergence_ratio
            for k in range(lowest_order_ratio,order):
                if (np.abs(e[k-1]-e[k]) > convergence_ratio*np.abs(e[k-2]-e[k-1])):
                    concave = False                   
                #print(np.abs(e[k-1]-e[k])/np.abs(e[k-2]-e[k-1]))
             # If convergence is smooth, accept
            if concave:
                print('accepted H matrix',flush=True)
                accept_hmat = accept_hmat + 1
                e_list = np.vstack((e_list,e))
                if (e_list.shape[0] > 1):
                    print('e_exact: ',e_exact)
                    print('e_start: ',e_start)
                    print('e_list mean: ',np.mean(e_list))
                    print('e_list std: ',np.std(e_list))
                    print()

                    #print(e_exact,e_start,np.mean(e_list,0),np.std(e_list,0))

                # Compute overlaps, noisy vs exact at each order, and noisy vs highest exact order
                overlap_temp = np.zeros(order)
                max_order_overlap_temp = np.zeros(order)
                vsmall = np.zeros((order,order))
                for k in range(order):
                    dtemp,vtemp = np.linalg.eig(np.dot(nmat[:k+1,:k+1],np.linalg.inv(hmat[:k+1,:k+1])))
                    vsmall[:k+1,k] = vtemp[:k+1, np.argmin(dtemp)]
                    overlap_temp[k] = overlap(vsmall_exact[:k+1,k], vsmall[:k+1,k],nmat_exact[:k+1,:k+1])
                    max_order_overlap_temp[k] = overlap(vsmall_exact[:order,order-1],vsmall[:k+1,k],nmat_exact[:order,:order])
                # Store results in overlap list
                overlap_list = np.vstack((overlap_list,overlap_temp))
                max_order_overlap_list = np.vstack((max_order_overlap_list,overlap_temp))

                
                if e_list.shape[1] > 1:
                    print()
                    print(np.ones((1,order)))
                    print('overlap_start: ',overlap_start)
                    print('overlap_start mean: ',np.mean(overlap_start))
                    print('overlap_start std: ',np.std(overlap_start))

                    print()
                    print('max_order_overlap_exact: ',max_order_overlap_exact)
                    print('max_order_overlap_start: ',max_order_overlap_start)
                    print('max_order_overlap_list mean: ',np.mean(max_order_overlap_list))
                    print('max_order_overlap_list std: ',np.std(max_order_overlap_list))
                    print('---------------------------------')
                    #print(np.ones((1,order)),overlap_start,np.mean(overlap_start),np.std(overlap_start))
                    #print(max_order_overlap_exact,max_order_overlap_start,np.mean(max_order_overlap_list,0),np.std(max_order_overlap_list,0))

    #vsmall_start[:k,k-1] = vtemp[:k, np.argmin(dtemp)]

#    \
 #           np.abs(np.dot(np.dot(vsmall_exact[:order,order-1].T,nmat_exact[:order,:k]),vsmall_start[:k,k-1]))/\
  #          np.sqrt(np.abs(np.dot(np.dot(vsmall_exact[:order,order-1].T,nmat_exact[:order,:order]),vsmall_exact[:order,order-1])*\
   #         np.abs(np.dot(np.dot(vsmall_start[:k,k-1].T,nmat_exact[:k,:k]),vsmall_start[:k,k-1]))))
#overlap_start[k-1]= \
     #    np.abs(np.dot(np.dot(vsmall_exact[:k,k-1].T,nmat_exact[:k,:k]),vsmall_start[:k,k-1]))/\
    #     np.sqrt(np.abs(np.dot(np.dot(vsmall_exact[:k,k-1].T,nmat_exact[:k,:k]),vsmall_exact[:k,k-1])*\
     #    np.abs(np.dot(np.dot(vsmall_start[:k,k-1].T,nmat_exact[:k,:k]),vsmall_start[:k,k-1]))))

#    \
 #           np.abs(np.dot(np.dot(vsmall_exact[:order,order-1].T,nmat_exact[:order,:k]),vsmall_start[:k,k-1]))/\
  #          np.sqrt(np.abs(np.dot(np.dot(vsmall_exact[:order,order-1].T,nmat_exact[:order,:order]),vsmall_exact[:order,order-1])*\
   #         np.abs(np.dot(np.dot(vsmall_start[:k,k-1].T,nmat_exact[:k,:k]),vsmall_start[:k,k-1]))))

#    \
#            np.abs(np.dot(np.dot(vsmall_exact[:order,order-1].T,nmat_exact[:order,:k]),vsmall_exact[:k,k-1]))/\
#            np.sqrt(np.abs(np.dot(np.dot(vsmall_exact[:order,order-1].T,nmat_exact[:order,:order]),vsmall_exact[:order,order-1])*\
#            np.abs(np.dot(np.dot(vsmall_exact[:k,k-1].T,nmat_exact[:k,:k]),vsmall_exact[:k,k-1]))))

 #                   \
  #                      np.abs(np.dot(np.dot(vsmall_exact[:k+1,k].T,nmat_exact[:k+1,:k+1]),vsmall[:k+1,k]))/\
   #                     np.sqrt(np.abs(np.dot(np.dot(vsmall_exact[:k+1,k].T,nmat_exact[:k+1,:k+1]),vsmall_exact[:k+1,k])*\
    #                    np.abs(np.dot(np.dot(vsmall[:k+1,k].T,nmat_exact[:k+1,:k+1]),vsmall[:k+1,k]))))

 #                    \
  #                      np.abs(np.dot(np.dot(vsmall_exact[:order,order].T,nmat_exact[:order,:k+1]),vsmall[:k+1,k]))/\
   #                     np.sqrt(np.abs(np.dot(np.dot(vsmall_exact[:order,order].T,nmat_exact[:order,:order]),vsmall_exact[:order,order])*\
    #                    np.abs(np.dot(np.dot(vsmall[:k+1,k].T,nmat_exact[:k+1,:k]),vsmall[:k+1,k]))))
