import numpy as np
import main

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


np.random.seed(7) #If set to 4, converged instantly. When set to 3, did not converge

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

main.stabilize(nmat_start,hmat_start)
