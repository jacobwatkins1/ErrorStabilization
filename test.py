import numpy as np
import main

N = 100                 # Dimension of Hilbert space
order = 5             # Max order of eigenvector continuation
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
autotune=True

np.random.seed(9) #

# The following section
H0 = 2*np.random.rand(N,N)-1
H1 = 2*np.random.rand(N,N)-1
H0 = (H0+H0.T)/2
H1 = (H1+H1.T)/2

np.random.seed(1)
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
nmat_err = 2*np.random.rand(order,order)-1
nmat_err = (nmat_err + nmat_err.T)/2*np.sqrt(2)
nmat_start = nmat_exact+errsize_nmat*nmat_err

# Generate noisy H matrix
hmat_err = 2*np.random.rand(order,order)-1
hmat_err = (hmat_err + hmat_err.T)/2*np.sqrt(2)
hmat_start = hmat_exact + errsize_hmat*hmat_err

# Solve eigenvalues/vectors of H exactly, in subspace
vsmall_exact = np.zeros((order,order))
for k in range(1,order+1):
    dtemp,vtemp = np.linalg.eig(np.dot(np.linalg.inv(nmat_exact[:k,:k]),hmat_exact[:k,:k]))
    vsmall_exact[:k,k-1]=vtemp[:k,np.argmin(dtemp)]

# List of ground states at each order k of EC, exact and with noise
e_exact = np.array([min(np.linalg.eigvals(np.dot(np.linalg.inv(nmat_exact[:k,:k]),hmat_exact[:k,:k]))) for k in range(1,order+1)])
e_start = np.array([min(np.linalg.eigvals(np.dot(np.linalg.inv(nmat_start[:k,:k]),hmat_start[:k,:k]))) for k in range(1,order+1)])

#print('hmat_start')
#print(hmat_start)
#print('hmat_exact')
#print(hmat_exact)
#print('nmat_start')
#print(nmat_start)
#print('nmat_exact')
#print(nmat_exact)
#print(np.min(np.linalg.eigvals(nmat_exact)))
#print(np.min(np.linalg.eigvals(nmat_start)))



print(E0)
print("Exact gs energy at each order:", e_exact)
print("With noise:", e_start)
print('',flush=True)
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
    return np.abs(dotp)**2/(np.sqrt(np.abs(inprod(u,u,N[:usize,:usize])*inprod(v,v,N[:vsize,:vsize]))))**2

overlap_start = np.zeros(order)
max_order_overlap_start = np.zeros(order)
max_order_overlap_exact = np.zeros(order)
vsmall_start = np.zeros((order,order))

#Compute overlaps (i.e. cosine theta) at each order of EC
for k in range(1,order+1):
    dtemp,vtemp = np.linalg.eig(np.dot(np.linalg.inv(nmat_start[:k,:k]),hmat_start[:k,:k]))
    vsmall_start[:k,k-1] = vtemp[:k, np.argmin(dtemp)]

for k in range(1,order+1):
    overlap_start[k-1] = overlap(vsmall_exact[:k,k-1],vsmall_start[:k,k-1],nmat_exact[:k,:k])
    max_order_overlap_start[k-1] = overlap(vsmall_exact[:order,order-1],vsmall_start[:k,k-1],nmat_exact[:order,:order])
    max_order_overlap_exact[k-1] = overlap(vsmall_exact[:order,order-1],vsmall_exact[:k,k-1],nmat_exact[:order,:order])

#Hstab,Nstab,e_stab = main.stabilize(nmat_start,hmat_start,errsize_nmat=errsize_nmat,errsize_hmat=errsize_hmat,autotune=autotune)

e_stab,vsmall_stab = main.stabilize(nmat_start,hmat_start,autotune=autotune,errsizeN=errsize_nmat,errsizeH=errsize_hmat)
print("Overlap of...")
print("noisy vs exact EC at each order:\t\t", overlap_start)
print("noisy kth order with with exact max order:\t", max_order_overlap_start)
print("kth exact with exact max order:\t", max_order_overlap_exact)

#e = vsmall_start[:,-1]
#concavity = np.max([np.abs(e[k-1]-e[k])/np.abs(e[k-2]-e[k-1]) for k in range(lowest_order_ratio,order)])
#print(concavity)



overlap_stab = np.zeros(order)
max_order_overlap_stab = np.zeros(order)
#vsmall_stab = np.zeros((order,order))
#for k in range(1,order+1):
#    dtemp,vtemp = np.linalg.eig(np.dot(np.linalg.inv(Nstab[:k,:k]),Hstab[:k,:k]))
#    vsmall_stab[:k,k-1] = vtemp[:k, np.argmin(dtemp)]

for k in range(1,order+1):
    overlap_stab[k-1] = overlap(vsmall_exact[:k,k-1],vsmall_stab[:k,k-1],nmat_exact[:k,:k])
    max_order_overlap_stab[k-1] = overlap(vsmall_exact[:order,order-1],vsmall_stab[:k,k-1],nmat_exact[:order,:order])



#print('n exact')
#print(nmat_exact)
#print('h exact')
#print(hmat_exact)
#print()
#print('n stabilized')
#print(Nstab)
#print('h stabilized')
#print(Hstab)

print("stabilized vs exact EC at each order:\t\t", overlap_stab)
print("stabilized kth order with with exact max order:\t", max_order_overlap_stab)
print()

print('E exact')
print(E0)
print('exact gs energies:')
print(e_exact)

print()
print('noisy gs energies:')
print(e_start)

print()
print('stabilized gs energies:')
print(e_stab)

print()
print('noisy error:')
print(np.abs(np.array(e_start)-np.array(e_exact))/np.abs(np.array(e_exact)))

print()
print('Stabilized error:')
print(np.abs(np.array(e_stab)-np.array(e_exact))/np.abs(np.array(e_exact)))





print()
#e2stab = []
#e2start = []
#e2exact = []
#for k in range(2,order+1):
#    e2stab.append(np.sort(np.linalg.eigvals(np.dot(np.linalg.inv(Nstab[:k,:k]),Hstab[:k,:k])))[1])
#    e2start.append(np.sort(np.linalg.eigvals(np.dot(np.linalg.inv(nmat_start[:k,:k]),hmat_start[:k,:k])))[1])
#    e2exact.append(np.sort(np.linalg.eigvals(np.dot(np.linalg.inv(nmat_exact[:k,:k]),hmat_exact[:k,:k])))[1])

#print(e2exact)
#print(e2start)
#print(e2stab)



#e_list=[]


#if e_list.shape[0]>1:
#    print('e_exact: ',e_exact)
#    print('e_start: ',e_start)
#    print('e_list mean: ',np.mean(e_list))
#    print('e_list std: ',np.std(e_list))
#    print()
#    print(e_list)
#    print()
                    
 #   overlap_temp = np.zeros(order)
 #   max_order_overlap_temp = np.zeros(order)
 #   vsmall = np.zeros((order,order))
#
 #   overlap_list = np.zeros((0,order))
 #   max_order_overlap_list = np.zeros((0,order))



  #  for hmat in Hlist:
  #      for k in range(order):
  #          dtemp,vtemp = np.linalg.eig(np.dot(nmat[:k+1,:k+1],np.linalg.inv(hmat[:k+1,:k+1])))
  #          vsmall[:k+1,k] = vtemp[:k+1, np.argmin(dtemp)]
  #          overlap_temp[k] = overlap(vsmall_exact[:k+1,k], vsmall[:k+1,k],nmat_exact[:k+1,:k+1])
  #          max_order_overlap_temp[k] = overlap(vsmall_exact[:order,order-1],vsmall[:k+1,k],nmat_exact[:order,:order])
  #              # Store results in overlap list
  #          overlap_list = np.vstack((overlap_list,overlap_temp))
  #          max_order_overlap_list = np.vstack((max_order_overlap_list,overlap_temp))


   #         if e_list.shape[1] > 1:
   #             print()
   #             print(np.ones((1,order)))
   #             print('overlap_start: ',overlap_start)
   #             print('overlap_start mean: ',np.mean(overlap_start))
   #             print('overlap_start std: ',np.std(overlap_start))
#
#                print()
#                print('max_order_overlap_exact: ',max_order_overlap_exact)
#                print('max_order_overlap_start: ',max_order_overlap_start)
#                print('max_order_overlap_list mean: ',np.mean(max_order_overlap_list))
#                print('max_order_overlap_list std: ',np.std(max_order_overlap_list))
#                print('---------------------------------')


