import numpy as np

N = 100
order = 5
lowest_order_ratio = 3
convergence_ratio = 0.6
EC_step_coupling = 0.1
target_coupling = 1.0
errsize_nmat = 0.001
errsize_hmat = 0.01
ntrials_nmat = 1000000
ntrials_hmat = 10000
metro_step = 0.1
nmat_delta = 0.0001


np.random.seed(3)

H0 = 2*np.random.rand(N,N)-1
H1 = 2*np.random.rand(N,N)-1
H0 = (H0+H0.T)/2
H1 = (H1+H1.T)/2

np.random.seed(13)
v = np.zeros([N,order])
for nc in range(order):
    vv,dd = np.linalg.eig(H0+nc*EC_step_coupling*H1)
    v[:,nc] = vv[:,0]

Ht = H0 + target_coupling*H1
E0 = np.min(np.linalg.eigvals(Ht))

nmat_exact = np.dot(v.T,v)
hmat_exact = np.dot(v.T,np.dot(Ht,v))

nmat_err = np.random.rand(order,order)
nmat_err = (nmat_err + nmat_err.T)/2*np.sqrt(2)
nmat_start = nmat_exact+errsize_nmat*nmat_err

hmat_err = randn(order,order)
hmat_err = (hmat_err + hmat_err.T)/2*np.sqrt(2)
hmat_start = hmat_exact + errsize_hmat*hmat_err

vsmall_exact = np.zeros(order,order)
for k in range(order):
    vtemp,dtemp = np.linalg.eig(np.dot(nmat_exact[:k,:k],np.linalg.inv(hmat_exact[:k,:k])))
    vsmall_exact[:k,k]=vtemp[:k,np.argmin(dtemp)]

e_exact = np.array([min(np.linalg.eig(np.dot(nmat_exact[:k,:k],np.linalg.inv(hmat_exact[:k,:k])))) for k in range(order)])
e_start = np.array([min(np.linalg.eig(np.dot(nmat_start[:k,:k],np.linalg.inv(hmat_start[:k,:k])))) for k in range(order)])

print(e_exact)
print(e_start)
overlap_start = np.zeros(k)
max_order_overlap_start = np.zeros(k)
max_order_overlap_exact = np.zeros(k)

for k in range(order):
    vtemp,dtemp = np.linalg.eig(np.dot(nmat_start[:k,:k],np.linalg.inv(hmat_start[:k,:k])))
    vsmall_start[:k,k] = vtemp[:k, np.argmin(dtemp)]
    overlap_start[k]=np.abs(vsmall_start[:k,k].T
