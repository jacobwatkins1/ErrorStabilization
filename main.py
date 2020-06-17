import numpy as np
from overlap import *
import Nstep
verbose = True

def stabilize(N,H,ntrials_nmat=10000,ntrials_hmat=10000,metro_step=0.1,nmat_delta=0.0001,errsize_nmat=0.01,errsize_hmat=0.01,lowest_order_ratio=3,convergence_ratio=0.6):
    if verbose: print('Stabilization Running')
    maxSize = np.max(N.shape)
    nmat = N.copy()
    hmat = H.copy()
    mineig = np.min(np.linalg.eigvals(nmat))
    guide = 1/(np.exp(-mineig/nmat_delta)+1) #How is nmat_delta defined?
    gauss = np.exp(-sum(sum((nmat-N)**2))/(2*errsize_nmat**2)) #How is errsize_nmat known?

    accept_nmat = 0
    accept_hmat = 0
    order = H.shape[0]
    e_list = np.zeros((0,order))
    overlap_list = np.zeros((0,order))
    max_order_overlap_list = np.zeros((0,order))

#    np.random.seed(13)

    for ii in range(ntrials_nmat):
        nmat,guide,gauss,accDelta,mineig = Nstep.__Nstep__(nmat,N,errsize_nmat,metro_step,guide,gauss,nmat_delta)
        accept_nmat = accept_nmat + accDelta
        print(mineig)
        if mineig>0:
            if verbose: print('accepted N matrix',flush=True)
            fraction = ntrials_hmat/guide-np.floor(ntrials_hmat/guide)
            if np.random.rand()<fraction:
                iterations_hmat=np.floor(ntrials_hmat/guide)+1
            else:
                iterations_hmat=np.floor(ntrials_hmat/guide)

            for jj in range(int(iterations_hmat)):
                #order = H.size[0]
                hmat_err = np.random.rand(order,order)
                hmat_err = (hmat_err+hmat_err.T)/2*np.sqrt(2)
                hmat=H+errsize_hmat*hmat_err

                e = []
                for k in range(1,order+1):
                    e.append(np.min(np.linalg.eigvals(np.dot(nmat[:k,:k],hmat[:k,:k]))))
                    
                concave = True
                for k in range(lowest_order_ratio,order):
                    if (np.abs(e[k-1]-e[k]) > convergence_ratio*np.abs(e[k-2]-e[k-1])):
                        concave = False
                print(np.abs(e[k-1]-e[k])/np.abs(e[k-2]-e[k-1]))
                if concave:
                    if verbose: print('h matrix accepted',flush=True)
                    accept_hmat = accept_hmat+1
                    e_list = np.vstack((e_list,e))
                    if (e_list.shape[0] > 1):
                        print('e_exact: ',e_exact)
                        print('e_start: ',e_start)
                        print('e_list mean: ',np.mean(e_list))
                        print('e_list std: ',np.std(e_list))
                        print()
                    
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


