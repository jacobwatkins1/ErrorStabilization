import numpy as np

def __Nstep__(N,nmat_start,err,s,guide,gauss,nmat_delta):

    maxSize = np.max(N.shape)
    
    nmat_eps = s*err*np.random.rand(maxSize,maxSize)
    nmat_eps = (nmat_eps+nmat_eps.T)/2*np.sqrt(2)

    nmat_new = N + nmat_eps

    min_eig_new = np.min(np.linalg.eigvals(nmat_new))
    guide_new = 1/(np.exp(-min_eig_new/nmat_delta)+1)
    gauss_new = np.exp(-sum(sum((nmat_new-nmat_start)**2))/(2*err**2))
    if (np.random.rand() < (guide_new*gauss_new)/(guide*gauss)):
        return nmat_new,guide_new,gauss_new,1,min_eig_new
    else:
        return N,guide,gauss,0,np.min(np.linalg.eigvals(N))

