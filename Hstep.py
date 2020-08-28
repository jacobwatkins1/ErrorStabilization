import numpy as np
from hGuide import *

#def __Hstep__(H,N,hmat_start,err,s,guide,gauss,hmat_delta,lowest_order_ratio,convergence_ratio,hconvFloorRatio):
#    order = np.max(H.shape)

#    hmat_err = s*err*2*(np.random.rand(order,order)-0.5)
#    hmat_err = (hmat_err+hmat_err.T)/2*np.sqrt(2)

#    hmat_new = H + hmat_err

#    e = []
#    for k in range(1,order+1):
#        e.append(np.min(np.linalg.eigvals(np.dot(np.linalg.inv(N[:k,:k]),hmat_new[:k,:k]))))
    
#    concavity = np.max([np.abs(e[k-1]-e[k])/np.max([np.abs(e[k-2]-e[k-1]),np.abs(hconvFloorRatio*e[k-2])]) for k in range(lowest_order_ratio,order)])
#    guide_new = 1/(np.exp((concavity-convergence_ratio)/hmat_delta)+1)
#    gauss_new = np.exp(-sum(sum((hmat_new-hmat_start)**2))/(2*err**2))
    
#    if (np.random.rand() < (guide_new*gauss_new)/(guide*gauss)):
#        return hmat_new,guide_new,gauss_new,1,e,concavity
#    else:
#        return H,guide,gauss,0,e,concavity

def __Hstep__(h,H,N,guide,stepH,deltaH,errsizeH,lowestOrderRatio,convergenceRatio):
    order = H.shape[0]

    eps = stepH*errsizeH*2*(np.random.rand(order,order)-0.5)
    eps = (eps+eps.T)/2*np.sqrt(2)

    hTrial = h + eps

    guideNew,convergence = hGuide(hTrial,H,N,deltaH,errsizeH,lowestOrderRatio,convergenceRatio)

    if np.random.rand() < guideNew/guide:
        return hTrial,guideNew,True,convergence
    else:
        return h,guide,False,convergence

