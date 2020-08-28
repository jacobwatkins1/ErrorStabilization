import numpy as np

def hConcavity(h,N,lowestOrderRatio,convergenceRatio):
    order = h.shape[0]
    e = []
    for k in range(1,order+1):
        e.append(np.min(np.linalg.eigvals(np.dot(np.linalg.inv(N[:k,:k]),h[:k,:k]))))

    concavity = np.max([np.abs(e[k-1]-e[k])/np.abs(e[k-2]-e[k-1]) for k in range(lowestOrderRatio,order)])
    return concavity

def hGuide(h,H,N,deltaH,errsizeH,lowestOrderRatio,convergenceRatio):
    concavity = hConcavity(h,N,lowestOrderRatio,convergenceRatio)

    sigmoid = 1/(np.exp((concavity-convergenceRatio)/deltaH)+1)
    gauss = np.exp(-sum(sum((h-H)**2))/(2*errsizeH**2))

    return sigmoid*gauss,concavity

