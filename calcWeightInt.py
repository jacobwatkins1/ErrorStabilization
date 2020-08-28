import numpy as np
from hGuide import *

def calcWeightInt(hList,H,N,stepH,deltaH,errsizeH,lowestOrderRatio,convergenceRatio):
    hAvg = sum(hList)/len(hList)
    order = H.shape[0]
    cList = []
    for h in hList:
        L = int(1/2*(order+1)*order)

        c = []
        for i in range(order):
            for j in range(i,order):
                for k in range(order):
                    for m in range(k,order):
                        c.append((h[i,j]-hAvg[i,j])*(h[k,m]-hAvg[k,m]))
        c = np.reshape(c,(L,L))
        cList.append(c)

    cAvg = sum(cList)/len(cList)
    mu = np.array([hAvg[i,j] for i in range(order) for j in range(i,order)])
    gaussian = lambda x: np.exp(-1/2*np.dot(x-mu,np.dot(np.linalg.inv(cAvg),x-mu)))/np.sqrt((2*np.pi)**(L)*np.abs(np.linalg.det(cAvg)))

    h = hAvg
    gMax = gaussian(mu)

    hSamples = []
    while len(hSamples) < len(hList):
        eps = stepH*errsizeH*2*(np.random.rand(order,order)-0.5)
        eps = (eps+eps.T)/2*np.sqrt(2)
        hTrial = h + eps

        x = np.array([hTrial[i,j] for i in range(order) for j in range(i,order)])
        g = gaussian(x)
        
        if np.random.rand()*gMax < g:
            hSamples.append(hTrial)
            h = hTrial

    wInt = 0
    for h in hSamples:
        guide,convergence = hGuide(hTrial,H,N,deltaH,errsizeH,lowestOrderRatio,convergenceRatio)
        x = np.array([hTrial[i,j] for i in range(order) for j in range(i,order)])
        g = gaussian(x)

        wInt = wInt + guide/g*1/len(hSamples)

    return wInt,hAvg

