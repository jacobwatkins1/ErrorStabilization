import numpy as np
from overlap import *
from nGuide import *
from hGuide import *
from calcWeightInt import *
import Nstep
import Hstep

verbose = True

class Pair(object):
    def __init__(self,H,N,wInt):
        self.H = H
        self.N = N
        self.wInt = wInt


def stabilize(N,H,numPairs=10,stepN=0.1,stepH=0.1,deltaN=1e-4,deltaH=1e-4,errsizeN=0.01,errsizeH=0.01,lowestOrderRatio=2,convergenceRatio=0.5,hSamples=10000,autotune=False):
    n = N.copy()
    h = H.copy()

    order = H.shape[0] 

    guideN = nGuide(n,N,deltaN,errsizeN)

    if verbose: print('Stabilization running',flush=True)

    #Calculate a list of N matrices
    nList = []
    while len(nList) < numPairs:
        n,guideN,accepted = Nstep.__Nstep__(n,N,guideN,stepN,deltaN,errsizeN)
        if accepted:
            minEig = np.min(np.linalg.eigvals(n))
            if minEig > 0:
                nList.append(n)
                if verbose: print('N matrix accepted. Total N matrices: ',len(nList),flush=True)
    
    #For each N matrix, find an acceptable H matrix and the integral of the sampling function
    pairs = []
    for n in nList:
        guideH,convergence = hGuide(h,H,n,deltaH,errsizeH,lowestOrderRatio,convergenceRatio)
        
        if autotune:
            if verbose: print('beginning autotune',flush=True)
            bestDelta = deltaH
            bestStep = stepH
            bestFittness = 99999
            for d in np.logspace(-4,0,10):
                for s in np.logspace(-3,0,10):
                    convList = []
                    for i in range(1000):
                        h,guideH,accepted,convergence = Hstep.__Hstep__(h,H,n,guideH,s,d,errsizeH,lowestOrderRatio,convergenceRatio)
                        convList.append(convergence)
                    mean = np.mean(convList)
                    std = np.std(convList)
                    if mean+std < bestFittness:
                        bestDelta = d
                        bestStep = s
                        bestFittness = mean+std
            coolSchedule = np.linspace(bestDelta,deltaH,1000)
            stepH = bestStep
            if verbose: print('deltaH: ',bestDelta,'\nstepH: ',bestStep,flush=True)
        else:
            coolSchedule = np.ones(100)*deltaH

        
        hList = []
        convList = []
        trials = 0
        coolCycles = 0
        addNorm = True
        while len(hList) < hSamples:
            trials+=1
            deltaH = coolSchedule[0]
            h,guideH,accepted,convergence = Hstep.__Hstep__(h,H,n,guideH,stepH,deltaH,errsizeH,lowestOrderRatio,convergenceRatio)
            #print(convergence,flush=True)
            
            convList = np.append(convList,convergence)
            if len(convList) > 1000:
                convList = np.delete(convList,0)
            convMean = np.mean(convList)
            convStd = np.std(convList)
            #print(convMean,'\t',convStd,'\t',convStd/convMean,'\t',len(coolSchedule),flush=True)
            if convStd/convMean < trials*1e-6 and accepted == False:
                coolSchedule = np.linspace(deltaH*100,deltaH,1000) #recool
                trials = 0
                coolCycles += 1
                if verbose: print('resetting cooling',flush=True)
            if accepted and len(coolSchedule) > 1:
                coolSchedule = np.delete(coolSchedule,0)
            #if accepted and len(coolSchedule)<2 and np.abs(convMean-np.mean(convList[-100:])) < convStd/200 and convergence < np.mean(convList[-100:]):
            #    convergence = 0
            if coolCycles > 20 and len(hList) < hSamples/10:
                #This matrix isn't working, abort.
                if verbose: print('Insufficient valid H matrices. Proceeding to next matrix')
                #nList.remove(n)
                addNorm = False
                break
            if accepted and convergence < convergenceRatio and len(coolSchedule)<2:
                hList.append(h)
                print('H matrix accepted. Total H matrices: ',len(hList),flush=True)

        if addNorm: wInt,h = calcWeightInt(hList,H,n,stepH,deltaH,errsizeH,lowestOrderRatio,convergenceRatio)
        if addNorm: pairs.append(Pair(h,n,wInt))
    wTot = 0
    for p in pairs:
        wTot += p.wInt
    e = np.zeros(order)
    vSmall = np.zeros((order,order))
    for p in pairs:
        vSmallTemp = np.zeros((order,order))
        eTemp = np.zeros(order)
        for k in range(1,order+1):
            dTemp,vTemp = np.linalg.eig(np.dot(np.linalg.inv(p.N[:k,:k]),p.H[:k,:k]))
            E = min(dTemp)
            eTemp[k-1]=E
            vSmallTemp[:k,k-1] = vTemp[:k,np.argmin(dTemp)]
            print(np.min(np.linalg.eigvals(p.N)))
        e += eTemp*p.wInt/wTot
        vSmall += vSmallTemp*p.wInt/wTot
    return e,vSmall


                





        

