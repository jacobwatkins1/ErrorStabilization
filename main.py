import numpy as np
from overlap import *
from nGuide import *
from hGuide import *
from calcWeightInt import *
import Nstep
import Hstep
from scipy.linalg import eigh
import multiprocessing as mp
import os
import time
class Pair(object):
    def __init__(self,H,N,wInt):
        self.H = H
        self.N = N
        self.wInt = wInt
def stabH(args):
    n,h,H,errsizeH,stepH,deltaH,hSamples,lowestOrderRatio,convergenceRatio,autotune,verbose = args


    stepH0 = stepH
    H = H.copy()
    h = h.copy()
    if verbose: print('process id: ',os.getpid(),flush=True)
    guideH,convergence = hGuide(h,H,n,deltaH,errsizeH,lowestOrderRatio,convergenceRatio)

    coolSchedule = np.ones(100)*deltaH

    if autotune:
        stepH = 2*stepH0*1/(np.log(np.linalg.cond(H)))

    hList = []
    convList = []
    trials = 0
    coolCycles = 0
    addNorm = True
    while len(hList) < hSamples:
        trials+=1
        deltaH = coolSchedule[0]
        h,guideH,accepted,convergence = Hstep.__Hstep__(h,H,n,guideH,stepH,deltaH,errsizeH,lowestOrderRatio,convergenceRatio)
        convList = np.append(convList,convergence)
        if len(convList) > 1000:
            convList = np.delete(convList,0)
        convMean = np.mean(convList)
        convStd = np.std(convList)
        #print(convMean,'\t',convStd,'\t',convStd/convMean,'\t',len(coolSchedule),flush=True)
        if convStd/convMean < trials*1e-6 and accepted == False:
            coolSchedule = np.linspace(deltaH*200,deltaH,2000) #recool
            trials = 0
            coolCycles += 1
            if verbose: print('resetting cooling',flush=True)
        if accepted and len(coolSchedule) > 1:
            coolSchedule = np.delete(coolSchedule,0)
        #if accepted and len(coolSchedule)<2 and np.abs(convMean-np.mean(convList[-100:])) < convStd/200 and convergence < np.mean(convList[-100:]):
        #    convergence = 0
        if coolCycles > 30:
            #This matrix isn't working, abort.
            if verbose: print('Insufficient valid H matrices. Proceeding to next matrix')
            #nList.remove(n)
            if len(hList) <= hSamples/2:
                addNorm = False
            else:
                if verbose: print('Keeping '+str(len(hList))+'/'+str(hSamples)+' matrices.')
            break
        if accepted and convergence < convergenceRatio and len(coolSchedule)<2:
            hList.append(h)
            if verbose and len(hList)%int(hSamples/10) == 0:
                print('H matrix accepted. Total H matrices: ',len(hList),flush=True)
    if addNorm: wInt,h = calcWeightInt(hList,H,n,stepH,deltaH,errsizeH,lowestOrderRatio,convergenceRatio)
    if addNorm: return Pair(h,n,wInt)

def stabN(args):
    n,N,deltaN,errsizeN,stepN,numPairs,verbose = args
    n = n.copy()
    np.random.seed(int(os.getpid()))
     #Calculate a list of N matrices
    guideN = nGuide(n,N,deltaN,errsizeN)
    nList = []
    numAccepted = 0
    while len(nList) < numPairs:
        n,guideN,accepted = Nstep.__Nstep__(n,N,guideN,stepN,deltaN,errsizeN)
        if accepted:
            minEig = np.min(np.linalg.eigvals(n))
            if minEig > 0:
                numAccepted += 1
                if numAccepted %100 == 0:
                    nList.append(n)
                    if verbose: print('N matrix accepted. Total N matrices: ',len(nList),flush=True)
    return nList


def stabilize(N,H,numPairs=200,stepN=0.1,stepH=0.1,deltaN=1e-4,deltaH=1e-4,errsizeN=0.01,errsizeH=0.01,lowestOrderRatio=2,convergenceRatio=0.8,hSamples=1000,autotune=False,verbose=False):
    n = N.copy()
    h = H.copy()

    order = H.shape[0] 


    if verbose: print('Stabilization running',flush=True)




#    #Calculate a list of N matrices
#    nList = []
#    while len(nList) < numPairs:
#        n,guideN,accepted = Nstep.__Nstep__(n,N,guideN,stepN,deltaN,errsizeN)
#        if accepted:
#            minEig = np.min(np.linalg.eigvals(n))
#            if minEig > 0:
#                nList.append(n)
#                if verbose: print('N matrix accepted. Total N matrices: ',len(nList),flush=True)
#    
    

    cores = mp.cpu_count()-2
    if verbose: print('Running on '+str(cores)+' cores.')
    
    pairsPerCore = numPairs//cores
    argsList = []
    for i in range(cores):
        argsList.append((n,N,deltaN,errsizeN,stepN,pairsPerCore,verbose))
        

    p = mp.Pool(cores)
    nRawList = (p.map(stabN,argsList))
    p.close()
    p.join()

    nList = []
    for i in nRawList:
        nList.extend(i)

    #For each N matrix, find an acceptable H matrix and the integral of the sampling function
    argsList = []
    for n in nList:
        argsList.append((n,h,H,errsizeH,stepH,deltaH,hSamples,lowestOrderRatio,convergenceRatio,autotune,verbose))




    #p = mp.Pool(mp.cpu_count())
    p = mp.Pool(mp.cpu_count()-2)
    pairs = (p.map(stabH,argsList))
    p.close()
    p.join()
    pairs = [p for p in pairs if p is not None]
    wTot = 0
    if len(pairs) == 0:
        print("Warning: No valid N matrices found. Reverting to original matrices")
        pairs.append(Pair(H,N,1))
    for p in pairs:
        wTot += p.wInt
    e = np.zeros(order)
    vSmall = np.zeros((order,order))
    for p in pairs:
        vSmallTemp = np.zeros((order,order))
        eTemp = np.zeros(order)
        for k in range(1,order+1):
            #dTemp,vTemp = np.linalg.eig(np.dot(np.linalg.inv(p.N[:k,:k]),p.H[:k,:k]))
            dTemp,vTemp = eigh(p.H[:k,:k],p.N[:k,:k])
            E = np.min(dTemp)
            eTemp[k-1]=E
            vSmallTemp[:k,k-1] = vTemp[:k,np.argmin(dTemp)]
        e += eTemp*p.wInt/wTot
        vSmall += vSmallTemp*p.wInt/wTot
    return e,vSmall
                





        

