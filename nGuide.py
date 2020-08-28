import numpy as np

def nGuide(n,oldN,deltaN,errsizeN):
    minEig = np.min(np.linalg.eigvals(n))
    sigmoid = 1/(np.exp(-minEig/deltaN)+1)
    gauss = np.exp(-sum(sum((n-oldN)**2))/(2*errsizeN**2))

    return sigmoid*gauss
