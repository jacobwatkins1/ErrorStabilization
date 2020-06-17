from numpy import dot,abs,sqrt

def __inprod__(u,v,N):
    return dot(dot(u,N),v)

def overlap(u,v,N):
    dotp = __inprod__(u,v,N[:u.size,:v.size])
    if dotp == 0: return 0
    return abs(dotp)/(sqrt(abs(__inprod__(u,u,N[:u.size,:u.size])*inprod(v,v,N[:v.size,:v.size]))))

