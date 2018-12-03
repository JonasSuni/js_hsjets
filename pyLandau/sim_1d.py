import numpy as np
import matplotlib.pyplot as plt

def divr(r,f):

    return (r**-2)*np.gradient((r**2)*f)

def sim_n(n0=100.0,v0=500.0,t=10000,dt=1):

    r = np.arange(1,216).astype(float)

    n = (-n0/(max(r)-min(r)))*r+n0

    dn = -v0*divr(r,n)*dt
    dn[np.abs(dn)>1e+3] = np.sign(dn[np.abs(dn)>1e+3])*1e+3

    stop = False

    i = 0

    while not stop:

        old_n = n

        n = np.concatenate((np.array([n[0]]),n[1:-1] + dn[1:-1],np.array([n[-1]])))
        #n += dn

        #n[n<0] = 0

        dn = -v0*divr(r,n)*dt
        dn[np.abs(dn)>1e+3] = np.sign(dn[np.abs(dn)>1e+3])*1e+3

        i += 1

        if i>t:
            stop = True

    return n