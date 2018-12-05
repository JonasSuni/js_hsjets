import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.constants as sc

def divr(r,flux):

    return (2*flux-np.append(0,flux)[:-1]-np.append(flux,0)[1:])/(r**2)

def divr_adv(r,f):

    diff_for = f*r**2 - np.append(f*r**2,0)[1:]
    diff_back = f*r**2 - np.append(0,f*r**2)[:-1]

    diff_for[diff_for < 0] = 0.0
    diff_back[diff_back > 0] = 0.0

    return (diff_for+diff_back)/(r**2)

def grad(f):

    diff_for = f-np.append(f,0)[1:]
    diff_back = f-np.append(0,f)[:-1]

    diff_for[diff_for < 0] = 0.0
    diff_back[diff_back > 0] = 0.0

    return (diff_for+diff_back)

def fit_sim(r,a1,a2):

    return a1*(r**a2)

def sim_n(n0=100.0,v0=0.2,t=10000,dt=1,rmax=10,dr=0.01):

    r = np.arange(1,rmax,dr).astype(float)

    n = (-n0/(max(r)-min(r)))*(r-min(r))+n0
    #n = np.zeros_like(r)
    n[0] = n0
    n[-1] = 0

    dn = -v0*divr_adv(r,n)*dt
    dn[0] = 0

    stop = False

    i = 0

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("R",fontsize=20,labelpad=10)
    ax.set_ylabel("n", fontsize=20,labelpad=10)
    ax.tick_params(labelsize=20)
    plt.grid()
    plt.tight_layout()
    line1, = ax.plot(r,n,"x")

    while not stop:

        n = n + dn

        dn = -v0*divr_adv(r,n)*dt
        dn[0] = 0

        if i%100 == 0:
            line1.set_ydata(n)
            fig.canvas.draw()
            fig.canvas.flush_events()

        i += 1

        if np.linalg.norm(dn) < 0.00001*v0:
            line1.set_ydata(n)
            fig.canvas.draw()
            fig.canvas.flush_events()
            stop = True

        if i>t:
            line1.set_ydata(n)
            fig.canvas.draw()
            fig.canvas.flush_events()
            stop = True

    popt,pcov = scipy.optimize.curve_fit(fit_sim,r,n,p0=[n0,-2])
    y = fit_sim(r,popt[0],popt[1])
    ax.plot(r,y,color="red")
    fig.show()
    print(popt)

    return n
