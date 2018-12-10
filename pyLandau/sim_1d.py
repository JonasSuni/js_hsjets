import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.constants as sc

def divr(flux,r):

    return np.gradient(flux)/(r**2)

def fit_n(r,a1,a2):

    return a1*(r**a2)

def fit_v(r,a1,a2,a3):

    return a1+a2*(np.log(r))**a3

def sim_n_rk4(n0=100.0,v0=1,t=10000,dt=0.1,rmax=10,dr=0.01):

    r = np.arange(1,rmax,dr).astype(float)

    n = np.zeros_like(r)
    n[0]=n0

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

        k1 = dt*(-divr(v0*n*r**2,r))
        k2 = dt*(-divr(v0*(n+k1/2)*r**2,r))
        k3 = dt*(-divr(v0*(n+k2/2)*r**2,r))
        k4 = dt*(-divr(v0*(n+k3)*r**2,r))

        n = n + (k1+2*k2+2*k3+k4)/6
        n[0] = n0

        if i%100 == 0:
            line1.set_ydata(n)
            fig.canvas.draw()
            fig.canvas.flush_events()

        i += 1

        if np.linalg.norm((k1+2*k2+2*k3+k4)/6) < 0.001*n0*dt:
            line1.set_ydata(n)
            fig.canvas.draw()
            fig.canvas.flush_events()
            stop = True

        if i>t:
            line1.set_ydata(n)
            fig.canvas.draw()
            fig.canvas.flush_events()
            stop = True

    popt,pcov = scipy.optimize.curve_fit(fit_n,r,n,p0=[n0,-2])
    y = fit_sim(r,popt[0],popt[1])
    ax.plot(r,y,color="red")
    fig.show()
    print(popt)

    return n

def sim_nv_rk4(n0=100,v0=100,T0=1,t=10000,dt=1,rmax=10,dr=0.1):

    r = np.arange(1,rmax,dr).astype(float)
    
    n = n0*(r**-2)
    v = np.zeros_like(r)+v0
    n[0] = n0
    v[0] = v0
    
    stop = False
    
    i = 0
    
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    plt.grid()
    line1, = ax1.plot(r,n,"x")
    line2, = ax2.plot(r,v,"x")
    plt.tight_layout()
    
    while not stop:
    
        k1n = dt*(-divr(v*n*r**2,r))
        k1v = dt*(-v*np.gradient(v)-T0*np.gradient(n)/n)
        
        k2n = dt*(-divr((v+k1v/2)*(n+k1n/2)*r**2,r))
        k2v = dt*(-(v+k1v/2)*np.gradient(v+k1v/2)-T0*np.gradient(n+k1n/2)/(n+k1n/2))
        
        k3n = dt*(-divr((v+k2v/2)*(n+k2n/2)*r**2,r))
        k3v = dt*(-(v+k2v/2)*np.gradient(v+k2v/2)-T0*np.gradient(n+k2n/2)/(n+k2n/2))
        
        k4n = dt*(-divr((v+k3v)*(n+k3n)*r**2,r))
        k4v = dt*(-(v+k3v)*np.gradient(v+k3v)-T0*np.gradient(n+k3n)/(n+k3n))
        
        n = n + (k1n+2*k2n+2*k3n+k4n)/6
        n[0] = n0
        
        v = v + (k1v+2*k2v+2*k3v+k4v)/6
        v[0] = v0
        
        i += 1
        
        if i%100 == 0:
            line1.set_ydata(n)
            line2.set_ydata(v)
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        if np.linalg.norm(k1v+2*k2v+2*k3v+k4v) < 0.001*dt*T0*v0:
            stop = True
            line1.set_ydata(n)
            line2.set_ydata(v)
            ax2.set_ylim(v0,max(v))
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            
        if i>t:
            stop = True
            line1.set_ydata(n)
            line2.set_ydata(v)
            ax2.set_ylim(v0,max(v))
            fig.canvas.draw()
            fig.canvas.flush_events()
    
    plt.tight_layout()

    popt,pcov = scipy.optimize.curve_fit(fit_n,r,n,p0=[n0,-2])
    y = fit_n(r,popt[0],popt[1])
    ax1.plot(r,y,color="red")
    fig.show()
    print(popt)

    poptv,pcovv = scipy.optimize.curve_fit(fit_v,r,v,p0=[v0,0.1,0.5])
    yv = fit_v(r,poptv[0],poptv[1],poptv[2])
    ax2.plot(r,yv,color="red")
    fig.show()
    print(poptv)
    
    return [n,v]