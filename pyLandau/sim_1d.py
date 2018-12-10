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
    ax.set_xlabel("r",fontsize=20,labelpad=10)
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

        if max(np.abs(k1+2*k2+2*k3+k46)/6) < 0.001*n0*dt:
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

def sim_nv_rk4(n0=100,v0=100,T0=1,t=10000,dt=1,rmax=10,dr=0.1,figname="unnamed"):

    r = np.arange(1,rmax,dr).astype(float)
    
    n = n0*(r**-2)
    #v = np.zeros_like(r)+v0
    v = 2*T0*(np.log(r)**0.5)+v0
    n[0] = n0
    v[0] = v0
    
    stop = False
    
    i = 0
    
    plt.ion()
    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.grid(b=True)
    ax2.grid(b=True)
    ax1.set_xlabel("r",fontsize=20,labelpad=10)
    ax2.set_xlabel("r",fontsize=20,labelpad=10)
    ax1.set_ylabel("n",fontsize=20,labelpad=10)
    ax2.set_ylabel("v",fontsize=20,labelpad=10)
    ax1.tick_params(labelsize=10)
    ax2.tick_params(labelsize=10)
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
            
        if max(np.abs(k1v+2*k2v+2*k3v+k4v)/6) < 0.0001*dt*T0*v0:
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

    fig.savefig("landau/simulations/sim_nv_rk4_"+figname+".png")

    f = open("landau/simulations/sim_nv_rk4_pars_"+figname+".txt","w")
    f.write(str(popt[0])+" "+str(popt[1])+"\n"+str(poptv[0])+" "+str(poptv[1])+" "+str(poptv[2]))
    f.close()
    
    return [n,v]

def sim_nvT_rk4(n0=100,v0=100,T0=0.1,qpar=0.1,qperp=0.1,t=10000,dt=1,rmax=10,dr=0.1,figname="unnamed"):

    r = np.arange(1,rmax,dr).astype(float)
    
    n = n0*(r**-2)
    v = (5.0/3)*T0*(np.log(r)**0.5)+v0
    Tpar = np.zeros_like(r)+T0
    Tperp = np.zeros_like(r)+T0
    n[0] = n0
    v[0] = v0
    Tpar[0] = T0
    Tperp[0] = T0
    
    stop = False
    
    i = 0
    
    plt.ion()
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224)
    ax1.grid(b=True)
    ax2.grid(b=True)
    ax3.grid(b=True)
    ax4.grid(b=True)
    ax3.set_xlabel("r",fontsize=20,labelpad=10)
    ax4.set_xlabel("r",fontsize=20,labelpad=10)
    ax1.set_ylabel("n",fontsize=20,labelpad=10)
    ax2.set_ylabel("v",fontsize=20,labelpad=10)
    ax3.set_ylabel("Tpar",fontsize=20,labelpad=10)
    ax4.set_ylabel("Tperp",fontsize=20,labelpad=10)
    ax1.tick_params(labelsize=10)
    ax2.tick_params(labelsize=10)
    ax3.tick_params(labelsize=10)
    ax4.tick_params(labelsize=10)
    line1, = ax1.plot(r,n,"x")
    line2, = ax2.plot(r,v,"x")
    line3, = ax3.plot(r,Tpar,"x")
    line4, = ax4.plot(r,Tperp,"x")
    plt.tight_layout()
    
    while not stop:
    
        k1n = dt*(-divr(v*n*r**2,r))
        k1v = dt*(-v*np.gradient(v)-Tpar*np.gradient(n)/n)
        k1T1 = -dt*(divr(v*n*Tpar*r**2,r)+2*n*Tpar*np.gradient(v))
        k1T2 = -dt*(divr(v*n*Tperp*r**2,r)+n*Tperp*divr(v,r)-n*Tperp*np.gradient(v))
        
        k2n = dt*(-divr((v+k1v/2)*(n+k1n/2)*r**2,r))
        k2v = dt*(-(v+k1v/2)*np.gradient(v+k1v/2)-(Tpar+k1T1/2)*np.gradient(n+k1n/2)/(n+k1n/2))
        k2T1 = 
        k2T2 = 

        k3n = dt*(-divr((v+k2v/2)*(n+k2n/2)*r**2,r))
        k3v = dt*(-(v+k2v/2)*np.gradient(v+k2v/2)-(Tpar+k2T1/2)*np.gradient(n+k2n/2)/(n+k2n/2))
        k3T1 = 
        k3T2 = 

        k4n = dt*(-divr((v+k3v)*(n+k3n)*r**2,r))
        k4v = dt*(-(v+k3v)*np.gradient(v+k3v)-(Tpar+k3T1)*np.gradient(n+k3n)/(n+k3n))
        k4T1 = 
        k4T2 = 

        n = n + (k1n+2*k2n+2*k3n+k4n)/6
        n[0] = n0
        
        v = v + (k1v+2*k2v+2*k3v+k4v)/6
        v[0] = v0
        
        Tpar = Tpar + (k1T1+2*k2T1+2*k3T1+k4T1)/6
        Tpar[0] = T0

        Tperp = Tperp + (k1T2+2*k2T2+2*k3T2+k4T2)/6
        Tpar[0] = T0

        i += 1
        
        if i%100 == 0:
            line1.set_ydata(n)
            line2.set_ydata(v)
            line3.set_ydata(Tpar)
            line2.set_ydata(Tperp)
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        if max(np.abs(k1v+2*k2v+2*k3v+k4v)/6) < 0.01*dt*T0*v0:
            stop = True
            line1.set_ydata(n)
            line2.set_ydata(v)
            line3.set_ydata(Tpar)
            line4.set_ydata(Tperp)
            ax2.set_ylim(v0,max(v))
            ax3.set_ylim(min(Tpar),max(Tpar))
            ax4.set_ylim(min(Tperp),max(Tperp))
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            
        if i>t:
            stop = True
            line1.set_ydata(n)
            line2.set_ydata(v)
            line3.set_ydata(Tpar)
            line4.set_ydata(Tperp)
            ax2.set_ylim(v0,max(v))
            ax3.set_ylim(min(Tpar),max(Tpar))
            ax4.set_ylim(min(Tperp),max(Tperp))
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

    fig.savefig("landau/simulations/sim_nvT_rk4_"+figname+".png")

    f = open("landau/simulations/sim_nvT_rk4_pars_"+figname+".txt","w")
    f.write(str(popt[0])+" "+str(popt[1])+"\n"+str(poptv[0])+" "+str(poptv[1])+" "+str(poptv[2]))
    f.close()
    
    return [n,v]