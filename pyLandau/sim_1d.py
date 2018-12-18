import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.constants as sc

def divr(flux,r):

    #return divr_false(flux,r)
    return divr_true(flux,r)

def grad(f):

    #return grad_false(f)
    return grad_true(f)

def grad_false(f):

    return (2*f-np.append(f[0],f)[:-1]-np.append(f,f[-1])[1:])/(2*695e+6)

def grad_true(f):

    return np.gradient(f,695e+6)

def divr_true(flux,r):

    return np.gradient(flux,695e+6)/(r**2)

def divr_false(flux,r):

    return (2*flux-np.append(flux[0],flux)[:-1]-np.append(flux,flux[-1])[1:])/(2*695e+6*r**2)

def fit_n(r,a1,a2):

    return a1*((r/2)**a2)

def fit_v(r,a1,a2,a3):

    return a1+a2*(np.log(r/2))**a3

def sim_n_rk4(n0=100.0,v0=1,t=10000,dt=0.1,rmax=10,dr=0.01,figname="unnamed",animate=True,cpf=1000):

    r = np.arange(2,rmax,dr).astype(float)
    rt = r*695e+6

    n = np.zeros_like(r)
    n[0]=n0

    stop = False

    i = 0

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("r",fontsize=20,labelpad=10)
    ax.set_ylabel("n", fontsize=20,labelpad=10)
    ax.tick_params(labelsize=15)
    plt.grid()
    if animate:
        line1, = ax.plot(r,n,"x")
    plt.tight_layout()

    while not stop:

        k1 = dt*(-divr(v0*n*rt**2,rt))
        k2 = dt*(-divr(v0*(n+k1/2)*rt**2,rt))
        k3 = dt*(-divr(v0*(n+k2/2)*rt**2,rt))
        k4 = dt*(-divr(v0*(n+k3)*rt**2,rt))

        n = n + (k1+2*k2+2*k3+k4)/6
        n[0] = n0

        if i%cpf == 0 and animate:
            line1.set_ydata(n)
            fig.canvas.draw()
            fig.canvas.flush_events()

        i += 1

        if i>t/dt:
            if animate:
                line1.set_ydata(n)
                fig.canvas.draw()
                fig.canvas.flush_events()
            else:
                ax.plot(r,n,"x")
            stop = True

    popt,pcov = scipy.optimize.curve_fit(fit_n,r,n,p0=[n0,-2])
    y = fit_n(r,popt[0],popt[1])
    ax.plot(r,y,color="red")
    fig.show()
    print(popt)

    fig.savefig("landau/simulations/sim_n_rk4_"+figname+".png")

    f = open("landau/simulations/sim_n_rk4_pars_"+figname+".txt","w")
    f.write(str(popt[0])+" "+str(popt[1]))
    f.close()

    return n

def sim_nv_rk4(n0=100,v0=100,T0=1,t=10000,dt=1,rmax=10,dr=0.1,figname="unnamed",animate=True,cpf=1000):

    r = np.arange(2,rmax,dr).astype(float)

    rt = r*695e+6
    
    T0 = T0*(sc.k/sc.m_p)
    GS = sc.G*1.988e+30

    n = n0*((r/r[0])**-2)
    #n = (-n0/218)*(r-r[0])+n0
    v = np.zeros_like(r)+v0
    v = 90000*(np.log(r/r[0])**0.5)+v0
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
    ax1.tick_params(labelsize=15)
    ax2.tick_params(labelsize=15)
    ax2.set_ylim(0,5*v0)
    if animate:
        line1, = ax1.plot(r,n,"x")
        line2, = ax2.plot(r,v,"x")
    plt.tight_layout()
    
    while not stop:
    
        k1n = dt*(-divr(v*n*rt**2,rt))
        k1v = dt*(-v*grad(v)-T0*divr(n*rt**2,rt)/n-GS/(rt**2))
        
        k2n = dt*(-divr((v+k1v/2)*(n+k1n/2)*rt**2,rt))
        k2v = dt*(-(v+k1v/2)*grad(v+k1v/2)-T0*divr((n+k1n/2)*rt**2,rt)/(n+k1n/2)-GS/(rt**2))
        
        k3n = dt*(-divr((v+k2v/2)*(n+k2n/2)*rt**2,rt))
        k3v = dt*(-(v+k2v/2)*grad(v+k2v/2)-T0*divr((n+k2n/2)*rt**2,rt)/(n+k2n/2)-GS/(rt**2))
        
        k4n = dt*(-divr((v+k3v)*(n+k3n)*rt**2,rt))
        k4v = dt*(-(v+k3v)*grad(v+k3v)-T0*divr((n+k3n)*rt**2,rt)/(n+k3n)-GS/(rt**2))
        
        n = n + (k1n+2*k2n+2*k3n+k4n)/6
        n[0] = n0
        
        v = v + (k1v+2*k2v+2*k3v+k4v)/6
        v[0] = v0
        
        i += 1
        
        if i%cpf == 0 and animate:
            line1.set_ydata(n)
            line2.set_ydata(v)
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        if i>t/dt:
            stop = True
            if animate:
                line1.set_ydata(n)
                line2.set_ydata(v)
                ax2.set_ylim(v0,max(v))
                fig.canvas.draw()
                fig.canvas.flush_events()
            else:
                ax1.plot(r,n,"x")
                ax2.plot(r,v,"x")

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

def sim_nvT_rk4(n0=100,v0=100,T0=0.1,qpar=0.1,qperp=0.1,t=10000,dt=1,rmax=10,dr=0.1,figname="unnamed",animate=True,cpf=1000):

    r = np.arange(2,rmax,dr).astype(float)
    rt = r*695e+6

    T0 = T0*sc.k
    GS = sc.m_p*sc.G*1.988e+30
    
    n = n0*((r/r[0])**-2)
    v = (5.0/3)*T0*(np.log(r/r[0])**0.5)+v0
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
    ax4 = fig.add_subplot(224)
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
    ax1.tick_params(labelsize=15)
    ax2.tick_params(labelsize=15)
    ax3.tick_params(labelsize=15)
    ax4.tick_params(labelsize=15)
    if animate:
        line1, = ax1.plot(r,n,"x")
        line2, = ax2.plot(r,v,"x")
        line3, = ax3.plot(r,Tpar,"x")
        line4, = ax4.plot(r,Tperp,"x")
    plt.tight_layout()
    
    while not stop:
    
        k1n = dt*(-divr(v*n*rt**2,rt))
        k1v = dt*(-v*grad(v)-grad(Tpar*n)/(sc.m_p*n)-GS*n/(rt**2))
        k1T1 = -dt*(divr(v*n*Tpar*rt**2,rt)+2*n*Tpar*grad(v))
        k1T2 = -dt*(divr(v*n*Tperp*rt**2,rt)+n*Tperp*divr(v,rt)-n*Tperp*grad(v))
        
        k2n = dt*(-divr((v+k1v/2)*(n+k1n/2)*rt**2,rt))
        k2v = dt*(-(v+k1v/2)*grad(v+k1v/2)-grad((Tpar+k1T1/2)*(n+k1n/2))/(sc.m_p*(n+k1n/2))-GS*(n+k1n/2)/(rt**2))
        k2T1 = -dt*(divr((v+k1v/2)*(n+k1n/2)*(Tpar+k1T1/2)*rt**2,rt)+2*(n+k1n/2)*(Tpar+k1T1/2)*grad(v+k1v/2))
        k2T2 = -dt*(divr((v+k1v/2)*(n+k1n/2)*(Tperp+k1T2/2)*rt**2,rt)+(n+k1n/2)*(Tperp+k1T2/2)*divr((v+k1v/2),rt)-(n+k1n/2)*(Tperp+k1T2/2)*grad(v+k1v/2))

        k3n = dt*(-divr((v+k2v/2)*(n+k2n/2)*rt**2,rt))
        k3v = dt*(-(v+k2v/2)*grad(v+k2v/2)-grad((Tpar+k2T1/2)*(n+k2n/2))/(sc.m_p*(n+k2n/2))-GS*(n+k2n/2)/(rt**2))
        k3T1 = -dt*(divr((v+k2v/2)*(n+k2n/2)*(Tpar+k2T1/2)*rt**2,rt)+2*(n+k2n/2)*(Tpar+k2T1/2)*grad(v+k2v/2))
        k3T2 = -dt*(divr((v+k2v/2)*(n+k2n/2)*(Tperp+k2T2/2)*rt**2,rt)+(n+k2n/2)*(Tperp+k2T2/2)*divr((v+k2v/2),rt)-(n+k2n/2)*(Tperp+k2T2/2)*grad(v+k2v/2))

        k4n = dt*(-divr((v+k3v)*(n+k3n)*rt**2,rt))
        k4v = dt*(-(v+k3v)*grad(v+k3v)-grad((Tpar+k3T1)*(n+k3n))/(sc.m_p*(n+k3n))-GS*(n+k2n/2)/(rt**2))
        k4T1 = -dt*(divr((v+k3v)*(n+k3n)*(Tpar+k3T1)*rt**2,rt)+2*(n+k3n)*(Tpar+k3T1)*grad(v+k3v))
        k4T2 = -dt*(divr((v+k3v)*(n+k3n)*(Tperp+k3T2)*rt**2,rt)+(n+k3n)*(Tperp+k3T2)*divr((v+k3v),rt)-(n+k3n)*(Tperp+k3T2)*grad(v+k3v))

        n = n + (k1n+2*k2n+2*k3n+k4n)/6
        n[0] = n0
        
        v = v + (k1v+2*k2v+2*k3v+k4v)/6
        v[0] = v0
        
        Tpar = Tpar + (k1T1+2*k2T1+2*k3T1+k4T1)/6
        Tpar[0] = T0

        Tperp = Tperp + (k1T2+2*k2T2+2*k3T2+k4T2)/6
        Tpar[0] = T0

        i += 1
        
        if i%cpf == 0 and animate:
            line1.set_ydata(n)
            line2.set_ydata(v)
            line3.set_ydata(Tpar)
            line2.set_ydata(Tperp)
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        if i>t/dt:
            stop = True
            if animate:
                line1.set_ydata(n)
                line2.set_ydata(v)
                line3.set_ydata(Tpar)
                line4.set_ydata(Tperp)
                ax2.set_ylim(v0,max(v))
                ax3.set_ylim(min(Tpar),max(Tpar))
                ax4.set_ylim(min(Tperp),max(Tperp))
                fig.canvas.draw()
                fig.canvas.flush_events()
            else:
                ax1.plot(r,n,"x")
                ax2.plot(r,v,"x")
                ax3.plot(r,Tpar,"x")
                ax4.plot(r,Tperp,"x")

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