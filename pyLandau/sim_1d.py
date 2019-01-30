import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.constants as sc

gS = sc.G*1.988e+30

def divr(f,r):

    return grad(f*r**2,r)/(r**2)

def grad(f,r):

    dr = np.ediff1d(r)[0]

    res = (-np.pad(f,(0,2),mode="reflect",reflect_type="odd")[2:]+8*np.pad(f,(0,1),mode="reflect",reflect_type="odd")[1:]-8*np.pad(f,(1,0),mode="reflect",reflect_type="odd")[:-1]+np.pad(f,(2,0),mode="reflect",reflect_type="odd")[:-2])/(12*dr)

    return res.astype(float)

def dn(n,v,r):

    return -n*divr(v,r)-v*grad(n,r)
    #return -grad(r**2*n*v,r)/(r**2)

def dv(n,v,T,r):

    return -v*grad(v,r)-grad(2*n*sc.k*T,r)/(sc.m_p*n)-gS/(r**2)

def sim_n_rk4(n0=100.0,v0=1,t=10000,dt=0.1,rmax=10,dr=0.01,figname="unnamed",animate=True,cpf=1000):

    r = np.arange(1,rmax,dr).astype(float)
    r = r*1e+9

    n = np.ones_like(r)*n0
    v = np.ones_like(r)*v0
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

        k1 = dt*dn(n,v,r)
        k2 = dt*dn(n+k1/2,v,r)
        k3 = dt*dn(n+k2/2,v,r)
        k4 = dt*dn(n+k3,v,r)

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

    return n

def rk4_nv(n,v,T,r,dt):

    n0 = n[0]

    kn1 = dt*dn(n,v,r)
    kn1[0] = 0
    kn2 = dt*dn(n+kn1/2,v,r)
    kn2[0] = 0
    kn3 = dt*dn(n+kn2/2,v,r)
    kn3[0] = 0
    kn4 = dt*dn(n+kn3,v,r)

    n = n + (kn1+2*kn2+2*kn3+kn4)/6
    n[0] = n0

    kv1 = dt*dv(n,v,T,r)
    kv2 = dt*dv(n,v+kv1/2,T,r)
    kv3 = dt*dv(n,v+kv2/2,T,r)
    kv4 = dt*dv(n,v+kv3,T,r)

    v = v + (kv1+2*kv2+2*kv3+kv4)/6

    return [n,v]

def ssprk3_nv(n,v,T,r,dt):

    n0 = n[0]

    kn1 = dt*dn(n,v,r)
    kn1[0] = 0
    kn2 = dt*dn(n+kn1,v,r)
    kn2[0] = 0
    kn3 = dt*dn(n+kn1/4+kn2/4,v,r)
    
    n = n + (kn1+kn2+4*kn3)/6
    n[0] = n0

    kv1 = dt*dv(n,v,T,r)
    kv2 = dt*dv(n,v+kv1,T,r)
    kv3 = dt*dv(n,v+kv1/4+kv2/4,T,r)
    
    v = v + (kv1+kv2+4*kv3)/6

    return [n,v]

def rk4_38_nv(n,v,T,r,dt):

    n0 = n[0]

    kn1 = dt*dn(n,v,r)
    kn2 = dt*dn(n+kn1/3,v,r)
    kn3 = dt*dn(n-kn1/3+kn2,v,r)
    kn4 = dt*dn(n+kn1-kn2+kn3,v,r)

    n = n + (kn1+3*kn2+3*kn3+kn4)/8
    n[0] = n0

    kv1 = dt*dv(n,v,T,r)
    kv2 = dt*dv(n,v+kv1/3,T,r)
    kv3 = dt*dv(n,v-kv1/3+kv2,T,r)
    kv4 = dt*dv(n,v+kv1-kv2+kv3,T,r)
    
    v = v + (kv1+3*kv2+3*kv3+kv4)/8

    return [n,v]

def sim_nv_rk4(n0=100,v0=100,T0=1,t=10000,dt=1,rmax=10,dr=0.1,figname="unnamed",animate=True,cpf=1000):

    r = np.arange(1,rmax,dr).astype(float)
    r = r*1e+9

    n = n0*(r/r[0])**-3
    #n = np.ones_like(r)
    v = np.ones_like(r)*v0
    T = np.ones_like(r)*T0
    n[0] = n0

    stop = False
    
    i = 0
    
    plt.ion()
    fig = plt.figure(figsize=(15,10))
    fig.suptitle("$T_0$ = {} K".format(T0),fontsize=20)
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
    ax2.set_ylim(0,1000000)
    ax2.set_yticks([100000,200000,300000,400000,500000,600000,700000,800000,900000])
    if animate:
        line1, = ax1.plot(r,n,"x")
        line2, = ax2.plot(r,v,"x")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust()
    
    for i in xrange(int(t/dt)):
        
        n,v = rk4_38_nv(n,v,T,r,dt)

        if i%cpf == 0 and animate:
            line1.set_ydata(n)
            line2.set_ydata(v)
            fig.canvas.draw()
            fig.canvas.flush_events()
            

    if animate:
        line1.set_ydata(n)
        line2.set_ydata(v)
        ax2.set_ylim(min(v),max(v))
        ax1.set_ylim(0,n0)
        fig.canvas.draw()
        fig.canvas.flush_events()
    else:
        ax1.plot(r,n,"x")
        ax2.plot(r,v,"x")

    plt.tight_layout()
    
    return [n,v]

def sim_nvT_rk4(n0=100,v0=100,T0=0.1,q0=0,t=10000,dt=1,rmax=10,dr=0.1,figname="unnamed",animate=True,cpf=1000):

    r = np.arange(2,rmax,dr).astype(float)
    rr = r*695e+6

    n = np.zeros_like(r)*n0
    v = np.ones_like(r)*v0
    T = np.ones_like(r)*T0
    q = np.ones_like(r)*q0
    
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
    
        # WIP

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