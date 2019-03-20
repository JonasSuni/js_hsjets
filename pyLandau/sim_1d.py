import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.constants as sc
import scipy.signal as ssi
import scipy.special

gS = sc.G*1.988e+30
Rs = 695.0e+6
f_max = 9.0
a_s = 5.0e+10
R1 = 1.29*Rs

def divr(f,r):

    return grad(f*r**2,r)/(r**2)

def get_a(r):

    f1 = 1 - (f_max - 1)*np.exp((Rs-R1)/R1)
    f = (f_max*np.exp((r-R1)/R1)+f1)/(np.exp((r-R1)/R1)+1)
    a = a_s*f*(r/Rs)**2

    return a

def divb(r):

    a = get_a(r)

    return grad(a,r)/a

def grad(f,r):

    dr = np.ediff1d(r)[0]

    res = (-np.pad(f,(0,2),mode="reflect",reflect_type="odd")[2:]+8*np.pad(f,(0,1),mode="reflect",reflect_type="odd")[1:]-8*np.pad(f,(1,0),mode="reflect",reflect_type="odd")[:-1]+np.pad(f,(2,0),mode="reflect",reflect_type="odd")[:-2])/(12*dr)

    return res.astype(float)

def dn_a(n,v,a,r):

    return -n*grad(a*v,r)/a-v*grad(n,r)

def dv_a(n,v,T,a,r):

    return -v*grad(v,r)-sc.k*grad(2*n*T,r)/(n*sc.m_p)-sc.k*T*grad(a,r)/(a*sc.m_p)-gS/(r**2)

def dB2Tpn2_a(n,v,T,q,B,a,R):
    # Returns d/dt(B^2*T/n^2)!

    return -v*grad(B**2*T/n**2)-2*B**2*grad(a*q)/(n**3*sc.k*a)

def rk2_nva(n,v,T,a,r,dt):

    n0 = n[0]
    kn1 = dt*dn_a(n,v,a,r)
    d_n = dt*dn_a(n+kn1/2,v,a,r)

    kv1 = dt*dv_a(n,v,T,a,r)
    d_v = dt*dv_a(n,v+kv1/2,T,a,r)

    n += d_n
    v += d_v

    n[0] = n0
    n[-1] = 2*n[-2]-n[-3]

    v[-1] = 2*v[-2]-v[-3]
    v[0] = 2*v[1]-v[2]

    return [n,v]

def dn(n,v,r):

    return -n*divr(v,r)-v*grad(n,r)

def dv(n,v,T,r):

    return -v*grad(v,r)-grad(2*n*sc.k*T,r)/(sc.m_p*n)-gS/(r**2)

def dp(v,p,q,r):

    return -(p*divr(v,r)+v*grad(p,r))-grad(q,r)-2*p*grad(v,r)

def dq(n,v,p,q,r,k):

    T = p/(2*n*sc.k)

    mom4 = sc.m_p*rk_mom(k=k,alpha=0.000001,T=T,l=4)/rk_mom(k=k,alpha=0.000001,T=T,l=0)

    return -q*divr(v,r)-v*grad(q,r)-grad(mom4,r)-3*q*grad(v,r)+3*p*grad(p,r)/(n*sc.m_p)+3*p**2*divr(1,r)/(n*sc.m_p)

def rk_mom(k,alpha,T,l):

    theta = np.sqrt(2*sc.k*T/sc.m_p/3) # Division by 3 needed to get correct result
    v = l + 2
    a = (v-1)/2.0
    b0,b1,b2,b3,b4,b5 = (2*k-v+2*np.array([0,1,2,3,4,5])-5)/2.0

    C1 = alpha**2*(2*alpha**2*k+v-1)*(2-k)*k**((v+3)/2.0)
    C2 = (4*alpha**4*k**2+((4*v-4)*alpha**2-2*v+2)*k+v**2-1)*k**((v+1)/2.0)
    C3 = k**(k+1)*alpha**(1+2*k-v)*(alpha**2+1)
    C4 = (v-3)*k**(k+1)*alpha**(3+2*k-v)

    print(scipy.special.gamma(b1))
    print(scipy.special.hyp1f1(a,-b0,alpha**2*k))
    print(scipy.special.gamma(b2))
    print(scipy.special.hyp1f1(a,-b1,alpha**2*k))
    print(scipy.special.gamma(-b3))
    print(scipy.special.hyp1f1(k,b4,alpha**2*k))
    print(scipy.special.gamma(-b4))
    print(scipy.special.hyp1f1(k,b5,alpha**2*k))

    res = 4*np.pi*theta**(1+v)*( \
        - C1 * scipy.special.gamma(a) * scipy.special.gamma(b1) * scipy.special.hyp1f1(a,-b0,alpha**2*k)/4/scipy.special.gamma(k + 1) \
        - C2 * scipy.special.gamma(a) * scipy.special.gamma(b2) * scipy.special.hyp1f1(a,-b1,alpha**2*k)/8/scipy.special.gamma(k + 1) \
        + C3 * scipy.special.gamma(-b3) * scipy.special.hyp1f1(k,b4,alpha**2*k)/2 \
        - C4 * scipy.special.gamma(-b4) * scipy.special.hyp1f1(k,b5,alpha**2*k)/4)

    return res

def sim_n_rk4(n0=100.0,v0=1,t=10000,dt=0.1,rmax=10,dr=0.01,figname="unnamed",animate=True,cpf=1000):

    r = np.arange(1,rmax,dr).astype(float)
    r = r*1e+9

    n = np.ones_like(r)*n0/2.0
    v = np.ones_like(r)*v0
    a = get_a(r)
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
    ax.set_ylim(0,n0)
    if animate:
        line1, = ax.plot(r,n,"x")
    plt.tight_layout()

    while not stop:

        k1 = dt*dn_a(n,v,a,r)
        k2 = dt*dn_a(n+k1/2,v,a,r)
        k3 = dt*dn_a(n+k2/2,v,a,r)
        k4 = dt*dn_a(n+k3,v,a,r)

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

def sim_nva(T0=4000000,t=10000,dt=1,rmax=10,dr=0.1,figname="unnamed",animate=True,cpf=1000):

    r = np.arange(1,rmax,dr).astype(float)
    r *= Rs
    
    Bs = 11.8e-4
    ns = 1.0e14
    Ts = 7.0e+5

    a = get_a(r)

    B = Bs*a_s/a
    v = 655000*(1+20*(r[0]/r)**3)**-1
    n = ns*v[0]*a_s/(v*a)
    #T = Ts*(3-2*r[0]/r)*(r/r[0])**(-2.0/7)
    T = np.ones_like(r)*T0
    
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
        
        n,v = rk2_nva(n,v,T,a,r,dt)

        if i%cpf == 0 and animate:
            line1.set_ydata(n)
            line2.set_ydata(v)
            fig.canvas.draw()
            fig.canvas.flush_events()
            

    if animate:
        line1.set_ydata(n)
        line2.set_ydata(v)
        ax2.set_ylim(0,max(v))
        fig.canvas.draw()
        fig.canvas.flush_events()
    else:
        ax1.plot(r,n,"x")
        ax2.plot(r,v,"x")

    
    return [n,v]

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
    kn2 = dt*dn(n+kn1,v,r)
    kn3 = dt*dn(n+kn1/4+kn2/4,v,r)
    
    n = n + (kn1+kn2+4*kn3)/6
    n[0] = n0

    kv1 = dt*dv(n,v,T,r)
    kv2 = dt*dv(n,v+kv1,T,r)
    kv3 = dt*dv(n,v+kv1/4+kv2/4,T,r)
    
    v = v + (kv1+kv2+4*kv3)/6

    return [n,v]

def rk2_nv(n,v,T,r,dt,clips=None,ret_deltas=False):

    n0 = n[0]
    kn1 = dt*dn(n,v,r)
    d_n = dt*dn(n+kn1/2,v,r)


    kv1 = dt*dv(n,v,T,r)
    d_v = dt*dv(n,v+kv1/2,T,r)
    
    if ret_deltas:
        return [d_n,d_v]

    d_n = np.clip(d_n,-clips[0],clips[0])
    d_v = np.clip(d_v,-clips[1],clips[1])

    n = n + d_n
    n[0] = n0
    v = v + d_v

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

    n = n0*(r/r[0])**-2
    #n = np.ones_like(r)
    #v = np.ones_like(r)*v0
    v = np.log(r/r[0])**0.5
    T = np.ones_like(r)*T0
    n[0] = n0


    stop = False
    
    i = 0

    d_n,d_v = rk2_nv(n,v,T,r,dt,ret_deltas=True)
    std_dn = np.std(d_n,ddof=1)
    mean_dn = np.mean(d_n)
    std_dv = np.std(d_v,ddof=1)
    mean_dv = np.mean(d_v)
    clips = [mean_dn+2*std_dn,mean_dv+2*std_dv]
    
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
        
        n,v = rk2_nv(n,v,T,r,dt,clips=clips)

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

    
    return [n,v]

def rk2_nvT(n,v,T,q,r,dt,clips=None,ret_deltas=False):
    
    p = 2*n*sc.k*T
    p0 = p[0]
    kp1 = dt*dp(v,p,q,r)
    d_p = dt*dp(v,p+kp1/2,q,r)

    n0 = n[0]
    kn1 = dt*dn(n,v,r)
    d_n = dt*dn(n+kn1/2,v,r)

    kv1 = dt*dv(n,v,T,r)
    d_v = dt*dv(n,v+kv1/2,T,r)
    
    if ret_deltas:
        return [d_n,d_v,d_p]

    d_p = np.clip(d_p,-clips[2],clips[2])
    d_v = np.clip(d_v,-clips[1],clips[1])
    d_n = np.clip(d_n,-clips[0],clips[0])

    p = p + d_p
    p[0] = p0

    v = v + d_v

    n = n + d_n
    n[0] = n0
    
    T = p/(2*n*sc.k)
    
    return [n,v,T]

def rk1_nvT(n,v,T,q,r,dt,clips=None,ret_deltas=False):
    
    p = 2*n*sc.k*T
    p0 = p[0]
    d_p = dt*dp(v,p,q,r)

    n0 = n[0]
    d_n = dt*dn(n,v,r)

    d_v = dt*dv(n,v,T,r)
    
    if ret_deltas:
        return [d_n,d_v,d_p]

    d_p = np.clip(d_p,-clips[2],clips[2])
    d_v = np.clip(d_v,-clips[1],clips[1])
    d_n = np.clip(d_n,-clips[0],clips[0])

    p = p + d_p
    p[0] = p0

    v = v + d_v

    n = n + d_n
    n[0] = n0
    T = p/(2*n*sc.k)
    
    return [n,v,T]


def sim_nvT_rk4(n0=100,v0=100,T0=0.1,q0=0,t=10000,dt=1,rmax=10,dr=0.1,figname="unnamed",animate=True,cpf=1000):

    r = np.arange(1,rmax,dr).astype(float)
    r = r*1e+9

    n = n0*(r/r[0])**-2
    #v = np.ones_like(r)*v0
    v = v0*np.log(r/r[0])**0.5
    #T = np.ones_like(r)*T0
    T = T0*(r/r[0])**-2
    q = np.ones_like(r)*q0
    
    stop = False
    
    i = 0

    d_n,d_v,d_p = rk2_nvT(n,v,T,q,r,dt,ret_deltas=True)
    std_dn = np.std(d_n,ddof=1)
    mean_dn = np.mean(d_n)
    std_dv = np.std(d_v,ddof=1)
    mean_dv = np.mean(d_v)
    std_dp = np.std(d_p,ddof=1)
    mean_dp = np.mean(d_p)
    clips = [mean_dn+1*std_dn,mean_dv+1*std_dv,mean_dp+1*std_dp]
    
    plt.ion()
    fig = plt.figure(figsize=(15,15))
    fig.suptitle("$T_0$ = {} K".format(T0),fontsize=20)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax1.grid(b=True)
    ax2.grid(b=True)
    ax3.grid(b=True)
    ax2.set_xlabel("r",fontsize=20,labelpad=10)
    ax3.set_xlabel("r",fontsize=20,labelpad=10)
    ax1.set_ylabel("n",fontsize=20,labelpad=10)
    ax2.set_ylabel("v",fontsize=20,labelpad=10)
    ax3.set_ylabel("T",fontsize=20,labelpad=10)
    ax1.tick_params(labelsize=15)
    ax2.tick_params(labelsize=15)
    ax3.tick_params(labelsize=15)
    ax2.set_ylim(0,1000000)
    ax3.set_ylim(0,max(T))
    if animate:
        line1, = ax1.plot(r,n,"x")
        line2, = ax2.plot(r,v,"x")
        line3, = ax3.plot(r,T,"x")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust()
    
    for i in xrange(int(t/dt)):    
        
        n,v,T = rk2_nvT(n,v,T,q,r,dt,clips=clips)
        
        if i%cpf == 0 and animate:
            line1.set_ydata(n)
            line2.set_ydata(v)
            line3.set_ydata(T)
            fig.canvas.draw()
            fig.canvas.flush_events()

        if np.isnan(v).any():
            raise ValueError("Simulation destabilised!")
            

    if animate:
        line1.set_ydata(n)
        line2.set_ydata(v)
        line3.set_ydata(T)
        ax2.set_ylim(min(v),max(v))
        ax3.set_ylim(min(T),max(T))
        fig.canvas.draw()
        fig.canvas.flush_events()
    else:
        ax1.plot(r,n,"x")
        ax2.plot(r,v,"x")
        ax3.plot(r,T,"x")

    fig.savefig("landau/simulations/sim_nvT_{}.png".format(figname))
    np.savetxt("landau/simulations/sim_nvT_n_{}.txt".format(figname),n)
    np.savetxt("landau/simulations/sim_nvT_v_{}.txt".format(figname),v)
    np.savetxt("landau/simulations/sim_nvT_T_{}.txt".format(figname),T)

    return [n,v,T]

def rk2_nvTq(n,v,T,q,r,dt,clips=None,ret_deltas=False):
    
    p = 2*n*sc.k*T
    p0 = p[0]
    kp1 = dt*dp(v,p,q,r)
    d_p = dt*dp(v,p+kp1/2,q,r)

    n0 = n[0]
    kn1 = dt*dn(n,v,r)
    d_n = dt*dn(n+kn1/2,v,r)

    kv1 = dt*dv(n,v,T,r)
    d_v = dt*dv(n,v+kv1/2,T,r)

    kq1 = dt*dq(n,v,p,q,r,k=2)
    d_q = dt*dq(n,v,p,q+kq1/2,r,k=2)
    
    if ret_deltas:
        return [d_n,d_v,d_p,d_q]

    d_p = np.clip(d_p,-clips[2],clips[2])
    d_v = np.clip(d_v,-clips[1],clips[1])
    d_n = np.clip(d_n,-clips[0],clips[0])
    d_q = np.clip(d_q,-clips[3],clips[3])

    p = p + d_p
    p[0] = p0

    v = v + d_v

    n = n + d_n
    n[0] = n0

    q = q + d_q
    
    T = p/(2*n*sc.k)
    
    return [n,v,T,q]

def sim_nvTq_rk4(n0=100,v0=100,T0=0.1,q0=0,t=10000,dt=1,rmax=10,dr=0.1,figname="unnamed",animate=True,cpf=1000):

    r = np.arange(1,rmax,dr).astype(float)
    r = r*1e+9

    n = n0*(r/r[0])**-2
    #v = np.ones_like(r)*v0
    v = v0*np.log(r/r[0])**0.5+v0/10.0
    #T = np.ones_like(r)*T0
    T = T0*(r/r[0])**-4
    q = np.ones_like(r)*q0
    
    stop = False
    
    i = 0

    d_n,d_v,d_p,d_q = rk2_nvTq(n,v,T,q,r,dt,ret_deltas=True)
    std_dn = np.nanstd(d_n,ddof=1)
    mean_dn = np.nanmean(d_n)
    std_dv = np.nanstd(d_v,ddof=1)
    mean_dv = np.nanmean(d_v)
    std_dp = np.nanstd(d_p,ddof=1)
    mean_dp = np.nanmean(d_p)
    std_dq = np.nanstd(d_q,ddof=1)
    mean_dq = np.nanmean(d_q)
    clips = [mean_dn+1*std_dn,mean_dv+1*std_dv,mean_dp+1*std_dp,mean_dq+1*std_dq]
    
    plt.ion()
    fig = plt.figure(figsize=(15,15))
    fig.suptitle("$T_0$ = {} K".format(T0),fontsize=20)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.grid(b=True)
    ax2.grid(b=True)
    ax3.grid(b=True)
    ax4.grid(b=True)
    ax2.set_xlabel("r",fontsize=20,labelpad=10)
    ax4.set_xlabel("r",fontsize=20,labelpad=10)
    ax1.set_ylabel("n",fontsize=20,labelpad=10)
    ax2.set_ylabel("v",fontsize=20,labelpad=10)
    ax3.set_ylabel("T",fontsize=20,labelpad=10)
    ax4.set_ylabel("q",fontsize=20,labelpad=10)
    ax1.tick_params(labelsize=15)
    ax2.tick_params(labelsize=15)
    ax3.tick_params(labelsize=15)
    ax4.tick_params(labelsize=15)
    ax2.set_ylim(0,1000000)
    ax3.set_ylim(0,max(T))
    ax4.set_ylim(0,0.1)
    if animate:
        line1, = ax1.plot(r,n,"x")
        line2, = ax2.plot(r,v,"x")
        line3, = ax3.plot(r,T,"x")
        line4, = ax4.plot(r,q,"x")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust()
    
    for i in xrange(int(t/dt)):    
        
        n,v,T,q = rk2_nvTq(n,v,T,q,r,dt,clips=clips)
        
        if i%cpf == 0 and animate:
            line1.set_ydata(n)
            line2.set_ydata(v)
            line3.set_ydata(T)
            line4.set_ydata(q)
            fig.canvas.draw()
            fig.canvas.flush_events()

        if np.isnan(v).any():
            raise ValueError("Simulation destabilised!")    

    if animate:
        line1.set_ydata(n)
        line2.set_ydata(v)
        line3.set_ydata(T)
        line4.set_ydata(q)
        ax2.set_ylim(min(v),max(v))
        ax3.set_ylim(min(T),max(T))
        ax4.set_ylim(min(q),max(q))
        fig.canvas.draw()
        fig.canvas.flush_events()
    else:
        ax1.plot(r,n,"x")
        ax2.plot(r,v,"x")
        ax3.plot(r,T,"x")
        ax4.plot(r,q,"x")

    fig.savefig("landau/simulations/sim_nvTq_{}.png".format(figname))
    np.savetxt("landau/simulations/sim_nvTq_n_{}.txt".format(figname),n)
    np.savetxt("landau/simulations/sim_nvTq_v_{}.txt".format(figname),v)
    np.savetxt("landau/simulations/sim_nvTq_T_{}.txt".format(figname),T)
    np.savetxt("landau/simulations/sim_nvTq_q_{}.txt".format(figname),q)

    return [n,v,T,q]