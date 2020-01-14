import matplotlib as mpl
import jet_aux as jx
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[jx.violet, jx.medium_blue, jx.dark_blue, jx.orange])

import pytools as pt
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import plot_contours as pc
import jet_analyser as ja
import jet_io as jio

wrkdir_DNR = os.environ["WRK"]+"/"
homedir = os.environ["HOME"]+"/"
try:
    vlasdir = os.environ["VLAS"]
except:
    vlasdir="/proj/vlasov"

def get_transient_xseries(runid,jetid,transient="jet"):

    if type(jetid) is not str:
        jetid = str(jetid).zfill(5)

    try:
        props = jio.PropReader(jetid,runid,transient=transient)
    except:
        return 1

    time_arr = props.read("time")
    x_arr = props.read("x_mean")
    y_arr = props.read("y_mean")
    pd_arr = props.read("pd_avg")

    bs_dist_arr = jx.bs_dist_markus(runid,time_arr,x_arr,y_arr)

    pd_arr = pd_arr[np.argsort(bs_dist_arr)]
    bs_dist_arr.sort()

    return (bs_dist_arr,pd_arr)

def jh2020_SEA(runid,transient="slamsjet"):

    pd_sw = jx.sw_par_dict(runid)[3]/1.0e-9

    epoch_arr = np.arange(-2.0,2.005,0.01)
    SEA_arr = np.zeros_like(epoch_arr)
    SEA_mean = np.zeros_like(epoch_arr)
    SEA_std = np.zeros_like(epoch_arr)

    for n in range(3000):
        jetid = str(n).zfill(5)
        try:
            props = jio.PropReader(jetid,runid,transient=transient)
        except:
            continue

        bs_dist,pd_arr = get_transient_xseries(runid,jetid,transient=transient)
        pd_arr = pd_arr[np.argsort(bs_dist)]
        bs_dist.sort()

        pd_epoch = np.interp(epoch_arr,bs_dist,pd_arr,left=0.0,right=0.0)
        SEA_arr = np.vstack((SEA_arr,pd_epoch))

    SEA_arr = SEA_arr[1:]
    SEA_mean = np.mean(SEA_arr,axis=0)
    SEA_std = np.std(SEA_arr,axis=0,ddof=1)

    return (epoch_arr,SEA_mean,SEA_std)

def jh2020_fig3():

    epoch_arr,SEA_mean_ABA,SEA_std_ABA = jh2020_SEA("ABA")
    epoch_arr,SEA_mean_ABC,SEA_std_ABC = jh2020_SEA("ABC")
    epoch_arr,SEA_mean_AEA,SEA_std_AEA = jh2020_SEA("AEA")
    epoch_arr,SEA_mean_AEC,SEA_std_AEC = jh2020_SEA("AEC")

    fig,ax = plt.subplots(1,1,figsize=(10,7))

    ax.set_xlabel("$\mathrm{X-X_{bs}~[R_e]}$",labelpad=10)
    ax.set_ylabel("$\mathrm{P_{dyn,mean}~[nPa]}$",labelpad=10)
    ax.set_xlim(-2.0,2.0)

    ax.plot(epoch_arr,SEA_mean_ABA,label="ABA")
    ax.plot(epoch_arr,SEA_mean_ABC,label="ABC")
    ax.plot(epoch_arr,SEA_mean_AEA,label="AEA")
    ax.plot(epoch_arr,SEA_mean_AEC,label="AEC")

    ax.legend(frameon=False,numpoints=1,markerscale=3)

    if not os.path.exists(homedir+"Figures/jh2020"):
        try:
            os.makedirs(homedir+"Figures/jh2020")
        except OSError:
            pass

    fig.savefig(homedir+"Figures/jh2020/fig3.png")
    plt.close(fig)
