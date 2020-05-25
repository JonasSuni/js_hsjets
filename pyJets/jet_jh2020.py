import sys
import matplotlib as mpl
import jet_aux as jx
if sys.version_info.major == 3:
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["black", jx.medium_blue, jx.dark_blue, jx.orange])
elif sys.version_info.major == 2:
    mpl.rcParams['axes.color_cycle'] = ["black", jx.medium_blue, jx.dark_blue, jx.orange]
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

r_e = 6.371e+6
m_p = 1.672621898e-27

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

    bs_dist_arr = jx.bs_dist(runid,time_arr,x_arr,y_arr)

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
        pd_arr = pd_arr[np.argsort(bs_dist)]/pd_sw
        bs_dist.sort()

        pd_epoch = np.interp(epoch_arr,bs_dist,pd_arr,left=np.nan,right=np.nan)
        SEA_arr = np.vstack((SEA_arr,pd_epoch))

    SEA_arr = SEA_arr[1:]
    SEA_mean = np.nanmean(SEA_arr,axis=0)
    SEA_std = np.nanstd(SEA_arr,axis=0,ddof=1)

    return (epoch_arr,SEA_mean,SEA_std)

def jh2020_hist(runid,transient="slamsjet"):

    hist_arr = np.array([])

    sj_counter = 0
    for n in range(3000):
        jetid = str(n).zfill(5)
        try:
            props = jio.PropReader(jetid,runid,transient=transient)
        except:
            continue

        bs_dist,pd_arr = get_transient_xseries(runid,jetid,transient=transient)
        bs_dist.sort()

        hist_arr = np.append(hist_arr,bs_dist)
        sj_counter += 1

    weight_arr = np.ones_like(hist_arr)/sj_counter
    hist,bin_edges = np.histogram(hist_arr,bins=20,range=(-1.5,1.5),weights=weight_arr)

    return (hist,bin_edges)

def jh2020_fig3():

    # epoch_arr,SEA_mean_ABA,SEA_std_ABA = jh2020_SEA("ABA")
    # epoch_arr,SEA_mean_ABC,SEA_std_ABC = jh2020_SEA("ABC")
    # epoch_arr,SEA_mean_AEA,SEA_std_AEA = jh2020_SEA("AEA")
    # epoch_arr,SEA_mean_AEC,SEA_std_AEC = jh2020_SEA("AEC")

    hist_ABA,bin_edges = jh2020_hist("ABA")
    hist_ABC,bin_edges = jh2020_hist("ABC")
    hist_AEA,bin_edges = jh2020_hist("AEA")
    hist_AEC,bin_edges = jh2020_hist("AEC")

    bins = bin_edges[:-1]

    fig,ax = plt.subplots(1,1,figsize=(10,7))

    ax.set_xlabel("$\mathrm{X-X_{bs}~[R_e]}$",labelpad=10,fontsize=20)
    # ax.set_ylabel("$\mathrm{P_{dyn,mean}~[P_{dyn,SW}]}$",labelpad=10,fontsize=20)
    ax.set_ylabel("Normalised count",labelpad=10,fontsize=20)
    #ax.set_xlim(-2.0,2.0)
    ax.axvline(0,linestyle="dashed",linewidth="0.5")
    ax.tick_params(labelsize=20)

    # ax.plot(epoch_arr,SEA_mean_ABA,label="ABA")
    # ax.plot(epoch_arr,SEA_mean_ABC,label="ABC")
    # ax.plot(epoch_arr,SEA_mean_AEA,label="AEA")
    # ax.plot(epoch_arr,SEA_mean_AEC,label="AEC")

    ax.step(bins,hist_ABA,where="post",label="ABA")
    ax.step(bins,hist_ABC,where="post",label="ABC")
    ax.step(bins,hist_AEA,where="post",label="AEA")
    ax.step(bins,hist_AEC,where="post",label="AEC")


    ax.legend(frameon=False,numpoints=1,markerscale=3)

    if not os.path.exists(homedir+"Figures/jh2020"):
        try:
            os.makedirs(homedir+"Figures/jh2020")
        except OSError:
            pass

    fig.savefig(homedir+"Figures/jh2020/fig3.png")
    plt.close(fig)

def get_cut_through(runid,start,stop,min_cellid,max_cellid,vars=["Pdyn","rho","v","B","Temperature"],save=True,custom=False):

    outputdir = wrkdir_DNR+"timeseries/{}/{}_{}/".format(runid,min_cellid,max_cellid)

    var_list = ["rho","v","vx","vy","vz","B","Bx","By","Bz","Pdyn","TParallel","TPerpendicular","beta"]
    vlsv_var_list = ["rho","v","v","v","v","B","B","B","B","Pdyn","TParallel","TPerpendicular","beta"]
    op_list = ["pass","magnitude","x","y","z","magnitude","x","y","z","pass","pass","pass","pass"]

    vars = vars+["Mmsx","TNonBackstream"]
    if custom:
        vars = ["Pdyn","rho","Pressure","Pmag","Ptot"]+["Mmsx","TNonBackstream"]

    if custom:
        cellid_range = jet_424_center_cells()
        outputdir = wrkdir_DNR+"timeseries/{}/{}_{}/".format(runid,"custom","424")
    else:
        cellid_range = np.arange(min_cellid,max_cellid+1,dtype=int)

    output_arr = np.zeros((len(vars),stop-start+1,cellid_range.size))

    bulkpath = jx.find_bulkpath(runid)

    for filenr in range(start,stop+1):
        print(filenr)
        bulkname = "bulk.{}.vlsv".format(str(filenr).zfill(7))
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
        for m in range(len(vars)):
            var = vars[m]
            if var in var_list:
                vlsv_var = vlsv_var_list[var_list.index(var)]
                vlsv_op = op_list[var_list.index(var)]
                output_arr[m][filenr-start] = vlsvobj.read_variable(vlsv_var,operator=vlsv_op,cellids=cellid_range)
            else:
                output_arr[m][filenr-start] = vlsvobj.read_variable(var,cellids=cellid_range)

    if save:
        if not os.path.exists(outputdir):
            try:
                os.makedirs(outputdir)
            except OSError:
                pass
        np.save(outputdir+"{}_{}".format(start,stop),output_arr)
        return None
    else:
        return output_arr

def find_one_jet():

    nrange = range(1,3000)
    for n in nrange:
        try:
            jetobj = jio.PropReader(str(n).zfill(5),"ABC",transient="slamsjet")
        except:
            continue
        if jetobj.read("time")[0] == 412.5 and not jetobj.read("is_slams").astype(bool).any():
            print(n)

    return None

def jet_424_center_cells():

    jetobj = jio.PropReader(str(424).zfill(5),"ABC",transient="slamsjet")
    vlsvobj = pt.vlsvfile.VlsvReader(vlasdir+"/2D/ABC/bulk/bulk.0000825.vlsv")
    x_arr = jetobj.read("x_mean")*r_e
    y_arr = jetobj.read("y_mean")*r_e
    z_arr = np.zeros_like(x_arr)

    coords = np.array([x_arr,y_arr,z_arr]).T
    cells = np.array([vlsvobj.get_cellid(coord) for coord in coords])

    return cells

def jh2020_cut_plot(runid,filenr,min_cellid=1814480,max_cellid=1814540):

    bulkpath = jx.find_bulkpath(runid)
    bulkname = "bulk.{}.vlsv".format(str(filenr).zfill(7))
    vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

    cell_range = np.arange(min_cellid,max_cellid+1)
    x_range = np.array([jx.get_cell_coordinates(runid,cell)[0]/r_e for cell in cell_range])
    y = jx.get_cell_coordinates(runid,cell_range[0])[1]/r_e

    var_list = ["rho","pdyn","B","v","TParallel","TPerpendicular"]
    norm_list = [1.e6,1.e-9,1.e-9,1.e3,1.e6,1.e6]
    label_list = ["$\mathrm{\\rho~[cm^{-3}]}$","$\mathrm{P_{dyn}~[nPa]}$","$\mathrm{B~[nT]}$","$\mathrm{v~[kms^{-1}]}$","$\mathrm{T~[MK]}$"]
    lim_list = [(0,30),(0,8),(-35,35),(-650,650),(0,20)]
    color_list = ["black", jx.medium_blue, jx.dark_blue, jx.orange]

    annot_list_list = [[""],[""],["B","Bx","By","Bz"],["v","vx","vy","vz"],["TPar","TPerp"]]

    raw_data_list = [vlsvobj.read_variable(var,cellids=cell_range)/norm_list[var_list.index(var)] for var in var_list]

    rho = raw_data_list[0]
    pdyn = raw_data_list[1]
    TPar = raw_data_list[4]
    TPerp = raw_data_list[5]
    v = raw_data_list[3]
    vmag = np.linalg.norm(v,axis=-1)
    B = raw_data_list[2]
    Bmag = np.linalg.norm(B,axis=-1)

    Ttot = np.array([TPar,TPerp]).T
    vtot = np.vstack((vmag,v.T)).T
    Btot = np.vstack((Bmag,B.T)).T

    x_1 = x_range
    x_2 = np.array((x_1,x_1)).T
    x_4 = np.array((x_1,x_1,x_1,x_1)).T

    x_list = [x_1,x_1,x_4,x_4,x_2]
    data_list = [rho,pdyn,Btot,vtot,Ttot]

    plt.ioff()

    fig,ax_list = plt.subplots(len(data_list),1,figsize=(10,15),sharex=True)
    fig.suptitle("Y = {:.3f} Re\nt = {} s".format(y,filenr/2),fontsize=20)

    for n in range(len(data_list)):
        ann_list = annot_list_list[n]
        ax = ax_list[n]
        ax.grid()
        ax.set_xlim(x_range[0],x_range[-1])
        ax.set_ylim(lim_list[n])
        x = x_list[n]
        data = data_list[n]
        ax.tick_params(labelsize=15)
        ax.plot(x,data)
        ax.set_ylabel(label_list[n],fontsize=20)
        if n == len(var_list)-1:
            ax.set_xlabel("$\mathrm{X~[R_e]}$",fontsize=20)
        for m in range(len(ann_list)):
            ax.annotate(ann_list[m],xy=(0.8+m*0.2/len(ann_list),0.05),xycoords="axes fraction",color=color_list[m])

    if not os.path.exists(wrkdir_DNR+"Figures/jh2020"):
        try:
            os.makedirs(wrkdir_DNR+"Figures/jh2020")
        except OSError:
            pass

    fig.savefig(wrkdir_DNR+"Figures/jh2020/cut_{}_{}_{}.png".format(filenr,min_cellid,max_cellid))
    plt.close(fig)

    return None

def event_for_mesh(runid,filenr,y,minx,maxx):

    event_props = np.array(jio.eventprop_read(runid,filenr,transient="slamsjet"),dtype=float)
    x_arr = event_props[:,1]
    y_arr = event_props[:,2]
    y_arr = y_arr[np.logical_and(x_arr<maxx,x_arr>minx)]
    x_arr = x_arr[np.logical_and(x_arr<maxx,x_arr>minx)]
    if np.min(np.abs(y_arr-y))<0.5:
        return x_arr[np.argmin(np.abs(y_arr-y))]
    else:
        return np.nan

def event_424_cut():

    var_list = ["Pdyn","rho","Pressure","Pmag","Ptot"]
    norm_list = [1.e-9,1.e6,1.e-9,1.e-9,1.e-9]
    cell_arr = jet_424_center_cells()
    x_arr = np.arange(cell_arr.size)

    data_arr = np.load(wrkdir_DNR+"/timeseries/{}/{}_{}/{}_{}.npy".format("ABC","custom","424",725,925))

    plt.ioff()

    fig,ax_list = plt.subplots(len(var_list),1,figsize=(10,20),sharex=True)

    for n in range(len(var_list)):
        data = data_arr[n][100]/norm_list[n]
        ax = ax_list[n]
        ax.tick_params(labelsize=15)
        ax.plot(x_arr,data)
        ax.set_xlim(x_arr[0],x_arr[-1])
        ax.set_ylabel(var_list[n],fontsize=10)
    ax_list[-1].set_xlabel("Pos along path",fontsize=20)

    if not os.path.exists(wrkdir_DNR+"Figures/jh2020"):
        try:
            os.makedirs(wrkdir_DNR+"Figures/jh2020")
        except OSError:
            pass
    fig.savefig(wrkdir_DNR+"Figures/jh2020/event_424_cut.png")
    plt.close(fig)

    return None

def jh2020_fig2_mesh(runid="ABC",start=400,stop=799,min_cellid=1814480,max_cellid=1814540,fromfile=True,clip="none",custom=False):

    var_list = ["Pdyn","rho","v","B","Temperature"]
    norm_list = [1.e-9,1.e6,1.e3,1.e-9,1.e6]
    if custom:
        var_list = ["Pdyn","rho","Pressure","Pmag","Ptot"]
        norm_list = [1.e-9,1.e6,1.e-9,1.e-9,1.e-9]

    if clip == "none":
        vmin_list = [0,0,0,0,0]
        vmax_list = [8,30,650,35,20]
    elif clip == "high":
        vmin_list = [0,0,0,0,0]
        vmax_list = [4.5,6.6,600,15,5]
    elif clip == "low":
        vmin_list = [1,6.6,150,5,0.5]
        vmax_list = [8,30,650,35,20]
    elif clip == "optimal":
        vmin_list = [0.0,3.3,100,5,0.5]
        vmax_list = [4.5,20,700,20,15]
    if custom:
        vmin_list = [0.0,3.3,0.2,0.0,1.0]
        vmax_list = [3,20,2,0.25,4]

    cell_arr = np.arange(min_cellid,max_cellid+1,dtype=int)
    x_arr = np.array([jx.get_cell_coordinates(runid,cell)[0]/r_e for cell in cell_arr])
    if custom:
        cell_arr = jet_424_center_cells()
        x_arr = np.arange(cell_arr.size)
    y = jx.get_cell_coordinates(runid,cell_arr[0])[1]/r_e
    time_arr = np.arange(start,stop+1)/2.0
    XmeshXT,TmeshXT = np.meshgrid(x_arr,time_arr)

    if min_cellid==1814480 and not custom:
        eventx_arr = np.array([event_for_mesh(runid,fnr,y,x_arr[0],x_arr[-1]) for fnr in np.arange(start,stop+1,dtype=int)])
    elif min_cellid==1784477 and not custom:
        onejet_obj = jio.PropReader(str(424).zfill(5),"ABC",transient="slamsjet")
        ox = onejet_obj.read("x_mean")
        oy = onejet_obj.read("y_mean")
        ot = onejet_obj.read("time")
        ox = ox[np.abs(oy-0.6212)<=0.5]
        ot = ot[np.abs(oy-0.6212)<=0.5]

    rho_sw = 3.3e6
    T_sw = 0.5e6

    if custom:
        data_arr = np.load(wrkdir_DNR+"/timeseries/{}/{}_{}/{}_{}.npy".format(runid,"custom","424",start,stop))
    else:
        if not fromfile:
            data_arr = get_cut_through(runid,start,stop,min_cellid,max_cellid,vars=var_list,save=False)
        else:
            data_arr = np.load(wrkdir_DNR+"/timeseries/{}/{}_{}/{}_{}.npy".format(runid,min_cellid,max_cellid,start,stop))

    rho_mask = (data_arr[1]>=2*rho_sw).astype(int)
    mms_mask = (data_arr[-2]<=1).astype(int)
    tcore_mask = (data_arr[-1]>=3*T_sw).astype(int)

    plt.ioff()

    fig,ax_list = plt.subplots(1,len(var_list),figsize=(20,10),sharex=True,sharey=True)
    im_list = []
    cb_list = []

    for n in range(len(var_list)):
        data = data_arr[n]/norm_list[n]
        ax = ax_list[n]
        if min_cellid == 1814480 and not custom:
            ax.axhline(328,color="black",linewidth=0.8)
            ax.axhline(337,color="black",linewidth=0.8)
            ax.axhline(345,color="black",linewidth=0.8)
        elif min_cellid == 1814480+60000+10 and not custom:
            ax.axhline(365,color="black",linewidth=0.8)
            ax.axhline(370,color="black",linewidth=0.8)
            ax.axhline(360,color="black",linewidth=0.8)
        elif min_cellid == 1784477 and not custom:
            ax.axhline(412.5,color="black",linewidth=0.8)
        if custom:
            ax.axhline(412.5,color="black",linewidth=0.8)
            ax.axhline(447.5,color="black",linewidth=0.8)
            ax.plot([x_arr[0],x_arr[-1]],[412.5,447.5],color="black",linewidth=0.8,linestyle="dashed")
        im_list.append(ax.pcolormesh(x_arr,time_arr,data,vmin=vmin_list[n],vmax=vmax_list[n]))
        cb_list.append(fig.colorbar(im_list[n],ax=ax))
        ax.contour(XmeshXT,TmeshXT,rho_mask,[0.5],linewidths=1.0,colors="black")
        ax.contour(XmeshXT,TmeshXT,mms_mask,[0.5],linewidths=1.0,colors=jx.violet)
        ax.contour(XmeshXT,TmeshXT,tcore_mask,[0.5],linewidths=1.0,colors=jx.orange)
        ax.tick_params(labelsize=15)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        #ax.xaxis.set_major_locator(MaxNLocator(nbins=6,prune="lower"))
        ax.set_title(var_list[n],fontsize=20)
        if min_cellid==1814480 and not custom:
            ax.plot(eventx_arr,time_arr,"o",color="red",markersize=2)
        elif min_cellid == 1784477 and not custom:
            ax.plot(ox,ot,"o",color="red",markersize=2)
        ax.set_xlim(x_arr[0],x_arr[-1])
        if n == 0:
            ax.set_ylabel("Simulation time [s]",fontsize=20)
            if custom:
                ax.set_xlabel("Pos along path",fontsize=20)
            else:
                ax.set_xlabel("$\mathrm{X~[R_e]}$",fontsize=20)

    fig.suptitle("Y = {:.3f} Re".format(y),fontsize=20)

    if not os.path.exists(wrkdir_DNR+"Figures/jh2020"):
        try:
            os.makedirs(wrkdir_DNR+"Figures/jh2020")
        except OSError:
            pass
    if custom:
        fig.savefig(wrkdir_DNR+"Figures/jh2020/fig2_mesh_{}_clip{}.png".format("custom",clip))
    else:
        fig.savefig(wrkdir_DNR+"Figures/jh2020/fig2_mesh_{}_clip{}.png".format(min_cellid,clip))
    plt.close(fig)

    return None

def get_timeseries_data(runid,start,stop,cellid):

    outputdir = wrkdir_DNR+"timeseries/{}/{}/".format(runid,cellid)

    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    bulkpath = jx.find_bulkpath(runid)
    var_list = ["rho","v","v","v","v","B","B","B","B","Pdyn","TParallel","TPerpendicular","beta"]
    op_list = ["pass","magnitude","x","y","z","magnitude","x","y","z","pass","pass","pass","pass"]
    output_arr = np.zeros((stop-start+1,len(var_list)+1))
    for filenr in range(start,stop+1):
        bulkname = "bulk.{}.vlsv".format(str(filenr).zfill(7))
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
        output_arr[filenr-start][0] = vlsvobj.read_parameter("t")
        for n in range(len(var_list)):
            data = vlsvobj.read_variable(var_list[n],operator=op_list[n],cellids=cellid)
            output_arr[filenr-start][n+1] = data

    np.savetxt(outputdir+"{}_{}".format(start,stop),output_arr)

    return None

def jh2020_fig2(xlim=[200.,399.5]):

    # time_arr = np.arange(580./2,1179./2+1./2,0.5)
    # time_list = [time_arr,np.array([time_arr,time_arr,time_arr,time_arr]).T,np.array([time_arr,time_arr,time_arr,time_arr]).T,time_arr,np.array([time_arr,time_arr]).T,time_arr]
    norm_list = [1.e6,1.e3,1.e3,1.e3,1.e3,1.e-9,1.e-9,1.e-9,1.e-9,1.e-9,1.e6,1.e6,1.]
    color_list = ["black", jx.medium_blue, jx.dark_blue, jx.orange]

    # data_in = np.loadtxt("taito_wrkdir/timeseries/ABC/1814507/580_1179").T
    # data_out = np.loadtxt("taito_wrkdir/timeseries/ABC/1814525/580_1179").T
    data_in = np.loadtxt("taito_wrkdir/timeseries/ABC/1814506/400_799").T
    data_out = np.loadtxt("taito_wrkdir/timeseries/ABC/1794536/400_799").T

    time_arr = data_in[0]
    time_list = [time_arr,np.array([time_arr,time_arr,time_arr,time_arr]).T,np.array([time_arr,time_arr,time_arr,time_arr]).T,time_arr,np.array([time_arr,time_arr]).T,time_arr]

    data_in = np.array([data_in[n+1]/norm_list[n] for n in range(len(data_in)-1)])
    data_out = np.array([data_out[n+1]/norm_list[n] for n in range(len(data_out)-1)])

    label_list = ["$\mathrm{\\rho~[cm^{-3}]}$","$\mathrm{v~[km/s]}$","$\mathrm{B~[nT]}$","$\mathrm{P_{dyn}~[nPa]}$","$\mathrm{T~[MK]}$","$\mathrm{\\beta}$"]

    fig,ax_list = plt.subplots(6,2,figsize=(15,15),sharex=True,sharey="row")

    #annot_list_list = [[""],["vx","vy","vz","v"],["Bx","By","Bz","B"],[""],["TPar","TPerp"],[""]]
    annot_list_list = [[""],["v","vx","vy","vz"],["B","Bx","By","Bz"],[""],["TPar","TPerp"],[""]]
    #re_arr_arr = np.array([3,0,1,2])
    re_arr_arr = np.array([0,1,2,3])

    for col in range(2):

        data = [data_in,data_out][col]
        xtitle = ["Inside bow shock","Outside bow shock"][col]
        data_list = [data[0],data[1:5][re_arr_arr].T,data[5:9][re_arr_arr].T,data[9],data[10:12].T,data[12]]
        for row in range(6):
            ann_list = annot_list_list[row]
            var = data_list[row]
            time = time_list[row]
            ax = ax_list[row][col]
            ax.tick_params(labelsize=15)
            ax.axvline(338.5,linestyle="dashed",linewidth=0.8)
            if len(var.T) == 4:
                ax.axhline(0,linestyle="dashed",linewidth=0.8)
            ax.plot(time,var)
            if col == 0:
                ax.set_ylabel(label_list[row],fontsize=15,labelpad=10)
                ax.axvspan(340,356,color="red",alpha=0.3,ec="none")
            if col == 1:
                ax.axvspan(325,335.5,color="red",alpha=0.3,ec="none")
            if row == 0:
                ax.set_title(xtitle,fontsize=15)
            if row == 5:
                ax.set_xlabel("Simulation time [s]",fontsize=15,labelpad=10)
            ax.set_xlim(xlim[0],xlim[1])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6,prune="lower"))
            for m in range(len(ann_list)):
                ax.annotate(ann_list[m],xy=(0.8+m*0.2/len(ann_list),0.05),xycoords="axes fraction",color=color_list[m])

    if not os.path.exists(homedir+"Figures/jh2020"):
        try:
            os.makedirs(homedir+"Figures/jh2020")
        except OSError:
            pass

    fig.savefig(homedir+"Figures/jh2020/fig2.png")
    plt.close(fig)

def mag_thresh_plot(allow_splinters=True):

    runid_list = ["ABA","ABC","AEA","AEC"]
    #mt_str_list = ["1.1","1.2","1.3","1.4","1.5","1.6","1.7","1.8","1.9","2.0","2.1","2.2","2.5","2.8","3.0"]
    mt_str_list = ["1.1","1.3","1.5","1.7","1.9","2.1","2.3","2.5","2.7","3.0"]

    share_arr = np.zeros((len(mt_str_list),len(runid_list)),dtype=float)
    slams_share_arr = np.zeros((len(mt_str_list),len(runid_list)),dtype=float)
    slams_number_arr = np.zeros((len(mt_str_list),len(runid_list)),dtype=float)
    jet_number_arr = np.zeros((len(mt_str_list),len(runid_list)),dtype=float)
    for n in range(len(mt_str_list)):
        #print(mt_str_list[n])
        data = np.loadtxt(wrkdir_DNR+"sjn_counts/sjn_count_{}_{}.txt".format(mt_str_list[n],allow_splinters)).astype(float)
        share = data[0]/(data[0]+data[1])
        slams_share = data[0]/(data[0]+data[2])
        slams_number = data[2]+data[0]
        jet_number = data[1]+data[0]
        share_arr[n] = share
        slams_share_arr[n] = slams_share
        slams_number_arr[n] = slams_number
        jet_number_arr[n] = jet_number

    share_arr = share_arr.T
    slams_share_arr = slams_share_arr.T
    slams_number_arr = slams_number_arr.T
    jet_number_arr = jet_number_arr.T
    mt_arr = np.array(list(map(float,mt_str_list)))

    fig,ax_list = plt.subplots(4,1,figsize=(8,10))
    for m in range(len(runid_list)):
        #ax_list[0].semilogy(mt_arr,slams_number_arr[m],label=runid_list[m])
        ax_list[0].plot(mt_arr,jet_number_arr[m],label=runid_list[m])
        ax_list[1].plot(mt_arr,slams_number_arr[m],label=runid_list[m])
        ax_list[2].plot(mt_arr,slams_share_arr[m],label=runid_list[m])
        ax_list[3].plot(mt_arr,share_arr[m],label=runid_list[m])

    ax_list[3].set_xlabel("Foreshock structure threshold $|B|/B_{IMF}$",fontsize=20,labelpad=10)
    ax_list[0].set_ylabel("Number of jets",fontsize=15,labelpad=10)
    ax_list[1].set_ylabel("Number of SLAMS",fontsize=15,labelpad=10)
    ax_list[2].set_ylabel("Slamsjets per SLAMS",fontsize=15,labelpad=10)
    ax_list[3].set_ylabel("Slamsjets per jet",fontsize=15,labelpad=10)
    ax_list[0].set_title("Allow splinters = {}".format(allow_splinters),fontsize=20)
    ax_list[0].legend(frameon=False,numpoints=1,markerscale=3)
    for ax in ax_list:
        ax.grid()
        ax.set_xlim(mt_arr[0],mt_arr[-1])
        ax.tick_params(labelsize=15)
    ax_list[0].set_ylim(bottom=0)
    ax_list[1].set_ylim(bottom=0)
    ax_list[2].set_ylim(0,1)
    ax_list[3].set_ylim(0,1)
    plt.tight_layout()

    fig.savefig(wrkdir_DNR+"sjratio_fig_{}.png".format(allow_splinters))
    plt.close(fig)
    return None

def sj_non_counter(allow_splinters=True,mag_thresh=1.4):

    runids = ["ABA","ABC","AEA","AEC"]

    data_arr = np.array([separate_jets(runid,allow_splinters) for runid in runids]).flatten()
    count_arr = np.array([arr.size for arr in data_arr])
    count_arr = np.reshape(count_arr,(4,3)).T

    print("Runs:           ABA ABC AEA AEC\n")
    print("SJ Jets:        {}\n".format(count_arr[0]))
    print("Non-SJ Jets:    {}\n".format(count_arr[1]))
    print("Non-SJ SLAMS:   {}\n".format(count_arr[2]))
    print("SJ/jet ratio:   {}\n".format(count_arr[0].astype(float)/(count_arr[0]+count_arr[1])))
    print("SJ/SLAMS ratio: {}\n".format(count_arr[0].astype(float)/(count_arr[0]+count_arr[2])))

    np.savetxt(wrkdir_DNR+"sjn_counts/sjn_count_{}_{}.txt".format(mag_thresh,allow_splinters),count_arr)

    return np.reshape(data_arr,(4,3))

def separate_jets(runid,allow_splinters=True):

    runids = ["ABA","ABC","AEA","AEC"]
    run_cutoff_dict = dict(zip(runids,[10,8,10,8]))

    sj_jet_ids = []
    non_sj_ids = []
    pure_slams_ids = []

    for n1 in range(3000):

        try:
            props = jio.PropReader(str(n1).zfill(5),runid,transient="slamsjet")
        except:
            continue

        if np.logical_and(props.read("is_slams")==1,props.read("is_jet")==1).any():
            if not allow_splinters and "splinter" in props.meta:
                splinter_time = props.read("time")[props.read("is_splinter")==1][0]
                non_jet_time = props.read("time")[props.read("is_jet")==1][0]-0.5
                extra_splin_times = np.array(props.get_splin_times())
                if splinter_time > non_jet_time:
                    continue
                elif (extra_splin_times > non_jet_time).any():
                    continue
                else :
                    sj_jet_ids.append(n1)
            else:
                sj_jet_ids.append(n1)
        elif (props.read("is_jet")==1).any():
            if not allow_splinters and "splinter" in props.meta:
                continue
            elif props.read("at_bow_shock")[0] != 1:
                continue
            else:
                non_sj_ids.append(n1)
        elif (props.read("is_slams")==1).any():
            # if not allow_splinters and "splinter" in props.meta:
            #     continue
            # else:
            #     pure_slams_ids.append(n1)
            pure_slams_ids.append(n1)

    return [np.array(sj_jet_ids),np.array(non_sj_ids),np.array(pure_slams_ids)]

def separate_jets_old(runid):

    runids = ["ABA","ABC","AEA","AEC"]
    run_cutoff_dict = dict(zip(runids,[10,8,10,8]))

    sj_jet_ids = []
    non_sj_ids = []

    for n1 in range(3000):
        try:
            props = jio.PropReader(str(n1).zfill(5),runid,transient="jet")
        except:
            continue

        # if "splinter" in props.meta:
        #     continue

        if props.read("sep_from_bs")[0] > 0.5:
            continue

        jet_first_cells = props.get_cells()[0]
        jet_first_time = props.read("time")[0]

        for n2 in range(3000):
            if n2 == 2999:
                if "splinter" not in props.meta:
                    non_sj_ids.append(n1)
                break

            try:
                props_sj = jio.PropReader(str(n2).zfill(5),runid,transient="slamsjet")
            except:
                continue

            sj_cells = props_sj.get_cells()
            sj_times = props_sj.read("time")
            try:
                matched_cells = sj_cells[np.where(sj_times==jet_first_time)[0][0]]
            except:
                continue

            if np.intersect1d(jet_first_cells,matched_cells).size > 0.05*len(jet_first_cells):
                sj_jet_ids.append(n1)
                break


    return [np.array(sj_jet_ids),np.array(non_sj_ids)]

def find_slams_of_jet(runid):

    sj_ids=[]
    slams_ids=[]

    for n1 in range(3000):
        try:
            props_sj = jio.PropReader(str(n1).zfill(5),runid,transient="slamsjet")
        except:
            continue

        sj_first_cells = props_sj.get_cells()[0]
        for n2 in range(3000):
            try:
                props_slams = jio.PropReader(str(n2).zfill(5),runid,transient="slams")
            except:
                continue
            slams_first_cells = props_slams.get_cells()[0]
            if np.intersect1d(slams_first_cells,sj_first_cells).size > 0.25*len(slams_first_cells):
                sj_ids.append(n1)
                slams_ids.append(n2)
                break

    return [np.array(sj_ids),np.array(slams_ids)]

def get_indent_depth(runid,crit="ew_pd"):

    x_res = 227000/r_e

    sj_ids,slams_ids=find_slams_of_jet(runid)
    indents = []
    depths = []

    for n in range(sj_ids.size):
        sj_props = jio.PropReader(str(sj_ids[n]).zfill(5),runid,transient="slamsjet")
        x_sj = sj_props.read("x_mean")
        y_sj = sj_props.read("y_mean")
        t_sj = sj_props.read("time")
        sj_dist = jx.bs_rd(runid,t_sj,x_sj,y_sj)
        sj_dist_min = np.min(sj_dist)

        slams_props = jio.PropReader(str(slams_ids[n]).zfill(5),runid,transient="slams")
        is_upstream_slams = slams_props.read("is_upstream")
        if np.all(is_upstream_slams==0.0):
            continue
        t_slams = slams_props.read("time")
        last_time = t_slams[is_upstream_slams>0][-1]


        if crit == "ew_pd":
            bow_shock_value = slams_props.read_at_time("ew_pd_enh",last_time)/ja.sw_normalisation(runid,"pd_avg")
        elif crit == "nonloc":
            bs_ch = slams_props.read_at_time("xbs_ch",last_time)
            bs_rho = slams_props.read_at_time("xbs_rho",last_time)
            bs_mms = slams_props.read_at_time("xbs_mms",last_time)
            bow_shock_value = np.linalg.norm([bs_ch-bs_rho,bs_rho-bs_mms,bs_mms-bs_ch])
        else:
            slams_cells = slams_props.get_cells()
            last_cells = np.array(slams_cells)[is_upstream_slams>0][-1]
            cell_pos = np.array([jx.get_cell_coordinates(runid,cellid)/r_e for cellid in last_cells])
            cell_x = cell_pos[:,0]
            cell_y = cell_pos[:,1]
            cell_t_arr = np.ones_like(cell_x)*(t_slams[is_upstream_slams>0][-1])
            slams_bs_dist = jx.bs_rd(runid,cell_t_arr,cell_x,cell_y)
            upstream_dist_min = np.min(slams_bs_dist)
            bow_shock_value = upstream_dist_min-x_res

        depths.append(sj_dist_min)
        indents.append(bow_shock_value)

    return [np.array(depths),np.array(indents)]

def jh2020_fig4(crit="ew_pd"):

    runids = ["ABA","ABC","AEA","AEC"]
    marker_list = ["x","o","^","v"]

    fig,ax = plt.subplots(1,1,figsize=(10,10))
    for runid in runids:
        depths,indents = get_indent_depth(runid,crit=crit)
        ax.plot(depths,indents,marker_list[runids.index(runid)],label=runid)

    ax.set_xlabel("$\mathrm{Last~X-X_{bs}~[R_e]}$",fontsize=20,labelpad=10)
    #ax.set_ylabel("$\mathrm{Indentation~[R_e]}$",fontsize=20,labelpad=10)
    if crit == "ew_pd":
        ax.set_ylabel("$\mathrm{Mean~earthward~P_{dyn}~[P_{dyn,sw}]}$",fontsize=20,labelpad=10)
    elif crit == "nonloc":
        ax.set_ylabel("$\mathrm{Bow~shock~nonlocality~[R_e]}$",fontsize=20,labelpad=10)
    else:
        ax.set_ylabel("$\mathrm{Bow~shock~indentation~[R_e]}$",fontsize=20,labelpad=10)
    ax.legend(frameon=False,numpoints=1,markerscale=2)
    ax.tick_params(labelsize=20)
    ax.axvline(0,linestyle="dashed",linewidth=0.6,color="black")
    #ax.axhline(0,linestyle="dashed",linewidth=0.6,color="black")
    #ax.plot([-3.0,3.0],[-3.0,3.0],linestyle="dashed",linewidth=0.6,color="black")
    ax.set_xlim(-2.5,0.5)
    #ax.set_ylim(-0.3,0.6)

    if not os.path.exists(homedir+"Figures/jh2020"):
        try:
            os.makedirs(homedir+"Figures/jh2020")
        except OSError:
            pass

    fig.savefig(homedir+"Figures/jh2020/fig4_{}.png".format(crit))
    plt.close(fig)

def jh2020_fig1(var="pdyn"):

    vars_list = ["pdyn","core_heating","rho","Mms","B"]
    var_index = vars_list.index(var)
    label_list = ["nPa","$T_{sw}$","$cm^{-3}$","","nT"]
    vmax_list = [4.5,3.0,6.6,1,10]
    expr_list = [pc.expr_pdyn,pc.expr_coreheating,pc.expr_srho,pc.expr_mms,pc.expr_B]
    sj_jet_ids,non_sj_ids = separate_jets("ABC")

    global filenr_g
    global runid_g
    global sj_jetobs
    global non_sjobs
    global draw_arrows

    draw_arrows = True

    runid_g = "ABC"

    sj_jetobs = [jio.PropReader(str(n).zfill(5),"ABC",transient="slamsjet") for n in sj_jet_ids]
    non_sjobs = [jio.PropReader(str(n).zfill(5),"ABC",transient="slamsjet") for n in non_sj_ids]

    outputdir = wrkdir_DNR+"Figures/jh2020/"

    #filepath = "/scratch/project_2000203/sunijona/vlasiator/2D/ABC/bulk/bulk.0000677.vlsv"
    #filepath = "/scratch/project_2000203/2D/ABC/bulk/bulk.0000714.vlsv"
    #filepath = vlasdir+"/2D/ABC/bulk/bulk.0000714.vlsv"
    filepath = vlasdir+"/2D/ABC/bulk/bulk.0000825.vlsv"

    #filenr_g = 677
    #filenr_g = 714
    filenr_g = 825

    colmap = "parula"
    if var == "Mms":
        colmap = "parula"

    #pt.plot.plot_colormap(filename=filepath,outputfile=outputdir+"fig1a_{}.png".format(var),usesci=0,lin=1,expression=expr_list[var_index],vmin=0,vmax=vmax_list[var_index],colormap=colmap,cbtitle=label_list[var_index],pass_vars=["rho","v","CellID","Pdyn","RhoNonBackstream","PTensorNonBackstreamDiagonal","Mms","B"],Earth=1)

    pt.plot.plot_colormap(filename=filepath,outputfile=outputdir+"fig1b_{}.png".format(var),boxre=[6,18,-6,6],usesci=0,lin=1,expression=expr_list[var_index],vmin=0,vmax=vmax_list[var_index],colormap=colmap,cbtitle=label_list[var_index],external=jh20f1_ext,pass_vars=["rho","v","CellID","Pdyn","RhoNonBackstream","PTensorNonBackstreamDiagonal","Mmsx","B","core_heating"])

def jh2020_movie(runid,start,stop,var="Pdyn",arr_draw=False,debug=False):

    runid_list = ["ABA","ABC","AEA","AEC"]
    maxfnr_list = [839,1179,1339,879]
    if start > maxfnr_list[runid_list.index(runid)]:
        return 0

    vars_list = ["Pmag","Ptot","Pressure","Pdyn","rho","B","v","Temperature"]
    var_index = vars_list.index(var)
    #label_list = ["nPa","nPa","$T_{sw}$","$cm^{-3}$","","nT"]
    vmax_list = [0.25,4,2,3.0,20,20,700,15]
    vmin_list = [0.0,1.0,0.2,0,3.3,5,100,0.5]
    vscale_list = [1e9,1e9,1e9,1e9,1.0e-6,1e9,1e-3,1e-6]
    #expr_list = [pc.expr_pdyn,pc.expr_coreheating,pc.expr_srho,pc.expr_mms,pc.expr_B]
    sj_jet_ids,non_sj_ids,pure_slams_ids = separate_jets(runid)

    global filenr_g
    global runid_g
    global sj_jetobs
    global non_sjobs
    global draw_arrows

    draw_arrows = arr_draw

    runid_g = runid

    sj_jetobs = [jio.PropReader(str(n).zfill(5),runid,transient="slamsjet") for n in sj_jet_ids]
    non_sjobs = [jio.PropReader(str(n).zfill(5),runid,transient="slamsjet") for n in non_sj_ids]

    outputdir = wrkdir_DNR+"jh2020_movie/{}/{}/".format(runid,var)
    if debug:
        outputdir = wrkdir_DNR+"jh2020_debug/{}/{}/".format(runid,var)
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    bulkpath = jx.find_bulkpath(runid)

    vmax = vmax_list[var_index]
    vmin = vmin_list[var_index]
    vscale = vscale_list[var_index]
    boxre = [6,18,-6,6]
    if runid in ["ABA","AEA"]:
        if var == "Pdyn":
            vmax = 1.5
        boxre = [6,18,-8,6]

    for itr in range(start,stop+1):
        filepath = bulkpath+"bulk.{}.vlsv".format(str(itr).zfill(7))
        filenr_g = itr

        colmap = "parula"
        if var == "Mmsx":
            colmap = "parula"

        #pt.plot.plot_colormap(filename=filepath,outputfile=outputdir+"{}.png".format(str(itr).zfill(5)),boxre=boxre,usesci=0,lin=1,expression=expr_list[var_index],tickinterval=2,vmin=0,vmax=vmax,colormap=colmap,cbtitle=label_list[var_index],external=jh20f1_ext,pass_vars=["RhoNonBackstream","PTensorNonBackstreamDiagonal","B","v","rho","core_heating","CellID","Mmsx"])

        pt.plot.plot_colormap(filename=filepath,outputfile=outputdir+"{}.png".format(str(itr).zfill(5)),boxre=boxre,usesci=0,lin=1,var=var,tickinterval=2,vmin=vmin,vmax=vmax,vscale=vscale,colormap=colmap,external=jh20f1_ext,pass_vars=["RhoNonBackstream","PTensorNonBackstreamDiagonal","B","v","rho","core_heating","CellID","Mmsx"])

        pt.plot.plot_colormap(filename=filepath,outputfile=outputdir+"zoom/{}.png".format(str(itr).zfill(5)),boxre=[8,12,-2,2],usesci=0,lin=1,vscale=vscale,var=var,tickinterval=1,vmin=vmin,vmax=vmax,colormap=colmap,external=jh20f1_ext,pass_vars=["RhoNonBackstream","PTensorNonBackstreamDiagonal","B","v","rho","core_heating","CellID","Mmsx"])

def jh20f1_ext(ax, XmeshXY,YmeshXY, pass_maps):

    cellids = pass_maps["CellID"]
    rho = pass_maps["rho"]
    mmsx = pass_maps["Mmsx"]
    core_heating = pass_maps["core_heating"]
    if runid_g in ["ABA","AEA"]:
        rho_sw = 1.e6
    else:
        rho_sw = 3.3e6

    #slams_cells = jio.eventfile_read("ABC",filenr_g,transient="slams")
    #slams_cells = np.array([item for sublist in slams_cells for item in sublist])
    #jet_cells = jio.eventfile_read("ABC",filenr_g,transient="jet")
    #jet_cells = np.array([item for sublist in jet_cells for item in sublist])
    slams_cells = np.loadtxt("/wrk/users/jesuni/working/SLAMS/Masks/{}/{}.mask".format(runid_g,filenr_g)).astype(int)
    jet_cells = np.loadtxt("/wrk/users/jesuni/working/jets/Masks/{}/{}.mask".format(runid_g,filenr_g)).astype(int)

    slams_mask = np.in1d(cellids,slams_cells).astype(int)
    slams_mask = np.reshape(slams_mask,cellids.shape)

    jet_mask = np.in1d(cellids,jet_cells).astype(int)
    jet_mask = np.reshape(jet_mask,cellids.shape)

    ch_mask = (core_heating > 3*0.5e6).astype(int)
    mach_mask = (mmsx < 1).astype(int)
    rho_mask = (rho > 2*rho_sw).astype(int)

    #x_list = []
    #y_list = []

    # for n in range(3000):
    #     try:
    #         props = jio.PropReader(str(n).zfill(5),"ABC",transient="slamsjet")
    #     except:
    #         continue
    #     if filenr_g/2.0 in props.read("time"):
    #         x_list.append(props.read_at_time("x_mean",filenr_g/2.0))
    #         y_list.append(props.read_at_time("y_mean",filenr_g/2.0))

    sj_xlist = []
    sj_ylist = []
    non_xlist = []
    non_ylist = []

    for jetobj in sj_jetobs:
        if filenr_g/2.0 in jetobj.read("time"):
            sj_xlist.append(jetobj.read_at_time("x_mean",filenr_g/2.0))
            sj_ylist.append(jetobj.read_at_time("y_mean",filenr_g/2.0))
    for jetobj in non_sjobs:
        if filenr_g/2.0 in jetobj.read("time"):
            non_xlist.append(jetobj.read_at_time("x_mean",filenr_g/2.0))
            non_ylist.append(jetobj.read_at_time("y_mean",filenr_g/2.0))

    #bs_fit = jx.bow_shock_jonas(runid_g,filenr_g)[::-1]
    #mp_fit = jx.mag_pause_jonas(runid_g,filenr_g)[::-1]
    #y_bs = np.arange(-8,6.01,0.05)
    #x_bs = np.polyval(bs_fit,y_bs)
    #x_mp = np.polyval(mp_fit,y_bs)

    #bs_cont, = ax.plot(x_bs,y_bs,color="black",linewidth=0.8)
    #mp_cont, = ax.plot(x_mp,y_bs,color="black",linewidth=0.8)

    rho_cont = ax.contour(XmeshXY,YmeshXY,rho_mask,[0.5],linewidths=0.6,colors="black")
    mach_cont = ax.contour(XmeshXY,YmeshXY,mach_mask,[0.5],linewidths=0.6,colors=jx.violet)
    ch_cont = ax.contour(XmeshXY,YmeshXY,ch_mask,[0.5],linewidths=0.6,colors=jx.orange)

    slams_cont = ax.contour(XmeshXY,YmeshXY,slams_mask,[0.5],linewidths=0.8,colors=jx.dark_blue)
    jet_cont = ax.contour(XmeshXY,YmeshXY,jet_mask,[0.5],linewidths=0.8,colors="brown")

    non_pos, = ax.plot(non_xlist,non_ylist,"o",color="black",markersize=1.5)
    sj_pos, = ax.plot(sj_xlist,sj_ylist,"o",color="red",markersize=1.5)

    if draw_arrows:
        arrow_coords = jx.bs_norm(runid_g,filenr_g)
        for n in range(1,len(arrow_coords)):
            nx,ny,dnx,dny = arrow_coords[n]
            if ny//0.5 > arrow_coords[n-1][1]//0.5:
                ax.arrow(nx,ny,dnx,dny,head_width=0.1,width=0.01,color=jx.orange)

    #xy_pos, = ax.plot(x_list,y_list,"o",color=jx.crimson,markersize=2)

    #is_coords = jx.get_cell_coordinates(runid_g,1814480)/r_e
    #os_coords = jx.get_cell_coordinates(runid_g,1814540)/r_e

    #is2 = jx.get_cell_coordinates("ABC",1814480+2000*30+10)/r_e
    #os2 = jx.get_cell_coordinates("ABC",1814540+2000*30+10)/r_e

    # is_pos, = ax.plot(is_coords[0],is_coords[1],">",color="black",markersize=2)
    # os_pos, = ax.plot(os_coords[0],os_coords[1],"<",color="black",markersize=2)

    #cut_through_plot, = ax.plot([is_coords[0],os_coords[0]],[is_coords[1],os_coords[1]],color="black",linewidth=0.8)
    #cut_through_plot2, = ax.plot([is2[0],os2[0]],[is2[1],os2[1]],color="black",linewidth=0.8)

def jh20_slams_movie(start,stop,var="Pdyn",vmax=15e-9):

    outputdir = wrkdir_DNR+"jh20_slams_movie/{}/".format(var)
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    bulkpath = jx.find_bulkpath("ABC")
    for itr in range(start,stop+1):
        filepath = bulkpath+"bulk.{}.vlsv".format(str(itr).zfill(7))

        colmap = "parula"

        pt.plot.plot_colormap(filename=filepath,outputfile=outputdir+"{}.png".format(str(itr).zfill(5)),boxre=[6,18,-6,6],var=var,usesci=0,lin=1,vmin=0,vmax=15e-9,colormap=colmap,external=jh20_slams_ext,pass_vars=["rho","v","CellID","Pdyn","RhoNonBackstream","PTensorNonBackstreamDiagonal","Mms","B","X","Y"])

def jh20_slams_ext(ax, XmeshXY,YmeshXY, pass_maps):

    cellids = pass_maps["CellID"]
    B = pass_maps["B"]
    X = pass_maps["X"]
    Y = pass_maps["Y"]
    rho = pass_maps["rho"]
    pdyn = pass_maps["Pdyn"]
    pr_PTDNBS = pass_maps["PTensorNonBackstreamDiagonal"]
    pr_rhonbs = pass_maps["RhoNonBackstream"]

    T_sw = 0.5e+6
    epsilon = 1.e-10
    kb = 1.38065e-23

    pr_pressurenbs = (1.0/3.0) * (pr_PTDNBS.sum(-1))
    pr_TNBS = pr_pressurenbs/ ((pr_rhonbs + epsilon) * kb)

    B_sw = 5.0e-9
    pd_sw = 3.3e6*600e3*600e3*m_p

    Bmag = np.linalg.norm(B,axis=-1)

    slams = np.ma.masked_greater_equal(Bmag,3.0*B_sw)
    #slams.mask[pr_TNBS >= 2.0*T_sw] = False
    #slams.mask[pdyn<=0.5*pd_sw] = False
    slams_mask = slams.mask.astype(int)

    slams_cont = ax.contour(XmeshXY,YmeshXY,slams_mask,[0.5],linewidths=0.8,colors=jx.orange)
