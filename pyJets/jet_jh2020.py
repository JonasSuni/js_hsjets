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

r_e = 6.371e+6

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
        pd_arr = pd_arr[np.argsort(bs_dist)]/pd_sw
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

    ax.set_xlabel("$\mathrm{X-X_{bs}~[R_e]}$",labelpad=10,fontsize=20)
    ax.set_ylabel("$\mathrm{P_{dyn,mean}~[P_{dyn,SW}]}$",labelpad=10,fontsize=20)
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
            if np.intersect1d(slams_first_cells,sj_first_cells).size > 0.75*len(slams_first_cells):
                sj_ids.append(n1)
                slams_ids.append(n2)
                break

    return [np.array(sj_ids),np.array(slams_ids)]

def get_indent_depth(runid):

    sj_ids,slams_ids=find_slams_of_jet(runid)
    indents = []
    depths = []

    for n in range(sj_ids.size):
        sj_props = jio.PropReader(str(sj_ids[n]).zfill(5),runid,transient="slamsjet")
        x_sj = sj_props.read("x_mean")
        y_sj = sj_props.read("y_mean")
        t_sj = sj_props.read("time")
        sj_dist = jx.bs_dist_markus(runid,t_sj,x_sj,y_sj)
        sj_dist_min = np.min(sj_dist)

        slams_props = jio.PropReader(str(slams_ids[n]).zfill(5),runid,transient="slams")
        is_upstream_slams = slams_props.read("is_upstream")
        if np.all(is_upstream_slams==0.0):
            continue
        t_slams = slams_props.read("time")
        slams_cells = slams_props.get_cells()
        last_cells = np.array(slams_cells)[is_upstream_slams>0][-1]
        cell_pos = np.array([jx.get_cell_coordinates(runid,cellid)/r_e for cellid in last_cells])
        cell_x = cell_pos[:,0]
        cell_y = cell_pos[:,1]
        cell_t_arr = np.ones_like(cell_x)*(t_slams[is_upstream_slams>0][-1])
        slams_bs_dist = jx.bs_dist_markus(runid,cell_t_arr,cell_x,cell_y)
        upstream_dist_min = np.min(slams_bs_dist)

        depths.append(sj_dist_min)
        indents.append(upstream_dist_min)

    return [np.array(depths),np.array(indents)]

def jh2020_fig4():

    runids = ["ABA","ABC","AEA","AEC"]

    fig,ax = plt.subplots(1,1,figsize=(10,10))
    for runid in runids:
        depths,indents = get_indent_depth(runid)
        ax.plot(depths,indents,"x",label=runid)

    ax.set_xlabel("$\mathrm{Last~X-X_{bs}~[R_e]}$",fontsize=20,labelpad=10)
    ax.set_ylabel("$\mathrm{Indentation~[R_e]}$",fontsize=20,labelpad=10)
    ax.legend(frameon=False,numpoints=1,markerscale=3)

    if not os.path.exists(homedir+"Figures/jh2020"):
        try:
            os.makedirs(homedir+"Figures/jh2020")
        except OSError:
            pass

    fig.savefig(homedir+"Figures/jh2020/fig4_old.png")
    plt.close(fig)

def jh2020_fig1():

    outputdir = homedir+"Figures/jh2020/"

    filepath = "/scratch/project_2000203/sunijona/vlasiator/2D/ABC/bulk/bulk.0000677.vlsv"

    pt.plot.plot_colormap(filename=filepath,outputfile=outputdir+"fig1a.png",usesci=0,lin=1,expression=pc.expr_pdyn,vmin=0,vmax=4.5,colormap="parula",cbtitle="nPa",pass_vars=["rho","v","CellID","Pdyn"],Earth=1)

    pt.plot.plot_colormap(filename=filepath,outputfile=outputdir+"fig1b.png",boxre=[6,18,-6,6],usesci=0,lin=1,expression=pc.expr_pdyn,vmin=0,vmax=4.5,colormap="parula",cbtitle="nPa",external=jh20f1_ext,pass_vars=["rho","v","CellID","Pdyn"])

def jh20f1_ext(ax, XmeshXY,YmeshXY, pass_maps):

    cellids = pass_maps["CellID"]

    slams_cells = jio.eventfile_read("ABC",677,transient="slams")
    slams_cells = np.array([item for sublist in slams_cells for item in sublist])
    jet_cells = jio.eventfile_read("ABC",677,transient="jet")
    jet_cells = np.array([item for sublist in jet_cells for item in sublist])

    slams_mask = np.in1d(cellids,slams_cells).astype(int)
    slams_mask = np.reshape(slams_mask,cellids.shape)

    jet_mask = np.in1d(cellids,jet_cells).astype(int)
    jet_mask = np.reshape(jet_mask,cellids.shape)

    x_list = []
    y_list = []

    for n in range(3000):
        try:
            props = jio.PropReader(str(n).zfill(5),"ABC",transient="slamsjet")
        except:
            continue
        if 338.5 in props.read("time"):
            x_list.append(props.read_at_time("x_mean",338.5))
            y_list.append(props.read_at_time("y_mean",338.5))

    bs_fit = jx.bow_shock_markus("ABC",677)[::-1]
    y_bs = np.arange(-6,6.01,0.05)
    x_bs = np.polyval(bs_fit,y_bs)

    bs_cont, = ax.plot(x_bs,y_bs,color="black")

    slams_cont = ax.contour(XmeshXY,YmeshXY,slams_mask,[0.5],linewidths=0.8,colors=jx.dark_blue)
    jet_cont = ax.contour(XmeshXY,YmeshXY,jet_mask,[0.5],linewidths=0.8,colors=jx.orange)

    xy_pos, = ax.plot(x_list,y_list,"o",color=jx.crimson,markersize=2)
