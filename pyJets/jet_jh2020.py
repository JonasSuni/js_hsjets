import matplotlib as mpl
import jet_aux as jx
#mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[jx.violet, jx.medium_blue, jx.dark_blue, jx.orange])
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["black", jx.medium_blue, jx.dark_blue, jx.orange])

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
    ax.axvline(0,linestyle="dashed",linewidth="0.5")
    ax.tick_params(labelsize=20)

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

def separate_jets(runid):

    sj_jet_ids = []
    non_sj_ids = []

    for n1 in range(3000):
        try:
            props = jio.PropReader(str(n1).zfill(5),runid,transient="jet")
        except:
            continue

        if "splinter" in props.meta:
            continue

        x_0 = props.read("x_mean")[0]
        xbs_0 = props.read("xbs_ch")[0]

        if x_0-xbs_0 > 0:
            sj_jet_ids.append(n1)
        else:
            non_sj_ids.append(n1)

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
            if np.intersect1d(slams_first_cells,sj_first_cells).size > 0.75*len(slams_first_cells):
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
        # slams_cells = slams_props.get_cells()
        # last_cells = np.array(slams_cells)[is_upstream_slams>0][-1]
        # cell_pos = np.array([jx.get_cell_coordinates(runid,cellid)/r_e for cellid in last_cells])
        # cell_x = cell_pos[:,0]
        # cell_y = cell_pos[:,1]
        # cell_t_arr = np.ones_like(cell_x)*(t_slams[is_upstream_slams>0][-1])
        # slams_bs_dist = jx.bs_rd(runid,cell_t_arr,cell_x,cell_y)
        # upstream_dist_min = np.min(slams_bs_dist)
        if crit == "ew_pd":
            bow_shock_value = slams_props.read_at_time("ew_pd_enh",last_time)/ja.sw_normalisation(runid,"pd_avg")
        elif crit == "nonloc":
            bs_ch = slams_props.read_at_time("xbs_ch",last_time)
            bs_rho = slams_props.read_at_time("xbs_rho",last_time)
            bs_mms = slams_props.read_at_time("xbs_mms",last_time)
            bow_shock_value = np.linalg.norm([bs_ch-bs_rho,bs_rho-bs_mms,bs_mms-bs_ch])

        depths.append(sj_dist_min)
        #indents.append(upstream_dist_min-x_res)
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

    global filenr_g

    outputdir = homedir+"Figures/jh2020/"

    filepath = "/scratch/project_2000203/sunijona/vlasiator/2D/ABC/bulk/bulk.0000677.vlsv"

    filenr_g = 677

    colmap = "parula"
    if var == "Mms":
        colmap = "parula"

    pt.plot.plot_colormap(filename=filepath,outputfile=outputdir+"fig1a_{}.png".format(var),usesci=0,lin=1,expression=expr_list[var_index],vmin=0,vmax=vmax_list[var_index],colormap=colmap,cbtitle=label_list[var_index],pass_vars=["rho","v","CellID","Pdyn","RhoNonBackstream","PTensorNonBackstreamDiagonal","Mms","B"],Earth=1)

    pt.plot.plot_colormap(filename=filepath,outputfile=outputdir+"fig1b_{}.png".format(var),boxre=[6,18,-6,6],usesci=0,lin=1,expression=expr_list[var_index],vmin=0,vmax=vmax_list[var_index],colormap=colmap,cbtitle=label_list[var_index],external=jh20f1_ext,pass_vars=["rho","v","CellID","Pdyn","RhoNonBackstream","PTensorNonBackstreamDiagonal","Mms","B"])

def jh2020_movie(start,stop,var="pdyn"):

    vars_list = ["pdyn","core_heating","rho","Mms","B"]
    var_index = vars_list.index(var)
    label_list = ["nPa","$T_{sw}$","$cm^{-3}$","","nT"]
    vmax_list = [4.5,3.0,6.6,1,10]
    expr_list = [pc.expr_pdyn,pc.expr_coreheating,pc.expr_srho,pc.expr_mms,pc.expr_B]

    global filenr_g

    outputdir = wrkdir_DNR+"jh2020_movie/{}/".format(var)
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    bulkpath = jx.find_bulkpath("ABC")
    for itr in range(start,stop+1):
        filepath = bulkpath+"bulk.{}.vlsv".format(str(itr).zfill(7))
        filenr_g = itr

        colmap = "parula"
        if var == "Mms":
            colmap = "parula"

        pt.plot.plot_colormap(filename=filepath,outputfile=outputdir+"{}.png".format(str(itr).zfill(5)),boxre=[6,18,-6,6],usesci=0,lin=1,expression=expr_list[var_index],vmin=0,vmax=vmax_list[var_index],colormap=colmap,cbtitle=label_list[var_index],external=jh20f1_ext,pass_vars=["rho","v","CellID","Pdyn","RhoNonBackstream","PTensorNonBackstreamDiagonal","Mms","B"])

def jh20f1_ext(ax, XmeshXY,YmeshXY, pass_maps):

    cellids = pass_maps["CellID"]

    slams_cells = jio.eventfile_read("ABC",filenr_g,transient="slams")
    slams_cells = np.array([item for sublist in slams_cells for item in sublist])
    jet_cells = jio.eventfile_read("ABC",filenr_g,transient="jet")
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
        if filenr_g/2.0 in props.read("time"):
            x_list.append(props.read_at_time("x_mean",filenr_g/2.0))
            y_list.append(props.read_at_time("y_mean",filenr_g/2.0))

    bs_fit = jx.bow_shock_jonas("ABC",filenr_g)[::-1]
    mp_fit = jx.mag_pause_jonas("ABC",filenr_g)[::-1]
    y_bs = np.arange(-6,6.01,0.05)
    x_bs = np.polyval(bs_fit,y_bs)
    x_mp = np.polyval(mp_fit,y_bs)

    bs_cont, = ax.plot(x_bs,y_bs,color="black",linewidth=0.8)
    mp_cont, = ax.plot(x_mp,y_bs,color="black",linewidth=0.8)

    slams_cont = ax.contour(XmeshXY,YmeshXY,slams_mask,[0.5],linewidths=0.8,colors=jx.dark_blue)
    jet_cont = ax.contour(XmeshXY,YmeshXY,jet_mask,[0.5],linewidths=0.8,colors=jx.orange)

    xy_pos, = ax.plot(x_list,y_list,"o",color=jx.crimson,markersize=2)

    is_coords = jx.get_cell_coordinates("ABC",1814506)/r_e
    os_coords = jx.get_cell_coordinates("ABC",1794536)/r_e

    is_pos, = ax.plot(is_coords[0],is_coords[1],">",color="black",markersize=2)
    os_pos, = ax.plot(os_coords[0],os_coords[1],"<",color="black",markersize=2)
