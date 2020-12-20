import pytools as pt
import jet_jh2020 as jh20
import jet_io as jio
import jet_analyser as ja
import jet_aux as jx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

wrkdir_DNR = os.environ["WRK"]+"/"
homedir = os.environ["HOME"]+"/"
try:
    vlasdir = os.environ["VLAS"]
except:
    vlasdir="/proj/vlasov"

def fcs_jet_histogram(transient="jet",weight_by_run=False,magt=1.5):

    label_list = ["$\mathrm{\\Delta n~[n_{sw}]}$","$\mathrm{\\Delta |v|~[v_{sw}]}$","$\mathrm{\\Delta P_{dyn}~[P_{dyn,sw}]}$","$\mathrm{\\Delta |B|~[B_{IMF}]}$","$\mathrm{\\Delta T~[T_{sw}]}$","$\mathrm{Lifetime~[s]}$","$\mathrm{Tangential~size~[R_e]}$","$\mathrm{Size~ratio}$"]
    if transient == "slams":
        bins_list = [np.linspace(-1.5,1.5,10+1),np.linspace(-0.1,0.3,10+1),np.linspace(0,2,10+1),np.linspace(-1.5,1.5,10+1),np.linspace(-5,5,10+1),np.linspace(0,60,10+1),np.linspace(0,0.5,10+1),np.linspace(0,5,10+1)]
    else:
        bins_list = [np.linspace(-2,4,10+1),np.linspace(-0.1,0.4,10+1),np.linspace(0,2,10+1),np.linspace(-2,2,10+1),np.linspace(-5,5,10+1),np.linspace(0,60,10+1),np.linspace(0,0.5,10+1),np.linspace(0,5,10+1)]
    pos_list = ["left","left","left","left","right","right","right","right"]

    runids = ["ABA","ABC","AEA","AEC"]

    if weight_by_run:
        jet_counts = [0,0,0,0]

        jet_Dn = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]
        jet_Dv = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]
        jet_Dpd = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]
        jet_DB = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]
        jet_DT = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]

        jet_dur = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]
        jet_tsiz = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]
        jet_srat = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]

        for idx,runid in enumerate(runids):
            sj_ids,non_sj_ids,fcs_ids = jh20.separate_jets_god(runid,False)
            jet_ids = np.union1d(sj_ids,non_sj_ids)
            if transient=="slams":
                jet_ids = fcs_ids
            jet_counts[idx] = jet_ids.size
            for jet_id in jet_ids:
                trans_obj = jio.PropReader(str(jet_id).zfill(5),runid,580,transient=transient)
                jet_Dn[idx] = np.append(jet_Dn[idx],trans_obj.read_at_randt("Dn")/ja.sw_normalisation(runid,"Dn"))
                jet_Dv[idx] = np.append(jet_Dv[idx],trans_obj.read_at_randt("Dv")/ja.sw_normalisation(runid,"Dv"))
                jet_Dpd[idx] = np.append(jet_Dpd[idx],trans_obj.read_at_randt("Dpd")/ja.sw_normalisation(runid,"Dpd"))
                jet_DB[idx] = np.append(jet_DB[idx],trans_obj.read_at_randt("DB")/ja.sw_normalisation(runid,"DB"))
                jet_DT[idx] = np.append(jet_DT[idx],trans_obj.read_at_randt("DT")/ja.sw_normalisation(runid,"DT"))

                jet_dur[idx] = np.append(jet_dur[idx],trans_obj.read_at_randt("duration"))
                jet_tsiz[idx] = np.append(jet_tsiz[idx],trans_obj.read_at_randt("size_tan"))
                jet_srat[idx] = np.append(jet_srat[idx],trans_obj.read_at_randt("size_ratio"))
    else:
        jet_counts = 0

        jet_Dn = np.array([],dtype=float)
        jet_Dv = np.array([],dtype=float)
        jet_Dpd = np.array([],dtype=float)
        jet_DB = np.array([],dtype=float)
        jet_DT = np.array([],dtype=float)

        jet_dur = np.array([],dtype=float)
        jet_tsiz = np.array([],dtype=float)
        jet_srat = np.array([],dtype=float)

        for idx,runid in enumerate(runids):
            sj_ids,non_sj_ids,fcs_ids = jh20.separate_jets_god(runid,False)
            jet_ids = np.union1d(sj_ids,non_sj_ids)
            if transient=="slams":
                jet_ids = fcs_ids

            jet_counts += jet_ids.size
            for jet_id in jet_ids:
                trans_obj = jio.PropReader(str(jet_id).zfill(5),runid,580,transient=transient)
                jet_Dn = np.append(jet_Dn,trans_obj.read_at_randt("Dn")/ja.sw_normalisation(runid,"Dn"))
                jet_Dv = np.append(jet_Dv,trans_obj.read_at_randt("Dv")/ja.sw_normalisation(runid,"Dv"))
                jet_Dpd = np.append(jet_Dpd,trans_obj.read_at_randt("Dpd")/ja.sw_normalisation(runid,"Dpd"))
                jet_DB = np.append(jet_DB,trans_obj.read_at_randt("DB")/ja.sw_normalisation(runid,"DB"))
                jet_DT = np.append(jet_DT,trans_obj.read_at_randt("DT")/ja.sw_normalisation(runid,"DT"))

                jet_dur = np.append(jet_dur,trans_obj.read_at_randt("duration"))
                jet_tsiz = np.append(jet_tsiz,trans_obj.read_at_randt("size_tan"))
                jet_srat = np.append(jet_srat,trans_obj.read_at_randt("size_ratio"))

        weights = np.ones(jet_counts,dtype=float)/float(jet_counts)
        data_arr = np.array([jet_Dn,jet_Dv,jet_Dpd,jet_DB,jet_DT,jet_dur,jet_tsiz,jet_srat])
        #data_meds = np.array([np.nanmedian(arr) for arr in data_arr])
        #data_stds = np.array([np.nanstd(arr,ddof=1) for arr in data_arr])
        data_meds = np.array([np.median(arr) for arr in data_arr])
        data_stds = np.array([np.std(arr,ddof=1) for arr in data_arr])

    fig,ax_list = plt.subplots(4,2,figsize=(7,11))

    ax_flat = ax_list.T.flatten()

    for idx,ax in enumerate(ax_flat):
        ax.hist(data_arr[idx],weights=weights,bins=bins_list[idx],histtype="step",label="med:{:.2f}\nstd:{:.2f}".format(data_meds[idx],data_stds[idx]))
        leg = ax.legend(fontsize=20,frameon=False,markerscale=0.5)
        jx.legend_compact(leg)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3,prune="lower"))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_ylim(0,0.8)
        ax.tick_params(labelsize=15)
        ax.set_ylabel(label_list[idx],labelpad=10,fontsize=20)
        ax.yaxis.set_label_position(pos_list[idx])

    fig.suptitle("transient: {} magt: {}".format(transient,magt),fontsize=24)
    plt.tight_layout()

    fig.savefig(wrkdir_DNR+"Figures/thesis/{}_stats_runweight_{}_magt_{}.png".format(transient,weight_by_run,magt))
    plt.close(fig)
