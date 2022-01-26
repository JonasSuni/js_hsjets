import sys
import matplotlib.style
import matplotlib as mpl
import jet_aux as jx
from pyJets.jet_aux import CB_color_cycle
import pytools as pt
import os
import scipy
import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D

import plot_contours as pc
import jet_analyser as ja
import jet_io as jio
import jet_jh2020 as jh20

# mpl.rc("text", usetex=True)
# params = {"text.latex.preamble": [r"\usepackage{amsmath}"]}
# plt.rcParams.update(params)

r_e = 6.371e6
m_p = 1.672621898e-27

wrkdir_DNR = os.environ["WRK"] + "/"
homedir = os.environ["HOME"] + "/"
try:
    vlasdir = os.environ["VLAS"]
except:
    vlasdir = "/proj/vlasov"


def sj_non_timeseries(runid):
    """
    Variables: t, n, v, Pdyn, B, Tperp, Tpar
    Count: 7
    """

    # Solar wind parameters for the different runs
    # n [m^-3], v [m/s], B [T], T [K]
    runid_list = ["ABA", "ABC", "AEA", "AEC"]
    sw_pars = [
        [1e6, 750e3, 5e-9, 0.5e6],
        [3.3e6, 600e3, 5e-9, 0.5e6],
        [1e6, 750e3, 10e-9, 0.5e6],
        [3.3e6, 600e3, 10e-9, 0.5e6],
    ]
    n_sw, v_sw, B_sw, T_sw = sw_pars[runid_list.index(runid)]

    # Path to vlsv files for current run
    bulkpath = jx.find_bulkpath(runid)

    # Get IDs of fcs-jets and non-fcs-jets
    sj_jet_ids, jet_ids, slams_ids = jh20.separate_jets_god(runid, False)
    non_sj_ids = jet_ids[np.in1d(jet_ids, sj_jet_ids) == False]

    # Loop through fcs-jet IDs
    for sj_id in sj_jet_ids:
        print("FCS jets for run {}: {}".format(runid, sj_id))
        out_arr = []

        # Read jet position, time and filenumber at time of birth
        props = jio.PropReader(str(sj_id).zfill(5), runid, transient="jet")
        x0, y0 = (props.read("x_mean")[0], props.read("y_mean")[0])
        t0 = props.read("time")[0]
        fnr0 = int(t0 * 2)

        # Initialise +-10s array of file numbers
        fnr_arr = np.arange(fnr0 - 20, fnr0 + 21)

        # Get cellid of initial position
        cellid = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
        ).get_cellid([x0 * r_e, y0 * r_e, 0 * r_e])

        # Loop through filenumbers in time series
        for fnr in fnr_arr:

            # Try if file exists
            try:
                vlsvobj = pt.vlsvfile.VlsvReader(
                    bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
                )

                # Read time, n, v, Pdyn, B, Tperp, Tpar
                t = float(fnr) / 2.0
                n = vlsvobj.read_variable("rho", cellids=cellid)
                v = vlsvobj.read_variable("v", cellids=cellid, operator="magnitude")
                Pdyn = m_p * n * v * v
                B = vlsvobj.read_variable("B", cellids=cellid, operator="magnitude")
                Tperp = vlsvobj.read_variable("TPerpendicular", cellids=cellid)
                Tpar = vlsvobj.read_variable("TParallel", cellids=cellid)

                # Append normalised properties to array
                out_arr.append(
                    np.array(
                        [
                            t,
                            n / n_sw,
                            v / v_sw,
                            Pdyn / (m_p * n_sw * v_sw * v_sw),
                            B / B_sw,
                            Tperp / T_sw,
                            Tpar / T_sw,
                        ]
                    )
                )
            except:
                # If vlsv file doesn't exist, append nans
                out_arr.append(
                    np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                )

        out_arr = np.array(out_arr)

        # Save time series as txt file
        np.savetxt(
            wrkdir_DNR + "papu22/fcs_jets/{}/timeseries_{}.txt".format(runid, sj_id),
            out_arr,
            fmt="%.7f",
        )

    # Loop through non-fcs-jet IDs
    for non_id in non_sj_ids:
        print("Non-FCS jets for run {}: {}".format(runid, non_id))
        out_arr = []

        # Read jet position, time and filenumber at time of birth
        props = jio.PropReader(str(non_id).zfill(5), runid, transient="jet")
        x0, y0 = (props.read("x_mean")[0], props.read("y_mean")[0])
        t0 = props.read("time")[0]
        fnr0 = int(t0 * 2)

        # Initialise +-10s array of file numbers
        fnr_arr = np.arange(fnr0 - 20, fnr0 + 21)

        # Get cellid of initial position
        cellid = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
        ).get_cellid([x0 * r_e, y0 * r_e, 0 * r_e])

        # Loop through filenumbers in time series
        for fnr in fnr_arr:

            # Try if file exists
            try:
                vlsvobj = pt.vlsvfile.VlsvReader(
                    bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
                )

                # Read time, n, v, Pdyn, B, Tperp, Tpar
                t = float(fnr) / 2.0
                n = vlsvobj.read_variable("rho", cellids=cellid)
                v = vlsvobj.read_variable("v", cellids=cellid, operator="magnitude")
                Pdyn = m_p * n * v * v
                B = vlsvobj.read_variable("B", cellids=cellid, operator="magnitude")
                Tperp = vlsvobj.read_variable("TPerpendicular", cellids=cellid)
                Tpar = vlsvobj.read_variable("TParallel", cellids=cellid)

                # Append normalised properties to array
                out_arr.append(
                    np.array(
                        [
                            t,
                            n / n_sw,
                            v / v_sw,
                            Pdyn / (m_p * n_sw * v_sw * v_sw),
                            B / B_sw,
                            Tperp / T_sw,
                            Tpar / T_sw,
                        ]
                    )
                )
            except:
                # If vlsv file doesn't exist, append nans
                out_arr.append(
                    np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                )

        out_arr = np.array(out_arr)

        # Save time series as txt file
        np.savetxt(
            wrkdir_DNR + "papu22/non_jets/{}/timeseries_{}.txt".format(runid, non_id),
            out_arr,
            fmt="%.7f",
        )


def SEA_plots():
    """
    Superposed epoch analysis of fcs-jet vs. non-fcs-jet start location properties
    """

    # Initialise array of times relative to epoch time
    t_arr = np.arange(-10.0, 10.05, 0.5)

    # Initialise number of fcs-jets and non-fcs-jets
    fcs_jet_count = 0
    non_jet_count = 0

    # Initialise arrays of averages
    fcs_jet_avg = np.zeros((6, 41), dtype=float)
    non_jet_avg = np.zeros((6, 41), dtype=float)

    # Initialise figure, add grids, add axis labels
    fig, ax_list = plt.subplots(6, 1, sharex=True, figsize=(8, 10))

    for ax in ax_list:
        ax.grid()

    ax_list[0].set_ylabel("$\\Delta n~[n_\mathrm{sw}]$")
    ax_list[1].set_ylabel("$\\Delta v~[v_\mathrm{sw}]$")
    ax_list[2].set_ylabel("$\\Delta P_\mathrm{dyn}~[P_\mathrm{dyn,sw}]$")
    ax_list[3].set_ylabel("$\\Delta B~[B_\mathrm{IMF}]$")
    ax_list[4].set_ylabel("$\\Delta T_\mathrm{perp}~[T_\mathrm{sw}]$")
    ax_list[5].set_ylabel("$\\Delta T_\mathrm{par}~[T_\mathrm{sw}]$")
    ax_list[-1].set_xlabel("$\\Delta t~[\mathrm{s}]$")

    # ax_list[0].set_ylabel("delta n")
    # ax_list[1].set_ylabel("delta v")
    # ax_list[2].set_ylabel("delta Pdyn")
    # ax_list[3].set_ylabel("delta B")
    # ax_list[4].set_ylabel("delta Tperp")
    # ax_list[5].set_ylabel("delta Tpar")
    # ax_list[-1].set_xlabel("delta t")

    # Loop over runs
    for runid in ["ABA", "ABC", "AEA", "AEC"]:

        # Loop over arbitrary large number
        for n1 in range(4000):

            # Try reading fcs-jet timeseries
            try:
                data = np.loadtxt(
                    wrkdir_DNR
                    + "papu22/fcs_jets/{}/timeseries_{}.txt".format(runid, n1)
                ).T[1:]
                fcs_jet_count += 1  # Iterate fcs-jet count

                # Loop over n,v,pdyn,B,Tperp,Tpar
                for n2 in range(6):

                    # Plot timeseries of deltas relative to epoch time
                    ax_list[n2].plot(
                        t_arr, data[n2] - data[n2][20], color="darkgray", zorder=0
                    )

                    # Add timeseries of deltas relative to epoch time to average array
                    fcs_jet_avg[n2] += data[n2] - data[n2][20]
            except:
                pass

            # Try reading non-fcs-jet timeseries
            try:
                data = np.loadtxt(
                    wrkdir_DNR
                    + "papu22/non_jets/{}/timeseries_{}.txt".format(runid, n1)
                ).T[1:]
                non_jet_count += 1  # Iterate fcs-jet count

                # Loop over n,v,pdyn,B,Tperp,Tpar
                for n2 in range(6):

                    # Plot timeseries of deltas relative to epoch time
                    ax_list[n2].plot(
                        t_arr, data[n2] - data[n2][20], color="lightgray", zorder=1
                    )

                    # Add timeseries of deltas relative to epoch time to average array
                    non_jet_avg[n2] += data[n2] - data[n2][20]
            except:
                pass

    # Calculate averages
    fcs_jet_avg /= fcs_jet_count
    non_jet_avg /= non_jet_count

    # Plot averages of n,v,pdyn,B,Tperp,Tpar
    for n2 in range(6):
        ax_list[n2].plot(
            t_arr,
            fcs_jet_avg[n2],
            color=jx.CB_color_cycle[0],
            label="FCS-jets",
            zorder=2,
        )
        ax_list[n2].plot(
            t_arr,
            non_jet_avg[n2],
            color=jx.CB_color_cycle[1],
            label="non-FCS-jets",
            zorder=2,
        )

    # Add legend
    ax_list[0].legend()

    # Save as pdf and png and close figure
    plt.tight_layout()

    fig.savefig(wrkdir_DNR + "papu22/Figures/SEA_plot.pdf")
    fig.savefig(wrkdir_DNR + "papu22/Figures/SEA_plot.png")
    plt.close(fig)


def fcs_non_jet_hist(lastbs=False):

    # Solar wind parameters for the different runs
    # n [m^-3], v [m/s], B [T], T [K]
    runid_list = ["ABA", "ABC", "AEA", "AEC"]
    sw_pars = [
        [1, 750, 5, 0.5],
        [3.3, 600, 5, 0.5],
        [1, 750, 10, 0.5],
        [3.3, 600, 10, 0.5],
    ]

    # Initialise arrays for variables to be read and their figure labels, histogram bins and label positions
    # delta n, delta v, delta Pdyn, delta B, delta T, Lifetime, Tangential size, Size ratio
    # Count: 8
    vars_list = ["Dn", "Dv", "Dpd", "DB", "DT", "duration", "size_tan", "size_ratio"]
    label_list = [
        "$\\Delta n~[n_\mathrm{sw}]$",
        "$\\Delta |v|~[v_\mathrm{sw}]$",
        "$\\Delta P_\mathrm{dyn}~[P_\mathrm{dyn,sw}]$",
        "$\\Delta |B|~[B_\mathrm{IMF}]$",
        "$\\Delta T~[T_\mathrm{sw}]$",
        "$Lifetime~[\mathrm{s}]}$",
        "$Tangential~size~[R_\mathrm{E}]$",
        "$Size~ratio$",
    ]
    bins_list = [
        np.linspace(-2, 5, 10 + 1),
        np.linspace(-0.1, 0.6, 10 + 1),
        np.linspace(0, 3, 10 + 1),
        np.linspace(-2, 3, 10 + 1),
        np.linspace(-10, 5, 10 + 1),
        np.linspace(0, 60, 10 + 1),
        np.linspace(0, 0.5, 10 + 1),
        np.linspace(0, 5, 10 + 1),
    ]
    pos_list = ["left", "left", "left", "left", "right", "right", "right", "right"]
    fcs_jet_props = [[], [], [], [], [], [], [], []]
    non_jet_props = [[], [], [], [], [], [], [], []]

    # Loop over runs
    for runid in ["ABA", "ABC", "AEA", "AEC"]:

        # Get solar wind values and make normalisation array
        n_sw, v_sw, B_sw, T_sw = sw_pars[runid_list.index(runid)]
        sw_norm = [
            n_sw,
            v_sw,
            m_p * n_sw * 1e6 * v_sw * 1e3 * v_sw * 1e3 * 1e9,
            B_sw,
            T_sw,
            1,
            1,
            1,
        ]

        # Get IDs of fcs-jets and non-fcs-jets
        sj_jet_ids, jet_ids, slams_ids = jh20.separate_jets_god(runid, False)
        non_sj_ids = jet_ids[np.in1d(jet_ids, sj_jet_ids) == False]

        # Loop over fcs-jets
        for sj_id in sj_jet_ids:

            # Read properties
            props = jio.PropReader(str(sj_id).zfill(5), runid, transient="jet")

            # Loop over variables
            for n1 in range(8):

                # Should properties be taken at last time at bow shock...
                if lastbs:
                    fcs_jet_props[n1].append(
                        props.read_at_lastbs(vars_list[n1]) / sw_norm[n1]
                    )

                # ...or at the time of maximum area?
                else:
                    fcs_jet_props[n1].append(
                        props.read_at_amax(vars_list[n1]) / sw_norm[n1]
                    )

        # Loop over non-fcs-jets
        for non_id in non_sj_ids:

            # Read properties
            props = jio.PropReader(str(non_id).zfill(5), runid, transient="jet")

            # Loop over variables
            for n1 in range(8):

                # Should properties be taken at last time at bow shock...
                if lastbs:
                    non_jet_props[n1].append(
                        props.read_at_lastbs(vars_list[n1]) / sw_norm[n1]
                    )

                # ...or at the time of maximum area?
                else:
                    non_jet_props[n1].append(
                        props.read_at_amax(vars_list[n1]) / sw_norm[n1]
                    )

    # Make figure
    fig, ax_list = plt.subplots(4, 2, figsize=(7, 11))

    ax_flat = ax_list.T.flatten()

    for n1, ax in enumerate(ax_flat):
        ax.set_ylabel(label_list[n1], labelpad=10, fontsize=20)
        ax.yaxis.set_label_position(pos_list[n1])
        ax.grid()
        ax.tick_params(labelsize=15)

        ax.hist(
            fcs_jet_props[n1],
            histtype="step",
            weights=np.ones(len(fcs_jet_props[n1])) / float(len(fcs_jet_props[n1])),
            bins=bins_list[n1],
            color=jx.CB_color_cycle[0],
            label="FCS-jets",
        )
        ax.hist(
            non_jet_props[n1],
            histtype="step",
            weights=np.ones(len(non_jet_props[n1])) / float(len(non_jet_props[n1])),
            bins=bins_list[n1],
            color=jx.CB_color_cycle[1],
            label="non-FCS-jets",
        )

    ax_flat[5].legend(frameon=False, markerscale=0.5)

    # Save figure
    plt.tight_layout()

    fig.savefig(wrkdir_DNR + "papu22/Figures/FCS_non_hist_lastbs_{}.pdf".format(lastbs))
    fig.savefig(wrkdir_DNR + "papu22/Figures/FCS_non_hist_lastbs_{}.png".format(lastbs))
    plt.close(fig)


def papu22_mov_script(runid):

    sj_jet_ids, jet_ids, slams_ids = jh20.separate_jets_god(runid, False)
    non_sj_ids = jet_ids[np.in1d(jet_ids, sj_jet_ids) == False]

    runid_list = ["ABA", "ABC", "AEA", "AEC"]
    stop_list = [839, 1179, 879, 1339]

    for filenr in range(580, stop_list[runid_list.index(runid)] + 1):
        colormap_with_contours(runid, filenr, sj_ids=sj_jet_ids, non_ids=non_sj_ids)


def colormap_with_contours(runid, filenr, sj_ids=[], non_ids=[]):

    global runid_g, filenr_g, sj_ids_g, non_ids_g
    runid_g = runid
    filenr_g = filenr
    sj_ids_g = sj_ids
    non_ids_g = non_ids

    # Path to vlsv files for current run
    bulkpath = jx.find_bulkpath(runid)
    bulkname = "bulk.{}.vlsv".format(str(filenr).zfill(7))

    # Solar wind parameters for the different runs
    # n [m^-3], v [m/s], B [T], T [K]
    runid_list = ["ABA", "ABC", "AEA", "AEC"]
    sw_pars = [
        [1e6, 750e3, 5e-9, 0.5e6],
        [3.3e6, 600e3, 5e-9, 0.5e6],
        [1e6, 750e3, 10e-9, 0.5e6],
        [3.3e6, 600e3, 10e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runid_list.index(runid)]
    vmax = [1.5, 3.0, 1.5, 3.0][runid_list.index(runid)]
    if runid in ["ABA", "AEA"]:
        boxre = [6, 18, -8, 6]
    else:
        boxre = [6, 18, -6, 6]

    pt.plot.plot_colormap(
        filename=bulkpath + bulkname,
        outputfile=wrkdir_DNR
        + "papu22/Figures/{}/contours_{}.png".format(runid, str(filenr).zfill(7)),
        boxre=boxre,
        usesci=0,
        lin=1,
        var="Pdyn",
        tickinterval=1,
        vmin=0,
        vmax=vmax,
        wmark=True,
        vscale=1e9,
        colormap="Greys",
        external=ext_contours,
        pass_vars=[
            "RhoNonBackstream",
            "PTensorNonBackstreamDiagonal",
            "B",
            "v",
            "rho",
            "core_heating",
            "CellID",
            "Mmsx",
        ],
        title="$t~=~$ {:.1f} ".format(filenr_g / 2.0) + "$~\mathrm{s}$",
        cbtitle="$P_\mathrm{dyn}~[\mathrm{nPa}]$",
    )


def ext_contours(ax, XmeshXY, YmeshXY, pass_maps):

    B = pass_maps["B"]
    rho = pass_maps["rho"]
    cellids = pass_maps["CellID"]
    mmsx = pass_maps["Mmsx"]
    core_heating = pass_maps["core_heating"]
    Bmag = np.linalg.norm(B, axis=-1)

    slams_cells = np.loadtxt(
        "/wrk/users/jesuni/working/SLAMS/Masks/{}/{}.mask".format(runid_g, filenr_g)
    ).astype(int)
    jet_cells = np.loadtxt(
        "/wrk/users/jesuni/working/jets/Masks/{}/{}.mask".format(runid_g, filenr_g)
    ).astype(int)

    sj_jetobs = [
        jio.PropReader(str(sj_id).zfill(5), runid_g, transient="jet")
        for sj_id in sj_ids_g
    ]
    non_sjobs = [
        jio.PropReader(str(non_id).zfill(5), runid_g, transient="jet")
        for non_id in non_ids_g
    ]

    sj_xlist = []
    sj_ylist = []
    non_xlist = []
    non_ylist = []

    for jetobj in sj_jetobs:
        if filenr_g / 2.0 in jetobj.read("time"):
            sj_xlist.append(jetobj.read_at_time("x_mean", filenr_g / 2.0))
            sj_ylist.append(jetobj.read_at_time("y_mean", filenr_g / 2.0))
    for jetobj in non_sjobs:
        if filenr_g / 2.0 in jetobj.read("time"):
            non_xlist.append(jetobj.read_at_time("x_mean", filenr_g / 2.0))
            non_ylist.append(jetobj.read_at_time("y_mean", filenr_g / 2.0))

    slams_mask = np.in1d(cellids, slams_cells).astype(int)
    slams_mask = np.reshape(slams_mask, cellids.shape)

    jet_mask = np.in1d(cellids, jet_cells).astype(int)
    jet_mask = np.reshape(jet_mask, cellids.shape)

    ch_mask = (core_heating > 3 * 0.5e6).astype(int)
    mach_mask = (mmsx < 1).astype(int)
    rho_mask = (rho > 2 * rho_sw).astype(int)

    cav_shfa_mask = (Bmag < 0.8 * B_sw).astype(int)
    cav_shfa_mask[rho >= 0.8 * rho_sw] = 0

    CB_color_cycle = jx.CB_color_cycle

    jet_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        jet_mask,
        [0.5],
        linewidths=0.6,
        colors=CB_color_cycle[0],
        linestyles=["solid"],
    )

    ch_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        ch_mask,
        [0.5],
        linewidths=0.6,
        colors=CB_color_cycle[1],
        linestyles=["solid"],
    )

    slams_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        slams_mask,
        [0.5],
        linewidths=0.6,
        colors=CB_color_cycle[2],
        linestyles=["solid"],
    )

    rho_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        rho_mask,
        [0.5],
        linewidths=0.6,
        colors=CB_color_cycle[3],
        linestyles=["solid"],
    )

    mach_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        mach_mask,
        [0.5],
        linewidths=0.6,
        colors=CB_color_cycle[4],
        linestyles=["solid"],
    )

    cav_shfa_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        cav_shfa_mask,
        [0.5],
        linewidths=0.6,
        colors=CB_color_cycle[5],
        linestyles=["solid"],
    )

    (non_pos,) = ax.plot(
        non_xlist,
        non_ylist,
        "o",
        color="black",
        markersize=2,
        markeredgecolor="white",
        fillstyle="full",
        mew=0.2,
        label="Non-FCS-jet",
    )
    (sj_pos,) = ax.plot(
        sj_xlist,
        sj_ylist,
        "o",
        color="red",
        markersize=2,
        markeredgecolor="white",
        fillstyle="full",
        mew=0.2,
        label="FCS-jet",
    )

    # print(jet_cont.collections[0])

    # jet_cont.collections[0].set_label("Jet")
    # ch_cont.collections[0].set_label("BS CH")
    # slams_cont.collections[0].set_label("FCS")
    # rho_cont.collections[0].set_label("BS rho")
    # mach_cont.collections[0].set_label("BS Mmsx")
    # cav_shfa_cont.collections[0].set_label("Cav/SHFA")

    jet_line = Line2D([0], [0], linestyle="none", color=CB_color_cycle[0])
    ch_line = Line2D([0], [0], linestyle="none", color=CB_color_cycle[1])
    slams_line = Line2D([0], [0], linestyle="none", color=CB_color_cycle[2])
    rho_line = Line2D([0], [0], linestyle="none", color=CB_color_cycle[3])
    mach_line = Line2D([0], [0], linestyle="none", color=CB_color_cycle[4])
    cav_shfa_line = Line2D([0], [0], linestyle="none", color=CB_color_cycle[5])

    proxy = [plt.Rectangle((0, 0), 1, 1, fc=CB_color_cycle[itr]) for itr in range(6)]

    ax.legend(
        # (jet_line, ch_line, slams_line, rho_line, mach_line, cav_shfa_line),
        proxy,
        ("Jet", "BS CH", "FCS", "BS rho", "BS Mmsx", "Cav/SHFA"),
        frameon=True,
        numpoints=1,
        markerscale=1,
        loc="upper right",
        fontsize=5,
    )
