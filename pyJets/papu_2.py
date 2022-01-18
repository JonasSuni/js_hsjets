import sys
import matplotlib.style
import matplotlib as mpl
import jet_aux as jx
import pytools as pt
import os
import scipy
import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        print("FCS jets: {}".format(sj_id))
        out_arr = []

        # Read jet position, time and filenumber at time of birth
        props = jio.PropReader(str(sj_id).zfill(5), runid, transient="jet")
        x0, y0 = (props.read("x_mean")[0], props.read("y_mean")[0])
        t0 = props.read("time")[0]
        fnr0 = int(t0 * 2)

        # Initialise +-5s array of file numbers
        fnr_arr = np.arange(fnr0 - 10, fnr0 + 11)

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
        print("Non-FCS jets: {}".format(non_id))
        out_arr = []

        # Read jet position, time and filenumber at time of birth
        props = jio.PropReader(str(non_id).zfill(5), runid, transient="jet")
        x0, y0 = (props.read("x_mean")[0], props.read("y_mean")[0])
        t0 = props.read("time")[0]
        fnr0 = int(t0 * 2)

        # Initialise +-5s array of file numbers
        fnr_arr = np.arange(fnr0 - 10, fnr0 + 11)

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
    t_arr = np.arange(-5.0, 5.05, 0.5)

    # Initialise number of fcs-jets and non-fcs-jets
    fcs_jet_count = 0
    non_jet_count = 0

    # Initialise arrays of averages
    fcs_jet_avg = np.zeros((6, 21), dtype=float)
    non_jet_avg = np.zeros((6, 21), dtype=float)

    # Initialise figure, add grids, add axis labels
    fig, ax_list = plt.subplots(6, 1, sharex=True)

    for ax in ax_list:
        ax.grid()

    ax_list[0].set_ylabel("$\\Delta n~[n_\mathrm{sw}]$")
    ax_list[1].set_ylabel("$\\Delta v~[v_\mathrm{sw}]$")
    ax_list[2].set_ylabel("$\\Delta P_\mathrm{dyn}~[P_\mathrm{dyn,sw}]$")
    ax_list[3].set_ylabel("$\\Delta B~[B_\mathrm{IMF}]$")
    ax_list[4].set_ylabel("$\\Delta T_\mathrm{perp}~[T_\mathrm{sw}]$")
    ax_list[5].set_ylabel("$\\Delta T_\mathrm{par}~[T_\mathrm{sw}]$")
    ax_list[-1].set_xlabel("$\\Delta t~[\mathrm{s}]$")

    # Loop over runs
    for runid in ["ABA", "ABC", "AEA", "AEC"]:

        # Loop over arbitrary large number
        for n1 in range(4000):

            # Try reading fcs-jet timeseries
            try:
                data = np.loadtxt(
                    wrkdir_DNR
                    + "papu22/fcs_jets/{}/timeseries_{}.txt".format(runid, n1)
                ).T
                fcs_jet_count += 1  # Iterate fcs-jet count

                # Loop over n,v,pdyn,B,Tperp,Tpar
                for n2 in range(6):

                    # Plot timeseries of deltas relative to epoch time
                    ax_list[n2].plot(
                        t_arr, data[n2] - data[n2][10], color="darkgray", zorder=1
                    )

                    # Add timeseries of deltas relative to epoch time to average array
                    fcs_jet_avg[n2] += data[n2] - data[n2][10]
            except:
                pass

            # Try reading non-fcs-jet timeseries
            try:
                data = np.loadtxt(
                    wrkdir_DNR
                    + "papu22/non_jets/{}/timeseries_{}.txt".format(runid, n1)
                ).T
                non_jet_count += 1  # Iterate fcs-jet count

                # Loop over n,v,pdyn,B,Tperp,Tpar
                for n2 in range(6):

                    # Plot timeseries of deltas relative to epoch time
                    ax_list[n2].plot(
                        t_arr, data[n2] - data[n2][10], color="darkgray", zorder=1
                    )

                    # Add timeseries of deltas relative to epoch time to average array
                    non_jet_avg[n2] += data[n2] - data[n2][10]
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
