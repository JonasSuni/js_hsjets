import analysator as pt
import os
import sys
from random import choice
from copy import deepcopy

from pyJets.jet_aux import CB_color_cycle

from scipy.linalg import eig
from scipy.fft import rfft2
from scipy.signal import butter, sosfilt
from scipy.ndimage import uniform_filter1d
from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from multiprocessing import Pool

mpl.rcParams["hatch.linewidth"] = 0.1

from matplotlib.ticker import MaxNLocator, ScalarFormatter

# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

# import plot_contours as pc
# import jet_analyser as ja
# import jet_io as jio
# import jet_jh2020 as jh20

# mpl.rc("text", usetex=True)
# params = {"text.latex.preamble": [r"\usepackage{amsmath}"]}
# plt.rcParams.update(params)

plt.rcParams.update(
    {
        "ps.useafm": True,
        "pdf.use14corefonts": True,
        "text.usetex": True,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "mathtext.bf": "Helvetica:bold",
        "mathtext.fallback": None,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "text.latex.preamble": r"\usepackage{siunitx} \sisetup{detect-all} \usepackage{helvet} \usepackage{sansmath} \sansmath",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

r_e = 6.371e6
m_p = 1.672621898e-27
q_p = 1.602176634e-19
mu0 = 1.25663706212e-06
kb = 1.380649e-23

wrkdir_DNR = os.environ["WRK"] + "/"
homedir = os.environ["HOME"] + "/"
try:
    vlasdir = os.environ["VLAS"]
except:
    vlasdir = "/proj/vlasov"

try:
    tavgdir = os.environ["TAVG"] + "/"
except:
    tavgdir = wrkdir_DNR + "tavg/"

wrkdir_DNR = wrkdir_DNR + "jets_3D/"
wrkdir_other = os.environ["WRK"] + "/"

bulkpath_FIF = "/wrk-vakka/group/spacephysics/vlasiator/3D/FIF/bulk1/"


def get_msh_VDF_coordinates():

    outdir = wrkdir_DNR + "FIF/msh_vdf_locs/"

    fnr0 = 600
    fnr1 = 991
    boxre = [8, 15, -10, 0, -10, 10]

    for fnr in range(fnr0, fnr1):
        fname = "bulk1.{}.vlsv".format(str(fnr).zfill(7))
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath_FIF + fname)
        cellids = vlsvobj.read_variable("CellID")
        fsaved = vlsvobj.read_variable("vg_f_saved")
        vdf_cellids = cellids[fsaved == 1]
        x, y, z = np.array(
            [vlsvobj.get_cell_coordinates(ci) / r_e for ci in vdf_cellids]
        ).T
        maskx = np.logical_and(x >= boxre[0], x <= boxre[1])
        masky = np.logical_and(y >= boxre[2], y <= boxre[3])
        maskz = np.logical_and(z >= boxre[4], z <= boxre[5])
        mask = np.logical_and(maskx, np.logical_and(masky, maskz))

        outarr = np.array([vdf_cellids[mask], x[mask], y[mask], z[mask]])
        np.savetxt(outdir + "{}.txt".format(fnr), outarr.T)


def process_timestep_VSC_timeseries(args):
    """Helper function for parallel processing in VSC_timeseries"""
    (
        fnr,
        var_list,
        scales,
        bulkpath,
        ops,
        x0,
        y0,
        z0,
    ) = args
    try:
        result = np.zeros(len(var_list) + 7, dtype=float)
        vlsvobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk1.{}.vlsv".format(str(fnr).zfill(7))
        )
        for idx2, var in enumerate(var_list):
            result[idx2] = (
                vlsvobj.read_interpolated_variable(
                    var, [x0 * r_e, y0 * r_e, z0 * r_e], operator=ops[idx2]
                )
                * scales[idx2]
            )
        return fnr, result
    except Exception as e:
        print(f"Error processing timestep {fnr}: {str(e)}")
        return fnr, None


def VSC_timeseries(
    runid,
    cellid,
    coords,
    t0,
    t1,
    pdx=False,
    delta=None,
    vlines=[],
    fmt="-",
    dirprefix="",
    skip=False,
    fromtxt=False,
    jett0=0.0,
    n_processes=1,
    draw=True,
):
    bulkpath = bulkpath_FIF
    x0, y0, z0 = coords

    figdir = wrkdir_DNR + "Figs/timeseries/{}".format(dirprefix)
    txtdir = wrkdir_DNR + "txts/timeseries/{}".format(dirprefix)
    if not os.path.exists(figdir):
        try:
            os.makedirs(figdir)
        except OSError:
            pass
    if not os.path.exists(txtdir):
        try:
            os.makedirs(txtdir)
        except OSError:
            pass
    if skip and os.path.isfile(
        figdir
        + "{}_x{:.3f}_y{:.3f}_z{:.3f}_t0{}_t1{}_delta{}.png".format(
            runid, x0, y0, z0, t0, t1, delta
        )
    ):
        print("Skip is True and file already exists, exiting.")
        return None

    var_list = [
        "proton/vg_rho",
        "proton/vg_v",
        "proton/vg_v",
        "proton/vg_v",
        "proton/vg_v",
        "proton/vg_Pdyn",
        "vg_b_vol",
        "vg_b_vol",
        "vg_b_vol",
        "vg_b_vol",
        "vg_e_vol",
        "vg_e_vol",
        "vg_e_vol",
        "vg_e_vol",
        "proton/vg_t_parallel",
        "proton/vg_t_perpendicular",
    ]
    plot_labels = [
        None,
        "$v_x$",
        "$v_y$",
        "$v_z$",
        "$|v|$",
        "$P_\\mathrm{dyn}$",
        "$B_x$",
        "$B_y$",
        "$B_z$",
        "$|B|$",
        "$E_x$",
        "$E_y$",
        "$E_z$",
        "$|E|$",
        "$T_\\parallel$",
        "$T_\\perp$",
    ]
    scales = [
        1e-6,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        1e9,
        1e9,
        1e9,
        1e9,
        1e9,
        1e3,
        1e3,
        1e3,
        1e3,
        1e-6,
        1e-6,
    ]
    draw_legend = [
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        True,
    ]
    ylabels = [
        "$\\rho~[\\mathrm{cm}^{-3}]$",
        "$v~[\\mathrm{km/s}]$",
        "$P_\\mathrm{dyn}~[\\mathrm{nPa}]$",
        "$B~[\\mathrm{nT}]$",
        "$E~[\\mathrm{mV/m}]$",
        "$T~[\\mathrm{MK}]$",
    ]
    if delta:
        for idx in range(len(ylabels)):
            ylabels[idx] = "$\\delta " + ylabels[idx][1:]
    e_sw = 750e3 * 5e-9 * q_p / m_p * 1e3
    pdsw_npa = m_p * 1e6 * 750e3 * 750e3 / 1e-9
    ops = [
        "pass",
        "x",
        "y",
        "z",
        "magnitude",
        "pass",
        "x",
        "y",
        "z",
        "magnitude",
        "x",
        "y",
        "z",
        "magnitude",
        "pass",
        "pass",
    ]
    plot_index = [0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5]
    plot_colors = [
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        "k",
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
    ]

    t_arr = np.arange(t0, t1 + 0.1, 1)
    fnr0 = int(t0)
    fnr_arr = np.arange(fnr0, int(t1) + 1, dtype=int)
    data_arr = np.zeros((len(var_list) + 7, fnr_arr.size), dtype=float)
    tavg_arr = np.zeros(fnr_arr.size, dtype=float)
    tavg_x_arr = np.zeros(fnr_arr.size, dtype=float)

    if fromtxt:
        data_arr = np.loadtxt(
            txtdir
            + "{}_x{:.3f}_y{:.3f}_z{:.3f}_t0{}_t1{}_delta{}.txt".format(
                runid, x0, y0, z0, t0, t1, delta
            )
        )
        tavg_arr = data_arr[-1, :]
    else:
        # Prepare arguments for parallel processing
        args_list = [
            (
                fnr,
                var_list,
                scales,
                bulkpath,
                ops,
                x0,
                y0,
                z0,
            )
            for fnr in fnr_arr
        ]

        # Use multiprocessing Pool

        with Pool(processes=n_processes) as pool:
            results = pool.map(process_timestep_VSC_timeseries, args_list)

            # Process results
            for fnr, result in results:
                if result is not None:
                    idx = np.where(fnr_arr == fnr)[0][0]
                    data_arr[:, idx] = result

    tavg_arr = uniform_filter1d(
        data_arr[5, :], 180, mode="constant", cval=np.nanmean(data_arr[5, :])
    )
    pdynx = m_p * data_arr[0] * 1e6 * data_arr[1] * 1e3 * data_arr[1] * 1e3 * 1e9
    tavg_x_arr = uniform_filter1d(pdynx, 180, mode="constant", cval=np.nanmean(pdynx))

    if draw:
        fig, ax_list = plt.subplots(
            len(ylabels) + 1, 1, sharex=True, figsize=(7, 9), constrained_layout=True
        )
        ax_list[0].set_title(
            "Run: {}, $x_0$: {:.3f}, $y_0$: {:.3f}, $z_0$: {:.3f}, cell: {}".format(
                runid, x0, y0, z0, int(cellid)
            )
        )
        for idx in range(len(var_list)):
            ax = ax_list[plot_index[idx]]
            for vline in vlines:
                ax.axvline(vline, linestyle="dashed", linewidth=0.6)
            if delta:
                ax.plot(
                    t_arr,
                    data_arr[idx] - uniform_filter1d(data_arr[idx], size=delta),
                    fmt,
                    color=plot_colors[idx],
                    label=plot_labels[idx],
                )
            else:
                ax.plot(
                    t_arr,
                    data_arr[idx],
                    fmt,
                    color=plot_colors[idx],
                    label=plot_labels[idx],
                )
            if idx == 5 and True and not delta:
                ax.plot(
                    t_arr,
                    2 * tavg_arr,
                    color=CB_color_cycle[1],
                    linestyle="dashed",
                    label="$2\\langle P_\\mathrm{dyn}\\rangle$",
                )
                # ax.axhline(
                #     0.5 * pdsw_npa,
                #     color=CB_color_cycle[2],
                #     linestyle="dotted",
                #     label="$0.5P_\\mathrm{dyn,sw}$",
                # )
                # ax.axhline(
                #     0.25 * pdsw_npa,
                #     color=CB_color_cycle[3],
                #     linestyle="dotted",
                #     label="$0.25P_\\mathrm{dyn,sw}$",
                # )
            if idx == 5 and pdx:
                if delta:
                    ax.plot(
                        t_arr,
                        pdynx - uniform_filter1d(pdynx, size=delta),
                        fmt,
                        color=CB_color_cycle[0],
                        label="$P_{\\mathrm{dyn},x}$",
                    )
                else:
                    ax.plot(
                        t_arr,
                        pdynx,
                        fmt,
                        color=CB_color_cycle[0],
                        label="$P_{\\mathrm{dyn},x}$",
                    )
                    ax.plot(
                        t_arr,
                        3 * tavg_x_arr,
                        color=CB_color_cycle[2],
                        linestyle="dashed",
                        label="$3\\langle P_{\\mathrm{dyn},x}\\rangle$",
                    )

            ax.set_xlim(t_arr[0]+90, t_arr[-1]-90)
            if draw_legend[idx]:
                ncols = 1
                if idx == 5:
                    ncols = 1
                ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncols=ncols)
        ylabels.append("$P_\\mathrm{dyn}$\ncontribution")

    rho_lp = m_p * data_arr[0, :] * 1e6
    vx_lp = data_arr[1, :] * 1e3
    vy_lp = data_arr[2, :] * 1e3
    vz_lp = data_arr[3, :] * 1e3
    vt_lp = data_arr[4, :] * 1e3
    pd_lp = data_arr[5, :] * 1e-9

    rho_term = rho_lp * np.nanmean(vt_lp**2) / np.nanmean(pd_lp)
    vx_term = np.nanmean(rho_lp) * vx_lp**2 / np.nanmean(pd_lp)
    vy_term = np.nanmean(rho_lp) * vy_lp**2 / np.nanmean(pd_lp)
    vz_term = np.nanmean(rho_lp) * vz_lp**2 / np.nanmean(pd_lp)

    data_arr[-7, :] = rho_term
    data_arr[-6, :] = vx_term
    data_arr[-5, :] = vy_term
    data_arr[-4, :] = vz_term
    data_arr[-3, :] = tavg_arr
    data_arr[-2, :] = t_arr
    data_arr[-1, :] = np.ones_like(t_arr) * jett0

    if draw:
        ax_list[-1].plot(t_arr, rho_term, color="black", label="$\\rho$")
        ax_list[-1].plot(t_arr, vx_term, color=CB_color_cycle[0], label="$v_x^2$")
        ax_list[-1].plot(t_arr, vy_term, color=CB_color_cycle[1], label="$v_y^2$")
        ax_list[-1].plot(t_arr, vz_term, color=CB_color_cycle[2], label="$v_z^2$")

        ax_list[-1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncols=1)
        for vline in vlines:
            ax_list[-1].axvline(vline, linestyle="dashed", linewidth=0.6)
        ax_list[-1].set_xlim(t_arr[0], t_arr[-1])
        ax_list[-1].set_xlabel("Simulation time [s]")
        for idx, ax in enumerate(ax_list):
            ax.grid()
            ax.set_ylabel(ylabels[idx])
            ax.axvline(t0, linestyle="dashed")
            if True:
                ax.fill_between(
                    t_arr,
                    0,
                    1,
                    where=data_arr[5, :] > 2 * tavg_arr,
                    color="red",
                    alpha=0.2,
                    transform=ax.get_xaxis_transform(),
                    linewidth=0,
                )
                ax.fill_between(
                    t_arr,
                    0,
                    1,
                    where=pdynx > 3 * tavg_x_arr,
                    color="green",
                    alpha=0.2,
                    transform=ax.get_xaxis_transform(),
                    linewidth=0,
                )

        fig.savefig(
            figdir
            + "{}_x{:.3f}_y{:.3f}_z{:.3f}_t0{}_t1{}_delta{}.png".format(
                runid, x0, y0, z0, t0, t1, delta
            ),
            dpi=300,
        )
        plt.close(fig)
    np.savetxt(
        txtdir
        + "{}_x{:.3f}_y{:.3f}_z{:.3f}_t0{}_t1{}_delta{}.txt".format(
            runid, x0, y0, z0, t0, t1, delta
        ),
        data_arr,
    )


def L3_vdf_timeseries(n_processes=16, skip=False, fromtxt=False):

    fnr0 = 600
    fnr1 = 991

    data_600 = np.loadtxt(wrkdir_DNR + "FIF/msh_vdf_locs/600.txt")
    cellids = data_600.T[0]
    vobj_600 = pt.vlsvfile.VlsvReader(bulkpath_FIF + "bulk1.0000600.vlsv")

    for ci in cellids:
        refl = vobj_600.read_variable("vg_reflevel", cellids=int(ci))
        print(refl)
        if refl != 3:
            continue

        coords = vobj_600.get_cell_coordinates(ci) / r_e
        try:
            VSC_timeseries(
                "FIF",
                ci,
                coords,
                fnr0,
                fnr1,
                pdx=True,
                delta=None,
                vlines=[],
                fmt="-",
                dirprefix="",
                skip=skip,
                fromtxt=fromtxt,
                jett0=0.0,
                n_processes=n_processes,
                draw=True,
            )
        except:
            pass
