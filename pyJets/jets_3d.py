import analysator as pt
import os, subprocess
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
from matplotlib.animation import FuncAnimation, FFMpegFileWriter

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

            ax.set_xlim(t_arr[0] + 90, t_arr[-1] - 90)
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
        ax_list[-1].set_xlim(t_arr[0] + 90, t_arr[-1] - 90)
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


def L3_good_timeseries_global_vdfs_all():

    cellids, t0, t1 = np.loadtxt(
        wrkdir_DNR + "FIF/good_jet_intervals_1.txt", dtype=int
    ).T
    vobj_600 = pt.vlsvfile.VlsvReader(bulkpath_FIF + "bulk1.0000600.vlsv")

    for idx in range(len(cellids)):
        coords = vobj_600.get_cell_coordinates(cellids[idx]) / r_e

        outdir = "/tmp/FIF/{}".format(idx)

        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except OSError:
                pass

        make_timeseries_global_vdf_anim(
            cellids[idx], coords, t0[idx], t1[idx], outdir=outdir
        )


def L3_good_timeseries_global_vdfs_one(
    idx, limitedsize=True, n_processes=16, oned=False
):

    global limitedsize_g

    limitedsize_g = limitedsize

    cellids, t0, t1 = np.loadtxt(
        wrkdir_DNR + "FIF/good_jet_intervals_1.txt", dtype=int
    ).T
    vobj_600 = pt.vlsvfile.VlsvReader(bulkpath_FIF + "bulk1.0000600.vlsv")

    try:
        print(t0[idx])
        coords = vobj_600.get_cell_coordinates(cellids[idx]) / r_e
    except:
        print("Index out of range, exiting gracefully!")

    outdir = "/tmp/FIF/{}".format(idx)

    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    # DEBUG ONLY
    # make_timeseries_global_vdf_anim(
    #     cellids[idx], coords, t0[idx], t0[idx] + 5, outdir=outdir
    # )

    # make_timeseries_global_vdf_anim(
    #     cellids[idx], coords, t0[idx], t1[idx], outdir=outdir
    # )

    args_list = []
    fnr_range = np.arange(t0[idx], t1[idx] + 0.1, 1, dtype=int)

    for fnr in fnr_range:
        args_list.append(
            (cellids[idx], coords, t0[idx], t1[idx], fnr, limitedsize, outdir)
        )

    # Use multiprocessing Pool

    if oned:
        outfilename = (
            "/wrk-vakka/users/jesuni/jets_3D/ani_1d/FIF/c{}_t{}_{}.mp4".format(
                cellids[idx], t0[idx], t1[idx]
            )
        )
        with Pool(processes=n_processes) as pool:
            pool.map(make_timeseries_1d_vdf_one, args_list)
    else:
        outfilename = "/wrk-vakka/users/jesuni/jets_3D/ani/FIF/c{}_t{}_{}.mp4".format(
            cellids[idx], t0[idx], t1[idx]
        )
        with Pool(processes=n_processes) as pool:
            pool.map(make_timeseries_global_vdf_one, args_list)

    # os.environ["FIF_ANIM_FILENAME"] = "/wrk-vakka/users/jesuni/jets_3D/ani/FIF/c{}_t{}_{}.mp4"
    # subprocess.run(
    #     "cat $(find {} -maxdepth 1 -name '*.png' | sort -V) | ffmpeg -framerate 5 -i - -pix_fmt yuv420p -b:v 2000k -vf scale=1600:-2 -y {}".format(
    #         outdir, outfilename
    #     ),
    #     shell=True,
    # )
    subprocess.run(
        "cat $(find {} -maxdepth 1 -name '*.png' | sort -V) | ffmpeg -framerate 5 -i - -b:v 2500k -vf scale=1600:-2 -y {}".format(
            outdir, outfilename
        ),
        shell=True,
    )
    subprocess.run("rm {} -rf".format(outdir), shell=True)


def make_timeseries_1d_vdf_one(args):

    ci, coords, t0, t1, fnr, limitedsize, outdir = args

    txtdir = wrkdir_DNR + "txts/timeseries/{}".format("")
    ts_data = np.loadtxt(
        txtdir
        + "{}_x{:.3f}_y{:.3f}_z{:.3f}_t0{}_t1{}_delta{}.txt".format(
            "FIF", coords[0], coords[1], coords[2], 600, 991, None
        )
    )
    fig, axes = plt.subplots(8, 1, figsize=(16, 24), layout="compressed")
    ts_axes = axes[2:]
    vdf_axes = axes[:2]

    generate_ts_plot(ts_axes, ts_data, ci, coords, t0, t1)
    axvlines = []
    for ax in ts_axes:
        axvlines.append(ax.axvline(t0, linestyle="dashed"))

    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath_FIF + "bulk1.{}.vlsv".format(str(int(fnr)).zfill(7))
    )
    vdf_axes[0].set_title("t = {}s".format(int(vlsvobj.read_parameter("time"))))
    vdf_axes[1].set_xlabel("v [km/s]")

    for ax in vdf_axes:
        ax.grid()
        ax.set_ylabel("$f_v$ [$s/m^4$]")
        ax.set_xlim(-2000, 2000)
        ax.set_ylim(0, 10)
    try:
        generate_1d_vdf_plots(vdf_axes, vlsvobj, ci)
    except:
        pass
    for linepl in axvlines:
        linepl.set_xdata([fnr, fnr])

    fig.savefig(outdir + "/{}.png".format(int(fnr)), dpi=300, bbox_inches="tight")

    print("Saved animation of cellid {} at time {}".format(ci, fnr))
    plt.close(fig)


def make_timeseries_global_vdf_one(args):

    ci, coords, t0, t1, fnr, limitedsize, outdir = args

    txtdir = wrkdir_DNR + "txts/timeseries/{}".format("")
    ts_data = np.loadtxt(
        txtdir
        + "{}_x{:.3f}_y{:.3f}_z{:.3f}_t0{}_t1{}_delta{}.txt".format(
            "FIF", coords[0], coords[1], coords[2], 600, 991, None
        )
    )
    fig = plt.figure(figsize=(24, 16), layout="compressed")
    axes = generate_axes(fig)
    ts_axes = []
    for axname in ["rho", "v", "pdyn", "b", "e", "t"]:
        ts_axes.append(axes[axname])
    vdf_axes = [axes["vdf_xy"], axes["vdf_xz"], axes["vdf_yz"]]
    cmap_axes = [axes["cmap_xy"], axes["cmap_xz"], axes["cmap_yz"]]

    generate_ts_plot(ts_axes, ts_data, ci, coords, t0, t1)
    axvlines = []
    for ax in ts_axes:
        axvlines.append(ax.axvline(t0, linestyle="dashed"))

    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath_FIF + "bulk1.{}.vlsv".format(str(int(fnr)).zfill(7))
    )
    generate_cmap_plots(
        cmap_axes, vlsvobj, coords[0], coords[1], coords[2], limitedsize
    )
    try:
        generate_vdf_plots(vdf_axes, vlsvobj, ci)
    except:
        pass
    for linepl in axvlines:
        linepl.set_xdata([fnr, fnr])

    fig.savefig(outdir + "/{}.png".format(int(fnr)), dpi=300, bbox_inches="tight")

    print("Saved animation of cellid {} at time {}".format(ci, fnr))
    plt.close(fig)


def make_timeseries_global_vdf_anim(ci, coords, t0, t1, outdir=""):

    global vdf_axes, cmap_axes, ci_g, x_g, y_g, z_g, axvlines

    x_g, y_g, z_g = coords

    ci_g = ci

    txtdir = wrkdir_DNR + "txts/timeseries/{}".format("")
    ts_data = np.loadtxt(
        txtdir
        + "{}_x{:.3f}_y{:.3f}_z{:.3f}_t0{}_t1{}_delta{}.txt".format(
            "FIF", coords[0], coords[1], coords[2], 600, 991, None
        )
    )
    fig = plt.figure(figsize=(25, 16), layout="compressed")
    axes = generate_axes(fig)
    ts_axes = []
    for axname in ["rho", "v", "pdyn", "b", "e", "t"]:
        ts_axes.append(axes[axname])
    vdf_axes = [axes["vdf_xy"], axes["vdf_xz"], axes["vdf_yz"]]
    cmap_axes = [axes["cmap_xy"], axes["cmap_xz"], axes["cmap_yz"]]

    generate_ts_plot(ts_axes, ts_data, ci, coords, t0, t1)
    axvlines = []
    for ax in ts_axes:
        axvlines.append(ax.axvline(t0, linestyle="dashed"))

    for fnr in np.arange(t0, t1 + 0.1, 1):
        ts_glob_vdf_update(fnr)
        fig.savefig(outdir + "/{}.png".format(int(fnr)), dpi=300, bbox_inches="tight")

    print("Saved animation of cellid {} from t {} to {}".format(ci, t0, t1))
    plt.close(fig)


def ts_glob_vdf_update(fnr):
    print("Current time: {}".format(fnr))
    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath_FIF + "bulk1.{}.vlsv".format(str(int(fnr)).zfill(7))
    )
    for ax in vdf_axes:
        ax.clear()
    for ax in cmap_axes:
        ax.clear()
    try:
        generate_vdf_plots(vdf_axes, vlsvobj, ci_g)
    except:
        pass
    generate_cmap_plots(cmap_axes, vlsvobj, x_g, y_g, z_g, limitedsize_g)
    for linepl in axvlines:
        linepl.set_xdata([fnr, fnr])


def vspace_reducer(
    vlsvobj,
    cellid,
    operator,
    dv=40e3,
    vmin=None,
    vmax=None,
    b=None,
    v=None,
    binw=40e3,
    fmin=1e-16,
):
    """
    Function for reducing a 3D VDF to 1D
    (object) vlsvobj = Analysator VLSV file object
    (int) cellid = ID of cell whose VDF you want
    (str) operator = "x", "y", or "z", which velocity component to retain after reduction, or "magnitude" to get the distribution of speeds (untested)
    (float) dv = Velocity space resolution in m/s
    """

    # List of valid operators from which to get an index
    op_list = ["x", "y", "z"]

    # Read velocity cell keys and values from vlsv file
    velcels = vlsvobj.read_velocity_cells(cellid)
    vc_coords = vlsvobj.get_velocity_cell_coordinates(list(velcels.keys()))
    vc_vals = np.array(list(velcels.values()))

    ii_fm = np.where(vc_vals >= fmin)
    vc_vals = vc_vals[ii_fm]
    vc_coords = vc_coords[ii_fm, :][0, :, :]

    # Select coordinates of chosen velocity component
    if operator in op_list:
        vc_coord_arr = vc_coords[:, op_list.index(operator)]
    elif operator == "magnitude":
        vc_coord_arr = np.sqrt(
            vc_coords[:, 0] ** 2 + vc_coords[:, 1] ** 2 + vc_coords[:, 2] ** 2
        )
    elif operator == "par":
        # print("par")
        vc_coord_arr = np.dot(vc_coords, b)
    elif operator == "perp1":
        # print("perp")
        bxv = np.cross(b, v)
        vc_coord_arr = np.dot(vc_coords, bxv)
    elif operator == "perp2":
        bxv = np.cross(b, v)
        bxbxv = np.cross(b, bxv)
        vc_coord_arr = np.dot(vc_coords, bxbxv)

    # Create histogram bins, one for each unique coordinate of the chosen velocity component
    vbins = np.sort(np.unique(vc_coord_arr))
    if vmin or vmax:
        vbins = np.arange(vmin - binw / 2, vmax + binw / 2 + binw / 4, binw)
    else:
        vbins = np.arange(
            np.min(vbins) - binw / 2, np.max(vbins) + binw / 2 + binw / 4, binw
        )

    # Create weights, <3D VDF value>*<vspace cell side area>, so that the histogram binning essentially performs an integration
    vweights = vc_vals * dv * dv

    # Integrate over the perpendicular directions
    hist, bin_edges = np.histogram(vc_coord_arr, bins=vbins, weights=vweights)

    # Return the 1D VDF values in units of s/m^4 as well as the bin edges to assist in plotting
    return (hist, bin_edges / 1e3)


def generate_1d_vdf_plots(vdf_axes, vobj, ci):

    B = vobj.read_variable("vg_b_vol", cellids=ci)
    V = vobj.read_variable("proton/vg_v", cellids=ci)
    b = B / np.linalg.norm(B)
    v = V / np.linalg.norm(V)

    hist, bin_edges = vspace_reducer(vobj, ci, "x", b=b, v=v, vmin=-2000e3, vmax=2000e3)
    bin_centers = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    vdf_axes[0].plot(bin_centers, hist, "-", color=CB_color_cycle[0], label="x")

    hist, bin_edges = vspace_reducer(vobj, ci, "y", b=b, v=v, vmin=-2000e3, vmax=2000e3)
    bin_centers = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    vdf_axes[0].plot(bin_centers, hist, "-", color=CB_color_cycle[1], label="y")

    hist, bin_edges = vspace_reducer(vobj, ci, "z", b=b, v=v, vmin=-2000e3, vmax=2000e3)
    bin_centers = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    vdf_axes[0].plot(bin_centers, hist, "-", color=CB_color_cycle[2], label="z")

    hist, bin_edges = vspace_reducer(
        vobj, ci, "par", b=b, v=v, vmin=-2000e3, vmax=2000e3
    )
    bin_centers = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    vdf_axes[1].plot(bin_centers, hist, "-", color=CB_color_cycle[0], label="par")

    hist, bin_edges = vspace_reducer(
        vobj, ci, "perp1", b=b, v=v, vmin=-2000e3, vmax=2000e3
    )
    bin_centers = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    vdf_axes[1].plot(bin_centers, hist, "-", color=CB_color_cycle[1], label="perp1")

    hist, bin_edges = vspace_reducer(
        vobj, ci, "perp2", b=b, v=v, vmin=-2000e3, vmax=2000e3
    )
    bin_centers = bin_edges[:-1] + 0.5 * (bin_edges[1] - bin_edges[0])
    vdf_axes[1].plot(bin_centers, hist, "-", color=CB_color_cycle[2], label="perp2")

    vdf_axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncols=1)
    vdf_axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncols=1)


def generate_vdf_plots(vdf_axes, vobj, ci):

    boxwidth = 3000e3

    pt.plot.plot_vdf(
        axes=vdf_axes[0],
        vlsvobj=vobj,
        cellids=[ci],
        colormap="batlow",
        bvector=1,
        xy=1,
        slicethick=1,
        box=[-boxwidth, boxwidth, -boxwidth, boxwidth],
        setThreshold=1e-16,
        scale=1.3,
        fmin=1e-10,
        fmax=1e-4,
        contours=7,
        # cbaxes=vdf_cb_ax,
        # cb_horizontal=True,
        title="",
    )
    pt.plot.plot_vdf(
        axes=vdf_axes[1],
        vlsvobj=vobj,
        cellids=[ci],
        colormap="batlow",
        bvector=1,
        xz=1,
        slicethick=1,
        box=[-boxwidth, boxwidth, -boxwidth, boxwidth],
        setThreshold=1e-16,
        scale=1.3,
        fmin=1e-10,
        fmax=1e-4,
        contours=7,
        # nocb=True,
        title="",
    )
    pt.plot.plot_vdf(
        axes=vdf_axes[2],
        vlsvobj=vobj,
        cellids=[ci],
        colormap="batlow",
        bvector=1,
        yz=1,
        slicethick=1,
        box=[-boxwidth, boxwidth, -boxwidth, boxwidth],
        setThreshold=1e-16,
        scale=1.3,
        fmin=1e-10,
        fmax=1e-4,
        contours=7,
        # nocb=True,
        title="",
    )


def ext_bs_mp(ax, XmeshXY, YmeshXY, pass_maps):
    beta_star = pass_maps["vg_beta_star"]
    rho = pass_maps["proton/vg_rho"]

    ax.contour(XmeshXY, YmeshXY, rho, [2e6], colors=["red"])
    # try:
    #     ax.contour(XmeshXY, YmeshXY, Tcore, [1.5e6], colors=["red"])
    # except:
    #     pass
    ax.contour(XmeshXY, YmeshXY, beta_star, [0.3], colors=["white"])


def generate_cmap_plots(cmap_axes, vobj, x0, y0, z0, limitedsize):

    boxwidth = 4

    pt.plot.plot_colormap3dslice(
        axes=cmap_axes[0],
        vlsvobj=vobj,
        var="proton/vg_Pdyn",
        vmin=0.05,
        vmax=2,
        lin=5,
        vscale=1e9,
        cbtitle="$P_\\mathrm{dyn}$ [nPa]",
        usesci=0,
        boxre=[x0 - boxwidth, x0 + boxwidth, y0 - boxwidth, y0 + boxwidth],
        # cbaxes=cmap_cb_ax,
        # cb_horizontal=True,
        colormap="batlow",
        scale=1.3,
        tickinterval=1.0,
        normal="z",
        cutpointre=z0,
        # title="",
        limitedsize=limitedsize,
        external=ext_bs_mp,
        pass_vars=["vg_beta_star", "proton/vg_rho"],
    )
    cmap_axes[0].axvline(x0, linestyle="dashed", linewidth=0.6, color="k")
    cmap_axes[0].axhline(y0, linestyle="dashed", linewidth=0.6, color="k")

    pt.plot.plot_colormap3dslice(
        axes=cmap_axes[1],
        vlsvobj=vobj,
        var="proton/vg_Pdyn",
        vmin=0.05,
        vmax=2,
        lin=5,
        vscale=1e9,
        cbtitle="$P_\\mathrm{dyn}$ [nPa]",
        usesci=0,
        boxre=[x0 - boxwidth, x0 + boxwidth, z0 - boxwidth, z0 + boxwidth],
        # nocb=True,
        colormap="batlow",
        scale=1.3,
        tickinterval=1.0,
        normal="y",
        cutpointre=y0,
        title="",
        limitedsize=limitedsize,
        external=ext_bs_mp,
        pass_vars=["vg_beta_star", "proton/vg_rho"],
    )
    cmap_axes[1].axvline(x0, linestyle="dashed", linewidth=0.6, color="k")
    cmap_axes[1].axhline(z0, linestyle="dashed", linewidth=0.6, color="k")

    pt.plot.plot_colormap3dslice(
        axes=cmap_axes[2],
        vlsvobj=vobj,
        var="proton/vg_Pdyn",
        vmin=0.05,
        vmax=2,
        lin=5,
        vscale=1e9,
        cbtitle="$P_\\mathrm{dyn}$ [nPa]",
        usesci=0,
        boxre=[y0 - boxwidth, y0 + boxwidth, z0 - boxwidth, z0 + boxwidth],
        # nocb=True,
        colormap="batlow",
        scale=1.3,
        tickinterval=1.0,
        normal="x",
        cutpointre=x0,
        title="",
        limitedsize=limitedsize,
        external=ext_bs_mp,
        pass_vars=["vg_beta_star", "proton/vg_rho"],
    )
    cmap_axes[2].axvline(y0, linestyle="dashed", linewidth=0.6, color="k")
    cmap_axes[2].axhline(z0, linestyle="dashed", linewidth=0.6, color="k")


def generate_ts_plot(ts_axes, ts_data, ci, coords, t0, t1):

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

    t_arr = np.arange(600, 991 + 0.1, 1)
    tavg_arr = uniform_filter1d(
        ts_data[5, :], 180, mode="constant", cval=np.nanmean(ts_data[5, :])
    )
    pdynx = m_p * ts_data[0] * 1e6 * ts_data[1] * 1e3 * ts_data[1] * 1e3 * 1e9
    tavg_x_arr = uniform_filter1d(pdynx, 180, mode="constant", cval=np.nanmean(pdynx))

    ts_axes[0].set_title(
        "Run: {}, $x_0$: {:.3f}, $y_0$: {:.3f}, $z_0$: {:.3f}, cell: {}".format(
            "FIF", coords[0], coords[1], coords[2], int(ci)
        )
    )
    ts_axes[-1].set_xlabel("t [s]")
    for idx in range(len(plot_labels)):
        ax = ts_axes[plot_index[idx]]
        ax.plot(
            t_arr,
            ts_data[idx],
            "-",
            color=plot_colors[idx],
            label=plot_labels[idx],
        )
        if idx == 5:
            ax.plot(
                t_arr,
                2 * tavg_arr,
                color=CB_color_cycle[1],
                linestyle="dashed",
                label="$2\\langle P_\\mathrm{dyn}\\rangle$",
            )
            ax.plot(
                t_arr,
                pdynx,
                "-",
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

        ax.set_xlim(min(t0, 690), max(t1, 900))
        if draw_legend[idx]:
            ncols = 1
            if idx == 5:
                ncols = 1
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncols=ncols)

    for idx, ax in enumerate(ts_axes):
        ax.grid()
        ax.set_ylabel(ylabels[idx])
        ax.fill_between(
            t_arr,
            0,
            1,
            where=ts_data[5, :] > 2 * tavg_arr,
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


def generate_axes(fig):
    gridspec = fig.add_gridspec(nrows=6, ncols=9)
    axes = {}
    axes["cmap_xy"] = fig.add_subplot(gridspec[0:2, 0:2])
    axes["cmap_xz"] = fig.add_subplot(gridspec[2:4, 0:2])
    axes["cmap_yz"] = fig.add_subplot(gridspec[4:6, 0:2])
    axes["vdf_xy"] = fig.add_subplot(gridspec[0:2, 2:4])
    axes["vdf_xz"] = fig.add_subplot(gridspec[2:4, 2:4])
    axes["vdf_yz"] = fig.add_subplot(gridspec[4:6, 2:4])
    axes["rho"] = fig.add_subplot(gridspec[0:1, 4:9])
    axes["v"] = fig.add_subplot(gridspec[1:2, 4:9])
    axes["pdyn"] = fig.add_subplot(gridspec[2:3, 4:9])
    axes["b"] = fig.add_subplot(gridspec[3:4, 4:9])
    axes["e"] = fig.add_subplot(gridspec[4:5, 4:9])
    axes["t"] = fig.add_subplot(gridspec[5:6, 4:9])
    return axes


def make_yz_slice_one(fnr):

    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath_FIF + "bulk1.{}.vlsv".format(str(int(fnr)).zfill(7))
    )

    fig, ax_list = plt.subplots(2, 2, figsize=(15, 15), layout="compressed")
    ax_flat = ax_list.flatten()

    xcuts = [11, 12, 13, 14]

    for idx in range(4):
        pt.plot.plot_colormap3dslice(
            axes=ax_flat[idx],
            vlsvobj=vlsvobj,
            var="proton/vg_Pdyn",
            vmin=0.05,
            vmax=2,
            lin=5,
            vscale=1e9,
            cbtitle="$P_\\mathrm{dyn}$ [nPa]",
            usesci=0,
            boxre=[-10, 10, -10, 10],
            # nocb=True,
            colormap="batlow",
            scale=1.3,
            tickinterval=1.0,
            normal="x",
            cutpointre=xcuts[idx],
            limitedsize=True,
            external=ext_bs_mp,
            pass_vars=["vg_beta_star", "proton/vg_rho"],
        )

    fig.savefig(
        wrkdir_DNR + "xcuts/{}.png".format(int(fnr)), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def make_yz_anim(n_processes=16):

    fnr_range = np.arange(690, 901, 1)

    outfilename = "/wrk-vakka/users/jesuni/jets_3D/yz_cuts.mp4"
    with Pool(processes=n_processes) as pool:
        pool.map(make_yz_slice_one, fnr_range)

    subprocess.run(
        "cat $(find /wrk-vakka/users/jesuni/jets_3D/xcuts -maxdepth 1 -name '*.png' | sort -V) | ffmpeg -framerate 5 -i - -b:v 2500k -vf scale=1600:-2 -y {}".format(
            outfilename
        ),
        shell=True,
    )
