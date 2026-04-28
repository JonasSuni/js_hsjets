import analysator as pt
import os, subprocess
import sys
from random import choice
from copy import deepcopy

from pyJets.jet_aux import CB_color_cycle

from scipy.linalg import eig, lstsq
from scipy.fft import rfft2
from scipy.signal import butter, sosfilt, argrelextrema
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
import analysator.plot

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

# wrkdir_DNR = wrkdir_DNR + "jets_3D/"
wrkdir_NEW = "/turso/home/jesuni/wrk/jets_3D/"
wrkdir_DNR = wrkdir_NEW
wrkdir_other = os.environ["WRK"] + "/"

bulkpath_FIF = "/turso/group/spacephysics/vlasiator/data/L0/3D/FIF/bulk1/"

plot_B_vdfs = False
slicethick_g = 1
calc_rel_dens_g = True
plot_gmm = None
scale_g = 1.3


def array_to_disjoint_naive(data_arr, bool_arr, len_thresh=1):

    out_arr = []
    sub_arr = []
    for idx in range(data_arr.size):
        if bool_arr[idx]:
            sub_arr.append(data_arr[idx])
        else:
            if len(sub_arr) >= len_thresh:
                out_arr.append(deepcopy(sub_arr))
                sub_arr = []
            else:
                pass
    if len(sub_arr) >= len_thresh:
        out_arr.append(sub_arr)

    return out_arr


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
    idx, limitedsize=True, n_processes=16, plot_type=1
):

    global limitedsize_g

    limitedsize_g = limitedsize

    # cellids, t0, t1 = np.loadtxt(
    #     wrkdir_DNR + "FIF/good_jet_intervals_1.txt", dtype=int
    # ).T
    cellids, t0, t1 = np.loadtxt(wrkdir_DNR + "good.txt", dtype=float).astype(int).T
    vobj_600 = pt.vlsvfile.VlsvReader(bulkpath_FIF + "bulk1.0000600.vlsv")

    try:
        print(t0[idx])
        coords = vobj_600.get_cell_coordinates(cellids[idx]) / r_e
    except:
        print("Index out of range, exiting gracefully!")

    outdir = "/tmp/FIF/{}/".format(idx)

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

    if plot_type == 2:
        outfilename = (
            "/wrk-vakka/users/jesuni/jets_3D/ani_1d/FIF/c{}_t{}_{}.mp4".format(
                cellids[idx], t0[idx], t1[idx]
            )
        )
        with Pool(processes=n_processes) as pool:
            pool.map(make_timeseries_1d_vdf_one, args_list)
    elif plot_type == 1:
        outfilename = "/wrk-vakka/users/jesuni/jets_3D/ani/FIF/c{}_t{}_{}.mp4".format(
            cellids[idx], t0[idx], t1[idx]
        )
        with Pool(processes=n_processes) as pool:
            result = pool.map(make_timeseries_global_vdf_one, args_list)

        np.savetxt(
            wrkdir_DNR
            + "txts/rel_dens/c{}_t{}_{}.txt".format(cellids[idx], t0[idx], t1[idx]),
            result,
        )
    elif plot_type == 3:
        outfilename = (
            "/wrk-vakka/users/jesuni/jets_3D/ani_vdf/FIF/c{}_t{}_{}.mp4".format(
                cellids[idx], t0[idx], t1[idx]
            )
        )
        with Pool(processes=n_processes) as pool:
            pool.map(make_global_vdf_one, args_list)

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


def extract_all_vdf(n_processes=16, fmin=1e-16, prepost_time=30):

    archer_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/archer_intervals.txt", dtype=int
    )
    koller_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/koller_intervals.txt", dtype=int
    )
    archerkoller_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/archerkoller_intervals.txt", dtype=int
    )

    for p in archer_data:
        ci, t0, t1, tjet = p
        args_list = []
        for fnr in np.arange(t0 - prepost_time, t1 + prepost_time + 0.1, 1, dtype=int):
            args_list.append([ci, fnr, fmin])
        with Pool(processes=n_processes) as pool:
            pool.map(vspace_extracter, args_list)

    for p in koller_data:
        ci, t0, t1, tjet = p
        args_list = []
        for fnr in np.arange(t0 - prepost_time, t1 + prepost_time + 0.1, 1, dtype=int):
            args_list.append([ci, fnr, fmin])
        with Pool(processes=n_processes) as pool:
            pool.map(vspace_extracter, args_list)

    for p in archerkoller_data:
        ci, t0, t1, tjet = p
        args_list = []
        for fnr in np.arange(t0 - prepost_time, t1 + prepost_time + 0.1, 1, dtype=int):
            args_list.append([ci, fnr, fmin])
        with Pool(processes=n_processes) as pool:
            pool.map(vspace_extracter, args_list)


def create_dir_if_not_exist(outdir):

    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
            print("Created directory {}".format(outdir))
        except OSError:
            print("Directory {} already exists".format(outdir))
            pass


def jet_interval_snap_all(
    limitedsize=False,
    B_vdfs=False,
    slicethick=1,
    calc_rel_dens=True,
    gmm=None,
    scale=1.3,
):

    global limitedsize_g, scale_g

    limitedsize_g = limitedsize
    scale_g = scale

    archer_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/archer_intervals.txt", dtype=int
    )
    koller_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/koller_intervals.txt", dtype=int
    )
    archerkoller_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/archerkoller_intervals.txt", dtype=int
    )
    create_dir_if_not_exist(wrkdir_DNR + "Figs/jet_gmm/archer/")
    create_dir_if_not_exist(wrkdir_DNR + "Figs/jet_gmm/koller/")
    create_dir_if_not_exist(wrkdir_DNR + "Figs/jet_gmm/archerkoller/")

    vobj_600 = pt.vlsvfile.VlsvReader(bulkpath_FIF + "bulk1.0000600.vlsv")

    global plot_B_vdfs, slicethick_g, calc_rel_dens_g, plot_gmm
    plot_B_vdfs = B_vdfs
    slicethick_g = slicethick
    calc_rel_dens_g = calc_rel_dens
    plot_gmm = gmm

    print("Plot GMM is {}".format(plot_gmm))

    for p in archer_data:
        ci, t0, t1, tjet = p
        coords = vobj_600.get_cell_coordinates(ci) / r_e
        outdir = wrkdir_DNR + "Figs/jet_gmm/archer/{}_{}_{}_".format(ci, t0, t1)
        args = (ci, coords, t0, t1, tjet, limitedsize, outdir)
        make_timeseries_global_vdf_one(args)

    for p in koller_data:
        ci, t0, t1, tjet = p
        coords = vobj_600.get_cell_coordinates(ci) / r_e
        outdir = wrkdir_DNR + "Figs/jet_gmm/koller/{}_{}_{}_".format(ci, t0, t1)
        args = (ci, coords, t0, t1, tjet, limitedsize, outdir)
        make_timeseries_global_vdf_one(args)

    for p in archerkoller_data:
        ci, t0, t1, tjet = p
        coords = vobj_600.get_cell_coordinates(ci) / r_e
        outdir = wrkdir_DNR + "Figs/jet_gmm/archerkoller/{}_{}_{}_".format(ci, t0, t1)
        args = (ci, coords, t0, t1, tjet, limitedsize, outdir)
        make_timeseries_global_vdf_one(args)


def jet_interval_anim_all(
    limitedsize=False,
    n_processes=16,
    plot_type=1,
    only_rel_dens=False,
    prepost_time=10,
    B_vdfs=False,
    slicethick=1,
    calc_rel_dens=True,
    gmm=None,
):

    archer_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/archer_intervals.txt", dtype=int
    )
    koller_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/koller_intervals.txt", dtype=int
    )
    archerkoller_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/archerkoller_intervals.txt", dtype=int
    )

    vobj_600 = pt.vlsvfile.VlsvReader(bulkpath_FIF + "bulk1.0000600.vlsv")

    global plot_B_vdfs, slicethick_g, calc_rel_dens_g, plot_gmm
    plot_B_vdfs = B_vdfs
    slicethick_g = slicethick
    calc_rel_dens_g = calc_rel_dens
    plot_gmm = gmm

    print("Plot GMM is {}".format(plot_gmm))

    for p in archer_data:
        ci, t0, t1, tjet = p
        coords = vobj_600.get_cell_coordinates(ci) / r_e
        # t0 = t0 - prepost_time
        # t1 = t1 + prepost_time
        if not only_rel_dens:
            jet_intervals_anim_one(
                ci,
                coords,
                t0,
                t1,
                limitedsize=limitedsize,
                n_processes=n_processes,
                plot_type=plot_type,
                jet_type="archer",
                prepost_time=prepost_time,
                tjet=int(tjet),
            )
        if only_rel_dens:
            rel_dens_plotter(
                ci, t0, t1, tjet, jet_type="archer", prepost_time=prepost_time
            )

    for p in koller_data:
        ci, t0, t1, tjet = p
        coords = vobj_600.get_cell_coordinates(ci) / r_e
        # t0 = t0 - prepost_time
        # t1 = t1 + prepost_time
        if not only_rel_dens:
            jet_intervals_anim_one(
                ci,
                coords,
                t0,
                t1,
                limitedsize=limitedsize,
                n_processes=n_processes,
                plot_type=plot_type,
                jet_type="koller",
                prepost_time=prepost_time,
                tjet=int(tjet),
            )
        if only_rel_dens:
            rel_dens_plotter(
                ci, t0, t1, tjet, jet_type="koller", prepost_time=prepost_time
            )

    for p in archerkoller_data:
        ci, t0, t1, tjet = p
        coords = vobj_600.get_cell_coordinates(ci) / r_e
        # t0 = t0 - prepost_time
        # t1 = t1 + prepost_time
        if not only_rel_dens:
            jet_intervals_anim_one(
                ci,
                coords,
                t0,
                t1,
                limitedsize=limitedsize,
                n_processes=n_processes,
                plot_type=plot_type,
                jet_type="archerkoller",
                prepost_time=prepost_time,
                tjet=int(tjet),
            )
        if only_rel_dens:
            rel_dens_plotter(
                ci, t0, t1, tjet, jet_type="archerkoller", prepost_time=prepost_time
            )


def rel_dens_plotter(ci, t0, t1, tjet, jet_type="archer", prepost_time=10):

    data = np.loadtxt(wrkdir_DNR + "txts/rel_dens/c{}_t{}_{}.txt".format(ci, t0, t1))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), layout="compressed")

    t_arr = np.arange(t0 - prepost_time, t1 + prepost_time + 0.001, 1)
    ax.plot(t_arr, data)
    ax.set_xlim(t0 - prepost_time, t1 + prepost_time)
    ax.set_ylim(0, 1.1)
    ax.grid()
    ax.axvline(tjet, linestyle="dashed", color="red")
    ax.fill_between(
        t_arr,
        0,
        1,
        where=np.logical_and(t_arr >= t0, t_arr <= t1),
        color="green",
        alpha=0.2,
        transform=ax.get_xaxis_transform(),
        linewidth=0,
    )
    ax.label_outer()
    ax.tick_params(labelsize=12)
    ax.set_xlabel("t [s]", fontsize=24, labelpad=10)
    ax.set_ylabel("$r_\\mathrm{sw}$", fontsize=24, labelpad=10)

    fig.savefig(
        wrkdir_DNR + "Figs/rel_dens/{}/c{}_t{}_{}.png".format(jet_type, ci, t0, t1),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def jet_intervals_anim_one(
    ci,
    coords,
    t0,
    t1,
    limitedsize=True,
    n_processes=16,
    plot_type=1,
    jet_type="archer",
    prepost_time=10,
    tjet=None,
):

    global limitedsize_g

    limitedsize_g = limitedsize

    outdir = "/tmp/FIF/{}/".format(ci)

    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    args_list = []
    fnr_range = np.arange(t0 - prepost_time, t1 + prepost_time + 0.1, 1, dtype=int)

    for fnr in fnr_range:
        args_list.append(
            (ci, coords, t0 - prepost_time, t1 + prepost_time, fnr, limitedsize, outdir)
        )

    # Use multiprocessing Pool

    if plot_type == 2:
        outfilename = wrkdir_DNR + "ani_1d/FIF/{}/c{}_t{}_{}.mp4".format(
            jet_type, ci, t0, t1
        )
        with Pool(processes=n_processes) as pool:
            pool.map(make_timeseries_1d_vdf_one, args_list)
    elif plot_type == 1:
        outfilename = wrkdir_DNR + "ani/FIF/{}/c{}_t{}_{}.mp4".format(
            jet_type, ci, t0, t1
        )
        with Pool(processes=n_processes) as pool:
            result = pool.map(make_timeseries_global_vdf_one, args_list)

        if calc_rel_dens_g:
            np.savetxt(
                wrkdir_DNR + "txts/rel_dens/c{}_t{}_{}.txt".format(ci, t0, t1), result
            )
    elif plot_type == 3:
        outfilename = wrkdir_DNR + "ani_vdf/FIF/{}/c{}_t{}_{}.mp4".format(
            jet_type, ci, t0, t1
        )
        with Pool(processes=n_processes) as pool:
            pool.map(make_global_vdf_one, args_list)

    subprocess.run(
        "cat $(find {} -maxdepth 1 -name '*.png' | sort -V) | ffmpeg -framerate 5 -i - -b:v 2500k -vf scale=1600:-2 -y {}".format(
            outdir, outfilename
        ),
        shell=True,
    )
    subprocess.run("rm {} -rf".format(outdir), shell=True)


def jet_interval_sorter(len_thresh=1):

    cellids, t0_arr, t1_arr = (
        np.loadtxt(wrkdir_DNR + "good.txt", dtype=float).astype(int).T
    )
    vobj_600 = pt.vlsvfile.VlsvReader(bulkpath_FIF + "bulk1.0000600.vlsv")

    pd_intervals_all = []
    pdx_intervals_all = []

    for idx in range(100):
        try:
            print(t0_arr[idx])
            coords = vobj_600.get_cell_coordinates(cellids[idx]) / r_e
        except:
            print("Index out of range, exiting gracefully!")
            break

        ci = cellids[idx]
        t0 = t0_arr[idx]
        t1 = t1_arr[idx]

        txtdir = wrkdir_DNR + "txts/timeseries/{}".format("")
        ts_data = np.loadtxt(
            txtdir
            + "{}_x{:.3f}_y{:.3f}_z{:.3f}_t0{}_t1{}_delta{}.txt".format(
                "FIF", coords[0], coords[1], coords[2], 600, 991, None
            )
        )

        t_arr = np.arange(600, 991 + 0.1, 1)
        t_restr = np.arange(690, 900 + 0.1, 1)
        tavg_arr = uniform_filter1d(
            ts_data[5, :], 180, mode="constant", cval=np.nanmean(ts_data[5, :])
        )
        pdynx = m_p * ts_data[0] * 1e6 * ts_data[1] * 1e3 * ts_data[1] * 1e3 * 1e9
        tavg_x_arr = uniform_filter1d(
            pdynx, 180, mode="constant", cval=np.nanmean(pdynx)
        )

        bool_arr = ts_data[5, :] >= 2 * tavg_arr
        bool_x_arr = pdynx >= 3 * tavg_x_arr

        pd_intervals = array_to_disjoint_naive(t_arr, bool_arr, len_thresh)
        pdx_intervals = array_to_disjoint_naive(t_arr, bool_x_arr, len_thresh)

        for intval in pd_intervals:
            t_masked = t_arr[np.isin(t_arr, intval)]
            pd_masked = ts_data[5, :][np.isin(t_arr, intval)]
            t_pdmax = t_masked[np.argmax(pd_masked)]
            if np.isin(t_restr, [t_pdmax]).any():
                pd_intervals_all.append([ci, intval[0], intval[-1], t_pdmax])

        for intval in pdx_intervals:
            t_masked = t_arr[np.isin(t_arr, intval)]
            pd_masked = ts_data[5, :][np.isin(t_arr, intval)]
            t_pdmax = t_masked[np.argmax(pd_masked)]
            if np.isin(t_restr, [t_pdmax]).any():
                pdx_intervals_all.append([ci, intval[0], intval[-1], t_pdmax])

    a_intervals = []
    k_intervals = []
    ak_intervals = []

    pd_intervals_all_short = np.array(pd_intervals_all)[:, [0, 3]].tolist()
    pdx_intervals_all_short = np.array(pdx_intervals_all)[:, [0, 3]].tolist()

    for intval in pd_intervals_all:
        if [intval[0], intval[3]] in pdx_intervals_all_short:
            pdx_intval = pdx_intervals_all[
                pdx_intervals_all_short.index([intval[0], intval[3]])
            ]
            intval[1] = min(intval[1], pdx_intval[1])
            intval[2] = max(intval[2], pdx_intval[2])
            ak_intervals.append(intval)
        else:
            a_intervals.append(intval)

    for intval in pdx_intervals_all:
        if [intval[0], intval[3]] in pd_intervals_all_short:
            pass
        else:
            k_intervals.append(intval)

    outdir = wrkdir_DNR + "txts/jet_intervals/"
    np.savetxt(
        outdir + "archer_intervals.txt",
        a_intervals,
        fmt="%d",
    )
    np.savetxt(
        outdir + "koller_intervals.txt",
        k_intervals,
        fmt="%d",
    )
    np.savetxt(
        outdir + "archerkoller_intervals.txt",
        ak_intervals,
        fmt="%d",
    )


def location_plot():

    vobj_600 = pt.vlsvfile.VlsvReader(bulkpath_FIF + "bulk1.0000600.vlsv")

    archer_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/archer_intervals.txt", dtype=int
    )
    koller_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/koller_intervals.txt", dtype=int
    )
    archerkoller_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/archerkoller_intervals.txt", dtype=int
    )

    cat_coords = [[], [], []]

    cat_list = [archer_data, koller_data, archerkoller_data]

    for idx in range(len(cat_list)):
        for p in cat_list[idx]:
            ci, t0, t1, tjet = p
            coords = vobj_600.get_cell_coordinates(ci) / r_e
            cat_coords[idx].append(coords)

    fig, ax_list = plt.subplots(3, 3, figsize=(20, 20), layout="compressed")

    titles = ["Archer", "Koller", "Archerkoller"]

    # boxre = [8, 15, -10, 0, -10, 10]
    dr = 1000e3 / r_e
    # xbins = np.arange(8, 15 + dr / 2.0, dr)
    # ybins = np.arange(-10, 0 + dr / 2.0, dr)
    # zbins = np.arange(-10, 10 + dr / 2.0, dr)
    xbins = np.linspace(8, 15, 20)
    ybins = np.linspace(-10, 0, 20)
    zbins = np.linspace(-10, 10, 20)

    for idx in range(len(cat_list)):
        ax_col = ax_list[:, idx]
        data = np.array(cat_coords[idx])
        xarr, yarr, zarr = data.T
        hxy, xedges, yedges = np.histogram2d(
            xarr,
            yarr,
            bins=[xbins, ybins],
        )
        hxz, xedges, zedges = np.histogram2d(
            xarr,
            zarr,
            bins=[xbins, zbins],
        )
        hyz, yedges, zedges = np.histogram2d(
            yarr,
            zarr,
            bins=[ybins, zbins],
        )

        hxy[hxy == 0] = np.nan
        im_xy = ax_col[0].pcolormesh(xedges, yedges, hxy.T, cmap="batlow", zorder=6)
        cb_xy = fig.colorbar(im_xy, ax=ax_col[0])
        cb_xy.set_label("Count", fontsize=12, labelpad=10, rotation=270)
        ax_col[0].set_title(titles[idx], fontsize=24, pad=10)
        pt.plot.plot_colormap3dslice(
            vlsvobj=vobj_600,
            axes=ax_col[0],
            var="vg_connection",
            vmin=42,
            vmax=43,
            colormap="Grays",
            nocb=True,
            normal="z",
            title="",
            external=ext_rho,
            pass_vars=["proton/vg_rho"],
        )
        ax_col[0].set(xlim=(8, 15), ylim=(-10, 0))

        hxz[hxz == 0] = np.nan
        im_xz = ax_col[1].pcolormesh(xedges, zedges, hxz.T, cmap="batlow", zorder=6)
        cb_xz = fig.colorbar(im_xz, ax=ax_col[1])
        cb_xz.set_label("Count", fontsize=12, labelpad=10, rotation=270)
        pt.plot.plot_colormap3dslice(
            vlsvobj=vobj_600,
            axes=ax_col[1],
            var="vg_connection",
            vmin=42,
            vmax=43,
            colormap="Grays",
            nocb=True,
            normal="y",
            title="",
            external=ext_rho,
            pass_vars=["proton/vg_rho"],
        )
        ax_col[1].set(xlim=(8, 15), ylim=(-10, 10))

        hyz[hyz == 0] = np.nan
        im_yz = ax_col[2].pcolormesh(yedges, zedges, hyz.T, cmap="batlow", zorder=6)
        cb_yz = fig.colorbar(im_yz, ax=ax_col[2])
        cb_yz.set_label("Count", fontsize=12, labelpad=10, rotation=270)
        pt.plot.plot_colormap3dslice(
            vlsvobj=vobj_600,
            axes=ax_col[2],
            var="vg_connection",
            vmin=42,
            vmax=43,
            colormap="Grays",
            nocb=True,
            normal="x",
            title="",
            external=ext_rho,
            pass_vars=["proton/vg_rho"],
        )
        ax_col[2].set(xlim=(-10, 0), ylim=(-10, 10))

    for ax in ax_list.flatten():
        ax.grid()
        ax.tick_params(labelsize=12)

    fig.savefig(wrkdir_DNR + "Figs/locplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def ext_rho(ax, XmeshXY, YmeshXY, pass_maps):

    rho = pass_maps["proton/vg_rho"]

    ax.contour(XmeshXY, YmeshXY, rho, [2e6], colors=["red"], zorder=7)


def archerplot(prejet_window_size=30):

    vobj_600 = pt.vlsvfile.VlsvReader(bulkpath_FIF + "bulk1.0000600.vlsv")

    archer_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/archer_intervals.txt", dtype=int
    )
    koller_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/koller_intervals.txt", dtype=int
    )
    archerkoller_data = np.loadtxt(
        wrkdir_DNR + "txts/jet_intervals/archerkoller_intervals.txt", dtype=int
    )

    a_toplot = archer_data[:, [0, 3]].tolist()
    k_toplot = koller_data[:, [0, 3]].tolist()
    ak_toplot = archerkoller_data[:, [0, 3]].tolist()

    a_contribs = []
    k_contribs = []
    ak_contribs = []
    all_contribs = []

    cat_contribs = [a_contribs, k_contribs, ak_contribs]
    cat_toplot = [a_toplot, k_toplot, ak_toplot]

    for idx in range(len(cat_toplot)):
        for p in cat_toplot[idx]:
            ci, t0 = p
            t0_idx = t0 - 600
            # print(t0_idx)
            coords = vobj_600.get_cell_coordinates(ci) / r_e
            data_arr = np.loadtxt(
                wrkdir_DNR
                + "txts/timeseries/{}".format("")
                + "{}_x{:.3f}_y{:.3f}_z{:.3f}_t0{}_t1{}_delta{}.txt".format(
                    "FIF", coords[0], coords[1], coords[2], 600, 991, None
                )
            )
            pdyn = data_arr[5, :]
            v = data_arr[4, :]
            v2 = v**2
            rho = data_arr[0, :]

            rhocontrib = (
                rho[t0_idx] - np.nanmean(rho[t0_idx - prejet_window_size : t0_idx])
            ) / np.nanmean(rho[t0_idx - prejet_window_size : t0_idx])
            vcontrib = (
                v2[t0_idx] - np.nanmean(v2[t0_idx - prejet_window_size : t0_idx])
            ) / np.nanmean(v2[t0_idx - prejet_window_size : t0_idx])
            pdyncontrib = (
                pdyn[t0_idx] - np.nanmean(pdyn[t0_idx - prejet_window_size : t0_idx])
            ) / np.nanmean(pdyn[t0_idx - prejet_window_size : t0_idx])

            cat_contribs[idx].append([rhocontrib / pdyncontrib, vcontrib / pdyncontrib])
            all_contribs.append([rhocontrib / pdyncontrib, vcontrib / pdyncontrib])

    fig, ax_list = plt.subplots(
        2, 2, figsize=(16, 16), layout="compressed", sharex=True, sharey=True
    )
    ax_flat = ax_list.flatten()

    titles = ["Archer", "Koller", "Archer-Koller", ""]

    for idx in range(len(cat_contribs)):
        ax = ax_flat[idx]
        xvals, yvals = np.array(cat_contribs[idx]).T
        ax.plot(xvals, yvals, "o")
        ax.set_title(titles[idx], fontsize=24, pad=10)

    ax = ax_flat[3]
    xvals, yvals = np.array(all_contribs).T
    h, xedges, yedges = np.histogram2d(
        xvals,
        yvals,
        bins=[np.arange(-1, 2.5 + 0.0001, 0.1), np.arange(-1, 2.5 + 0.0001, 0.1)],
    )
    h[h == 0] = np.nan
    im = ax.pcolormesh(xedges, yedges, h.T, cmap="batlow")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Count", fontsize=24, labelpad=20, rotation=270)
    # ax.plot(xvals, yvals, "o")
    ax.set_title("All", fontsize=24, pad=10)

    for ax in ax_flat:
        ax.set_xlabel(
            "$\\frac{\\delta\\rho(P_\\mathrm{dyn,max})}{\\langle \\rho \\rangle_\\mathrm{pre-jet}} / \\frac{\\delta P_\\mathrm{dyn} (P_\\mathrm{dyn,max})}{\\langle P_\\mathrm{dyn} \\rangle_\\mathrm{pre-jet}}$",
            fontsize=24,
            labelpad=10,
        )
        ax.set_ylabel(
            "$\\frac{\\delta v^2 (P_\\mathrm{dyn,max})}{\\langle v^2 \\rangle_\\mathrm{pre-jet}} / \\frac{\\delta P_\\mathrm{dyn} (P_\\mathrm{dyn,max})}{\\langle P_\\mathrm{dyn} \\rangle_\\mathrm{pre-jet}}$",
            fontsize=24,
            labelpad=10,
        )
        # ax.legend()
        ax.axvline(0, linestyle="dashed", linewidth=0.6)
        ax.axhline(0, linestyle="dashed", linewidth=0.6)
        ax.grid()
        ax.set_xlim(-1, 2.5)
        ax.set_ylim(-1, 2.5)
        ax.label_outer()
        ax.tick_params(labelsize=12)

    fig.savefig(wrkdir_DNR + "Figs/archerplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_global_vdf_one(args):
    ci, coords, t0, t1, fnr, limitedsize, outdir = args

    fig, axes = plt.subplots(3, 3, figsize=(16, 16), layout="compressed")
    cmap_axes = axes[0, :]
    vdf_xyz_axes = axes[1, :]
    vdf_b_axes = axes[2, :]

    cmap_cb_ax = fig.add_axes((0, 1.01, 1, 0.01))
    vdf_cb_ax = fig.add_axes((0, -0.03, 1, 0.01))

    cmap_axes = np.append(cmap_axes, cmap_cb_ax)
    vdf_xyz_axes = np.append(vdf_xyz_axes, vdf_cb_ax)
    vdf_b_axes = np.append(vdf_b_axes, vdf_cb_ax)

    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath_FIF + "bulk1.{}.vlsv".format(str(int(fnr)).zfill(7))
    )
    generate_cmap_plots(
        cmap_axes, vlsvobj, coords[0], coords[1], coords[2], limitedsize
    )
    try:
        generate_vdf_plots(vdf_xyz_axes, vlsvobj, ci)
        generate_vdf_B_plots(vdf_b_axes, vlsvobj, ci)
    except:
        pass

    fig.savefig(outdir + "/{}.png".format(int(fnr)), dpi=300, bbox_inches="tight")

    print("Saved frame of cellid {} at time {}".format(ci, fnr))
    plt.close(fig)


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
        ax.set_ylim(0, 15)
    try:
        generate_1d_vdf_plots(vdf_axes, vlsvobj, ci)
    except:
        pass
    for linepl in axvlines:
        linepl.set_xdata([fnr, fnr])

    fig.savefig(outdir + "{}.png".format(int(fnr)), dpi=300, bbox_inches="tight")

    print("Saved frame of cellid {} at time {}".format(ci, fnr))
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
    vdf_axes = [axes["vdf_xy"], axes["vdf_xz"], axes["vdf_yz"], axes["vdf_cb"]]
    cmap_axes = [axes["cmap_xy"], axes["cmap_xz"], axes["cmap_yz"], axes["cmap_cb"]]

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
    res = None
    try:
        if plot_B_vdfs:
            generate_vdf_B_plots(vdf_axes, vlsvobj, ci)
        else:
            generate_vdf_plots(vdf_axes, vlsvobj, ci)
        if calc_rel_dens_g:
            res = density_rel_to_mb(vlsvobj, ci)
    except:
        pass
    for linepl in axvlines:
        linepl.set_xdata([fnr, fnr])

    fig.savefig(outdir + "{}.png".format(int(fnr)), dpi=300, bbox_inches="tight")

    print("Saved frame of cellid {} at time {}".format(ci, fnr))
    plt.close(fig)

    vlsvobj.optimize_clear_fileindex_for_cellid()

    if res is not None:
        return res
    else:
        return np.nan


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
    vdf_axes = [axes["vdf_xy"], axes["vdf_xz"], axes["vdf_yz"], axes["vdf_cb"]]
    cmap_axes = [axes["cmap_xy"], axes["cmap_xz"], axes["cmap_yz"], axes["cmap_cb"]]

    generate_ts_plot(ts_axes, ts_data, ci, coords, t0, t1)
    axvlines = []
    for ax in ts_axes:
        axvlines.append(ax.axvline(t0, linestyle="dashed"))

    for fnr in np.arange(t0, t1 + 0.1, 1):
        ts_glob_vdf_update(fnr)
        fig.savefig(outdir + "{}.png".format(int(fnr)), dpi=300, bbox_inches="tight")

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


def density_rel_to_mb(
    vlsvobj, cellid, nsw=1e6, vsw=(-750e3, 0, 0), Tsw=500e3, fmin=1e-16, dv=40e3
):

    # Read velocity cell keys and values from vlsv file
    velcels = vlsvobj.read_velocity_cells(cellid)
    vc_coords = vlsvobj.get_velocity_cell_coordinates(list(velcels.keys()))
    vc_vals = np.array(list(velcels.values()))

    ii_fm = np.where(vc_vals >= fmin)
    vc_vals = vc_vals[ii_fm]
    vc_coords = vc_coords[ii_fm, :][0, :, :]

    vc_coords_sw_frame = np.subtract(vc_coords, vsw)
    vc_coords_mag_sw_frame = np.linalg.norm(vc_coords_sw_frame, axis=-1)

    mb_vals = (
        nsw
        * (m_p / 2.0 / np.pi / kb / Tsw) ** (3.0 / 2)
        * np.exp(-m_p * vc_coords_mag_sw_frame**2 / 2.0 / kb / Tsw)
    )
    vc_vals_weighted = np.sqrt(mb_vals * vc_vals)

    res = np.sum(dv * dv * dv * vc_vals_weighted) / vlsvobj.read_variable(
        "proton/vg_rho", cellids=cellid
    )

    return res


def vspace_smasher(
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
    Function for reducing a 3D VDF to 2D
    (object) vlsvobj = Analysator VLSV file object
    (int) cellid = ID of cell whose VDF you want
    (str) operator = "x", "y", or "z", which velocity component to retain after reduction, or "magnitude" to get the distribution of speeds (untested)
    (float) dv = Velocity space resolution in m/s
    """

    # List of valid operators from which to get an index
    op_list = ["x", "y", "z"]
    op_pairs = [[1, 2], [0, 2], [0, 1]]

    bop_list = ["par", "perp1", "perp2"]

    # Read velocity cell keys and values from vlsv file
    velcels = vlsvobj.read_velocity_cells(cellid)
    vc_coords = vlsvobj.get_velocity_cell_coordinates(list(velcels.keys()))
    vc_vals = np.array(list(velcels.values()))

    ii_fm = np.where(vc_vals >= fmin)
    vc_vals = vc_vals[ii_fm]
    vc_coords = vc_coords[ii_fm, :][0, :, :]

    if operator in op_list:
        vc_coord_arr = vc_coords[:, op_pairs[op_list.index(operator)]]
    elif operator in bop_list:
        bxv = np.cross(b, v)
        bxbxv = np.cross(b, bxv)
        bxbxv = bxbxv / np.linalg.norm(bxbxv)
        bxv = bxv / np.linalg.norm(bxv)
        vc_coord_arr = np.array(
            [np.dot(vc_coords, b), np.dot(vc_coords, bxv), np.dot(vc_coords, bxbxv)]
        ).T
        vc_coord_arr = vc_coord_arr[:, op_pairs[bop_list.index(operator)]]

    vbins = np.sort(np.unique(vc_coords.flatten()))
    if vmin or vmax:
        vbins = np.arange(vmin - binw / 2, vmax + binw / 2 + binw / 4, binw)
    else:
        vbins = np.arange(
            np.min(vbins) - binw / 2, np.max(vbins) + binw / 2 + binw / 4, binw
        )

    vweights = vc_vals * dv

    # Integrate over the perpendicular direction
    hist, xedges, yedges = np.histogram2d(
        vc_coord_arr[:, 0], vc_coord_arr[:, 1], bins=[vbins, vbins], weights=vweights
    )

    # Return the 2D VDF values in units of s^2/m^5 as well as the bin edges to assist in plotting
    return (hist, xedges / 1e3, yedges / 1e3)


def vspace_rotator(
    vlsvobj,
    cellid,
    b=None,
    v=None,
    vlim=8e6,
    dv=40e3,
    fmin=1e-16,
):

    binw = dv

    # Read velocity cell keys and values from vlsv file
    velcels = vlsvobj.read_velocity_cells(cellid)
    vc_coords = vlsvobj.get_velocity_cell_coordinates(list(velcels.keys()))
    vc_vals = np.array(list(velcels.values()))

    ii_fm = np.where(vc_vals >= fmin)
    vc_vals = vc_vals[ii_fm]
    vc_coords = vc_coords[ii_fm, :][0, :, :]

    bxv = np.cross(b, v)
    bxbxv = np.cross(b, bxv)
    bxbxv = bxbxv / np.linalg.norm(bxbxv)
    bxv = bxv / np.linalg.norm(bxv)
    vc_coord_arr = np.array(
        [np.dot(vc_coords, b), np.dot(vc_coords, bxv), np.dot(vc_coords, bxbxv)]
    ).T

    vbins = np.arange(-vlim - binw / 2, vlim + binw / 2 + binw / 4, binw)

    hist, xedges, yedges, zedges = np.histogramdd(
        vc_coord_arr, bins=[vbins, vbins, vbins], weights=vc_vals
    )

    xmesh, ymesh, zmesh = np.meshgrid(
        xedges[:-1] + binw / 2, yedges[:-1] + binw / 2, zedges[:-1] + binw / 2
    )

    vc_vals = hist.flatten()
    vc_coords = np.array([xmesh.flatten(), ymesh.flatten(), zmesh.flatten()]).T

    ii_fm = np.where(vc_vals >= fmin)
    vc_vals = vc_vals[ii_fm]
    vc_coords = vc_coords[ii_fm, :][0, :, :]

    return (vc_coords, vc_vals)


def vspace_extracter(args):

    cellid, fnr, fmin = args

    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath_FIF + "bulk1.{}.vlsv".format(str(int(fnr)).zfill(7))
    )

    outdir = wrkdir_DNR + "vdf_txts/"

    try:
        velcels = vlsvobj.read_velocity_cells(cellid)
    except:
        return None
    vc_coords = vlsvobj.get_velocity_cell_coordinates(list(velcels.keys()))
    vc_vals = np.array(list(velcels.values()))

    ii_fm = np.where(vc_vals >= fmin)
    vc_vals = vc_vals[ii_fm]
    vc_coords = vc_coords[ii_fm, :][0, :, :]

    vc_x, vc_y, vc_z = vc_coords.T

    out_arr = np.array([vc_x, vc_y, vc_z, vc_vals]).T

    if not os.path.exists(outdir + "c{}".format(cellid)):
        try:
            os.makedirs(outdir + "c{}".format(cellid))
        except OSError:
            pass

    np.savetxt(outdir + "c{}/f{}.txt".format(cellid, fnr), out_arr)

    vlsvobj.optimize_clear_fileindex_for_cellid()


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
        bxv = bxv / np.linalg.norm(bxv)
        vc_coord_arr = np.dot(vc_coords, bxv)
    elif operator == "perp2":
        bxv = np.cross(b, v)
        bxbxv = np.cross(b, bxv)
        bxbxv = bxbxv / np.linalg.norm(bxbxv)
        vc_coord_arr = np.dot(vc_coords, bxbxv)

    # Create histogram bins, one for each unique coordinate of the chosen velocity component
    vbins = np.sort(np.unique(vc_coords.flatten()))
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


def ellipse_params(mean, cov, normal):
    normals = ["x", "y", "z"]
    idx = normals.index(normal)

    pairs = [[1, 2], [0, 2], [0, 1]]

    mean_proj = mean[pairs[idx]]
    cov_proj = cov[pairs[idx], :][:, pairs[idx]]

    vals, vecs = np.linalg.eigh(cov_proj)
    order = np.argsort(vals)[::-1]
    major_val, minor_val = vals[order]
    major_vec = vecs[:, order[0]]

    width, height = 2 * np.sqrt(major_val), 2 * np.sqrt(minor_val)
    angle = np.degrees(np.arctan2(major_vec[1], major_vec[0]))

    return (mean_proj, width, height, angle)

    # ellipse = Ellipse( mean, width, height, angle=angle,
    #                 edgecolor='black', facecolor='none', lw=2)

    # ax.add_patch(ellipse)
    # ax.plot(mean[0], mean[1], 'bo')


def plot_ellipses(means, covs, weights, ax, normal):

    edgecolors = [
        "#000000",
        "#377eb8",
        "#e41a1c",
        "#999999",
        "#a65628",
        "#4daf4a",
        "#ff7f00",
        "#f781bf",
        "#984ea3",
        "#dede00",
    ]
    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    niter = plot_gmm

    for idx in range(niter):
        mean, width, height, angle = ellipse_params(means[idx], covs[idx], normal)
        ellipse = mpatches.Ellipse(
            mean,
            width,
            height,
            angle=angle,
            edgecolor=edgecolors[idx],
            facecolor="none",
            lw=2,
        )
        ax.add_patch(ellipse)
        ax.plot(mean[0], mean[1], "o", color=edgecolors[idx])


def generate_vdf_plots(vdf_axes, vobj, ci):

    gmm_success = True

    print("Plot GMM is {}".format(plot_gmm))

    boxwidth = 3000e3
    fnr = int(vobj.read_parameter("time"))
    if plot_gmm:
        try:
            gmm_fit = np.loadtxt(
                wrkdir_DNR + "vdf_gmm/n{}/c{}/f{}.fit".format(plot_gmm, int(ci), fnr)
            )
            weights = []
            means = []
            covs = []
            traces = []
            for idx in range(plot_gmm):
                weights.append(gmm_fit[idx, 0])
                means.append(gmm_fit[idx, 1:4] / 1e3)
                covs.append(np.reshape(gmm_fit[idx, 4:13], (3, 3)) / 1e6)
                traces.append(np.trace(np.reshape(gmm_fit[idx, 4:13], (3, 3))))
            # weights_sorted = np.array(weights)[np.argsort(traces)]
            # means_sorted = np.array(means)[np.argsort(traces), :]
            # covs_sorted = np.array(covs)[np.argsort(traces), :, :]
            weights_sorted = weights
            means_sorted = means
            covs_sorted = covs
        except:
            gmm_success = False

    pt.plot.plot_vdf(
        axes=vdf_axes[0],
        vlsvobj=vobj,
        cellids=[ci],
        colormap="batlow",
        bvector=1,
        xy=1,
        slicethick=slicethick_g,
        box=[-boxwidth, boxwidth, -boxwidth, boxwidth],
        setThreshold=1e-16,
        scale=scale_g,
        fmin=1e-10,
        fmax=1e-4,
        contours=7,
        nocb=True,
        title="",
    )
    pt.plot.plot_vdf(
        axes=vdf_axes[1],
        vlsvobj=vobj,
        cellids=[ci],
        colormap="batlow",
        bvector=1,
        xz=1,
        slicethick=slicethick_g,
        box=[-boxwidth, boxwidth, -boxwidth, boxwidth],
        setThreshold=1e-16,
        scale=scale_g,
        fmin=1e-10,
        fmax=1e-4,
        contours=7,
        nocb=True,
        title="",
    )
    pt.plot.plot_vdf(
        axes=vdf_axes[2],
        vlsvobj=vobj,
        cellids=[ci],
        colormap="batlow",
        bvector=1,
        yz=1,
        slicethick=slicethick_g,
        box=[-boxwidth, boxwidth, -boxwidth, boxwidth],
        setThreshold=1e-16,
        scale=scale_g,
        fmin=1e-10,
        fmax=1e-4,
        contours=7,
        cbaxes=vdf_axes[3],
        cb_horizontal=True,
        title="",
    )
    if plot_gmm and gmm_success:
        plot_ellipses(means_sorted, covs_sorted, weights_sorted, vdf_axes[0], "z")
        plot_ellipses(means_sorted, covs_sorted, weights_sorted, vdf_axes[1], "y")
        plot_ellipses(means_sorted, covs_sorted, weights_sorted, vdf_axes[2], "x")


def generate_vdf_B_plots(vdf_axes, vobj, ci):

    boxwidth = 3000e3

    pt.plot.plot_vdf(
        axes=vdf_axes[0],
        vlsvobj=vobj,
        cellids=[ci],
        colormap="batlow",
        # bvector=1,
        bpara=1,
        slicethick=slicethick_g,
        box=[-boxwidth, boxwidth, -boxwidth, boxwidth],
        setThreshold=1e-16,
        scale=scale_g,
        fmin=1e-10,
        fmax=1e-4,
        contours=7,
        nocb=True,
        title="",
    )
    pt.plot.plot_vdf(
        axes=vdf_axes[1],
        vlsvobj=vobj,
        cellids=[ci],
        colormap="batlow",
        # bvector=1,
        bpara1=1,
        slicethick=slicethick_g,
        box=[-boxwidth, boxwidth, -boxwidth, boxwidth],
        setThreshold=1e-16,
        scale=scale_g,
        fmin=1e-10,
        fmax=1e-4,
        contours=7,
        nocb=True,
        title="",
    )
    pt.plot.plot_vdf(
        axes=vdf_axes[2],
        vlsvobj=vobj,
        cellids=[ci],
        colormap="batlow",
        # bvector=1,
        bperp=1,
        slicethick=slicethick_g,
        box=[-boxwidth, boxwidth, -boxwidth, boxwidth],
        setThreshold=1e-16,
        scale=scale_g,
        fmin=1e-10,
        fmax=1e-4,
        contours=7,
        cbaxes=vdf_axes[3],
        cb_horizontal=True,
        title="",
    )


def ext_bs_mp(ax, XmeshXY, YmeshXY, pass_maps):
    beta_star = pass_maps["vg_beta_star"]
    rho = pass_maps["proton/vg_rho"]
    pdynx = pass_maps["proton/vg_pdynx"]

    ax.contour(XmeshXY, YmeshXY, rho, [2e6], colors=["red"])
    # try:
    #     ax.contour(XmeshXY, YmeshXY, Tcore, [1.5e6], colors=["red"])
    # except:
    #     pass
    ax.contour(XmeshXY, YmeshXY, beta_star, [0.3], colors=["white"])
    ax.contour(
        XmeshXY, YmeshXY, pdynx, [0.5 * m_p * 1e6 * 750e3 * 750e3], colors=["black"]
    )


def generate_cmap_plots(cmap_axes, vobj, x0, y0, z0, limitedsize):

    boxwidth = 4
    fsaved = "yellow"

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
        nocb=True,
        colormap="batlow",
        scale=scale_g,
        tickinterval=1.0,
        normal="z",
        cutpointre=z0,
        # title="",
        limitedsize=limitedsize,
        external=ext_bs_mp,
        pass_vars=["vg_beta_star", "proton/vg_rho", "proton/vg_pdynx"],
        fsaved=fsaved,
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
        nocb=True,
        colormap="batlow",
        scale=scale_g,
        tickinterval=1.0,
        normal="y",
        cutpointre=y0,
        title="",
        limitedsize=limitedsize,
        external=ext_bs_mp,
        pass_vars=["vg_beta_star", "proton/vg_rho", "proton/vg_pdynx"],
        fsaved=fsaved,
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
        cbaxes=cmap_axes[3],
        cb_horizontal=True,
        colormap="batlow",
        scale=scale_g,
        tickinterval=1.0,
        normal="x",
        cutpointre=x0,
        title="",
        limitedsize=limitedsize,
        external=ext_bs_mp,
        pass_vars=["vg_beta_star", "proton/vg_rho", "proton/vg_pdynx"],
        fsaved=fsaved,
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
        ),
        fontsize=20,
        pad=10,
    )
    ts_axes[-1].set_xlabel("t [s]", fontsize=16, labelpad=10)
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
            ax.legend(
                loc="center left", bbox_to_anchor=(1.01, 0.5), ncols=ncols, fontsize=16
            )

    pdynx_peaks = argrelextrema(pdynx, np.greater)[0]
    pdyn_peaks = argrelextrema(ts_data[5, :], np.greater)[0]

    if pdynx_peaks.size > 0:
        pdynx_peak_times = t_arr[
            pdynx_peaks[pdynx[pdynx_peaks] >= 3 * tavg_x_arr[pdynx_peaks]]
        ]
    if pdyn_peaks.size > 0:
        pdyn_peak_times = t_arr[
            pdyn_peaks[ts_data[5, :][pdyn_peaks] >= 2 * tavg_arr[pdyn_peaks]]
        ]

    for idx, ax in enumerate(ts_axes):
        ax.grid()
        ax.set_ylabel(ylabels[idx], fontsize=16, labelpad=10)
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
        ax.tick_params(labelsize=16)
        [ax.axvline(x, color="red", linestyle="dotted") for x in pdyn_peak_times]
        [ax.axvline(x, color="green", linestyle="dotted") for x in pdynx_peak_times]


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
    axes["cmap_cb"] = fig.add_axes((0, -0.03, 0.2, 0.01))
    axes["vdf_cb"] = fig.add_axes((0.25, -0.03, 0.2, 0.01))
    return axes


def make_yz_slice_one(fnr):

    global fnr_g
    fnr_g = fnr
    global xcut_g

    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath_FIF + "bulk1.{}.vlsv".format(str(int(fnr)).zfill(7))
    )

    fig, ax_list = plt.subplots(2, 2, figsize=(12, 12), layout="compressed")
    ax_flat = ax_list.flatten()
    cbax = fig.add_axes((1.01, 0, 0.05, 1))

    # xcuts = [11, 11.5, 12, 12.5]
    xcuts = [11, 12, 12.5, 13]

    pdynsw = m_p * 1e6 * 750e3 * 750e3 * 1e9

    for idx in range(4):
        xcut_g = xcuts[idx]
        pt.plot.plot_colormap3dslice(
            axes=ax_flat[idx],
            vlsvobj=vlsvobj,
            cbaxes=cbax,
            var="proton/vg_Pdyn",
            vmin=0.0 * pdynsw,
            vmax=2.0 * pdynsw,
            lin=5,
            vscale=1e9,
            cbtitle="$P_\\mathrm{dyn}$ [nPa]",
            usesci=0,
            boxre=[-15, 15, -15, 15],
            # nocb=True,
            colormap="roma_r",
            scale=1.3,
            tickinterval=3.0,
            normal="x",
            cutpointre=xcuts[idx],
            limitedsize=True,
            external=ext_jet,
            pass_vars=[
                "vg_beta_star",
                "proton/vg_rho",
                "proton/vg_pdynx",
                "proton/vg_pdyn",
            ],
        )
        ax_flat[idx].label_outer()

    fig.savefig(
        wrkdir_DNR + "xcuts/{}.png".format(int(fnr)), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def save_yz_slice_one(fnr):

    normal = "x"
    # normal = [0.8660254037844387, -0.5, 0]

    global fnr_g
    fnr_g = fnr
    global xcut_g

    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath_FIF + "bulk1.{}.vlsv".format(str(int(fnr)).zfill(7))
    )

    fig, ax_list = plt.subplots(2, 2, figsize=(15, 15), layout="compressed")
    ax_flat = ax_list.flatten()
    cbax = fig.add_axes((1.01, 0, 0.05, 1))

    # xcuts = [11, 11.5, 12, 12.5]
    xcuts = [13, 13.5, 14, 14.5]

    pdynsw = m_p * 1e6 * 750e3 * 750e3 * 1e9

    for idx in range(4):
        xcut_g = xcuts[idx]
        pt.plot.plot_colormap3dslice(
            axes=ax_flat[idx],
            vlsvobj=vlsvobj,
            cbaxes=cbax,
            var="proton/vg_Pdyn",
            vmin=0.0 * pdynsw,
            vmax=2 * pdynsw,
            lin=5,
            vscale=1e9,
            cbtitle="$P_\\mathrm{dyn}$ [nPa]",
            usesci=0,
            boxre=[-15, 15, -15, 15],
            # nocb=True,
            colormap="roma_r",
            scale=1.3,
            tickinterval=3.0,
            normal=normal,
            cutpointre=xcuts[idx],
            limitedsize=True,
            external=ext_save,
            pass_vars=["proton/vg_pdynx", "proton/vg_pdyn"],
        )

    plt.close(fig)


def ext_save(ax, XmeshXY, YmeshXY, pass_maps):
    pdyn = pass_maps["proton/vg_pdyn"]
    pdynx = pass_maps["proton/vg_pdynx"]

    np.savetxt(
        "/wrk-vakka/users/jesuni/jets_3D/txts/xcuts/{}_{}.pdyn".format(fnr_g, xcut_g),
        pdyn,
    )
    np.savetxt(
        "/wrk-vakka/users/jesuni/jets_3D/txts/xcuts/{}_{}.pdynx".format(fnr_g, xcut_g),
        pdynx,
    )


def ext_jet(ax, XmeshXY, YmeshXY, pass_maps):

    beta_star = pass_maps["vg_beta_star"]
    rho = pass_maps["proton/vg_rho"]
    pdynx = pass_maps["proton/vg_pdynx"]
    pdyn = pass_maps["proton/vg_pdyn"]

    pdynx_avg = np.loadtxt(
        "/wrk-vakka/users/jesuni/jets_3D/txts/xcut_avgs/{}_{}.pdynx".format(
            fnr_g, xcut_g
        )
    )
    pdyn_avg = np.loadtxt(
        "/wrk-vakka/users/jesuni/jets_3D/txts/xcut_avgs/{}_{}.pdyn".format(
            fnr_g, xcut_g
        )
    )

    ax.contour(XmeshXY, YmeshXY, rho, [2e6], colors=["orange"])
    ax.contour(XmeshXY, YmeshXY, beta_star, [0.3], colors=["white"])
    ax.contour(
        XmeshXY, YmeshXY, pdynx, [0.5 * m_p * 1e6 * 750e3 * 750e3], colors=["black"]
    )
    ax.contour(XmeshXY, YmeshXY, pdynx / pdynx_avg, [3.0], colors=["green"])
    ax.contour(XmeshXY, YmeshXY, pdyn / pdyn_avg, [2.0], colors=["red"])


def calc_xcut_avgs(xcut):

    fnr_range_full = np.arange(600, 991, 1)
    fnr_range = np.arange(690, 901, 1)
    template_arr_shape = np.loadtxt(
        "/wrk-vakka/users/jesuni/jets_3D/txts/xcuts/600_11.pdyn"
    ).shape
    full_arr = np.zeros(
        (template_arr_shape[0], template_arr_shape[1], fnr_range_full.size), dtype=float
    )
    full_arr_x = np.zeros(
        (template_arr_shape[0], template_arr_shape[1], fnr_range_full.size), dtype=float
    )
    for idx in range(fnr_range_full.size):
        full_arr[:, :, idx] = np.loadtxt(
            "/wrk-vakka/users/jesuni/jets_3D/txts/xcuts/{}_{}.pdyn".format(
                fnr_range_full[idx], xcut
            )
        )
        full_arr_x[:, :, idx] = np.loadtxt(
            "/wrk-vakka/users/jesuni/jets_3D/txts/xcuts/{}_{}.pdynx".format(
                fnr_range_full[idx], xcut
            )
        )
    for idx in range(fnr_range.size):
        avg = np.nanmean(full_arr[:, :, idx : idx + 180], axis=-1)
        avgx = np.nanmean(full_arr_x[:, :, idx : idx + 180], axis=-1)
        np.savetxt(
            "/wrk-vakka/users/jesuni/jets_3D/txts/xcut_avgs/{}_{}.pdyn".format(
                fnr_range[idx], xcut
            ),
            avg,
        )
        np.savetxt(
            "/wrk-vakka/users/jesuni/jets_3D/txts/xcut_avgs/{}_{}.pdynx".format(
                fnr_range[idx], xcut
            ),
            avgx,
        )


def make_yz_anim(n_processes=16, sav=False):

    fnr_range = np.arange(690, 901, 1)
    # fnr_range = np.arange(690, 701, 1)

    if sav:
        fnr_range = np.arange(600, 991, 1)
        with Pool(processes=n_processes) as pool:
            pool.map(save_yz_slice_one, fnr_range)
    else:
        outfilename = "/wrk-vakka/users/jesuni/jets_3D/yz_cuts.mp4"
        with Pool(processes=n_processes) as pool:
            pool.map(make_yz_slice_one, fnr_range)

        subprocess.run(
            "cat $(find /wrk-vakka/users/jesuni/jets_3D/xcuts -maxdepth 1 -name '*.png' | sort -V) | ffmpeg -framerate 5 -i - -b:v 2500k -vf scale=1600:-2 -y {}".format(
                outfilename
            ),
            shell=True,
        )
        subprocess.run("rm /wrk-vakka/users/jesuni/jets_3D/xcuts/* -f", shell=True)


def make_shell_anim(n_processes=16, shellre=13.5):

    fnr_range = np.arange(690, 901, 1)

    outfilename = "/wrk-vakka/users/jesuni/jets_3D/shell_{}.mp4".format(shellre)
    args = [[fnr, shellre] for fnr in fnr_range]
    with Pool(processes=n_processes) as pool:
        pool.map(make_shell_map_one, args)

    subprocess.run(
        "cat $(find /wrk-vakka/users/jesuni/jets_3D/shells/{} -maxdepth 1 -name '*.png' | sort -V) | ffmpeg -framerate 5 -i - -b:v 2500k -vf scale=1600:-2 -y {}".format(
            shellre, outfilename
        ),
        shell=True,
    )
    subprocess.run(
        "rm /wrk-vakka/users/jesuni/jets_3D/shells/{}/* -f".format(shellre), shell=True
    )


def spherical_to_cartesian(r, theta, phi):

    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)

    return (x, y, z)


def find_bs(vlsvobj, r0, theta, phi, dr=1000e3, tol=1e-3, maxiter=1000):

    coord = np.array(spherical_to_cartesian(r0, theta, phi))
    u = coord / np.linalg.norm(coord)

    rho_thresh = 2e6
    iter = 0

    rho = vlsvobj.read_interpolated_variable("proton/vg_rho", coord)
    diff = np.abs(rho - rho_thresh)
    old_diff = np.abs(rho - rho_thresh)

    while np.abs(diff) / rho_thresh >= tol:
        coord = coord + u * dr
        rho = vlsvobj.read_interpolated_variable("proton/vg_rho", coord)
        diff = np.abs(rho - rho_thresh)
        # print("BS", diff, old_diff)
        if diff > old_diff:
            dr = -dr / 2.0

        iter += 1
        if iter > maxiter:
            break

    return coord


def find_mp(vlsvobj, r0, theta, phi, dr=1000e3, tol=1e-3, maxiter=1000):

    coord = np.array(spherical_to_cartesian(r0, theta, phi))
    u = coord / np.linalg.norm(coord)

    bstar_thresh = 0.3
    iter = 0

    bstar = vlsvobj.read_interpolated_variable("proton/vg_beta_star", coord)
    diff = np.abs(bstar - bstar_thresh)
    old_diff = np.abs(bstar - bstar_thresh)

    while np.abs(diff) >= tol:
        coord = coord + u * dr
        bstar = vlsvobj.read_interpolated_variable("proton/vg_beta_star", coord)
        diff = np.abs(bstar - bstar_thresh)
        # print("MP", diff, old_diff)
        if diff > old_diff:
            dr = -dr / 2.0

        iter += 1
        if iter > maxiter:
            break

    return coord


def polyfit_2d(coord_arr):

    x, y, z = coord_arr.T

    mix_arr = np.array(
        [
            1.0 + 0 * y,
            y,
            z,
            y**2,
            z * y**2,
            z**2 * y**2,
            z**2 * y,
            z**2,
            z * y,
        ]
    ).T

    coeff, r, rank, s = lstsq(mix_arr, x)

    return coeff


def polyval_2d(coeff, y, z):

    return (
        coeff[0]
        + coeff[1] * y
        + coeff[2] * z
        + coeff[3] * y**2
        + coeff[4] * z * y**2
        + coeff[5] * z**2 * y**2
        + coeff[6] * z**2 * y
        + coeff[7] * z**2
        + coeff[8] * z * y
    )


def make_bs_mp_map_one(args):

    fnr, idx = args

    outdir = wrkdir_DNR + "bs_mp"
    create_dir_if_not_exist(outdir)

    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath_FIF + "bulk1.{}.vlsv".format(str(int(fnr)).zfill(7))
    )

    phi_range = np.linspace(-np.deg2rad(30), np.deg2rad(30), 10)
    theta_range = np.linspace(-np.deg2rad(30), np.deg2rad(30), 10)
    thetamesh, phimesh = np.meshgrid(theta_range, phi_range)
    thetaflat = thetamesh.flatten()
    phiflat = phimesh.flatten()

    bs_xyz = np.zeros((thetaflat.size, 3), dtype=float)
    # mp_xyz = np.zeros((thetaflat.size, 3), dtype=float)

    for idx in range(thetaflat.size):
        theta = thetaflat[idx]
        phi = phiflat[idx]
        bs_xyz[idx] = find_bs(vlsvobj, 14 * r_e, theta, phi, dr=100e3, tol=0.01) / r_e
        # mp_xyz[idx] = find_mp(vlsvobj, 10 * r_e, theta, phi, dr=100e3, tol=0.01) / r_e

    bs_coeff = polyfit_2d(bs_xyz)
    # mp_coeff = polyfit_2d(mp_xyz)

    np.savetxt(outdir + "/{}.bs".format(int(fnr)), bs_coeff)
    # np.savetxt(outdir + "/{}.mp".format(int(fnr)), mp_coeff)


def make_bs_mp_map_all(fnr0, fnr1, n_processes=16):

    fnr_arr = np.arange(fnr0, fnr1 + 0.1, 1, dtype=int)
    args_list = []
    for idx in range(fnr_arr.size):
        args_list.append([fnr_arr[idx], idx])

    with Pool(processes=n_processes) as pool:
        pool.map(make_bs_mp_map_one, args_list)


def plot_bs_map_all():

    fnr_arr = np.arange(600, 991 + 0.1, 1, dtype=int)
    y_arr = np.linspace(-20, 20, 100)
    z_arr = np.linspace(-20, 20, 100)

    outdir = wrkdir_DNR + "Figs/bs_mp"
    create_dir_if_not_exist(outdir)

    for fnr in fnr_arr:
        coeff = np.loadtxt(wrkdir_DNR + "bs_mp/{}.bs".format(fnr))
        x_of_y = polyval_2d(coeff, y_arr, np.zeros_like(z_arr))
        x_of_z = polyval_2d(coeff, np.zeros_like(y_arr), z_arr)
        fig, ax_list = plt.subplots(1, 2, figsize=(20, 10), layout="compressed")
        ax_list[0].plot(x_of_y, y_arr)
        ax_list[1].plot(x_of_z, z_arr)
        for ax in ax_list:
            ax.grid()
            ax.set_xlabel("X")
            ax.set_xlim(0, 20)
        ax_list[0].set_ylabel("Y")
        ax_list[1].set_ylabel("Z")

        fig.savefig(outdir + "/{}.png".format(fnr), dpi=300, bbox_inches="tight")
        plt.close(fig)


def make_shell_map_one(args):

    fnr, shellre = args

    outdir = wrkdir_DNR + "shells/{}".format(shellre)

    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath_FIF + "bulk1.{}.vlsv".format(str(int(fnr)).zfill(7))
    )

    yrange = np.arange(-15 * r_e, 15 * r_e + 1, 1000e3)
    zrange = np.arange(-15 * r_e, 15 * r_e + 1, 1000e3)

    pdyn_sw = m_p * 1e6 * 750e3 * 750e3

    ymesh, zmesh = np.meshgrid(yrange, zrange)

    pdyn_arr = np.empty_like(ymesh, dtype=float)
    pdyn_arr.fill(np.nan)

    pdynx_arr = np.empty_like(ymesh, dtype=float)
    pdynx_arr.fill(np.nan)

    for idy in range(yrange.size):
        for idz in range(zrange.size):
            y = ymesh[idy, idz]
            z = zmesh[idy, idz]
            xsq = (shellre * r_e) ** 2 - y**2 - z**2
            if xsq < 0:
                continue
            x = np.sqrt(xsq)
            pdyn_arr[idy, idz] = vlsvobj.read_interpolated_variable(
                "proton/vg_pdyn", [x, y, z]
            )
            pdynx_arr[idy, idz] = vlsvobj.read_interpolated_variable(
                "proton/vg_pdynx", [x, y, z]
            )

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), layout="compressed")
    ax.set_aspect(1)
    cax = fig.add_axes((1.01, 0, 0.02, 1))
    im = ax.pcolormesh(
        ymesh / r_e,
        zmesh / r_e,
        pdyn_arr / 1e-9,
        cmap="roma_r",
        vmin=0,
        vmax=2,
        shading="nearest",
    )
    ax.contour(ymesh / r_e, zmesh / r_e, pdynx_arr, [0.5 * pdyn_sw], colors=["black"])
    fig.colorbar(im, cax=cax)

    fig.savefig(outdir + "/{}.png".format(fnr), dpi=300, bbox_inches="tight")
    plt.close(fig)
