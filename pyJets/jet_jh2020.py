import sys
import matplotlib.style
import matplotlib as mpl
import jet_aux as jx

if sys.version_info.major == 3:
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=["black", jx.medium_blue, jx.dark_blue, jx.orange]
    )
elif sys.version_info.major == 2:
    mpl.rcParams["axes.color_cycle"] = [
        "black",
        jx.medium_blue,
        jx.dark_blue,
        jx.orange,
    ]
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

r_e = 6.371e6
m_p = 1.672621898e-27

wrkdir_DNR = os.environ["WRK"] + "/"
homedir = os.environ["HOME"] + "/"
try:
    vlasdir = os.environ["VLAS"]
except:
    vlasdir = "/proj/vlasov"


def get_cut_through(
    runid,
    start,
    stop,
    min_cellid,
    max_cellid,
    vars=["Pdyn", "rho", "v", "B", "Temperature"],
    save=True,
    custom=False,
):

    outputdir = wrkdir_DNR + "timeseries/{}/{}_{}/".format(
        runid, min_cellid, max_cellid
    )

    var_list = [
        "rho",
        "v",
        "vx",
        "vy",
        "vz",
        "B",
        "Bx",
        "By",
        "Bz",
        "Pdyn",
        "TParallel",
        "TPerpendicular",
        "beta",
    ]
    vlsv_var_list = [
        "rho",
        "v",
        "v",
        "v",
        "v",
        "B",
        "B",
        "B",
        "B",
        "Pdyn",
        "TParallel",
        "TPerpendicular",
        "beta",
    ]
    op_list = [
        "pass",
        "magnitude",
        "x",
        "y",
        "z",
        "magnitude",
        "x",
        "y",
        "z",
        "pass",
        "pass",
        "pass",
        "pass",
    ]

    vars = vars + ["Mmsx", "TNonBackstream"]
    if custom:
        vars = ["Pdyn", "rho", "Pressure", "Pmag", "Ptot"] + ["Mmsx", "TNonBackstream"]

    if custom:
        cellid_range = jet_424_center_cells()
        outputdir = wrkdir_DNR + "timeseries/{}/{}_{}/".format(runid, "custom", "424")
    else:
        cellid_range = np.arange(min_cellid, max_cellid + 1, dtype=int)

    output_arr = np.zeros((len(vars), stop - start + 1, cellid_range.size))

    bulkpath = jx.find_bulkpath(runid)

    for filenr in range(start, stop + 1):
        print(filenr)
        bulkname = "bulk.{}.vlsv".format(str(filenr).zfill(7))
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath + bulkname)
        for m in range(len(vars)):
            var = vars[m]
            if var in var_list:
                vlsv_var = vlsv_var_list[var_list.index(var)]
                vlsv_op = op_list[var_list.index(var)]
                output_arr[m][filenr - start] = vlsvobj.read_variable(
                    vlsv_var, operator=vlsv_op, cellids=cellid_range
                )
            else:
                output_arr[m][filenr - start] = vlsvobj.read_variable(
                    var, cellids=cellid_range
                )

    if save:
        if not os.path.exists(outputdir):
            try:
                os.makedirs(outputdir)
            except OSError:
                pass
        np.save(outputdir + "{}_{}".format(start, stop), output_arr)
        return None
    else:
        return output_arr


def find_one_jet():

    nrange = range(1, 3000)
    for n in nrange:
        try:
            jetobj = jio.PropReader(str(n).zfill(5), "ABC", transient="slamsjet")
        except:
            continue
        if (
            jetobj.read("time")[0] == 412.5
            and not jetobj.read("is_slams").astype(bool).any()
        ):
            print(n)

    return None


def find_markus_FCS(dist_thresh=1e5):

    markus_time = 428.0
    markus_pos = np.array([7.0e7, -2.7e7])

    nrange = range(1, 3000)
    for n in nrange:
        try:
            jetobj = jio.PropReader(str(n).zfill(5), "ABC", transient="slams")
        except:
            continue
        if "splinter" in jetobj.meta or "merger" in jetobj.meta:
            continue
        fcs_times = jetobj.read("time")
        fcs_x = jetobj.read("x_mean")
        fcs_y = jetobj.read("y_mean")
        if markus_time in fcs_times:
            if (
                np.abs(jetobj.read_at_time("x_mean", markus_time) * r_e - markus_pos[0])
                < dist_thresh
                and np.abs(
                    jetobj.read_at_time("y_mean", markus_time) * r_e - markus_pos[1]
                )
                < dist_thresh
            ):
                text_arr = np.array([fcs_times, fcs_x, fcs_y]).T
                np.savetxt(wrkdir_DNR + "markus_fcs.txt", text_arr, fmt="%.2f")
                print(fcs_times)
                print(fcs_x)
                print(fcs_y)
                print(str(n).zfill(5))

    return None


def jet_424_center_cells():

    jetobj = jio.PropReader(str(424).zfill(5), "ABC", transient="slamsjet")
    vlsvobj = pt.vlsvfile.VlsvReader(vlasdir + "/2D/ABC/bulk/bulk.0000825.vlsv")
    x_arr = jetobj.read("x_mean") * r_e
    y_arr = jetobj.read("y_mean") * r_e
    z_arr = np.zeros_like(x_arr)

    coords = np.array([x_arr, y_arr, z_arr]).T
    cells = np.array([vlsvobj.get_cellid(coord) for coord in coords])

    return cells


def jh2020_cut_plot(runid, filenr, min_cellid=1814480, max_cellid=1814540):

    bulkpath = jx.find_bulkpath(runid)
    bulkname = "bulk.{}.vlsv".format(str(filenr).zfill(7))
    vlsvobj = pt.vlsvfile.VlsvReader(bulkpath + bulkname)

    cell_range = np.arange(min_cellid, max_cellid + 1)
    x_range = np.array(
        [jx.get_cell_coordinates(runid, cell)[0] / r_e for cell in cell_range]
    )
    y = jx.get_cell_coordinates(runid, cell_range[0])[1] / r_e

    var_list = ["rho", "pdyn", "B", "v", "TParallel", "TPerpendicular"]
    norm_list = [1.0e6, 1.0e-9, 1.0e-9, 1.0e3, 1.0e6, 1.0e6]
    label_list = [
        "$\mathrm{\\rho~[cm^{-3}]}$",
        "$\mathrm{P_{dyn}~[nPa]}$",
        "$\mathrm{B~[nT]}$",
        "$\mathrm{v~[kms^{-1}]}$",
        "$\mathrm{T~[MK]}$",
    ]
    lim_list = [(0, 30), (0, 8), (-35, 35), (-650, 650), (0, 20)]
    color_list = ["black", jx.medium_blue, jx.dark_blue, jx.orange]

    annot_list_list = [
        [""],
        [""],
        ["B", "Bx", "By", "Bz"],
        ["v", "vx", "vy", "vz"],
        ["TPar", "TPerp"],
    ]

    raw_data_list = [
        vlsvobj.read_variable(var, cellids=cell_range) / norm_list[var_list.index(var)]
        for var in var_list
    ]

    rho = raw_data_list[0]
    pdyn = raw_data_list[1]
    TPar = raw_data_list[4]
    TPerp = raw_data_list[5]
    v = raw_data_list[3]
    vmag = np.linalg.norm(v, axis=-1)
    B = raw_data_list[2]
    Bmag = np.linalg.norm(B, axis=-1)

    Ttot = np.array([TPar, TPerp]).T
    vtot = np.vstack((vmag, v.T)).T
    Btot = np.vstack((Bmag, B.T)).T

    x_1 = x_range
    x_2 = np.array((x_1, x_1)).T
    x_4 = np.array((x_1, x_1, x_1, x_1)).T

    x_list = [x_1, x_1, x_4, x_4, x_2]
    data_list = [rho, pdyn, Btot, vtot, Ttot]

    plt.ioff()

    fig, ax_list = plt.subplots(len(data_list), 1, figsize=(10, 15), sharex=True)
    fig.suptitle("Y = {:.3f} Re\nt = {} s".format(y, filenr / 2), fontsize=20)

    for n in range(len(data_list)):
        ann_list = annot_list_list[n]
        ax = ax_list[n]
        ax.grid()
        ax.set_xlim(x_range[0], x_range[-1])
        ax.set_ylim(lim_list[n])
        x = x_list[n]
        data = data_list[n]
        ax.tick_params(labelsize=15)
        ax.plot(x, data)
        ax.set_ylabel(label_list[n], fontsize=20)
        if n == len(var_list) - 1:
            ax.set_xlabel("$\mathrm{X~[R_e]}$", fontsize=20)
        for m in range(len(ann_list)):
            ax.annotate(
                ann_list[m],
                xy=(0.8 + m * 0.2 / len(ann_list), 0.05),
                xycoords="axes fraction",
                color=color_list[m],
            )

    if not os.path.exists(wrkdir_DNR + "Figures/jh2020"):
        try:
            os.makedirs(wrkdir_DNR + "Figures/jh2020")
        except OSError:
            pass

    fig.savefig(
        wrkdir_DNR
        + "Figures/jh2020/cut_{}_{}_{}.png".format(filenr, min_cellid, max_cellid)
    )
    plt.close(fig)

    return None


def event_for_mesh(runid, filenr, y, minx, maxx):

    event_props = np.array(
        jio.eventprop_read(runid, filenr, transient="slamsjet"), dtype=float
    )
    x_arr = event_props[:, 1]
    y_arr = event_props[:, 2]
    y_arr = y_arr[np.logical_and(x_arr < maxx, x_arr > minx)]
    x_arr = x_arr[np.logical_and(x_arr < maxx, x_arr > minx)]
    if np.min(np.abs(y_arr - y)) < 0.5:
        return x_arr[np.argmin(np.abs(y_arr - y))]
    else:
        return np.nan


def event_424_cut(time=825):

    var_list = ["Pdyn", "rho", "Pressure", "Pmag", "Ptot"]
    norm_list = [1.0e-9, 1.0e6, 1.0e-9, 1.0e-9, 1.0e-9]
    cell_arr = jet_424_center_cells()
    x_arr = np.arange(cell_arr.size)
    vmin_list = [0.0, 0, 1, 0.0, 1]
    vmax_list = [2.5, 20, 2.5, 0.25, 4.5]

    data_arr = np.load(
        wrkdir_DNR
        + "/timeseries/{}/{}_{}/{}_{}.npy".format("ABC", "custom", "424", 725, 925)
    )

    plt.ioff()

    fig, ax_list = plt.subplots(len(var_list), 1, figsize=(10, 10), sharex=True)

    for n in range(len(var_list)):
        data = data_arr[n][time - 725] / norm_list[n]
        ax = ax_list[n]
        ax.tick_params(labelsize=15)
        ax.plot(x_arr, data)
        ax.set_xlim(x_arr[0], x_arr[-1])
        ax.set_ylabel(var_list[n], fontsize=15)
        ax.set_ylim(vmin_list[n], vmax_list[n])
        ax.grid()
    ax_list[-1].set_xlabel("Pos along path", fontsize=20)
    ax_list[0].set_title("Time = {}s".format(float(time) / 2), fontsize=20)

    if not os.path.exists(wrkdir_DNR + "Figures/jh2020/event_424_cut"):
        try:
            os.makedirs(wrkdir_DNR + "Figures/jh2020/event_424_cut")
        except OSError:
            pass
    fig.savefig(wrkdir_DNR + "Figures/jh2020/event_424_cut/{}.png".format(time))
    plt.close(fig)

    return None


def jh2020_fig2_mesh(
    runid="ABC",
    start=400,
    stop=799,
    min_cellid=1814480,
    max_cellid=1814540,
    fromfile=True,
    clip="none",
    custom=False,
):

    var_list = ["Pdyn", "rho", "v", "B", "Temperature"]
    norm_list = [1.0e-9, 1.0e6, 1.0e3, 1.0e-9, 1.0e6]
    if custom:
        var_list = ["Pdyn", "rho", "Pressure", "Pmag", "Ptot"]
        norm_list = [1.0e-9, 1.0e6, 1.0e-9, 1.0e-9, 1.0e-9]

    if clip == "none":
        vmin_list = [0, 0, 0, 0, 0]
        vmax_list = [8, 30, 650, 35, 20]
    elif clip == "high":
        vmin_list = [0, 0, 0, 0, 0]
        vmax_list = [4.5, 6.6, 600, 15, 5]
    elif clip == "low":
        vmin_list = [1, 6.6, 150, 5, 0.5]
        vmax_list = [8, 30, 650, 35, 20]
    elif clip == "optimal":
        vmin_list = [0.0, 3.3, 100, 5, 0.5]
        vmax_list = [4.5, 20, 700, 20, 15]
    if custom:
        vmin_list = [0.0, 3.3, 0.2, 0.0, 1.0]
        vmax_list = [3, 20, 2, 0.25, 4]

    cell_arr = np.arange(min_cellid, max_cellid + 1, dtype=int)
    x_arr = np.array(
        [jx.get_cell_coordinates(runid, cell)[0] / r_e for cell in cell_arr]
    )
    if custom:
        cell_arr = jet_424_center_cells()
        x_arr = np.arange(cell_arr.size)
    y = jx.get_cell_coordinates(runid, cell_arr[0])[1] / r_e
    time_arr = np.arange(start, stop + 1) / 2.0
    XmeshXT, TmeshXT = np.meshgrid(x_arr, time_arr)

    if min_cellid == 1814480 and not custom:
        eventx_arr = np.array(
            [
                event_for_mesh(runid, fnr, y, x_arr[0], x_arr[-1])
                for fnr in np.arange(start, stop + 1, dtype=int)
            ]
        )
    elif min_cellid == 1784477 and not custom:
        onejet_obj = jio.PropReader(str(424).zfill(5), "ABC", transient="slamsjet")
        ox = onejet_obj.read("x_mean")
        oy = onejet_obj.read("y_mean")
        ot = onejet_obj.read("time")
        ox = ox[np.abs(oy - 0.6212) <= 0.5]
        ot = ot[np.abs(oy - 0.6212) <= 0.5]

    rho_sw = 3.3e6
    T_sw = 0.5e6

    if custom:
        data_arr = np.load(
            wrkdir_DNR
            + "/timeseries/{}/{}_{}/{}_{}.npy".format(
                runid, "custom", "424", start, stop
            )
        )
    else:
        if not fromfile:
            data_arr = get_cut_through(
                runid, start, stop, min_cellid, max_cellid, vars=var_list, save=False
            )
        else:
            data_arr = np.load(
                wrkdir_DNR
                + "/timeseries/{}/{}_{}/{}_{}.npy".format(
                    runid, min_cellid, max_cellid, start, stop
                )
            )

    rho_mask = (data_arr[1] >= 2 * rho_sw).astype(int)
    mms_mask = (data_arr[-2] <= 1).astype(int)
    tcore_mask = (data_arr[-1] >= 3 * T_sw).astype(int)

    plt.ioff()

    fig, ax_list = plt.subplots(
        1, len(var_list), figsize=(20, 10), sharex=True, sharey=True
    )
    im_list = []
    cb_list = []

    for n in range(len(var_list)):
        data = data_arr[n] / norm_list[n]
        ax = ax_list[n]
        if min_cellid == 1814480 and not custom:
            ax.axhline(328, color="black", linewidth=0.8)
            ax.axhline(337, color="black", linewidth=0.8)
            ax.axhline(345, color="black", linewidth=0.8)
        elif min_cellid == 1814480 + 60000 + 10 and not custom:
            ax.axhline(365, color="black", linewidth=0.8)
            ax.axhline(370, color="black", linewidth=0.8)
            ax.axhline(360, color="black", linewidth=0.8)
        elif min_cellid == 1784477 and not custom:
            ax.axhline(412.5, color="black", linewidth=0.8)
        if custom:
            ax.axhline(412.5, color="black", linewidth=0.8)
            ax.axhline(447.5, color="black", linewidth=0.8)
            ax.plot(
                [x_arr[0], x_arr[-1]],
                [412.5, 447.5],
                color="black",
                linewidth=0.8,
                linestyle="dashed",
            )
        im_list.append(
            ax.pcolormesh(x_arr, time_arr, data, vmin=vmin_list[n], vmax=vmax_list[n])
        )
        cb_list.append(fig.colorbar(im_list[n], ax=ax))
        ax.contour(XmeshXT, TmeshXT, rho_mask, [0.5], linewidths=1.0, colors="black")
        ax.contour(XmeshXT, TmeshXT, mms_mask, [0.5], linewidths=1.0, colors=jx.violet)
        ax.contour(
            XmeshXT, TmeshXT, tcore_mask, [0.5], linewidths=1.0, colors=jx.orange
        )
        ax.tick_params(labelsize=15)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        # ax.xaxis.set_major_locator(MaxNLocator(nbins=6,prune="lower"))
        ax.set_title(var_list[n], fontsize=20)
        if min_cellid == 1814480 and not custom:
            ax.plot(eventx_arr, time_arr, "o", color="red", markersize=2)
        elif min_cellid == 1784477 and not custom:
            ax.plot(ox, ot, "o", color="red", markersize=2)
        ax.set_xlim(x_arr[0], x_arr[-1])
        if n == 0:
            ax.set_ylabel("Simulation time [s]", fontsize=20)
            if custom:
                ax.set_xlabel("Pos along path", fontsize=20)
            else:
                ax.set_xlabel("$\mathrm{X~[R_e]}$", fontsize=20)

    fig.suptitle("Y = {:.3f} Re".format(y), fontsize=20)

    if not os.path.exists(wrkdir_DNR + "Figures/jh2020"):
        try:
            os.makedirs(wrkdir_DNR + "Figures/jh2020")
        except OSError:
            pass
    if custom:
        fig.savefig(
            wrkdir_DNR + "Figures/jh2020/fig2_mesh_{}_clip{}.png".format("custom", clip)
        )
    else:
        fig.savefig(
            wrkdir_DNR
            + "Figures/jh2020/fig2_mesh_{}_clip{}.png".format(min_cellid, clip)
        )
    plt.close(fig)

    return None


def get_timeseries_data(runid, start, stop, cellid):

    outputdir = wrkdir_DNR + "timeseries/{}/{}/".format(runid, cellid)

    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    bulkpath = jx.find_bulkpath(runid)
    var_list = [
        "rho",
        "v",
        "v",
        "v",
        "v",
        "B",
        "B",
        "B",
        "B",
        "Pdyn",
        "TParallel",
        "TPerpendicular",
        "beta",
    ]
    op_list = [
        "pass",
        "magnitude",
        "x",
        "y",
        "z",
        "magnitude",
        "x",
        "y",
        "z",
        "pass",
        "pass",
        "pass",
        "pass",
    ]
    output_arr = np.zeros((stop - start + 1, len(var_list) + 1))
    for filenr in range(start, stop + 1):
        bulkname = "bulk.{}.vlsv".format(str(filenr).zfill(7))
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath + bulkname)
        output_arr[filenr - start][0] = vlsvobj.read_parameter("t")
        for n in range(len(var_list)):
            data = vlsvobj.read_variable(
                var_list[n], operator=op_list[n], cellids=cellid
            )
            output_arr[filenr - start][n + 1] = data

    np.savetxt(outputdir + "{}_{}".format(start, stop), output_arr)

    return None


def mag_thresh_plot(allow_splinters=False):

    epsilon = 1.0e-27

    run_length = np.array([839, 1179, 1339, 879]) / 2.0 - 290.0

    if allow_splinters:
        jet_count_list = [238, 466, 782, 250]
    else:
        jet_count_list = [197, 381, 733, 240]

    color_arr = ["black", jx.medium_blue, "green", jx.orange]

    runid_list = ["ABA", "ABC", "AEA", "AEC"]
    runname_list = ["HM30", "HM05", "LM30", "LM05"]
    # mt_str_list = ["1.1","1.2","1.3","1.4","1.5","1.6","1.7","1.8","1.9","2.0","2.1","2.2","2.5","2.8","3.0"]
    mt_str_list = ["1.1", "1.3", "1.5", "1.7", "1.9", "2.1", "2.3", "2.5", "2.7", "3.0"]

    share_arr = np.zeros((len(mt_str_list), len(runid_list)), dtype=float)
    slams_share_arr = np.zeros((len(mt_str_list), len(runid_list)), dtype=float)
    slams_number_arr = np.zeros((len(mt_str_list), len(runid_list)), dtype=float)
    jet_number_arr = np.zeros((len(mt_str_list), len(runid_list)), dtype=float)
    sj_number_arr = np.zeros((len(mt_str_list), len(runid_list)), dtype=float)
    for n in range(len(mt_str_list)):
        # print(mt_str_list[n])
        data = np.loadtxt(
            wrkdir_DNR
            + "sjn_counts/sjn_count_{}_{}.txt".format(mt_str_list[n], allow_splinters)
        ).astype(float)
        share = (data[0]) / (data[1])
        slams_share = (data[0]) / (data[2])
        slams_number = data[2]
        jet_number = data[1]
        sj_number = data[0]
        share_arr[n] = share
        slams_share_arr[n] = slams_share
        slams_number_arr[n] = slams_number
        jet_number_arr[n] = jet_number
        sj_number_arr[n] = sj_number

    share_arr = share_arr.T
    slams_share_arr = slams_share_arr.T
    slams_number_arr = slams_number_arr.T
    jet_number_arr = jet_number_arr.T
    sj_number_arr = sj_number_arr.T
    mt_arr = np.array(list(map(float, mt_str_list)))

    ann_locs = [(0.03, 0.1), (0.03, 0.1), (0.03, 0.1), (0.03, 0.8), (0.03, 0.1)]
    ann_labs = ["a)", "b)", "c)", "d)", "e)"]

    fig, ax_list = plt.subplots(5, 1, figsize=(8, 10))
    for m in range(len(runid_list)):
        # ax_list[0].semilogy(mt_arr,slams_number_arr[m],label=runid_list[m])
        ax_list[2].plot(
            mt_arr,
            sj_number_arr[m] / run_length[m],
            label=runname_list[m],
            color=color_arr[m],
            linewidth=1.2,
        )
        ax_list[1].plot(
            mt_arr,
            jet_number_arr[m] / run_length[m],
            label=runname_list[m],
            color=color_arr[m],
            linewidth=1.2,
        )
        ax_list[0].plot(
            mt_arr,
            slams_number_arr[m] / run_length[m],
            label=runname_list[m],
            color=color_arr[m],
            linewidth=1.2,
        )
        ax_list[3].plot(
            mt_arr,
            slams_share_arr[m],
            label=runname_list[m],
            color=color_arr[m],
            linewidth=1.2,
        )
        ax_list[4].plot(
            mt_arr,
            share_arr[m],
            label=runname_list[m],
            color=color_arr[m],
            linewidth=1.2,
        )
    for m in range(len(runid_list)):
        # ax_list[0].axhline(jet_count_list[m],linestyle="dashed",color=color_arr[m],linewidth=0.8)
        pass

    ax_list[-1].set_xlabel(
        "FCS magnetic threshold $\mathrm{\\eta=|B|/B_{IMF}}$", fontsize=20, labelpad=10
    )
    ax_list[2].set_ylabel("FCS-jets/s", fontsize=15, labelpad=10)
    ax_list[1].set_ylabel("Jets/s", fontsize=15, labelpad=10)
    ax_list[0].set_ylabel("FCSs/s", fontsize=15, labelpad=10)
    ax_list[3].set_ylabel("FCS-jets\nper FCS", fontsize=15, labelpad=10)
    ax_list[4].set_ylabel("Fraction of jets\ncaused by FCS", fontsize=15, labelpad=10)
    # ax_list[0].set_title("Allow splinters = {}".format(allow_splinters),fontsize=20)
    ax_list[1].legend(frameon=False, numpoints=1, markerscale=3, loc="lower right")
    for ix, ax in enumerate(ax_list):
        ax.annotate(ann_labs[ix], ann_locs[ix], xycoords="axes fraction", fontsize=20)
        ax.grid()
        ax.set_xlim(mt_arr[0], mt_arr[-1])
        ax.tick_params(labelsize=15)
        for lb in ["bottom", "top", "left", "right"]:
            ax.spines[lb].set_linewidth(1.5)
    ax_list[2].set_ylim(bottom=0, top=0.99)
    ax_list[1].set_ylim(bottom=0, top=1.25)
    ax_list[0].set_ylim(bottom=0)
    ax_list[3].set_ylim(bottom=0, top=1.99)
    ax_list[4].set_ylim(0, 0.99)
    for axe in fig.get_axes():
        axe.label_outer()
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1)

    # fig.savefig(wrkdir_DNR+"sjratio_fig_{}.png".format(allow_splinters))
    fig.savefig(wrkdir_DNR + "Figures/sj_figs/fig2.png")
    plt.close(fig)
    return None


def sj_non_counter(allow_splinters=True, mag_thresh=1.5):

    epsilon = 1.0e-27

    runids = ["ABA", "ABC", "AEA", "AEC"]

    count_arr = np.empty((3, 4), dtype=int)

    for ix, runid in enumerate(runids):
        counts = np.array(
            [arr.size for arr in separate_jets_god(runid, allow_splinters)], dtype=int
        )
        count_arr[:, ix] = counts

    # data_arr = np.array([separate_jets_god(runid,allow_splinters) for runid in runids]).flatten()
    # count_arr = np.array([arr.size for arr in data_arr])
    # count_arr = np.reshape(count_arr,(4,3)).T

    print("Runs:           ABA ABC AEA AEC\n")
    print("SJ Jets:        {}\n".format(count_arr[0]))
    print("Jets:           {}\n".format(count_arr[1]))
    print("SLAMS:          {}\n".format(count_arr[2]))
    print(
        "SJ/jet ratio:   {}\n".format(
            (count_arr[0].astype(float) + epsilon) / (count_arr[1] + epsilon)
        )
    )
    print(
        "SJ/SLAMS ratio: {}\n".format(
            (count_arr[0].astype(float) + epsilon) / (count_arr[2] + epsilon)
        )
    )

    np.savetxt(
        wrkdir_DNR
        + "sjn_counts/sjn_count_{}_{}.txt".format(mag_thresh, allow_splinters),
        count_arr,
        fmt="%.0f",
    )

    # return np.reshape(data_arr,(4,3))
    return None


def separate_jets_god(runid, allow_relatives=True):

    runids = ["ABA", "ABC", "AEA", "AEC"]

    sj_ids = []
    jet_ids = []
    slams_ids = []

    for n1 in range(6000):

        try:
            props = jio.PropReader(str(n1).zfill(5), runid, transient="jet")
        except:
            continue

        if props.read("at_bow_shock")[0] != 1:
            continue

        if not allow_relatives:
            if "splinter" in props.meta:
                non_bs_time = (props.read("time")[props.read("at_bow_shock") == 1])[
                    -1
                ] + 0.5
                splinter_time = props.read("time")[props.read("is_splinter") == 1][
                    0
                ]  # time of first splintering
                splin_times = np.array([splinter_time] + props.get_splin_times())
                if (splin_times >= non_bs_time).any():
                    continue

        if (props.read("at_slams") == 1).any():
            jet_ids.append(n1)
            sj_ids.append(n1)
        else:
            jet_ids.append(n1)

    for n2 in range(6000):

        try:
            props = jio.PropReader(str(n2).zfill(5), runid, transient="slams")
        except:
            continue

        if not allow_relatives:
            if "merger" in props.meta:
                if (props.read("at_bow_shock") == 1).any():
                    bs_time = (props.read("time")[props.read("at_bow_shock") == 1])[0]
                else:
                    bs_time = (props.read("time")[props.read("at_bow_shock") == 0])[
                        -1
                    ] + 0.5
                merge_time = props.read("time")[props.read("is_merger") == 1][0]
                if merge_time < bs_time:
                    continue

        slams_ids.append(n2)

    return [np.unique(sj_ids), np.unique(jet_ids), np.unique(slams_ids)]


def pendep_hist(runids=["ABA", "ABC", "AEA", "AEC"], panel_one=True):

    mpl.style.use("default")

    runid_dict = ["ABA", "ABC", "AEA", "AEC"]
    run_length = np.array([839, 1179, 1339, 879]) / 2.0 - 290.0

    runids_list = runids
    sj_pendeps = np.full([2000], np.nan)
    non_pendeps = np.full([2000], np.nan)
    sj_weights = np.full([2000], np.nan)
    non_weights = np.full([2000], np.nan)
    sj_time_weights = np.full([2000], np.nan)
    non_time_weights = np.full([2000], np.nan)
    sj_counter = 0
    non_counter = 0

    opstring = r"\_".join(runids)

    for runid in runids_list:
        sj_jet_ids, jet_ids, slams_ids = separate_jets_god(runid, False)
        non_sj_ids = jet_ids[np.in1d(jet_ids, sj_jet_ids) == False]
        sj_amount = sj_jet_ids.size
        non_amount = non_sj_ids.size
        jet_amount = jet_ids.size
        tracking_duration = run_length[runid_dict.index(runid)]
        for sj_id in sj_jet_ids:
            props = jio.PropReader(str(sj_id).zfill(5), runid, transient="jet")
            x_mean = props.read("x_mean")
            bs_dist = props.read("bs_distance")
            pendep = (x_mean - bs_dist)[-1]
            # sj_pendeps.append(pendep)
            sj_pendeps[sj_counter] = pendep
            # sj_weights.append(1.0/run_length[runid_dict.index(runid)])
            sj_weights[sj_counter] = 1.0 / (4.0 * sj_amount)
            sj_time_weights[sj_counter] = 1.0 / (4.0 * tracking_duration)
            sj_counter += 1

        for non_id in non_sj_ids:
            props = jio.PropReader(str(non_id).zfill(5), runid, transient="jet")
            x_mean = props.read("x_mean")
            bs_dist = props.read("bs_distance")
            pendep = (x_mean - bs_dist)[-1]
            # non_pendeps.append(pendep)
            non_pendeps[non_counter] = pendep
            # non_weights.append(1.0/run_length[runid_dict.index(runid)])
            non_weights[non_counter] = 1.0 / (4.0 * non_amount)
            non_time_weights[non_counter] = 1.0 / (4.0 * tracking_duration)
            non_counter += 1

    # sj_pendeps = np.array(sj_pendeps,ndmin=1)
    # non_pendeps = np.array(non_pendeps,ndmin=1)
    sj_pendeps = sj_pendeps[:sj_counter]
    sj_weights = sj_weights[:sj_counter]
    sj_time_weights = sj_time_weights[:sj_counter]

    non_pendeps = non_pendeps[:non_counter]
    non_weights = non_weights[:non_counter]
    non_time_weights = non_time_weights[:non_counter]

    bins = np.linspace(-2.5, 0, 25 + 1)
    # sj_weights = np.ones(sj_pendeps.shape,dtype=float)/sj_pendeps.size
    # non_weights = np.ones(non_pendeps.shape,dtype=float)/non_pendeps.size
    # sj_weights = np.array(sj_weights)/len(runids)
    # non_weights = np.array(non_weights)/len(runids)

    if panel_one:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        # plt.grid()
        ax[0].hist(
            sj_pendeps,
            bins=bins,
            weights=sj_time_weights,
            histtype="step",
            color="red",
            label="FCS-originated",
        )
        ax[0].hist(
            non_pendeps,
            bins=bins,
            weights=non_time_weights,
            histtype="step",
            color="black",
            label="Non-FCS-originated",
        )
        ax[1].set_xlabel("$\mathrm{X_{last}-X_{BS}}~[R_e]$", fontsize=25, labelpad=10)
        sj_hist, sj_bins, sj_patch = ax[1].hist(
            sj_pendeps,
            bins=bins,
            weights=sj_weights,
            histtype="step",
            color="red",
            label="FCS-originated",
            cumulative=True,
        )
        non_hist, non_bins, non_patch = ax[1].hist(
            non_pendeps,
            bins=bins,
            weights=non_weights,
            histtype="step",
            color="black",
            label="Non-FCS-originated",
            cumulative=True,
        )
        # ax[0].legend(frameon=False,numpoints=1,markerscale=2,fontsize=15,loc="upper left")
        # ax[1].legend(frameon=False,numpoints=1,markerscale=2,fontsize=15,loc="upper left")
        ax[0].set_ylabel("Jets/s", fontsize=25, labelpad=10)
        ax[1].set_ylabel("Cumulative fraction of jets", fontsize=25, labelpad=10)
        ax[0].set_xlabel("$\mathrm{X_{last}-X_{BS}}~[R_e]$", fontsize=25, labelpad=10)
        ax[0].tick_params(labelsize=15)
        ax[1].tick_params(labelsize=15)
        ax[1].set_ylim(bottom=0)
        ax[0].set_xlim(right=0)
        ax[1].set_xlim(right=0)
        # for axe in fig.get_axes():
        #    axe.label_outer()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # plt.grid()
        ax.set_xlabel("$\mathrm{X_{last}-X_{BS}}~[R_e]$", fontsize=20, labelpad=10)
        sj_hist, sj_bins, sj_patch = ax.hist(
            sj_pendeps,
            bins=bins,
            weights=sj_weights,
            histtype="step",
            color="red",
            label="FCS-originated",
            cumulative=True,
        )
        non_hist, non_bins, non_patch = ax.hist(
            non_pendeps,
            bins=bins,
            weights=non_weights,
            histtype="step",
            color="black",
            label="Non-FCS-originated",
            cumulative=True,
        )
        # ax.legend(frameon=False,numpoints=1,markerscale=2,fontsize=15,loc="upper left")
        ax.set_ylabel("Cumulative fraction of jets", fontsize=20, labelpad=10)
        ax.tick_params(labelsize=15)
        ax.set_ylim(bottom=0)
        ax.set_xlim(right=0)
    # ax.set_title(opstring,fontsize=20,pad=10)
    binwidth = np.ediff1d(sj_bins)[0]
    sj_bins = sj_bins[:-1] + binwidth / 2.0

    global maxbin, jet_max

    maxbin = sj_bins[-1]
    jet_max = sj_hist[-1]

    sj_popt, sj_pcov = scipy.optimize.curve_fit(
        expfit_pendep, sj_bins, sj_hist, p0=[1.0, 0.0]
    )

    jet_max = non_hist[-1]

    non_popt, non_pcov = scipy.optimize.curve_fit(
        expfit_pendep, sj_bins, non_hist, p0=[1.0, 0.0]
    )

    xinterp = np.linspace(-2.5, 0, 100 + 1)

    if panel_one:
        jet_max = sj_hist[-1]
        ax[1].plot(
            xinterp,
            expfit_pendep(xinterp, sj_popt[0], sj_popt[1]),
            color="red",
            linestyle="dashed",
            label="EF: {:.2f}Re".format(-1.0 / sj_popt[0]),
        )
        jet_max = non_hist[-1]
        ax[1].plot(
            xinterp,
            expfit_pendep(xinterp, non_popt[0], non_popt[1]),
            color="black",
            linestyle="dashed",
            label="EF: {:.2f}Re".format(-1.0 / non_popt[0]),
        )
        ax[0].annotate("a)", (0.05, 0.1), xycoords="axes fraction", fontsize=20)
        ax[1].annotate("b)", (0.05, 0.1), xycoords="axes fraction", fontsize=20)
        ax[0].legend(
            frameon=False, numpoints=1, markerscale=2, fontsize=15, loc="upper left"
        )
        ax[1].legend(
            frameon=False, numpoints=1, markerscale=2, fontsize=15, loc="upper left"
        )
    else:
        jet_max = sj_hist[-1]
        ax.plot(
            xinterp,
            expfit_pendep(xinterp, sj_popt[0], sj_popt[1]),
            color="red",
            linestyle="dashed",
            label="EFL: {:.2f}Re".format(-1.0 / sj_popt[0]),
        )
        jet_max = non_hist[-1]
        ax.plot(
            xinterp,
            expfit_pendep(xinterp, non_popt[0], non_popt[1]),
            color="black",
            linestyle="dashed",
            label="EFL: {:.2f}Re".format(-1.0 / non_popt[0]),
        )
        ax.legend(
            frameon=False, numpoints=1, markerscale=2, fontsize=15, loc="upper left"
        )

    plt.tight_layout()

    # fig.savefig(wrkdir_DNR+"pendep_{}.png".format("_".join(runids)))
    fig.savefig(wrkdir_DNR + "Figures/sj_figs/fig3.png")
    plt.close(fig)


def expfit_pendep(xdata, a1, a2):

    return jet_max * np.exp(-a1 * (xdata - maxbin)) + a2


def jh2020_fig1(var="pdyn"):

    vars_list = ["pdyn", "core_heating", "rho", "Mms", "B"]
    var_index = vars_list.index(var)
    label_list = ["nPa", "$T_{sw}$", "$cm^{-3}$", "", "nT"]
    vmax_list = [4.5, 3.0, 6.6, 1, 10]
    expr_list = [
        pc.expr_pdyn,
        pc.expr_coreheating,
        pc.expr_srho,
        pc.expr_mms,
        pc.expr_B,
    ]
    sj_jet_ids, non_sj_ids = separate_jets("ABC")

    global filenr_g
    global runid_g
    global sj_jetobs
    global non_sjobs
    global draw_arrows

    draw_arrows = True

    runid_g = "ABC"

    sj_jetobs = [
        jio.PropReader(str(n).zfill(5), "ABC", transient="slamsjet") for n in sj_jet_ids
    ]
    non_sjobs = [
        jio.PropReader(str(n).zfill(5), "ABC", transient="slamsjet") for n in non_sj_ids
    ]

    outputdir = wrkdir_DNR + "Figures/jh2020/"

    # filepath = "/scratch/project_2000203/sunijona/vlasiator/2D/ABC/bulk/bulk.0000677.vlsv"
    # filepath = "/scratch/project_2000203/2D/ABC/bulk/bulk.0000714.vlsv"
    # filepath = vlasdir+"/2D/ABC/bulk/bulk.0000714.vlsv"
    filepath = vlasdir + "/2D/ABC/bulk/bulk.0000825.vlsv"

    # filenr_g = 677
    # filenr_g = 714
    filenr_g = 825

    colmap = "parula"
    if var == "Mms":
        colmap = "parula"

    # pt.plot.plot_colormap(filename=filepath,outputfile=outputdir+"fig1a_{}.png".format(var),usesci=0,lin=1,expression=expr_list[var_index],vmin=0,vmax=vmax_list[var_index],colormap=colmap,cbtitle=label_list[var_index],pass_vars=["rho","v","CellID","Pdyn","RhoNonBackstream","PTensorNonBackstreamDiagonal","Mms","B"],Earth=1)

    pt.plot.plot_colormap(
        filename=filepath,
        outputfile=outputdir + "fig1b_{}.png".format(var),
        boxre=[6, 18, -6, 6],
        usesci=0,
        lin=1,
        expression=expr_list[var_index],
        vmin=0,
        vmax=vmax_list[var_index],
        colormap=colmap,
        cbtitle=label_list[var_index],
        external=jh20f1_ext,
        pass_vars=[
            "rho",
            "v",
            "CellID",
            "Pdyn",
            "RhoNonBackstream",
            "PTensorNonBackstreamDiagonal",
            "Mmsx",
            "B",
            "core_heating",
        ],
    )


def jh2020_movie(
    runid,
    start,
    stop,
    var="Pdyn",
    arr_draw=False,
    dbg=False,
    fig5=False,
    fig1=False,
    magt=1.5,
    fig1mov=False,
):

    if fig1:
        fig5 = False

    runid_list = ["ABA", "ABC", "AEA", "AEC"]
    run_index = runid_list.index(runid)
    maxfnr_list = [839, 1179, 1339, 879]
    if start > maxfnr_list[runid_list.index(runid)]:
        return 0

    vars_list = ["Pmag", "Ptot", "Pressure", "Pdyn", "rho", "B", "v", "Temperature"]
    var_index = vars_list.index(var)
    # label_list = ["nPa","nPa","$T_{sw}$","$cm^{-3}$","","nT"]
    vmax_list = [
        [0.25, 4, 2, 1.5, 6, 20, 850, 15],
        [0.25, 4, 2, 3.0, 20, 20, 700, 15],
        [0.25, 4, 2, 1.5, 6, 40, 850, 15],
        [0.25, 4, 2, 3.0, 20, 40, 700, 15],
    ][run_index]
    vmin_list = [
        [0.0, 1.0, 0.2, 0, 1.0, 5, 250, 0.5],
        [0.0, 1.0, 0.2, 0, 3.3, 5, 100, 0.5],
        [0.0, 1.0, 0.2, 0, 1.0, 10, 250, 0.5],
        [0.0, 1.0, 0.2, 0, 3.3, 10, 100, 0.5],
    ][run_index]
    vscale_list = [1e9, 1e9, 1e9, 1e9, 1.0e-6, 1e9, 1e-3, 1e-6]
    # expr_list = [pc.expr_pdyn,pc.expr_coreheating,pc.expr_srho,pc.expr_mms,pc.expr_B]
    sj_jet_ids, jet_ids, slams_ids = separate_jets_god(runid, False)
    non_sj_ids = jet_ids[np.in1d(jet_ids, sj_jet_ids) == False]

    global filenr_g
    global runid_g
    global sj_jetobs
    global non_sjobs
    global draw_arrows
    global fig5_g
    global fig1_g

    fig5_g = fig5
    fig1_g = fig1

    draw_arrows = arr_draw

    runid_g = runid

    sj_jetobs = [
        jio.PropReader(str(n).zfill(5), runid, transient="jet") for n in sj_jet_ids
    ]
    non_sjobs = [
        jio.PropReader(str(n).zfill(5), runid, transient="jet") for n in non_sj_ids
    ]

    outputdir = wrkdir_DNR + "jh2020_movie/{}/{}/{}/".format(runid, var, magt)
    fluxfile = None
    fluxdir = None
    if dbg:
        outputdir = wrkdir_DNR + "jh2020_debug/{}/{}/{}/".format(runid, var, magt)
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    bulkpath = jx.find_bulkpath(runid)

    vmax = vmax_list[var_index]
    vmin = vmin_list[var_index]
    vscale = vscale_list[var_index]
    boxre = [6, 18, -6, 6]
    if runid in ["ABA", "AEA"]:
        boxre = [6, 18, -8, 6]

    colmap = "Blues"

    if fig1mov:
        fig1_g = True
        for itr in range(start, stop + 1):
            filepath = bulkpath + "bulk.{}.vlsv".format(str(itr).zfill(7))
            filenr_g = itr
            pt.plot.plot_colormap(
                filename=filepath,
                outputfile=wrkdir_DNR
                + "Figures/thesis/mov/{}.png".format(str(itr).zfill(7)),
                boxre=boxre,
                usesci=0,
                lin=1,
                var=var,
                tickinterval=2,
                vmin=vmin,
                vmax=vmax,
                vscale=vscale,
                colormap=colmap,
                external=jh20f1_ext,
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
            )

        return None

    if fig1:
        filepath = bulkpath + "bulk.0000895.vlsv"
        filenr_g = 895

        pt.plot.plot_colormap(
            filename=filepath,
            outputfile=wrkdir_DNR + "Figures/sj_figs/fig1.png",
            boxre=boxre,
            usesci=0,
            lin=1,
            var=var,
            tickinterval=2,
            vmin=vmin,
            vmax=vmax,
            vscale=vscale,
            colormap=colmap,
            external=jh20f1_ext,
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
            fluxfile=fluxfile,
            fluxdir=fluxdir,
            fluxlines=40,
        )

        return None

    if fig5:
        noborder = True
        scale = 2.0

        fig = plt.figure(figsize=(8, 8))
        gs = fig.add_gridspec(2, 15)
        ax_ul = fig.add_subplot(gs[0, 0:7])
        ax_ur = fig.add_subplot(gs[0, 7:-1])
        ax_ll = fig.add_subplot(gs[1, 0:7])
        ax_lr = fig.add_subplot(gs[1, 7:-1])
        cbax = fig.add_subplot(gs[:, -1])

        filepath = bulkpath + "bulk.0000954.vlsv"
        filenr_g = 954

        pt.plot.plot_colormap(
            filename=filepath,
            outputfile=wrkdir_DNR + "Figures/sj_figs/fig4a.png",
            boxre=[9, 13, -3, 1],
            usesci=0,
            lin=1,
            vscale=vscale,
            var=var,
            tickinterval=1,
            vmin=vmin,
            vmax=vmax,
            colormap=colmap,
            external=jh20f1_ext,
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
            fluxfile=fluxfile,
            fluxdir=fluxdir,
            fluxlines=80,
            axes=ax_ul,
            nocb=True,
            noxlabels=True,
            noborder=noborder,
            scale=scale,
        )
        ax_ul.annotate("a)", xy=(0.05, 0.9), xycoords="axes fraction", fontsize=20)

        filepath = bulkpath + "bulk.0000962.vlsv"
        filenr_g = 962

        pt.plot.plot_colormap(
            filename=filepath,
            outputfile=wrkdir_DNR + "Figures/sj_figs/fig4b.png",
            boxre=[9, 13, -3, 1],
            usesci=0,
            lin=1,
            vscale=vscale,
            var=var,
            tickinterval=1,
            vmin=vmin,
            vmax=vmax,
            colormap=colmap,
            external=jh20f1_ext,
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
            fluxfile=fluxfile,
            fluxdir=fluxdir,
            fluxlines=80,
            axes=ax_ur,
            nocb=True,
            noxlabels=True,
            noylabels=True,
            noborder=noborder,
            scale=scale,
        )
        ax_ur.annotate("b)", xy=(0.05, 0.9), xycoords="axes fraction", fontsize=20)

        filepath = bulkpath + "bulk.0000970.vlsv"
        filenr_g = 970

        pt.plot.plot_colormap(
            filename=filepath,
            outputfile=wrkdir_DNR + "Figures/sj_figs/fig4c.png",
            boxre=[9, 13, -3, 1],
            usesci=0,
            lin=1,
            vscale=vscale,
            var=var,
            tickinterval=1,
            vmin=vmin,
            vmax=vmax,
            colormap=colmap,
            external=jh20f1_ext,
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
            fluxfile=fluxfile,
            fluxdir=fluxdir,
            fluxlines=80,
            axes=ax_ll,
            nocb=True,
            noborder=noborder,
            scale=scale,
        )
        ax_ll.annotate("c)", xy=(0.05, 0.9), xycoords="axes fraction", fontsize=20)

        filepath = bulkpath + "bulk.0000996.vlsv"
        filenr_g = 996

        pt.plot.plot_colormap(
            filename=filepath,
            outputfile=wrkdir_DNR + "Figures/sj_figs/fig4d.png",
            boxre=[9, 13, -3, 1],
            usesci=0,
            lin=1,
            vscale=vscale,
            var=var,
            tickinterval=1,
            vmin=vmin,
            vmax=vmax,
            colormap=colmap,
            external=jh20f1_ext,
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
            fluxfile=fluxfile,
            fluxdir=fluxdir,
            fluxlines=80,
            axes=ax_lr,
            noylabels=True,
            cbaxes=cbax,
            noborder=noborder,
            scale=scale,
            cbtitle="",
        )
        ax_lr.annotate("d)", xy=(0.05, 0.9), xycoords="axes fraction", fontsize=20)

        # fig.subplots_adjust(wspace=0.05)
        plt.tight_layout()
        fig.savefig(wrkdir_DNR + "Figures/sj_figs/fig4.png")
        plt.close(fig)
        return None

    for itr in range(start, stop + 1):
        filepath = bulkpath + "bulk.{}.vlsv".format(str(itr).zfill(7))
        filenr_g = itr
        if dbg:
            fluxdir = bulkpath + "../flux/"
            fluxfile = "flux.{}.bin".format(str(itr).zfill(7))

        if fig5:
            pt.plot.plot_colormap(
                filename=filepath,
                outputfile=outputdir + "fig5/{}.png".format(str(itr).zfill(5)),
                boxre=[9, 12, -3, 1],
                usesci=0,
                lin=1,
                vscale=vscale,
                var=var,
                tickinterval=1,
                vmin=vmin,
                vmax=vmax,
                colormap=colmap,
                external=jh20f1_ext,
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
                fluxfile=fluxfile,
                fluxdir=fluxdir,
                fluxlines=80,
            )

        else:

            pt.plot.plot_colormap(
                filename=filepath,
                outputfile=outputdir + "{}.png".format(str(itr).zfill(5)),
                boxre=boxre,
                usesci=0,
                lin=1,
                var=var,
                tickinterval=2,
                vmin=vmin,
                vmax=vmax,
                vscale=vscale,
                colormap=colmap,
                external=jh20f1_ext,
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
                fluxfile=fluxfile,
                fluxdir=fluxdir,
                fluxlines=40,
            )

            pt.plot.plot_colormap(
                filename=filepath,
                outputfile=outputdir + "zoom/{}.png".format(str(itr).zfill(5)),
                boxre=[8, 14, -2, 2],
                usesci=0,
                lin=1,
                vscale=vscale,
                var=var,
                tickinterval=1,
                vmin=vmin,
                vmax=vmax,
                colormap=colmap,
                external=jh20f1_ext,
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
                fluxfile=fluxfile,
                fluxdir=fluxdir,
                fluxlines=80,
            )


def jh20f1_ext(ax, XmeshXY, YmeshXY, pass_maps):

    cellids = pass_maps["CellID"]
    rho = pass_maps["rho"]
    mmsx = pass_maps["Mmsx"]
    core_heating = pass_maps["core_heating"]
    if runid_g in ["ABA", "AEA"]:
        rho_sw = 1.0e6
    else:
        rho_sw = 3.3e6

    # slams_cells = jio.eventfile_read("ABC",filenr_g,transient="slams")
    # slams_cells = np.array([item for sublist in slams_cells for item in sublist])
    # jet_cells = jio.eventfile_read("ABC",filenr_g,transient="jet")
    # jet_cells = np.array([item for sublist in jet_cells for item in sublist])
    slams_cells = np.loadtxt(
        "/wrk/users/jesuni/working/SLAMS/Masks/{}/{}.mask".format(runid_g, filenr_g)
    ).astype(int)
    jet_cells = np.loadtxt(
        "/wrk/users/jesuni/working/jets/Masks/{}/{}.mask".format(runid_g, filenr_g)
    ).astype(int)

    slams_mask = np.in1d(cellids, slams_cells).astype(int)
    slams_mask = np.reshape(slams_mask, cellids.shape)

    jet_mask = np.in1d(cellids, jet_cells).astype(int)
    jet_mask = np.reshape(jet_mask, cellids.shape)

    ch_mask = (core_heating > 3 * 0.5e6).astype(int)
    mach_mask = (mmsx < 1).astype(int)
    rho_mask = (rho > 2 * rho_sw).astype(int)

    # x_list = []
    # y_list = []

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
        if filenr_g / 2.0 in jetobj.read("time"):
            sj_xlist.append(jetobj.read_at_time("x_mean", filenr_g / 2.0))
            sj_ylist.append(jetobj.read_at_time("y_mean", filenr_g / 2.0))
    for jetobj in non_sjobs:
        if filenr_g / 2.0 in jetobj.read("time"):
            non_xlist.append(jetobj.read_at_time("x_mean", filenr_g / 2.0))
            non_ylist.append(jetobj.read_at_time("y_mean", filenr_g / 2.0))
        if fig5_g:
            if 962 / 2.0 in jetobj.read("time"):
                if np.abs(jetobj.read_at_time("x_mean", 962 / 2.0) - 11) < 1.0:
                    xpos = jetobj.read_at_time("x_mean", 962 / 2.0)
                    ypos = jetobj.read_at_time("y_mean", 962 / 2.0)

    # bs_fit = jx.bow_shock_jonas(runid_g,filenr_g)[::-1]
    # mp_fit = jx.mag_pause_jonas(runid_g,filenr_g)[::-1]
    # y_bs = np.arange(-8,6.01,0.05)
    # x_bs = np.polyval(bs_fit,y_bs)
    # x_mp = np.polyval(mp_fit,y_bs)

    # bs_cont, = ax.plot(x_bs,y_bs,color="black",linewidth=0.8)
    # mp_cont, = ax.plot(x_mp,y_bs,color="black",linewidth=0.8)

    markscaler = 1.0
    if fig5_g:
        markscaler = 2.0

    if fig5_g:
        rho_cont = ax.contour(
            XmeshXY,
            YmeshXY,
            rho_mask,
            [0.5],
            linewidths=markscaler * 0.6,
            colors="black",
        )
        mach_cont = ax.contour(
            XmeshXY,
            YmeshXY,
            mach_mask,
            [0.5],
            linewidths=markscaler * 0.6,
            colors=jx.violet,
        )
        rho_cont.collections[0].set_label("$n \geq 2n_{sw}$")
        mach_cont.collections[0].set_label("$M_{MS,x} \leq 1$")
        ax.annotate(
            "",
            xy=(xpos - 0.125, ypos + 0.125),
            xytext=(xpos - 0.75, ypos + 0.75),
            xycoords="data",
            textcoords="data",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        )
        # if filenr_g in [962,970]:
        #     for ix,xpos in enumerate(non_xlist):
        #         if np.abs(xpos-11) < 1.0:
        #             ypos = non_ylist[ix]
        #             ax.annotate("",xy=(xpos-0.075,ypos+0.075),xytext=(xpos-0.75,ypos+0.75),xycoords="data",textcoords="data",arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))

    ch_cont = ax.contour(
        XmeshXY, YmeshXY, ch_mask, [0.5], linewidths=markscaler * 0.6, colors=jx.orange
    )
    ch_cont.collections[0].set_label("$T_{core} \geq 3T_{sw}$")

    slams_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        slams_mask,
        [0.5],
        linewidths=markscaler * 0.6,
        colors="yellow",
    )
    jet_cont = ax.contour(
        XmeshXY, YmeshXY, jet_mask, [0.5], linewidths=markscaler * 0.6, colors=jx.green
    )
    slams_cont.collections[0].set_label("FCS")
    jet_cont.collections[0].set_label("Jet")

    (non_pos,) = ax.plot(
        non_xlist,
        non_ylist,
        "o",
        color="black",
        markersize=markscaler * 4,
        markeredgecolor="white",
        fillstyle="full",
        mew=markscaler * 0.4,
        label="Non-FCS-jet",
    )
    (sj_pos,) = ax.plot(
        sj_xlist,
        sj_ylist,
        "o",
        color="red",
        markersize=markscaler * 4,
        markeredgecolor="white",
        fillstyle="full",
        mew=markscaler * 0.4,
        label="FCS-jet",
    )

    if draw_arrows:
        arrow_coords = jx.bs_norm(runid_g, filenr_g)
        for n in range(1, len(arrow_coords)):
            nx, ny, dnx, dny = arrow_coords[n]
            if ny // 0.5 > arrow_coords[n - 1][1] // 0.5:
                ax.arrow(nx, ny, dnx, dny, head_width=0.1, width=0.01, color=jx.orange)

    if fig1_g:
        ax.legend(
            frameon=True, numpoints=1, markerscale=1, loc="upper right", fontsize=5
        )

    # xy_pos, = ax.plot(x_list,y_list,"o",color=jx.crimson,markersize=2)

    # is_coords = jx.get_cell_coordinates(runid_g,1814480)/r_e
    # os_coords = jx.get_cell_coordinates(runid_g,1814540)/r_e

    # is2 = jx.get_cell_coordinates("ABC",1814480+2000*30+10)/r_e
    # os2 = jx.get_cell_coordinates("ABC",1814540+2000*30+10)/r_e

    # is_pos, = ax.plot(is_coords[0],is_coords[1],">",color="black",markersize=2)
    # os_pos, = ax.plot(os_coords[0],os_coords[1],"<",color="black",markersize=2)

    # cut_through_plot, = ax.plot([is_coords[0],os_coords[0]],[is_coords[1],os_coords[1]],color="black",linewidth=0.8)
    # cut_through_plot2, = ax.plot([is2[0],os2[0]],[is2[1],os2[1]],color="black",linewidth=0.8)

