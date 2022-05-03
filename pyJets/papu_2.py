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


def jet_pos_plot():
    runids = ["AEC", "AEA", "ABC", "ABA"]
    CB_color_cycle = jx.CB_color_cycle
    kinds = ["foreshock", "beam", "complex", "stripe"]
    draw_labels = [False, True, False, False]

    fig, ax_list = plt.subplots(2, 2, figsize=(10, 10))
    ax_flat = ax_list.flatten()

    yarr = np.arange(-15, 15, 0.1)
    bs_fit = [jx.bs_mp_fit(runid, 800, boxre=[6, 18, -15, 15])[1] for runid in runids]
    bs_x = [
        np.polyval(bs_fit[idx], yarr) - bs_fit[idx][-1] for idx in range(len(runids))
    ]

    vlsvobj_arr = [jx.read_bulkfile(runid, 800) for runid in runids]
    cellid_arr = [
        vlsvobj_arr[idx].read_variable("CellID") for idx in range(len(runids))
    ]
    Yun_arr = [
        np.unique(jx.xyz_reconstruct(vlsvobj_arr[idx])[1][np.argsort(cellid_arr[idx])])
        / r_e
        for idx in range(len(runids))
    ]
    Xun_arr = [
        np.unique(jx.xyz_reconstruct(vlsvobj_arr[idx])[0][np.argsort(cellid_arr[idx])])
        / r_e
        for idx in range(len(runids))
    ]
    Bz_arr = [
        vlsvobj_arr[idx].read_variable("B", operator="z")[np.argsort(cellid_arr[idx])]
        for idx in range(len(runids))
    ]
    RhoBS_arr = [
        vlsvobj_arr[idx].read_variable("RhoBackstream")[np.argsort(cellid_arr[idx])]
        for idx in range(len(runids))
    ]

    Bz_arr = [
        np.reshape(Bz_arr[idx], (Yun_arr[idx].size, Xun_arr[idx].size))
        for idx in range(len(runids))
    ]
    RhoBS_arr = [
        np.reshape(RhoBS_arr[idx], (Yun_arr[idx].size, Xun_arr[idx].size))
        for idx in range(len(runids))
    ]

    for idx, ax in enumerate(ax_flat):

        ax.contour(
            Xun_arr[idx] - bs_fit[idx][-1],
            Yun_arr[idx],
            Bz_arr[idx],
            [-0.5e-9, 0.5e-9],
            colors=[CB_color_cycle[4], CB_color_cycle[5]],
            linewidths=[0.6],
        )
        ax.contour(
            Xun_arr[idx] - bs_fit[idx][-1],
            Yun_arr[idx],
            np.abs(RhoBS_arr[idx]),
            [1],
            colors=["black"],
            linewidths=[0.6],
            linestyles=["dashed"],
        )
        ax.plot(bs_x[idx], yarr, color="black")

    for n1, runid in enumerate(runids):
        ax = ax_flat[n1]
        for n2, kind in enumerate(kinds):
            label_bool = draw_labels[n1]
            non_ids = np.loadtxt(
                wrkdir_DNR + "papu22/id_txts/new/{}_{}.txt".format(runid, kind),
                dtype=int,
                ndmin=1,
            )
            for non_id in non_ids:
                props = jio.PropReader(str(non_id).zfill(5), runid, transient="jet")
                x0, y0, t0 = (
                    props.read("x_mean")[0],
                    props.read("y_mean")[0],
                    props.read("time")[0],
                )
                bs_x_y0 = np.polyval(jx.bs_mp_fit(runid, int(t0 * 2))[1], y0)
                if label_bool:
                    ax.plot(
                        np.polyval(bs_fit[n1], y0) - bs_fit[n1][-1] + (x0 - bs_x_y0),
                        y0,
                        "x",
                        color=CB_color_cycle[n2],
                        label=kinds[n2].capitalize(),
                    )
                    label_bool = False
                else:
                    ax.plot(
                        np.polyval(bs_fit[n1], y0) - bs_fit[n1][-1] + (x0 - bs_x_y0),
                        y0,
                        "x",
                        color=CB_color_cycle[n2],
                    )
        label_bool = draw_labels[n1]
        ax.grid()
        ax.set_xlim(-3, 4)
        if runid in ["ABA", "AEA"]:
            ax.set_ylim(-15, 15)
        else:
            ax.set_ylim(-15, 15)
        if label_bool:
            ax.legend()
        ax_flat[0].set_ylabel("10 nT")
        ax_flat[2].set_ylabel("5 nT")
        ax_flat[2].set_xlabel("5 deg")
        ax_flat[3].set_xlabel("30 deg")

    # Save figure
    plt.tight_layout()

    fig.savefig(wrkdir_DNR + "papu22/Figures/BS_plot.png", dpi=300)
    plt.close(fig)


def get_fcs_jets(runid):

    runids = ["ABA", "ABC", "AEA", "AEC"]

    fcs_ids = []

    for n1 in range(6000):

        try:
            props = jio.PropReader(str(n1).zfill(5), runid, transient="jet")
        except:
            continue

        if props.read("at_bow_shock")[0] != 1:
            continue

        if props.read("time")[0] == 290.0:
            continue

        if "splinter" in props.meta:
            continue

        if not (props.read("at_slams") == 1).any():
            continue
        else:
            fcs_ids.append(n1)

    return np.unique(fcs_ids)


def get_non_jets(runid):

    runids = ["ABA", "ABC", "AEA", "AEC"]

    non_ids = []

    for n1 in range(6000):

        try:
            props = jio.PropReader(str(n1).zfill(5), runid, transient="jet")
        except:
            continue

        if props.read("at_bow_shock")[0] != 1:
            continue

        if props.read("time")[0] == 290.0:
            continue

        if "splinter" in props.meta:
            continue

        if (props.read("at_slams") == 1).any():
            continue
        else:
            non_ids.append(n1)

    return np.unique(non_ids)


def foreshock_jplot_SEA(run_id):

    if run_id == "all":
        runid_list = ["ABA", "ABC", "AEA", "AEC"]
    else:
        runid_list = [run_id]

    x0 = 0.0
    t0 = 0.0
    t_range = np.arange(t0 - 15, t0 + 15 + 0.1, 0.5)
    dx = 227e3 / r_e
    x_range = np.arange(x0 - 20 * dx, x0 + 20 * dx + 0.5 * dx, dx)
    XmeshXY, YmeshXY = np.meshgrid(x_range, t_range)
    vmin_norm = [1.0, 1.0 / 4, 1.0 / 4, 1.0, 5.0, 0.25]
    vmax_norm = [4.0, 1.25, 1.5, 3.0, 25.0, 1.0]

    rho_avg = np.zeros_like(XmeshXY)
    v_avg = np.zeros_like(XmeshXY)
    pdyn_avg = np.zeros_like(XmeshXY)
    B_avg = np.zeros_like(XmeshXY)
    T_avg = np.zeros_like(XmeshXY)
    Tcore_avg = np.zeros_like(XmeshXY)
    mmsx_avg = np.zeros_like(XmeshXY)
    type_count = 0

    sj_rho_avg = np.zeros_like(XmeshXY)
    sj_v_avg = np.zeros_like(XmeshXY)
    sj_pdyn_avg = np.zeros_like(XmeshXY)
    sj_B_avg = np.zeros_like(XmeshXY)
    sj_T_avg = np.zeros_like(XmeshXY)
    sj_Tcore_avg = np.zeros_like(XmeshXY)
    sj_mmsx_avg = np.zeros_like(XmeshXY)
    sj_count = 0

    for runid in runid_list:
        non_ids = np.loadtxt(
            wrkdir_DNR + "papu22/id_txts/new/{}_{}.txt".format(runid, "foreshock"),
            dtype=int,
            ndmin=1,
        )
        sj_ids = get_fcs_jets(runid)

        for non_id in non_ids:
            try:
                rho, v, pdyn, B, T, Tcore, mmsx = np.load(
                    wrkdir_DNR
                    + "papu22/jmap_txts/{}/{}_{}.npy".format(
                        runid, runid, str(non_id).zfill(5)
                    )
                )
                type_count += 1
                rho_avg = rho_avg + rho
                v_avg = v_avg + v
                pdyn_avg = pdyn_avg + pdyn
                B_avg = B_avg + B
                T_avg = T_avg + T
                Tcore_avg = Tcore_avg + Tcore
                mmsx_avg = mmsx_avg + mmsx

            except:
                continue

        for sj_id in sj_ids:
            try:
                rho, v, pdyn, B, T, Tcore, mmsx = np.load(
                    wrkdir_DNR
                    + "papu22/sj_jmap_txts/{}/{}_{}.npy".format(
                        runid, runid, str(sj_id).zfill(5)
                    )
                )
                sj_count += 1
                sj_rho_avg = sj_rho_avg + rho
                sj_v_avg = sj_v_avg + v
                sj_pdyn_avg = sj_pdyn_avg + pdyn
                sj_B_avg = sj_B_avg + B
                sj_T_avg = sj_T_avg + T
                sj_Tcore_avg = sj_Tcore_avg + Tcore
                sj_mmsx_avg = sj_mmsx_avg + mmsx

            except:
                continue

    if type_count != 0:
        rho_avg /= type_count
        v_avg /= type_count
        pdyn_avg /= type_count
        B_avg /= type_count
        T_avg /= type_count
        Tcore_avg /= type_count
        mmsx_avg /= type_count
    else:
        print("No jets of type {} found in run {}".format("foreshock", run_id))
        return 0

    if sj_count != 0:
        sj_rho_avg /= sj_count
        sj_v_avg /= sj_count
        sj_pdyn_avg /= sj_count
        sj_B_avg /= sj_count
        sj_T_avg /= sj_count
        sj_Tcore_avg /= sj_count
        sj_mmsx_avg /= sj_count
    else:
        return 0

    varname_list = [
        "$n$ [$n_\mathrm{sw}$]",
        "$v$ [$v_\mathrm{sw}$]",
        "$P_\mathrm{dyn}$ [$P_\mathrm{dyn,sw}$]",
        "$B$ [$B_\mathrm{IMF}$]",
        "$T$ [$T_\mathrm{sw}$]",
        "$M_{\mathrm{MS},x}$",
    ]

    data_arr = [rho_avg, v_avg, pdyn_avg, B_avg, T_avg, mmsx_avg]

    sj_data_arr = [sj_rho_avg, sj_v_avg, sj_pdyn_avg, sj_B_avg, sj_T_avg, sj_mmsx_avg]

    fig, ax_list = plt.subplots(
        2, len(varname_list), figsize=(24, 16), sharex=True, sharey=True
    )
    im_list = []
    cb_list = []
    sj_im_list = []
    sj_cb_list = []
    fig.suptitle(
        "Run: {}, Type: {}, Nnon = {}, Nfcs = {}".format(
            run_id, "foreshock vs. FCS-jet", type_count, sj_count
        ),
        fontsize=20,
    )
    for idx, ax in enumerate(ax_list[0]):
        ax.tick_params(labelsize=15)
        im_list.append(
            ax.pcolormesh(
                x_range,
                t_range,
                data_arr[idx],
                shading="nearest",
                cmap="viridis",
                # vmin=vmin_norm[idx],
                # vmax=vmax_norm[idx],
                vmin=np.min(data_arr[idx]),
                vmax=np.max(data_arr[idx]),
            )
        )
        cb_list.append(fig.colorbar(im_list[idx], ax=ax))
        ax.contour(XmeshXY, YmeshXY, rho_avg, [2], colors=["black"])
        ax.contour(XmeshXY, YmeshXY, Tcore_avg, [3], colors=[CB_color_cycle[1]])
        ax.contour(XmeshXY, YmeshXY, mmsx_avg, [1.0], colors=[CB_color_cycle[4]])
        ax.set_title(varname_list[idx], fontsize=20, pad=10)
        ax.set_xlim(x_range[0], x_range[-1])
        ax.set_ylim(t_range[0], t_range[-1])
        # ax.set_xlabel("Epoch $x$ [$R_\mathrm{E}$]", fontsize=20, labelpad=10)
        ax.axhline(t0, linestyle="dashed", linewidth=0.6)
        ax.axvline(x0, linestyle="dashed", linewidth=0.6)
    ax_list[0][0].set_ylabel("Epoch time [s]", fontsize=20, labelpad=10)

    for idx, ax in enumerate(ax_list[1]):
        ax.tick_params(labelsize=15)
        sj_im_list.append(
            ax.pcolormesh(
                x_range,
                t_range,
                sj_data_arr[idx],
                shading="nearest",
                cmap="viridis",
                # vmin=vmin_norm[idx],
                # vmax=vmax_norm[idx],
                vmin=np.min(data_arr[idx]),
                vmax=np.max(data_arr[idx]),
            )
        )
        sj_cb_list.append(fig.colorbar(sj_im_list[idx], ax=ax))
        ax.contour(XmeshXY, YmeshXY, sj_rho_avg, [2], colors=["black"])
        ax.contour(XmeshXY, YmeshXY, sj_Tcore_avg, [3], colors=[CB_color_cycle[1]])
        ax.contour(XmeshXY, YmeshXY, sj_mmsx_avg, [1.0], colors=[CB_color_cycle[4]])
        # ax.set_title(varname_list[idx], fontsize=20, pad=10)
        ax.set_xlim(x_range[0], x_range[-1])
        ax.set_ylim(t_range[0], t_range[-1])
        ax.set_xlabel("Epoch $x$ [$R_\mathrm{E}$]", fontsize=20, labelpad=10)
        ax.axhline(t0, linestyle="dashed", linewidth=0.6)
        ax.axvline(x0, linestyle="dashed", linewidth=0.6)
    ax_list[1][0].set_ylabel("Epoch time [s]", fontsize=20, labelpad=10)

    # Save figure
    plt.tight_layout()

    fig.savefig(
        wrkdir_DNR
        + "papu22/Figures/jmap_SEA_foreshock_comparison_{}.png".format(run_id),
        dpi=300,
    )
    plt.close(fig)


def types_jplot_SEA(run_id, kind="beam", version="new"):

    if run_id == "all":
        runid_list = ["ABA", "ABC", "AEA", "AEC"]
    else:
        runid_list = [run_id]

    x0 = 0.0
    t0 = 0.0
    t_range = np.arange(t0 - 15, t0 + 15 + 0.1, 0.5)
    dx = 227e3 / r_e
    x_range = np.arange(x0 - 20 * dx, x0 + 20 * dx + 0.5 * dx, dx)
    XmeshXY, YmeshXY = np.meshgrid(x_range, t_range)

    rho_avg = np.zeros_like(XmeshXY)
    v_avg = np.zeros_like(XmeshXY)
    pdyn_avg = np.zeros_like(XmeshXY)
    B_avg = np.zeros_like(XmeshXY)
    T_avg = np.zeros_like(XmeshXY)
    Tcore_avg = np.zeros_like(XmeshXY)
    mmsx_avg = np.zeros_like(XmeshXY)
    type_count = 0

    for runid in runid_list:
        if version == "old":
            non_ids = np.loadtxt(
                wrkdir_DNR + "papu22/id_txts/{}_{}.txt".format(runid, kind),
                dtype=int,
                ndmin=1,
            )
        elif version == "new":
            non_ids = np.loadtxt(
                wrkdir_DNR + "papu22/id_txts/new/{}_{}.txt".format(runid, kind),
                dtype=int,
                ndmin=1,
            )
        print(non_ids)
        for non_id in non_ids:
            try:
                rho, v, pdyn, B, T, Tcore, mmsx = np.load(
                    wrkdir_DNR
                    + "papu22/jmap_txts/{}/{}_{}.npy".format(
                        runid, runid, str(non_id).zfill(5)
                    )
                )
                type_count += 1
                rho_avg = rho_avg + rho
                v_avg = v_avg + v
                pdyn_avg = pdyn_avg + pdyn
                B_avg = B_avg + B
                T_avg = T_avg + T
                Tcore_avg = Tcore_avg + Tcore
                mmsx_avg = mmsx_avg + mmsx

            except:
                continue

    if type_count != 0:
        rho_avg /= type_count
        v_avg /= type_count
        pdyn_avg /= type_count
        B_avg /= type_count
        T_avg /= type_count
        Tcore_avg /= type_count
        mmsx_avg /= type_count
    else:
        print("No jets of type {} found in run {}".format(kind, run_id))
        return 0

    varname_list = [
        "$n$ [$n_\mathrm{sw}$]",
        "$v$ [$v_\mathrm{sw}$]",
        "$P_\mathrm{dyn}$ [$P_\mathrm{dyn,sw}$]",
        "$B$ [$B_\mathrm{IMF}$]",
        "$T$ [$T_\mathrm{sw}$]",
        "$M_{\mathrm{MS},x}$",
    ]

    data_arr = [rho_avg, v_avg, pdyn_avg, B_avg, T_avg, mmsx_avg]

    fig, ax_list = plt.subplots(
        1, len(varname_list), figsize=(24, 10), sharex=True, sharey=True
    )
    im_list = []
    cb_list = []
    fig.suptitle(
        "Run: {}, Type: {}, N = {}".format(run_id, kind, type_count),
        fontsize=20,
    )
    for idx, ax in enumerate(ax_list):
        ax.tick_params(labelsize=15)
        im_list.append(
            ax.pcolormesh(
                x_range,
                t_range,
                data_arr[idx],
                shading="nearest",
                cmap="viridis",
            )
        )
        cb_list.append(fig.colorbar(im_list[idx], ax=ax))
        ax.contour(XmeshXY, YmeshXY, rho_avg, [2], colors=["black"])
        ax.contour(XmeshXY, YmeshXY, Tcore_avg, [3], colors=[CB_color_cycle[1]])
        ax.contour(XmeshXY, YmeshXY, mmsx_avg, [1.0], colors=[CB_color_cycle[4]])
        ax.set_title(varname_list[idx], fontsize=20, pad=10)
        ax.set_xlim(x_range[0], x_range[-1])
        ax.set_ylim(t_range[0], t_range[-1])
        ax.set_xlabel("Epoch $x$ [$R_\mathrm{E}$]", fontsize=20, labelpad=10)
        ax.axhline(t0, linestyle="dashed", linewidth=0.6)
        ax.axvline(x0, linestyle="dashed", linewidth=0.6)
    ax_list[0].set_ylabel("Epoch time [s]", fontsize=20, labelpad=10)

    # Save figure
    plt.tight_layout()

    fig.savefig(
        wrkdir_DNR + "papu22/Figures/jmap_SEA_{}_{}.png".format(run_id, kind), dpi=300
    )
    plt.close(fig)


def types_P_jplot_SEA(run_id, kind="beam", version="new"):

    if run_id == "all":
        runid_list = ["ABA", "ABC", "AEA", "AEC"]
    else:
        runid_list = [run_id]

    vmin_norm = [0, 0, 0]
    vmax_norm = [1, 1, 1]

    x0 = 0.0
    t0 = 0.0
    t_range = np.arange(t0 - 15, t0 + 15 + 0.1, 0.5)
    dx = 227e3 / r_e
    x_range = np.arange(x0 - 20 * dx, x0 + 20 * dx + 0.5 * dx, dx)
    XmeshXY, YmeshXY = np.meshgrid(x_range, t_range)

    rho_avg = np.zeros_like(XmeshXY)
    pdyn_avg = np.zeros_like(XmeshXY)
    pth_avg = np.zeros_like(XmeshXY)
    pmag_avg = np.zeros_like(XmeshXY)
    ptot_avg = np.zeros_like(XmeshXY)
    Tcore_avg = np.zeros_like(XmeshXY)
    mmsx_avg = np.zeros_like(XmeshXY)
    type_count = 0

    for runid in runid_list:
        if version == "old":
            non_ids = np.loadtxt(
                wrkdir_DNR + "papu22/id_txts/{}_{}.txt".format(runid, kind),
                dtype=int,
                ndmin=1,
            )
        elif version == "new":
            non_ids = np.loadtxt(
                wrkdir_DNR + "papu22/id_txts/new/{}_{}.txt".format(runid, kind),
                dtype=int,
                ndmin=1,
            )
        print(non_ids)
        for non_id in non_ids:
            try:
                rho, pdyn, pmag, pth, ptot, Tcore, mmsx = np.load(
                    wrkdir_DNR
                    + "papu22/P_jmap_txts/{}/{}_{}.npy".format(
                        runid, runid, str(non_id).zfill(5)
                    )
                )
                type_count += 1
                rho_avg = rho_avg + rho
                pdyn_avg = pdyn_avg + pdyn
                pth_avg = pth_avg + pth
                pmag_avg = pmag_avg + pmag
                ptot_avg = ptot_avg + ptot
                Tcore_avg = Tcore_avg + Tcore
                mmsx_avg = mmsx_avg + mmsx

            except:
                continue

    if type_count != 0:
        rho_avg /= type_count
        pdyn_avg /= type_count
        pth_avg /= type_count
        pmag_avg /= type_count
        ptot_avg /= type_count
        Tcore_avg /= type_count
        mmsx_avg /= type_count
    else:
        print("No jets of type {} found in run {}".format(kind, run_id))
        return 0

    varname_list = [
        "$P_{th}/P_{tot}$",
        "$P_{dyn}/P_{tot}$",
        "$P_{mag}/P_{tot}$",
    ]

    data_arr = [pth_avg / ptot_avg, pdyn_avg / ptot_avg, pmag_avg / ptot_avg]

    fig, ax_list = plt.subplots(
        1, len(varname_list), figsize=(24, 10), sharex=True, sharey=True
    )
    im_list = []
    cb_list = []
    fig.suptitle(
        "Run: {}, Type: {}, N = {}".format(run_id, kind, type_count),
        fontsize=20,
    )
    for idx, ax in enumerate(ax_list):
        ax.tick_params(labelsize=15)
        im_list.append(
            ax.pcolormesh(
                x_range,
                t_range,
                data_arr[idx],
                shading="nearest",
                cmap="viridis",
                vmin=vmin_norm[idx],
                vmax=vmax_norm[idx],
            )
        )
        cb_list.append(fig.colorbar(im_list[idx], ax=ax))
        ax.contour(XmeshXY, YmeshXY, rho_avg, [2], colors=["black"])
        ax.contour(XmeshXY, YmeshXY, Tcore_avg, [3], colors=[CB_color_cycle[1]])
        ax.contour(XmeshXY, YmeshXY, mmsx_avg, [1.0], colors=[CB_color_cycle[4]])
        ax.set_title(varname_list[idx], fontsize=20, pad=10)
        ax.set_xlim(x_range[0], x_range[-1])
        ax.set_ylim(t_range[0], t_range[-1])
        ax.set_xlabel("Epoch $x$ [$R_\mathrm{E}$]", fontsize=20, labelpad=10)
        ax.axhline(t0, linestyle="dashed", linewidth=0.6)
        ax.axvline(x0, linestyle="dashed", linewidth=0.6)
    ax_list[0].set_ylabel("Epoch time [s]", fontsize=20, labelpad=10)

    # Save figure
    plt.tight_layout()

    fig.savefig(
        wrkdir_DNR + "papu22/Figures/P_jmap_SEA_{}_{}.png".format(run_id, kind), dpi=300
    )
    plt.close(fig)


def fcs_jet_jplot_txtonly(runid):

    dx = 227e3 / r_e

    # Solar wind parameters for the different runs
    # n [m^-3], v [m/s], B [T], T [K]
    runid_list = ["ABA", "ABC", "AEA", "AEC"]
    sw_pars = [
        [1.0, 750.0, 5.0, 0.5],
        [3.3, 600.0, 5.0, 0.5],
        [1.0, 750.0, 10.0, 0.5],
        [3.3, 600.0, 10.0, 0.5],
    ]
    n_sw, v_sw, B_sw, T_sw = sw_pars[runid_list.index(runid)]

    # Path to vlsv files for current run
    bulkpath = jx.find_bulkpath(runid)

    # Get IDs of fcs-jets
    sj_ids = get_fcs_jets(runid)

    # Loop through non-fcs-jet IDs
    for sj_id in sj_ids:
        print("Non-FCS jets for run {}: {}".format(runid, sj_id))
        props = jio.PropReader(str(sj_id).zfill(5), runid, transient="jet")
        x0, y0 = (props.read("x_mean")[0], props.read("y_mean")[0])
        t0 = props.read("time")[0]
        fnr0 = int(t0 * 2)

        fnr_range = np.arange(fnr0 - 30, fnr0 + 30 + 1)
        t_range = np.arange(t0 - 15, t0 + 15 + 0.1, 0.5)
        # Get cellid of initial position
        cellid = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
        ).get_cellid([x0 * r_e, y0 * r_e, 0 * r_e])

        cell_range = np.arange(cellid - 20, cellid + 20 + 1)
        x_range = np.arange(x0 - 20 * dx, x0 + 20 * dx + 0.5 * dx, dx)

        rho_arr = []
        v_arr = []
        pdyn_arr = []
        B_arr = []
        T_arr = []
        Tcore_arr = []
        mmsx_arr = []

        for fnr in fnr_range:
            vlsvobj = pt.vlsvfile.VlsvReader(
                bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
            )
            rho_arr.append(vlsvobj.read_variable("rho", cellids=cell_range))
            v_arr.append(
                vlsvobj.read_variable("v", operator="magnitude", cellids=cell_range)
            )
            pdyn_arr.append(vlsvobj.read_variable("Pdyn", cellids=cell_range))
            B_arr.append(
                vlsvobj.read_variable("B", operator="magnitude", cellids=cell_range)
            )
            T_arr.append(vlsvobj.read_variable("Temperature", cellids=cell_range))
            Tcore_arr.append(vlsvobj.read_variable("core_heating", cellids=cell_range))
            mmsx_arr.append(vlsvobj.read_variable("Mmsx", cellids=cell_range))

        rho_arr = np.array(rho_arr) / 1.0e6 / n_sw
        v_arr = np.array(v_arr) / 1.0e3 / v_sw
        pdyn_arr = (
            np.array(pdyn_arr) / 1.0e-9 / (m_p * n_sw * 1e6 * v_sw * v_sw * 1e6 / 1e-9)
        )
        B_arr = np.array(B_arr) / 1.0e-9 / B_sw
        T_arr = np.array(T_arr) / 1.0e6 / T_sw
        Tcore_arr = np.array(Tcore_arr) / 1.0e6 / T_sw
        mmsx_arr = np.array(mmsx_arr)

        np.save(
            wrkdir_DNR
            + "papu22/sj_jmap_txts/{}/{}_{}".format(runid, runid, str(sj_id).zfill(5)),
            np.array([rho_arr, v_arr, pdyn_arr, B_arr, T_arr, Tcore_arr, mmsx_arr]),
        )


def non_jet_jplots(runid):

    CB_color_cycle = jx.CB_color_cycle

    dx = 227e3 / r_e
    varname_list = [
        "$n$ [$n_\mathrm{sw}$]",
        "$v$ [$v_\mathrm{sw}$]",
        "$P_\mathrm{dyn}$ [$P_\mathrm{dyn,sw}$]",
        "$B$ [$B_\mathrm{IMF}$]",
        "$T$ [$T_\mathrm{sw}$]",
    ]

    # Solar wind parameters for the different runs
    # n [m^-3], v [m/s], B [T], T [K]
    runid_list = ["ABA", "ABC", "AEA", "AEC"]
    sw_pars = [
        [1.0, 750.0, 5.0, 0.5],
        [3.3, 600.0, 5.0, 0.5],
        [1.0, 750.0, 10.0, 0.5],
        [3.3, 600.0, 10.0, 0.5],
    ]
    n_sw, v_sw, B_sw, T_sw = sw_pars[runid_list.index(runid)]
    vmin_norm = [1.0 / 2, 1.0 / 6, 1.0 / 6, 1.0 / 2, 1.0]
    vmax_norm = [6.0, 2.0, 2.0, 6.0, 36.0]

    # Path to vlsv files for current run
    bulkpath = jx.find_bulkpath(runid)

    # Get IDs of non-fcs-jets
    non_sj_ids = get_non_jets(runid)

    # Loop through non-fcs-jet IDs
    for non_id in non_sj_ids:
        print("Non-FCS jets for run {}: {}".format(runid, non_id))
        props = jio.PropReader(str(non_id).zfill(5), runid, transient="jet")
        x0, y0 = (props.read("x_mean")[0], props.read("y_mean")[0])
        t0 = props.read("time")[0]
        fnr0 = int(t0 * 2)

        fnr_range = np.arange(fnr0 - 30, fnr0 + 30 + 1)
        t_range = np.arange(t0 - 15, t0 + 15 + 0.1, 0.5)
        # Get cellid of initial position
        cellid = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
        ).get_cellid([x0 * r_e, y0 * r_e, 0 * r_e])

        cell_range = np.arange(cellid - 20, cellid + 20 + 1)
        x_range = np.arange(x0 - 20 * dx, x0 + 20 * dx + 0.5 * dx, dx)

        XmeshXY, YmeshXY = np.meshgrid(x_range, t_range)

        rho_arr = []
        v_arr = []
        pdyn_arr = []
        B_arr = []
        T_arr = []
        Tcore_arr = []
        mmsx_arr = []

        for fnr in fnr_range:
            vlsvobj = pt.vlsvfile.VlsvReader(
                bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
            )
            rho_arr.append(vlsvobj.read_variable("rho", cellids=cell_range))
            v_arr.append(
                vlsvobj.read_variable("v", operator="magnitude", cellids=cell_range)
            )
            pdyn_arr.append(vlsvobj.read_variable("Pdyn", cellids=cell_range))
            B_arr.append(
                vlsvobj.read_variable("B", operator="magnitude", cellids=cell_range)
            )
            T_arr.append(vlsvobj.read_variable("Temperature", cellids=cell_range))
            Tcore_arr.append(vlsvobj.read_variable("core_heating", cellids=cell_range))
            mmsx_arr.append(vlsvobj.read_variable("Mmsx", cellids=cell_range))

        rho_arr = np.array(rho_arr) / 1.0e6 / n_sw
        v_arr = np.array(v_arr) / 1.0e3 / v_sw
        pdyn_arr = (
            np.array(pdyn_arr) / 1.0e-9 / (m_p * n_sw * 1e6 * v_sw * v_sw * 1e6 / 1e-9)
        )
        B_arr = np.array(B_arr) / 1.0e-9 / B_sw
        T_arr = np.array(T_arr) / 1.0e6 / T_sw
        Tcore_arr = np.array(Tcore_arr) / 1.0e6 / T_sw
        mmsx_arr = np.array(mmsx_arr)

        data_arr = [rho_arr, v_arr, pdyn_arr, B_arr, T_arr]

        fig, ax_list = plt.subplots(
            1, len(varname_list), figsize=(20, 10), sharex=True, sharey=True
        )
        im_list = []
        cb_list = []
        fig.suptitle(
            "Run: {}, JetID: {}, $y$ = {:.3f} ".format(runid, non_id, y0)
            + "$R_\mathrm{E}$",
            fontsize=20,
        )
        for idx, ax in enumerate(ax_list):
            ax.tick_params(labelsize=15)
            im_list.append(
                ax.pcolormesh(
                    x_range,
                    t_range,
                    data_arr[idx],
                    shading="nearest",
                    cmap="Greys",
                    vmin=vmin_norm[idx],
                    vmax=vmax_norm[idx],
                )
            )
            cb_list.append(fig.colorbar(im_list[idx], ax=ax))
            ax.contour(XmeshXY, YmeshXY, rho_arr, [2], colors=["black"])
            ax.contour(XmeshXY, YmeshXY, Tcore_arr, [3], colors=[CB_color_cycle[1]])
            ax.contour(XmeshXY, YmeshXY, mmsx_arr, [1.0], colors=[CB_color_cycle[4]])
            ax.set_title(varname_list[idx], fontsize=20, pad=10)
            ax.set_xlim(x_range[0], x_range[-1])
            ax.set_ylim(t_range[0], t_range[-1])
            ax.set_xlabel("$x$ [$R_\mathrm{E}$]", fontsize=20, labelpad=10)
            ax.axhline(t0, linestyle="dashed", linewidth=0.6)
            ax.axvline(x0, linestyle="dashed", linewidth=0.6)
        ax_list[0].set_ylabel("Simulation time [s]", fontsize=20, labelpad=10)

        # Save figure
        plt.tight_layout()

        # fig.savefig(
        #     wrkdir_DNR
        #     + "papu22/Figures/jmaps/{}_{}.pdf".format(runid, str(non_id).zfill(5))
        # )
        fig.savefig(
            wrkdir_DNR
            + "papu22/Figures/jmaps/{}_{}.png".format(runid, str(non_id).zfill(5)),
            dpi=300,
        )
        plt.close(fig)

        np.save(
            wrkdir_DNR
            + "papu22/jmap_txts/{}/{}_{}".format(runid, runid, str(non_id).zfill(5)),
            np.array([rho_arr, v_arr, pdyn_arr, B_arr, T_arr, Tcore_arr, mmsx_arr]),
        )


def P_jplots(runid):

    CB_color_cycle = jx.CB_color_cycle

    dx = 227e3 / r_e
    varname_list = [
        "$P_{th}/P_{tot}$",
        "$P_{dyn}/P_{tot}$",
        "$P_{mag}/P_{tot}$",
    ]

    # Solar wind parameters for the different runs
    # n [m^-3], v [m/s], B [T], T [K]
    runid_list = ["ABA", "ABC", "AEA", "AEC"]
    sw_pars = [
        [1.0, 750.0, 5.0, 0.5],
        [3.3, 600.0, 5.0, 0.5],
        [1.0, 750.0, 10.0, 0.5],
        [3.3, 600.0, 10.0, 0.5],
    ]
    n_sw, v_sw, B_sw, T_sw = sw_pars[runid_list.index(runid)]
    vmin_norm = [0, 0, 0]
    vmax_norm = [1, 1, 1]

    # Path to vlsv files for current run
    bulkpath = jx.find_bulkpath(runid)

    # Get IDs of non-fcs-jets
    non_sj_ids = get_non_jets(runid)

    # Loop through non-fcs-jet IDs
    for non_id in non_sj_ids:
        print("Non-FCS jets for run {}: {}".format(runid, non_id))
        props = jio.PropReader(str(non_id).zfill(5), runid, transient="jet")
        x0, y0 = (props.read("x_mean")[0], props.read("y_mean")[0])
        t0 = props.read("time")[0]
        fnr0 = int(t0 * 2)

        fnr_range = np.arange(fnr0 - 30, fnr0 + 30 + 1)
        t_range = np.arange(t0 - 15, t0 + 15 + 0.1, 0.5)
        # Get cellid of initial position
        cellid = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
        ).get_cellid([x0 * r_e, y0 * r_e, 0 * r_e])

        cell_range = np.arange(cellid - 20, cellid + 20 + 1)
        x_range = np.arange(x0 - 20 * dx, x0 + 20 * dx + 0.5 * dx, dx)

        XmeshXY, YmeshXY = np.meshgrid(x_range, t_range)

        rho_arr = []
        pdyn_arr = []
        pmag_arr = []
        pth_arr = []
        ptot_arr = []
        Tcore_arr = []
        mmsx_arr = []

        for fnr in fnr_range:
            vlsvobj = pt.vlsvfile.VlsvReader(
                bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
            )
            rho_arr.append(vlsvobj.read_variable("rho", cellids=cell_range))
            pdyn_arr.append(vlsvobj.read_variable("Pdyn", cellids=cell_range))
            pmag_arr.append(vlsvobj.read_variable("Pmag", cellids=cell_range))
            pth_arr.append(vlsvobj.read_variable("Pressure", cellids=cell_range))
            ptot_arr.append(vlsvobj.read_variable("Ptot", cellids=cell_range))
            Tcore_arr.append(vlsvobj.read_variable("core_heating", cellids=cell_range))
            mmsx_arr.append(vlsvobj.read_variable("Mmsx", cellids=cell_range))

        rho_arr = np.array(rho_arr) / 1.0e6 / n_sw
        Tcore_arr = np.array(Tcore_arr) / 1.0e6 / T_sw
        mmsx_arr = np.array(mmsx_arr)
        pdyn_arr = np.array(pdyn_arr)
        pmag_arr = np.array(pmag_arr)
        pth_arr = np.array(pth_arr)
        ptot_arr = np.array(ptot_arr)

        data_arr = [pth_arr / ptot_arr, pdyn_arr / ptot_arr, pmag_arr / ptot_arr]

        fig, ax_list = plt.subplots(
            1, len(varname_list), figsize=(20, 10), sharex=True, sharey=True
        )
        im_list = []
        cb_list = []
        fig.suptitle(
            "Run: {}, JetID: {}, $y$ = {:.3f} ".format(runid, non_id, y0)
            + "$R_\mathrm{E}$",
            fontsize=20,
        )
        for idx, ax in enumerate(ax_list):
            ax.tick_params(labelsize=15)
            im_list.append(
                ax.pcolormesh(
                    x_range,
                    t_range,
                    data_arr[idx],
                    shading="nearest",
                    cmap="Greys",
                    vmin=vmin_norm[idx],
                    vmax=vmax_norm[idx],
                )
            )
            cb_list.append(fig.colorbar(im_list[idx], ax=ax))
            ax.contour(XmeshXY, YmeshXY, rho_arr, [2], colors=["black"])
            ax.contour(XmeshXY, YmeshXY, Tcore_arr, [3], colors=[CB_color_cycle[1]])
            ax.contour(XmeshXY, YmeshXY, mmsx_arr, [1.0], colors=[CB_color_cycle[4]])
            ax.set_title(varname_list[idx], fontsize=20, pad=10)
            ax.set_xlim(x_range[0], x_range[-1])
            ax.set_ylim(t_range[0], t_range[-1])
            ax.set_xlabel("$x$ [$R_\mathrm{E}$]", fontsize=20, labelpad=10)
            ax.axhline(t0, linestyle="dashed", linewidth=0.6)
            ax.axvline(x0, linestyle="dashed", linewidth=0.6)
        ax_list[0].set_ylabel("Simulation time [s]", fontsize=20, labelpad=10)

        # Save figure
        plt.tight_layout()

        # fig.savefig(
        #     wrkdir_DNR
        #     + "papu22/Figures/jmaps/{}_{}.pdf".format(runid, str(non_id).zfill(5))
        # )
        fig.savefig(
            wrkdir_DNR
            + "papu22/Figures/P_jmaps/{}_{}.png".format(runid, str(non_id).zfill(5)),
            dpi=300,
        )
        plt.close(fig)

        np.save(
            wrkdir_DNR
            + "papu22/P_jmap_txts/{}/{}_{}".format(runid, runid, str(non_id).zfill(5)),
            np.array(
                [rho_arr, pdyn_arr, pmag_arr, pth_arr, ptot_arr, Tcore_arr, mmsx_arr]
            ),
        )


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


def SEA_types(run_id="all"):
    """
    Superposed epoch analysis of different types of non-fcs-jets
    """

    if run_id == "all":
        runid_list = ["ABA", "ABC", "AEA", "AEC"]
    elif run_id == "30":
        runid_list = ["ABA", "AEA"]
    elif run_id == "05":
        runid_list = ["ABC", "AEC"]
    else:
        runid_list = [run_id]

    # Initialise array of times relative to epoch time
    t_arr = np.arange(-10.0, 10.05, 0.5)

    # Initialise number of jets of different types
    beam_count = 0
    stripe_count = 0
    reformation_count = 0
    foreshock_count = 0
    complex_count = 0

    # Initialise arrays of averages
    beam_avg = np.zeros((6, 41), dtype=float)
    stripe_avg = np.zeros((6, 41), dtype=float)
    reformation_avg = np.zeros((6, 41), dtype=float)
    foreshock_avg = np.zeros((6, 41), dtype=float)
    complex_avg = np.zeros((6, 41), dtype=float)

    # Initialise figure, add grids, add axis labels
    fig, ax_list = plt.subplots(6, 1, sharex=True, figsize=(8, 10))

    for ax in ax_list:
        ax.grid()

    ax_list[0].set_ylabel("$n~[n_\mathrm{sw}]$")
    ax_list[1].set_ylabel("$v~[v_\mathrm{sw}]$")
    ax_list[2].set_ylabel("$P_\mathrm{dyn}~[P_\mathrm{dyn,sw}]$")
    ax_list[3].set_ylabel("$B~[B_\mathrm{IMF}]$")
    ax_list[4].set_ylabel("$T_\mathrm{perp}~[T_\mathrm{sw}]$")
    ax_list[5].set_ylabel("$T_\mathrm{par}~[T_\mathrm{sw}]$")
    ax_list[-1].set_xlabel("$\\Delta t~[\mathrm{s}]$")

    # Loop over runs
    for runid in runid_list:

        jet_ids = np.loadtxt(
            wrkdir_DNR + "papu22/id_txts/new/{}_beam.txt".format(runid),
            dtype=int,
            ndmin=1,
        )
        for jet_id in jet_ids:
            data = np.loadtxt(
                wrkdir_DNR
                + "papu22/non_jets/{}/timeseries_{}.txt".format(runid, jet_id)
            ).T[1:]
            beam_count += 1  # Iterate jet count

            # Loop over n,v,pdyn,B,Tperp,Tpar
            for n2 in range(6):

                # Plot timeseries of deltas relative to epoch time
                ax_list[n2].plot(t_arr, data[n2], color="lightgray", zorder=1)

                # Add timeseries of deltas relative to epoch time to average array
                beam_avg[n2] += data[n2]

        jet_ids = np.loadtxt(
            wrkdir_DNR + "papu22/id_txts/new/{}_stripe.txt".format(runid),
            dtype=int,
            ndmin=1,
        )
        for jet_id in jet_ids:
            data = np.loadtxt(
                wrkdir_DNR
                + "papu22/non_jets/{}/timeseries_{}.txt".format(runid, jet_id)
            ).T[1:]
            stripe_count += 1  # Iterate jet count

            # Loop over n,v,pdyn,B,Tperp,Tpar
            for n2 in range(6):

                # Plot timeseries of deltas relative to epoch time
                ax_list[n2].plot(t_arr, data[n2], color="lightgray", zorder=1)

                # Add timeseries of deltas relative to epoch time to average array
                stripe_avg[n2] += data[n2]

        jet_ids = np.loadtxt(
            wrkdir_DNR + "papu22/id_txts/new/{}_reformation.txt".format(runid),
            dtype=int,
            ndmin=1,
        )
        for jet_id in jet_ids:
            data = np.loadtxt(
                wrkdir_DNR
                + "papu22/non_jets/{}/timeseries_{}.txt".format(runid, jet_id)
            ).T[1:]
            reformation_count += 1  # Iterate jet count

            # Loop over n,v,pdyn,B,Tperp,Tpar
            for n2 in range(6):

                # Plot timeseries of deltas relative to epoch time
                ax_list[n2].plot(t_arr, data[n2], color="lightgray", zorder=1)

                # Add timeseries of deltas relative to epoch time to average array
                reformation_avg[n2] += data[n2]

        jet_ids = np.loadtxt(
            wrkdir_DNR + "papu22/id_txts/new/{}_foreshock.txt".format(runid),
            dtype=int,
            ndmin=1,
        )
        for jet_id in jet_ids:
            data = np.loadtxt(
                wrkdir_DNR
                + "papu22/non_jets/{}/timeseries_{}.txt".format(runid, jet_id)
            ).T[1:]
            foreshock_count += 1  # Iterate jet count

            # Loop over n,v,pdyn,B,Tperp,Tpar
            for n2 in range(6):

                # Plot timeseries of deltas relative to epoch time
                ax_list[n2].plot(t_arr, data[n2], color="lightgray", zorder=1)

                # Add timeseries of deltas relative to epoch time to average array
                foreshock_avg[n2] += data[n2]

        jet_ids = np.loadtxt(
            wrkdir_DNR + "papu22/id_txts/new/{}_complex.txt".format(runid),
            dtype=int,
            ndmin=1,
        )
        for jet_id in jet_ids:
            data = np.loadtxt(
                wrkdir_DNR
                + "papu22/non_jets/{}/timeseries_{}.txt".format(runid, jet_id)
            ).T[1:]
            complex_count += 1  # Iterate jet count

            # Loop over n,v,pdyn,B,Tperp,Tpar
            for n2 in range(6):

                # Plot timeseries of deltas relative to epoch time
                ax_list[n2].plot(t_arr, data[n2], color="lightgray", zorder=1)

                # Add timeseries of deltas relative to epoch time to average array
                complex_avg[n2] += data[n2]

    # Calculate averages
    if beam_count != 0:
        beam_avg /= beam_count
    else:
        beam_avg[:, :] = np.nan

    if stripe_count != 0:
        stripe_avg /= stripe_count
    else:
        stripe_avg[:, :] = np.nan

    if reformation_count != 0:
        reformation_avg /= reformation_count
    else:
        reformation_avg[:, :] = np.nan

    if foreshock_count != 0:
        foreshock_avg /= foreshock_count
    else:
        foreshock_avg[:, :] = np.nan

    if complex_count != 0:
        complex_avg /= complex_count
    else:
        complex_avg[:, :] = np.nan

    # Plot averages of n,v,pdyn,B,Tperp,Tpar
    for n2 in range(6):
        ax_list[n2].plot(
            t_arr,
            beam_avg[n2],
            color=jx.CB_color_cycle[0],
            label="Beam",
            zorder=2,
        )
        ax_list[n2].plot(
            t_arr,
            stripe_avg[n2],
            color=jx.CB_color_cycle[1],
            label="Stripe",
            zorder=2,
        )
        ax_list[n2].plot(
            t_arr,
            reformation_avg[n2],
            color=jx.CB_color_cycle[2],
            label="Reformation",
            zorder=2,
        )
        ax_list[n2].plot(
            t_arr,
            foreshock_avg[n2],
            color=jx.CB_color_cycle[3],
            label="Foreshock",
            zorder=2,
        )
        ax_list[n2].plot(
            t_arr,
            complex_avg[n2],
            color=jx.CB_color_cycle[4],
            label="Complex",
            zorder=2,
        )

    # Add legend
    ax_list[0].legend()
    ax_list[0].set_title(run_id)

    # Save as pdf and png and close figure
    plt.tight_layout()

    # fig.savefig(wrkdir_DNR + "papu22/Figures/SEA_types_{}.pdf".format(run_id))
    fig.savefig(wrkdir_DNR + "papu22/Figures/SEA_types_{}.png".format(run_id), dpi=300)
    plt.close(fig)


def SEA_plots(zero_level=False, run_id="all"):
    """
    Superposed epoch analysis of fcs-jet vs. non-fcs-jet start location properties
    """

    if run_id == "all":
        runid_list = ["ABA", "ABC", "AEA", "AEC"]
    elif run_id == "30":
        runid_list = ["ABA", "AEA"]
    elif run_id == "05":
        runid_list = ["ABC", "AEC"]
    else:
        runid_list = [run_id]

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

    if zero_level:
        ax_list[0].set_ylabel("$\\Delta n~[n_\mathrm{sw}]$")
        ax_list[1].set_ylabel("$\\Delta v~[v_\mathrm{sw}]$")
        ax_list[2].set_ylabel("$\\Delta P_\mathrm{dyn}~[P_\mathrm{dyn,sw}]$")
        ax_list[3].set_ylabel("$\\Delta B~[B_\mathrm{IMF}]$")
        ax_list[4].set_ylabel("$\\Delta T_\mathrm{perp}~[T_\mathrm{sw}]$")
        ax_list[5].set_ylabel("$\\Delta T_\mathrm{par}~[T_\mathrm{sw}]$")
    else:
        ax_list[0].set_ylabel("$n~[n_\mathrm{sw}]$")
        ax_list[1].set_ylabel("$v~[v_\mathrm{sw}]$")
        ax_list[2].set_ylabel("$P_\mathrm{dyn}~[P_\mathrm{dyn,sw}]$")
        ax_list[3].set_ylabel("$B~[B_\mathrm{IMF}]$")
        ax_list[4].set_ylabel("$T_\mathrm{perp}~[T_\mathrm{sw}]$")
        ax_list[5].set_ylabel("$T_\mathrm{par}~[T_\mathrm{sw}]$")
    ax_list[-1].set_xlabel("$\\Delta t~[\mathrm{s}]$")

    # Loop over runs
    for runid in runid_list:

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
                    if not zero_level:
                        ax_list[n2].plot(t_arr, data[n2], color="darkgray", zorder=0)
                    else:
                        ax_list[n2].plot(
                            t_arr, data[n2] - data[n2][20], color="darkgray", zorder=0
                        )

                    # Add timeseries of deltas relative to epoch time to average array
                    if not zero_level:
                        fcs_jet_avg[n2] += data[n2]
                    else:
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
                    if not zero_level:
                        ax_list[n2].plot(t_arr, data[n2], color="lightgray", zorder=1)
                    else:
                        ax_list[n2].plot(
                            t_arr, data[n2] - data[n2][20], color="lightgray", zorder=1
                        )

                    # Add timeseries of deltas relative to epoch time to average array
                    if not zero_level:
                        non_jet_avg[n2] += data[n2]
                    else:
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
    ax_list[0].set_title(run_id)

    # Save as pdf and png and close figure
    plt.tight_layout()

    # fig.savefig(
    #    wrkdir_DNR + "papu22/Figures/SEA_plot_zl{}_{}.pdf".format(zero_level, run_id)
    # )
    fig.savefig(
        wrkdir_DNR + "papu22/Figures/SEA_plot_zl{}_{}.png".format(zero_level, run_id),
        dpi=300,
    )
    plt.close(fig)


def non_type_hist(run_id="all"):
    if run_id == "all":
        run_arr = ["ABA", "ABC", "AEA", "AEC"]
    elif run_id == "30":
        runid_list = ["ABA", "AEA"]
    elif run_id == "05":
        runid_list = ["ABC", "AEC"]
    else:
        run_arr = [run_id]

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
    # delta n, delta v, delta Pdyn, delta B, delta T, Lifetime, Tangential size, Size ratio, first cone, first y
    # Count: 10
    vars_list = [
        "Dn",
        "Dv",
        "Dpd",
        "DB",
        "DT",
        "duration",
        "size_tan",
        "size_ratio",
        "first_cone",
        "first_y",
    ]
    label_list = [
        "$\\Delta n~[n_\mathrm{sw}]$",
        "$\\Delta |v|~[v_\mathrm{sw}]$",
        "$\\Delta P_\mathrm{dyn}~[P_\mathrm{dyn,sw}]$",
        "$\\Delta |B|~[B_\mathrm{IMF}]$",
        "$\\Delta T~[T_\mathrm{sw}]$",
        "$Lifetime~[\mathrm{s}]$",
        "$Tangential$\n$size~[R_\mathrm{E}]$",
        "Size Ratio",
        "First cone [$^\circ$]",
        "First $y$ [$R_\mathrm{E}$]",
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
        np.linspace(-60, 60, 10 + 1),
        np.linspace(-6, 6, 10 + 1),
    ]
    pos_list = [
        "left",
        "left",
        "left",
        "left",
        "left",
        "right",
        "right",
        "right",
        "right",
        "right",
    ]
    beam_props = [[], [], [], [], [], [], [], [], [], []]
    stripe_props = [[], [], [], [], [], [], [], [], [], []]
    reformation_props = [[], [], [], [], [], [], [], [], [], []]
    foreshock_props = [[], [], [], [], [], [], [], [], [], []]
    complex_props = [[], [], [], [], [], [], [], [], [], []]

    # Loop over runs
    for runid in run_arr:

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
            1,
            1,
        ]

        for jet_id in np.loadtxt(
            wrkdir_DNR + "papu22/id_txts/new/{}_beam.txt".format(runid),
            dtype=int,
            ndmin=1,
        ):
            # Read properties
            props = jio.PropReader(str(jet_id).zfill(5), runid, transient="jet")

            # Loop over variables
            for n1 in range(len(vars_list)):

                # ...or at the time of maximum area?
                beam_props[n1].append(props.read_at_amax(vars_list[n1]) / sw_norm[n1])

        for jet_id in np.loadtxt(
            wrkdir_DNR + "papu22/id_txts/new/{}_stripe.txt".format(runid),
            dtype=int,
            ndmin=1,
        ):
            # Read properties
            props = jio.PropReader(str(jet_id).zfill(5), runid, transient="jet")

            # Loop over variables
            for n1 in range(len(vars_list)):

                # ...or at the time of maximum area?
                stripe_props[n1].append(props.read_at_amax(vars_list[n1]) / sw_norm[n1])

        for jet_id in np.loadtxt(
            wrkdir_DNR + "papu22/id_txts/new/{}_reformation.txt".format(runid),
            dtype=int,
            ndmin=1,
        ):
            # Read properties
            props = jio.PropReader(str(jet_id).zfill(5), runid, transient="jet")

            # Loop over variables
            for n1 in range(len(vars_list)):

                # ...or at the time of maximum area?
                reformation_props[n1].append(
                    props.read_at_amax(vars_list[n1]) / sw_norm[n1]
                )

        for jet_id in np.loadtxt(
            wrkdir_DNR + "papu22/id_txts/new/{}_foreshock.txt".format(runid),
            dtype=int,
            ndmin=1,
        ):
            # Read properties
            props = jio.PropReader(str(jet_id).zfill(5), runid, transient="jet")

            # Loop over variables
            for n1 in range(len(vars_list)):

                # ...or at the time of maximum area?
                foreshock_props[n1].append(
                    props.read_at_amax(vars_list[n1]) / sw_norm[n1]
                )

        for jet_id in np.loadtxt(
            wrkdir_DNR + "papu22/id_txts/new/{}_complex.txt".format(runid),
            dtype=int,
            ndmin=1,
        ):
            # Read properties
            props = jio.PropReader(str(jet_id).zfill(5), runid, transient="jet")

            # Loop over variables
            for n1 in range(len(vars_list)):

                # ...or at the time of maximum area?
                complex_props[n1].append(
                    props.read_at_amax(vars_list[n1]) / sw_norm[n1]
                )

    # Make figure
    fig, ax_list = plt.subplots(5, 2, figsize=(7, 13))

    ax_flat = ax_list.T.flatten()

    for n1, ax in enumerate(ax_flat):
        ax.set_ylabel(label_list[n1], labelpad=10, fontsize=20)
        ax.yaxis.set_label_position(pos_list[n1])
        ax.grid()
        ax.tick_params(labelsize=15)

        ax.hist(
            beam_props[n1],
            histtype="step",
            weights=np.ones(len(beam_props[n1])) / float(len(beam_props[n1])),
            bins=bins_list[n1],
            color=jx.CB_color_cycle[0],
            label="Beam",
        )

        ax.hist(
            stripe_props[n1],
            histtype="step",
            weights=np.ones(len(stripe_props[n1])) / float(len(stripe_props[n1])),
            bins=bins_list[n1],
            color=jx.CB_color_cycle[1],
            label="Stripe",
        )

        ax.hist(
            reformation_props[n1],
            histtype="step",
            weights=np.ones(len(reformation_props[n1]))
            / float(len(reformation_props[n1])),
            bins=bins_list[n1],
            color=jx.CB_color_cycle[2],
            label="Reformation",
        )

        ax.hist(
            foreshock_props[n1],
            histtype="step",
            weights=np.ones(len(foreshock_props[n1])) / float(len(foreshock_props[n1])),
            bins=bins_list[n1],
            color=jx.CB_color_cycle[3],
            label="Foreshock",
        )

        ax.hist(
            complex_props[n1],
            histtype="step",
            weights=np.ones(len(complex_props[n1])) / float(len(complex_props[n1])),
            bins=bins_list[n1],
            color=jx.CB_color_cycle[4],
            label="Complex",
        )

    ax_flat[6].legend(frameon=False, markerscale=0.5)
    ax_flat[0].set_title(run_id, fontsize=20)

    # Save figure
    plt.tight_layout()

    # fig.savefig(wrkdir_DNR + "papu22/Figures/FCS_type_hist_{}.pdf".format(run_id))
    fig.savefig(
        wrkdir_DNR + "papu22/Figures/FCS_type_hist_{}.png".format(run_id), dpi=300
    )
    plt.close(fig)


def fcs_non_jet_hist(lastbs=False, run_id="all"):

    if run_id == "all":
        run_arr = ["ABA", "ABC", "AEA", "AEC"]
    elif run_id == "30":
        runid_list = ["ABA", "AEA"]
    elif run_id == "05":
        runid_list = ["ABC", "AEC"]
    else:
        run_arr = [run_id]

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
    # delta n, delta v, delta Pdyn, delta B, delta T, Lifetime, Tangential size, Size ratio, leaves bs, dies at bs
    # Count: 10
    vars_list = [
        "Dn",
        "Dv",
        "Dpd",
        "DB",
        "DT",
        "duration",
        "size_tan",
        "size_ratio",
        "leaves_bs",
        "dies_at_bs",
    ]
    label_list = [
        "$\\Delta n~[n_\mathrm{sw}]$",
        "$\\Delta |v|~[v_\mathrm{sw}]$",
        "$\\Delta P_\mathrm{dyn}~[P_\mathrm{dyn,sw}]$",
        "$\\Delta |B|~[B_\mathrm{IMF}]$",
        "$\\Delta T~[T_\mathrm{sw}]$",
        "$Lifetime~[\mathrm{s}]$",
        "$Tangential$\n$size~[R_\mathrm{E}]$",
        "Size Ratio",
        "Leaves BS",
        "Dies\nat BS",
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
        np.linspace(0, 1, 2 + 1),
        np.linspace(0, 1, 2 + 1),
    ]
    pos_list = [
        "left",
        "left",
        "left",
        "left",
        "left",
        "right",
        "right",
        "right",
        "right",
        "right",
    ]
    fcs_jet_props = [[], [], [], [], [], [], [], [], [], []]
    non_jet_props = [[], [], [], [], [], [], [], [], [], []]

    # Loop over runs
    for runid in run_arr:

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
            for n1 in range(len(vars_list)):

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
            for n1 in range(len(vars_list)):

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
    fig, ax_list = plt.subplots(5, 2, figsize=(7, 13))

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

    ax_flat[6].legend(frameon=False, markerscale=0.5)
    ax_flat[0].set_title(run_id, fontsize=20)

    # Save figure
    plt.tight_layout()

    # fig.savefig(
    #    wrkdir_DNR
    #    + "papu22/Figures/FCS_non_hist_lastbs_{}_{}.pdf".format(lastbs, run_id)
    # )
    fig.savefig(
        wrkdir_DNR
        + "papu22/Figures/FCS_non_hist_lastbs_{}_{}.png".format(lastbs, run_id),
        dpi=300,
    )
    plt.close(fig)


def papu22_mov_script(runid):

    sj_jet_ids, jet_ids, slams_ids = jh20.separate_jets_god(runid, False)
    non_sj_ids = jet_ids[np.in1d(jet_ids, sj_jet_ids) == False]

    runid_list = ["ABA", "ABC", "AEA", "AEC"]
    stop_list = [839, 1179, 1339, 879]

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
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runid_list.index(runid)]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw
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
            "Pdyn",
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
    Pdyn = pass_maps["Pdyn"]

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

    ch_mask = (core_heating > 3 * T_sw).astype(int)
    mach_mask = (mmsx < 1).astype(int)
    rho_mask = (rho > 2 * rho_sw).astype(int)

    cav_shfa_mask = (Bmag < 0.8 * B_sw).astype(int)
    cav_shfa_mask[rho >= 0.8 * rho_sw] = 0

    diamag_mask = (Pdyn >= 1.2 * Pdyn_sw).astype(int)
    diamag_mask[Bmag > B_sw] = 0

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

    # cav_shfa_cont = ax.contour(
    #     XmeshXY,
    #     YmeshXY,
    #     cav_shfa_mask,
    #     [0.5],
    #     linewidths=0.6,
    #     colors=CB_color_cycle[5],
    #     linestyles=["solid"],
    # )

    diamag_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        diamag_mask,
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
        markersize=4,
        markeredgecolor="white",
        fillstyle="full",
        mew=0.4,
        label="Non-FCS-jet",
    )
    (sj_pos,) = ax.plot(
        sj_xlist,
        sj_ylist,
        "o",
        color="red",
        markersize=4,
        markeredgecolor="white",
        fillstyle="full",
        mew=0.4,
        label="FCS-jet",
    )

    # print(jet_cont.collections[0])

    # jet_cont.collections[0].set_label("Jet")
    # ch_cont.collections[0].set_label("BS CH")
    # slams_cont.collections[0].set_label("FCS")
    # rho_cont.collections[0].set_label("BS rho")
    # mach_cont.collections[0].set_label("BS Mmsx")
    # cav_shfa_cont.collections[0].set_label("Cav/SHFA")

    # jet_line = Line2D([0], [0], linestyle="none", color=CB_color_cycle[0])
    # ch_line = Line2D([0], [0], linestyle="none", color=CB_color_cycle[1])
    # slams_line = Line2D([0], [0], linestyle="none", color=CB_color_cycle[2])
    # rho_line = Line2D([0], [0], linestyle="none", color=CB_color_cycle[3])
    # mach_line = Line2D([0], [0], linestyle="none", color=CB_color_cycle[4])
    # cav_shfa_line = Line2D([0], [0], linestyle="none", color=CB_color_cycle[5])

    proxy = [plt.Rectangle((0, 0), 1, 1, fc=CB_color_cycle[itr]) for itr in range(6)]

    ax.legend(
        # (jet_line, ch_line, slams_line, rho_line, mach_line, cav_shfa_line),
        proxy,
        # ("Jet", "BS CH", "FCS", "BS rho", "BS Mmsx", "Cav/SHFA"),
        ("Jet", "BS CH", "FCS", "BS rho", "BS Mmsx", "Diamag"),
        frameon=True,
        numpoints=1,
        markerscale=1,
        loc="upper right",
        fontsize=5,
    )
