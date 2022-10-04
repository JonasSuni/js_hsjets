from operator import ge
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
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

import plot_contours as pc
import jet_analyser as ja
import jet_io as jio
import jet_jh2020 as jh20

# mpl.rc("text", usetex=True)
# params = {"text.latex.preamble": [r"\usepackage{amsmath}"]}
# plt.rcParams.update(params)

r_e = 6.371e6
m_p = 1.672621898e-27
mu0 = 1.25663706212e-06
kb = 1.380649e-23

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

    fig, ax_list = plt.subplots(2, 2, figsize=(9, 12), sharex=True, sharey=True)
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

    annot = ["a)", "b)", "c)", "d)"]

    for idx, ax in enumerate(ax_flat):
        print(Xun_arr[idx].shape)
        print(Yun_arr[idx].shape)
        Xun_minus_bs = np.array(
            [
                Xun_arr[idx] - np.polyval(bs_fit[idx], Yun_arr[idx][i2])
                for i2 in range(len(Yun_arr[idx]))
            ]
        )
        print(Xun_minus_bs.shape)

        cont = ax.contour(
            # Xun_arr[idx] - bs_fit[idx][-1],
            Xun_arr[idx],
            Yun_arr[idx],
            np.ma.masked_where(Xun_minus_bs < 0, Bz_arr[idx]),
            [-0.5e-9, 0.5e-9],
            colors=[CB_color_cycle[4], CB_color_cycle[5]],
            linewidths=[0.6, 0.6],
            zorder=1,
        )
        for c in cont.collections:
            c.set_rasterized(True)
        cont = ax.contour(
            # Xun_arr[idx] - bs_fit[idx][-1],
            Xun_arr[idx],
            Yun_arr[idx],
            np.ma.masked_where(Xun_minus_bs < 0, np.abs(RhoBS_arr[idx])),
            [1],
            colors=["black"],
            linewidths=[0.6],
            linestyles=["dashed"],
            zorder=1,
        )
        for c in cont.collections:
            c.set_rasterized(True)
        # ax.plot(bs_x[idx], yarr, color="black")
        ax.annotate(annot[idx], (0.05, 0.90), xycoords="axes fraction", fontsize=20)

    for n1, runid in enumerate(runids):
        ax = ax_flat[n1]
        for n2, kind in enumerate(kinds):
            label_bool = draw_labels[n1]
            non_ids = np.loadtxt(
                wrkdir_DNR + "papu22/id_txts/2D/{}_{}.txt".format(runid, kind),
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
                # bs_x_y0 = np.polyval(jx.bs_mp_fit(runid, int(t0 * 2))[1], y0)
                if label_bool:
                    ax.plot(
                        # np.polyval(bs_fit[n1], y0) - bs_fit[n1][-1] + (x0 - bs_x_y0),
                        x0,
                        y0,
                        "x",
                        color=CB_color_cycle[n2],
                        label=kinds[n2].capitalize(),
                        rasterized=True,
                        zorder=2,
                    )
                    label_bool = False
                else:
                    ax.plot(
                        # np.polyval(bs_fit[n1], y0) - bs_fit[n1][-1] + (x0 - bs_x_y0),
                        x0,
                        y0,
                        "x",
                        color=CB_color_cycle[n2],
                        rasterized=True,
                        zorder=2,
                    )
        label_bool = draw_labels[n1]
        fcs_jet_ids = get_fcs_jets(runid)
        for sj_id in fcs_jet_ids:
            props = jio.PropReader(str(sj_id).zfill(5), runid, transient="jet")
            x0, y0, t0 = (
                props.read("x_mean")[0],
                props.read("y_mean")[0],
                props.read("time")[0],
            )
            # bs_x_y0 = np.polyval(jx.bs_mp_fit(runid, int(t0 * 2))[1], y0)
            if label_bool:
                ax.plot(
                    # np.polyval(bs_fit[n1], y0) - bs_fit[n1][-1] + (x0 - bs_x_y0),
                    x0,
                    y0,
                    "x",
                    color="gray",
                    label="FCS-jets",
                    rasterized=True,
                    zorder=0,
                    alpha=0.3,
                )
                label_bool = False
            else:
                ax.plot(
                    # np.polyval(bs_fit[n1], y0) - bs_fit[n1][-1] + (x0 - bs_x_y0),
                    x0,
                    y0,
                    "x",
                    color="gray",
                    rasterized=True,
                    zorder=0,
                    alpha=0.3,
                )
        label_bool = draw_labels[n1]
        ax.grid()
        # ax.set_xlim(-3, 2)
        ax.set_xlim(6, 18)
        # ax.set_aspect("equal")
        ax.tick_params(labelsize=16)
        if runid in ["ABA", "AEA"]:
            ax.set_ylim(-10, 10)
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.set_ylim(-10, 10)
            ax.set_aspect("equal", adjustable="box")
        if label_bool:
            ax.legend(fontsize=12, loc="center left")
    ax_flat[0].set_ylabel("$B_\mathrm{IMF}=10$ nT\n\n$Y~[R_\mathrm{E}]$", fontsize=20)
    ax_flat[2].set_ylabel("$B_\mathrm{IMF}=5$ nT\n\n$Y~[R_\mathrm{E}]$", fontsize=20)
    # ax_flat[2].set_xlabel(
    #     "$X-X_\mathrm{nose}~[R_\mathrm{E}]$\n\n$\\theta_\mathrm{cone}=5^\circ$",
    #     fontsize=20,
    # )
    # ax_flat[3].set_xlabel(
    #     "$X-X_\mathrm{nose}~[R_\mathrm{E}]$\n\n$\\theta_\mathrm{cone}=30^\circ$",
    #     fontsize=20,
    # )
    ax_flat[2].set_xlabel(
        "$X~[R_\mathrm{E}]$\n\n$\\theta_\mathrm{cone}=5^\circ$", fontsize=20,
    )
    ax_flat[3].set_xlabel(
        "$X~[R_\mathrm{E}]$\n\n$\\theta_\mathrm{cone}=30^\circ$", fontsize=20,
    )

    # Save figure
    plt.tight_layout()

    fig.savefig(wrkdir_DNR + "papu22/Figures/BS_plot.pdf", dpi=300)
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
            wrkdir_DNR + "papu22/id_txts/2D/{}_{}.txt".format(runid, "foreshock"),
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
        "$v_x$ [$v_\mathrm{sw}$]",
        "$P_\mathrm{dyn}$ [$P_\mathrm{dyn,sw}$]",
        "$B$ [$B_\mathrm{IMF}$]",
        "$T$ [$T_\mathrm{sw}$]",
        # "$M_{\mathrm{MS},x}$",
    ]

    data_arr = [rho_avg, v_avg, pdyn_avg, B_avg, T_avg, mmsx_avg]

    sj_data_arr = [sj_rho_avg, sj_v_avg, sj_pdyn_avg, sj_B_avg, sj_T_avg, sj_mmsx_avg]

    fig, ax_list = plt.subplots(
        2, len(varname_list), figsize=(24, 10), sharex=True, sharey=True
    )
    im_list = []
    cb_list = []
    sj_im_list = []
    sj_cb_list = []
    # fig.suptitle(
    #     "Run: {}, Type: {}, Nnon = {}, Nfcs = {}".format(
    #         run_id, "foreshock vs. FCS-jet", type_count, sj_count
    #     ),
    #     fontsize=20,
    # )

    vmin = [
        np.min(data_arr[0]),
        -1,
        np.min(data_arr[2]),
        np.min(data_arr[3]),
        np.min(data_arr[4]),
    ]
    vmax = [
        np.max(data_arr[0]),
        0,
        np.max(data_arr[2]),
        np.max(data_arr[3]),
        np.max(data_arr[4]),
    ]
    cmap = ["batlow", "Blues_r", "batlow", "batlow", "batlow"]
    annot = ["a)", "b)", "c)", "d)", "e)"]
    annot_sj = ["f)", "g)", "h)", "i)", "j)"]

    for idx, ax in enumerate(ax_list[0]):
        ax.tick_params(labelsize=20)
        im_list.append(
            ax.pcolormesh(
                x_range,
                t_range,
                data_arr[idx],
                shading="nearest",
                cmap=cmap[idx],
                # vmin=vmin_norm[idx],
                # vmax=vmax_norm[idx],
                vmin=vmin[idx],
                vmax=vmax[idx],
                rasterized=True,
            )
        )
        # if idx == 1:
        #     ax.contourf(x_range, t_range, data_arr[idx], [0, 10], colors="red")
        if idx == 1:
            cb_list.append(fig.colorbar(im_list[idx], ax=ax, extend="max"))
            cb_list[idx].cmap.set_over("red")
        else:
            cb_list.append(fig.colorbar(im_list[idx], ax=ax))
        cb_list[idx].ax.tick_params(labelsize=20)
        ax.contour(XmeshXY, YmeshXY, rho_avg, [2], colors=["black"])
        ax.contour(XmeshXY, YmeshXY, Tcore_avg, [3], colors=[CB_color_cycle[1]])
        ax.contour(XmeshXY, YmeshXY, mmsx_avg, [1.0], colors=[CB_color_cycle[4]])
        ax.set_title(varname_list[idx], fontsize=24, pad=10)
        ax.set_xlim(x_range[0], x_range[-1])
        ax.set_ylim(t_range[0], t_range[-1])
        # ax.set_xlabel("Epoch $x$ [$R_\mathrm{E}$]", fontsize=20, labelpad=10)
        ax.axhline(t0, linestyle="dashed", linewidth=0.6)
        ax.axvline(x0, linestyle="dashed", linewidth=0.6)
        ax.annotate(annot[idx], (0.05, 0.90), xycoords="axes fraction", fontsize=24)
    ax_list[0][0].set_ylabel(
        "Foreshock jets\n\nEpoch time [s]", fontsize=28, labelpad=10
    )

    for idx, ax in enumerate(ax_list[1]):
        ax.tick_params(labelsize=20)
        sj_im_list.append(
            ax.pcolormesh(
                x_range,
                t_range,
                sj_data_arr[idx],
                shading="nearest",
                cmap=cmap[idx],
                # vmin=vmin_norm[idx],
                # vmax=vmax_norm[idx],
                vmin=vmin[idx],
                vmax=vmax[idx],
                rasterized=True,
            )
        )
        # if idx == 1:
        #     ax.contourf(x_range, t_range, sj_data_arr[idx], [0, 10], colors="red")
        if idx == 1:
            sj_cb_list.append(fig.colorbar(sj_im_list[idx], ax=ax, extend="max"))
            sj_cb_list[idx].cmap.set_over("red")
        else:
            sj_cb_list.append(fig.colorbar(sj_im_list[idx], ax=ax))
        sj_cb_list[idx].ax.tick_params(labelsize=20)
        ax.contour(XmeshXY, YmeshXY, sj_rho_avg, [2], colors=["black"])
        ax.contour(XmeshXY, YmeshXY, sj_Tcore_avg, [3], colors=[CB_color_cycle[1]])
        ax.contour(XmeshXY, YmeshXY, sj_mmsx_avg, [1.0], colors=[CB_color_cycle[4]])
        # ax.set_title(varname_list[idx], fontsize=20, pad=10)
        ax.set_xlim(x_range[0], x_range[-1])
        ax.set_ylim(t_range[0], t_range[-1])
        ax.set_xlabel("Epoch $x$ [$R_\mathrm{E}$]", fontsize=24, labelpad=10)
        ax.axhline(t0, linestyle="dashed", linewidth=0.6)
        ax.axvline(x0, linestyle="dashed", linewidth=0.6)
        ax.annotate(annot_sj[idx], (0.05, 0.90), xycoords="axes fraction", fontsize=24)
    ax_list[1][0].set_ylabel("FCS-jets\n\nEpoch time [s]", fontsize=28, labelpad=10)

    # Save figure
    plt.tight_layout()

    fig.savefig(
        wrkdir_DNR
        + "papu22/Figures/jmap_SEA_foreshock_comparison_{}.pdf".format(run_id),
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
                wrkdir_DNR + "papu22/id_txts/2D/{}_{}.txt".format(runid, kind),
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
        "$v_x$ [$v_\mathrm{sw}$]",
        "$P_\mathrm{dyn}$ [$P_\mathrm{dyn,sw}$]",
        "$B$ [$B_\mathrm{IMF}$]",
        "$T$ [$T_\mathrm{sw}$]",
        # "$M_{\mathrm{MS},x}$",
    ]

    cmap = ["batlow", "vik", "batlow", "batlow", "batlow"]
    vmin = [None, -1, None, None, None]
    vmax = [None, 1, None, None, None]
    annot = ["a)", "b)", "c)", "d)", "e)"]

    data_arr = [rho_avg, v_avg, pdyn_avg, B_avg, T_avg, mmsx_avg]

    fig, ax_list = plt.subplots(
        2,
        int(np.ceil(len(varname_list) / 2.0)),
        figsize=(20, 10),
        sharex=True,
        sharey=True,
    )
    ax_list = ax_list.flatten()
    im_list = []
    cb_list = []
    # fig.suptitle(
    #     "Run: {}, Type: {}, N = {}".format(run_id, kind, type_count), fontsize=20,
    # )
    for idx, ax in enumerate(ax_list[: len(varname_list)]):
        ax.tick_params(labelsize=20)
        im_list.append(
            ax.pcolormesh(
                x_range,
                t_range,
                data_arr[idx],
                shading="nearest",
                cmap=cmap[idx],
                vmin=vmin[idx],
                vmax=vmax[idx],
                rasterized=True,
            )
        )
        # ax.set_aspect(0.1, adjustable="box")
        cb_list.append(fig.colorbar(im_list[idx], ax=ax))
        cb_list[idx].ax.tick_params(labelsize=20)
        # cb_list[idx].ax.set_aspect(0.1, adjustable="box")
        ax.contour(XmeshXY, YmeshXY, rho_avg, [2], colors=["black"])
        ax.contour(XmeshXY, YmeshXY, Tcore_avg, [3], colors=[CB_color_cycle[1]])
        ax.contour(XmeshXY, YmeshXY, mmsx_avg, [1.0], colors=[CB_color_cycle[4]])
        ax.set_title(varname_list[idx], fontsize=24, pad=10)
        ax.set_xlim(x_range[0], x_range[-1])
        ax.set_ylim(t_range[0], t_range[-1])
        ax.set_xlabel("Epoch $x$ [$R_\mathrm{E}$]", fontsize=24, labelpad=10)
        ax.axhline(t0, linestyle="dashed", linewidth=0.6)
        ax.axvline(x0, linestyle="dashed", linewidth=0.6)
        ax.annotate(annot[idx], (0.05, 0.90), xycoords="axes fraction", fontsize=24)
    ax_list[0].set_ylabel("Epoch time [s]", fontsize=28, labelpad=10)
    ax_list[int(np.ceil(len(varname_list) / 2.0))].set_ylabel(
        "Epoch time [s]", fontsize=28, labelpad=10
    )
    ax_list[-1].set_axis_off()
    fig.suptitle("{} jets".format(kind.capitalize()), fontsize=28)

    # Save figure
    plt.tight_layout()

    fig.savefig(
        wrkdir_DNR + "papu22/Figures/jmap_SEA_{}_{}.pdf".format(run_id, kind), dpi=300
    )
    plt.close(fig)


def types_P_jplot_SEA(run_id, kind="beam", version="new", shfa=False):

    if run_id == "all":
        runid_list = ["ABA", "ABC", "AEA", "AEC"]
    else:
        runid_list = [run_id]

    if shfa:
        vmin_norm = [None, None, None]
        vmax_norm = [None, None, None]
    else:
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
                Tcore_avg = Tcore_avg + Tcore
                mmsx_avg = mmsx_avg + mmsx

                if shfa:
                    pth_avg = pth_avg + rho
                    pmag_avg = pmag_avg + np.sqrt(2 * mu0 * pmag)
                    ptot_avg = ptot_avg + pth / (rho * kb)
                else:
                    pth_avg = pth_avg + pth
                    pmag_avg = pmag_avg + pmag
                    ptot_avg = ptot_avg + ptot

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
    if shfa:
        norm = [colors.LogNorm(), colors.LogNorm(), colors.LogNorm()]
        varname_list = [
            "$n$",
            "$B$",
            "$T$",
        ]
    else:
        norm = [
            colors.Normalize(vmin=vmin_norm[idx], vmax=vmax_norm[idx])
            for idx in range(len(vmax_norm))
        ]
        varname_list = [
            "$P_{th}/P_{tot}$",
            "$P_{dyn}/P_{tot}$",
            "$P_{mag}/P_{tot}$",
        ]

    if shfa:
        data_arr = [pth_avg, pmag_avg, ptot_avg]
    else:
        data_arr = [pth_avg / ptot_avg, pdyn_avg / ptot_avg, pmag_avg / ptot_avg]

    fig, ax_list = plt.subplots(
        1, len(varname_list), figsize=(24, 10), sharex=True, sharey=True
    )
    im_list = []
    cb_list = []
    fig.suptitle(
        "Run: {}, Type: {}, N = {}".format(run_id, kind, type_count), fontsize=20,
    )
    for idx, ax in enumerate(ax_list):
        ax.tick_params(labelsize=20)
        im_list.append(
            ax.pcolormesh(
                x_range,
                t_range,
                data_arr[idx],
                shading="nearest",
                cmap="batlow",
                # vmin=vmin_norm[idx],
                # vmax=vmax_norm[idx],
                norm=norm[idx],
                rasterized=True,
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
    if shfa:
        fig.savefig(
            wrkdir_DNR + "papu22/Figures/shfa_jmap_SEA_{}_{}.png".format(run_id, kind),
            dpi=300,
        )
    else:
        fig.savefig(
            wrkdir_DNR + "papu22/Figures/P_jmap_SEA_{}_{}.png".format(run_id, kind),
            dpi=300,
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
            v_arr.append(vlsvobj.read_variable("v", operator="x", cellids=cell_range))
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


def non_jet_jplots(runid, txt=False):

    CB_color_cycle = jx.CB_color_cycle

    dx = 227e3 / r_e
    varname_list = [
        "$n$ [$n_\mathrm{sw}$]",
        "$v_x$ [$v_\mathrm{sw}$]",
        "$P_\mathrm{dyn}$ [$P_\mathrm{dyn,sw}$]",
        "$B$ [$B_\mathrm{IMF}$]",
        "$T$ [$T_\mathrm{sw}$]",
    ]

    # Solar wind parameters for the different runs
    # n [m^-3], v [m/s], B [T], T [K]
    runid_list = ["ABA", "ABC", "AEA", "AEC"]
    runids_paper = ["HM30", "HM05", "LM30", "LM05"]
    sw_pars = [
        [1.0, 750.0, 5.0, 0.5],
        [3.3, 600.0, 5.0, 0.5],
        [1.0, 750.0, 10.0, 0.5],
        [3.3, 600.0, 10.0, 0.5],
    ]
    n_sw, v_sw, B_sw, T_sw = sw_pars[runid_list.index(runid)]
    vmin_norm = [1.0 / 2, -2.0, 1.0 / 6, 1.0 / 2, 1.0]
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

        if txt:
            rho_arr, v_arr, pdyn_arr, B_arr, T_arr, Tcore_arr, mmsx_arr = np.load(
                wrkdir_DNR
                + "papu22/jmap_txts/{}/{}_{}.npy".format(
                    runid, runid, str(non_id).zfill(5)
                )
            )
        else:
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
                    vlsvobj.read_variable("v", operator="x", cellids=cell_range)
                )
                pdyn_arr.append(vlsvobj.read_variable("Pdyn", cellids=cell_range))
                B_arr.append(
                    vlsvobj.read_variable("B", operator="magnitude", cellids=cell_range)
                )
                T_arr.append(vlsvobj.read_variable("Temperature", cellids=cell_range))
                Tcore_arr.append(
                    vlsvobj.read_variable("core_heating", cellids=cell_range)
                )
                mmsx_arr.append(vlsvobj.read_variable("Mmsx", cellids=cell_range))

            rho_arr = np.array(rho_arr) / 1.0e6 / n_sw
            v_arr = np.array(v_arr) / 1.0e3 / v_sw
            pdyn_arr = (
                np.array(pdyn_arr)
                / 1.0e-9
                / (m_p * n_sw * 1e6 * v_sw * v_sw * 1e6 / 1e-9)
            )
            B_arr = np.array(B_arr) / 1.0e-9 / B_sw
            T_arr = np.array(T_arr) / 1.0e6 / T_sw
            Tcore_arr = np.array(Tcore_arr) / 1.0e6 / T_sw
            mmsx_arr = np.array(mmsx_arr)

        data_arr = [rho_arr, v_arr, pdyn_arr, B_arr, T_arr]
        cmap = ["batlow", "vik", "batlow", "batlow", "batlow"]
        annot = ["a)", "b)", "c)", "d)", "e)"]

        # fig, ax_list = plt.subplots(
        #     1, len(varname_list), figsize=(20, 5), sharex=True, sharey=True
        # )
        fig, ax_list = plt.subplots(
            2,
            int(np.ceil(len(varname_list) / 2.0)),
            figsize=(20, 10),
            sharex=True,
            sharey=True,
        )
        ax_list = ax_list.flatten()
        im_list = []
        cb_list = []
        fig.suptitle(
            "Run: {}, JetID: {}, $y$ = {:.3f} ".format(
                runids_paper[runid_list.index(runid)], non_id, y0
            )
            + "$R_\mathrm{E}$",
            fontsize=28,
        )
        for idx in range(len(varname_list)):
            ax = ax_list[idx]
            ax.tick_params(labelsize=20)
            im_list.append(
                ax.pcolormesh(
                    x_range,
                    t_range,
                    data_arr[idx],
                    shading="nearest",
                    cmap=cmap[idx],
                    vmin=vmin_norm[idx],
                    vmax=vmax_norm[idx],
                    rasterized=True,
                )
            )
            cb_list.append(fig.colorbar(im_list[idx], ax=ax))
            cb_list[idx].ax.tick_params(labelsize=20)
            ax.contour(XmeshXY, YmeshXY, rho_arr, [2], colors=["black"])
            ax.contour(XmeshXY, YmeshXY, Tcore_arr, [3], colors=[CB_color_cycle[1]])
            ax.contour(XmeshXY, YmeshXY, mmsx_arr, [1.0], colors=[CB_color_cycle[4]])
            ax.set_title(varname_list[idx], fontsize=24, pad=10)
            ax.set_xlim(x_range[0], x_range[-1])
            ax.set_ylim(t_range[0], t_range[-1])
            ax.set_xlabel("$x$ [$R_\mathrm{E}$]", fontsize=24, labelpad=10)
            ax.axhline(t0, linestyle="dashed", linewidth=0.6)
            ax.axvline(x0, linestyle="dashed", linewidth=0.6)
            ax.annotate(annot[idx], (0.05, 0.90), xycoords="axes fraction", fontsize=24)
        ax_list[0].set_ylabel("Simulation time [s]", fontsize=28, labelpad=10)
        ax_list[int(np.ceil(len(varname_list) / 2.0))].set_ylabel(
            "Epoch time [s]", fontsize=28, labelpad=10
        )
        ax_list[-1].set_axis_off()

        # Save figure
        plt.tight_layout()

        # fig.savefig(
        #     wrkdir_DNR
        #     + "papu22/Figures/jmaps/{}_{}.pdf".format(runid, str(non_id).zfill(5))
        # )
        fig.savefig(
            wrkdir_DNR
            + "papu22/Figures/jmaps/{}_{}.pdf".format(runid, str(non_id).zfill(5)),
            dpi=300,
        )
        plt.close(fig)

        if not txt:
            np.save(
                wrkdir_DNR
                + "papu22/jmap_txts/{}/{}_{}".format(
                    runid, runid, str(non_id).zfill(5)
                ),
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
            ax.tick_params(labelsize=20)
            im_list.append(
                ax.pcolormesh(
                    x_range,
                    t_range,
                    data_arr[idx],
                    shading="nearest",
                    cmap="Greys",
                    vmin=vmin_norm[idx],
                    vmax=vmax_norm[idx],
                    rasterized=True,
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


def jet_avg_std(kind, version="2D"):

    runids = ["ABA", "ABC", "AEA", "AEC"]

    rho_sw = [1e6, 3.3e6, 1e6, 3.3e6]
    v_sw = [750e3, 600e3, 750e3, 600e3]
    pdyn_sw = [m_p * rho_sw[idx] * v_sw[idx] * v_sw[idx] for idx in range(len(runids))]

    data_arr = np.empty((3, 6000), dtype=float)
    data_arr.fill(np.nan)

    counter = 0
    for runid in runids:
        if kind == "fcs":
            jet_ids = get_fcs_jets(runid)
        else:
            jet_ids = np.loadtxt(
                wrkdir_DNR + "papu22/id_txts/{}/{}_{}.txt".format(version, runid, kind),
                dtype=int,
                ndmin=1,
            )
        for jet_id in jet_ids:
            props = jio.PropReader(str(jet_id).zfill(5), runid, transient="jet")
            duration = props.read("duration")[0]
            pendep = props.read("x_mean")[-1] - props.read("bs_distance")[-1]
            pd_max = np.max(props.read("pd_max")) / (pdyn_sw[runids.index(runid)] * 1e9)
            data_arr[0][counter] = duration
            data_arr[1][counter] = pendep
            data_arr[2][counter] = pd_max
            counter += 1

    print("\n")
    # print("RUN: {}".format(runid))
    print("KIND: {}".format(kind.capitalize()))
    print("N = {}".format(counter))
    print(
        "Duration = {:.3f} +- {:.3f} s".format(
            np.nanmean(data_arr[0]), np.nanstd(data_arr[0], ddof=1)
        )
    )
    print(
        "Penetration depth = {:.3f} +- {:.3f} RE".format(
            np.nanmean(data_arr[1]), np.nanstd(data_arr[1], ddof=1)
        )
    )
    print(
        "Max Pdyn = {:.3f} +- {:.3f} Pdyn_sw".format(
            np.nanmean(data_arr[2]), np.nanstd(data_arr[2], ddof=1)
        )
    )
    print("\n")

    iqr_dur = np.subtract.reduce(np.nanpercentile(data_arr[0], [75, 25]))
    iqr_pen = np.subtract.reduce(np.nanpercentile(data_arr[1], [75, 25]))
    iqr_pd = np.subtract.reduce(np.nanpercentile(data_arr[2], [75, 25]))

    bins_dur = (np.nanmax(data_arr[0]) - np.nanmin(data_arr[0])) / (
        2 * iqr_dur / float(counter) ** (1.0 / 3)
    )
    bins_pen = (np.nanmax(data_arr[1]) - np.nanmin(data_arr[1])) / (
        2 * iqr_pen / float(counter) ** (1.0 / 3)
    )
    bins_pd = (np.nanmax(data_arr[2]) - np.nanmin(data_arr[2])) / (
        2 * iqr_pd / float(counter) ** (1.0 / 3)
    )

    return (
        np.histogram(data_arr[0][:counter], bins=int(bins_dur)),
        np.histogram(data_arr[1][:counter], bins=int(bins_pen)),
        np.histogram(data_arr[0][:counter], bins=int(bins_pd)),
    )


def kind_timeseries(runid, kind):

    bulkpath = jx.find_bulkpath(runid)

    runids = ["ABA", "ABC", "AEA", "AEC"]
    non_ids = np.loadtxt(
        wrkdir_DNR + "papu22/id_txts/2D/{}_{}.txt".format(runid, kind),
        dtype=int,
        ndmin=1,
    )
    vars = [
        "rho",
        "v",
        "v",
        "v",
        "v",
        "Pdyn",
        "B",
        "B",
        "B",
        "B",
        "TParallel",
        "TPerpendicular",
    ]
    plot_labels = [
        None,
        "$v_x$",
        "$v_y$",
        "$v_z$",
        "$|v|$",
        None,
        "$B_x$",
        "$B_y$",
        "$B_z$",
        "$|B|$",
        "TPar",
        "TPerp",
    ]
    scales = [1e-6, 1e-3, 1e-3, 1e-3, 1e-3, 1e9, 1e9, 1e9, 1e9, 1e9, 1e-6, 1e-6]
    draw_legend = [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
    ]
    ylabels = [
        "$\\rho~[\mathrm{cm}^{-3}]$",
        "$v~[\mathrm{km/s}]$",
        "$P_\mathrm{dyn}~[\mathrm{nPa}]$",
        "$B~[\mathrm{nT}]$",
        "$T~[\mathrm{MK}]$",
    ]
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
        "pass",
        "pass",
    ]
    plot_index = [0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 4]
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
    ]

    for non_id in non_ids:
        print("Jet {} of kind {} in run {}".format(non_id, kind, runid))
        props = jio.PropReader(str(non_id).zfill(5), runid, transient="jet")
        x0, y0 = (props.read("x_mean")[0], props.read("y_mean")[0])
        t0 = props.read("time")[0]
        t_arr = np.arange(t0 - 10.0, t0 + 10.1, 0.5)
        fnr0 = int(t0 * 2)
        fnr_arr = np.arange(fnr0 - 20, fnr0 + 21)
        cellid = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
        ).get_cellid([x0 * r_e, y0 * r_e, 0 * r_e])
        data_arr = np.zeros((len(vars), fnr_arr.size), dtype=float)

        for idx, fnr in enumerate(fnr_arr):
            try:
                vlsvobj = pt.vlsvfile.VlsvReader(
                    bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
                )
                for idx2, var in enumerate(vars):
                    data_arr[idx2, idx] = (
                        vlsvobj.read_variable(var, cellids=cellid, operator=ops[idx2])
                        * scales[idx2]
                    )
            except:
                data_arr[:, idx] = np.nan

        fig, ax_list = plt.subplots(len(ylabels), 1, sharex=True, figsize=(6, 8))
        ax_list[0].set_title(
            "Run: {}, Jet: {}, Kind: {}".format(runid, non_id, kind.capitalize())
        )
        for idx in range(len(vars)):
            ax = ax_list[plot_index[idx]]
            ax.plot(
                t_arr, data_arr[idx], color=plot_colors[idx], label=plot_labels[idx]
            )
            ax.set_xlim(t_arr[0], t_arr[-1])
            if draw_legend[idx]:
                ax.legend()
        ax_list[-1].set_xlabel("Simulation time [s]")
        for idx, ax in enumerate(ax_list):
            ax.grid()
            ax.set_ylabel(ylabels[idx])
            ax.axvline(t0, linestyle="dashed")
        plt.tight_layout()
        fig.savefig(
            wrkdir_DNR
            + "papu22/Figures/timeseries/{}/{}/{}.png".format(
                runid, kind, str(non_id).zfill(5)
            ),
            dpi=300,
        )
        plt.close(fig)


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
            t_arr, beam_avg[n2], color=jx.CB_color_cycle[0], label="Beam", zorder=2,
        )
        ax_list[n2].plot(
            t_arr, stripe_avg[n2], color=jx.CB_color_cycle[1], label="Stripe", zorder=2,
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
    types_list = ["foreshock", "beam", "complex", "stripe"]
    all_props = [[[]] * 8] * 4

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
        fsaved="black",
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
        "/wrk-vakka/users/jesuni/working/SLAMS/Masks/{}/{}.mask".format(
            runid_g, int(filenr_g)
        )
    ).astype(int)
    jet_cells = np.loadtxt(
        "/wrk-vakka/users/jesuni/working/jets/Masks/{}/{}.mask".format(
            runid_g, int(filenr_g)
        )
    ).astype(int)

    sj_jetobs = [
        jio.PropReader(str(int(sj_id)).zfill(5), runid_g, transient="jet")
        for sj_id in sj_ids_g
    ]
    non_sjobs = [
        jio.PropReader(str(int(non_id)).zfill(5), runid_g, transient="jet")
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

    # diamag_cont = ax.contour(
    #     XmeshXY,
    #     YmeshXY,
    #     diamag_mask,
    #     [0.5],
    #     linewidths=0.6,
    #     colors=CB_color_cycle[5],
    #     linestyles=["solid"],
    # )

    # (non_pos,) = ax.plot(
    #     non_xlist,
    #     non_ylist,
    #     "o",
    #     color="black",
    #     markersize=4,
    #     markeredgecolor="white",
    #     fillstyle="full",
    #     mew=0.4,
    #     label="Non-FCS-jet",
    # )
    # (sj_pos,) = ax.plot(
    #     sj_xlist,
    #     sj_ylist,
    #     "o",
    #     color="red",
    #     markersize=4,
    #     markeredgecolor="white",
    #     fillstyle="full",
    #     mew=0.4,
    #     label="FCS-jet",
    # )

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

    # proxy = [plt.Rectangle((0, 0), 1, 1, fc=CB_color_cycle[itr]) for itr in range(6)]

    # ax.legend(
    #     # (jet_line, ch_line, slams_line, rho_line, mach_line, cav_shfa_line),
    #     proxy,
    #     # ("Jet", "BS CH", "FCS", "BS rho", "BS Mmsx", "Cav/SHFA"),
    #     ("Jet", "BS CH", "FCS", "BS rho", "BS Mmsx", "Diamag"),
    #     frameon=True,
    #     numpoints=1,
    #     markerscale=1,
    #     loc="upper right",
    #     fontsize=5,
    # )

    proxy = [plt.Rectangle((0, 0), 1, 1, fc=CB_color_cycle[itr]) for itr in range(5)]

    ax.legend(
        # (jet_line, ch_line, slams_line, rho_line, mach_line, cav_shfa_line),
        proxy,
        # ("Jet", "BS CH", "FCS", "BS rho", "BS Mmsx", "Cav/SHFA"),
        ("Jet", "BS CH", "FCS", "BS rho", "BS Mmsx"),
        frameon=True,
        numpoints=1,
        markerscale=1,
        loc="upper right",
        fontsize=5,
    )


def vdf_plotter(runid, cellid, t0, zoom=1):

    # make outputdir if it doesn't already exist
    if not os.path.exists(wrkdir_DNR + "papu22/VDFs/{}/{}/".format(runid, cellid)):
        try:
            os.makedirs(wrkdir_DNR + "papu22/VDFs/{}/{}/".format(runid, cellid))
        except OSError:
            pass

    runids = ["ABA", "ABC", "AEA", "AEC"]

    global runid_g
    global filenr_g
    global sj_ids_g
    global non_ids_g

    sj_ids_g = []
    non_ids_g = []

    runid_g = runid

    sw_pars = [
        [1e6, 750e3, 5e-9, 0.5e6],
        [3.3e6, 600e3, 5e-9, 0.5e6],
        [1e6, 750e3, 10e-9, 0.5e6],
        [3.3e6, 600e3, 10e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runids.index(runid)]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw

    pdmax = [1.5, 3.5, 1.5, 3.5][runids.index(runid)]
    bulkpath = jx.find_bulkpath(runid)
    obj_580 = pt.vlsvfile.VlsvReader(bulkpath + "bulk.0000580.vlsv")
    cellids = obj_580.read_variable("CellID")
    if obj_580.check_variable("fSaved"):
        fsaved = obj_580.read_variable("fSaved")
    else:
        fsaved = obj_580.read_variable("vg_f_saved")

    vdf_cells = cellids[fsaved == 1]

    if float(cellid) not in vdf_cells:
        print("CellID {} in run {} does not contain VDF data!".format(cellid, runid))
        return 1

    for tc in np.arange(t0 - 10, t0 + 10.01, 0.5):
        fnr = int(tc * 2)
        filenr_g = fnr
        fname = "bulk.{}.vlsv".format(str(fnr).zfill(7))
        x_re, y_re, z_re = obj_580.get_cell_coordinates(cellid) / r_e

        fig, ax_list = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)

        pt.plot.plot_colormap(
            axes=ax_list[0][0],
            filename=bulkpath + fname,
            var="Pdyn",
            vmin=0,
            vmax=pdmax * 1e-9,
            # vscale=1e9,
            cbtitle="$P_\mathrm{dyn}$ [Pa]",
            usesci=1,
            boxre=[
                x_re - 2.0 / zoom,
                x_re + 2.0 / zoom,
                y_re - 2.0 / zoom,
                y_re + 2.0 / zoom,
            ],
            # internalcb=True,
            lin=1,
            colormap="batlow",
            scale=1.3,
            tickinterval=1.0,
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
        )
        ax_list[0][0].axhline(y_re, linestyle="dashed", linewidth=0.6, color="k")
        ax_list[0][0].axvline(x_re, linestyle="dashed", linewidth=0.6, color="k")

        pt.plot.plot_vdf(
            axes=ax_list[0][1],
            filename=bulkpath + fname,
            cellids=[cellid],
            colormap="batlow",
            bvector=1,
            xy=1,
            slicethick=1e9,
            box=[-2e6, 2e6, -2e6, 2e6],
            # internalcb=True,
            setThreshold=1e-15,
            scale=1.3,
        )
        pt.plot.plot_vdf(
            axes=ax_list[1][0],
            filename=bulkpath + fname,
            cellids=[cellid],
            colormap="batlow",
            bvector=1,
            xz=1,
            slicethick=1e9,
            box=[-2e6, 2e6, -2e6, 2e6],
            # internalcb=True,
            setThreshold=1e-15,
            scale=1.3,
        )
        pt.plot.plot_vdf(
            axes=ax_list[1][1],
            filename=bulkpath + fname,
            cellids=[cellid],
            colormap="batlow",
            bvector=1,
            yz=1,
            slicethick=1e9,
            box=[-2e6, 2e6, -2e6, 2e6],
            # internalcb=True,
            setThreshold=1e-15,
            scale=1.3,
        )

        # plt.subplots_adjust(wspace=1, hspace=1)

        fig.suptitle("Run: {}, Cellid: {}, Time: {}s".format(runid, cellid, tc))
        fig.savefig(wrkdir_DNR + "papu22/VDFs/{}/{}/{}.png".format(runid, cellid, fnr))
        plt.close(fig)

    return None


def jet_vdf_plotter(runid):

    runids = ["ABA", "ABC", "AEA", "AEC"]
    pdmax = [1.5, 3.5, 1.5, 3.5][runids.index(runid)]
    bulkpath = jx.find_bulkpath(runid)
    obj_580 = pt.vlsvfile.VlsvReader(bulkpath + "bulk.0000580.vlsv")
    cellids = obj_580.read_variable("CellID")
    if obj_580.check_variable("fSaved"):
        fsaved = obj_580.read_variable("fSaved")
    else:
        fsaved = obj_580.read_variable("vg_f_saved")

    vdf_cells = cellids[fsaved == 1]

    sj_ids, jet_ids, fcs_ids = jh20.separate_jets_god(runid, False)

    for jet_id in jet_ids:
        props = jio.PropReader(str(jet_id).zfill(5), runid)
        jet_times = props.get_times()
        jet_cells = props.get_cells()

        for idx, t in enumerate(jet_times):
            if np.intersect1d(jet_cells[idx], vdf_cells).size == 0:
                continue
            else:
                vdf_cellid = np.intersect1d(jet_cells[idx], vdf_cells)[0]

            for tc in np.arange(t - 10, t + 10.01, 0.5):
                fnr = int(tc * 2)
                fname = "bulk.{}.vlsv".format(str(fnr).zfill(7))
                x_re, y_re, z_re = obj_580.get_cell_coordinates(vdf_cellid) / r_e

                fig, ax_list = plt.subplots(
                    2, 2, figsize=(11, 10), constrained_layout=True
                )

                pt.plot.plot_colormap(
                    axes=ax_list[0][0],
                    filename=bulkpath + fname,
                    var="Pdyn",
                    vmin=0,
                    vmax=pdmax,
                    vscale=1e9,
                    cbtitle="$P_\mathrm{dyn}$ [nPa]",
                    usesci=0,
                    boxre=[x_re - 2, x_re + 2, y_re - 2, y_re + 2],
                    # internalcb=True,
                    lin=1,
                    colormap="batlow",
                    scale=1.3,
                    tickinterval=1.0,
                )
                ax_list[0][0].axhline(
                    y_re, linestyle="dashed", linewidth=0.6, color="k"
                )
                ax_list[0][0].axvline(
                    x_re, linestyle="dashed", linewidth=0.6, color="k"
                )

                pt.plot.plot_vdf(
                    axes=ax_list[0][1],
                    filename=bulkpath + fname,
                    cellids=[vdf_cellid],
                    colormap="batlow",
                    bvector=1,
                    xy=1,
                    slicethick=1e9,
                    box=[-2e6, 2e6, -2e6, 2e6],
                    # internalcb=True,
                    setThreshold=1e-15,
                    scale=1.3,
                )
                pt.plot.plot_vdf(
                    axes=ax_list[1][0],
                    filename=bulkpath + fname,
                    cellids=[vdf_cellid],
                    colormap="batlow",
                    bvector=1,
                    xz=1,
                    slicethick=1e9,
                    box=[-2e6, 2e6, -2e6, 2e6],
                    # internalcb=True,
                    setThreshold=1e-15,
                    scale=1.3,
                )
                pt.plot.plot_vdf(
                    axes=ax_list[1][1],
                    filename=bulkpath + fname,
                    cellids=[vdf_cellid],
                    colormap="batlow",
                    bvector=1,
                    yz=1,
                    slicethick=1e9,
                    box=[-2e6, 2e6, -2e6, 2e6],
                    # internalcb=True,
                    setThreshold=1e-15,
                    scale=1.3,
                )

                # plt.subplots_adjust(wspace=1, hspace=1)

                fig.suptitle("Run: {}, Jet: {}, Time: {}s".format(runid, jet_id, tc))
                fig.savefig(
                    wrkdir_DNR
                    + "papu22/VDFs/{}/jet_vdf_{}_{}.png".format(runid, jet_id, fnr)
                )
                plt.close(fig)
            break

    return None


def kind_animations(runid):

    # sj_ids = get_fcs_jets(runid)
    # for sj_id in sj_ids:
    #     jet_animator(runid, sj_id, "FCS-jet")

    for kind in ["foreshock", "beam", "stripe", "complex"]:
        non_ids = np.loadtxt(
            wrkdir_DNR + "papu22/id_txts/new/{}_{}.txt".format(runid, kind),
            dtype=int,
            ndmin=1,
        )
        for non_id in non_ids:
            jet_animator(runid, non_id, kind)

    return None


def jet_animator(runid, jetid, kind):
    global ax, x0, y0, pdmax, bulkpath, jetid_g
    global runid_g, sj_ids_g, non_ids_g, kind_g
    kind_g = kind
    jetid_g = jetid
    runid_g = runid
    runids = ["ABA", "ABC", "AEA", "AEC"]
    sw_pars = [
        [1e6, 750e3, 5e-9, 0.5e6],
        [3.3e6, 600e3, 5e-9, 0.5e6],
        [1e6, 750e3, 10e-9, 0.5e6],
        [3.3e6, 600e3, 10e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runids.index(runid)]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw
    bulkpath = jx.find_bulkpath(runid)
    pdmax = [1.5, 3.5, 1.5, 3.5][runids.index(runid)]

    sj_ids_g = get_fcs_jets(runid)
    non_ids_g = get_non_jets(runid)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    props = jio.PropReader(str(jetid).zfill(5), runid)
    t0 = props.read("time")[0]
    x0 = props.read("x_mean")[0]
    y0 = props.read("y_mean")[0]

    fnr0 = int(t0 * 2)

    ani = FuncAnimation(
        fig, jet_update, frames=np.arange(fnr0 - 20, fnr0 + 20 + 0.1, 1), blit=False
    )
    ani.save(
        wrkdir_DNR + "papu22/jet_ani/{}/{}/{}.mp4".format(runid, kind, jetid),
        fps=5,
        dpi=150,
        bitrate=1000,
    )
    print("Saved animation of jet {} in run {}".format(jetid, runid))
    plt.close(fig)


def jet_update(fnr):
    ax.clear()
    fname = "bulk.{}.vlsv".format(str(int(fnr)).zfill(7))
    global filenr_g
    filenr_g = fnr
    pt.plot.plot_colormap(
        axes=ax,
        filename=bulkpath + fname,
        var="Pdyn",
        vmin=0,
        vmax=pdmax,
        vscale=1e9,
        cbtitle="$P_\mathrm{dyn}$ [nPa]",
        usesci=0,
        scale=2,
        title="",
        boxre=[x0 - 2, x0 + 2, y0 - 2, y0 + 2],
        # internalcb=True,
        lin=1,
        colormap="batlow",
        tickinterval=1.0,
        external=ext_jet,
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
    )
    ax.set_title(
        "Run: {}, ID: {}, Kind: {}\n t = {}s".format(
            runid_g, jetid_g, kind_g, float(fnr) / 2.0
        ),
        pad=10,
        fontsize=20,
    )
    ax.axhline(y0, linestyle="dashed", linewidth=0.6, color="k")
    ax.axvline(x0, linestyle="dashed", linewidth=0.6, color="k")
    plt.tight_layout()


def ext_jet(ax, XmeshXY, YmeshXY, pass_maps):

    B = pass_maps["B"]
    rho = pass_maps["rho"]
    cellids = pass_maps["CellID"]
    mmsx = pass_maps["Mmsx"]
    core_heating = pass_maps["core_heating"]
    Bmag = np.linalg.norm(B, axis=-1)
    Pdyn = pass_maps["Pdyn"]

    try:
        slams_cells = np.loadtxt(
            "/wrk-vakka/users/jesuni/working/SLAMS/Masks/{}/{}.mask".format(
                runid_g, int(filenr_g)
            )
        ).astype(int)
    except:
        slams_cells = []
    try:
        jet_cells = np.loadtxt(
            "/wrk-vakka/users/jesuni/working/jets/Masks/{}/{}.mask".format(
                runid_g, int(filenr_g)
            )
        ).astype(int)
    except:
        jet_cells = []

    sj_jetobs = [
        jio.PropReader(str(int(sj_id)).zfill(5), runid_g, transient="jet")
        for sj_id in sj_ids_g
    ]
    non_sjobs = [
        jio.PropReader(str(int(non_id)).zfill(5), runid_g, transient="jet")
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
        linewidths=1.0,
        colors=CB_color_cycle[0],
        linestyles=["solid"],
    )

    ch_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        ch_mask,
        [0.5],
        linewidths=1.0,
        colors=CB_color_cycle[1],
        linestyles=["solid"],
    )

    slams_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        slams_mask,
        [0.5],
        linewidths=1.0,
        colors=CB_color_cycle[2],
        linestyles=["solid"],
    )

    rho_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        rho_mask,
        [0.5],
        linewidths=1.0,
        colors=CB_color_cycle[3],
        linestyles=["solid"],
    )

    mach_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        mach_mask,
        [0.5],
        linewidths=1.0,
        colors=CB_color_cycle[4],
        linestyles=["solid"],
    )

    (non_pos,) = ax.plot(
        non_xlist,
        non_ylist,
        "o",
        color="black",
        markersize=10,
        markeredgecolor="white",
        fillstyle="full",
        mew=1,
        label="Non-FCS-jet",
    )
    (sj_pos,) = ax.plot(
        sj_xlist,
        sj_ylist,
        "o",
        color="red",
        markersize=10,
        markeredgecolor="white",
        fillstyle="full",
        mew=1,
        label="FCS-jet",
    )

    proxy = [
        plt.Rectangle((0, 0), 1, 1, fc=CB_color_cycle[itr]) for itr in range(5)
    ] + [non_pos, sj_pos]

    ax.legend(
        proxy,
        ("Jet", "BS CH", "FCS", "BS rho", "BS Mmsx", "Non-FCS jet", "FCS-jet"),
        frameon=True,
        numpoints=1,
        markerscale=1,
        loc="upper right",
        fontsize=16,
    )
