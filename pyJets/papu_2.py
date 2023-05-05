# from operator import ge
# import sys
# import matplotlib.style
# import matplotlib as mpl
import jet_aux as jx
from pyJets.jet_aux import CB_color_cycle
import pytools as pt
import os

# import scipy
# import scipy.linalg
from scipy.linalg import eig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib.ticker as ticker

# from matplotlib.ticker import MaxNLocator
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.lines import Line2D
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
    runid_labels = ["LM05", "LM30", "HM05", "HM30"]
    rect_anchor = [(6, -6), (6, -8), (6, -6), (6, -8)]
    rect_ex = [(12, 12), (12, 14), (12, 12), (12, 14)]
    CB_color_cycle = jx.CB_color_cycle
    # kinds = ["foreshock", "beam", "complex", "stripe"]
    kinds = ["beam", "foreshock"]
    # kinds_pub = ["Antisunward\njets", "Flankward\njets"]
    kinds_pub = ["Flankward\njets", "Antisunward\njets"]
    marker = ["^", "o"]
    draw_labels = [False, True, False, False]

    fig, ax_list = plt.subplots(2, 2, figsize=(11, 12), sharex=True, sharey=True)
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
        # for c in cont.collections:
        #     c.set_rasterized(True)
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
        # for c in cont.collections:
        # c.set_rasterized(True)
        # ax.plot(bs_x[idx], yarr, color="black")
        ax.annotate(annot[idx], (0.05, 0.90), xycoords="axes fraction", fontsize=20)

    for n1, runid in enumerate(runids):
        ax = ax_flat[n1]
        for n2, kind in enumerate(kinds):
            label_bool = draw_labels[n1]
            non_ids = np.loadtxt(
                wrkdir_DNR + "papu22/id_txts/auto/{}_{}.txt".format(runid, kind),
                dtype=int,
                ndmin=1,
            )
            for non_id in non_ids:
                props = jio.PropReader(str(non_id).zfill(5), runid, transient="jet")
                x0, y0, t0 = (
                    props.read("x_wmean")[0],
                    props.read("y_wmean")[0],
                    props.read("time")[0],
                )
                # bs_x_y0 = np.polyval(jx.bs_mp_fit(runid, int(t0 * 2))[1], y0)
                if label_bool:
                    ax.plot(
                        # np.polyval(bs_fit[n1], y0) - bs_fit[n1][-1] + (x0 - bs_x_y0),
                        x0,
                        y0,
                        marker[n2],
                        color=CB_color_cycle[n2],
                        label=kinds_pub[n2].capitalize(),
                        # rasterized=True,
                        zorder=2,
                        alpha=0.7,
                    )
                    label_bool = False
                else:
                    if (kind == "beam" and runid == "AEA" and non_id == 920) or (
                        kind == "foreshock" and runid == "ABC" and non_id == 153
                    ):
                        ax.plot(
                            # np.polyval(bs_fit[n1], y0) - bs_fit[n1][-1] + (x0 - bs_x_y0),
                            x0,
                            y0,
                            "*",
                            color=CB_color_cycle[n2],
                            # rasterized=True,
                            zorder=3,
                            alpha=1,
                            markersize=10,
                            mec="k",
                            mew=0.02,
                        )
                    else:
                        ax.plot(
                            # np.polyval(bs_fit[n1], y0) - bs_fit[n1][-1] + (x0 - bs_x_y0),
                            x0,
                            y0,
                            marker[n2],
                            color=CB_color_cycle[n2],
                            # rasterized=True,
                            zorder=2,
                            alpha=0.7,
                        )
        label_bool = draw_labels[n1]
        fcs_jet_ids = get_fcs_jets(runid)
        for sj_id in fcs_jet_ids:
            props = jio.PropReader(str(sj_id).zfill(5), runid, transient="jet")
            x0, y0, t0 = (
                props.read("x_wmean")[0],
                props.read("y_wmean")[0],
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
                    # rasterized=True,
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
                    # rasterized=True,
                    zorder=0,
                    alpha=0.3,
                )
        label_bool = draw_labels[n1]
        ax.grid()
        # ax.set_xlim(-3, 2)
        ax.set_xlim(4, 20)
        # ax.set_aspect("equal")
        ax.tick_params(labelsize=16)
        ax.set_title("{}".format(runid_labels[n1]), fontsize=20, pad=10)
        ax.add_patch(
            plt.Rectangle(
                rect_anchor[n1],
                rect_ex[n1][0],
                rect_ex[n1][1],
                fill=None,
                linestyle="dotted",
                color="k",
                linewidth=0.8,
            )
        )
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
        "$X~[R_\mathrm{E}]$\n\n$\\theta_\mathrm{cone}=5^\circ$",
        fontsize=20,
    )
    ax_flat[3].set_xlabel(
        "$X~[R_\mathrm{E}]$\n\n$\\theta_\mathrm{cone}=30^\circ$",
        fontsize=20,
    )

    # Save figure
    plt.tight_layout()

    fig.savefig(wrkdir_DNR + "papu22/Figures/fig7.pdf", dpi=300)
    plt.close(fig)


def get_fcs_jets(runid):
    runids = ["ABA", "ABC", "AEA", "AEC"]

    fcs_ids = []
    singular_counter = 0

    for n1 in range(6000):
        try:
            props = jio.PropReader(str(n1).zfill(5), runid, transient="jet")
        except:
            continue

        if props.read("at_bow_shock")[0] != 1:
            continue

        if props.read("time")[0] == 290.0:
            continue

        if props.read("time")[-1] - props.read("time")[0] == 0:
            singular_counter += 1
            continue

        if "splinter" in props.meta:
            continue

        if not (props.read("at_slams") == 1).any():
            continue
        else:
            fcs_ids.append(n1)

    # print("Run {} singular FCS jets: {}".format(runid, singular_counter))

    return np.unique(fcs_ids)


def get_non_jets(runid):
    runids = ["ABA", "ABC", "AEA", "AEC"]

    non_ids = []

    singular_counter = 0

    for n1 in range(6000):
        try:
            props = jio.PropReader(str(n1).zfill(5), runid, transient="jet")
        except:
            continue

        if props.read("at_bow_shock")[0] != 1:
            continue

        if props.read("time")[0] == 290.0:
            continue

        if props.read("time")[-1] - props.read("time")[0] == 0:
            singular_counter += 1
            continue

        if "splinter" in props.meta:
            continue

        if (props.read("at_slams") == 1).any():
            continue
        else:
            non_ids.append(n1)

    # print("Run {} singular non jets: {}".format(runid, singular_counter))

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
            wrkdir_DNR + "papu22/id_txts/auto/{}_{}.txt".format(runid, "foreshock"),
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

    # vmin = [
    #     np.min(data_arr[0]),
    #     -1,
    #     np.min(data_arr[2]),
    #     np.min(data_arr[3]),
    #     np.min(data_arr[4]),
    # ]
    # vmax = [
    #     np.max(data_arr[0]),
    #     0,
    #     np.max(data_arr[2]),
    #     np.max(data_arr[3]),
    #     np.max(data_arr[4]),
    # ]
    vmin = [0, -1, 0.25, 0, 10]
    vmax = [4, 0, 1, 4, 30]
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
                wrkdir_DNR + "papu22/id_txts/auto/{}_{}.txt".format(runid, kind),
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

    cmap = ["batlow", "Blues_r", "batlow", "batlow", "batlow"]
    vmin = [0, -1, 0.25, 0, 10]
    vmax = [4, 0, 1, 4, 30]
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
        # cb_list.append(fig.colorbar(im_list[idx], ax=ax))
        if idx == 1:
            cb_list.append(fig.colorbar(im_list[idx], ax=ax, extend="max"))
            cb_list[idx].cmap.set_over("red")
        else:
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
        "Run: {}, Type: {}, N = {}".format(run_id, kind, type_count),
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
        x0, y0 = (props.read("x_wmean")[0], props.read("y_wmean")[0])
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


def non_jet_jplots(runid, txt=False, draw=False):
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
    vmin = [0, -1, 0.25, 0, 10]
    vmax = [4, 0, 1, 4, 30]

    # Path to vlsv files for current run
    bulkpath = jx.find_bulkpath(runid)

    # Get IDs of non-fcs-jets
    non_sj_ids = get_non_jets(runid)

    # Loop through non-fcs-jet IDs
    for non_id in non_sj_ids:
        print("Non-FCS jets for run {}: {}".format(runid, non_id))
        props = jio.PropReader(str(non_id).zfill(5), runid, transient="jet")
        x0, y0 = (props.read("x_wmean")[0], props.read("y_wmean")[0])
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
                # rho_arr.append(
                #     [
                #         vlsvobj.read_interpolated_variable(
                #             "rho",
                #             [xpos * r_e, y0 * r_e, 0],
                #         )
                #         for xpos in x_range
                #     ]
                # )
                # v_arr.append(
                #     [
                #         vlsvobj.read_interpolated_variable(
                #             "v",
                #             [xpos * r_e, y0 * r_e, 0],
                #             operator="magnitude",
                #         )
                #         for xpos in x_range
                #     ]
                # )
                # pdyn_arr.append(
                #     [
                #         vlsvobj.read_interpolated_variable(
                #             "Pdyn",
                #             [xpos * r_e, y0 * r_e, 0],
                #         )
                #         for xpos in x_range
                #     ]
                # )
                # B_arr.append(
                #     [
                #         vlsvobj.read_interpolated_variable(
                #             "B",
                #             [xpos * r_e, y0 * r_e, 0],
                #             operator="magnitude",
                #         )
                #         for xpos in x_range
                #     ]
                # )
                # T_arr.append(
                #     [
                #         vlsvobj.read_interpolated_variable(
                #             "Temperature",
                #             [xpos * r_e, y0 * r_e, 0],
                #         )
                #         for xpos in x_range
                #     ]
                # )
                # Tcore_arr.append(
                #     [
                #         vlsvobj.read_interpolated_variable(
                #             "core_heating",
                #             [xpos * r_e, y0 * r_e, 0],
                #         )
                #         for xpos in x_range
                #     ]
                # )
                # mmsx_arr.append(
                #     [
                #         vlsvobj.read_interpolated_variable(
                #             "Mmsx",
                #             [xpos * r_e, y0 * r_e, 0],
                #         )
                #         for xpos in x_range
                #     ]
                # )

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
        cmap = ["batlow", "Blues_r", "batlow", "batlow", "batlow"]
        annot = ["a)", "b)", "c)", "d)", "e)"]

        # fig, ax_list = plt.subplots(
        #     1, len(varname_list), figsize=(20, 5), sharex=True, sharey=True
        # )
        if draw:
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
                        vmin=vmin[idx],
                        vmax=vmax[idx],
                        rasterized=True,
                    )
                )
                if idx == 1:
                    cb_list.append(fig.colorbar(im_list[idx], ax=ax, extend="max"))
                    cb_list[idx].cmap.set_over("red")
                else:
                    cb_list.append(fig.colorbar(im_list[idx], ax=ax))
                # cb_list.append(fig.colorbar(im_list[idx], ax=ax))
                cb_list[idx].ax.tick_params(labelsize=20)
                ax.contour(XmeshXY, YmeshXY, rho_arr, [2], colors=["black"])
                ax.contour(XmeshXY, YmeshXY, Tcore_arr, [3], colors=[CB_color_cycle[1]])
                ax.contour(
                    XmeshXY, YmeshXY, mmsx_arr, [1.0], colors=[CB_color_cycle[4]]
                )
                ax.set_title(varname_list[idx], fontsize=24, pad=10)
                ax.set_xlim(x_range[0], x_range[-1])
                ax.set_ylim(t_range[0], t_range[-1])
                ax.set_xlabel("$x$ [$R_\mathrm{E}$]", fontsize=24, labelpad=10)
                ax.axhline(t0, linestyle="dashed", linewidth=0.6)
                ax.axvline(x0, linestyle="dashed", linewidth=0.6)
                ax.annotate(
                    annot[idx], (0.05, 0.90), xycoords="axes fraction", fontsize=24
                )
            ax_list[0].set_ylabel("Simulation time [s]", fontsize=28, labelpad=10)
            ax_list[int(np.ceil(len(varname_list) / 2.0))].set_ylabel(
                "Simulation time [s]", fontsize=28, labelpad=10
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
                + "papu22/Figures/jmaps/{}_{}_jm.png".format(
                    runid, str(non_id).zfill(5)
                ),
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
        x0, y0 = (props.read("x_wmean")[0], props.read("y_wmean")[0])
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


def jet_avg_std(kind, version="squish"):
    runids = ["ABA", "ABC", "AEA", "AEC"]

    rho_sw = [1e6, 3.3e6, 1e6, 3.3e6]
    v_sw = [750e3, 600e3, 750e3, 600e3]
    pdyn_sw = [m_p * rho_sw[idx] * v_sw[idx] * v_sw[idx] for idx in range(len(runids))]

    data_arr = np.empty((5, 6000), dtype=float)
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
            pendep = props.read("x_wmean")[-1] - props.read("bs_distance")[-1]
            pd_max = np.max(props.read("pd_max")) / (pdyn_sw[runids.index(runid)] * 1e9)
            first_cone = props.read("first_cone")[0]
            min_r = np.min(props.read("r_mean"))
            data_arr[0][counter] = duration
            data_arr[1][counter] = -pendep
            data_arr[2][counter] = pd_max
            data_arr[3][counter] = first_cone
            data_arr[4][counter] = min_r
            counter += 1

    print("\n")
    # print("RUN: {}".format(runid))
    print("KIND: {}".format(kind.capitalize()))
    print("N = {}".format(counter))
    for idx, s in enumerate(["Duration", "Pendep", "Max pd", "First cone", "Min r"]):
        print(
            "{}, q1,med,q3 = ({:.3g}, {:.3g}, {:.3g})".format(
                s,
                np.nanpercentile(data_arr[idx], 25),
                np.nanmedian(data_arr[idx]),
                np.nanpercentile(data_arr[idx], 75),
            )
        )

    # print(
    #     "Duration = {:.3f} +- {:.3f} s".format(
    #         np.nanmedian(data_arr[0]), np.nanstd(data_arr[0], ddof=1)
    #     )
    # )
    # print(
    #     "Penetration depth = {:.3f} +- {:.3f} RE".format(
    #         np.nanmedian(data_arr[1]), np.nanstd(data_arr[1], ddof=1)
    #     )
    # )
    # print(
    #     "Max Pdyn = {:.3f} +- {:.3f} Pdyn_sw".format(
    #         np.nanmedian(data_arr[2]), np.nanstd(data_arr[2], ddof=1)
    #     )
    # )
    # print(
    #     "First cone = {:.3f} +- {:.3f} deg".format(
    #         np.nanmedian(data_arr[3]), np.nanstd(data_arr[3], ddof=1)
    #     )
    # )
    # print(
    #     "Min r = {:.3f} +- {:.3f} RE".format(
    #         np.nanmedian(data_arr[4]), np.nanstd(data_arr[4], ddof=1)
    #     )
    # )
    print("\n")

    # iqr_dur = np.subtract.reduce(np.nanpercentile(data_arr[0], [75, 25]))
    # iqr_pen = np.subtract.reduce(np.nanpercentile(data_arr[1], [75, 25]))
    # iqr_pd = np.subtract.reduce(np.nanpercentile(data_arr[2], [75, 25]))

    # bins_dur = (np.nanmax(data_arr[0]) - np.nanmin(data_arr[0])) / (
    #     2 * iqr_dur / float(counter) ** (1.0 / 3)
    # )
    # bins_pen = (np.nanmax(data_arr[1]) - np.nanmin(data_arr[1])) / (
    #     2 * iqr_pen / float(counter) ** (1.0 / 3)
    # )
    # bins_pd = (np.nanmax(data_arr[2]) - np.nanmin(data_arr[2])) / (
    #     2 * iqr_pd / float(counter) ** (1.0 / 3)
    # )

    # return (
    #     np.histogram(data_arr[0][:counter], bins=int(bins_dur)),
    #     np.histogram(data_arr[1][:counter], bins=int(bins_pen)),
    #     np.histogram(data_arr[0][:counter], bins=int(bins_pd)),
    # )


def kind_SEA_timeseries(kind):
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
        # "$\\rho~[\mathrm{cm}^{-3}]$",
        # "$v~[\mathrm{km/s}]$",
        # "$P_\mathrm{dyn}~[\mathrm{nPa}]$",
        # "$B~[\mathrm{nT}]$",
        # "$T~[\mathrm{MK}]$",
        "$\\rho~[\\rho_\mathrm{sw}]$",
        "$v~[v_\mathrm{sw}]$",
        "$P_\mathrm{dyn}~[P_\mathrm{dyn,sw}]$",
        "$B~[B_\mathrm{IMF}]$",
        "$T~[T_\mathrm{sw}]$",
    ]
    vmins = [1.5, -0.8, 0.4, -3.75, 10]
    vmaxs = [4.6, 0.8, 1.6, 3.75, 35]
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

    t_arr = np.arange(0 - 10.0, 0 + 10.1, 0.5)
    fnr_arr = np.arange(0 - 20, 0 + 21)
    avg_arr = np.zeros((len(plot_labels), fnr_arr.size), dtype=float)
    counter = 0
    for runid in ["ABA", "ABC", "AEA", "AEC"]:
        if kind == "fcs":
            non_ids = get_fcs_jets(runid)
        else:
            non_ids = np.loadtxt(
                wrkdir_DNR + "papu22/id_txts/auto/{}_{}.txt".format(runid, kind),
                dtype=int,
                ndmin=1,
            )
        for non_id in non_ids:
            data_arr = np.loadtxt(
                wrkdir_DNR
                + "papu22/timeseries_txts/{}_{}.txt".format(runid, str(non_id).zfill(5))
            )
            avg_arr = avg_arr + data_arr
            counter += 1

    avg_arr = avg_arr / counter
    fig, ax_list = plt.subplots(len(ylabels), 1, sharex=True, figsize=(6, 8))
    ax_list[0].set_title("Kind: {}".format(kind.capitalize()))
    for idx in range(len(plot_labels)):
        ax = ax_list[plot_index[idx]]
        ax.plot(t_arr, avg_arr[idx], color=plot_colors[idx], label=plot_labels[idx])
        ax.set_xlim(t_arr[0], t_arr[-1])
        if draw_legend[idx]:
            ax.legend()
    ax_list[-1].set_xlabel("Epoch time [s]")
    for idx, ax in enumerate(ax_list):
        ax.grid()
        ax.set_ylabel(ylabels[idx])
        ax.axvline(0, linestyle="dashed")
        ax.set_ylim(vmins[idx], vmaxs[idx])
    plt.tight_layout()
    fig.savefig(
        wrkdir_DNR + "papu22/Figures/timeseries_SEA_{}.pdf".format(kind),
        dpi=300,
    )
    plt.close(fig)


def SEA_trifecta(kind):
    plot_labels = ["VS1", "VS2", "VS3"]
    ylabels = [
        "$\\rho~[\\rho_\mathrm{sw}]$",
        "$B~[B_\mathrm{IMF}]$",
        "$v~[v_\mathrm{sw}]$",
        "$P_\mathrm{dyn}~[P_\mathrm{dyn,sw}]$",
        "$T~[T_\mathrm{sw}]$",
    ]
    plot_colors = [
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
    ]

    vmins = [1.5, 1.75, 0.2, 0.4, 10]
    vmaxs = [5, 3.75, 0.8, 1.6, 35]

    t_arr = np.arange(0 - 10.0, 0 + 10.1, 0.5)
    fnr_arr = np.arange(0 - 20, 0 + 21)
    avg_arr = np.zeros((3, len(ylabels) + 3 + 1, fnr_arr.size), dtype=float)
    counter = 0

    for runid in ["ABA", "ABC", "AEA", "AEC"]:
        if kind == "fcs":
            non_ids = get_fcs_jets(runid)
        else:
            non_ids = np.loadtxt(
                wrkdir_DNR + "papu22/id_txts/auto/{}_{}.txt".format(runid, kind),
                dtype=int,
                ndmin=1,
            )
        for non_id in non_ids:
            data_arr = np.load(
                wrkdir_DNR
                + "papu22/trifecta_txts/{}_{}.npy".format(runid, str(non_id).zfill(5))
            )
            avg_arr = avg_arr + data_arr
            counter += 1

    avg_arr = avg_arr / counter
    fig, ax_list = plt.subplots(len(ylabels), 1, sharex=True, figsize=(6, 8))
    ax_list[0].set_title("Kind: {}".format(kind.capitalize()))
    for idx in range(len(ylabels)):
        ax = ax_list[idx]
        for idx2 in range(len(plot_labels)):
            ax.plot(
                t_arr,
                avg_arr[idx2, idx],
                color=plot_colors[idx2],
                label=plot_labels[idx2],
            )
        ax.set_xlim(t_arr[0], t_arr[-1])
        if idx == 0:
            ax.legend()
    ax_list[-1].set_xlabel("Epoch time [s]")
    for idx, ax in enumerate(ax_list):
        ax.grid()
        ax.set_ylabel(ylabels[idx])
        ax.axvline(0, linestyle="dashed")
        ax.set_ylim(vmins[idx], vmaxs[idx])
    plt.tight_layout()
    fig.savefig(
        wrkdir_DNR + "papu22/Figures/trifecta_SEA_{}.pdf".format(kind),
        dpi=300,
    )
    plt.close(fig)

    # (i1,) = np.where(avg_arr[0, 3] == np.max(avg_arr[0, 3]))
    # (i2,) = np.where(avg_arr[1, 3] == np.max(avg_arr[1, 3]))
    # (i3,) = np.where(avg_arr[2, 3] == np.max(avg_arr[2, 3]))

    # d_cell = 227e3
    # dmatrix = np.array(
    #     [
    #         [-d_cell, -d_cell, -t_arr[i1][0]],
    #         [0, d_cell, -t_arr[i2][0]],
    #         [d_cell, -d_cell, -t_arr[i3][0]],
    #     ]
    # )
    # avec = np.array([1.0, 1.0, 1.0])
    # Xinv = np.linalg.pinv(dmatrix)
    # params = np.matmul(Xinv, avec)

    # vx = params[2] / params[0]
    # vy = params[2] / params[1]

    # print(
    #     "KIND: {}, VX = {:.3g} km/s, VY = {:.3g} km/s".format(kind, vx / 1e3, vy / 1e3)
    # )
    avg_res = avg_arr[0, -1]
    print(
        "\nLUCILES AVGS\nKIND: {}, VX = {:.3g} km/s, VY = {:.3g} km/s, VX_rel = {:.3g}, VY_rel = {:.3g}\n".format(
            kind,
            avg_res[0],
            avg_res[1],
            avg_res[2],
            avg_res[3],
        )
    )

    results = jx.timing_analysis_datadict(avg_arr)
    print(
        "\nLUCILES ALGORITHM\nKIND: {}, VX = {:.3g} km/s, VY = {:.3g} km/s, VX_rel = {:.3g}, VY_rel = {:.3g}\n".format(
            kind,
            results["wave_velocity_sc_frame"] * results["wave_vector"][0][0],
            results["wave_velocity_sc_frame"] * results["wave_vector"][1][0],
            results["wave_velocity_relative2sc"][0],
            results["wave_velocity_relative2sc"][1],
        )
    )


def trifecta(runid, kind="non", draw=False):
    bulkpath = jx.find_bulkpath(runid)

    runids = ["ABA", "ABC", "AEA", "AEC"]
    if kind == "fcs":
        non_ids = get_fcs_jets(runid)
    else:
        # non_ids = np.loadtxt(
        #     wrkdir_DNR + "papu22/id_txts/auto/{}_{}.txt".format(runid, kind),
        #     dtype=int,
        #     ndmin=1,
        # )
        non_ids = get_non_jets(runid)

    var_list = ["rho", "B", "v", "Pdyn", "Temperature", "core_heating"]
    plot_labels = ["VS1", "VS2", "VS3"]
    scales = [1e-6, 1e9, 1e-3, 1e9, 1e-6, 1e-6]
    ylabels = [
        "$\\rho~[\\rho_\mathrm{sw}]$",
        "$B~[B_\mathrm{IMF}]$",
        "$v~[v_\mathrm{sw}]$",
        "$P_\mathrm{dyn}~[P_\mathrm{dyn,sw}]$",
        "$T~[T_\mathrm{sw}]$",
        "$T_{core}~[T_\mathrm{sw}]$",
    ]
    norm = [
        [1, 5, 750, 0.9408498320756251, 0.5, 0.5],
        [3.3, 5, 600, 1.9870748453437201, 0.5, 0.5],
        [1, 10, 750, 0.9408498320756251, 0.5, 0.5],
        [3.3, 10, 600, 1.9870748453437201, 0.5, 0.5],
    ]
    ops = ["pass", "magnitude", "magnitude", "pass", "pass", "pass"]
    plot_colors = [
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
    ]
    v_ops = ["x", "y", "z"]
    run_norm = norm[runids.index(runid)]

    for non_id in non_ids:
        print("Jet {} in run {}".format(non_id, runid))
        props = jio.PropReader(str(non_id).zfill(5), runid, transient="jet")
        x0, y0 = (props.read("x_wmean")[0], props.read("y_wmean")[0])
        t0 = props.read("time")[0]
        t_arr = np.arange(t0 - 10.0, t0 + 10.1, 0.5)
        fnr0 = int(t0 * 2)
        fnr_arr = np.arange(fnr0 - 20, fnr0 + 21)
        d_cell = 227e3
        vlsvobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
        )
        cellids = [
            vlsvobj.get_cellid([x0 * r_e - d_cell, y0 * r_e - d_cell, 0 * r_e]),
            vlsvobj.get_cellid([x0 * r_e, y0 * r_e + d_cell, 0 * r_e]),
            vlsvobj.get_cellid([x0 * r_e + d_cell, y0 * r_e - d_cell, 0 * r_e]),
        ]
        coords = [
            [
                x0 * r_e + np.sin(np.deg2rad(phi)) * d_cell,
                y0 * r_e + np.cos(np.deg2rad(phi)) * d_cell,
                0,
            ]
            for phi in [-120, 0, 120]
        ]
        data_arr = np.zeros((3, len(var_list) + 6 + 1, fnr_arr.size), dtype=float)

        for idx, fnr in enumerate(fnr_arr):
            try:
                vlsvobj = pt.vlsvfile.VlsvReader(
                    bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
                )
                for idx2, var in enumerate(var_list):
                    data_arr[:, idx2, idx] = (
                        np.array(
                            [
                                vlsvobj.read_interpolated_variable(
                                    var, coords[idx3], operator=ops[idx2]
                                )
                                for idx3 in range(3)
                            ]
                        )
                        * scales[idx2]
                        / run_norm[idx2]
                    )
                for idx2 in range(3):
                    data_arr[:, len(var_list) + idx2, idx] = (
                        np.array(
                            [
                                vlsvobj.read_interpolated_variable(
                                    "v", coords[idx3], operator=v_ops[idx2]
                                )
                                for idx3 in range(3)
                            ]
                        )
                        * scales[2]
                    )
                    data_arr[:, len(var_list) + 3 + idx2, idx] = np.array(
                        [
                            vlsvobj.read_interpolated_variable(
                                "B", coords[idx3], operator=v_ops[idx2]
                            )
                            / np.sqrt(
                                m_p
                                * mu0
                                * vlsvobj.read_interpolated_variable(
                                    "rho", coords[idx3]
                                )
                            )
                            * scales[2]
                            for idx3 in range(3)
                        ]
                    )
            except:
                data_arr[:, :, idx] = np.nan

        results = jx.timing_analysis_datadict(data_arr)
        wave_vector = results["wave_vector"]
        wave_v_sc = results["wave_velocity_sc_frame"]
        vpl = results["wave_velocity_plasma_frame"]
        c = np.min(results["cross_corr_values"])
        out_results = [
            wave_v_sc * wave_vector[0][0],
            wave_v_sc * wave_vector[1][0],
            results["wave_velocity_relative2sc"][0],
            results["wave_velocity_relative2sc"][1],
            results["bulk_velocity"][0],
            results["bulk_velocity"][1],
            results["alfven_velocity"][0],
            results["alfven_velocity"][1],
            wave_v_sc * wave_vector[2][0],
            results["wave_velocity_relative2sc"][2],
            results["bulk_velocity"][2],
            results["alfven_velocity"][2],
            c,
        ]

        data_arr[0, len(var_list) + 6, :13] = out_results

        if draw:
            fig, ax_list = plt.subplots(len(ylabels), 1, sharex=True, figsize=(6, 6))
            ax_list[0].set_title("Run: {}, Jet: {}".format(runid, non_id))
            for idx in range(len(var_list)):
                ax = ax_list[idx]
                for idx2 in range(len(plot_labels)):
                    ax.plot(
                        t_arr,
                        data_arr[idx2, idx],
                        color=plot_colors[idx2],
                        label=plot_labels[idx2],
                    )
                ax.set_xlim(t_arr[0], t_arr[-1])
                if idx == 0:
                    ax.legend()

            ax_list[-1].set_xlabel(
                "Simulation time [s]\nWave (vx,vy) = ({:.3g},{:.3g}), RelSC (vx,vy) = ({:.3g},{:.3g})".format(
                    out_results[0], out_results[1], out_results[2], out_results[3]
                )
            )
            for idx, ax in enumerate(ax_list):
                ax.grid()
                ax.set_ylabel(ylabels[idx])
                ax.axvline(t0, linestyle="dashed")
            plt.tight_layout()
            fig.savefig(
                wrkdir_DNR
                + "papu22/Figures/trifecta/{}_{}_tf.png".format(
                    runid, str(non_id).zfill(5)
                ),
                dpi=300,
            )
            plt.close(fig)

        np.save(
            wrkdir_DNR
            + "papu22/trifecta_txts/{}_{}".format(runid, str(non_id).zfill(5)),
            data_arr,
        )


def kind_timeseries(runid, kind="non"):
    bulkpath = jx.find_bulkpath(runid)

    runids = ["ABA", "ABC", "AEA", "AEC"]
    if kind == "fcs":
        non_ids = get_fcs_jets(runid)
    else:
        # non_ids = np.loadtxt(
        #     wrkdir_DNR + "papu22/id_txts/auto/{}_{}.txt".format(runid, kind),
        #     dtype=int,
        #     ndmin=1,
        # )
        non_ids = get_non_jets(runid)

    var_list = [
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
        # "$\\rho~[\mathrm{cm}^{-3}]$",
        # "$v~[\mathrm{km/s}]$",
        # "$P_\mathrm{dyn}~[\mathrm{nPa}]$",
        # "$B~[\mathrm{nT}]$",
        # "$T~[\mathrm{MK}]$",
        "$\\rho~[\\rho_\mathrm{sw}]$",
        "$v~[v_\mathrm{sw}]$",
        "$P_\mathrm{dyn}~[P_\mathrm{dyn,sw}]$",
        "$B~[B_\mathrm{IMF}]$",
        "$T~[T_\mathrm{sw}]$",
    ]
    norm = [
        [1, 750, 750, 750, 750, 0.9408498320756251, 5, 5, 5, 5, 0.5, 0.5],
        [3.3, 600, 600, 600, 600, 1.9870748453437201, 5, 5, 5, 5, 0.5, 0.5],
        [1, 750, 750, 750, 750, 0.9408498320756251, 10, 10, 10, 10, 0.5, 0.5],
        [3.3, 600, 600, 600, 600, 1.9870748453437201, 10, 10, 10, 10, 0.5, 0.5],
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

    run_norm = norm[runids.index(runid)]
    for non_id in non_ids:
        print("Jet {} in run {}".format(non_id, runid))
        props = jio.PropReader(str(non_id).zfill(5), runid, transient="jet")
        x0, y0 = (props.read("x_wmean")[0], props.read("y_wmean")[0])
        t0 = props.read("time")[0]
        t_arr = np.arange(t0 - 10.0, t0 + 10.1, 0.5)
        fnr0 = int(t0 * 2)
        fnr_arr = np.arange(fnr0 - 20, fnr0 + 21)
        cellid = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
        ).get_cellid([x0 * r_e, y0 * r_e, 0 * r_e])
        data_arr = np.zeros((len(var_list), fnr_arr.size), dtype=float)

        for idx, fnr in enumerate(fnr_arr):
            try:
                vlsvobj = pt.vlsvfile.VlsvReader(
                    bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
                )
                for idx2, var in enumerate(var_list):
                    data_arr[idx2, idx] = (
                        vlsvobj.read_interpolated_variable(
                            var, [x0 * r_e, y0 * r_e, 0], operator=ops[idx2]
                        )
                        * scales[idx2]
                        / run_norm[idx2]
                    )
            except:
                data_arr[:, idx] = np.nan

        fig, ax_list = plt.subplots(len(ylabels), 1, sharex=True, figsize=(6, 8))
        ax_list[0].set_title("Run: {}, Jet: {}".format(runid, non_id))
        for idx in range(len(var_list)):
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
            + "papu22/Figures/timeseries/{}_{}_ts.png".format(
                runid, str(non_id).zfill(5)
            ),
            dpi=300,
        )
        np.savetxt(
            wrkdir_DNR
            + "papu22/timeseries_txts/{}_{}.txt".format(runid, str(non_id).zfill(5)),
            data_arr,
        )
        plt.close(fig)


def vz_timeseries(runid, kind="non"):
    bulkpath = jx.find_bulkpath(runid)

    runids = ["ABA", "ABC", "AEA", "AEC"]
    if kind == "fcs":
        non_ids = get_fcs_jets(runid)
    else:
        non_ids = get_non_jets(runid)

    var_list = [
        "rho",
        "v",
        "v",
        "v",
        "Pdyn",
        "B",
        "B",
        "B",
        "TParallel",
        "TPerpendicular",
    ]
    scales = [1e-6, 1e-3, 1e-3, 1e-3, 1e9, 1e9, 1e9, 1e9, 1e-6, 1e-6]
    norm = [
        [1, 750, 750, 750, 0.9408498320756251, 5, 5, 5, 0.5, 0.5],
        [3.3, 600, 600, 600, 1.9870748453437201, 5, 5, 5, 0.5, 0.5],
        [1, 750, 750, 750, 0.9408498320756251, 10, 10, 10, 0.5, 0.5],
        [3.3, 600, 600, 600, 1.9870748453437201, 10, 10, 10, 0.5, 0.5],
    ]
    ops = [
        "pass",
        "x",
        "yz",
        "magnitude",
        "pass",
        "x",
        "yz",
        "magnitude",
        "pass",
        "pass",
    ]

    run_norm = norm[runids.index(runid)]
    for non_id in non_ids:
        print("Jet {} in run {}".format(non_id, runid))
        props = jio.PropReader(str(non_id).zfill(5), runid, transient="jet")
        x0, y0 = (props.read("x_wmean")[0], props.read("y_wmean")[0])
        t0 = props.read("time")[0]
        t_arr = np.arange(t0 - 10.0, t0 + 10.1, 0.5)
        fnr0 = int(t0 * 2)
        fnr_arr = np.arange(fnr0 - 20, fnr0 + 21)
        cellid = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
        ).get_cellid([x0 * r_e, y0 * r_e, 0 * r_e])
        data_arr = np.zeros((len(var_list), fnr_arr.size), dtype=float)

        for idx, fnr in enumerate(fnr_arr):
            try:
                vlsvobj = pt.vlsvfile.VlsvReader(
                    bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
                )
                for idx2, var in enumerate(var_list):
                    if ops[idx2] == "yz":
                        data_arr[idx2, idx] = (
                            np.sqrt(
                                vlsvobj.read_interpolated_variable(
                                    var, [x0 * r_e, y0 * r_e, 0], operator="y"
                                )
                                ** 2
                                + vlsvobj.read_interpolated_variable(
                                    var, [x0 * r_e, y0 * r_e, 0], operator="z"
                                )
                                ** 2
                            )
                            * scales[idx2]
                            / run_norm[idx2]
                        )
                    else:
                        data_arr[idx2, idx] = (
                            vlsvobj.read_interpolated_variable(
                                var, [x0 * r_e, y0 * r_e, 0], operator=ops[idx2]
                            )
                            * scales[idx2]
                            / run_norm[idx2]
                        )
            except:
                data_arr[:, idx] = np.nan

        np.savetxt(
            wrkdir_DNR
            + "papu22/timeseries_yz_txts/{}_{}.txt".format(runid, str(non_id).zfill(5)),
            data_arr,
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
        x0, y0 = (props.read("x_wmean")[0], props.read("y_wmean")[0])
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
        x0, y0 = (props.read("x_wmean")[0], props.read("y_wmean")[0])
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

    for kind in ["foreshock", "beam"]:
        non_ids = np.loadtxt(
            wrkdir_DNR + "papu22/id_txts/auto/{}_{}.txt".format(runid, kind),
            dtype=int,
            ndmin=1,
        )
        for non_id in non_ids:
            jet_animator(runid, non_id, kind)

    return None


def jet_animator(runid, jetid, kind):
    global ax, x0, y0, pdmax, bulkpath, jetid_g
    global runid_g, sj_ids_g, non_ids_g, kind_g, Blines_g
    kind_g = kind
    jetid_g = jetid
    runid_g = runid
    Blines_g = False
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
    x0 = props.read("x_wmean")[0]
    y0 = props.read("y_wmean")[0]

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
        vmax=3.5,
        vscale=1e9,
        # cbtitle="$\\rho_{st}/\\rho_{th}",
        usesci=0,
        scale=2,
        title="",
        boxre=[x0 - 2, x0 + 2, y0 - 2, y0 + 2],
        # internalcb=True,
        lin=1,
        colormap="Blues_r",
        tickinterval=1.0,
        external=ext_jet,
        # expression=expr_rhoratio,
        pass_vars=[
            "RhoNonBackstream",
            "RhoBackstream",
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
            sj_xlist.append(jetobj.read_at_time("x_wmean", filenr_g / 2.0))
            sj_ylist.append(jetobj.read_at_time("y_wmean", filenr_g / 2.0))
    for jetobj in non_sjobs:
        if filenr_g / 2.0 in jetobj.read("time"):
            non_xlist.append(jetobj.read_at_time("x_wmean", filenr_g / 2.0))
            non_ylist.append(jetobj.read_at_time("y_wmean", filenr_g / 2.0))

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

    start_points = np.array(
        [np.ones(20) * x0 + 0.5, np.linspace(y0 - 0.9, y0 + 0.9, 20)]
    ).T
    # start_points = np.array([np.linspace(x0 - 0.9, x0 + 0.9, 10), np.ones(10) * y0]).T

    if Blines_g:
        stream = ax.streamplot(
            XmeshXY,
            YmeshXY,
            B[:, :, 0],
            B[:, :, 1],
            # arrowstyle="-",
            # broken_streamlines=False,
            color="k",
            linewidth=0.6,
            # minlength=4,
            density=35,
            start_points=start_points,
        )

    jet_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        jet_mask,
        [0.5],
        linewidths=2.2,
        colors=CB_color_cycle[2],
        linestyles=["solid"],
    )

    ch_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        ch_mask,
        [0.5],
        linewidths=2.2,
        colors=CB_color_cycle[1],
        linestyles=["solid"],
    )

    slams_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        slams_mask,
        [0.5],
        linewidths=2.2,
        colors=CB_color_cycle[7],
        linestyles=["solid"],
    )

    rho_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        rho_mask,
        [0.5],
        linewidths=2.2,
        colors=CB_color_cycle[3],
        linestyles=["solid"],
    )

    mach_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        mach_mask,
        [0.5],
        linewidths=2.2,
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

    itr_jumbled = [3, 1, 4, 2, 7]

    # proxy = [
    #     plt.Rectangle((0, 0), 1, 1, fc=CB_color_cycle[itr_jumbled[itr]])
    #     for itr in range(5)
    # ] + [non_pos, sj_pos]

    # proxy = [
    #     mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[itr]])
    #     for itr in range(5)
    # ] + [non_pos, sj_pos]

    # proxy_labs = (
    #         "$n=2n_\mathrm{sw}$",
    #         "$T_\mathrm{core}=3T_\mathrm{sw}$",
    #         "$M_{\mathrm{MS},x}=1$",
    #         "Jet",
    #         "FCS",
    #         "Non-FCS jet",
    #         "FCS-jet"
    #     )
    proxy = [
        mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[itr]])
        for itr in range(3)
    ]
    proxy_labs = [
        "$n=2n_\mathrm{sw}$",
        "$T_\mathrm{core}=3T_\mathrm{sw}$",
        "$M_{\mathrm{MS},x}=1$",
    ]

    xmin, xmax, ymin, ymax = (
        np.min(XmeshXY),
        np.max(XmeshXY),
        np.min(YmeshXY),
        np.max(YmeshXY),
    )

    if ~(jet_mask == 0).all():
        proxy.append(mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[3]]))
        proxy_labs.append("Jet")
    if ~(slams_mask == 0).all():
        proxy.append(mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[4]]))
        proxy_labs.append("FCS")
    if np.logical_and(
        np.logical_and(non_xlist >= xmin, non_xlist <= xmax),
        np.logical_and(non_ylist >= ymin, non_ylist <= ymax),
    ).any():
        proxy.append(non_pos)
        proxy_labs.append("Non-FCS jet")
    if np.logical_and(
        np.logical_and(sj_xlist >= xmin, sj_xlist <= xmax),
        np.logical_and(sj_ylist >= ymin, sj_ylist <= ymax),
    ).any():
        proxy.append(sj_pos)
        proxy_labs.append("FCS-jet")

    ax.legend(
        proxy,
        proxy_labs,
        frameon=True,
        numpoints=1,
        markerscale=1,
        loc="lower left",
        fontsize=14,
    )

    global gprox, gprox_labs

    gprox = proxy
    gprox_labs = proxy_labs


def expr_rhoratio(pass_maps):
    rho_th = pass_maps["RhoNonBackstream"]
    rho_st = pass_maps["RhoBackstream"]

    return rho_st / (rho_st + rho_th + 1e-27)


def fig0(runid="ABC", jetid=596):
    var = "Pdyn"
    vscale = 1e9
    vmax = 1.5
    if runid in ["ABC", "AEC"]:
        vmax = 3
    runids = ["ABA", "ABC", "AEA", "AEC"]
    runids_pub = ["HM30", "HM05", "LM30", "LM05"]

    rect_anchor = [(6, -8), (6, -6), (6, -8), (6, -6)]
    rect_ex = [(12, 14), (12, 12), (12, 14), (12, 12)]

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g
    runid_g = runid
    Blines_g = False

    bulkpath = jx.find_bulkpath(runid)

    non_ids = get_non_jets(runid)
    sj_ids = get_fcs_jets(runid)

    sj_ids_g = sj_ids
    non_ids_g = non_ids

    pdmax = [1.5, 3.5, 1.5, 3.5][runids.index(runid)]
    sw_pars = [
        [1e6, 750e3, 5e-9, 0.5e6],
        [3.3e6, 600e3, 5e-9, 0.5e6],
        [1e6, 750e3, 10e-9, 0.5e6],
        [3.3e6, 600e3, 10e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runids.index(runid)]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw

    print("Jet {} in run {}".format(jetid, runid))

    global x0, y0
    props = jio.PropReader(str(jetid).zfill(5), runid, transient="jet")
    t0 = props.read("time")[0]
    x0 = props.read("x_wmean")[0]
    y0 = props.read("y_wmean")[0]
    fnr0 = int(t0 * 2)

    filenr_g = fnr0

    fname = "bulk.{}.vlsv".format(str(int(fnr0)).zfill(7))

    fnr_range = np.arange(fnr0 - 30, fnr0 + 30 + 1)
    t_range = np.arange(t0 - 15, t0 + 15 + 0.1, 0.5)
    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
    )
    # Get cellid of initial position
    cellid = vlsvobj.get_cellid([x0 * r_e, y0 * r_e, 0 * r_e])
    dx = 227e3 / r_e

    cell_range = np.arange(cellid - 20, cellid + 20 + 1)
    x_range = np.arange(x0 - 20 * dx, x0 + 20 * dx + 0.5 * dx, dx)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    pt.plot.plot_colormap(
        axes=ax[0],
        filename=bulkpath + fname,
        # outputfile=wrkdir_DNR
        # + "papu22/Figures/var_plots/{}_{}_var_{}.png".format(
        #     runid, str(non_id).zfill(5), var
        # ),
        var=var,
        vmin=0,
        # vmax=1,
        vmax=vmax,
        vscale=vscale,
        # cbtitle="$P_{dyn}$ [nPa]",
        # cbtitle="",
        usesci=0,
        scale=2,
        # title="Run: {}, ID: {}\n t = {}s".format(
        #     runids_pub[runids.index(runid)], non_id, float(fnr0) / 2.0
        # ),
        # boxre=[x0 - 2, x0 + 2, y0 - 2, y0 + 2],
        internalcb=True,
        lin=1,
        colormap="batlow",
        tickinterval=5.0,
        useimshow=True,
        # external=ext_jet,
        # expression=expr_rhoratio,
        pass_vars=[
            "RhoNonBackstream",
            "RhoBackstream",
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

    ax[0].set_title(
        "Run: {}, t = {}s".format(runids_pub[runids.index(runid)], float(fnr0) / 2.0),
        pad=10,
        fontsize=24,
    )

    ax[0].add_patch(
        plt.Rectangle(
            rect_anchor[runids.index(runid)],
            rect_ex[runids.index(runid)][0],
            rect_ex[runids.index(runid)][1],
            fill=None,
            linestyle="dotted",
            color="k",
            linewidth=1.0,
        )
    )

    pt.plot.plot_colormap(
        axes=ax[1],
        filename=bulkpath + fname,
        # outputfile=wrkdir_DNR
        # + "papu22/Figures/var_plots/{}_{}_var_{}.png".format(
        #     runid, str(non_id).zfill(5), var
        # ),
        var=var,
        vmin=0,
        # vmax=1,
        vmax=vmax,
        vscale=vscale,
        useimshow=True,
        # cbtitle="$P_{dyn}$ [nPa]",
        # cbtitle="",
        usesci=0,
        scale=2,
        # title="Run: {}, ID: {}\n t = {}s".format(
        #     runids_pub[runids.index(runid)], non_id, float(fnr0) / 2.0
        # ),
        title="",
        boxre=[
            rect_anchor[runids.index(runid)][0],
            rect_anchor[runids.index(runid)][0] + rect_ex[runids.index(runid)][0],
            rect_anchor[runids.index(runid)][1],
            rect_anchor[runids.index(runid)][1] + rect_ex[runids.index(runid)][1],
        ],
        # internalcb=True,
        lin=1,
        colormap="batlow",
        tickinterval=2.0,
        external=ext_jet,
        nocb=True,
        # expression=expr_rhoratio,
        pass_vars=[
            "RhoNonBackstream",
            "RhoBackstream",
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

    ax[1].legend(gprox, gprox_labs, fontsize=20, loc="upper right")

    # plt.tight_layout()
    fig.savefig(wrkdir_DNR + "papu22/Figures/fig1.pdf")

    plt.close(fig)


def jet_var_plotter(runid, var):
    vars_list = [
        "Pdyn",
        "RhoNonBackstream",
        "RhoBackstream",
        "B",
        "vNonBackstream",
        "vBackstream",
    ]
    vscale = [1e9, 1e-6, 1e-6, 1e9, 1e-3, 1e-3][vars_list.index(var)]
    vmax = [1.5, 10, 10, 10, 250, 250][vars_list.index(var)]
    if runid in ["ABC", "AEC"]:
        vmax = [3, 10, 10, 10, 250, 250][vars_list.index(var)]

    runids = ["ABA", "ABC", "AEA", "AEC"]
    runids_pub = ["HM30", "HM05", "LM30", "LM05"]

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g
    runid_g = runid
    Blines_g = False

    bulkpath = jx.find_bulkpath(runid)

    non_ids = get_non_jets(runid)
    sj_ids = get_fcs_jets(runid)

    sj_ids_g = sj_ids
    non_ids_g = non_ids

    pdmax = [1.5, 3.5, 1.5, 3.5][runids.index(runid)]
    sw_pars = [
        [1e6, 750e3, 5e-9, 0.5e6],
        [3.3e6, 600e3, 5e-9, 0.5e6],
        [1e6, 750e3, 10e-9, 0.5e6],
        [3.3e6, 600e3, 10e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runids.index(runid)]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw

    for non_id in non_ids:
        print("Jet {} in run {}".format(non_id, runid))

        global x0, y0
        props = jio.PropReader(str(non_id).zfill(5), runid, transient="jet")
        t0 = props.read("time")[0]
        x0 = props.read("x_wmean")[0]
        y0 = props.read("y_wmean")[0]
        fnr0 = int(t0 * 2)

        filenr_g = fnr0

        fname = "bulk.{}.vlsv".format(str(int(fnr0)).zfill(7))

        fnr_range = np.arange(fnr0 - 30, fnr0 + 30 + 1)
        t_range = np.arange(t0 - 15, t0 + 15 + 0.1, 0.5)
        vlsvobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
        )
        # Get cellid of initial position
        cellid = vlsvobj.get_cellid([x0 * r_e, y0 * r_e, 0 * r_e])
        dx = 227e3 / r_e

        cell_range = np.arange(cellid - 20, cellid + 20 + 1)
        x_range = np.arange(x0 - 20 * dx, x0 + 20 * dx + 0.5 * dx, dx)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        pt.plot.plot_colormap(
            axes=ax,
            filename=bulkpath + fname,
            # outputfile=wrkdir_DNR
            # + "papu22/Figures/var_plots/{}_{}_var_{}.png".format(
            #     runid, str(non_id).zfill(5), var
            # ),
            var=var,
            vmin=0,
            # vmax=1,
            vmax=vmax,
            vscale=vscale,
            # cbtitle="$P_{dyn}$ [nPa]",
            # cbtitle="",
            usesci=0,
            scale=2,
            # title="Run: {}, ID: {}\n t = {}s".format(
            #     runids_pub[runids.index(runid)], non_id, float(fnr0) / 2.0
            # ),
            boxre=[x0 - 2, x0 + 2, y0 - 2, y0 + 2],
            internalcb=True,
            lin=1,
            colormap="batlow",
            tickinterval=1.0,
            external=ext_jet,
            # expression=expr_rhoratio,
            pass_vars=[
                "RhoNonBackstream",
                "RhoBackstream",
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
            "Run: {}, ID: {}\n t = {}s".format(
                runids_pub[runids.index(runid)], non_id, float(fnr0) / 2.0
            ),
            pad=10,
            fontsize=24,
        )
        plt.tight_layout()
        fig.savefig(
            wrkdir_DNR
            + "papu22/Figures/var_plots/{}_{}_var_{}.png".format(
                runid, str(non_id).zfill(5), var
            )
        )

        plt.close(fig)


def non_jet_omni(runid, only_man_figs=True):
    runids = ["ABA", "ABC", "AEA", "AEC"]
    runids_pub = ["HM30", "HM05", "LM30", "LM05"]

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g
    runid_g = runid

    Blines_g = True

    bulkpath = jx.find_bulkpath(runid)

    non_ids = get_non_jets(runid)
    sj_ids = get_fcs_jets(runid)

    sj_ids_g = sj_ids
    non_ids_g = non_ids

    pdmax = [1.5, 3.5, 1.5, 3.5][runids.index(runid)]
    sw_pars = [
        [1e6, 750e3, 5e-9, 0.5e6],
        [3.3e6, 600e3, 5e-9, 0.5e6],
        [1e6, 750e3, 10e-9, 0.5e6],
        [3.3e6, 600e3, 10e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runids.index(runid)]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw

    for non_id in non_ids:
        if only_man_figs:
            if (runid == "ABC" and non_id == 153) or (runid == "AEA" and non_id == 920):
                pass
            else:
                continue

        print("Jet {} in run {}".format(non_id, runid))

        global x0, y0
        props = jio.PropReader(str(non_id).zfill(5), runid, transient="jet")
        t0 = props.read("time")[0]
        x0 = props.read("x_wmean")[0]
        y0 = props.read("y_wmean")[0]
        fnr0 = int(t0 * 2)

        filenr_g = fnr0

        fname = "bulk.{}.vlsv".format(str(int(fnr0)).zfill(7))

        fnr_range = np.arange(fnr0 - 30, fnr0 + 30 + 1)
        t_range = np.arange(t0 - 15, t0 + 15 + 0.1, 0.5)
        vlsvobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
        )
        # Get cellid of initial position
        cellid = vlsvobj.get_cellid([x0 * r_e, y0 * r_e, 0 * r_e])
        dx = 227e3 / r_e

        cell_range = np.arange(cellid - 20, cellid + 20 + 1)
        x_range = np.arange(x0 - 20 * dx, x0 + 20 * dx + 0.5 * dx, dx)

        fig = plt.figure(figsize=(16, 16), constrained_layout=True)
        # gs = fig.add_gridspec(11, 11)
        # ax_nw = fig.add_subplot(gs[0:5, 0:5])
        # ax_ne = fig.add_subplot(gs[0:5, 6:11])
        # ax_sw = fig.add_subplot(gs[6:11, 0:5])
        # ax_se_list = [fig.add_subplot(gs[6 + n, 6:11]) for n in range(5)]
        gs = fig.add_gridspec(20, 2)
        ax_nw = fig.add_subplot(gs[0:9, 0])
        ax_ne = fig.add_subplot(gs[0:9, 1])
        ax_sw = fig.add_subplot(gs[10:20, 0])
        ax_se_list = [
            fig.add_subplot(gs[10 + 2 * n : 10 + 2 * n + 2, 1]) for n in range(5)
        ]

        pt.plot.plot_colormap(
            axes=ax_nw,
            filename=bulkpath + fname,
            var="Pdyn",
            vmin=0,
            # vmax=pdmax,
            vmax=1,
            # vscale=1e9,
            vscale=1.0 / Pdyn_sw,
            # cbtitle="$P_{dyn}$ [nPa]",
            cbtitle="",
            usesci=0,
            scale=2,
            title="",
            boxre=[x0 - 1, x0 + 1, y0 - 1, y0 + 1],
            useimshow=True,
            # internalcb=True,
            lin=1,
            # colormap="batlow",
            colormap="Blues_r",
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
        ax_nw.set_title(
            "Run: {}, ID: {}\n $t_0$ = {}s".format(
                runids_pub[runids.index(runid)], non_id, float(fnr0) / 2.0
            ),
            pad=10,
            fontsize=24,
        )
        ax_nw.axhline(y0, linestyle="dashed", linewidth=0.6, color="k")
        ax_nw.axvline(x0, linestyle="dashed", linewidth=0.6, color="k")
        ax_nw.annotate(
            "a)",
            (0.05, 0.90),
            xycoords="axes fraction",
            fontsize=20,
            bbox=dict(
                boxstyle="square,pad=0.2",
                fc="white",
                ec="k",
                lw=1,
            ),
        )
        ax_nw.annotate(
            "jet",
            xy=(x0, y0),
            xytext=(x0 - 0.5, y0),
            fontsize=20,
            arrowprops=dict(
                facecolor=CB_color_cycle[2],
                ec=CB_color_cycle[2],
                shrink=0.1,
                width=2,
                headwidth=6,
            ),
            bbox=dict(
                boxstyle="square,pad=0.2",
                fc="white",
                ec=CB_color_cycle[2],
                lw=1,
            ),
        )
        ax_nw.annotate(
            # "$P_\mathrm{dyn}$ [nPa]",
            "$P_\mathrm{dyn}~[P_\mathrm{dyn,sw}]$",
            (0.95, 1.05),
            xycoords="axes fraction",
            fontsize=20,
        )

        rho_arr, v_arr, pdyn_arr, B_arr, T_arr, Tcore_arr, mmsx_arr = np.load(
            wrkdir_DNR
            + "papu22/jmap_txts/{}/{}_{}.npy".format(runid, runid, str(non_id).zfill(5))
        )
        XmeshXY, YmeshXY = np.meshgrid(x_range, t_range)
        ax_ne.tick_params(labelsize=16)
        im = ax_ne.pcolormesh(
            x_range,
            t_range,
            pdyn_arr,
            shading="nearest",
            cmap="Blues_r",
            vmin=1.0 / 6,
            vmax=2,
            rasterized=True,
        )
        cb = fig.colorbar(im, ax=ax_ne, location="top", pad=-0.05)
        # cb_list.append(fig.colorbar(im_list[idx], ax=ax))
        cb.ax.tick_params(labelsize=16)
        cb.set_label("$P_\mathrm{dyn}~[P_\mathrm{dyn,sw}]$", fontsize=24, labelpad=10)
        ax_ne.contour(
            XmeshXY,
            YmeshXY,
            rho_arr,
            [2],
            colors=[CB_color_cycle[3]],
            linewidths=[2],
        )
        ax_ne.contour(
            XmeshXY,
            YmeshXY,
            Tcore_arr,
            [3],
            colors=[CB_color_cycle[1]],
            linewidths=[2],
        )
        ax_ne.contour(
            XmeshXY,
            YmeshXY,
            mmsx_arr,
            [1.0],
            colors=[CB_color_cycle[4]],
            linewidths=[2],
        )
        # ax_ne.set_title("$P_\mathrm{dyn}~[P_\mathrm{dyn,sw}]$", fontsize=24, pad=10)
        ax_ne.set_xlim(x_range[0], x_range[-1])
        ax_ne.set_ylim(t0 - 10, t0 + 10)
        ax_ne.set_xlabel("$x$ [$R_\mathrm{E}$]", fontsize=22, labelpad=10)
        ax_ne.axhline(t0, linestyle="dashed", linewidth=0.6)
        ax_ne.axvline(x0, linestyle="dashed", linewidth=0.6)
        ax_ne.annotate(
            "b)",
            (0.05, 0.90),
            xycoords="axes fraction",
            fontsize=20,
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="k", lw=1),
        )
        ax_ne.set_ylabel("Simulation time [s]", fontsize=24, labelpad=10)

        try:
            # trifecta_data = np.load(
            #     wrkdir_DNR
            #     + "papu22/trifecta_txts/{}_{}.npy".format(runid, str(non_id).zfill(5))
            # )
            trifecta_data = np.load(
                wrkdir_DNR
                + "papu22/trifecta_txts/{}_{}.npy".format(runid, str(non_id).zfill(5))
            )
            res = trifecta_data[0, -1]
            c = res[12]
            if c < 0.8:
                res[[0, 1, 2, 3]] = np.nan
            tlist, xlist, ylist = np.loadtxt(
                wrkdir_DNR
                + "papu22/jet_prop_v_txts/{}_{}.txt".format(runid, str(non_id).zfill(5))
            ).T

            ch_mask = trifecta_data[1, 5, :] >= 3

            t0, x0, y0 = tlist[0], xlist[0], ylist[0]

            propvx = (xlist[tlist - t0 < 2.5][-1] - x0) / (
                tlist[tlist - t0 < 2.5][-1] - t0 + 1e-27
            )
            propvy = (ylist[tlist - t0 < 2.5][-1] - y0) / (
                tlist[tlist - t0 < 2.5][-1] - t0 + 1e-27
            )
            propvx_full = (xlist[-1] - x0) / (tlist[-1] - t0 + 1e-27)
            propvy_full = (ylist[-1] - y0) / (tlist[-1] - t0 + 1e-27)

            va = (
                np.nanmean(
                    B_sw
                    * trifecta_data[1, 1, :][ch_mask]
                    / np.sqrt(mu0 * m_p * rho_sw * trifecta_data[1, 0, :][ch_mask])
                )
                / 1e3
            )
            vs = (
                np.nanmean(
                    np.sqrt(5.0 / 3 * kb * T_sw * trifecta_data[1, 4, :][ch_mask] / m_p)
                )
                / 1e3
            )
            vms = np.sqrt(va**2 + vs**2)
            va_xy = (
                va
                * np.array(
                    [
                        (np.cos(theta), np.sin(theta))
                        for theta in np.arange(0, np.pi * 2, 0.01)
                    ]
                ).T
            )
            vms_xy = (
                vms
                * np.array(
                    [
                        (np.cos(theta), np.sin(theta))
                        for theta in np.arange(0, np.pi * 2, 0.01)
                    ]
                ).T
            )

            B = vlsvobj.read_variable("B", cellids=cellid)
            n = vlsvobj.read_variable("rho", cellids=cellid)
            ax_sw.plot(
                vms_xy[0],
                vms_xy[1],
                color="k",
                linestyle="dashed",
                label="$v_\mathrm{MS}$",
            )
            ax_sw.plot(
                va_xy[0],
                va_xy[1],
                color=CB_color_cycle[3],
                linestyle="dashed",
                label="$v_\mathrm{A}$",
            )
            ax_sw.quiver(
                0,
                0,
                propvx,
                propvy,
                color=CB_color_cycle[0],
                label="$v_\mathrm{tr}$",
                angles="xy",
                scale_units="xy",
                scale=1,
                linewidth=1,
                edgecolor="k",
            )
            # ax_sw.quiver(
            #     0,
            #     0,
            #     propvx_full,
            #     propvy_full,
            #     color=CB_color_cycle[6],
            #     label="$v_\mathrm{tr,full}$",
            #     angles="xy",
            #     scale_units="xy",
            #     scale=1,
            # )
            vx_arr = np.array(
                [
                    res[0],
                    res[2],
                    res[4],
                    # res[6],
                ]
            )
            vy_arr = np.array(
                [
                    res[1],
                    res[3],
                    res[5],
                    # res[7],
                ]
            )
            arrow_labels = [
                "$v_\mathrm{n}$",
                "$v_\mathrm{SC}$",
                "$v_\mathrm{bulk}$",
                # "$v_\mathrm{A}$",
            ]
            for idx in range(1, len(vx_arr)):
                ax_sw.quiver(
                    0,
                    0,
                    vx_arr[idx],
                    vy_arr[idx],
                    color=CB_color_cycle[idx],
                    label=arrow_labels[idx],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    linewidth=1,
                    edgecolor="k",
                )
            ax_sw.legend(fontsize=16, loc="lower right")
            lhand, llab = ax_sw.get_legend_handles_labels()
            order = [3, 2, 4, 0, 1]
            ax_sw.legend(
                [lhand[idx3] for idx3 in order],
                [llab[idx3] for idx3 in order],
                fontsize=16,
                loc="lower right",
            )
            ax_sw.set_xlabel("$v_x$ [km/s]", fontsize=24, labelpad=10)
            ax_sw.set_ylabel("$v_y$ [km/s]", fontsize=24, labelpad=10)
            maxv = np.max([np.max(np.abs(vx_arr)), np.max(np.abs(vy_arr))])
            ax_sw.set_xlim(-1.1 * v_sw / 1e3, 1.1 * v_sw / 1e3)
            ax_sw.set_ylim(-1.1 * v_sw / 1e3, 1.1 * v_sw / 1e3)
            ax_sw.grid()
            ax_sw.set_aspect("equal")
            ax_sw.tick_params(labelsize=16)
            # ax_sw.set_title("Timing analysis", fontsize=24, pad=10)
            ax_sw.annotate("c)", (0.05, 0.90), xycoords="axes fraction", fontsize=20)
        except:
            ax_sw.set_axis_off()

        try:
            timeseries_data = np.loadtxt(
                wrkdir_DNR
                + "papu22/timeseries_yz_txts/{}_{}.txt".format(
                    runid, str(non_id).zfill(5)
                )
            )
            timeseries_data[5, :] = np.abs(timeseries_data[5, :])  # Use |Bx|
            var_list = [
                "rho",
                "v",
                "v",
                "v",
                "Pdyn",
                "B",
                "B",
                "B",
                "TParallel",
                "TPerpendicular",
            ]
            plot_labels = [
                None,
                "$v_x$",
                "$|v_{yz}|$",
                "$|v|$",
                None,
                "$|B_x|$",
                "$|B_{yz}|$",
                "$|B|$",
                "$T_\\parallel$",
                "$T_\\perp$",
            ]
            draw_legend = [
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
                "$\\rho~[\\rho_\mathrm{sw}]$",
                "$v~[v_\mathrm{sw}]$",
                "$P_\mathrm{dyn}$\n$[P_\mathrm{dyn,sw}]$",
                "$B~[B_\mathrm{IMF}]$",
                "$T~[T_\mathrm{sw}]$",
            ]
            annots = ["d)", "e)", "f)", "g)", "h)"]
            plot_index = [0, 1, 1, 1, 2, 3, 3, 3, 4, 4]
            plot_colors = [
                "k",
                CB_color_cycle[0],
                CB_color_cycle[1],
                "k",
                "k",
                CB_color_cycle[0],
                CB_color_cycle[1],
                "k",
                CB_color_cycle[0],
                CB_color_cycle[1],
            ]
            t_arr = np.arange(t0 - 10, t0 + 10 + 0.1, 0.5)
            for idx in range(len(var_list)):
                ax = ax_se_list[plot_index[idx]]
                ax.plot(
                    t_arr,
                    timeseries_data[idx],
                    color=plot_colors[idx],
                    label=plot_labels[idx],
                )
                ax.set_xlim(t_arr[0], t_arr[-1])
                if draw_legend[idx]:
                    ax.legend(
                        loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=16
                    )
            ax_se_list[-1].set_xlabel("Simulation time [s]", fontsize=24)
            for idx, ax in enumerate(ax_se_list):
                ax.grid()
                ax.set_ylabel(ylabels[idx], fontsize=20)
                ax.axvline(t0, linestyle="dashed")
                ax.tick_params(labelsize=16)
                if idx != len(ax_se_list) - 1:
                    ax.set_xticklabels([])
                ax.annotate(
                    annots[idx],
                    (0.05, 0.75),
                    xycoords="axes fraction",
                    fontsize=20,
                    bbox=dict(
                        boxstyle="square,pad=0.2", fc="white", ec="k", lw=1, alpha=0.5
                    ),
                )
            # ax_se_list[0].set_title("Timeseries", fontsize=24, pad=10)
        except:
            for ax in ax_se_list:
                ax.set_axis_off()

        # plt.tight_layout()

        # fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0.2, wspace=0.2)

        fig.savefig(
            wrkdir_DNR
            + "papu22/Figures/omni/{}_{}_omni.png".format(runid, str(non_id).zfill(5))
        )
        if runid == "ABC" and non_id == 153:
            fig.savefig(
                wrkdir_DNR
                + "papu22/Figures/fig4.pdf".format(runid, str(non_id).zfill(5))
            )
        elif runid == "AEA" and non_id == 920:
            fig.savefig(
                wrkdir_DNR
                + "papu22/Figures/fig3.pdf".format(runid, str(non_id).zfill(5))
            )

        plt.close(fig)


def jmap_SEA_comp(run_id="all"):
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

    kinds = ["beam", "foreshock", "fcs"]
    kind_names = ["Flankward jets", "Antisunward jets", "FCS-jets"]
    prefix = ["", "", "sj_"]
    counts = [0, 0, 0]

    data_avg = np.zeros((3, 7, XmeshXY.shape[0], XmeshXY.shape[1]))

    for runid in runid_list:
        for idx, kind in enumerate(kinds):
            if kind == "fcs":
                non_ids = get_fcs_jets(runid)
            else:
                non_ids = np.loadtxt(
                    wrkdir_DNR + "papu22/id_txts/auto/{}_{}.txt".format(runid, kind),
                    dtype=int,
                    ndmin=1,
                )

            for non_id in non_ids:
                data = np.load(
                    wrkdir_DNR
                    + "papu22/{}jmap_txts/{}/{}_{}.npy".format(
                        prefix[idx], runid, runid, str(non_id).zfill(5)
                    )
                )
                data_avg[idx, :, :, :] = data_avg[idx, :, :, :] + data
                counts[idx] += 1

    for idx, kind in enumerate(kinds):
        if counts[idx] != 0:
            data_avg[idx, :, :, :] /= counts[idx]
        else:
            print("No jets of kind {} found in run {}".format(kind, run_id))
            return 0

    varname_list = [
        "$n$ [$n_\mathrm{sw}$]",
        "$v_x$ [$v_\mathrm{sw}$]",
        "$P_\mathrm{dyn}$ [$P_\mathrm{dyn,sw}$]",
        "$| \mathbf{B} |$ [$B_\mathrm{IMF}$]",
        "$T$ [$T_\mathrm{sw}$]",
        # "$M_{\mathrm{MS},x}$",
    ]

    fig, ax_list = plt.subplots(
        len(varname_list), len(kinds), figsize=(24, 16), sharex=True, sharey=True
    )
    im_list = []
    cb_list = []
    vmin = [1.25, -1, 0.25, 1.3, 15]
    vmax = [4, -0.25, 1.2, 3.5, 25]
    # cmap = [
    #     "batlow",
    #     "batlow",
    #     "batlow",
    #     "batlow",
    #     "batlow",
    # ]
    cmap = [
        "Blues_r",
        "Blues_r",
        "Blues_r",
        "Blues_r",
        "Blues_r",
    ]
    annot = [
        ["a)", "b)", "c)", "d)", "e)"],
        ["f)", "g)", "h)", "i)", "j)"],
        ["k)", "l)", "m)", "n)", "o)"],
    ]
    annot_sj = ["f)", "g)", "h)", "i)", "j)"]

    for idx2 in range(len(kinds)):
        for idx, ax in enumerate(ax_list[:, idx2]):
            ax.tick_params(labelsize=20)
            im_list.append(
                ax.pcolormesh(
                    x_range,
                    t_range,
                    data_avg[idx2, idx, :, :],
                    shading="nearest",
                    cmap=cmap[idx],
                    vmin=vmin[idx],
                    vmax=vmax[idx],
                    rasterized=True,
                )
            )
            if idx2 == 2:
                if idx == 11:  # Disabled
                    cb_list.append(
                        fig.colorbar(
                            im_list[idx2 * len(varname_list) + idx], ax=ax, extend="max"
                        )
                    )
                    # cb_list[idx2 * len(varname_list) + idx].cmap.set_over("red")
                    cb_list[idx].cmap.set_over("red")
                else:
                    cb_list.append(
                        fig.colorbar(im_list[idx2 * len(varname_list) + idx], ax=ax)
                    )
                # cb_list[idx2 * len(varname_list) + idx].ax.tick_params(labelsize=20)
                cb_list[idx].ax.tick_params(labelsize=20)
                cb_list[idx].set_label(varname_list[idx], fontsize=28, labelpad=10)
            ax.contour(
                XmeshXY,
                YmeshXY,
                data_avg[idx2, 0, :, :],
                [2],
                colors=[CB_color_cycle[3]],
                linewidths=[3],
            )
            ax.contour(
                XmeshXY,
                YmeshXY,
                data_avg[idx2, 5, :, :],
                [3],
                colors=[CB_color_cycle[1]],
                linewidths=[3],
            )
            ax.contour(
                XmeshXY,
                YmeshXY,
                data_avg[idx2, 6, :, :],
                [1.0],
                colors=[CB_color_cycle[4]],
                linewidths=[3],
            )
            # ax.set_title(varname_list[idx], fontsize=24, pad=10)
            ax.set_xlim(x_range[0], x_range[-1])
            ax.set_ylim(t_range[0], t_range[-1])
            ax.axhline(t0, linestyle="dashed", linewidth=0.6)
            ax.axvline(x0, linestyle="dashed", linewidth=0.6)
            ax.annotate(
                annot[idx2][idx],
                (0.05, 0.85),
                xycoords="axes fraction",
                fontsize=24,
                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="k", lw=1),
            )
        ax_list[0][idx2].set_title(kind_names[idx2], fontsize=32, pad=10)
        ax_list[-1][idx2].set_xlabel(
            "Epoch $x$ [$R_\mathrm{E}$]", fontsize=32, labelpad=10
        )
    for idx, ax in enumerate(ax_list[:, 0]):
        # ax.set_ylabel(
        #     "{}\nEpoch time [s]".format(varname_list[idx]), fontsize=28, labelpad=10
        # )
        ax.set_ylabel("Epoch time [s]", fontsize=28, labelpad=10)
    proxy = [
        mlines.Line2D([], [], color=CB_color_cycle[3]),
        mlines.Line2D([], [], color=CB_color_cycle[1]),
        mlines.Line2D([], [], color=CB_color_cycle[4]),
    ]

    ax_list[0][0].legend(
        proxy,
        (
            "$n=2n_\mathrm{sw}$",
            "$T_\mathrm{core}=3T_\mathrm{sw}$",
            "$M_{\mathrm{MS},x}=1$",
        ),
        frameon=True,
        numpoints=1,
        markerscale=1,
        loc="lower left",
        fontsize=18,
        framealpha=0.5,
    )
    # Save figure
    plt.tight_layout()

    fig.savefig(
        wrkdir_DNR + "papu22/Figures/fig5.pdf".format(run_id),
        dpi=300,
    )
    plt.close(fig)


def print_means_max():
    means_maxes = np.load(wrkdir_DNR + "papu22/SEA_timeseries_mean_max.npy")

    means, epochval = means_maxes

    varnames = [
        "rho",
        "vx",
        "vy",
        "vz",
        "vmag",
        "pdyn",
        "Bx",
        "By",
        "Bz",
        "Bmag",
        "Tpar",
        "Tperp",
    ]
    print("test")

    kinds = ["Flankward", "Antisunward", "FCS"]

    for idx, var in enumerate(varnames):
        print(var)
        for idx2, kind in enumerate(kinds):
            print(
                kind
                + ": Mean = {:.3g}, Epochval = {:.3g}, Enhancement = {:.3g}, Delta = {:.3g}".format(
                    means[idx2, idx],
                    epochval[idx2, idx],
                    epochval[idx2, idx] / means[idx2, idx],
                    epochval[idx2, idx] - means[idx2, idx],
                )
            )
        print("\n")

    return None


def calc_conv_ExB(v, B):
    Bmag = np.linalg.norm(B, axis=-1)

    return np.cross(-np.cross(v, B), B) / (Bmag**2)


def SEA_timeseries_comp():
    plot_labels = [
        None,
        "$v_x$",
        "$|v_{yz}|$",
        "$|v|$",
        None,
        "$|B_x|$",
        "$|B_{yz}|$",
        "$|B|$",
        "$T_\\parallel$",
        "$T_\\perp$",
    ]
    draw_legend = [
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
        "$n~[n_\mathrm{sw}]$",
        "$v~[v_\mathrm{sw}]$",
        "$P_\mathrm{dyn}~[P_\mathrm{dyn,sw}]$",
        "$B~[B_\mathrm{IMF}]$",
        "$T~[T_\mathrm{sw}]$",
    ]
    vmins = [0.2, -1.1, 0, -0.5, 0]
    vmaxs = [7, 1.1, 2.3, 5, 55]
    plot_index = [0, 1, 1, 1, 2, 3, 3, 3, 4, 4]
    offsets = np.array([0, -0.2, 0.2, 0, 0, -0.2, 0.2, 0, 0, -0.2]) - 0.1
    plot_colors = [
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        "k",
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
    ]

    annot = [
        ["a)", "b)", "c)", "d)", "e)", "f)"],
        ["g)", "h)", "i)", "j)", "k)", "l)"],
        ["m)", "n)", "o)", "p)", "q)", "r)"],
    ]

    kinds = ["beam", "foreshock", "fcs"]
    kind_labels = ["Flankward jets", "Antisunward jets", "FCS-jets"]
    vsw = [750e3, 600e3, 750e3, 600e3]
    Bsw = [5e-9, 5e-9, 10e-9, 10e-9]
    t_arr = np.arange(0 - 10.0, 0 + 10.1, 0.5)
    fnr_arr = np.arange(0 - 20, 0 + 21)
    avg_arr = np.zeros((len(kinds), len(plot_labels), fnr_arr.size), dtype=float)
    Tani_avg_arr = np.zeros((len(kinds), fnr_arr.size), dtype=float)
    epoch_mag_arr = np.empty((len(kinds), len(plot_labels), 3, 1000), dtype=float)
    # v_conv_ExB = np.zeros((len(kinds), 3, fnr_arr.size), dtype=float)
    # print(epoch_mag_arr.shape)
    epoch_mag_arr.fill(np.nan)
    counters = [0, 0, 0]
    for runid in ["ABA", "ABC", "AEA", "AEC"]:
        for idx, kind in enumerate(kinds):
            # run_vsw = vsw[["ABA", "ABC", "AEA", "AEC"].index(runid)]
            # run_Bsw = Bsw[["ABA", "ABC", "AEA", "AEC"].index(runid)]
            if kind == "fcs":
                non_ids = get_fcs_jets(runid)
            else:
                non_ids = np.loadtxt(
                    wrkdir_DNR + "papu22/id_txts/auto/{}_{}.txt".format(runid, kind),
                    dtype=int,
                    ndmin=1,
                )
            for non_id in non_ids:
                data_arr = np.loadtxt(
                    wrkdir_DNR
                    + "papu22/timeseries_yz_txts/{}_{}.txt".format(
                        runid, str(non_id).zfill(5)
                    )
                )
                if np.isnan(data_arr).any():
                    continue
                data_arr[5, :] = np.abs(data_arr[5, :])  # Use |Bx| instead of Bx
                avg_arr[idx] = avg_arr[idx] + data_arr
                Tani_avg_arr[idx] = (
                    Tani_avg_arr[idx] + data_arr[-1, :] / data_arr[-2, :]
                )
                # print(data_arr.shape)
                # epoch_mag_arr[idx, :, counters[idx]] = data_arr[:, 20][[0, 4, 5, 9]]
                epoch_mag_arr[idx, :, :, counters[idx]] = data_arr[:, 7::13]
                # data_v = run_vsw * data_arr[[1, 2, 3], :].T
                # data_B = run_Bsw * data_arr[[6, 7, 8], :].T
                # v_conv_ExB[idx] = (
                #     v_conv_ExB[idx] + (1.0 / run_vsw) * calc_conv_ExB(data_v, data_B).T
                # )
                counters[idx] += 1

    for idx in range(len(kinds)):
        avg_arr[idx] = avg_arr[idx] / counters[idx]
        Tani_avg_arr[idx] = Tani_avg_arr[idx] / counters[idx]
        # v_conv_ExB[idx] = v_conv_ExB[idx] / counters[idx]

    means = np.mean(avg_arr, axis=-1)
    epochval = avg_arr[:, :, 20]

    means_max_arr = np.array([means, epochval])
    np.save(wrkdir_DNR + "papu22/SEA_timeseries_mean_max", means_max_arr)

    T_ani = np.zeros((len(kinds), 3, 1000), dtype=float)
    for idx in range(len(kinds)):
        T_ani[idx, :, : counters[idx]] = (
            epoch_mag_arr[idx, -1, :, : counters[idx]]
            / epoch_mag_arr[idx, -2, :, : counters[idx]]
        )

    fig, ax_list = plt.subplots(len(ylabels) + 1, 3, sharex=True, figsize=(24, 24))
    for idx2, kind in enumerate(kinds):
        ax_list[0][idx2].set_title("{}".format(kind_labels[idx2]), fontsize=40, pad=10)
        for idx in range(len(plot_labels)):
            ax = ax_list[plot_index[idx]][idx2]
            ax.plot(
                t_arr,
                avg_arr[idx2, idx],
                color=plot_colors[idx],
                label=plot_labels[idx],
                linewidth=2,
            )
            # if idx in [1, 2, 3]:
            #     ax.plot(
            #         t_arr,
            #         v_conv_ExB[idx2, idx - 1],
            #         color=plot_colors[idx],
            #         linewidth=2,
            #         linestyle="dashed",
            #     )
            ax.boxplot(
                epoch_mag_arr[idx2, idx, :, : counters[idx2]].T,
                # positions=[0],
                positions=np.arange(-6.5, 7.5, 6.5) + offsets[idx],
                manage_ticks=False,
                widths=1.0,
                sym="",
                whis=1.5,
                notch=False,
                boxprops=dict(color=plot_colors[idx]),
                capprops=dict(color=plot_colors[idx]),
                whiskerprops=dict(color=plot_colors[idx]),
                # flierprops=dict(color=c, markeredgecolor=c),
                medianprops=dict(color=plot_colors[idx]),
            )
            ax.set_xlim(t_arr[0], t_arr[-1])
            if draw_legend[idx] and idx2 == 2:
                # ax.legend(loc="lower right", fontsize=22, ncols=3, framealpha=0.5)
                ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=24)
        ax_list[-1][idx2].set_xlabel("Epoch time [s]", fontsize=40, labelpad=10)
        for idx, ax in enumerate(ax_list[:-1, idx2]):
            ax.grid()
            # ax.set_xticks(np.arange(-10, 10.1, 2.5))
            ax.set_xticks(np.arange(-7.5, 10.1, 2.5))
            # ax.set_xticklabels(["", "", "-5", "", "0", "", "5", "", "10"])
            ax.tick_params(labelsize=22)
            if idx2 == 0:
                ax.set_ylabel(ylabels[idx], fontsize=40, labelpad=10)
            # ax.axvline(0, linestyle="dashed")
            ax.set_ylim(vmins[idx], vmaxs[idx])
            ax.annotate(
                annot[idx2][idx], (0.05, 0.85), xycoords="axes fraction", fontsize=32
            )
    for idx2, kind in enumerate(kinds):
        ax = ax_list[-1][idx2]
        ax.plot(
            t_arr,
            Tani_avg_arr[idx2],
            color="k",
            # label="$T_\perp/T_\parallel$",
            linewidth=2,
        )
        ax.boxplot(
            T_ani[idx2, :, : counters[idx2]].T,
            # positions=[0],
            positions=np.arange(-6.5, 7.5, 6.5) - 0.1,
            manage_ticks=False,
            widths=1.0,
            sym="",
            whis=1.5,
            notch=False,
            boxprops=dict(color="k"),
            capprops=dict(color="k"),
            whiskerprops=dict(color="k"),
            # flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color="k"),
        )
        ax.grid()
        # ax.set_xticks(np.arange(-10, 10.1, 2.5))
        ax.set_xticks(np.arange(-7.5, 10.1, 2.5))
        # ax.set_xticklabels(["", "", "-5", "", "0", "", "5", "", "10"])
        ax.tick_params(labelsize=22)
        if idx2 == 0:
            ax.set_ylabel("$T_\perp/T_\parallel$", fontsize=40, labelpad=10)
        # ax.axvline(0, linestyle="dashed")
        ax.set_ylim(0, 4)
        ax.annotate(
            annot[idx2][-1], (0.05, 0.85), xycoords="axes fraction", fontsize=32
        )
    for ax in ax_list.flat:
        ax.label_outer()
    plt.tight_layout()
    fig.savefig(
        wrkdir_DNR + "papu22/Figures/fig6.pdf",
        dpi=300,
    )
    plt.close(fig)


def kinds_pca():
    var_names = np.array(
        [
            "rho",
            "vx",
            "vyz",
            "vmag",
            "pdyn",
            "Bx",
            "Byz",
            "Bmag",
            "TPar",
            "TPerp",
        ]
    )
    vars = np.array([])

    kinds = ["beam", "foreshock", "fcs"]
    kind_labels = ["Flankward jets", "Antisunward jets", "FCS-jets"]

    data_arr = []
    classes_arr = [[], [], []]
    counters = [0, 0, 0]

    runid_arr = []
    id_arr = []

    for idx, kind in enumerate(kinds):
        for idx2, runid in enumerate(["ABA", "ABC", "AEA", "AEC"]):
            if kind == "fcs":
                non_ids = get_fcs_jets(runid)
            else:
                non_ids = np.loadtxt(
                    wrkdir_DNR + "papu22/id_txts/auto/{}_{}.txt".format(runid, kind),
                    dtype=int,
                    ndmin=1,
                )
            for non_id in non_ids:
                ts_data = np.loadtxt(
                    wrkdir_DNR
                    + "papu22/timeseries_txts/{}_{}.txt".format(
                        runid, str(non_id).zfill(5)
                    )
                )
                if np.isnan(ts_data).any():
                    continue
                # data_arr.append(ts_data[:, 20])
                # classes_arr[idx].append(ts_data[:, 20])
                # vars = var_names[:]

                # data_arr.append(ts_data[:, [0, 20, 40]].flatten())
                # classes_arr[idx].append(ts_data[:, [0, 20, 40]].flatten())

                # data_arr.append(ts_data.flatten())
                # classes_arr[idx].append(ts_data.flatten())

                data_arr.append(ts_data[[0, 1, 2, 5, 6, 8, 9], 20].flatten())
                classes_arr[idx].append(ts_data[[0, 1, 2, 5, 6, 8, 9], 20].flatten())
                vars = var_names[[0, 1, 2, 5, 6, 8, 9]]
                runid_arr.append(runid)
                id_arr.append(non_id)

                # data_arr.append(ts_data[[0, 1, 2, 5, 6, 7, 11], :].flatten())
                # data_arr.append(ts_data[[0, 1, 2, 3, 5, 6, 7, 8, 10, 11], 20])
                counters[idx] += 1

    Y = np.array(data_arr)
    Y_lda = [np.array(arr) for arr in classes_arr]

    n, p = Y.shape
    n_lda = [arr.shape[0] for arr in Y_lda]

    color_arr = (
        [CB_color_cycle[0]] * counters[0]
        + [CB_color_cycle[1]] * counters[1]
        + [CB_color_cycle[2]] * counters[2]
    )
    sym_arr = ["o"] * counters[0] + ["x"] * counters[1] + ["^"] * counters[2]
    zorder_arr = [3] * counters[0] + [2] * counters[1] + [1] * counters[2]

    mean_arr = np.mean(Y, axis=0)
    mean_lda = [np.mean(arr, axis=0) for arr in Y_lda]

    std_arr = np.std(Y, axis=0, ddof=1)
    std_lda = [np.std(arr, axis=0, ddof=1) for arr in Y_lda]

    ones_arr = np.ones((n, p))
    ones_lda = [np.ones((nl, p)) for nl in n_lda]

    X = Y - np.matmul(ones_arr, np.diag(mean_arr))
    X_lda = [
        Y_lda[idx] - np.matmul(ones_lda[idx], np.diag(mean_lda[idx]))
        for idx in range(len(kinds))
    ]

    V = np.diag(1.0 / std_arr)
    V_lda = [np.diag(1.0 / std_lda[idx]) for idx in range(len(kinds))]

    X = np.matmul(X, V)
    X_lda = [np.matmul(X_lda[idx], V_lda[idx]) for idx in range(len(kinds))]

    S = np.matmul(X.T, X) / (n - 1)
    S_lda = [
        np.matmul(X_lda[idx].T, X_lda[idx]) / (n_lda[idx] - 1)
        for idx in range(len(kinds))
    ]

    W_lda = np.zeros_like(S_lda[0])
    B_lda = np.zeros_like(np.outer(mean_lda[0], mean_lda[0]))
    for idx in range(len(kinds)):
        W_lda += (n_lda[idx] - 1) * S_lda[idx]
        B_lda += n_lda[idx] * np.outer(mean_lda[0], mean_lda[0])

    lbd, U = np.linalg.eig(S)
    print("S residual: {}".format(np.linalg.norm(S - U @ np.diag(lbd) @ U.T)))
    print("PCA var importance: {}".format(vars[np.argsort(lbd)[::-1]]))
    print("PCA lambdas: {}".format(lbd[np.argsort(lbd)[::-1]]))
    U = U[:, np.argsort(lbd)[::-1]]

    print(np.linalg.inv(W_lda).shape)
    print(B_lda.shape)

    lbd_lda, U_lda = np.linalg.eig(np.matmul(np.linalg.inv(W_lda), B_lda))
    print(
        "WinvB residual: {}".format(
            np.linalg.norm(
                np.linalg.inv(W_lda) @ B_lda
                - (U_lda @ np.diag(lbd_lda) @ np.linalg.inv(U_lda))
            )
        )
    )
    print("LDA var importance: {}".format(vars[np.argsort(lbd_lda)[::-1]]))
    print("LDA lambdas: {}".format(lbd_lda[np.argsort(lbd_lda)[::-1]]))
    U_lda = U_lda[:, np.argsort(lbd_lda)[::-1]]

    # U = U.T
    # U_lda = U_lda.T
    print(U.shape)

    Z = np.matmul(X, U)
    Z_lda = np.matmul(X, U_lda)
    print(Z.shape)

    fig, ax = plt.subplots(1, 2)

    ax[0].set_xlabel("PCA1")
    ax[1].set_xlabel("PCA1")
    ax[0].set_ylabel("PCA2")
    ax[1].set_ylabel("PCA3")
    ax[0].grid(zorder=0)
    ax[1].grid(zorder=0)

    for idx, row in enumerate(Z[:, 0]):
        ax[0].plot(
            Z[idx, 0],
            Z[idx, 1],
            sym_arr[idx],
            color=color_arr[idx],
            zorder=zorder_arr[idx],
        )
        ax[1].plot(
            Z[idx, 0],
            Z[idx, 2],
            sym_arr[idx],
            color=color_arr[idx],
            zorder=zorder_arr[idx],
        )
    plt.tight_layout()

    fig.savefig(
        wrkdir_DNR + "papu22/Figures/kinds_pca.pdf",
        dpi=300,
    )
    plt.close(fig)

    fig, ax = plt.subplots(1, 2)

    ax[0].set_xlabel("LDA1")
    ax[1].set_xlabel("LDA1")
    ax[0].set_ylabel("LDA2")
    ax[1].set_ylabel("LDA3")
    ax[0].grid(zorder=0)
    ax[1].grid(zorder=0)

    for idx, row in enumerate(Z[:, 0]):
        ax[0].plot(
            Z_lda[idx, 0],
            Z_lda[idx, 1],
            sym_arr[idx],
            color=color_arr[idx],
            zorder=zorder_arr[idx],
        )
        ax[1].plot(
            Z_lda[idx, 0],
            Z_lda[idx, 2],
            sym_arr[idx],
            color=color_arr[idx],
            zorder=zorder_arr[idx],
        )

    plt.tight_layout()

    fig.savefig(
        wrkdir_DNR + "papu22/Figures/kinds_lda.pdf",
        dpi=300,
    )
    plt.close(fig)

    for idx in range(counters[0]):
        if Z[idx, 0] < 2.0:
            print(
                "Outlier: Runid = {}, ID = {}, PCA1 = {}".format(
                    runid_arr[idx], id_arr[idx], Z[idx, 0]
                )
            )


def timing_comp():
    vsws = [750, 600, 750, 600]

    nsws = [1.0e6, 3.3e6, 1.0e6, 3.3e6]
    Bsws = [5.0e-9, 5.0e-9, 10.0e-9, 10.0e-9]

    kinds = ["beam", "foreshock", "fcs"]
    kind_labels = ["Flankward jets", "Antisunward jets", "FCS-jets"]
    annot = ["b)", "c)", "d)"]
    arrow_labels = [
        "$v_\mathrm{n}$",
        "$v_\mathrm{SC}$",
        # "$v_{\\langle \mathrm{SC} \\rangle}$",
        "$v_\mathrm{bulk}$",
        # "$v_\mathrm{A}$",
    ]
    ylabels = [
        "$\\rho~[\\rho_\mathrm{sw}]$",
        "$B~[B_\mathrm{IMF}]$",
        "$v~[v_\mathrm{sw}]$",
        "$P_\mathrm{dyn}~[P_\mathrm{dyn,sw}]$",
        "$T~[T_\mathrm{sw}]$",
        "$T_{core}~[T_\mathrm{sw}]$",
    ]

    fnr_arr = np.arange(0 - 20, 0 + 21)
    avg_arr = np.zeros((3, 3, len(ylabels) + 6 + 1, fnr_arr.size), dtype=float)
    timing_arrs = np.zeros((3, fnr_arr.size, 1000), dtype=float)
    alfven_arrs = np.zeros((3, 1000), dtype=float)
    sonic_arrs = np.zeros((3, 1000), dtype=float)
    special_arrs = np.zeros((3, 1000), dtype=float)
    propv_arrs = np.zeros((3, 2, 1000), dtype=float)
    propv_arrs_full = np.zeros((3, 2, 1000), dtype=float)
    # ts_avg_arr = np.zeros((3, 12, fnr_arr.size))
    # vax_avg_arr = np.zeros((3, fnr_arr.size))
    # vay_avg_arr = np.zeros((3, fnr_arr.size))
    counters = [0, 0, 0]

    for idx, kind in enumerate(kinds):
        for idx2, runid in enumerate(["ABA", "ABC", "AEA", "AEC"]):
            vsw = vsws[idx2]
            Bsw = Bsws[idx2]
            nsw = nsws[idx2]
            Tsw = 0.5e6
            if kind == "fcs":
                non_ids = get_fcs_jets(runid)
            else:
                non_ids = np.loadtxt(
                    wrkdir_DNR + "papu22/id_txts/auto/{}_{}.txt".format(runid, kind),
                    dtype=int,
                    ndmin=1,
                )
            for non_id in non_ids:
                # ts_data = np.loadtxt(
                #     wrkdir_DNR
                #     + "papu22/timeseries_txts/{}_{}.txt".format(
                #         runid, str(non_id).zfill(5)
                #     )
                # )
                data_arr = np.load(
                    wrkdir_DNR
                    + "papu22/trifecta_txts/{}_{}.npy".format(
                        runid, str(non_id).zfill(5)
                    )
                )
                tlist, xlist, ylist = np.loadtxt(
                    wrkdir_DNR
                    + "papu22/jet_prop_v_txts/{}_{}.txt".format(
                        runid, str(non_id).zfill(5)
                    )
                ).T
                if kind == "beam" and runid == "AEA" and non_id == 920:
                    special_arrs[idx, counters[idx]] = 1
                elif kind == "foreshock" and runid == "ABC" and non_id == 153:
                    special_arrs[idx, counters[idx]] = 1

                t0, x0, y0 = tlist[0], xlist[0], ylist[0]

                propvx = (xlist[tlist - t0 < 2.5][-1] - x0) / (
                    tlist[tlist - t0 < 2.5][-1] - t0 + 1e-27
                )
                propvy = (ylist[tlist - t0 < 2.5][-1] - y0) / (
                    tlist[tlist - t0 < 2.5][-1] - t0 + 1e-27
                )
                propvx_full = (xlist[-1] - x0) / (tlist[-1] - t0 + 1e-27)
                propvy_full = (ylist[-1] - y0) / (tlist[-1] - t0 + 1e-27)

                # propvx = np.array(propvx, ndmin=1)
                # propvy = np.array(propvy, ndmin=1)

                propv_arrs[idx, :, counters[idx]] = [
                    propvx / vsw,
                    propvy / vsw,
                ]
                propv_arrs_full[idx, :, counters[idx]] = [
                    propvx_full / vsw,
                    propvy_full / vsw,
                ]
                # print(data_arr[:, -1, :8])
                c = data_arr[0, -1, 12]
                data_arr[:, 6:, :] /= vsw
                # ts_avg_arr[idx] = ts_avg_arr[idx] + ts_data
                # vax_avg_arr[idx] = (
                #     vax_avg_arr[idx]
                #     + ts_data[6, :]
                #     * Bsw
                #     / np.sqrt(m_p * mu0 * ts_data[0, :] * nsw)
                #     / vsw
                #     / 1.0e3
                # )
                # vay_avg_arr[idx] = (
                #     vay_avg_arr[idx]
                #     + ts_data[7, :]
                #     * Bsw
                #     / np.sqrt(m_p * mu0 * ts_data[0, :] * nsw)
                #     / vsw
                #     / 1.0e3
                # )
                ch_mask = data_arr[1, 5, :] >= 3
                timing_arrs[idx, :, counters[idx]] = data_arr[0, -1, :]
                if c < 0.8:
                    timing_arrs[idx, [0, 1, 2, 3], counters[idx]] = np.nan
                alfven_arrs[idx, counters[idx]] = np.nanmean(
                    Bsw
                    * data_arr[1, 1, :][ch_mask]
                    / np.sqrt(mu0 * m_p * nsw * data_arr[1, 0, :][ch_mask])
                ) / (vsw * 1e3)
                sonic_arrs[idx, counters[idx]] = np.nanmean(
                    np.sqrt(5.0 / 3 * kb * Tsw * data_arr[1, 4, :][ch_mask] / m_p)
                ) / (vsw * 1e3)

                avg_arr[idx] = avg_arr[idx] + data_arr
                counters[idx] += 1

    for idx, kind in enumerate(kinds):
        avg_arr[idx] = avg_arr[idx] / counters[idx]
        # vax_avg_arr[idx] = vax_avg_arr[idx] / counters[idx]
        # vay_avg_arr[idx] = vay_avg_arr[idx] / counters[idx]

    fig, ax_list = plt.subplots(2, 2, figsize=(18, 18))
    magnetosonic_arrs = np.sqrt(alfven_arrs**2 + sonic_arrs**2)
    vx_all = []
    vy_all = []
    vms_all = []
    ax_flat = ax_list.flatten()
    for idx, ax in enumerate(ax_flat[1:]):
        ax.set_title("{}".format(kind_labels[idx]), fontsize=32, pad=10)
        avg_res = avg_arr[idx, 0, -1]
        # print(avg_res)
        a_med = np.nanmedian(alfven_arrs[idx, : counters[idx]])
        ms_med = np.nanmedian(magnetosonic_arrs[idx, : counters[idx]])
        circ_xy = np.array(
            [(np.cos(theta), np.sin(theta)) for theta in np.arange(0, np.pi * 2, 0.01)]
        )
        ms_xy = circ_xy * ms_med
        vms_all = vms_all + [ms_med]
        a_xy = circ_xy * a_med
        ax.plot(
            ms_xy.T[0],
            ms_xy.T[1],
            color="k",
            linestyle="dashed",
            label="$v_\mathrm{MS}$",
        )
        ax.plot(
            a_xy.T[0],
            a_xy.T[1],
            color=CB_color_cycle[3],
            linestyle="dashed",
            label="$v_\mathrm{A}$",
        )

        vx = [
            # avg_res[0],
            # avg_res[2],
            # avg_res[4],
            # avg_res[6],
            np.nanmedian(timing_arrs[idx, 0, : counters[idx]]),
            np.nanmedian(timing_arrs[idx, 2, : counters[idx]]),
            np.nanmedian(timing_arrs[idx, 4, : counters[idx]]),
            # np.nanmedian(timing_arrs[idx, 6, : counters[idx]]),
        ]
        vy = [
            # avg_res[1],
            # avg_res[3],
            # avg_res[5],
            # avg_res[7],
            np.nanmedian(timing_arrs[idx, 1, : counters[idx]]),
            np.nanmedian(timing_arrs[idx, 3, : counters[idx]]),
            np.nanmedian(timing_arrs[idx, 5, : counters[idx]]),
            # np.nanmedian(timing_arrs[idx, 7, : counters[idx]]),
        ]
        vx_all = vx_all + vx
        vy_all = vy_all + vy

        # ax.quiver(
        #     0,
        #     0,
        #     np.nanmedian(propv_arrs_full[idx, 0, : counters[idx]]),
        #     np.nanmedian(propv_arrs_full[idx, 1, : counters[idx]]),
        #     color=CB_color_cycle[6],
        #     label="$v_\mathrm{tr,full}$",
        #     angles="xy",
        #     scale_units="xy",
        #     scale=1,
        #     zorder=2,
        # )
        for idx2 in range(1, len(vx)):
            ax.quiver(
                0,
                0,
                vx[idx2],
                vy[idx2],
                color=CB_color_cycle[idx2],
                label=arrow_labels[idx2],
                angles="xy",
                scale_units="xy",
                scale=1,
                zorder=1,
                linewidth=1,
                edgecolor="k",
            )
            if idx2 != 2:
                for n in range(counters[idx]):
                    vx_one = timing_arrs[idx, :, n][2 * idx2]
                    vy_one = timing_arrs[idx, :, n][2 * idx2 + 1]
                    vx_all = vx_all + [vx_one]
                    vy_all = vy_all + [vy_one]
                    if special_arrs[idx, n] == 1.0:
                        ax.plot(
                            vx_one,
                            vy_one,
                            "*",
                            color=CB_color_cycle[idx2],
                            alpha=1,
                            zorder=3,
                            markersize=20,
                            mec="k",
                            mew=1,
                        )
                    else:
                        ax.plot(
                            vx_one,
                            vy_one,
                            "^",
                            color=CB_color_cycle[idx2],
                            alpha=0.5,
                            zorder=0,
                            markersize=10,
                        )
        for n in range(counters[idx]):
            if special_arrs[idx, n] == 1.0:
                ax.plot(
                    propv_arrs[idx, 0, n],
                    propv_arrs[idx, 1, n],
                    "*",
                    color=CB_color_cycle[0],
                    alpha=1,
                    zorder=3,
                    markersize=20,
                    mec="k",
                    mew=1,
                )
            else:
                ax.plot(
                    propv_arrs[idx, 0, n],
                    propv_arrs[idx, 1, n],
                    "o",
                    color=CB_color_cycle[0],
                    alpha=0.5,
                    zorder=0,
                    markersize=10,
                )
            vx_all = vx_all + [propv_arrs[idx, 0, n]]
            vy_all = vy_all + [propv_arrs[idx, 1, n]]
        ax.quiver(
            0,
            0,
            np.nanmedian(propv_arrs[idx, 0, : counters[idx]]),
            np.nanmedian(propv_arrs[idx, 1, : counters[idx]]),
            color=CB_color_cycle[0],
            label="$v_\mathrm{tr}$",
            angles="xy",
            scale_units="xy",
            scale=1,
            zorder=2,
            linewidth=1,
            edgecolor="k",
        )
        # ax.set_xlim(-1.1 * np.nanmax(np.abs(vx_all+vy_all)), 1.1 * np.nanmax(np.abs(vx_all+vy_all)))
        # ax.set_ylim(-1.1 * np.nanmax(np.abs(vx_all+vy_all)), 1.1 * np.nanmax(np.abs(vx_all+vy_all)))
        ax.set_xlim(-1.0 - np.nanmax(vms_all), 1.0 + np.nanmax(vms_all))
        ax.set_ylim(-1.0 - np.nanmax(vms_all), 1.0 + np.nanmax(vms_all))
        ax.annotate(annot[idx], (0.05, 0.90), xycoords="axes fraction", fontsize=32)
        ax.set_ylabel("$v_y$ [$v_{sw}$]", fontsize=32, labelpad=10)
        if idx == 1:
            ax.legend(fontsize=24, loc="lower right")
            lhand, llab = ax.get_legend_handles_labels()
            order = [2, 4, 3, 0, 1]
            # ax.legend(
            #     [lhand[idx3] for idx3 in order],
            #     [llab[idx3] for idx3 in order],
            #     fontsize=24,
            #     loc="lower right",
            # )
            ax.legend(
                [lhand[idx3] for idx3 in order],
                [llab[idx3] for idx3 in order],
                fontsize=24,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncols=3,
            )
        ax.set_xlabel("$v_x$ [$v_{sw}$]", fontsize=32, labelpad=10)
        ax.tick_params(labelsize=20)
        ax.grid()
        ax.set_aspect("equal")

    # for ax in ax_flat[1:]:
    #     ax.set_xlim(
    #         -1.1 * np.nanmax(np.abs(vx_all + vy_all)),
    #         1.1 * np.nanmax(np.abs(vx_all + vy_all)),
    #     )
    #     ax.set_ylim(
    #         -1.1 * np.nanmax(np.abs(vx_all + vy_all)),
    #         1.1 * np.nanmax(np.abs(vx_all + vy_all)),
    #     )

    top_ax = ax_flat[0]

    # draw gridlines
    top_ax.grid(which="major", axis="both", linestyle="-", color="k", linewidth=1)
    top_ax.set_xticks([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    top_ax.set_yticks([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    top_ax.set_xticklabels([])
    top_ax.set_yticklabels([])

    top_ax.tick_params(which="minor", labelsize=24)
    top_ax.xaxis.set_major_formatter(ticker.NullFormatter())
    top_ax.yaxis.set_major_formatter(ticker.NullFormatter())

    # Customize minor tick labels
    top_ax.xaxis.set_minor_locator(ticker.FixedLocator([-2, -1, 0, 1, 2]))
    top_ax.xaxis.set_minor_formatter(ticker.FixedFormatter([-2, -1, 0, 1, 2]))
    top_ax.yaxis.set_minor_locator(ticker.FixedLocator([-2, -1, 0, 1, 2]))
    top_ax.yaxis.set_minor_formatter(ticker.FixedFormatter([-2, -1, 0, 1, 2]))
    for idx, phi in enumerate([-120, 0, 120]):
        top_ax.plot(
            np.sin(np.deg2rad(phi)),
            np.cos(np.deg2rad(phi)),
            "o",
            color="C2",
            markersize=20,
        )

    top_ax.set_xlabel("$x-x_0$ [cells]", fontsize=32, labelpad=10)
    top_ax.set_ylabel("$y-y_0$ [cells]", fontsize=32, labelpad=10)
    top_ax.set_title("VSC formation", fontsize=32, pad=10)
    top_ax.set_xlim(-2.5, 2.5)
    top_ax.set_ylim(-2.5, 2.5)
    top_ax.set_aspect("equal")
    # top_ax.set_xticklabels(["-2", "-1", "0", "1", "2", ""], minor=True)
    # top_ax.set_yticklabels(["-2", "-1", "0", "1", "2", ""], minor=True)
    top_ax.annotate("a)", (0.05, 0.90), xycoords="axes fraction", fontsize=32)

    plt.tight_layout()
    fig.savefig(
        wrkdir_DNR + "papu22/Figures/fig2.pdf",
        dpi=300,
    )
    plt.close(fig)


def weighted_propagation_velocity(runid, kind="non"):
    runids = ["ABA", "ABC", "AEA", "AEC"]

    rho_sw = [1e6, 3.3e6, 1e6, 3.3e6]
    v_sw = [750e3, 600e3, 750e3, 600e3]
    pdyn_sws = [m_p * rho_sw[idx] * v_sw[idx] * v_sw[idx] for idx in range(len(runids))]
    pdyn_sw = pdyn_sws[runids.index(runid)]

    bulkpath = jx.find_bulkpath(runid)

    if kind == "non":
        jet_ids = get_non_jets(runid)
    elif kind == "fcs":
        jet_ids = get_fcs_jets(runid)

    for jetid in jet_ids:
        print("Jet {} in run {}".format(jetid, runid))
        props = jio.PropReader(str(jetid).zfill(5), runid, transient="jet")

        t_list = props.get_times()
        cell_list = props.get_cells()

        if len(t_list) == 1:
            prop_v = [[np.nan, np.nan]]
            np.savetxt(
                wrkdir_DNR
                + "papu22/jet_prop_v_txts/{}_{}.txt".format(runid, str(jetid).zfill(5)),
                prop_v,
            )
            continue

        xlist = []
        ylist = []
        for idx, t in enumerate(t_list):
            fnr = int(t * 2)

            wsum = 0
            wxsum = 0
            wysum = 0
            vlsvobj = pt.vlsvfile.VlsvReader(
                bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
            )
            cells = cell_list[idx]
            tpdynavg = np.load(
                wrkdir_DNR + "tavg/" + runid + "/" + str(fnr) + "_pdyn.npy"
            )
            # print(type(cells))
            # print(cells)
            for c in cells:
                x, y, z = vlsvobj.get_cell_coordinates(c)
                pdyn = vlsvobj.read_variable("Pdyn", cellids=c)
                pdyn_tavg = tpdynavg[c - 1] + 1e-27
                w = pdyn / pdyn_tavg - 2

                wsum += w
                wxsum += w * x
                wysum += w * y

            xavg = wxsum / wsum
            yavg = wysum / wsum
            xlist.append(xavg / 1000)
            ylist.append(yavg / 1000)

        xlist = np.array(xlist)
        ylist = np.array(ylist)
        t_list = np.array(t_list)

        # prop_v = np.array(
        #     [
        #         np.ediff1d(xlist) / np.ediff1d(t_list),
        #         np.ediff1d(ylist) / np.ediff1d(t_list),
        #     ]
        # ).T
        prop_v = np.array([t_list, xlist, ylist]).T
        np.savetxt(
            wrkdir_DNR
            + "papu22/jet_prop_v_txts/{}_{}.txt".format(runid, str(jetid).zfill(5)),
            prop_v,
        )


def jet_counter(runid="all", cc_thresh=0.8):
    if runid == "all":
        runids = ["ABA", "ABC", "AEA", "AEC"]
    else:
        runids = [runid]

    flank_counter = 0
    antisunward_counter = 0
    fcs_counter = 0
    total_non = 0

    for run_id in runids:
        antisunward, flankward = auto_classifier(run_id, cross_corr_threshold=cc_thresh)
        fcs_jets = get_fcs_jets(run_id)
        non_jets = get_non_jets(run_id)

        flank_counter += len(flankward)
        antisunward_counter += len(antisunward)
        fcs_counter += fcs_jets.size
        total_non += non_jets.size

        print(
            "{}: {} {} {} {}".format(
                run_id, non_jets.size, len(flankward), len(antisunward), fcs_jets.size
            )
        )

    return (total_non, flank_counter, antisunward_counter, fcs_counter)


def auto_classifier(runid, threshold_angle=np.pi / 4, cross_corr_threshold=0.8):
    runids = ["ABA", "ABC", "AEA", "AEC"]

    rho_sw = [1e6, 3.3e6, 1e6, 3.3e6]
    v_sw = [750e3, 600e3, 750e3, 600e3]
    pdyn_sws = [m_p * rho_sw[idx] * v_sw[idx] * v_sw[idx] for idx in range(len(runids))]
    pdyn_sw = pdyn_sws[runids.index(runid)]
    v_sw_run = v_sw[runids.index(runid)] / 1e3

    non_ids = get_non_jets(runid)
    flankward_list = []
    antisunward_list = []

    for non_id in non_ids:
        if runid == "ABA" and non_id in [157, 257, 586, 800]:
            continue
        elif runid == "ABC" and non_id in [
            93,
            176,
            231,
            273,
            285,
            458,
            620,
            686,
            691,
            724,
            732,
        ]:
            continue
        elif runid == "AEA" and non_id in [
            878,
            1073,
            1251,
            1340,
            1354,
            1404,
            1498,
            1566,
            1592,
            2698,
        ]:
            continue
        elif runid == "AEC" and non_id in [
            59,
            64,
            266,
            332,
            430,
        ]:
            continue

        data_arr = np.load(
            wrkdir_DNR
            + "papu22/trifecta_txts/{}_{}.npy".format(runid, str(non_id).zfill(5))
        )

        tlist, xlist, ylist = np.loadtxt(
            wrkdir_DNR
            + "papu22/jet_prop_v_txts/{}_{}.txt".format(runid, str(non_id).zfill(5))
        ).T

        if tlist[-1] - tlist[0] == 0:
            continue

        res_arr = data_arr[0, -1, :]

        t0, x0, y0 = tlist[0], xlist[0], ylist[0]

        propvx = (xlist[tlist - t0 < 2.5][-1] - x0) / (
            tlist[tlist - t0 < 2.5][-1] - t0 + 1e-27
        )
        propvy = (ylist[tlist - t0 < 2.5][-1] - y0) / (
            tlist[tlist - t0 < 2.5][-1] - t0 + 1e-27
        )
        propvx_full = (xlist[-1] - x0) / (tlist[-1] - t0 + 1e-27)
        propvy_full = (ylist[-1] - y0) / (tlist[-1] - t0 + 1e-27)

        vnx, vny, vscx, vscy, vbx, vby = res_arr[:6]
        c = res_arr[12]
        # print(c)

        mod_arg_pvfull = [
            np.sqrt(propvx_full**2 + propvy_full**2),
            (np.arctan2(propvy_full, propvx_full) + 2 * np.pi) % (2 * np.pi),
        ]
        mod_arg_pv = [
            np.sqrt(propvx**2 + propvy**2),
            (np.arctan2(propvy, propvx) + 2 * np.pi) % (2 * np.pi),
        ]
        mod_arg_vn = [
            np.sqrt(vnx**2 + vny**2),
            (np.arctan2(vny, vnx) + 2 * np.pi) % (2 * np.pi),
        ]
        mod_arg_vsc = [
            np.sqrt(vscx**2 + vscy**2),
            (np.arctan2(vscy, vscx) + 2 * np.pi) % (2 * np.pi),
        ]
        mod_arg_vb = [
            np.sqrt(vbx**2 + vby**2),
            (np.arctan2(vby, vbx) + 2 * np.pi) % (2 * np.pi),
        ]

        if (
            ~np.isnan(mod_arg_vsc[0])
            # and mod_arg_vsc[0] < v_sw_run
            and np.abs(mod_arg_vsc[1] - np.pi) < threshold_angle
            and c >= cross_corr_threshold
        ):
            antisunward_list.append(non_id)
            continue
        elif (
            ~np.isnan(mod_arg_vsc[0])
            # and mod_arg_vsc[0] < v_sw_run
            and np.abs(mod_arg_vsc[1] - np.pi) >= threshold_angle
            and c >= cross_corr_threshold
        ):
            flankward_list.append(non_id)
            continue
        elif (
            ~np.isnan(mod_arg_pv[0])
            # and mod_arg_pv[0] < v_sw_run
            and np.abs(mod_arg_pv[1] - np.pi) < threshold_angle
        ):
            antisunward_list.append(non_id)
            continue
        elif (
            ~np.isnan(mod_arg_pv[0])
            # and mod_arg_pv[0] < v_sw_run
            and np.abs(mod_arg_pv[1] - np.pi) >= threshold_angle
        ):
            flankward_list.append(non_id)
            continue
        # elif (
        #     ~np.isnan(mod_arg_vn[0])
        #     and mod_arg_vn[0] < v_sw_run
        #     and np.abs(mod_arg_vn[1] - np.pi) < threshold_angle
        # ):
        #     antisunward_list.append(non_id)
        #     continue
        # elif (
        #     ~np.isnan(mod_arg_vn[0])
        #     and mod_arg_vn[0] < v_sw_run
        #     and np.abs(mod_arg_vn[1] - np.pi) >= threshold_angle
        # ):
        #     flankward_list.append(non_id)
        #     continue

    np.savetxt(
        wrkdir_DNR + "papu22/id_txts/" + "auto/{}_foreshock.txt".format(runid),
        antisunward_list,
        fmt="%d",
    )
    np.savetxt(
        wrkdir_DNR + "papu22/id_txts/" + "auto/{}_beam.txt".format(runid),
        flankward_list,
        fmt="%d",
    )

    return (antisunward_list, flankward_list)
