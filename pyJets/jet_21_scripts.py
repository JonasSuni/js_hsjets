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


def make_flap_plots():

    # for n in range(12, 27, 1):
    #    tail_sheet_jplot(xcut=n)

    for n in range(0, 11, 1):
        tail_sheet_jplot_y(xcut=n)


def tail_sheet_jplot(xcut=20):

    fnr_range = np.arange(1200, 1501, 1)
    y_arr = np.loadtxt(
        wrkdir_DNR + "Figures/sum21/sheet_txt/x{}/1200.txt".format(xcut)
    )[:, 0]
    val_mesh = np.array(
        [
            np.loadtxt(
                wrkdir_DNR + "Figures/sum21/sheet_txt/x{}/{}.txt".format(xcut, fnr)
            )[:, 1]
            for fnr in fnr_range
        ]
    )

    fig, ax = plt.subplots(1, 1)
    ax.grid()
    ax.set(
        xlabel="Y [Re]",
        ylabel="Time [s]",
        title="X = -{} Re".format(xcut),
        xlim=(-10, 10),
    )

    ax.pcolormesh(y_arr, fnr_range, val_mesh, shading="nearest", cmap="seismic")

    for fnr in fnr_range:
        ffjs = np.loadtxt(
            "/wrk/group/spacephysics/vlasiator/3D/EGI/visualizations/FFJ/dx_2e6_series/{}/ascii_rxpoints_tail_neighbourhood_1_extend_4_4_4_000{}.dat".format(
                fnr, fnr
            )
        )
        x, y, z = ffjs.T
        y_plot = y[np.abs(x + xcut) < 0.2]
        t_plot = np.ones_like(y_plot) * fnr
        ax.plot(y_plot, t_plot, "^", color="black", markersize=1)

    plt.tight_layout()
    # fig.savefig(wrkdir_DNR + "Figures/sum21/tail_sheet_jplot_x{}.pdf".format(xcut))
    fig.savefig(wrkdir_DNR + "Figures/sum21/tail_sheet_jplot_x{}.png".format(xcut))
    plt.close(fig)

    return None


def tail_sheet_jplot_y(xcut=20):

    fnr_range = np.arange(1200, 1501, 1)
    x_arr = np.loadtxt(
        wrkdir_DNR + "Figures/sum21/sheet_txt/y{}/1200.txt".format(xcut)
    )[:, 0]
    val_mesh = np.array(
        [
            np.loadtxt(
                wrkdir_DNR + "Figures/sum21/sheet_txt/y{}/{}.txt".format(xcut, fnr)
            )[:, 1]
            for fnr in fnr_range
        ]
    )

    fig, ax = plt.subplots(1, 1)
    ax.grid()
    ax.set(
        xlabel="X [Re]",
        ylabel="Time [s]",
        title="Y = {} Re".format(xcut),
        xlim=(-10, 10),
    )

    ax.pcolormesh(x_arr, fnr_range, val_mesh, shading="nearest", cmap="seismic")

    for fnr in fnr_range:
        ffjs = np.loadtxt(
            "/wrk/group/spacephysics/vlasiator/3D/EGI/visualizations/FFJ/dx_2e6_series/{}/ascii_rxpoints_tail_neighbourhood_1_extend_4_4_4_000{}.dat".format(
                fnr, fnr
            )
        )
        x, y, z = ffjs.T
        x_plot = x[np.abs(y - xcut) < 0.2]
        t_plot = np.ones_like(x_plot) * fnr
        ax.plot(x_plot, t_plot, "^", color="black", markersize=1)

    plt.tight_layout()
    # fig.savefig(wrkdir_DNR + "Figures/sum21/tail_sheet_jplot_x{}.pdf".format(xcut))
    fig.savefig(wrkdir_DNR + "Figures/sum21/tail_sheet_jplot_y{}.png".format(xcut))
    plt.close(fig)

    return None


def make_plots(cb=False):

    plt.ioff()

    outpath = wrkdir_DNR + "Figures/tektal/"

    # runs = ["ABA", "ABC", "AEA", "AEC", "BCQ", "BFD"]
    runs = ["BCQ", "BFD"]
    # vars = ["Pdyn", "rho", "v", "B"]
    cmaps = ["warhol", "magma", "jet", "plasma", "viridis"]

    # bulk_paths = [
    #     "/wrk/group/spacephysics/vlasiator/2D/ABA/bulk/bulk.0001000.vlsv",
    #     "/wrk/group/spacephysics/vlasiator/2D/ABC/bulk/bulk.0001000.vlsv",
    #     "/wrk/group/spacephysics/vlasiator/2D/AEA/round_3_boundary_sw/bulk.0001000.vlsv",
    #     "/wrk/group/spacephysics/vlasiator/2D/AEC/bulk/bulk.0001000.vlsv",
    #     "/wrk/group/spacephysics/vlasiator/2D/BCQ/bulk/bulk.0002000.vlsv",
    #     "/wrk/group/spacephysics/vlasiator/2D/BFD/bulk/bulk.0002000.vlsv",
    # ]
    bulk_paths = [
        "/wrk/group/spacephysics/vlasiator/2D/BCQ/bulk/bulk.0002000.vlsv",
        "/wrk/group/spacephysics/vlasiator/2D/BFD/bulk/bulk.0002000.vlsv",
    ]
    if cb:
        bulk_paths = [
            "/wrk/group/spacephysics/vlasiator/2D/BCQ/bulk/bulk.0002000.vlsv",
        ]
        vars = ["v"]
        cmaps = ["warhol"]
        runs = ["BCQ"]

    for run in runs:
        bulkpath = bulk_paths[runs.index(run)]
        for var in vars:
            for cm in cmaps:
                outdir = wrkdir_DNR + "Figures/tektal/{}/{}/".format(run, var)
                if not os.path.exists(outdir):
                    try:
                        os.makedirs(outdir)
                    except OSError:
                        pass

                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.axis("off")
                pt.plot.plot_colormap(
                    filename=bulkpath,
                    var=var,
                    nocb=True,
                    lin=1,
                    colormap=cm,
                    noxlabels=True,
                    noylabels=True,
                    noborder=True,
                    title="",
                    Earth=1,
                    axes=ax,
                )

                fig.savefig(
                    outdir + "{}_{}_{}.png".format(run, var, cm),
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=300,
                )
                plt.close(fig)
                if cb:
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    ax.axis("off")
                    pt.plot.plot_colormap(
                        filename=bulkpath,
                        var=var,
                        internalcb=True,
                        lin=1,
                        colormap=cm,
                        noxlabels=True,
                        noylabels=True,
                        noborder=True,
                        title="",
                        Earth=1,
                        axes=ax,
                    )

                    fig.savefig(
                        outdir + "{}_{}_{}_scale.png".format(run, var, cm),
                        bbox_inches="tight",
                        pad_inches=0,
                        dpi=300,
                    )

    return None

