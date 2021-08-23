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

    #    for n in range(12, 27, 1):
    #        tail_sheet_jplot(xcut=n)

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

    im = ax.pcolormesh(
        y_arr, fnr_range, val_mesh, shading="nearest", cmap="seismic", vmin=-1, vmax=1
    )
    fig.colorbar(im, ax=ax)

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
        xlim=(-25, -10),
    )

    im = ax.pcolormesh(
        x_arr, fnr_range, val_mesh, shading="nearest", cmap="seismic", vmin=-1, vmax=1
    )
    fig.colorbar(im, ax=ax)

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


def vfield3_dot(a, b):
    """ Calculates dot product of vectors a and b in 3D vector field
    """

    return (
        a[:, :, :, 0] * b[:, :, :, 0]
        + a[:, :, :, 1] * b[:, :, :, 1]
        + a[:, :, :, 2] * b[:, :, :, 2]
    )


def vfield3_normalise(a):

    amag = np.linalg.norm(a, axis=-1)

    resx = a[:, :, :, 0] / amag
    resy = a[:, :, :, 1] / amag
    resz = a[:, :, :, 2] / amag

    return np.stack((resx, resy, resz), axis=-1)


def vfield3_matder(a, b, dr):
    """ Calculates material derivative of 3D vector fields a and b
    """

    bx = b[:, :, :, 0]
    by = b[:, :, :, 1]
    bz = b[:, :, :, 2]

    grad_bx = vfield3_grad(bx, dr)
    grad_by = vfield3_grad(by, dr)
    grad_bz = vfield3_grad(bz, dr)

    resx = vfield3_dot(a, grad_bx)
    resy = vfield3_dot(a, grad_by)
    resz = vfield3_dot(a, grad_bz)

    return np.stack((resx, resy, resz), axis=-1)


def vfield3_grad(a, dr):
    """ Calculates gradient of 3D scalar field a using central difference
    """

    gradx = (np.roll(a, -1, 0) - np.roll(a, 1, 0)) / 2.0 / dr
    grady = (np.roll(a, -1, 1) - np.roll(a, 1, 1)) / 2.0 / dr
    gradz = (np.roll(a, -1, 2) - np.roll(a, 1, 2)) / 2.0 / dr

    return np.stack((gradx, grady, gradz), axis=-1)


def ballooning_crit(B, P, beta):

    dr = 1000e3

    # Bmag = np.linalg.norm(B, axis=-1)

    b = vfield3_normalise(B)

    n = vfield3_matder(b, b, dr)
    nnorm = vfield3_normalise(n)

    kappaP = vfield3_dot(nnorm, vfield3_grad(P, dr)) / (P + 1e-27)
    # kappaB = vfield3_dot(n, vfield3_grad(Bmag, dr)) / Bmag
    kappaC = vfield3_dot(nnorm, n)

    return (2 + beta) / 4.0 * kappaP / (kappaC + 1e-27)


def plot_ballooning(tstep=1274, xcut=15):

    bulkfile = "/wrk/group/spacephysics/vlasiator/3D/EGI/bulk/dense_cold_hall1e5_afterRestart374/bulk1.{}.vlsv".format(
        str(tstep).zfill(7)
    )

    global zymesh_size
    global B_arr
    global P_arr
    global beta_arr
    global idx
    global ballooning_arr

    zymesh_size = [1, 2, 3]

    pt.plot.plot_colormap3dslice(
        filename=bulkfile,
        var="proton/vg_rho",
        draw=1,
        external=ext_get_meshsize,
        pass_vars=["vg_b_vol", "CellID"],
        boxre=[-20, -12, -1.5, 1.5],
        normal="y",
        cutpoint=-1 * xcut * r_e,
    )

    B_arr = np.empty((zymesh_size[0], 3, zymesh_size[1], zymesh_size[2]), dtype=float)
    P_arr = np.empty((zymesh_size[0], 3, zymesh_size[1]), dtype=float)
    beta_arr = np.empty((zymesh_size[0], 3, zymesh_size[1]), dtype=float)

    ballooning_arr = ballooning_crit(B_arr, P_arr, beta_arr)

    for idx in [0, 1, 2]:
        pt.plot.plot_colormap3dslice(
            filename=bulkfile,
            var="proton/vg_rho",
            draw=1,
            external=ext_get_cuts,
            pass_vars=["vg_b_vol", "proton/vg_pressure", "proton/vg_beta", "CellID"],
            boxre=[-20, -12, -1.5, 1.5],
            normal="y",
            cutpoint=-1 * xcut * r_e + 1000e3 * (idx - 1),
        )

    pt.plot.plot_colormap3dslice(
        filename=bulkfile,
        outputfile=wrkdir_DNR
        + "Figures/sum21/ballooning_t{}_y{}.png".format(tstep, xcut),
        var="vg_b_vol",
        colormap="seismic",
        operator="x",
        vmin=-2e-8,
        vmax=2e-8,
        lin=1,
        external=ext_plot_ballooning,
        pass_vars=[
            "vg_b_vol",
            "proton/vg_pressure",
            "proton/vg_beta",
            "proton/vg_v",
            "CellID",
        ],
        boxre=[-20, -12, -1.5, 1.5],
        normal="y",
        cutpoint=-1 * xcut * r_e,
    )

    return None


def ext_get_meshsize(ax, XmeshXY, YmeshXY, pass_maps):

    global zymesh_size

    B = pass_maps["vg_b_vol"]
    zymesh_size[0] = B.shape[0]
    zymesh_size[1] = B.shape[1]
    zymesh_size[2] = B.shape[2]

    return None


def ext_get_cuts(ax, XmeshXY, YmeshXY, pass_maps):

    global B_arr
    global P_arr
    global beta_arr

    B = pass_maps["vg_b_vol"]
    P = pass_maps["proton/vg_pressure"]
    beta = pass_maps["proton/vg_beta"]

    B_arr[:, idx, :, :] = B
    P_arr[:, idx, :] = P
    beta_arr[:, idx, :] = beta

    return None


def ext_plot_ballooning(ax, XmeshXY, YmeshXY, pass_maps):

    global ballooning_arr

    B = pass_maps["vg_b_vol"]
    P = pass_maps["proton/vg_pressure"]
    beta = pass_maps["proton/vg_beta"]
    v = pass_maps["proton/vg_v"]

    vx = v[:, :, 0]

    balloon = ballooning_arr[:, 1, :]
    balloon_masked = np.ma.masked_array(balloon, balloon < 1)
    balloon_masked.mask[beta > 2] = True
    balloon_masked.mask[balloon > 900000] = True

    ax.contour(XmeshXY, YmeshXY, vx, 0, colors="blue", linewidths=1.2)

    ax.pcolormesh(
        XmeshXY,
        YmeshXY,
        balloon_masked,
        vmin=1,
        vmax=3,
        cmap="YlOrBr",
        shading="nearest",
    )

    return None

