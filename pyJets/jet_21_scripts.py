import sys
import matplotlib.style
import matplotlib as mpl
import jet_aux as jx
import scipy.ndimage

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
mu_0 = 1.25663706212e-06

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
    """Calculates dot product of vectors a and b in 3D vector field"""

    return (
        a[:, :, :, 0] * b[:, :, :, 0]
        + a[:, :, :, 1] * b[:, :, :, 1]
        + a[:, :, :, 2] * b[:, :, :, 2]
    )


def vfield2_dot(a, b):
    """Calculates dot product of vectors a and b in 2D vector field"""

    return a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1] + a[:, :, 2] * b[:, :, 2]


def vfield2_normalise(a):

    amag = np.linalg.norm(a, axis=-1)

    resx = a[:, :, 0] / amag
    resy = a[:, :, 1] / amag
    resz = a[:, :, 2] / amag

    return np.stack((resx, resy, resz), axis=-1)


def vfield3_normalise(a):

    amag = np.linalg.norm(a, axis=-1)

    resx = a[:, :, :, 0] / amag
    resy = a[:, :, :, 1] / amag
    resz = a[:, :, :, 2] / amag

    return np.stack((resx, resy, resz), axis=-1)


def vfield3_matder(a, b, dr, normal="y"):
    """Calculates material derivative of 3D vector fields a and b"""

    bx = b[:, :, :, 0]
    by = b[:, :, :, 1]
    bz = b[:, :, :, 2]

    grad_bx = vfield3_grad(bx, dr, normal=normal)
    grad_by = vfield3_grad(by, dr, normal=normal)
    grad_bz = vfield3_grad(bz, dr, normal=normal)

    resx = vfield3_dot(a, grad_bx)
    resy = vfield3_dot(a, grad_by)
    resz = vfield3_dot(a, grad_bz)

    return np.stack((resx, resy, resz), axis=-1)


def vfield3_grad(a, dr, normal="y"):
    """Calculates gradient of 3D scalar field a using central difference"""

    ax_order = [[0, 2, 1], [2, 1, 0], [1, 0, 2]][["x", "y", "z"].index(normal)]

    gradx = (np.roll(a, -1, ax_order[0]) - np.roll(a, 1, ax_order[0])) / 2.0 / dr
    grady = (np.roll(a, -1, ax_order[1]) - np.roll(a, 1, ax_order[1])) / 2.0 / dr
    gradz = (np.roll(a, -1, ax_order[2]) - np.roll(a, 1, ax_order[2])) / 2.0 / dr

    return np.stack((gradx, grady, gradz), axis=-1)

    # return np.stack(np.gradient(a, dr,dr,dr), axis=-1)


def vfield3_grad_stencil5(a, dr, normal="y"):
    """Calculates gradient of 3D scalar field a using a 5 point stencil"""

    ax_order = [[0, 2, 1], [2, 1, 0], [1, 0, 2]][["x", "y", "z"].index(normal)]

    gradx = (
        (
            -np.roll(a, -2, ax_order[0])
            + 8 * np.roll(a, -1, ax_order[0])
            - 8 * np.roll(a, 1, ax_order[0])
            + np.roll(a, 2, ax_order[0])
        )
        / 12.0
        / dr
    )
    grady = (
        (
            -np.roll(a, -2, ax_order[1])
            + 8 * np.roll(a, -1, ax_order[1])
            - 8 * np.roll(a, 1, ax_order[1])
            + np.roll(a, 2, ax_order[1])
        )
        / 12.0
        / dr
    )
    gradz = (
        (
            -np.roll(a, -2, ax_order[2])
            + 8 * np.roll(a, -1, ax_order[2])
            - 8 * np.roll(a, 1, ax_order[2])
            + np.roll(a, 2, ax_order[2])
        )
        / 12.0
        / dr
    )


def vfield3_curl(a, dr, normal="y"):
    """Calculates curl of 3D vector field"""

    grad_ax = vfield3_grad(a[:, :, :, 0], dr, normal=normal)
    grad_ay = vfield3_grad(a[:, :, :, 1], dr, normal=normal)
    grad_az = vfield3_grad(a[:, :, :, 2], dr, normal=normal)

    resx = grad_az[:, :, :, 1] - grad_ay[:, :, :, 2]
    resy = grad_ax[:, :, :, 2] - grad_az[:, :, :, 0]
    resz = grad_ay[:, :, :, 0] - grad_ax[:, :, :, 1]

    return np.stack((resx, resy, resz), axis=-1)


def ballooning_crit(B, P, beta, dr=1000e3, normal="y"):

    # Bmag = np.linalg.norm(B, axis=-1)

    b = vfield3_normalise(B)

    n = vfield3_matder(b, b, dr, normal=normal)

    nnorm = vfield3_normalise(n)

    kappaP = vfield3_dot(nnorm, vfield3_grad(P, dr, normal=normal)) / P
    # kappaB = vfield3_dot(n, vfield3_grad(Bmag, dr)) / Bmag
    kappaC = vfield3_dot(nnorm, n)

    balloon = (2 + beta) / 4.0 * kappaP / (kappaC + 1e-27)

    return (balloon, nnorm, kappaC)


def plot_ballooning(
    tstep=1274,
    cut=15,
    normal="y",
    boxre=[-19, -9, -1.5, 1.5],
    dr=1000e3,
    op="mag",
    write_txt=False,
):

    bulkfile = "/wrk/group/spacephysics/vlasiator/3D/EGI/bulk/dense_cold_hall1e5_afterRestart374/bulk1.{}.vlsv".format(
        str(tstep).zfill(7)
    )

    global zymesh_size
    global B_arr
    global P_arr
    global beta_arr
    global idx_g
    global ballooning_arr, nnorm_arr, kappaC_arr, J_arr
    global normal_g, tstep_g, cut_g
    global op_g, zoom_g, write_txt_g

    op_g = op
    zoom_g = 1000e3 / dr
    write_txt_g = write_txt

    normal_g = normal
    tstep_g = tstep
    cut_g = cut

    zymesh_size = [1, 2, 3]

    pt.plot.plot_colormap3dslice(
        filename=bulkfile,
        var="proton/vg_rho",
        draw=1,
        external=ext_get_meshsize,
        pass_vars=["vg_b_vol", "CellID"],
        boxre=boxre,
        normal=normal,
        cutpoint=-1 * cut * r_e,
    )

    if normal == "x":
        B_arr = np.empty(
            (3, zymesh_size[0], zymesh_size[1], zymesh_size[2]), dtype=float
        )
        P_arr = np.empty((3, zymesh_size[0], zymesh_size[1]), dtype=float)
        beta_arr = np.empty((3, zymesh_size[0], zymesh_size[1]), dtype=float)
    elif normal == "y":
        B_arr = np.empty(
            (zymesh_size[0], 3, zymesh_size[1], zymesh_size[2]), dtype=float
        )
        P_arr = np.empty((zymesh_size[0], 3, zymesh_size[1]), dtype=float)
        beta_arr = np.empty((zymesh_size[0], 3, zymesh_size[1]), dtype=float)
    elif normal == "z":
        B_arr = np.empty(
            (zymesh_size[0], zymesh_size[1], 3, zymesh_size[2]), dtype=float
        )
        P_arr = np.empty((zymesh_size[0], zymesh_size[1], 3), dtype=float)
        beta_arr = np.empty((zymesh_size[0], zymesh_size[1], 3), dtype=float)

    for idx in [0, 1, 2]:
        idx_g = idx
        pt.plot.plot_colormap3dslice(
            filename=bulkfile,
            var="proton/vg_rho",
            draw=1,
            external=ext_get_cuts,
            pass_vars=["vg_b_vol", "proton/vg_pressure", "proton/vg_beta", "CellID"],
            boxre=boxre,
            normal=normal,
            cutpoint=-1 * cut * r_e + dr * (idx - 1),
        )

    ballooning_arr, nnorm_arr, kappaC_arr = ballooning_crit(
        B_arr, P_arr, beta_arr, dr=dr, normal=normal
    )
    J_arr = vfield3_curl(B_arr, dr, normal=normal) / mu_0

    pt.plot.plot_colormap3dslice(
        filename=bulkfile,
        outputfile=wrkdir_DNR
        + "Figures/sum21/balloon/ballooning_t{}_{}{}_{}.png".format(
            tstep, cut, normal, op
        ),
        var="proton/vg_pressure",
        colormap="viridis",
        vmax=1e-10,
        lin=1,
        external=ext_plot_ballooning,
        pass_vars=[
            "vg_b_vol",
            "proton/vg_pressure",
            "proton/vg_beta",
            "proton/vg_v",
            "CellID",
        ],
        boxre=boxre,
        normal=normal,
        cutpoint=-1 * cut * r_e,
        nocb=True,
        scale=0.8,
        tickinterval=2.0,
    )

    return None


def ext_get_meshsize(ax, XmeshXY, YmeshXY, pass_maps):

    global zymesh_size

    B = pass_maps["vg_b_vol"]

    B = scipy.ndimage.zoom(B, (zoom_g, zoom_g, 1), mode="grid-constant", grid_mode=True)

    zymesh_size[0] = B.shape[0]
    zymesh_size[1] = B.shape[1]
    zymesh_size[2] = B.shape[2]

    return None


def ext_get_cuts(ax, XmeshXY, YmeshXY, pass_maps):

    idx = idx_g

    global B_arr
    global P_arr
    global beta_arr

    B = scipy.ndimage.zoom(
        pass_maps["vg_b_vol"], (zoom_g, zoom_g, 1), mode="grid-constant", grid_mode=True
    )
    P = scipy.ndimage.zoom(
        pass_maps["proton/vg_pressure"], zoom_g, mode="grid-constant", grid_mode=True
    )
    beta = scipy.ndimage.zoom(
        pass_maps["proton/vg_beta"], zoom_g, mode="grid-constant", grid_mode=True
    )

    if normal_g == "x":
        B_arr[idx, :, :, :] = B
        P_arr[idx, :, :] = P
        beta_arr[idx, :, :] = beta
    elif normal_g == "y":
        B_arr[:, idx, :, :] = B
        P_arr[:, idx, :] = P
        beta_arr[:, idx, :] = beta
    elif normal_g == "z":
        B_arr[:, :, idx, :] = B
        P_arr[:, :, idx] = P
        beta_arr[:, :, idx] = beta

    return None


def ext_plot_ballooning(ax, XmeshXY, YmeshXY, pass_maps):

    global ballooning_arr

    B = scipy.ndimage.zoom(
        pass_maps["vg_b_vol"], (zoom_g, zoom_g, 1), mode="grid-constant", grid_mode=True
    )
    P = scipy.ndimage.zoom(
        pass_maps["proton/vg_pressure"], zoom_g, mode="grid-constant", grid_mode=True
    )
    beta = scipy.ndimage.zoom(
        pass_maps["proton/vg_beta"], zoom_g, mode="grid-constant", grid_mode=True
    )
    v = scipy.ndimage.zoom(
        pass_maps["proton/vg_v"],
        (zoom_g, zoom_g, 1),
        mode="grid-constant",
        grid_mode=True,
    )

    XmeshXY = scipy.ndimage.zoom(XmeshXY, zoom_g, mode="grid-constant", grid_mode=True)
    YmeshXY = scipy.ndimage.zoom(YmeshXY, zoom_g, mode="grid-constant", grid_mode=True)

    vx = v[:, :, 0]
    Bx = B[:, :, 0]

    if normal_g == "x":
        balloon = ballooning_arr[1, :, :]
        J = J_arr[1, :, :, :]
        U = nnorm_arr[1, :, :, 0]
        V = nnorm_arr[1, :, :, 2]
        C = nnorm_arr[1, :, :, 1]

        BU = B[:, :, 1]
        BV = B[:, :, 2]
    elif normal_g == "y":
        balloon = ballooning_arr[:, 1, :]
        J = J_arr[:, 1, :, :]
        U = nnorm_arr[:, 1, :, 0]
        V = nnorm_arr[:, 1, :, 2]
        C = nnorm_arr[:, 1, :, 1]

        BU = B[:, :, 0]
        BV = B[:, :, 2]
    elif normal_g == "z":
        balloon = ballooning_arr[:, :, 1]
        J = J_arr[:, :, 1, :]
        U = nnorm_arr[:, :, 1, 0]
        V = nnorm_arr[:, :, 1, 2]
        C = nnorm_arr[:, :, 1, 1]

        BU = B[:, :, 0]
        BV = B[:, :, 1]

    balloon_masked = np.ma.masked_less_equal(balloon, 1)
    balloon_masked.mask[beta >= 2] = True
    # balloon_masked.mask[balloon > 1e30] = True

    if op_g == "mag":
        Jmag = np.linalg.norm(J, axis=-1) / 1.0e-9
        J_im = ax.pcolormesh(
            XmeshXY,
            YmeshXY,
            Jmag,
            vmin=2,
            vmax=6,
            cmap="viridis_r",
            shading="nearest",
        )
    elif op_g == "fa":
        b = vfield2_normalise(B)
        Jfa = vfield2_dot(J, b) / 1.0e-9
        J_im = ax.pcolormesh(
            XmeshXY,
            YmeshXY,
            Jfa,
            vmin=-1,
            vmax=1,
            cmap="seismic",
            shading="nearest",
        )

    cax1 = ax.inset_axes([1.04, 0, 0.05, 1])
    # cax2 = ax.inset_axes([1.3, 0, 0.05, 1])

    Jcb = plt.colorbar(J_im, cax=cax1)
    Jcb.ax.tick_params(labelsize=6)
    Jcb.set_label("J [nA/m$^2$]", size=6)

    ax.contour(XmeshXY, YmeshXY, vx, 0, colors="cyan", linewidths=0.6)
    # ax.contour(XmeshXY, YmeshXY, Bx, 0, colors="red", linewidths=0.4)

    # Balloon_im = ax.pcolormesh(
    #     XmeshXY,
    #     YmeshXY,
    #     balloon_masked,
    #     vmin=1,
    #     vmax=10,
    #     cmap="YlOrBr",
    #     shading="nearest",
    # )

    if op_g == "mag":
        ax.contour(
            XmeshXY,
            YmeshXY,
            balloon_masked.mask.astype(int),
            0,
            colors="magenta",
            linewidths=0.6,
        )

    # Bcb = plt.colorbar(Balloon_im, cax=cax2)
    # Bcb.ax.tick_params(labelsize=6)
    # Bcb.set_label("Ballooning", size=6, loc="bottom")

    if normal_g == "y":
        ax.streamplot(
            XmeshXY,
            YmeshXY,
            BU,
            BV,
            linewidth=0.4,
            arrowstyle="-",
            color="gray",
            density=1.5,
        )

    if normal_g == "x" and op_g == "fa" and write_txt_g:
        msk = np.ones_like(XmeshXY).astype(bool)
        msk[XmeshXY < -1] = False
        msk[XmeshXY > 2] = False
        msk[YmeshXY < 3] = False
        msk[YmeshXY > 5] = False

        Jfa_min = np.min(Jfa[msk])
        ymin = XmeshXY[msk][Jfa[msk] == Jfa_min][0]
        zmin = YmeshXY[msk][Jfa[msk] == Jfa_min][0]
        Jfa_med = np.median(Jfa[msk])

        txt_out = np.array([ymin, zmin, Jfa_min, Jfa_med])

        np.savetxt(
            "/wrk/users/jesuni/Figures/sum21/fac_txt/x{}_t{}".format(cut_g, tstep_g),
            txt_out,
        )

    if normal_g == "x" and op_g == "mag" and write_txt_g:
        Bxmag = np.abs(Bx)
        Jsheet = np.array(
            [Jmag[idy, idx] for idx, idy in enumerate(np.argmin(Bxmag, axis=0))]
        )
        Balloonsheet = np.array(
            [
                balloon_masked[idy, idx]
                for idx, idy in enumerate(np.argmin(Bxmag, axis=0))
            ]
        )
        txt_out = np.array([XmeshXY[0], Jsheet, Balloonsheet]).T
        np.savetxt(
            "/wrk/users/jesuni/Figures/sum21/balloon_txt/x{}_t{}".format(
                cut_g, tstep_g
            ),
            txt_out,
        )

    # plt.colorbar(Balloon_im, ax=ax, label="Ballooning")

    # ax.quiver(XmeshXY, YmeshXY, U, V, C, cmap="seismic")

    return None


def fac_migration_plot():

    fnr_range = np.arange(1250, 1501, 1)

    yarr = np.zeros(fnr_range.shape, dtype=float)
    zarr = np.zeros(fnr_range.shape, dtype=float)
    min_arr = np.zeros(fnr_range.shape, dtype=float)
    med_arr = np.zeros(fnr_range.shape, dtype=float)

    for idx, fnr in fnr_range:
        data = np.loadtxt(wrkdir_DNR + "Figures/sum21/fac_txt/x{}_t{}".format(5.5, fnr))
        yarr[idx] = data[0]
        zarr[idx] = data[1]
        min_arr[idx] = data[2]
        med_arr[idx] = data[3]

    fig, axs = plt.subplots(4, 1)
    axs[0].plot(fnr_range, yarr)
    axs[1].plot(fnr_range, zarr)
    axs[2].plot(fnr_range, min_arr)
    axs[3].plot(fnr_range, med_arr)

    plt.tight_layout()
    fig.savefig(wrkdir_DNR + "Figures/sum21/fac_plot.png")
    plt.close(fig)

    return None


def tail_sheet_jplot_balloon(xcut=14):

    fnr_range = np.arange(1250, 1501, 1)
    y_arr = np.loadtxt(
        wrkdir_DNR + "Figures/sum21/balloon_txt/x{}_t{}".format(xcut, 1250)
    )[:, 0]
    J_mesh = np.array(
        [
            np.loadtxt(
                wrkdir_DNR + "Figures/sum21/balloon_txt/x{}_t{}".format(xcut, fnr)
            )[:, 1]
            for fnr in fnr_range
        ]
    )

    balloon_mesh = np.array(
        [
            np.loadtxt(
                wrkdir_DNR + "Figures/sum21/balloon_txt/x{}_t{}".format(xcut, fnr)
            )[:, 2]
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
        y_arr,
        fnr_range,
        J_mesh,
        shading="nearest",
        cmap="viridis_r",
        vmin=2e-9,
        vmax=6e-9,
    )
    fig.colorbar(im, ax=ax, label="$J_{mag}$")

    balloon_im = ax.pcolormesh(
        y_arr,
        fnr_range,
        balloon_mesh,
        shading="nearest",
        cmap="YlOrBr",
        vmin=1,
        vmax=10,
    )

    # fig.colorbar(balloon_im, ax=ax, label="Balloon")

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
    fig.savefig(
        wrkdir_DNR + "Figures/sum21/tail_sheet_jplot_balloon_x{}.png".format(xcut)
    )
    plt.close(fig)

    return None
