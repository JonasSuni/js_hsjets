outdir = ""

import pytools
import matplotlib.pyplot as plt
import numpy as np

r_e = 6.371e6


def ext_magpause(ax, XmeshXY, YmeshXY, pass_maps):

    vlsvobj = vlsvobj_g
    fileindex = fileindex_g
    time = time_g

    xarr = np.arange(10.1 * r_e, 7.0 * r_e, -1e6)
    slice = np.arange(xarr.size, dtype=float)
    theta = np.arange(0, 2 * np.pi + 0.001, np.pi / 30)
    Slice, Theta = np.meshgrid(slice, theta)
    psi = np.zeros_like(Slice, dtype=float)
    dvzdz = np.zeros_like(Slice, dtype=float)

    for irx, x in enumerate(xarr):
        coords = np.array([x, 0, 0], dtype=float)
        for itx, t in enumerate(theta):
            for r in np.arange(0, 10 * r_e, 1e6):
                y = np.cos(t) * r
                z = np.sin(t) * r
                coords[1] = y
                coords[2] = z
                cellid = vlsvobj.get_cellid(coords)
                bz = vlsvobj.read_variable("vg_b_vol", operator="z", cellids=cellid)
                if bz <= 0:
                    cellp = vlsvobj.get_cellid(coords + np.array([0, 0, 1e6]))
                    cellm = vlsvobj.get_cellid(coords - np.array([0, 0, 1e6]))
                    vz = vlsvobj.read_variable("vg_v", operator="z", cellids=cellid)
                    vzp = vlsvobj.read_variable("vg_v", operator="z", cellids=cellp)
                    vzm = vlsvobj.read_variable("vg_v", operator="z", cellids=cellm)
                    dvz = (vzp - vzm) / 1e3
                    dvzdz[itx, irx] = dvz
                    psi[itx, irx] = vz
                    break

    fig, ax = plt.subplots(1, 1)
    Xmesh = Slice * np.cos(Theta)
    Ymesh = Slice * np.sin(Theta)

    pcm = ax.pcolormesh(
        Xmesh, Ymesh, psi, cmap="seismic", vmin=-600e3, vmax=600e3, shading="gouraud"
    )
    cvz = ax.contour(
        Xmesh, Ymesh, dvzdz, [0.0], linewidths=1.5, colors="white", alpha=0.7
    )
    cf = ax.contour(Xmesh, Ymesh, psi, [0.0], linewidths=1.5, colors="black")
    fig.colorbar(pcm, ax=ax)
    ax.set_title("t = {} s".format(int(time)))
    plt.tight_layout()

    # CHANGE THE PATH HERE TO WHAT YOU WANT
    fig.savefig("magpause/magplot_{}.png".format(fileindex))
    plt.close(fig)


global vlsvobj_g
global fileindex_g
global time_g

# CHANGE IF YOU WANT TO PLOT A DIFFERENT TIME STEP
itr = 800
idx = str(itr).zfill(7)

# CHANGE PATH HERE IF YOU WANT TO PLOT A DIFFERENT RUN
f = pytools.vlsvfile.VlsvReader(
    "/wrk/group/spacephysics/vlasiator/3D/EGI/bulk/dense_cold_hall1e5_afterRestart374/bulk1."
    + idx
    + ".vlsv"
)

vlsvobj_g = f
fileindex_g = itr
time_g = f.read_parameter("time")

outputfile = "EGI_rho_vz_stream" + idx + ".png"

fig, ax_list = plt.subplots(1, 2)

pytools.plot.plot_colormap3dslice(
    vlsvobj=f,
    var="proton/vg_rho",
    normal="y",
    axes=ax_list[0],
    vmin=1e6,
    vmax=6e6,
    lin=1,
    boxre=[5, 15, -10, 10],
    scale=1.8,
    pass_vars=["vg_b_vol"],
    external=ext_magpause,
)

plt.tight_layout()

# fig.savefig(outdir+outputfile)
plt.close(fig)
