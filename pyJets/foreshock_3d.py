# from operator import ge
# import sys
# import matplotlib.style
# import matplotlib as mpl
# import jet_aux as jx
from pyJets.jet_aux import (
    CB_color_cycle,
    find_bulkpath,
    restrict_area,
    get_neighs,
    get_neighs_asym,
    xyz_reconstruct,
    bow_shock_jonas,
    mag_pause_jonas,
    BS_xy,
    MP_xy,
)
from pyJets.jet_analyser import get_cell_volume, sw_par_dict
import pytools as pt
import os
import sys
from random import choice
from copy import deepcopy

# import scipy
# import scipy.linalg
from scipy.linalg import eig
from scipy.fft import rfft2
from scipy.signal import butter, sosfilt, cwt, morlet2
from scipy.ndimage import uniform_filter1d
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

mpl.rcParams["hatch.linewidth"] = 0.1

# from matplotlib.ticker import MaxNLocator
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

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

wrkdir_DNR = wrkdir_DNR + "3d_foreshock/"


def resol_vdf(resol, cellid, box=[-6e6, 6e6, -6e6, 6e6]):

    fig, ax_list = plt.subplots(
        2,
        2,
        figsize=(8, 8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    ipshock_path = os.environ["WRK"] + "/ipshock_FIE/"
    ax_flat = ax_list.flatten()

    ax_flat[-1].set_axis_off()

    filename = sorted(os.listdir(ipshock_path + "{}/restart/".format(resol)))[-1]
    vobj = pt.vlsvfile.VlsvReader(
        ipshock_path + "{}/restart/{}".format(resol, filename)
    )

    pt.plot.plot_vdf(
        vlsvobj=vobj,
        cellids=[cellid],
        axes=ax_flat[0],
        # fmin=1e-18,
        xy=True,
        setThreshold=1e-18,
        box=box,
        # fmax=1e-5,
        # slicethick=1,
        # reducer="average",
        slicethick=0,
    )
    pt.plot.plot_vdf(
        vlsvobj=vobj,
        cellids=[cellid],
        axes=ax_flat[1],
        # fmin=1e-18,
        xz=True,
        setThreshold=1e-18,
        box=box,
        # fmax=1e-5,
        # slicethick=1,
        # reducer="average",
        slicethick=0,
    )
    pt.plot.plot_vdf(
        vlsvobj=vobj,
        cellids=[cellid],
        axes=ax_flat[2],
        # fmin=1e-18,
        yz=True,
        setThreshold=1e-18,
        box=box,
        # fmax=1e-5,
        # slicethick=1,
        # reducer="average",
        slicethick=0,
    )
    fig.suptitle("res = {}, cellid = {}".format(resol, cellid))
    ax_list.flatten()[-1].set_axis_off()
    res = str(resol).replace("/", "_")
    fig.savefig(wrkdir_DNR + "Figs/vdf_r{}_c{}.png".format(res, cellid))
    plt.close(fig)


def ipshock_1d_vdf(x0=20, cutoff=1e-18, resols=[250, 300, 500, 1000, 2000, 4000, 8000]):

    ipshock_path = os.environ["WRK"] + "/ipshock_FIE/"

    fig, ax_list = plt.subplots(
        3,
        3,
        figsize=(12, 12),
        constrained_layout=True,
    )

    ax_flat = ax_list.flatten()

    for idx, r in enumerate(resols):
        ax = ax_flat[idx]
        filename = sorted(os.listdir(ipshock_path + "{}/restart/".format(r)))[-1]
        vobj = pt.vlsvfile.VlsvReader(
            ipshock_path + "{}/restart/{}".format(r, filename)
        )
        pt.plot.plot_vdf(
            vlsvobj=vobj,
            coordre=[x0, 0, 0],
            axes=ax,
            fmin=1e-18,
            bpara=True,
            setThreshold=cutoff,
            box=[-6e6, 6e6, -6e6, 6e6],
            fmax=1e-5,
            slicethick=1,
            reducer="average",
        )
        pt_title = ax.get_title()
        ax.set_title("{}\nres: {}".format(pt_title, r))
    fig.suptitle("X = {} RE, threshold = {}".format(x0, str(cutoff)))
    for idx in range(len(resols), 9):
        ax_flat[idx].set_axis_off()

    resols = "_".join([r.replace("/", "_") for r in resols])
    fig.savefig(wrkdir_DNR + "Figs/vdf_comp_x{}_f{}_r{}.png".format(x0, cutoff, resols))
    plt.close(fig)


def ipshock_1d_amr_target(fnr=100, a1=0.4, a2=1, resol="v30/8000"):

    ipshock_path = os.environ["WRK"] + "/ipshock_FIE/"

    # var_list = [
    #     "proton/vg_rho",
    #     "proton/vg_rho_nonthermal",
    #     "proton/vg_v",
    #     "proton/vg_v_nonthermal",
    #     "vg_b_vol",
    # ]
    # ylabels = [
    #     "$\\rho~[\mathrm{cm}^{-3}]$",
    #     "$\\rho_\mathrm{non-th}~[\mathrm{cm}^{-3}]$",
    #     "$v_x~[\mathrm{km/s}]$",
    #     "$v_{\mathrm{non-th},x}~[\mathrm{km/s}]$",
    #     "$B_y~[\mathrm{nT}]$",
    # ]
    # scales = [1e-6, 1e-6, 1e-3, 1e-3, 1e9]
    # miny = [None, 10**-4, -1000, -500, -5]
    # maxy = [5, 5, 0, 1000, 5]
    # op = ["pass", "pass", "x", "x", "y"]
    # yscales = ["log", "log", "linear", "linear", "linear"]

    fig, ax_list = plt.subplots(
        2, 1, figsize=(8, 6), constrained_layout=True, sharex=True
    )

    vobj = pt.vlsvfile.VlsvReader(
        ipshock_path + "{}/bulk/bulk.{}.vlsv".format(resol, str(fnr).zfill(7))
    )
    cellids = vobj.read_variable("CellID")
    x_arr = np.array([vobj.get_cell_coordinates(c)[0] for c in np.sort(cellids)]) / r_e
    alpha1_arr = vobj.read_variable("vg_amr_alpha1")[np.argsort(cellids)]
    alpha2_arr = vobj.read_variable("vg_amr_alpha2")[np.argsort(cellids)]

    alpha1_target = np.ceil(np.log2(alpha1_arr + 1e-30) - np.log2(a1))
    alpha2_target = np.ceil(np.log2(alpha2_arr + 1e-30) - np.log2(a2))

    ax_list[0].plot(x_arr, alpha1_target, color="k")
    ax_list[1].plot(x_arr, alpha2_target, color="k")

    for idx, ax in enumerate(ax_list):
        ax.grid()
        ax.set_xlim(-20, 40)
        ax.set_ylabel("AMR alpha{} target".format(idx + 1))
        ax.set_ylim(-3, 3)
    ax_list[-1].set_xlabel("X [RE]")
    ax_list[0].set_title(
        "t = {}s, r: {}, a1: {}, a2: {}".format(fnr * 5, resol, a1, a2)
    )

    fig.savefig(
        wrkdir_DNR + "Figs/amr_target_{}_{}.png".format(fnr, resol.replace("/", "_"))
    )
    plt.close(fig)


def ipshock_1d_compare(fnr=36, resols=[250, 300, 500, 1000, 2000, 4000, 8000]):

    ipshock_path = os.environ["WRK"] + "/ipshock_FIE/"

    var_list = [
        "proton/vg_rho",
        "proton/vg_rho_nonthermal",
        "proton/vg_v",
        "proton/vg_v_nonthermal",
        "vg_b_vol",
    ]
    ylabels = [
        "$\\rho~[\mathrm{cm}^{-3}]$",
        "$\\rho_\mathrm{non-th}~[\mathrm{cm}^{-3}]$",
        "$v_x~[\mathrm{km/s}]$",
        "$v_{\mathrm{non-th},x}~[\mathrm{km/s}]$",
        "$B_y~[\mathrm{nT}]$",
    ]
    scales = [1e-6, 1e-6, 1e-3, 1e-3, 1e9]
    miny = [None, 10**-4, -1000, -500, -5]
    maxy = [5, 5, 0, 1000, 5]
    op = ["pass", "pass", "x", "x", "y"]
    yscales = ["log", "log", "linear", "linear", "linear"]

    fig, ax_list = plt.subplots(
        len(var_list), 1, figsize=(8, 12), constrained_layout=True, sharex=True
    )

    for idx, r in enumerate(resols):
        vobj = pt.vlsvfile.VlsvReader(
            ipshock_path + "{}/bulk/bulk.{}.vlsv".format(r, str(fnr).zfill(7))
        )

        cellids = vobj.read_variable("CellID")
        x_arr = (
            np.array([vobj.get_cell_coordinates(c)[0] for c in np.sort(cellids)]) / r_e
        )
        for idx2, var in enumerate(var_list):
            ax = ax_list[idx2]
            var_arr = (
                vobj.read_variable(var, operator=op[idx2])[np.argsort(cellids)]
                * scales[idx2]
            )

            ax.plot(x_arr, var_arr, color=CB_color_cycle[idx], label="{}".format(r))

    for idx, ax in enumerate(ax_list):
        ax.grid()
        # ax.set_xlim(x_arr[0], x_arr[-1])
        ax.set_xlim(-20, 40)
        ax.set_ylabel(ylabels[idx])
        ax.set_yscale(yscales[idx])
        ax.set_ylim(miny[idx], maxy[idx])
    ax_list[-1].set_xlabel("X [RE]")
    ax_list[0].legend(loc="upper right")
    ax_list[0].set_title("t = {}s".format(fnr * 5))

    resols = "_".join([r.replace("/", "_") for r in resols])

    fig.savefig(wrkdir_DNR + "Figs/res_comp_{}_r{}.png".format(fnr, resols))
    plt.close(fig)
