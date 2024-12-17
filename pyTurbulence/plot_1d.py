# from operator import ge
# import sys
# import matplotlib.style
# import matplotlib as mpl
# import jet_aux as jx
# from pyJets.jet_aux import (
#     CB_color_cycle,
#     find_bulkpath,
#     restrict_area,
#     get_neighs,
#     get_neighs_asym,
#     xyz_reconstruct,
#     bow_shock_jonas,
#     mag_pause_jonas,
#     BS_xy,
#     MP_xy,
# )
from pyJets.jet_analyser import get_cell_volume, sw_par_dict
from pyJets.jet_aux import CB_color_cycle
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

wrkdir_DNR = wrkdir_DNR + "turbulence/"
turbdir = "/wrk-vakka/group/spacephysics/turbulence/"


def plot_elsasser(fnr0, fnr1, dirname):

    figdir = wrkdir_DNR + "Figs/cuts/{}/".format(dirname)
    if not os.path.exists(figdir):
        try:
            os.makedirs(figdir)
        except OSError:
            pass

    bulkpath = turbdir + "MultiCircularAlfven/bulk/4_attempt/"

    fnr = np.arange(fnr0, fnr1 + 1)

    for idx in fnr:
        vlsvobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk." + str(idx).zfill(7) + ".vlsv"
        )
        cellids = vlsvobj.read_variable("cellID")
        ci_sorted = np.sort(cellids)
        x = np.array([vlsvobj.get_cell_coordinates(c)[0] / r_e for c in ci_sorted])
        By = vlsvobj.read_variable("vg_b_vol", operator="y")[cellids.argsort()]
        Bz = vlsvobj.read_variable("vg_b_vol", operator="z")[cellids.argsort()]
        B = np.array([np.zeros_like(By), By, Bz])

        vy = vlsvobj.read_variable("proton/vg_v", operator="y")[cellids.argsort()]
        vz = vlsvobj.read_variable("proton/vg_v", operator="z")[cellids.argsort()]
        v = np.array([np.zeros_like(vy), vy, vz])

        BvA = B / np.sqrt(
            mu0 * m_p * vlsvobj.read_variable("proton/vg_rho")[cellids.argsort()]
        )

        fig, ax = plt.subplots(1, 1, figsize=(8, 3), constrained_layout=True)

        ax.plot(
            x,
            np.linalg.norm(v + BvA, axis=0),
            label="$\\delta z^{+}$",
            color=CB_color_cycle[0],
        )
        ax.plot(
            x,
            np.linalg.norm(v - BvA, axis=0),
            label="$\\delta z^{-}$",
            color=CB_color_cycle[1],
        )

        ax.set_ylabel(r"$\delta z^{\pm}$")
        ax.set_xlabel(r"$x~[R_\mathrm{E}]$")
        ax.set_xlim([x[0], x[-1]])
        ax.set_ylim(0, 25000)
        ax.grid()
        ax.legend()

        fig.savefig(figdir + "{}_elsasser.png".format(idx))
        plt.close(fig)


def plot_Byz(fnr0, fnr1, dirname):

    figdir = wrkdir_DNR + "Figs/cuts/{}/".format(dirname)
    if not os.path.exists(figdir):
        try:
            os.makedirs(figdir)
        except OSError:
            pass

    bulkpath = turbdir + "MultiCircularAlfven/bulk/4_attempt/"

    fnr = np.arange(fnr0, fnr1 + 1)

    for idx in fnr:
        vlsvobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk." + str(idx).zfill(7) + ".vlsv"
        )
        cellids = vlsvobj.read_variable("cellID")
        ci_sorted = np.sort(cellids)
        x = np.array([vlsvobj.get_cell_coordinates(c)[0] / r_e for c in ci_sorted])
        By = vlsvobj.read_variable("vg_b_vol", operator="y")[cellids.argsort()] / 1e-9
        Bz = vlsvobj.read_variable("vg_b_vol", operator="z")[cellids.argsort()] / 1e-9

        fig, ax = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)

        ax[0].plot(x, By)
        ax[0].set_ylabel(r"$B_y$ [nT]")
        ax[0].set_title("t = {}".format(str(vlsvobj.read_parameter("time"))))
        ax[0].set_xlim([x[0], x[-1]])
        ax[0].set_ylim([-1, 1])
        ax[0].grid()

        ax[1].plot(x, Bz)
        ax[1].set_ylabel(r"$B_z$ [nT]")
        ax[1].set_xlim([x[0], x[-1]])
        ax[1].set_ylim([-1, 1])
        ax[1].grid()

        ax[2].plot(x, np.arctan2(Bz, By) * 360 / (2 * np.pi))
        ax[2].set_ylabel(r"$\theta$ [deg]")
        ax[2].set_xlabel(r"$x~[R_\mathrm{E}]$")
        ax[2].set_xlim([x[0], x[-1]])
        ax[2].set_ylim([-180, 180])
        ax[2].grid()

        fig.savefig(figdir + "{}.png".format(idx))
        plt.close(fig)
