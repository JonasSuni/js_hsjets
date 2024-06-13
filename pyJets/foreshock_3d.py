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

wrkdir_DNR = wrkdir_DNR + "foreshock_3d/"


def ipshock_1d_compare(fnr=36):

    resols = [250, 300, 500]
    ipshock_path = os.environ["WRK"] + "/ipshock_FIE/"

    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    for idx, r in enumerate(resols):
        vobj = pt.vlsvfile.VlsvReader(
            ipshock_path + "{}/bulk/bulk.{}.vlsv".format(r, str(fnr).zfill(7))
        )

        cellids = vobj.read_variable("CellID")
        x_arr = (
            np.array([vobj.get_cell_coordinates(c)[0] for c in cellids.sort()]) / r_e
        )
        rho_arr = vobj.read_variable("proton/vg_rho")[np.argsort(cellids)]

        ax.plot(x_arr, rho_arr, color=CB_color_cycle[idx], label="{}".format(r))

    ax.grid()
    ax.set_xlim(x_arr[0], x_arr[-1])
    ax.set_xlabel("X [RE]")
    ax.set_ylabel("$\\rho~[m^{-3}]$")

    fig.savefig(wrkdir_DNR + "res_comp.png")
    plt.close(fig)