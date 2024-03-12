# from operator import ge
# import sys
# import matplotlib.style
# import matplotlib as mpl
# import jet_aux as jx
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

import pyspedas
from datetime import datetime

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

wrkdir_DNR = wrkdir_DNR + "foreshock_bubble/"


def interpolate_nans(data):

    dummy_x = np.arange(data.size)
    mask = np.isnan(data)

    return np.interp(dummy_x, dummy_x[~mask], data[~mask])


def plot_ace_dscovr_wind(t0, t1):

    t0plot = datetime.strptime(t0, "%Y-%m-%d/%H:%M:%S")
    t1plot = datetime.strptime(t1, "%Y-%m-%d/%H:%M:%S")

    ace_data = pyspedas.ace.mfi(trange=[t0, t1], notplot=True)
    dscovr_data = pyspedas.dscovr.mag(trange=[t0, t1], notplot=True)
    wind_data = pyspedas.wind.mfi(trange=[t0, t1], notplot=True)

    ace_t = ace_data["BGSEc"]["x"]
    ace_B = ace_data["BGSEc"]["y"].T

    dscovr_t = dscovr_data["dsc_h0_mag_B1GSE"]["x"]
    dscovr_B = dscovr_data["dsc_h0_mag_B1GSE"]["y"].T

    wind_t = wind_data["BGSE"]["x"]
    wind_B = wind_data["BGSE"]["y"].T

    ace_clock, dscovr_clock, wind_clock = [
        np.rad2deg(np.arctan2(B[2], B[1])) for B in [ace_B, dscovr_B, wind_B]
    ]
    ace_cone, dscovr_cone, wind_cone = [
        np.rad2deg(np.arctan2(np.sqrt(B[2] ** 2 + B[1] ** 2), B[0]))
        for B in [ace_B, dscovr_B, wind_B]
    ]
    ace_Bmag, dscovr_Bmag, wind_Bmag = [
        np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2) for B in [ace_B, dscovr_B, wind_B]
    ]

    time_list = [ace_t, dscovr_t, wind_t]
    data_list = [
        [ace_B[0], ace_B[1], ace_B[2], ace_Bmag, ace_clock, ace_cone],
        [dscovr_B[0], dscovr_B[1], dscovr_B[2], dscovr_Bmag, dscovr_clock, dscovr_cone],
        [wind_B[0], wind_B[1], wind_B[2], wind_Bmag, wind_clock, wind_cone],
    ]
    ylabs = ["Bx", "By", "Bz", "Bmag", "Clock", "Cone"]

    fig, ax_list = plt.subplots(
        6, 3, figsize=(18, 18), constrained_layout=True, sharey="row"
    )

    for idx in range(3):
        for idx2 in range(6):
            ax = ax_list[idx2, idx]
            if idx == 0:
                ax.set_ylabel(ylabs[idx2])
            if idx == 2:
                ax.plot(time_list[idx], data_list[idx][idx2])
            else:
                ax.plot(
                    time_list[idx],
                    uniform_filter1d(interpolate_nans(data_list[idx][idx2]), size=60),
                )
            # ax.plot(time_list[idx], data_list[idx][idx2])
            ax.set_xlim(t0plot, t1plot)

    for ax in ax_list.flatten():
        ax.label_outer()

    outdir = wrkdir_DNR + "Figs/satellite/"
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    fig.savefig(outdir + "ace_dscovr_wind_t0{}_t1{}.png".format(t0plot, t1plot))
    plt.close(fig)
