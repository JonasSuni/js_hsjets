import analysator as pt
import os
import sys
from random import choice
from copy import deepcopy

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

from matplotlib.ticker import MaxNLocator, ScalarFormatter

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

wrkdir_DNR = wrkdir_DNR + "jets_3D/"
wrkdir_other = os.environ["WRK"] + "/"

bulkpath_FIF = "/wrk-vakka/group/spacephysics/vlasiator/3D/FIF/bulk1/"

def get_msh_VDF_coordinates():

    outdir = wrkdir_DNR+"msh_vdf_locs/"

    fnr0 = 600
    fnr1 = 991
    boxre = [8,15,-10,0,-10,10]

    for fnr in range(fnr0,fnr1):
        fname = "bulk1.{}.vlsv".format(str(fnr).zfill(7))
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath_FIF+fname)
        cellids = vlsvobj.read_variable("CellID")
        fsaved = vlsvobj.read_variable("vg_f_saved")
        vdf_cellids = cellids[fsaved==1]
        x,y,z = np.array([vlsvobj.get_cell_coordinates(ci)/r_e for ci in vdf_cellids]).T
        maskx = np.logical_and(x>=boxre[0],x<=boxre[1])
        masky = np.logical_and(y>=boxre[2],y<=boxre[3])
        maskz = np.logical_and(z>=boxre[4],z<=boxre[5])
        mask = np.logical_and(maskx,np.logical_and(masky,maskz))

        outarr = np.array([vdf_cellids[mask],x[mask],y[mask],z[mask]])
        np.savetxt(outdir+"{}.txt".format(fnr),outarr.T)