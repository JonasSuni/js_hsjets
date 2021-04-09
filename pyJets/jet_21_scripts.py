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


def make_plots():

    outpath = wrkdir_DNR + "Figures/tektal/"

    ABA_path = "/wrk/group/spacephysics/vlasiator/2D/ABA/bulk/bulk.0001000.vlsv"

    pt.plot.plot_colormap(
        filename=ABA_path,
        outputfile=outpath + "ABA_global_viridis.pdf",
        var="Pdyn",
        nocb=True,
        lin=1,
        colormap="viridis",
        noxlabels=True,
        noylabels=True,
        noborder=True,
        title="",
        Earth=1,
        vmax=3e-9,
    )
    pt.plot.plot_colormap(
        filename=ABA_path,
        outputfile=outpath + "ABA_global_plasma.pdf",
        var="Pdyn",
        nocb=True,
        lin=1,
        colormap="plasma",
        noxlabels=True,
        noylabels=True,
        noborder=True,
        title="",
        Earth=1,
        vmax=3e-9,
    )
    pt.plot.plot_colormap(
        filename=ABA_path,
        outputfile=outpath + "ABA_global_magma.pdf",
        var="Pdyn",
        nocb=True,
        lin=1,
        colormap="magma",
        noxlabels=True,
        noylabels=True,
        noborder=True,
        title="",
        Earth=1,
        vmax=3e-9,
    )

    return None

