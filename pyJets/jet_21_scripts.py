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

    plt.ioff()

    outpath = wrkdir_DNR + "Figures/tektal/"

    # runs = ["ABA", "ABC", "AEA", "AEC", "BCQ", "BFD"]
    runs = ["BCQ", "BFD"]
    vars = ["Pdyn", "rho", "v", "B"]
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

                fig, ax = plt.subplots(1, 1, figsize=(15, 15))
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

    return None

