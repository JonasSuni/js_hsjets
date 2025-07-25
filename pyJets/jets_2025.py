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
from pyJets.agf_jets import PropReader as AIC_PropReader
from pyJets.jet_io import PropReader as OLD_PropReader
from pyJets.papu_2 import get_fcs_jets
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

wrkdir_DNR = wrkdir_DNR + "jets_all_2D/"


def process_timestep_VSC_timeseries(args):
    """Helper function for parallel processing in VSC_timeseries"""
    (
        fnr,
        var_list,
        scales,
        bulkpath,
        ops,
        x0,
        y0,
    ) = args
    try:
        result = np.zeros(len(var_list) + 7, dtype=float)
        vlsvobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
        )
        for idx2, var in enumerate(var_list):
            result[idx2] = (
                vlsvobj.read_interpolated_variable(
                    var, [x0 * r_e, y0 * r_e, 0], operator=ops[idx2]
                )
                * scales[idx2]
            )
        return fnr, result
    except Exception as e:
        print(f"Error processing timestep {fnr}: {str(e)}")
        return fnr, None


def VSC_timeseries(
    runid,
    x0,
    y0,
    t0,
    t1,
    dirprefix="",
    skip=False,
    jett0=0.0,
    n_processes=1,
):
    bulkpath = find_bulkpath(runid)

    txtdir = wrkdir_DNR + "txts/timeseries/{}/{}".format(runid, dirprefix)
    if not os.path.exists(txtdir):
        try:
            os.makedirs(txtdir)
        except OSError:
            pass
    if skip and os.path.isfile(
        txtdir + "{}_x{:.3f}_y{:.3f}_t0{}_t1{}.txt".format(runid, x0, y0, t0, t1)
    ):
        print("Skip is True and file already exists, exiting.")
        return None

    if runid in ["AEA", "AEC", "AIC"]:
        var_list = [
            "proton/vg_rho",
            "proton/vg_v",
            "proton/vg_v",
            "proton/vg_v",
            "proton/vg_v",
            "proton/vg_Pdyn",
            "vg_b_vol",
            "vg_b_vol",
            "vg_b_vol",
            "vg_b_vol",
            "vg_e_vol",
            "vg_e_vol",
            "vg_e_vol",
            "vg_e_vol",
            "proton/vg_t_parallel",
            "proton/vg_t_perpendicular",
        ]
    else:
        var_list = [
            "rho",
            "v",
            "v",
            "v",
            "v",
            "Pdyn",
            "B",
            "B",
            "B",
            "B",
            "E",
            "E",
            "E",
            "E",
            "TParallel",
            "TPerpendicular",
        ]
    scales = [
        1e-6,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        1e9,
        1e9,
        1e9,
        1e9,
        1e9,
        1e3,
        1e3,
        1e3,
        1e3,
        1e-6,
        1e-6,
    ]
    ops = [
        "pass",
        "x",
        "y",
        "z",
        "magnitude",
        "pass",
        "x",
        "y",
        "z",
        "magnitude",
        "x",
        "y",
        "z",
        "magnitude",
        "pass",
        "pass",
    ]

    t_arr = np.arange(t0, t1 + 0.1, 0.5)
    fnr0 = int(t0 * 2)
    fnr_arr = np.arange(fnr0, int(t1 * 2) + 1, dtype=int)
    cellid = pt.vlsvfile.VlsvReader(
        bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
    ).get_cellid([x0 * r_e, y0 * r_e, 0 * r_e])
    data_arr = np.zeros((len(var_list) + 7, fnr_arr.size), dtype=float)
    tavg_arr = np.zeros(fnr_arr.size, dtype=float)

    # Prepare arguments for parallel processing
    args_list = [
        (
            fnr,
            var_list,
            scales,
            bulkpath,
            ops,
            x0,
            y0,
        )
        for fnr in fnr_arr
    ]

    # Use multiprocessing Pool

    with Pool(processes=n_processes) as pool:
        results = pool.map(process_timestep_VSC_timeseries, args_list)

        # Process results
        for fnr, result in results:
            if result is not None:
                idx = np.where(fnr_arr == fnr)[0][0]
                data_arr[:, idx] = result

    rho_lp = m_p * data_arr[0, :] * 1e6
    vx_lp = data_arr[1, :] * 1e3
    vy_lp = data_arr[2, :] * 1e3
    vz_lp = data_arr[3, :] * 1e3
    vt_lp = data_arr[4, :] * 1e3
    pd_lp = data_arr[5, :] * 1e-9

    rho_term = rho_lp * np.nanmean(vt_lp**2) / np.nanmean(pd_lp)
    vx_term = np.nanmean(rho_lp) * vx_lp**2 / np.nanmean(pd_lp)
    vy_term = np.nanmean(rho_lp) * vy_lp**2 / np.nanmean(pd_lp)
    vz_term = np.nanmean(rho_lp) * vz_lp**2 / np.nanmean(pd_lp)

    data_arr[-7, :] = rho_term
    data_arr[-6, :] = vx_term
    data_arr[-5, :] = vy_term
    data_arr[-4, :] = vz_term
    data_arr[-3, :] = tavg_arr
    data_arr[-2, :] = t_arr
    data_arr[-1, :] = np.ones_like(t_arr) * jett0

    np.savetxt(
        txtdir + "{}_x{:.3f}_y{:.3f}_t0{}_t1{}.txt".format(runid, x0, y0, t0, t1),
        data_arr,
    )

def plot_timeseries_at_jets_OLD(
    runid,
    boxre=None,
    tmin=None,
    tmax=None,
    folder_suffix="jets",
    skip=False,
    minduration=0,
    minsize=0,
    n_processes=1,
):

    if runid == "AIC":
        PropReader = AIC_PropReader
    else:
        PropReader = OLD_PropReader

    if folder_suffix == "fcs":
        jet_ids = get_fcs_jets(runid)
    else:
        kind = ["foreshock","beam"][["antisunward","flankward"].index(folder_suffix)]
        jet_ids = np.loadtxt(wrkdir_DNR + "papu22/id_txts/auto/{}_{}.txt".format(runid, kind),dtype=int,ndmin=1)

    for n1 in jet_ids:
        # try:
        #     props = PropReader(str(n1).zfill(5), runid, transient="jet")
        # except:
        #     continue
        props = PropReader(str(n1).zfill(5), runid, transient="jet")

        if props.read("at_bow_shock")[0] != 1:
            continue

        xmean = props.read("x_mean")
        ymean = props.read("y_mean")

        x0, y0 = (xmean[0], ymean[0])
        t0 = props.get_times()[0]
        tarr = props.read("time")
        duration = tarr[-1] - tarr[0] + 0.5
        maxsize = max(props.read("Nr_cells"))

        if tmin:
            if t0 < tmin:
                continue
        if tmax:
            if t0 > tmax:
                continue

        if boxre:
            if not (
                x0 >= boxre[0] and x0 <= boxre[1] and y0 >= boxre[2] and y0 <= boxre[3]
            ):
                continue

        if np.sqrt(x0**2 + y0**2) < 8:
            continue
        if duration < minduration:
            continue
        if maxsize < minsize:
            continue
        if "splinter" in props.meta:
            continue

        plott0 = t0 - 10
        plott1 = t0 + 10

        print(
            "Plotting timeseries at ({:.3f},{:.3f}) from t = {} to {} s, jet ID = {}".format(
                x0,
                y0,
                plott0,
                plott1,
                n1,
            )
        )

        VSC_timeseries(
            runid,
            x0,
            y0,
            plott0,
            plott1,
            dirprefix="{}/".format(folder_suffix),
            skip=skip,
            jett0=t0,
            n_processes=n_processes,
        )

def plot_timeseries_at_jets_AIC(
    runid,
    boxre=None,
    tmin=None,
    tmax=None,
    folder_suffix="jets",
    skip=False,
    minduration=0,
    minsize=0,
    n_processes=1,
):

    if runid == "AIC":
        PropReader = AIC_PropReader
    else:
        PropReader = OLD_PropReader

    for n1 in range(6000):
        try:
            props = PropReader(str(n1).zfill(5), runid, transient="jet")
        except:
            continue

        if props.read("at_bow_shock")[0] != 1:
            continue

        xmean = props.read("x_mean")
        ymean = props.read("y_mean")

        x0, y0 = (xmean[0], ymean[0])
        t0 = props.get_times()[0]
        tarr = props.read("time")
        duration = tarr[-1] - tarr[0] + 0.5
        maxsize = max(props.read("Nr_cells"))

        if t0 <= 391 or t0 > 1000:
            continue
        if tmin:
            if t0 < tmin:
                continue
        if tmax:
            if t0 > tmax:
                continue

        if boxre:
            if not (
                x0 >= boxre[0] and x0 <= boxre[1] and y0 >= boxre[2] and y0 <= boxre[3]
            ):
                continue

        if np.sqrt(x0**2 + y0**2) < 8:
            continue
        if duration < minduration:
            continue
        if maxsize < minsize:
            continue
        if "splinter" in props.meta:
            continue
        if props.at_ch_shock()[0] != True:
            continue

        plott0 = t0 - 10
        plott1 = t0 + 10

        print(
            "Plotting timeseries at ({:.3f},{:.3f}) from t = {} to {} s, jet ID = {}".format(
                x0,
                y0,
                plott0,
                plott1,
                n1,
            )
        )

        VSC_timeseries(
            runid,
            x0,
            y0,
            plott0,
            plott1,
            dirprefix="{}/".format(folder_suffix),
            skip=skip,
            jett0=t0,
            n_processes=n_processes,
        )


def all_cats_timeseries_script(n_processes=1, skip=True):

    boxres = [
        [8, 16, 3, 17],
        [8, 20, 3, 17],
        [8, 16, 3, 17],
        [8, 16, -17, 0],
        [8, 20, -17, 17],
        [8, 20, -17, -3],
        [8, 16, -17, -3],
    ]
    folder_suffixes = [
        "qpar_before",
        "qpar_after",
        "qpar_fb",
        "qperp_rd",
        "all",
        "qperp_after",
        "qperp_inter",
    ]
    tmins = [391, 470, 430, 430, 391, 600, 509]
    tmaxs = [426, 800, 470, 470, 800, 800, 600]

    for idx in range(len(folder_suffixes)):
        plot_timeseries_at_jets_AIC(
            "AIC",
            boxre=boxres[idx],
            tmin=tmins[idx],
            tmax=tmaxs[idx],
            folder_suffix=folder_suffixes[idx],
            skip=skip,
            minduration=1,
            minsize=4,
            pdavg=False,
            n_processes=n_processes,
        )

    for sfx in ["fcs","antisunward","flankward"]:
        for runid in ["ABA","ABC","AEA","AEC"]:
            plot_timeseries_at_jets_OLD(
                runid,
                folder_suffix=sfx,
                skip=skip,
                minduration=1,
                minsize=4,
                pdavg=False,
                n_processes=n_processes,
            )
