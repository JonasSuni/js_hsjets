# from operator import ge
# import sys
# import matplotlib.style
# import matplotlib as mpl
# import jet_aux as jx
from pyJets.jet_aux import CB_color_cycle, find_bulkpath, restrict_area
from pyJets.jet_io import eventprop_write, jet_sorter
import pytools as pt
import os

# import scipy
# import scipy.linalg
from scipy.linalg import eig
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.lines as mlines
import matplotlib.ticker as ticker

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


def mask_maker(runid, filenr, boxre=[6, 18, -8, 6], avgfile=True, mag_thresh=1.5):
    bulkpath = find_bulkpath(runid)
    bulkname = "bulk." + str(filenr).zfill(7) + ".vlsv"

    if bulkname not in os.listdir(bulkpath):
        print("Bulk file " + str(filenr) + " not found, exiting.")
        return 1

    # open vlsv file for reading
    vlsvreader = pt.vlsvfile.VlsvReader(bulkpath + bulkname)

    origid = vlsvreader.read_variable("CellID")
    sorigid = origid[np.argsort(origid)]

    pdyn = vlsvreader.read_variable("proton/vg_Pdyn")[np.argsort(origid)]
    B = vlsvreader.read_variable("vg_b_vol")[np.argsort(origid)]
    pr_rhonbs = vlsvreader.read_variable("proton/vg_rho_thermal")[np.argsort(origid)]
    pr_PTDNBS = vlsvreader.read_variable("proton/vg_ptensor_thermal_diagonal")[
        np.argsort(origid)
    ]

    T_sw = 0.5e6
    epsilon = 1.0e-10
    kb = 1.38065e-23

    pr_pressurenbs = (1.0 / 3.0) * (pr_PTDNBS.sum(-1))
    pr_TNBS = pr_pressurenbs / ((pr_rhonbs + epsilon) * kb)

    mmsx = vlsvreader.read_variable("proton/vg_mmsx")[np.argsort(origid)]

    Bmag = np.linalg.norm(B, axis=-1)

    rho_sw = 1e6
    v_sw = 750e3
    pdyn_sw = m_p * rho_sw * v_sw * v_sw
    B_sw = 3e-9

    # initialise time average of dynamic pressure
    tpdynavg = np.zeros(pdyn.shape)

    # range of timesteps to calculate average of
    timerange = range(filenr - 180, filenr + 180 + 1)

    missing_file_counter = 0

    vlsvobj_list = []

    if avgfile:
        tpdynavg = np.loadtxt(tavgdir + "/" + runid + "/" + str(filenr) + "_pdyn.tavg")
    else:
        for n_t in timerange:
            # exclude the main timestep
            if n_t == filenr:
                continue

            # find correct file path for current time step
            tfile_name = "bulk." + str(n_t).zfill(7) + ".vlsv"

            if tfile_name not in os.listdir(bulkpath):
                missing_file_counter += 1
                print("Bulk file " + str(n_t) + " not found, continuing")
                continue

            # open file for current time step
            vlsvobj_list.append(pt.vlsvfile.VlsvReader(bulkpath + tfile_name))

        for f in vlsvobj_list:
            # f.optimize_open_file()

            # if file has separate populations, read proton population
            tpdyn = f.read_variable("proton/vg_Pdyn")

            # read cellids for current time step
            cellids = f.read_variable("CellID")

            # sort dynamic pressures
            otpdyn = tpdyn[cellids.argsort()]

            tpdynavg = np.add(tpdynavg, otpdyn)

            # f.optimize_clear_fileindex_for_cellid()
            # f.optimize_close_file()

        # calculate time average of dynamic pressure
        tpdynavg /= len(timerange) - 1 - missing_file_counter

    # prevent divide by zero errors
    tpdynavg[tpdynavg == 0.0] = 1.0e-27

    slams = np.ma.masked_greater_equal(Bmag, mag_thresh * B_sw)
    slams.mask[pr_TNBS >= 3.0 * T_sw] = False
    slams.mask[pdyn < 1.2 * pdyn_sw] = False
    jet = np.ma.masked_greater_equal(pdyn, 2.0 * tpdynavg)
    jet.mask[pr_TNBS < 3.0 * T_sw] = False
    slamsjet = np.logical_or(slams, jet)

    if not os.path.exists(
        wrkdir_DNR + "up_down_stream/" + runid + "/{}.up".format(str(filenr))
    ):
        upstream = np.ma.masked_less(pr_TNBS, 3.0 * T_sw)
        upstream_ci = np.ma.array(sorigid, mask=~upstream.mask).compressed()

        upstream_mms = np.ma.masked_greater_equal(mmsx, 1)
        upstream_mms_ci = np.ma.array(sorigid, mask=~upstream_mms.mask).compressed()

    jet_ci = np.ma.array(sorigid, mask=~jet.mask).compressed()
    slams_ci = np.ma.array(sorigid, mask=~slams.mask).compressed()
    slamsjet_ci = np.ma.array(sorigid, mask=~slamsjet.mask).compressed()

    restr_ci = restrict_area(vlsvreader, boxre)

    if not os.path.exists(wrkdir_DNR + "working/jets/Masks/" + runid):
        os.makedirs(wrkdir_DNR + "working/jets/Masks/" + runid)
        os.makedirs(wrkdir_DNR + "working/SLAMS/Masks/" + runid)
        os.makedirs(wrkdir_DNR + "working/SLAMSJETS/Masks/" + runid)

    if not os.path.exists(wrkdir_DNR + "up_down_stream/" + runid):
        os.makedirs(wrkdir_DNR + "up_down_stream/" + runid)

    np.savetxt(
        wrkdir_DNR + "working/jets/Masks/" + runid + "/{}.mask".format(str(filenr)),
        np.intersect1d(jet_ci, restr_ci),
    )
    np.savetxt(
        wrkdir_DNR + "working/SLAMS/Masks/" + runid + "/{}.mask".format(str(filenr)),
        np.intersect1d(slams_ci, restr_ci),
    )
    np.savetxt(
        wrkdir_DNR
        + "working/SLAMSJETS/Masks/"
        + runid
        + "/{}.mask".format(str(filenr)),
        np.intersect1d(slamsjet_ci, restr_ci),
    )

    if not os.path.exists(
        wrkdir_DNR + "up_down_stream/" + runid + "/{}.up".format(str(filenr))
    ):
        np.savetxt(
            wrkdir_DNR + "up_down_stream/" + runid + "/{}.up".format(str(filenr)),
            np.intersect1d(upstream_ci, restr_ci),
        )
        np.savetxt(
            wrkdir_DNR + "up_down_stream/" + runid + "/{}.down".format(str(filenr)),
            restr_ci[~np.in1d(restr_ci, upstream_ci)],
        )

        np.savetxt(
            wrkdir_DNR + "up_down_stream/" + runid + "/{}.up.mms".format(str(filenr)),
            np.intersect1d(upstream_mms_ci, restr_ci),
        )
        np.savetxt(
            wrkdir_DNR + "up_down_stream/" + runid + "/{}.down.mms".format(str(filenr)),
            restr_ci[~np.in1d(restr_ci, upstream_mms_ci)],
        )

    return (
        np.intersect1d(jet_ci, restr_ci),
        np.intersect1d(slams_ci, restr_ci),
        np.intersect1d(slamsjet_ci, restr_ci),
    )


def jet_creator(
    runid,
    start,
    stop,
    boxre=[6, 18, -8, 6],
    maskfile=False,
    avgfile=True,
    nbrs=[2, 2, 0],
    mag_thresh=1.5,
):
    runid_list = ["ABA", "ABC", "AEA", "AEC"]
    maxfnr_list = [839, 1179, 1339, 879]
    if start > maxfnr_list[runid_list.index(runid)]:
        return 0

    global runid_g
    global filenr_g
    runid_g = runid

    global rho_sw_g

    rho_sw_g = 1e6

    # make outputdir if it doesn't already exist
    if not os.path.exists(wrkdir_DNR + "working/jets/events/" + runid + "/"):
        try:
            os.makedirs(wrkdir_DNR + "working/jets/events/" + runid + "/")
            os.makedirs(wrkdir_DNR + "working/SLAMS/events/" + runid + "/")
            os.makedirs(wrkdir_DNR + "working/SLAMSJETS/events/" + runid + "/")
        except OSError:
            pass

    bulkpath = find_bulkpath(runid)

    for file_nr in range(start, stop + 1):
        filenr_g = file_nr

        # find correct file based on file number and run id

        bulkname = "bulk." + str(file_nr).zfill(7) + ".vlsv"

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file " + str(file_nr) + " not found, continuing")
            continue

        # open vlsv file for reading
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath + bulkname)

        vlsvobj.optimize_open_file()

        # create mask
        if maskfile:
            jet_msk = np.loadtxt(
                wrkdir_DNR
                + "working/jets/Masks/"
                + runid
                + "/"
                + str(file_nr)
                + ".mask"
            ).astype(int)
            slams_msk = np.loadtxt(
                wrkdir_DNR
                + "working/SLAMS/Masks/"
                + runid
                + "/"
                + str(file_nr)
                + ".mask"
            ).astype(int)
            slamsjet_msk = np.loadtxt(
                wrkdir_DNR
                + "working/SLAMSJETS/Masks/"
                + runid
                + "/"
                + str(file_nr)
                + ".mask"
            ).astype(int)
        else:
            jet_msk, slams_msk, slamsjet_msk = mask_maker(
                runid, file_nr, boxre, avgfile, mag_thresh=mag_thresh
            )

        print("Current file number is " + str(file_nr))

        up_cells = np.loadtxt(
            wrkdir_DNR + "up_down_stream/" + runid + "/" + str(file_nr) + ".up"
        ).astype(int)
        down_cells = np.loadtxt(
            wrkdir_DNR + "up_down_stream/" + runid + "/" + str(file_nr) + ".down"
        ).astype(int)

        up_cells_mms = np.loadtxt(
            wrkdir_DNR + "up_down_stream/" + runid + "/" + str(file_nr) + ".up.mms"
        ).astype(int)
        down_cells_mms = np.loadtxt(
            wrkdir_DNR + "up_down_stream/" + runid + "/" + str(file_nr) + ".down.mms"
        ).astype(int)

        # sort jets
        jets, props_inc = jet_sorter(
            vlsvobj,
            jet_msk,
            slams_msk,
            slamsjet_msk,
            up_cells,
            down_cells,
            up_cells_mms,
            down_cells_mms,
            neighborhood_reach=nbrs,
        )
        jet_jets, jet_props_inc = jet_sorter(
            vlsvobj,
            jet_msk,
            slams_msk,
            jet_msk,
            up_cells,
            down_cells,
            up_cells_mms,
            down_cells_mms,
            neighborhood_reach=nbrs,
        )
        slams_jets, slams_props_inc = jet_sorter(
            vlsvobj,
            jet_msk,
            slams_msk,
            slams_msk,
            up_cells,
            down_cells,
            up_cells_mms,
            down_cells_mms,
            neighborhood_reach=nbrs,
        )

        props = [[float(file_nr) / 2.0] + line for line in props_inc]
        jet_props = [[float(file_nr) / 2.0] + line for line in jet_props_inc]
        slams_props = [[float(file_nr) / 2.0] + line for line in slams_props_inc]

        # print(len(jet_props))
        # print(len(jet_jets))
        #
        # print(len(slams_props))
        # print(len(slams_jets))

        eventprop_write(runid, file_nr, props, transient="slamsjet")
        eventprop_write(runid, file_nr, slams_props, transient="slams")
        eventprop_write(runid, file_nr, jet_props, transient="jet")

        # erase contents of output file

        open(
            wrkdir_DNR
            + "working/SLAMSJETS/events/"
            + runid
            + "/"
            + str(file_nr)
            + ".events",
            "w",
        ).close()

        # open output file
        fileobj = open(
            wrkdir_DNR
            + "working/SLAMSJETS/events/"
            + runid
            + "/"
            + str(file_nr)
            + ".events",
            "a",
        )

        # write jets to outputfile
        for jet in jets:
            fileobj.write(",".join(list(map(str, jet))) + "\n")

        fileobj.close()

        open(
            wrkdir_DNR
            + "working/SLAMS/events/"
            + runid
            + "/"
            + str(file_nr)
            + ".events",
            "w",
        ).close()
        fileobj_slams = open(
            wrkdir_DNR
            + "working/SLAMS/events/"
            + runid
            + "/"
            + str(file_nr)
            + ".events",
            "a",
        )

        for slams_jet in slams_jets:
            fileobj_slams.write(",".join(list(map(str, slams_jet))) + "\n")

        fileobj_slams.close()

        open(
            wrkdir_DNR
            + "working/jets/events/"
            + runid
            + "/"
            + str(file_nr)
            + ".events",
            "w",
        ).close()
        fileobj_jet = open(
            wrkdir_DNR
            + "working/jets/events/"
            + runid
            + "/"
            + str(file_nr)
            + ".events",
            "a",
        )

        for jet_jet in jet_jets:
            fileobj_jet.write(",".join(list(map(str, jet_jet))) + "\n")

        fileobj_jet.close()

        vlsvobj.optimize_close_file()

    return None
