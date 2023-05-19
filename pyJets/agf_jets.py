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
)
from pyJets.jet_analyser import get_cell_volume
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

propfile_var_list = [
    "time",
    "x_mean",
    "y_mean",
    "z_mean",
    "x_wmean",
    "y_wmean",
    "z_wmean",
    "A",
    "Nr_cells",
    "size_rad",
    "size_tan",
    "size_vpar",
    "size_vperp",
    "size_Bpar",
    "size_Bperp",
    "x_vmax",
    "y_vmax",
    "z_vmax",
    "n_avg",
    "n_med",
    "n_max",
    "v_avg",
    "v_med",
    "v_max",
    "B_avg",
    "B_med",
    "B_max",
    "T_avg",
    "T_med",
    "T_max",
    "TPar_avg",
    "TPar_med",
    "TPar_max",
    "TPerp_avg",
    "TPerp_med",
    "TPerp_max",
    "beta_avg",
    "beta_med",
    "beta_max",
    "x_min",
    "rho_vmax",
    "b_vmax",
    "pd_avg",
    "pd_med",
    "pd_max",
    "B_sheath",
    "TPar_sheath",
    "TPerp_sheath",
    "T_sheath",
    "n_sheath",
    "v_sheath",
    "pd_sheath",
    "is_upstream",
    "ew_pd_enh",
    "is_slams",
    "is_jet",
    "is_merger",
    "is_splinter",
    "at_bow_shock",
    "at_slams",
    "at_jet",
]
propfile_header_list = "time [s],x_mean [R_e],y_mean [R_e],z_mean [R_e],x_wmean [R_e],y_wmean [R_e],z_wmean [R_e],A [R_e^2],Nr_cells,size_rad [R_e],size_tan [R_e],size_vpar [R_e],size_vperp [R_e],size_Bpar [R_e],size_Bperp [R_e],x_max [R_e],y_max [R_e],z_max [R_e],n_avg [1/cm^3],n_med [1/cm^3],n_max [1/cm^3],v_avg [km/s],v_med [km/s],v_max [km/s],B_avg [nT],B_med [nT],B_max [nT],T_avg [MK],T_med [MK],T_max [MK],TPar_avg [MK],TPar_med [MK],TPar_max [MK],TPerp_avg [MK],TPerp_med [MK],TPerp_max [MK],beta_avg,beta_med,beta_max,x_min [R_e],rho_vmax [1/cm^3],b_vmax,pd_avg [nPa],pd_med [nPa],pd_max [nPa],B_sheath [nT],TPar_sheath [MK],TPerp_sheath [MK],T_sheath [MK],n_sheath [1/cm^3],v_sheath [km/s],pd_sheath [nPa],is_upstream [bool],ew_pd_enh [nPa],is_slams [bool],is_jet [bool],is_merger [bool],is_splinter [bool],at_bow_shock [bool],at_jet [bool],at_jet [bool]"


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


def jet_sorter(
    vlsvobj,
    jet_cells,
    slams_cells,
    sj_cells,
    up_cells,
    down_cells,
    up_cells_mms,
    down_cells_mms,
    min_size=1,
    max_size=100000000,
    neighborhood_reach=[2, 2, 0],
):
    cells = np.array(sj_cells, ndmin=1, dtype=int)
    events = []
    curr_event = np.array([], dtype=int)

    while cells.size != 0:
        curr_event = np.array([cells[0]], dtype=int)
        curr_event_size = curr_event.size

        curr_event = np.intersect1d(
            cells, get_neighs(runid_g, curr_event, neighborhood_reach)
        )

        while curr_event.size != curr_event_size:
            # print(curr_event.size)
            # print(curr_event_size)

            curr_event_size = curr_event.size

            curr_event = np.intersect1d(
                cells, get_neighs(runid_g, curr_event, neighborhood_reach)
            )

        events.append(curr_event.astype(int))
        cells = cells[~np.in1d(cells, curr_event)]

    events_culled = [
        jet for jet in events if jet.size >= min_size and jet.size <= max_size
    ]

    props = [
        calc_event_props(
            vlsvobj,
            event,
            jet_cells,
            slams_cells,
            up_cells,
            down_cells,
            up_cells_mms,
            down_cells_mms,
        )
        for event in events_culled
    ]

    return [events_culled, props]


def mean_med_max(var):
    var_mean = np.nanmean(var)
    var_med = np.nanmedian(var)
    var_max = np.nanmax(var)

    return [var_mean, var_med, var_max]


def get_sheath_cells(runid, cells, neighborhood_reach=[2, 2, 0]):
    plus_sheath_cells = get_neighs(runid, cells, neighborhood_reach)
    sheath_cells = plus_sheath_cells[~np.in1d(plus_sheath_cells, cells)]

    return sheath_cells


def get_sheath_cells_asym(runid, cells, neighborhood_reach=[-1, 1, -1, 1, 0, 0]):
    plus_sheath_cells = get_neighs_asym(runid, cells, neighborhood_reach)
    sheath_cells = plus_sheath_cells[~np.in1d(plus_sheath_cells, cells)]

    return sheath_cells


def calc_event_props(
    vlsvobj,
    cells,
    jet_cells=[],
    slams_cells=[],
    up_cells=[],
    down_cells=[],
    up_cells_mms=[],
    down_cells_mms=[],
):
    is_merger = 0
    is_splinter = 0
    is_slams = 0
    is_jet = 0
    at_jet = 0
    at_slams = 0
    at_bow_shock = 0

    upstream_slice = get_neighs_asym(
        runid_g, down_cells, neighborhood_reach=[0, 2, 0, 0, 0, 0]
    )
    downstream_slice = get_neighs_asym(
        runid_g, up_cells, neighborhood_reach=[-2, 0, 0, 0, 0, 0]
    )

    upstream_slice_mms = get_neighs_asym(
        runid_g, down_cells_mms, neighborhood_reach=[0, 2, 0, 0, 0, 0]
    )
    downstream_slice_mms = get_neighs_asym(
        runid_g, up_cells_mms, neighborhood_reach=[-2, 0, 0, 0, 0, 0]
    )

    bs_slice = np.intersect1d(upstream_slice, downstream_slice)
    bs_slice_mms = np.intersect1d(upstream_slice_mms, downstream_slice_mms)

    bs_slice_tot = np.union1d(bs_slice, bs_slice_mms)

    if np.intersect1d(cells, bs_slice_tot).size > 0:
        at_bow_shock = 1
    if np.intersect1d(cells, slams_cells).size > 0:
        is_slams = 1
    if np.intersect1d(cells, jet_cells).size > 0:
        is_jet = 1
    if (
        np.intersect1d(
            cells, get_neighs(runid_g, slams_cells, neighborhood_reach=[2, 2, 0])
        ).size
        > 0
    ):
        at_slams = 1
    if (
        np.intersect1d(
            cells, get_neighs(runid_g, jet_cells, neighborhood_reach=[2, 2, 0])
        ).size
        > 0
    ):
        at_jet = 1

    if np.argmin(vlsvobj.get_spatial_mesh_size()) == 1:
        sheath_cells = get_sheath_cells(runid_g, cells, neighborhood_reach=[2, 0, 2])
        ssh_cells = get_sheath_cells(runid_g, cells, neighborhood_reach=[1, 0, 1])
    else:
        sheath_cells = get_sheath_cells(runid_g, cells)
        ssh_cells = get_sheath_cells(runid_g, cells, neighborhood_reach=[1, 1, 0])

    up_cells_all = np.union1d(up_cells, up_cells_mms)
    down_cells_all = np.union1d(down_cells, down_cells_mms)
    if is_jet and not is_slams:
        sheath_cells = np.intersect1d(sheath_cells, down_cells_all)
    elif is_slams and not is_jet:
        sheath_cells = np.intersect1d(sheath_cells, up_cells_all)

    ew_cells = get_sheath_cells_asym(
        runid_g, cells, neighborhood_reach=[-10, 0, 0, 0, 0, 0]
    )

    # read variables
    X, Y, Z = xyz_reconstruct(vlsvobj, cellids=cells)
    X = np.array(X, ndmin=1)
    Y = np.array(Y, ndmin=1)
    Z = np.array(Z, ndmin=1)
    dA = get_cell_volume(vlsvobj)

    var_list_v5 = [
        "proton/vg_rho",
        "proton/vg_v",
        "vg_b_vol",
        "proton/vg_temperature",
        "CellID",
        "proton/vg_Pdyn" "proton/vg_beta",
        "proton/vg_t_parallel",
        "proton/vg_t_perpendicular",
    ]

    sheath_list_v5 = [
        "proton/vg_rho",
        "proton/vg_v",
        "vg_b_vol",
        "proton/vg_temperature",
        "proton/vg_t_parallel",
        "proton/vg_t_perpendicular",
        "proton/vg_Pdyn",
    ]

    rho, v, B, T, cellids, pdyn, beta, TParallel, TPerpendicular = [
        np.array(vlsvobj.read_variable(s, cellids=cells), ndmin=1) for s in var_list_v5
    ]
    (
        rho_sheath,
        v_sheath,
        B_sheath,
        T_sheath,
        TPar_sheath,
        TPerp_sheath,
        pd_sheath,
    ) = [
        np.array(vlsvobj.read_variable(s, cellids=sheath_cells), ndmin=1)
        for s in sheath_list_v5
    ]

    pr_rhonbs = np.array(
        vlsvobj.read_variable("proton/vg_rho_thermal", cellids=ssh_cells), ndmin=1
    )
    pr_PTDNBS = np.array(
        vlsvobj.read_variable("proton/vg_ptensor_thermal_diagonal", cellids=ssh_cells),
        ndmin=1,
    )
    ew_pdyn = np.array(
        vlsvobj.read_variable("proton/vg_Pdyn", cellids=ew_cells), ndmin=1
    )
    mmsx_ssh = np.array(
        vlsvobj.read_variable("proton/vg_mmsx", cellids=ssh_cells), ndmin=1
    )

    # rho_sw = rho_sw_g
    T_sw = 0.5e6

    epsilon = 1.0e-10
    kb = 1.38065e-23

    pr_pressurenbs = (1.0 / 3.0) * (pr_PTDNBS.sum(-1))
    pr_TNBS = pr_pressurenbs / ((pr_rhonbs + epsilon) * kb)

    # is_upstream = int(np.all(rho_ssh < 2*rho_sw))
    is_upstream = int(np.all(pr_TNBS < 3 * T_sw))
    # is_upstream = int(np.all(mmsx_ssh >= 1))

    # Scale variables
    rho /= 1.0e6
    v /= 1.0e3
    B /= 1.0e-9
    B_sheath /= 1.0e-9
    pdyn /= 1.0e-9
    T /= 1.0e6
    TParallel /= 1.0e6
    TPerpendicular /= 1.0e6
    TPar_sheath /= 1.0e6
    TPerp_sheath /= 1.0e6
    T_sheath /= 1.0e6
    rho_sheath /= 1.0e6
    v_sheath /= 1.0e3
    pd_sheath /= 1.0e-9
    ew_pdyn /= 1.0e-9

    ew_pd_enh = np.nanmean(ew_pdyn)

    # Calculate magnitudes of v and B
    vmag = np.array(np.linalg.norm(v, axis=-1), ndmin=1)
    Bmag = np.array(np.linalg.norm(B, axis=-1), ndmin=1)

    avg_v = np.nanmean(np.array(v, ndmin=2), axis=0)
    avg_B = np.nanmean(np.array(B, ndmin=2), axis=0)

    avg_vu = avg_v / np.linalg.norm(avg_v)
    avg_Bu = avg_B / np.linalg.norm(avg_B)

    B_sheath_mag = np.array(np.linalg.norm(B_sheath, axis=-1), ndmin=1)
    v_sheath_mag = np.array(np.linalg.norm(v_sheath, axis=-1), ndmin=1)

    if type(vmag) == float:
        vmag = np.array(vmag, ndmin=1)
    if type(Bmag) == float:
        Bmag = np.array(Bmag, ndmin=1)
    if type(B_sheath_mag) == float:
        B_sheath_mag = np.array(B_sheath_mag, ndmin=1)

    n_avg, n_med, n_max = mean_med_max(rho)

    v_avg, v_med, v_max = mean_med_max(vmag)

    B_avg, B_med, B_max = mean_med_max(Bmag)

    pd_avg, pd_med, pd_max = mean_med_max(pdyn)

    T_avg, T_med, T_max = mean_med_max(T)

    TPar_avg, TPar_med, TPar_max = mean_med_max(TParallel)

    TPerp_avg, TPerp_med, TPerp_max = mean_med_max(TPerpendicular)

    beta_avg, beta_med, beta_max = mean_med_max(beta)

    # Weighted geometric center of jet in cartesian coordinates
    tavg_pdyn = np.loadtxt(
        np.loadtxt(tavgdir + "/" + runid_g + "/" + str(filenr_g) + "_pdyn.tavg")
    )[np.array(cells) - 1]
    w = pdyn / tavg_pdyn - 2.0
    x_wmean = np.sum(X * w) / np.sum(w) / r_e
    y_wmean = np.sum(Y * w) / np.sum(w) / r_e
    z_wmean = np.sum(Z * w) / np.sum(w) / r_e

    # Geometric center of jet in cartesian coordinates
    x_mean = np.nanmean(X) / r_e
    y_mean = np.nanmean(Y) / r_e
    z_mean = np.nanmean(Z) / r_e

    # Position of maximum velocity in cartesian coordinates
    x_max = X[vmag == max(vmag)][0] / r_e
    y_max = Y[vmag == max(vmag)][0] / r_e
    z_max = Z[vmag == max(vmag)][0] / r_e

    # Minimum x and density at maximum velocity
    x_min = min(X) / r_e
    rho_vmax = rho[vmag == max(vmag)][0]
    b_vmax = beta[vmag == max(vmag)][0]

    # calculate jet size
    A = dA * len(cells) / (r_e**2)
    Nr_cells = len(cells)

    # calculate linear sizes of jet
    size_rad = (max(X) - min(X)) / r_e + np.sqrt(dA) / r_e
    size_tan = A / size_rad

    coords_shifted = np.array(
        [X / r_e - x_wmean, Y / r_e - y_wmean, Z / r_e - z_wmean]
    ).T
    dist_vpar = np.dot(coords_shifted, avg_vu)
    dist_Bpar = np.dot(coords_shifted, avg_Bu)
    size_vpar = np.max(dist_vpar) - np.min(dist_vpar)
    size_vperp = A / size_vpar
    size_Bpar = np.max(dist_Bpar) - np.min(dist_Bpar)
    size_Bperp = A / size_Bpar

    [
        B_sheath_avg,
        TPar_sheath_avg,
        TPerp_sheath_avg,
        T_sheath_avg,
        n_sheath_avg,
        v_sheath_avg,
        pd_sheath_avg,
    ] = [
        np.nanmean(v)
        for v in [
            B_sheath_mag,
            TPar_sheath,
            TPerp_sheath,
            T_sheath,
            rho_sheath,
            v_sheath_mag,
            pd_sheath,
        ]
    ]

    temp_arr = [
        x_mean,
        y_mean,
        z_mean,
        x_wmean,
        y_wmean,
        z_wmean,
        A,
        Nr_cells,
        size_rad,
        size_tan,
        size_vpar,
        size_vperp,
        size_Bpar,
        size_Bperp,
        x_max,
        y_max,
        z_max,
        n_avg,
        n_med,
        n_max,
        v_avg,
        v_med,
        v_max,
        B_avg,
        B_med,
        B_max,
        T_avg,
        T_med,
        T_max,
        TPar_avg,
        TPar_med,
        TPar_max,
        TPerp_avg,
        TPerp_med,
        TPerp_max,
        beta_avg,
        beta_med,
        beta_max,
        x_min,
        rho_vmax,
        b_vmax,
        pd_avg,
        pd_med,
        pd_max,
        B_sheath_avg,
        TPar_sheath_avg,
        TPerp_sheath_avg,
        T_sheath_avg,
        n_sheath_avg,
        v_sheath_avg,
        pd_sheath_avg,
        is_upstream,
        ew_pd_enh,
        is_slams,
        is_jet,
        is_merger,
        is_splinter,
        at_bow_shock,
        at_slams,
        at_jet,
    ]

    return temp_arr
