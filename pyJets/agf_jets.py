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


class NeoTransient:
    # Class for identifying and handling individual jets and their properties

    def __init__(self, ID, runid, birthday, transient="jet"):
        self.ID = ID  # Should be a string of 5 digits
        self.runid = runid  # Should be a string of 3 letters
        self.birthday = birthday  # Should be a float of accuracy to half a second
        self.cellids = []
        self.times = [birthday]
        self.props = []
        self.meta = ["META"]
        self.merge_time = np.inf
        self.splinter_time = np.inf
        self.transient = transient

        if debug_g:
            print("Created transient with ID " + self.ID)

    def return_cellid_string(self):
        # Return string of lists of cellids for printing to file

        return "\n".join([",".join(list(map(str, l))) for l in self.cellids])

    def return_time_string(self):
        # Return string of times for printing to file

        return "\n".join(list(map(str, self.times)))

    def jetprops_write(self, start):
        if self.times[-1] - self.times[0] >= 0:
            t_arr = np.array(self.times)
            splinter_arr = (t_arr >= self.splinter_time).astype(int)
            merge_arr = (t_arr >= self.merge_time).astype(int)
            for n in range(t_arr.size):
                self.props[n][-5] = merge_arr[n]
                self.props[n][-4] = splinter_arr[n]
            propfile_write(
                self.runid,
                start,
                self.ID,
                self.props,
                self.meta,
                transient=self.transient,
            )
        else:
            if debug_g:
                print(
                    "Transient {} too short-lived, propfile not written!".format(
                        self.ID
                    )
                )

        return None


class PropReader:
    # Class for reading jet property files

    def __init__(self, ID, runid, start=781, fname=None, transient="jet"):
        # Check for transient type
        if transient == "jet":
            inputdir = wrkdir_DNR + "working/jets/jets"
        elif transient == "slamsjet":
            inputdir = wrkdir_DNR + "working/SLAMSJETS/slamsjets"
        elif transient == "slams":
            inputdir = wrkdir_DNR + "working/SLAMS/slams"

        self.ID = ID  # Should be a string of 5 digits
        self.runid = runid  # Should be a string of 3 letters
        self.start = start  # Should be a float of accuracy to half a second
        self.transient = transient
        self.meta = []
        self.sw_pars = sw_par_dict(runid)  # Solar wind parameters for run
        self.sw_pars[0] /= 1.0e6  # rho in 1/cm^3
        self.sw_pars[1] /= 1.0e3  # v in km/s
        self.sw_pars[2] /= 1.0e-9  # Pdyn in nPa
        self.sw_pars[3] /= 1.0e-9  # B in nT

        # Check if passing free-form filename to function
        if type(fname) is not str:
            self.fname = str(start) + "." + ID + ".props"
        else:
            self.fname = fname

        # Try opening file
        try:
            props_f = open(inputdir + "/" + runid + "/" + self.fname)
        except IOError:
            raise IOError("File not found!")

        props = props_f.read()
        props_f.close()
        props = props.split("\n")
        if "META" in props[0]:
            self.meta = props[0].split(",")[1:]
            props = props[2:]
        else:
            props = props[1:]
        props = [line.split(",") for line in props]
        self.props = np.asarray(props, dtype="float")
        # self.times = timefile_read(self.runid,self.start,self.ID,transient=self.transient)
        # self.cells = jetfile_read(self.runid,self.start,self.ID,transient=self.transient)

        # Initialise list of variable names and associated dictionary
        var_list = propfile_var_list
        n_list = list(range(len(var_list)))
        self.var_dict = dict(zip(var_list, n_list))

        self.delta_list = ["DT", "Dn", "Dv", "Dpd", "DB", "DTPar", "DTPerp"]
        self.davg_list = [
            "T_avg",
            "n_max",
            "v_max",
            "pd_max",
            "B_max",
            "TPar_avg",
            "TPerp_avg",
        ]
        self.sheath_list = [
            "T_sheath",
            "n_sheath",
            "v_sheath",
            "pd_sheath",
            "B_sheath",
            "TPar_sheath",
            "TPerp_sheath",
        ]

    def get_splin_times(self):
        splin_arr = [s for s in self.meta if s != "splinter"]
        splin_arr = [s[-5:] for s in splin_arr if "splin" in s]
        splin_arr = list(map(float, splin_arr))

        return splin_arr

    def get_times(self):
        return timefile_read(self.runid, self.start, self.ID, transient=self.transient)

    def get_cells(self):
        return jetfile_read(self.runid, self.start, self.ID, transient=self.transient)

    def read(self, name):
        # Read data of specified variable

        if name in self.var_dict:
            return self.props[:, self.var_dict[name]]
        elif name in self.delta_list:
            return self.read(self.davg_list[self.delta_list.index(name)]) - self.read(
                self.sheath_list[self.delta_list.index(name)]
            )
        # elif name == "x_wmean":
        #     return (
        #         np.loadtxt(
        #             wrkdir_DNR
        #             + "papu22/jet_prop_v_txts/{}_{}.txt".format(
        #                 self.runid, str(self.ID).zfill(5)
        #             )
        #         ).T[1]
        #         * 1e3
        #         / r_e
        #     )
        # elif name == "y_wmean":
        #     return (
        #         np.loadtxt(
        #             wrkdir_DNR
        #             + "papu22/jet_prop_v_txts/{}_{}.txt".format(
        #                 self.runid, str(self.ID).zfill(5)
        #             )
        #         ).T[2]
        #         * 1e3
        #         / r_e
        #     )
        elif name == "pdyn_vmax":
            return 1.0e21 * m_p * self.read("rho_vmax") * self.read("v_max") ** 2
        elif name == "duration":
            t = self.read("time")
            return np.ones(t.shape) * (t[-1] - t[0] + 0.5)
        elif name == "size_ratio":
            return self.read("size_rad") / self.read("size_tan")
        elif name == "death_distance":
            x, y, z = (
                self.read("x_vmax")[-1],
                self.read("y_vmax")[-1],
                self.read("z_vmax")[-1],
            )
            t = self.read("time")[-1]
            outp = np.ones(t.shape)
            pfit = bow_shock_jonas(self.runid, int(t * 2))[::-1]
            x_bs = np.polyval(pfit, np.linalg.norm([y, z]))
            return outp * (x - x_bs)
        elif name == "bs_distance":
            y, t = self.read("y_mean"), self.read("time")
            x_bs = np.zeros_like(y)
            for n in range(y.size):
                p = bow_shock_jonas(self.runid, int(t[n] * 2))[::-1]
                x_bs[n] = np.polyval(p, y[n])
            return x_bs
        elif name == "mp_distance":
            y, t = self.read("y_mean"), self.read("time")
            x_mp = np.zeros_like(y)
            for n in range(y.size):
                p = mag_pause_jonas(self.runid, int(t[n] * 2))[::-1]
                x_mp[n] = np.polyval(p, y[n])
            return x_mp
        elif name == "r_mean":
            x, y, z = self.read("x_mean"), self.read("y_mean"), self.read("z_mean")
            return np.linalg.norm([x, y, z], axis=0)
        elif name == "sep_from_bs":
            x, x_size, x_bs = (
                self.read("x_mean"),
                self.read("size_rad"),
                self.read("bs_distance"),
            )
            return np.abs(np.abs(x - x_bs) - x_size / 2.0)
        elif name == "parcel_speed":
            x, y = self.read("x_mean"), self.read("y_mean")
            vx = np.gradient(x, 0.5)
            vy = np.gradient(y, 0.5)
            return np.array([vx, vy]).T
        elif name == "first_cone":
            t = self.read("time")
            x, y = self.read("x_mean")[0], self.read("y_mean")[0]
            cone_angle = np.rad2deg(np.arctan2(y, x))
            return np.ones_like(t) * cone_angle
        elif name == "first_y":
            t = self.read("time")
            y = self.read("y_mean")[0]
            return np.ones_like(t) * y
        elif name == "final_cone":
            t = self.read("time")
            x, y = self.read("x_mean")[-1], self.read("y_mean")[-1]
            cone_angle = np.rad2deg(np.arctan2(y, x))
            return np.ones_like(t) * cone_angle
        elif name == "leaves_bs":
            at_bow_shock = self.read("at_bow_shock")
            leaves_bs = int((at_bow_shock == 0).any())
            return np.ones_like(at_bow_shock) * leaves_bs
        elif name == "dies_at_bs":
            at_bow_shock = self.read("at_bow_shock")
            dies_at_bs = int(at_bow_shock[-1] == 1)
            return np.ones_like(at_bow_shock) * dies_at_bs
        else:
            print("Variable not found!")
            return None

    def amax_index(self):
        # Return list index of time when area is largest

        return self.read("A").argmax()

    def time_index(self, time):
        # Return list index of specified time

        time_arr = self.read("time")
        if time not in time_arr:
            raise IOError("Time not found!")
        else:
            return time_arr.tolist().index(time)

    def read_at_time(self, var, time):
        # Return variable data at specified time

        return self.read(var)[self.time_index(time)]

    def read_at_randt(self, var):
        time_arr = self.read("time")
        randt = choice(time_arr)

        return self.read_at_time(var, randt)

    def read_at_amax(self, name):
        # Return variable data at time when area is largest

        return self.read(name)[self.amax_index()]

    def read_at_lastbs(self, name):
        t0 = self.read("time")[self.read("at_bow_shock") == 1][-1]
        return self.read_at_time(name, t0)


def mask_maker(runid, filenr, boxre=[-10, 20, -20, 20], avgfile=True, mag_thresh=1.5):
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

    sw_pars = sw_par_dict(runid)
    rho_sw = sw_pars[0]
    v_sw = sw_pars[1]
    pdyn_sw = sw_pars[3]
    B_sw = sw_pars[2]

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
    boxre=[-10, 20, -20, 20],
    maskfile=False,
    avgfile=True,
    nbrs=[2, 2, 0],
    mag_thresh=1.1,
):
    runid_list = ["ABA", "ABC", "AEA", "AEC", "AGF", "AIA"]
    maxfnr_list = [839, 1179, 1339, 879, 1193, 1193]
    if start > maxfnr_list[runid_list.index(runid)]:
        return 0

    global runid_g
    global filenr_g
    runid_g = runid

    global rho_sw_g

    rho_sw_g = sw_par_dict(runid)[0]

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
        if file_nr > maxfnr_list[runid_list.index(runid)]:
            break

        filenr_g = file_nr

        # find correct file based on file number and run id

        bulkname = "bulk." + str(file_nr).zfill(7) + ".vlsv"

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file " + str(file_nr) + " not found, continuing")
            continue

        # open vlsv file for reading
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath + bulkname)

        # vlsvobj.optimize_open_file()

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
        # jets, props_inc = jet_sorter(
        #     vlsvobj,
        #     jet_msk,
        #     slams_msk,
        #     slamsjet_msk,
        #     up_cells,
        #     down_cells,
        #     up_cells_mms,
        #     down_cells_mms,
        #     neighborhood_reach=nbrs,
        # )
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
        # slams_jets, slams_props_inc = jet_sorter(
        #     vlsvobj,
        #     jet_msk,
        #     slams_msk,
        #     slams_msk,
        #     up_cells,
        #     down_cells,
        #     up_cells_mms,
        #     down_cells_mms,
        #     neighborhood_reach=nbrs,
        # )

        # props = [[float(file_nr) / 2.0] + line for line in props_inc]
        jet_props = [[float(file_nr) / 2.0] + line for line in jet_props_inc]
        # slams_props = [[float(file_nr) / 2.0] + line for line in slams_props_inc]

        # print(len(jet_props))
        # print(len(jet_jets))
        #
        # print(len(slams_props))
        # print(len(slams_jets))

        # eventprop_write(runid, file_nr, props, transient="slamsjet")
        # eventprop_write(runid, file_nr, slams_props, transient="slams")
        eventprop_write(runid, file_nr, jet_props, transient="jet")

        # erase contents of output file

        # open(
        #     wrkdir_DNR
        #     + "working/SLAMSJETS/events/"
        #     + runid
        #     + "/"
        #     + str(file_nr)
        #     + ".events",
        #     "w",
        # ).close()

        # # open output file
        # fileobj = open(
        #     wrkdir_DNR
        #     + "working/SLAMSJETS/events/"
        #     + runid
        #     + "/"
        #     + str(file_nr)
        #     + ".events",
        #     "a",
        # )

        # # write jets to outputfile
        # for jet in jets:
        #     fileobj.write(",".join(list(map(str, jet))) + "\n")

        # fileobj.close()

        # open(
        #     wrkdir_DNR
        #     + "working/SLAMS/events/"
        #     + runid
        #     + "/"
        #     + str(file_nr)
        #     + ".events",
        #     "w",
        # ).close()
        # fileobj_slams = open(
        #     wrkdir_DNR
        #     + "working/SLAMS/events/"
        #     + runid
        #     + "/"
        #     + str(file_nr)
        #     + ".events",
        #     "a",
        # )

        # for slams_jet in slams_jets:
        #     fileobj_slams.write(",".join(list(map(str, slams_jet))) + "\n")

        # fileobj_slams.close()

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

        # vlsvobj.optimize_close_file()

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
        "proton/vg_Pdyn",
        "proton/vg_beta",
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
        tavgdir + "/" + runid_g + "/" + str(filenr_g) + "_pdyn.tavg"
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
    size_vpar = np.max(dist_vpar) - np.min(dist_vpar) + np.sqrt(dA) / r_e
    size_vperp = A / size_vpar
    size_Bpar = np.max(dist_Bpar) - np.min(dist_Bpar) + np.sqrt(dA) / r_e
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


def eventprop_write(runid, filenr, props, transient="jet"):
    if transient == "jet":
        outputdir = wrkdir_DNR + "working/jets/event_props/" + runid
    elif transient == "slams":
        outputdir = wrkdir_DNR + "working/SLAMS/event_props/" + runid
    elif transient == "slamsjet":
        outputdir = wrkdir_DNR + "working/SLAMSJETS/event_props/" + runid

    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    # print(len(props))

    open(outputdir + "/{}.eventprops".format(str(filenr)), "w").close()
    epf = open(outputdir + "/{}.eventprops".format(str(filenr)), "a")

    epf.write(propfile_header_list + "\n")

    epf.write("\n".join([",".join(list(map(str, line))) for line in props]))
    epf.close()
    print("Wrote to " + outputdir + "/{}.eventprops".format(str(filenr)))


def timefile_read(runid, filenr, key, transient="jet"):
    # Read array of times from file

    # Check for transient type
    if transient == "jet":
        inputdir = wrkdir_DNR + "working/jets/jets"
    elif transient == "slamsjet":
        inputdir = wrkdir_DNR + "working/SLAMSJETS/slamsjets"
    elif transient == "slams":
        inputdir = wrkdir_DNR + "working/SLAMS/slams"

    tf = open("{}/{}/{}.{}.times".format(inputdir, runid, str(filenr), key), "r")
    contents = tf.read().split("\n")
    tf.close()

    return list(map(float, contents))


def jetfile_read(runid, filenr, key, transient="jet"):
    # Read array of cellids from file

    # Check for transient type
    if transient == "jet":
        inputdir = wrkdir_DNR + "working/jets/jets"
        extension = "jet"
    elif transient == "slamsjet":
        inputdir = wrkdir_DNR + "working/SLAMSJETS/slamsjets"
        extension = "slamsjet"
    elif transient == "slams":
        inputdir = wrkdir_DNR + "working/SLAMS/slams"
        extension = "slams"

    outputlist = []

    jf = open(
        "{}/{}/{}.{}.{}".format(inputdir, runid, str(filenr), key, extension), "r"
    )
    contents = jf.read()
    jf.close()
    lines = contents.split("\n")

    for line in lines:
        outputlist.append(list(map(int, line.split(","))))

    return outputlist


def eventfile_read(runid, filenr, transient="jet"):
    # Read array of arrays of cellids from file

    if transient == "jet":
        inputdir = wrkdir_DNR + "working/jets/events"
    elif transient == "slams":
        inputdir = wrkdir_DNR + "working/SLAMS/events"
    elif transient == "slamsjet":
        inputdir = wrkdir_DNR + "working/SLAMSJETS/events"

    outputlist = []

    ef = open("{}/{}/{}.events".format(inputdir, runid, str(filenr)), "r")
    contents = ef.read().strip("\n")
    ef.close()
    if contents == "":
        return []
    lines = contents.split("\n")

    for line in lines:
        outputlist.append(list(map(int, line.split(","))))

    return outputlist


def eventprop_read(runid, filenr, transient="jet"):
    if transient == "jet":
        inputname = wrkdir_DNR + "working/jets/event_props/{}/{}.eventprops".format(
            runid, str(filenr)
        )
    elif transient == "slams":
        inputname = wrkdir_DNR + "working/SLAMS/event_props/{}/{}.eventprops".format(
            runid, str(filenr)
        )
    elif transient == "slamsjet":
        inputname = (
            wrkdir_DNR
            + "working/SLAMSJETS/event_props/{}/{}.eventprops".format(
                runid, str(filenr)
            )
        )

    try:
        props_f = open(inputname)
    except IOError:
        raise IOError("File not found!")

    props = props_f.read()
    props_f.close()
    props = props.split("\n")[1:]
    if props == [] or props == [""]:
        return []
    props = [list(map(float, line.split(","))) for line in props]

    return props


def propfile_write(runid, filenr, key, props, meta, transient="jet"):
    # Write jet properties to file

    if transient == "jet":
        outputdir = wrkdir_DNR + "working/jets/jets"
    elif transient == "slams":
        outputdir = wrkdir_DNR + "working/SLAMS/slams"
    elif transient == "slamsjet":
        outputdir = wrkdir_DNR + "working/SLAMSJETS/slamsjets"

    open(
        outputdir + "/" + runid + "/" + str(filenr) + "." + key + ".props", "w"
    ).close()
    pf = open(outputdir + "/" + runid + "/" + str(filenr) + "." + key + ".props", "a")
    pf.write(",".join(meta) + "\n")
    pf.write(propfile_header_list + "\n")
    pf.write("\n".join([",".join(list(map(str, line))) for line in props]))
    pf.close()
    if debug_g:
        print(
            "Wrote to "
            + outputdir
            + "/"
            + runid
            + "/"
            + str(filenr)
            + "."
            + key
            + ".props"
        )


def check_threshold(A, B, thresh):
    return np.intersect1d(A, B).size > thresh * min(len(A), len(B))


def jet_tracker(runid, start, stop, threshold=0.5, transient="jet", dbg=False):
    if transient == "slamsjet":
        outputdir = wrkdir_DNR + "working/SLAMSJETS/slamsjets/" + runid
    elif transient == "jet":
        outputdir = wrkdir_DNR + "working/jets/jets/" + runid
    elif transient == "slams":
        outputdir = wrkdir_DNR + "working/SLAMS/slams/" + runid

    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    global debug_g

    debug_g = dbg

    # Read initial event files
    events_old = eventfile_read(runid, start, transient=transient)
    old_props = eventprop_read(runid, start, transient=transient)
    events_unsrt = eventfile_read(runid, start + 1, transient=transient)
    props_unsrt = eventprop_read(runid, start + 1, transient=transient)

    # Initialise list of jet objects
    jetobj_list = []
    dead_jetobj_list = []

    # Initialise unique ID counter
    counter = 1

    # Print current time
    if dbg:
        print("t = " + str(float(start + 1) / 2) + "s")

    # Look for jets at bow shock
    for event in events_unsrt:
        for old_event in events_old:
            if check_threshold(old_event, event, threshold):
                # Create unique ID
                curr_id = str(counter).zfill(5)

                # Create new jet object

                jetobj_new = NeoTransient(
                    curr_id, runid, float(start) / 2, transient=transient
                )

                # print(len(props_unsrt))
                # print(len(events_unsrt))
                # print(events_unsrt.index(event))

                # Append current events to jet object properties
                jetobj_new.cellids.append(old_event)
                jetobj_new.cellids.append(event)
                jetobj_new.props.append(old_props[events_old.index(old_event)])
                jetobj_new.props.append(props_unsrt[events_unsrt.index(event)])
                jetobj_new.times.append(float(start + 1) / 2)

                jetobj_list.append(jetobj_new)

                # Iterate counter
                counter += 1

                break

    # Track jets
    for n in range(start + 2, stop + 1):
        for jetobj in jetobj_list:
            if float(n) / 2 - jetobj.times[-1] + 0.5 > 2.5:
                if dbg:
                    print("Killing jet {}".format(jetobj.ID))
                dead_jetobj_list.append(jetobj)
                jetobj_list.remove(jetobj)

        # Print  current time
        if dbg:
            print("t = " + str(float(n) / 2) + "s")

        events_old = events_unsrt
        old_props = props_unsrt

        # Initialise flags for finding splintering jets
        flags = []

        # Read event file for current time step
        events_unsrt = eventfile_read(runid, n, transient=transient)
        props_unsrt = eventprop_read(runid, n, transient=transient)
        events = sorted(events_unsrt, key=len)
        events = events[::-1]

        # Iniatilise list of cells currently being tracked
        curr_jet_temp_list = []

        # Update existing jets
        for event in events:
            for jetobj in jetobj_list:
                if jetobj.ID not in flags:
                    if check_threshold(jetobj.cellids[-1], event, threshold):
                        # Append event to jet object properties
                        jetobj.cellids.append(event)
                        jetobj.props.append(props_unsrt[events_unsrt.index(event)])
                        jetobj.times.append(float(n) / 2)
                        # print("Updated jet "+jetobj.ID)

                        flags.append(jetobj.ID)
                        if event in curr_jet_temp_list:
                            if "merger" not in jetobj.meta:
                                jetobj.meta.append("merger")
                                jetobj.merge_time = float(n) / 2
                        else:
                            curr_jet_temp_list.append(event)

                else:
                    if event not in curr_jet_temp_list:
                        if check_threshold(jetobj.cellids[-2], event, threshold):
                            curr_id = str(counter).zfill(5)
                            # Iterate counter
                            counter += 1

                            jetobj_new = deepcopy(jetobj)
                            jetobj_new.ID = curr_id
                            if "splinter" not in jetobj_new.meta:
                                jetobj_new.meta.append("splinter")
                                jetobj_new.splinter_time = float(n) / 2
                            else:
                                jetobj.meta.append("splin{}".format(str(float(n) / 2)))
                            jetobj_new.cellids[-1] = event
                            jetobj_new.props[-1] = props_unsrt[
                                events_unsrt.index(event)
                            ]
                            jetobj_list.append(jetobj_new)
                            curr_jet_temp_list.append(event)

                            break
                        else:
                            continue

                    else:
                        continue

        # Look for new jets at bow shock
        for event in events:
            if event not in curr_jet_temp_list:
                for old_event in events_old:
                    if check_threshold(old_event, event, threshold):
                        # Create unique ID
                        curr_id = str(counter).zfill(5)

                        # Create new jet object
                        jetobj_new = NeoTransient(
                            curr_id, runid, float(n - 1) / 2, transient=transient
                        )

                        # Append current events to jet object properties
                        jetobj_new.cellids.append(old_event)
                        jetobj_new.cellids.append(event)
                        jetobj_new.props.append(old_props[events_old.index(old_event)])
                        jetobj_new.props.append(props_unsrt[events_unsrt.index(event)])
                        jetobj_new.times.append(float(n) / 2)

                        jetobj_list.append(jetobj_new)

                        # Iterate counter
                        counter += 1

                        break

    jetobj_list = jetobj_list + dead_jetobj_list

    for jetobj in jetobj_list:
        # Write jet object cellids and times to files
        jetfile = open(
            outputdir + "/" + str(start) + "." + jetobj.ID + "." + transient, "w"
        )
        timefile = open(outputdir + "/" + str(start) + "." + jetobj.ID + ".times", "w")

        jetfile.write(jetobj.return_cellid_string())
        timefile.write(jetobj.return_time_string())
        jetobj.jetprops_write(start)

        jetfile.close()
        timefile.close()

    return None


def ext_jet(ax, XmeshXY, YmeshXY, pass_maps):
    B = pass_maps["vg_b_vol"]
    rho = pass_maps["proton/vg_rho"]
    cellids = pass_maps["CellID"]
    mmsx = pass_maps["proton/vg_mmsx"]
    core_heating = pass_maps["proton/vg_core_heating"]
    Bmag = np.linalg.norm(B, axis=-1)
    Pdyn = pass_maps["proton/vg_Pdyn"]
    Pdynx = pass_maps["proton/vg_Pdynx"]
    beta_star = pass_maps["proton/vg_beta_star"]
    By = B[:, :, 1]

    # try:
    #     slams_cells = np.loadtxt(
    #         "/wrk-vakka/users/jesuni/foreshock_bubble/working/SLAMS/Masks/{}/{}.mask".format(
    #             runid_g, int(filenr_g)
    #         )
    #     ).astype(int)
    # except:
    #     slams_cells = []
    try:
        jet_cells = np.loadtxt(
            "/wrk-vakka/users/jesuni/foreshock_bubble/working/jets/Masks/{}/{}.mask".format(
                runid_g, int(filenr_g)
            )
        ).astype(int)
    except:
        jet_cells = []

    # sj_jetobs = [
    #     PropReader(str(int(sj_id)).zfill(5), runid_g, transient="jet")
    #     for sj_id in sj_ids_g
    # ]
    non_sjobs = [
        PropReader(str(int(non_id)).zfill(5), runid_g, transient="jet")
        for non_id in non_ids_g
    ]

    sj_xlist = []
    sj_ylist = []
    non_xlist = []
    non_ylist = []

    # for jetobj in sj_jetobs:
    #     if filenr_g / 2.0 in jetobj.read("time"):
    #         sj_xlist.append(jetobj.read_at_time("x_wmean", filenr_g / 2.0))
    #         sj_ylist.append(jetobj.read_at_time("y_wmean", filenr_g / 2.0))
    for jetobj in non_sjobs:
        if filenr_g / 2.0 in jetobj.read("time"):
            non_xlist.append(jetobj.read_at_time("x_wmean", filenr_g / 2.0))
            non_ylist.append(jetobj.read_at_time("y_wmean", filenr_g / 2.0))

    for idx in range(len(xg)):
        ax.plot(xg[idx], yg[idx], "x", color=CB_color_cycle[idx])

    # slams_mask = np.in1d(cellids, slams_cells).astype(int)
    # slams_mask = np.reshape(slams_mask, cellids.shape)

    jet_mask = np.in1d(cellids, jet_cells).astype(int)
    jet_mask = np.reshape(jet_mask, cellids.shape)

    ch_mask = (core_heating > 3 * T_sw).astype(int)
    mach_mask = (mmsx < 1).astype(int)
    rho_mask = (rho > 2 * rho_sw).astype(int)

    plaschke_mask = (Pdynx > 0.25 * Pdyn_sw).astype(int)
    plaschke_mask[core_heating < 3 * T_sw] = 0

    cav_shfa_mask = (Bmag < 0.8 * B_sw).astype(int)
    cav_shfa_mask[rho >= 0.8 * rho_sw] = 0

    diamag_mask = (Pdyn >= 1.2 * Pdyn_sw).astype(int)
    diamag_mask[Bmag > B_sw] = 0

    # CB_color_cycle

    # start_points = np.array(
    #     [np.ones(20) * x0 + 0.5, np.linspace(y0 - 0.9, y0 + 0.9, 20)]
    # ).T
    # nstp = 40
    # start_points = np.array([np.ones(nstp) * 17, np.linspace(-20, 20, nstp)]).T

    if Blines_g:
        blines_bx = np.copy(B[:, :, 0])
        blines_by = np.copy(B[:, :, 1])
        blines_bx[core_heating > 3 * T_sw] = np.nan
        blines_by[core_heating > 3 * T_sw] = np.nan
        stream = ax.streamplot(
            XmeshXY,
            YmeshXY,
            blines_bx,
            blines_by,
            arrowstyle="-",
            broken_streamlines=False,
            color="k",
            linewidth=0.4,
            # minlength=4,
            density=35,
            start_points=start_points,
        )

    lws = 0.6
    mrks = 2
    mews = 0.4

    if drawBy0:
        by_mask = np.ones_like(By, dtype=int)
        by_mask[np.logical_and(By > 0, YmeshXY < 0)] = 0
        by_mask[np.logical_and(By < 0, YmeshXY > 0)] = 0

        # by_mask[YmeshXY < 0] = 0
        by_mask[beta_star < 0.3] = -1
        by_mask[core_heating < 3 * T_sw] = -1

        by_cont = ax.contourf(
            XmeshXY,
            YmeshXY,
            by_mask,
            [-0.5, 0.5],
            # linewidths=lws,
            colors=[CB_color_cycle[6], CB_color_cycle[8]],
            # linestyles=["dashed"],
            hatches=["\\", "/"],
            alpha=0.3,
        )

        by0_cont = ax.contour(
            XmeshXY,
            YmeshXY,
            By,
            [0],
            linewidths=lws,
            colors="red",
            linestyles=["dashed"],
        )

    jet_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        jet_mask,
        [0.5],
        linewidths=lws,
        colors=CB_color_cycle[2],
        linestyles=["solid"],
    )

    # ch_cont = ax.contour(
    #     XmeshXY,
    #     YmeshXY,
    #     ch_mask,
    #     [0.5],
    #     linewidths=lws,
    #     colors=CB_color_cycle[1],
    #     linestyles=["solid"],
    # )
    bs_cont = ax.contour(
        XmeshXY,
        YmeshXY,
        beta_star,
        [0.3],
        linewidths=lws,
        colors=CB_color_cycle[1],
        linestyles=["solid"],
    )

    if plaschke_g:
        plaschke_cont = ax.contour(
            XmeshXY,
            YmeshXY,
            plaschke_mask,
            [0.5],
            linewidths=lws,
            colors=CB_color_cycle[7],
            linestyles=["solid"],
        )

    # slams_cont = ax.contour(
    #     XmeshXY,
    #     YmeshXY,
    #     slams_mask,
    #     [0.5],
    #     linewidths=lws,
    #     colors=CB_color_cycle[7],
    #     linestyles=["solid"],
    # )

    # rho_cont = ax.contour(
    #     XmeshXY,
    #     YmeshXY,
    #     rho_mask,
    #     [0.5],
    #     linewidths=lws,
    #     colors=CB_color_cycle[3],
    #     linestyles=["solid"],
    # )

    # mach_cont = ax.contour(
    #     XmeshXY,
    #     YmeshXY,
    #     mach_mask,
    #     [0.5],
    #     linewidths=lws,
    #     colors=CB_color_cycle[4],
    #     linestyles=["solid"],
    # )

    (non_pos,) = ax.plot(
        non_xlist,
        non_ylist,
        "o",
        color="black",
        markersize=mrks,
        markeredgecolor="white",
        fillstyle="full",
        mew=mews,
        label="Tracked jet",
    )
    # (sj_pos,) = ax.plot(
    #     sj_xlist,
    #     sj_ylist,
    #     "o",
    #     color="red",
    #     markersize=mrks,
    #     markeredgecolor="white",
    #     fillstyle="full",
    #     mew=mews,
    #     label="FCS-jet",
    # )

    # itr_jumbled = [3, 1, 4, 2, 7]

    itr_jumbled = [1, 1, 4, 2, 7]
    itr_jumbled = [1, 7, 4, 2, 7]

    # proxy = [
    #     plt.Rectangle((0, 0), 1, 1, fc=CB_color_cycle[itr_jumbled[itr]])
    #     for itr in range(5)
    # ] + [non_pos, sj_pos]

    # proxy = [
    #     mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[itr]])
    #     for itr in range(5)
    # ] + [non_pos, sj_pos]

    # proxy_labs = (
    #         "$n=2n_\mathrm{sw}$",
    #         "$T_\mathrm{core}=3T_\mathrm{sw}$",
    #         "$M_{\mathrm{MS},x}=1$",
    #         "Jet",
    #         "FCS",
    #         "Non-FCS jet",
    #         "FCS-jet"
    #     )

    proxy_labs = [
        # "$n=2n_\mathrm{sw}$",
        # "$T_\mathrm{core}=3T_\mathrm{sw}$",
        "$\\beta^* = 0.3$",
        # "$M_{\mathrm{MS},x}=1$",
        # "$P_\mathrm{dyn,x}>0.25 P_\mathrm{dyn,sw}$",
    ]

    proxy = [
        mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[itr]])
        for itr in range(len(proxy_labs))
    ]

    xmin, xmax, ymin, ymax = (
        np.min(XmeshXY),
        np.max(XmeshXY),
        np.min(YmeshXY),
        np.max(YmeshXY),
    )

    if plaschke_g:
        proxy.append(mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[7]]))
        proxy_labs.append("$P_\mathrm{dyn,x}>0.25 P_\mathrm{dyn,sw}$")
    if ~(jet_mask == 0).all():
        proxy.append(mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[3]]))
        proxy_labs.append(
            "$P_\mathrm{dyn} \geq 2 \\langle P_\mathrm{dyn} \\rangle_\mathrm{3min}$"
        )
    # if ~(slams_mask == 0).all():
    #     proxy.append(mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[4]]))
    #     proxy_labs.append("FCS")
    if Blines_g:
        proxy.append(mlines.Line2D([], [], color="k"))
        proxy_labs.append("$B$")
    if drawBy0:
        proxy.append(mlines.Line2D([], [], color="red", linestyle="dashed"))
        proxy_labs.append("$B_y=0$")
        proxy.append(
            plt.Rectangle(
                (-100, -100),
                1,
                1,
                fc=CB_color_cycle[6],
                # color="black",
                fill=True,
                hatch="\\",
                alpha=0.3,
            )
        )
        proxy_labs.append("Q$\\perp$ sheath")
    if np.logical_and(
        np.logical_and(non_xlist >= xmin, non_xlist <= xmax),
        np.logical_and(non_ylist >= ymin, non_ylist <= ymax),
    ).any():
        proxy.append(non_pos)
        proxy_labs.append("Tracked jet")
    # if np.logical_and(
    #     np.logical_and(sj_xlist >= xmin, sj_xlist <= xmax),
    #     np.logical_and(sj_ylist >= ymin, sj_ylist <= ymax),
    # ).any():
    #     proxy.append(sj_pos)
    #     proxy_labs.append("FCS-jet")

    ax.legend(
        proxy,
        proxy_labs,
        frameon=True,
        numpoints=1,
        markerscale=1,
        loc="lower left",
        fontsize=5,
    )

    global gprox, gprox_labs

    gprox = proxy
    gprox_labs = proxy_labs


def get_jets(runid):
    non_ids = []

    singular_counter = 0

    for n1 in range(6000):
        try:
            props = PropReader(str(n1).zfill(5), runid, transient="jet")
        except:
            continue

        if props.read("at_bow_shock")[0] != 1:
            continue

        # if props.read("time")[0] == 290.0:
        #     continue

        # if props.read("time")[-1] - props.read("time")[0] == 0:
        #     singular_counter += 1
        #     continue

        # if "splinter" in props.meta:
        #     continue

        # if (props.read("at_slams") == 1).any():
        #     continue
        # else:
        #     non_ids.append(n1)
        non_ids.append(n1)

    # print("Run {} singular non jets: {}".format(runid, singular_counter))

    return np.unique(non_ids)


def fig1(runid):
    var = "proton/vg_Pdyn"
    vscale = 1e9
    vmax = 1.0
    runids = ["AGF", "AIA"]

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g, start_points, drawBy0, plaschke_g
    runid_g = runid
    Blines_g = True
    drawBy0 = True
    plaschke_g = False

    start_points = np.array(
        # [np.ones(nstp) * boxre[1] - 1, np.linspace(boxre[2], boxre[3], nstp)]
        [
            np.ones(10) * 16,
            np.linspace(-14, 14, 10),
        ]
    ).T

    global xg, yg
    xg = []
    yg = []

    bulkpath = find_bulkpath(runid)

    non_ids = get_jets(runid)

    sj_ids_g = []
    non_ids_g = non_ids

    pdmax = [1.5, 1.5][runids.index(runid)]
    sw_pars = [
        [1e6, 750e3, 3e-9, 0.5e6],
        [1e6, 750e3, 3e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runids.index(runid)]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw

    outputdir = wrkdir_DNR + "Figs/pdfs/"
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    fig, ax_list = plt.subplots(1, 4, figsize=(8, 4), constrained_layout=True)

    legon = [True, False, False, False]
    nocb = [True, True, True, False]

    for idx, fnr in enumerate([820, 880, 935, 1190]):
        filenr_g = fnr

        fname = "bulk.{}.vlsv".format(str(int(fnr)).zfill(7))

        pt.plot.plot_colormap(
            filename=bulkpath + fname,
            axes=ax_list[idx],
            var=var,
            vmin=0.01,
            # vmax=1,
            vmax=vmax,
            vscale=vscale,
            # cbtitle="",
            # cbtitle="",
            usesci=0,
            # scale=3,
            title="$t = {}s".format(float(fnr) / 2.0),
            boxre=[0, 18, -15, 15],
            internalcb=True,
            nocb=nocb[idx],
            lin=None,
            colormap="batlow",
            tickinterval=3.0,
            fsaved=None,
            # useimshow=True,
            external=ext_jet,
            useimshow=True,
            # expression=expr_rhoratio,
            pass_vars=[
                "proton/vg_rho_thermal",
                "proton/vg_rho_nonthermal",
                "proton/vg_ptensor_thermal_diagonal",
                "vg_b_vol",
                "proton/vg_v",
                "proton/vg_rho",
                "proton/vg_core_heating",
                "CellID",
                "proton/vg_mmsx",
                "proton/vg_Pdyn",
                "proton/vg_Pdynx",
                "proton/vg_beta_star",
            ],
            # streamlines="vg_b_vol",
            # streamlinedensity=0.4,
            # streamlinecolor="red",
            # streamlinethick=0.7,
        )

        if not legon[idx]:
            ax_list[idx].get_legend().remove()

    for ax in ax_list:
        ax.label_outer()

    fig.savefig(outputdir + "fig1.pdf")
    plt.close(fig)


def v5_plotter(
    runid,
    start,
    stop,
    boxre=[-10, 20, -20, 20],
    tickint=5.0,
    blines=True,
    nstp=40,
    pdynmax=1.5,
    pdynmin=0.1,
    outdir="cmaps",
    pointsx=[],
    pointsy=[],
    fsaved=None,
    lin=1,
):
    var = "proton/vg_Pdyn"
    vscale = 1e9
    vmax = pdynmax
    runids = ["AGF", "AIA"]

    if len(pointsx) != len(pointsy):
        print("x and y must have same length!")
        return 1

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g, start_points, drawBy0
    runid_g = runid
    Blines_g = blines
    drawBy0 = True

    global xg, yg
    xg = pointsx
    yg = pointsy

    # nstp = 40
    start_points = np.array(
        # [np.ones(nstp) * boxre[1] - 1, np.linspace(boxre[2], boxre[3], nstp)]
        [
            np.linspace(boxre[0] + 0.1, boxre[1] - 0.1, nstp),
            np.ones(nstp) * (boxre[2] + 1),
        ]
    ).T

    bulkpath = find_bulkpath(runid)

    non_ids = get_jets(runid)

    sj_ids_g = []
    non_ids_g = non_ids

    pdmax = [1.5, 1.5][runids.index(runid)]
    sw_pars = [
        [1e6, 750e3, 3e-9, 0.5e6],
        [1e6, 750e3, 3e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runids.index(runid)]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw

    outputdir = wrkdir_DNR + "Figs/{}/".format(outdir)
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    # global x0, y0
    # props = jio.PropReader(str(jetid).zfill(5), runid, transient="jet")
    # t0 = props.read("time")[0]
    # x0 = props.read("x_wmean")[0]
    # y0 = props.read("y_wmean")[0]
    # fnr0 = int(t0 * 2)

    for fnr in range(start, stop + 1):
        filenr_g = fnr

        fname = "bulk.{}.vlsv".format(str(int(fnr)).zfill(7))

        pt.plot.plot_colormap(
            filename=bulkpath + fname,
            outputfile=outputdir + "pdyn_{}.png".format(str(fnr).zfill(7)),
            var=var,
            vmin=pdynmin,
            # vmax=1,
            vmax=vmax,
            vscale=vscale,
            # cbtitle="",
            # cbtitle="",
            usesci=0,
            # scale=3,
            title="Run: {}$~$t = {}s".format(runid, float(fnr) / 2.0),
            boxre=boxre,
            internalcb=False,
            lin=lin,
            colormap="batlow",
            tickinterval=tickint,
            fsaved=fsaved,
            # useimshow=True,
            external=ext_jet,
            # expression=expr_rhoratio,
            pass_vars=[
                "proton/vg_rho_thermal",
                "proton/vg_rho_nonthermal",
                "proton/vg_ptensor_thermal_diagonal",
                "vg_b_vol",
                "proton/vg_v",
                "proton/vg_rho",
                "proton/vg_core_heating",
                "CellID",
                "proton/vg_mmsx",
                "proton/vg_Pdyn",
                "proton/vg_Pdynx",
                "proton/vg_beta_star",
            ],
        )


def VSC_timeseries(runid, x0, y0, t0, tpm=20, pdavg=True):
    bulkpath = find_bulkpath(runid)

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
    plot_labels = [
        None,
        "$v_x$",
        "$v_y$",
        "$v_z$",
        "$|v|$",
        "$P_\mathrm{dyn}$",
        "$B_x$",
        "$B_y$",
        "$B_z$",
        "$|B|$",
        "$E_x$",
        "$E_y$",
        "$E_z$",
        "$|E|$",
        "$T_\\parallel$",
        "$T_\\perp$",
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
    draw_legend = [
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        True,
    ]
    ylabels = [
        "$\\rho~[\mathrm{cm}^{-3}]$",
        "$v~[\mathrm{km/s}]$",
        "$P_\mathrm{dyn}~[\mathrm{nPa}]$",
        "$B~[\mathrm{nT}]$",
        "$E~[\mathrm{mV/m}]$",
        "$T~[\mathrm{MK}]$",
        # "$\\rho~[\\rho_\mathrm{sw}]$",
        # "$v~[v_\mathrm{sw}]$",
        # "$P_\mathrm{dyn}~[P_\mathrm{dyn,sw}]$",
        # "$B~[B_\mathrm{IMF}]$",
        # "$E~[E_\mathrm{sw}]$",
        # "$T~[T_\mathrm{sw}]$",
    ]
    e_sw = 750e3 * 3e-9 * q_p / m_p * 1e3
    norm = [
        [
            1,
            750,
            750,
            750,
            750,
            0.9408498320756251,
            3,
            3,
            3,
            3,
            e_sw,
            e_sw,
            e_sw,
            e_sw,
            0.5,
            0.5,
        ],
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
    plot_index = [0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5]
    plot_colors = [
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        "k",
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
    ]

    run_norm = norm[0]

    t_arr = np.arange(t0 - tpm, t0 + tpm + 0.1, 0.5)
    fnr0 = int(t0 * 2)
    fnr_arr = np.arange(fnr0 - 2 * tpm, fnr0 + 2 * tpm + 1)
    cellid = pt.vlsvfile.VlsvReader(
        bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
    ).get_cellid([x0 * r_e, y0 * r_e, 0 * r_e])
    data_arr = np.zeros((len(var_list), fnr_arr.size), dtype=float)
    tavg_arr = np.zeros(fnr_arr.size, dtype=float)

    for idx, fnr in enumerate(fnr_arr):
        if pdavg:
            try:
                tavg_pdyn = np.loadtxt(
                    tavgdir + "/" + runid + "/" + str(fnr) + "_pdyn.tavg"
                )[int(cellid) - 1]
            except:
                tavg_pdyn = np.nan
            tavg_arr[idx] = tavg_pdyn * scales[5]  # / run_norm[5]
        try:
            vlsvobj = pt.vlsvfile.VlsvReader(
                bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
            )
            for idx2, var in enumerate(var_list):
                data_arr[idx2, idx] = (
                    vlsvobj.read_interpolated_variable(
                        var, [x0 * r_e, y0 * r_e, 0], operator=ops[idx2]
                    )
                    * scales[idx2]
                    # / run_norm[idx2]
                )
        except:
            data_arr[:, idx] = np.nan

    fig, ax_list = plt.subplots(
        len(ylabels), 1, sharex=True, figsize=(6, 8), constrained_layout=True
    )
    ax_list[0].set_title("Run: {}, $x_0$: {}, $y_0$: {}".format(runid, x0, y0))
    for idx in range(len(var_list)):
        ax = ax_list[plot_index[idx]]
        ax.plot(t_arr, data_arr[idx], color=plot_colors[idx], label=plot_labels[idx])
        if idx == 5 and pdavg:
            ax.plot(
                t_arr,
                2 * tavg_arr,
                color=CB_color_cycle[0],
                linestyle="dashed",
                label="$2\\langle P_\mathrm{dyn}\\rangle$",
            )
        ax.set_xlim(t_arr[0], t_arr[-1])
        if draw_legend[idx]:
            ax.legend(loc="center left")
    ax_list[-1].set_xlabel("Simulation time [s]")
    for idx, ax in enumerate(ax_list):
        ax.grid()
        ax.set_ylabel(ylabels[idx])
        ax.axvline(t0, linestyle="dashed")
    # plt.tight_layout()
    figdir = wrkdir_DNR + "Figs/timeseries/"
    txtdir = wrkdir_DNR + "txts/timeseries/"
    if not os.path.exists(figdir):
        try:
            os.makedirs(figdir)
        except OSError:
            pass
    if not os.path.exists(txtdir):
        try:
            os.makedirs(txtdir)
        except OSError:
            pass

    fig.savefig(
        figdir + "{}_x{}_y{}_t{}_pm{}.png".format(runid, x0, y0, t0, tpm),
        dpi=300,
    )
    np.savetxt(
        txtdir + "{}_x{}_y{}_t{}_pm{}.txt".format(runid, x0, y0, t0, tpm),
        data_arr,
    )
    plt.close(fig)


def multi_VSC_timeseries(runid="AGF", time0=480, x=[8], y=[7], pm=60, delta=False):
    if len(x) != len(y):
        print("x and y must have same length!")
        return 1

    nvsc = len(x)
    coords = np.array([[x[k] * r_e, y[k] * r_e, 0] for k in range(nvsc)])
    t_arr = np.arange(time0 - pm / 2.0, time0 + pm / 2.0 + 0.1, 0.5)
    fnr_arr = np.arange(time0 * 2 - pm, time0 * 2 + pm + 0.1, 1, dtype=int)
    nt = len(t_arr)

    ts_v_vars = [
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
    ts_v_ops = [
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
    ts_v_labels = [
        "$n~[cm^{-3}]$",
        "$v_x~[km/s]$",
        "$v_y~[km/s]$",
        "$v_z~[km/s]$",
        "$v~[km/s]$",
        "$P_\mathrm{dyn}~[nPa]$",
        "$B_x~[nT]$",
        "$B_y~[nT]$",
        "$B_z~[nT]$",
        "$B~[nT]$",
        "$E_x~[mV/m]$",
        "$E_y~[mV/m]$",
        "$E_z~[mV/m]$",
        "$E~[mV/m]$",
        "$T_\parallel~[MK]$",
        "$T_\perp~[MK]$",
    ]
    if delta:
        for idx, lab in enumerate(ts_v_labels):
            lnew = lab.split("~")[0]
            # ts_v_labels[idx] = (
            #     "$\\delta$" + lnew + "/ | \\langle {} \\rangle | $".format(lnew[1:])
            # )
            ts_v_labels[idx] = "$\\delta$" + lab
    nrows = len(ts_v_labels)

    ts_arr = np.zeros((nvsc, nrows, nt), dtype=float)
    for idx, fnr in enumerate(fnr_arr):
        bulkpath = find_bulkpath(runid)
        vlsvobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(int(fnr)).zfill(7))
        )
        for idx2 in range(nvsc):
            coord = coords[idx2]
            for idx3 in range(nrows):
                ts_arr[idx2, idx3, idx] = (
                    vlsvobj.read_interpolated_variable(
                        ts_v_vars[idx3], coord, operator=ts_v_ops[idx3]
                    )
                    * scales[idx3]
                )

    fig, ax_list = plt.subplots(
        int(np.ceil((nrows + 4) / 3)), 3, figsize=(24, 24), constrained_layout=True
    )
    ax_list = (ax_list.T).flatten()

    ax_list[-1].set_xlabel("Time [s]")
    for idx in range(nrows):
        a = ax_list[idx]
        a.grid()
        a.set_xlim(t_arr[0], t_arr[-1])
        a.set_ylabel(ts_v_labels[idx])
        for idx2 in range(nvsc):
            if delta:
                a.plot(
                    t_arr,
                    # (ts_arr[idx2, idx, :] - np.mean(ts_arr[idx2, idx, :]))
                    # / np.abs(np.mean(ts_arr[idx2, idx, :])),
                    ts_arr[idx2, idx, :] - np.mean(ts_arr[idx2, idx, :]),
                    color=CB_color_cycle[idx2],
                    label="VSC {}".format(idx2),
                )
            else:
                a.plot(
                    t_arr,
                    ts_arr[idx2, idx, :],
                    color=CB_color_cycle[idx2],
                    label="VSC {}".format(idx2),
                )

    for idx2 in range(nvsc):
        ax_list[nrows].grid()
        ax_list[nrows].set_xlim(t_arr[0], t_arr[-1])
        ax_list[nrows].plot(
            t_arr,
            (ts_arr[idx2, 0, :] - np.mean(ts_arr[idx2, 0, :]))
            * (ts_arr[idx2, 4, :] - np.mean(ts_arr[idx2, 4, :])),
            color=CB_color_cycle[idx2],
        )
        ax_list[nrows].set_ylabel("$\\delta n \\times \\delta v$")

        ax_list[nrows + 1].grid()
        ax_list[nrows + 1].set_xlim(t_arr[0], t_arr[-1])
        ax_list[nrows + 1].plot(
            t_arr,
            (ts_arr[idx2, 0, :] - np.mean(ts_arr[idx2, 0, :]))
            * (ts_arr[idx2, 9, :] - np.mean(ts_arr[idx2, 9, :])),
            color=CB_color_cycle[idx2],
        )
        ax_list[nrows + 1].set_ylabel("$\\delta n \\times \\delta B$")

        ax_list[nrows + 2].grid()
        ax_list[nrows + 2].set_xlim(t_arr[0], t_arr[-1])
        ax_list[nrows + 2].plot(
            t_arr,
            (ts_arr[idx2, 9, :] - np.mean(ts_arr[idx2, 9, :]))
            * (ts_arr[idx2, 4, :] - np.mean(ts_arr[idx2, 4, :])),
            color=CB_color_cycle[idx2],
        )
        ax_list[nrows + 2].set_ylabel("$\\delta B \\times \\delta v$")

        ax_list[nrows + 3].grid()
        ax_list[nrows + 3].set_xlim(t_arr[0], t_arr[-1])
        ax_list[nrows + 3].plot(
            t_arr,
            (ts_arr[idx2, 9, :] - np.mean(ts_arr[idx2, 9, :]))
            * (ts_arr[idx2, 13, :] - np.mean(ts_arr[idx2, 13, :])),
            color=CB_color_cycle[idx2],
        )
        ax_list[nrows + 3].set_ylabel("$\\delta B \\times \\delta E$")

    ax_list[0].legend(loc="lower left")
    ax_list[0].set_title("VSC: {}".format(coords[:, :2] / r_e))

    figdir = wrkdir_DNR + "Figs/multi_vsc/"
    # txtdir = wrkdir_DNR + "txts/timeseries/"
    if not os.path.exists(figdir):
        try:
            os.makedirs(figdir)
        except OSError:
            pass
    # if not os.path.exists(txtdir):
    #     try:
    #         os.makedirs(txtdir)
    #     except OSError:
    #         pass

    # fig.savefig(
    #     figdir + "{}_x{}_y{}_t{}.png".format(runid, x0, y0, t0),
    #     dpi=300,
    # )
    # np.savetxt(
    #     txtdir + "{}_x{}_y{}_t{}.txt".format(runid, x0, y0, t0),
    #     data_arr,
    # )

    fig.savefig(
        figdir
        + "vsc_{}_x{}_y{}_t{}_pm{}_delta{}.png".format(
            nvsc, x[0], y[0], time0, pm, delta
        )
    )

    plt.close(fig)


def moving_avg(A, w):
    ncols = A.shape[0]

    B = np.zeros_like(A)

    for idx in range(ncols):
        B[idx, :] = np.convolve(A[idx, :], np.ones(w), mode="same") / w

    return B


def calc_velocities(dx, dy, vx, vy, Bx, By, va, vs, vms):
    # Bx = moving_avg(Bx, 5)
    # By = moving_avg(By, 5)

    Bmag = np.sqrt(Bx**2 + By**2)
    vmag = np.sqrt(vx**2 + vy**2)

    vax = va * Bx / Bmag
    vay = va * By / Bmag

    vsx = vs * Bx / Bmag
    vsy = vs * By / Bmag

    vmsx = vms * By / Bmag
    vmsy = vms * Bx / Bmag

    vpvax = vx + vax
    vpvay = vy + vay

    vpvsx = vx + vsx
    vpvsy = vy + vsy

    vpvmsx = vx + vmsx
    vpvmsy = vy + vmsy

    vmvax = vx - vax
    vmvay = vy - vay

    vmvsx = vx - vsx
    vmvsy = vy - vsy

    vmvmsx = vx - vmsx
    vmvmsy = vy - vmsy

    vpvapar = (dx * vpvax + dy * vpvay) / (np.sqrt(dx**2 + dy**2))
    vmvapar = (dx * vmvax + dy * vmvay) / (np.sqrt(dx**2 + dy**2))

    vpvspar = (dx * vpvsx + dy * vpvsy) / (np.sqrt(dx**2 + dy**2))
    vmvspar = (dx * vmvsx + dy * vmvsy) / (np.sqrt(dx**2 + dy**2))

    vpvmspar = (dx * vpvmsx + dy * vpvmsy) / (np.sqrt(dx**2 + dy**2))
    vmvmspar = (dx * vmvmsx + dy * vmvmsy) / (np.sqrt(dx**2 + dy**2))

    vpar = (dx * vx + dy * vy) / (np.sqrt(dx**2 + dy**2))

    return (vpar, vpvapar, vmvapar, vpvspar, vmvspar, vpvmspar, vmvmspar)


def jplots(
    x0,
    y0,
    x1,
    y1,
    t0,
    t1,
    runid="AGF",
    txt=False,
    draw=True,
    bs_thresh=0.3,
    intpol=False,
    vel_lines=None,
    wavefan=None,
):
    dr = 300e3 / r_e
    dr_km = 300
    varname_list = [
        "$n$ [cm$^{-3}$]",
        "$v_x$ [km/s]",
        "$P_\mathrm{dyn}$ [nPa]",
        "$B$ [nT]",
        "$T$ [MK]",
    ]
    vars_list = [
        "proton/vg_rho",
        "proton/vg_v",
        "proton/vg_pdyn",
        "vg_b_vol",
        "proton/vg_temperature",
        # "proton/vg_core_heating",
        # "proton/vg_mmsx",
        "proton/vg_beta_star",
        "vg_va",
        "vg_vs",
        "vg_vms",
        "proton/vg_v",
        "proton/vg_v",
        "vg_b_vol",
        "vg_b_vol",
    ]
    ops_list = [
        "pass",
        "x",
        "pass",
        "magnitude",
        "pass",
        # "pass",
        # "pass",
        "pass",
        "pass",
        "pass",
        "pass",
        "x",
        "y",
        "x",
        "y",
    ]
    scale_list = [
        1e-6,
        1e-3,
        1e9,
        1e9,
        1e-6,
        # 1,
        # 1,
        1,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        1e9,
        1e9,
    ]

    # Solar wind parameters for the different runs
    # n [m^-3], v [m/s], B [T], T [K]
    runid_list = ["AGF", "AIA"]
    runids_paper = ["RDC", "RDC2"]
    sw_pars = [
        [1.0e6, 750.0e3, 3.0e-9, 0.5e6],
        [1.0e6, 750.0e3, 3.0e-9, 0.5e6],
    ]
    n_sw, v_sw, B_sw, T_sw = sw_pars[runid_list.index(runid)]
    pdyn_sw = m_p * n_sw * v_sw * v_sw

    # vmin_norm = [1.0 / 2, -2.0, 1.0 / 6, 1.0 / 2, 1.0]
    # vmax_norm = [6.0, 2.0, 2.0, 6.0, 36.0]
    vmin = [0, -500, 0, 0, 0]
    vmax = [5, 0, 0.8, 40, 25]

    # Path to vlsv files for current run
    bulkpath = find_bulkpath(runid)

    fnr0 = int(t0 * 2)
    fnr1 = int(t1 * 2)

    fnr_range = np.arange(fnr0, fnr1 + 1, 1, dtype=int)
    t_range = np.arange(t0, t1 + 0.1, 0.5)
    # Get cellid of initial position

    npoints = int(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / dr) + 1

    xlist = np.linspace(x0, x1, npoints)
    ylist = np.linspace(y0, y1, npoints)

    if intpol:
        coords = [[xlist[idx] * r_e, ylist[idx] * r_e, 0] for idx in range(xlist.size)]

    fobj = pt.vlsvfile.VlsvReader(bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7)))

    cellids = [
        int(fobj.get_cellid([xlist[idx] * r_e, ylist[idx] * r_e, 0]))
        for idx in range(xlist.size)
    ]
    cellnr = range(xlist.size)
    if xlist[-1] != xlist[0]:
        xplot_list = xlist
        xlab = "$X~[R_\mathrm{E}]$"
    else:
        xplot_list = ylist
        xlab = "$Y~[R_\mathrm{E}]$"

    XmeshXY, YmeshXY = np.meshgrid(xlist, t_range)

    data_arr = np.zeros((len(vars_list), xplot_list.size, t_range.size), dtype=float)
    vt_arr = np.ones((xplot_list.size, t_range.size), dtype=float)
    dx_arr = (x1 - x0) * vt_arr
    dy_arr = (y1 - y0) * vt_arr

    figdir = wrkdir_DNR + "Figs/jmaps/"
    txtdir = wrkdir_DNR + "txts/jmaps/"

    if txt:
        data_arr = np.load(
            txtdir
            + "{}_x0_{}_y0_{}_x1_{}_y1_{}_t0_{}_t1_{}.npy".format(
                runid, x0, y0, x1, y1, t0, t1
            )
        )
    else:
        for idx in range(fnr_range.size):
            fnr = fnr_range[idx]
            vlsvobj = pt.vlsvfile.VlsvReader(
                bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
            )
            for idx2 in range(len(vars_list)):
                if intpol:
                    data_arr[idx2, :, idx] = [
                        vlsvobj.read_interpolated_variable(
                            vars_list[idx2], coords[idx3], operator=ops_list[idx2]
                        )
                        * scale_list[idx2]
                        for idx3 in range(xlist.size)
                    ]
                else:
                    data_arr[idx2, :, idx] = (
                        vlsvobj.read_variable(
                            vars_list[idx2], operator=ops_list[idx2], cellids=cellids
                        )
                        * scale_list[idx2]
                    )

    # vpar,vpvapar,vmvapar,vpvspar,vmvspar,vpvmspar,vmvmspar
    outvels = calc_velocities(
        dx_arr,
        dy_arr,
        data_arr[9],
        data_arr[10],
        data_arr[11],
        data_arr[12],
        data_arr[6],
        data_arr[7],
        data_arr[8],
    )
    vels_list = ["vb", "vb+va", "vb-va", "vb+vs", "vb-vs", "vb+vms", "vb-vms"]
    vels_labels = [
        "$v_b$",
        "$v_b+v_a$",
        "$v_b-v_a$",
        "$v_b+v_s$",
        "$v_b-v_s$",
        "$v_b+v_{ms}$",
        "$v_b-v_{ms}$",
    ]
    if vel_lines:
        vel_to_plot = outvels[vels_list.index(vel_lines)] / dr_km
        nstp = 10
        start_points = np.array(
            [
                np.ones(nstp) * int(xplot_list.size / 2),
                np.linspace(t_range[1], t_range[-2], nstp),
            ]
        ).T

    # data_arr = [rho_arr, v_arr, pdyn_arr, B_arr, T_arr]
    cmap = ["Blues_r", "Blues_r", "Blues_r", "Blues_r", "Blues_r"]
    annot = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    # fig, ax_list = plt.subplots(
    #     1, len(varname_list), figsize=(20, 5), sharex=True, sharey=True
    # )
    if draw:
        fig, ax_list = plt.subplots(
            1,
            len(varname_list),
            figsize=(18, 10),
            sharex=True,
            sharey=True,
        )
        ax_list = ax_list.flatten()
        im_list = []
        cb_list = []
        fig.suptitle(
            "Run: {}, x0: {}, y0: {}, x1: {}, y1: {}".format(runid, x0, y0, x1, y1),
            fontsize=28,
        )
        for idx in range(len(varname_list)):
            ax = ax_list[idx]
            ax.tick_params(labelsize=20)
            im_list.append(
                ax.pcolormesh(
                    XmeshXY,
                    YmeshXY,
                    data_arr[idx].T,
                    shading="nearest",
                    cmap=cmap[idx],
                    vmin=vmin[idx],
                    vmax=vmax[idx],
                    rasterized=True,
                )
            )
            # if idx == 1:
            #     cb_list.append(fig.colorbar(im_list[idx], ax=ax, extend="max"))
            #     cb_list[idx].cmap.set_over("red")
            # else:
            cb_list.append(fig.colorbar(im_list[idx], ax=ax))
            # cb_list.append(fig.colorbar(im_list[idx], ax=ax))
            cb_list[idx].ax.tick_params(labelsize=20)
            ax.contour(XmeshXY, YmeshXY, data_arr[5].T, [bs_thresh], colors=["k"])
            ax.plot([1, 2], [0, 1], color="k", label="$\\beta^*=$ {}".format(bs_thresh))
            if vel_lines:
                ax.streamplot(
                    XmeshXY,
                    YmeshXY,
                    vel_to_plot.T,
                    vt_arr.T,
                    arrowstyle="-",
                    broken_streamlines=False,
                    color=CB_color_cycle[1],
                    linewidth=0.4,
                    # minlength=4,
                    density=35,
                    start_points=start_points,
                )
            if wavefan:
                for itr, vel in enumerate(outvels):
                    ax.streamplot(
                        XmeshXY,
                        YmeshXY,
                        vel.T / 6371,
                        vt_arr.T,
                        arrowstyle="-",
                        broken_streamlines=True,
                        color=CB_color_cycle[itr],
                        linewidth=0.6,
                        # minlength=4,
                        maxlength=1,
                        integration_direction="forward",
                        density=35,
                        start_points=np.array([wavefan]),
                    )
                    ax.plot(
                        [1, 2],
                        [0, 1],
                        color=CB_color_cycle[itr],
                        label=vels_labels[itr],
                    )
            # ax.contour(XmeshXY, YmeshXY, Tcore_arr, [3], colors=[CB_color_cycle[1]])
            # ax.contour(XmeshXY, YmeshXY, mmsx_arr, [1.0], colors=[CB_color_cycle[4]])
            ax.set_title(varname_list[idx], fontsize=24, pad=10)
            ax.set_xlim(xplot_list[0], xplot_list[-1])
            ax.set_ylim(t_range[0], t_range[-1])
            ax.set_xlabel(xlab, fontsize=24, labelpad=10)
            # ax.axhline(t0, linestyle="dashed", linewidth=0.6)
            # ax.axvline(x0, linestyle="dashed", linewidth=0.6)
            ax.annotate(annot[idx], (0.05, 0.90), xycoords="axes fraction", fontsize=24)
        ax_list[0].set_ylabel("Simulation time [s]", fontsize=28, labelpad=10)
        ax_list[1].legend(
            fontsize=12, bbox_to_anchor=(0.5, -0.12), loc="upper center", ncols=2
        )
        # ax_list[int(np.ceil(len(varname_list) / 2.0))].set_ylabel(
        #     "Simulation time [s]", fontsize=28, labelpad=10
        # )
        # ax_list[-1].set_axis_off()

        # Save figure
        plt.tight_layout()

        # fig.savefig(
        #     wrkdir_DNR
        #     + "papu22/Figures/jmaps/{}_{}.pdf".format(runid, str(non_id).zfill(5))
        # )
        if not os.path.exists(figdir):
            try:
                os.makedirs(figdir)
            except OSError:
                pass

        fig.savefig(
            figdir
            + "{}_x0_{}_y0_{}_x1_{}_y1_{}_t0_{}_t1_{}.png".format(
                runid, x0, y0, x1, y1, t0, t1
            ),
            dpi=300,
        )
        plt.close(fig)

    if not txt:
        if not os.path.exists(txtdir):
            try:
                os.makedirs(txtdir)
            except OSError:
                pass

        np.save(
            txtdir
            + "{}_x0_{}_y0_{}_x1_{}_y1_{}_t0_{}_t1_{}.npy".format(
                runid, x0, y0, x1, y1, t0, t1
            ),
            data_arr,
        )


def getNearestCellWithVspace(vlsvReader, cid):
    cell_candidates = vlsvReader.read(mesh="SpatialGrid", tag="CELLSWITHBLOCKS")
    if len(cell_candidates) == 0:
        print("Error: No velocity distributions found!")
        sys.exit()
    cell_candidate_coordinates = [
        vlsvReader.get_cell_coordinates(cell_candidate)
        for cell_candidate in cell_candidates
    ]
    cell_coordinates = vlsvReader.get_cell_coordinates(cid)
    norms = np.sum((cell_candidate_coordinates - cell_coordinates) ** 2, axis=-1) ** (
        1.0 / 2
    )
    norm, i = min((norm, idx) for (idx, norm) in enumerate(norms))
    return cell_candidates[i]


def pos_vdf_profile_plotter(runid, x, y, t0, t1):
    runids = ["AGF", "AIA"]
    pdmax = [1.0, 1.0][runids.index(runid)]
    bulkpath = find_bulkpath(runid)

    global xg, yg

    xg = []
    yg = []

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g, x0, y0
    runid_g = runid
    Blines_g = False

    non_ids = []
    sj_ids = []

    sj_ids_g = sj_ids
    non_ids_g = non_ids

    # pdmax = [1.5, 3.5, 1.5, 3.5][runids.index(runid)]
    # sw_pars = [
    #     [1e6, 750e3, 5e-9, 0.5e6],
    #     [3.3e6, 600e3, 5e-9, 0.5e6],
    #     [1e6, 750e3, 10e-9, 0.5e6],
    #     [3.3e6, 600e3, 10e-9, 0.5e6],
    # ]
    sw_pars = [
        [1e6, 750e3, 3e-9, 0.5e6],
        [1e6, 750e3, 3e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runids.index(runid)]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw

    for t in np.arange(t0, t1 + 0.1, 0.5):
        fnr = int(t * 2)
        filenr_g = fnr
        vobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
        )
        cellid = vobj.get_cellid([x * r_e, y * r_e, 0 * r_e])
        vdf_cellid = getNearestCellWithVspace(vobj, cellid)

        x_re, y_re, z_re = vobj.get_cell_coordinates(vdf_cellid) / r_e
        xhist, xbin_edges = vspace_reducer(vobj, vdf_cellid, operator="x")
        yhist, ybin_edges = vspace_reducer(vobj, vdf_cellid, operator="y")
        zhist, zbin_edges = vspace_reducer(vobj, vdf_cellid, operator="z")
        xbin_centers = xbin_edges[:-1] + 0.5 * (xbin_edges[1] - xbin_edges[0])
        ybin_centers = ybin_edges[:-1] + 0.5 * (ybin_edges[1] - ybin_edges[0])
        zbin_centers = zbin_edges[:-1] + 0.5 * (zbin_edges[1] - zbin_edges[0])

        x0 = x_re
        y0 = y_re

        fig, ax_list = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

        pt.plot.plot_colormap(
            axes=ax_list[0],
            vlsvobj=vobj,
            var="proton/vg_Pdyn",
            vmin=0,
            vmax=pdmax,
            vscale=1e9,
            cbtitle="$P_\mathrm{dyn}$ [nPa]",
            usesci=0,
            boxre=[x_re - 2, x_re + 2, y_re - 2, y_re + 2],
            # internalcb=True,
            lin=1,
            colormap="batlow",
            scale=1.3,
            tickinterval=1.0,
            external=ext_jet,
            pass_vars=[
                "proton/vg_rho_thermal",
                "proton/vg_rho_nonthermal",
                "proton/vg_ptensor_thermal_diagonal",
                "vg_b_vol",
                "proton/vg_v",
                "proton/vg_rho",
                "proton/vg_core_heating",
                "CellID",
                "proton/vg_mmsx",
                "proton/vg_Pdyn",
                "proton/vg_Pdynx",
                "proton/vg_beta_star",
            ],
        )
        ax_list[0].axhline(y_re, linestyle="dashed", linewidth=0.6, color="k")
        ax_list[0].axvline(x_re, linestyle="dashed", linewidth=0.6, color="k")

        # pt.plot.plot_vdf_profiles(
        #     axes=ax_list[1],
        #     filename=bulkpath + fname,
        #     cellids=[vdf_cellid],
        #     # colormap="batlow",
        #     # bvector=1,
        #     xy=1,
        #     # slicethick=0,
        #     # box=[-2e6, 2e6, -2e6, 2e6],
        #     # internalcb=True,
        #     setThreshold=1e-15,
        #     lin=None,
        #     fmin=1e-15,
        #     fmax=4e-10,
        #     vmin=-2000,
        #     vmax=2000,
        #     # scale=1.3,
        # )

        ax_list[1].step(xbin_centers * 1e-3, xhist, "k", label="vx")
        ax_list[1].step(ybin_centers * 1e-3, yhist, "r", label="vy")
        ax_list[1].step(zbin_centers * 1e-3, zhist, "b", label="vz")
        ax_list[1].legend(loc="upper right")
        ax_list[1].set_xlim(-2000, 2000)
        ax_list[1].set_xlabel("$v~[\mathrm{kms}^{-1}]$")
        ax_list[1].set_ylabel("$f(v)~[\mathrm{sm}^{-4}]$")

        # plt.subplots_adjust(wspace=1, hspace=1)

        outdir = wrkdir_DNR + "VDFs/{}/x_{:.3f}_y_{:.3f}_t0_{}_t1_{}_profile".format(
            runid, x_re, y_re, t0, t1
        )

        fig.suptitle(
            "Run: {}, x: {:.3f}, y: {:.3f}, Time: {}s".format(runid, x_re, y_re, t)
        )
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except OSError:
                pass
        fig.savefig(outdir + "/{}.png".format(fnr))
        plt.close(fig)

    return None


def pos_vdf_plotter(runid, x, y, t0, t1):
    runids = ["AGF", "AIA"]
    pdmax = [1.0, 1.0][runids.index(runid)]
    bulkpath = find_bulkpath(runid)

    global xg, yg

    xg = []
    yg = []

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g, x0, y0
    runid_g = runid
    Blines_g = False

    non_ids = []
    sj_ids = []

    sj_ids_g = sj_ids
    non_ids_g = non_ids

    # pdmax = [1.5, 3.5, 1.5, 3.5][runids.index(runid)]
    # sw_pars = [
    #     [1e6, 750e3, 5e-9, 0.5e6],
    #     [3.3e6, 600e3, 5e-9, 0.5e6],
    #     [1e6, 750e3, 10e-9, 0.5e6],
    #     [3.3e6, 600e3, 10e-9, 0.5e6],
    # ]
    sw_pars = [
        [1e6, 750e3, 3e-9, 0.5e6],
        [1e6, 750e3, 3e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runids.index(runid)]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw

    for t in np.arange(t0, t1 + 0.1, 0.5):
        fnr = int(t * 2)
        filenr_g = fnr
        vobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
        )
        cellid = vobj.get_cellid([x * r_e, y * r_e, 0 * r_e])
        vdf_cellid = getNearestCellWithVspace(vobj, cellid)

        x_re, y_re, z_re = vobj.get_cell_coordinates(vdf_cellid) / r_e

        x0 = x_re
        y0 = y_re

        fig, ax_list = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)

        pt.plot.plot_colormap(
            axes=ax_list[0][0],
            vlsvobj=vobj,
            var="proton/vg_Pdyn",
            vmin=0,
            vmax=pdmax,
            vscale=1e9,
            cbtitle="$P_\mathrm{dyn}$ [nPa]",
            usesci=0,
            boxre=[x_re - 2, x_re + 2, y_re - 2, y_re + 2],
            # internalcb=True,
            lin=1,
            colormap="batlow",
            scale=1.3,
            tickinterval=1.0,
            external=ext_jet,
            pass_vars=[
                "proton/vg_rho_thermal",
                "proton/vg_rho_nonthermal",
                "proton/vg_ptensor_thermal_diagonal",
                "vg_b_vol",
                "proton/vg_v",
                "proton/vg_rho",
                "proton/vg_core_heating",
                "CellID",
                "proton/vg_mmsx",
                "proton/vg_Pdyn",
                "proton/vg_Pdynx",
                "proton/vg_beta_star",
            ],
        )
        ax_list[0][0].axhline(y_re, linestyle="dashed", linewidth=0.6, color="k")
        ax_list[0][0].axvline(x_re, linestyle="dashed", linewidth=0.6, color="k")

        pt.plot.plot_vdf(
            axes=ax_list[0][1],
            vlsvobj=vobj,
            cellids=[vdf_cellid],
            colormap="batlow",
            bvector=1,
            xy=1,
            slicethick=0,
            box=[-2e6, 2e6, -2e6, 2e6],
            # internalcb=True,
            setThreshold=1e-15,
            scale=1.3,
            fmin=1e-10,
            fmax=1e-4,
        )
        pt.plot.plot_vdf(
            axes=ax_list[1][0],
            vlsvobj=vobj,
            cellids=[vdf_cellid],
            colormap="batlow",
            bvector=1,
            xz=1,
            slicethick=0,
            box=[-2e6, 2e6, -2e6, 2e6],
            # internalcb=True,
            setThreshold=1e-15,
            scale=1.3,
            fmin=1e-10,
            fmax=1e-4,
        )
        pt.plot.plot_vdf(
            axes=ax_list[1][1],
            vlsvobj=vobj,
            cellids=[vdf_cellid],
            colormap="batlow",
            bvector=1,
            yz=1,
            slicethick=0,
            box=[-2e6, 2e6, -2e6, 2e6],
            # internalcb=True,
            setThreshold=1e-15,
            scale=1.3,
            fmin=1e-10,
            fmax=1e-4,
        )

        # plt.subplots_adjust(wspace=1, hspace=1)

        outdir = wrkdir_DNR + "VDFs/{}/x_{:.3f}_y_{:.3f}_t0_{}_t1_{}".format(
            runid, x_re, y_re, t0, t1
        )

        fig.suptitle(
            "Run: {}, x: {:.3f}, y: {:.3f}, Time: {}s".format(runid, x_re, y_re, t)
        )
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except OSError:
                pass
        fig.savefig(outdir + "/{}.png".format(fnr))
        plt.close(fig)

    return None


def jet_vdf_plotter(runid, skip=[]):
    runids = ["AGF", "AIA"]
    pdmax = [1.0, 1.0][runids.index(runid)]
    bulkpath = find_bulkpath(runid)
    obj_580 = pt.vlsvfile.VlsvReader(bulkpath + "bulk.0000781.vlsv")
    cellids = obj_580.read_variable("CellID")
    if obj_580.check_variable("fSaved"):
        fsaved = obj_580.read_variable("fSaved")
    else:
        fsaved = obj_580.read_variable("vg_f_saved")

    vdf_cells = cellids[fsaved == 1]

    global xg, yg

    xg = []
    yg = []

    # asw_list, fw_list = auto_classifier(runid)
    # jet_ids = asw_list + fw_list

    jet_ids = get_jets(runid)

    # jet_ids = np.append(np.array(jet_ids, dtype=int), get_fcs_jets(runid))

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g, x0, y0
    runid_g = runid
    Blines_g = False

    non_ids = []
    sj_ids = []

    sj_ids_g = sj_ids
    non_ids_g = non_ids

    # pdmax = [1.5, 3.5, 1.5, 3.5][runids.index(runid)]
    # sw_pars = [
    #     [1e6, 750e3, 5e-9, 0.5e6],
    #     [3.3e6, 600e3, 5e-9, 0.5e6],
    #     [1e6, 750e3, 10e-9, 0.5e6],
    #     [3.3e6, 600e3, 10e-9, 0.5e6],
    # ]
    sw_pars = [
        [1e6, 750e3, 3e-9, 0.5e6],
        [1e6, 750e3, 3e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runids.index(runid)]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw

    for jet_id in jet_ids:
        if jet_id in skip:
            continue
        props = PropReader(str(jet_id).zfill(5), runid)
        jet_times = props.get_times()
        jet_cells = props.get_cells()

        for idx, t in enumerate(jet_times):
            vobj = pt.vlsvfile.VlsvReader(
                bulkpath + "bulk.{}.vlsv".format(str(int(t * 2)).zfill(7))
            )
            cellids = vobj.read_variable("CellID")
            fsaved = vobj.read_variable("vg_f_saved")
            vdf_cells = cellids[fsaved == 1]
            if np.intersect1d(jet_cells[idx], vdf_cells).size == 0:
                continue
            else:
                vdf_cellid = np.intersect1d(jet_cells[idx], vdf_cells)[0]

            for tc in np.arange(t - 10, t + 10.01, 0.5):
                fnr = int(tc * 2)
                filenr_g = fnr
                fname = "bulk.{}.vlsv".format(str(fnr).zfill(7))

                x_re, y_re, z_re = vobj.get_cell_coordinates(vdf_cellid) / r_e

                x0 = x_re
                y0 = y_re

                fig, ax_list = plt.subplots(
                    2, 2, figsize=(11, 10), constrained_layout=True
                )

                pt.plot.plot_colormap(
                    axes=ax_list[0][0],
                    filename=bulkpath + fname,
                    var="proton/vg_Pdyn",
                    vmin=0,
                    vmax=pdmax,
                    vscale=1e9,
                    cbtitle="$P_\mathrm{dyn}$ [nPa]",
                    usesci=0,
                    boxre=[x_re - 2, x_re + 2, y_re - 2, y_re + 2],
                    # internalcb=True,
                    lin=1,
                    colormap="batlow",
                    scale=1.3,
                    tickinterval=1.0,
                    external=ext_jet,
                    pass_vars=[
                        "proton/vg_rho_thermal",
                        "proton/vg_rho_nonthermal",
                        "proton/vg_ptensor_thermal_diagonal",
                        "vg_b_vol",
                        "proton/vg_v",
                        "proton/vg_rho",
                        "proton/vg_core_heating",
                        "CellID",
                        "proton/vg_mmsx",
                        "proton/vg_Pdyn",
                        "proton/vg_Pdynx",
                        "proton/vg_beta_star",
                    ],
                )
                ax_list[0][0].axhline(
                    y_re, linestyle="dashed", linewidth=0.6, color="k"
                )
                ax_list[0][0].axvline(
                    x_re, linestyle="dashed", linewidth=0.6, color="k"
                )

                pt.plot.plot_vdf(
                    axes=ax_list[0][1],
                    filename=bulkpath + fname,
                    cellids=[vdf_cellid],
                    colormap="batlow",
                    bvector=1,
                    xy=1,
                    slicethick=0,
                    box=[-2e6, 2e6, -2e6, 2e6],
                    # internalcb=True,
                    setThreshold=1e-15,
                    scale=1.3,
                    fmin=1e-10,
                    fmax=1e-4,
                )
                pt.plot.plot_vdf(
                    axes=ax_list[1][0],
                    filename=bulkpath + fname,
                    cellids=[vdf_cellid],
                    colormap="batlow",
                    bvector=1,
                    xz=1,
                    slicethick=0,
                    box=[-2e6, 2e6, -2e6, 2e6],
                    # internalcb=True,
                    setThreshold=1e-15,
                    scale=1.3,
                    fmin=1e-10,
                    fmax=1e-4,
                )
                pt.plot.plot_vdf(
                    axes=ax_list[1][1],
                    filename=bulkpath + fname,
                    cellids=[vdf_cellid],
                    colormap="batlow",
                    bvector=1,
                    yz=1,
                    slicethick=0,
                    box=[-2e6, 2e6, -2e6, 2e6],
                    # internalcb=True,
                    setThreshold=1e-15,
                    scale=1.3,
                    fmin=1e-10,
                    fmax=1e-4,
                )

                # plt.subplots_adjust(wspace=1, hspace=1)

                fig.suptitle("Run: {}, Jet: {}, Time: {}s".format(runid, jet_id, tc))
                if not os.path.exists(
                    wrkdir_DNR + "VDFs/{}/jet_vdf_{}".format(runid, jet_id)
                ):
                    try:
                        os.makedirs(
                            wrkdir_DNR + "VDFs/{}/jet_vdf_{}".format(runid, jet_id)
                        )
                    except OSError:
                        pass
                fig.savefig(
                    wrkdir_DNR + "VDFs/{}/jet_vdf_{}/{}.png".format(runid, jet_id, fnr)
                )
                plt.close(fig)
            break

    return None


def vspace_reducer(vlsvobj, cellid, operator):
    op_list = ["x", "y", "z"]

    velcels = vlsvobj.read_velocity_cells(cellid)
    vc_coords = vlsvobj.get_velocity_cell_coordinates(list(velcels.keys()))
    vc_vals = np.array(list(velcels.values()))

    vc_coord_arr = vc_coords[:, op_list.index(operator)]

    dv = 30e3

    vbins = np.sort(np.unique(vc_coord_arr))
    vbins = np.append(vbins - dv / 2, vbins[-1] + dv / 2)

    hist, bin_edges = np.histogram(vc_coord_arr, bins=vbins, weights=vc_vals * dv * dv)

    return (hist, bin_edges)


def jet_vdf_profile_plotter(runid, skip=[]):
    runids = ["AGF", "AIA"]
    pdmax = [1.0, 1.0][runids.index(runid)]
    bulkpath = find_bulkpath(runid)
    obj_580 = pt.vlsvfile.VlsvReader(bulkpath + "bulk.0000781.vlsv")
    cellids = obj_580.read_variable("CellID")
    if obj_580.check_variable("fSaved"):
        fsaved = obj_580.read_variable("fSaved")
    else:
        fsaved = obj_580.read_variable("vg_f_saved")

    global xg, yg

    xg = []
    yg = []

    vdf_cells = cellids[fsaved == 1]

    # asw_list, fw_list = auto_classifier(runid)
    # jet_ids = asw_list + fw_list

    jet_ids = get_jets(runid)
    # jet_ids = np.append(np.array(jet_ids, dtype=int), get_fcs_jets(runid))

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g, x0, y0
    runid_g = runid
    Blines_g = False

    non_ids = []
    sj_ids = []

    sj_ids_g = sj_ids
    non_ids_g = non_ids

    # pdmax = [1.5, 3.5, 1.5, 3.5][runids.index(runid)]
    # sw_pars = [
    #     [1e6, 750e3, 5e-9, 0.5e6],
    #     [3.3e6, 600e3, 5e-9, 0.5e6],
    #     [1e6, 750e3, 10e-9, 0.5e6],
    #     [3.3e6, 600e3, 10e-9, 0.5e6],
    # ]
    sw_pars = [
        [1e6, 750e3, 3e-9, 0.5e6],
        [1e6, 750e3, 3e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runids.index(runid)]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw

    for jet_id in jet_ids:
        if jet_id in skip:
            continue
        props = PropReader(str(jet_id).zfill(5), runid)
        jet_times = props.get_times()
        jet_cells = props.get_cells()

        for idx, t in enumerate(jet_times):
            vobj_t = pt.vlsvfile.VlsvReader(
                bulkpath + "bulk.{}.vlsv".format(str(int(t * 2)).zfill(7))
            )
            cellids = vobj_t.read_variable("CellID")
            fsaved = vobj_t.read_variable("vg_f_saved")
            vdf_cells = cellids[fsaved == 1]
            if np.intersect1d(jet_cells[idx], vdf_cells).size == 0:
                continue
            else:
                vdf_cellid = np.intersect1d(jet_cells[idx], vdf_cells)[0]

            for tc in np.arange(t - 10, t + 10.01, 0.5):
                fnr = int(tc * 2)
                filenr_g = fnr
                fname = "bulk.{}.vlsv".format(str(fnr).zfill(7))
                vobj = pt.vlsvfile.VlsvReader(bulkpath + fname)
                x_re, y_re, z_re = vobj.get_cell_coordinates(vdf_cellid) / r_e
                xhist, xbin_edges = vspace_reducer(vobj, vdf_cellid, operator="x")
                yhist, ybin_edges = vspace_reducer(vobj, vdf_cellid, operator="y")
                zhist, zbin_edges = vspace_reducer(vobj, vdf_cellid, operator="z")
                xbin_centers = xbin_edges[:-1] + 0.5 * (xbin_edges[1] - xbin_edges[0])
                ybin_centers = ybin_edges[:-1] + 0.5 * (ybin_edges[1] - ybin_edges[0])
                zbin_centers = zbin_edges[:-1] + 0.5 * (zbin_edges[1] - zbin_edges[0])

                x0 = x_re
                y0 = y_re

                fig, ax_list = plt.subplots(
                    1, 2, figsize=(11, 5), constrained_layout=True
                )

                pt.plot.plot_colormap(
                    axes=ax_list[0],
                    filename=bulkpath + fname,
                    var="proton/vg_Pdyn",
                    vmin=0,
                    vmax=pdmax,
                    vscale=1e9,
                    cbtitle="$P_\mathrm{dyn}$ [nPa]",
                    usesci=0,
                    boxre=[x_re - 2, x_re + 2, y_re - 2, y_re + 2],
                    # internalcb=True,
                    lin=1,
                    colormap="batlow",
                    scale=1.3,
                    tickinterval=1.0,
                    external=ext_jet,
                    pass_vars=[
                        "proton/vg_rho_thermal",
                        "proton/vg_rho_nonthermal",
                        "proton/vg_ptensor_thermal_diagonal",
                        "vg_b_vol",
                        "proton/vg_v",
                        "proton/vg_rho",
                        "proton/vg_core_heating",
                        "CellID",
                        "proton/vg_mmsx",
                        "proton/vg_Pdyn",
                        "proton/vg_Pdynx",
                        "proton/vg_beta_star",
                    ],
                )
                ax_list[0].axhline(y_re, linestyle="dashed", linewidth=0.6, color="k")
                ax_list[0].axvline(x_re, linestyle="dashed", linewidth=0.6, color="k")

                # pt.plot.plot_vdf_profiles(
                #     axes=ax_list[1],
                #     filename=bulkpath + fname,
                #     cellids=[vdf_cellid],
                #     # colormap="batlow",
                #     # bvector=1,
                #     xy=1,
                #     # slicethick=0,
                #     # box=[-2e6, 2e6, -2e6, 2e6],
                #     # internalcb=True,
                #     setThreshold=1e-15,
                #     lin=None,
                #     fmin=1e-15,
                #     fmax=4e-10,
                #     vmin=-2000,
                #     vmax=2000,
                #     # scale=1.3,
                # )

                ax_list[1].step(xbin_centers * 1e-3, xhist, "k", label="vx")
                ax_list[1].step(ybin_centers * 1e-3, yhist, "r", label="vy")
                ax_list[1].step(zbin_centers * 1e-3, zhist, "b", label="vz")
                ax_list[1].legend(loc="upper right")
                ax_list[1].set_xlim(-2000, 2000)
                ax_list[1].set_xlabel("$v~[\mathrm{kms}^{-1}]$")
                ax_list[1].set_ylabel("$f(v)~[\mathrm{sm}^{-4}]$")
                # ax_list[1].set_ylim(0, 30)

                # pt.plot.plot_vdf_profiles(
                #     axes=ax_list[1][0],
                #     filename=bulkpath + fname,
                #     cellids=[vdf_cellid],
                #     # colormap="batlow",
                #     # bvector=1,
                #     xz=1,
                #     # slicethick=0,
                #     # box=[-2e6, 2e6, -2e6, 2e6],
                #     # internalcb=True,
                #     setThreshold=1e-15,
                #     lin=5,
                #     fmin=0,
                #     fmax=5e-10,
                #     vmin=-2000,
                #     vmax=2000,
                #     # scale=1.3,
                # )
                # pt.plot.plot_vdf_profiles(
                #     axes=ax_list[1][1],
                #     filename=bulkpath + fname,
                #     cellids=[vdf_cellid],
                #     # colormap="batlow",
                #     # bvector=1,
                #     yz=1,
                #     # slicethick=0,
                #     # box=[-2e6, 2e6, -2e6, 2e6],
                #     # internalcb=True,
                #     setThreshold=1e-15,
                #     lin=5,
                #     fmin=0,
                #     fmax=5e-10,
                #     vmin=-2000,
                #     vmax=2000,
                #     # scale=1.3,
                # )

                # plt.subplots_adjust(wspace=1, hspace=1)

                fig.suptitle("Run: {}, Jet: {}, Time: {}s".format(runid, jet_id, tc))
                if not os.path.exists(
                    wrkdir_DNR + "VDFs/{}/jet_vdf_profile_{}".format(runid, jet_id)
                ):
                    try:
                        os.makedirs(
                            wrkdir_DNR
                            + "VDFs/{}/jet_vdf_profile_{}".format(runid, jet_id)
                        )
                    except OSError:
                        pass
                fig.savefig(
                    wrkdir_DNR
                    + "VDFs/{}/jet_vdf_profile_{}/{}.png".format(runid, jet_id, fnr)
                )
                plt.close(fig)
            break

    return None
