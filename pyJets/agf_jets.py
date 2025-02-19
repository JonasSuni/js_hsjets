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

default_globals = set(globals())


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
    runid_list = ["ABA", "ABC", "AEA", "AEC", "AGF", "AIA", "AIC"]
    maxfnr_list = [839, 1179, 1339, 879, 1193, 1193, 1960]
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

    my_globals = set(globals()) - default_globals

    B = pass_maps["vg_b_vol"]
    v = pass_maps["proton/vg_v"]
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
        ax.plot(xg[idx], yg[idx], "x", color=CB_color_cycle[idx], zorder=2)

    if linsg:
        ax.plot([linsg[0], lineg[0]], [linsg[1], lineg[1]], alpha=0.3, zorder=1)

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

    if "umagten_g" in my_globals:
        if globals()["umagten_g"]:
            magten_arr = magten_vec(cellids, B, v)
            magten_mag = np.sqrt(
                magten_arr[:, :, 0] ** 2
                + magten_arr[:, :, 1] ** 2
                # + magten_arr[:, :, 2] ** 2
            )
            magten_mag_full = np.sqrt(
                magten_arr[:, :, 0] ** 2
                + magten_arr[:, :, 1] ** 2
                + magten_arr[:, :, 2] ** 2
            )
            umagten_x = magten_arr[:, :, 0] / (magten_mag + 1e-27)
            umagten_y = magten_arr[:, :, 1] / (magten_mag + 1e-27)
            ax.quiver(
                XmeshXY[::10, ::10],
                YmeshXY[::10, ::10],
                umagten_x[::10, ::10],
                umagten_y[::10, ::10],
                magten_mag_full[::10, ::10],
                scale_units="xy",
                angles="xy",
                pivot="mid",
                # scale=1,
                cmap="lipari",
                norm="log",
            )

    if "xo_g" in my_globals:
        if globals()["xo_g"]:
            try:
                x_points = np.loadtxt(
                    "/wrk-vakka/group/spacephysics/vlasiator/2D/AIC/visualization/x_and_o_points/x_point_location_{}.txt".format(
                        filenr_g
                    )
                )
                if len(x_points.shape) == 1:
                    ez = (
                        vobj.read_interpolated_variable(
                            "vg_e_vol", x_points, operator="z"
                        )
                        / 1e-3
                    )
                    ax.plot(
                        x_points[0] / r_e,
                        x_points[1] / r_e,
                        "x",
                        color="yellow",
                        fillstyle="none",
                        markersize=2 * highres_g * np.abs(ez),
                        zorder=10,
                    )
                else:
                    for xp in x_points:
                        ez = (
                            vobj.read_interpolated_variable(
                                "vg_e_vol", xp, operator="z"
                            )
                            / 1e-3
                        )
                        ax.plot(
                            xp[0] / r_e,
                            xp[1] / r_e,
                            "x",
                            color="yellow",
                            fillstyle="none",
                            markersize=2 * highres_g * np.abs(ez),
                            zorder=10,
                        )
            except:
                pass
            try:
                pass
                # o_points = np.loadtxt(
                #     "/wrk-vakka/group/spacephysics/vlasiator/2D/AIC/visualization/x_and_o_points/o_point_location_{}.txt".format(
                #         filenr_g
                #     )
                # )
                # if len(o_points.shape) == 1:
                #     ax.plot(
                #         o_points[0] / r_e,
                #         o_points[1] / r_e,
                #         "o",
                #         color="red",
                #         fillstyle="none",
                #         markersize=3 * highres_g,
                #         zorder=10,
                #     )
                # else:
                #     for op in o_points:
                #         ax.plot(
                #             op[0] / r_e,
                #             op[1] / r_e,
                #             "o",
                #             color="red",
                #             fillstyle="none",
                #             markersize=3 * highres_g,
                #             zorder=10,
                #         )
            except:
                pass

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

    lws = 0.4 * highres_g
    mrks = 2 * highres_g
    mews = 0.4 * highres_g

    if draw_qperp:
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
            hatches=["++", "/"],
            alpha=0.3,
        )

    if drawBy0:

        by0_cont = ax.contour(
            XmeshXY,
            YmeshXY,
            By,
            [0],
            linewidths=lws,
            colors="red",
            linestyles=["dashed"],
        )

    if "archer_g" in my_globals:
        if globals()["archer_g"]:
            jet_cont = ax.contour(
                XmeshXY,
                YmeshXY,
                jet_mask,
                [0.5],
                linewidths=lws,
                colors=CB_color_cycle[2],
                linestyles=["solid"],
            )
        else:
            pass
    else:
        jet_cont = ax.contour(
            XmeshXY,
            YmeshXY,
            jet_mask,
            [0.5],
            linewidths=lws,
            colors=CB_color_cycle[2],
            linestyles=["solid"],
        )

    if chg:
        ch_cont = ax.contour(
            XmeshXY,
            YmeshXY,
            ch_mask,
            [0.5],
            linewidths=lws,
            colors=CB_color_cycle[1],
            linestyles=["solid"],
        )
    if bsg:
        bs_cont = ax.contour(
            XmeshXY,
            YmeshXY,
            beta_star,
            [0.3],
            linewidths=lws,
            colors=CB_color_cycle[0],
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

    if mmsg:
        mach_cont = ax.contour(
            XmeshXY,
            YmeshXY,
            mach_mask,
            [0.5],
            linewidths=lws,
            colors=CB_color_cycle[3],
            linestyles=["solid"],
        )

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
    #         "$n=2n_\\mathrm{sw}$",
    #         "$T_\\mathrm{core}=3T_\\mathrm{sw}$",
    #         "$M_{\\mathrm{MS},x}=1$",
    #         "Jet",
    #         "FCS",
    #         "Non-FCS jet",
    #         "FCS-jet"
    #     )

    # proxy_labs = [
    #     # "$n=2n_\\mathrm{sw}$",
    #     # "$T_\\mathrm{core}=3T_\\mathrm{sw}$",
    #     "$\\beta^* = 0.3$",
    #     # "$M_{\\mathrm{MS},x}=1$",
    #     # "$P_\\mathrm{dyn,x}>0.25 P_\\mathrm{dyn,sw}$",
    # ]

    # proxy = [
    #     mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[itr]])
    #     for itr in range(len(proxy_labs))
    # ]
    proxy = []
    proxy_labs = []

    if bsg:
        proxy.append(mlines.Line2D([], [], color=CB_color_cycle[0]))
        proxy_labs.append("$\\beta^* = 0.3$")

    if chg:
        proxy.append(mlines.Line2D([], [], color=CB_color_cycle[1]))
        proxy_labs.append("$T_\\mathrm{core}=3T_\\mathrm{sw}$")

    if mmsg:
        proxy.append(mlines.Line2D([], [], color=CB_color_cycle[3]))
        proxy_labs.append("$M_{\\mathrm{MS},x}=1$")

    xmin, xmax, ymin, ymax = (
        np.min(XmeshXY),
        np.max(XmeshXY),
        np.min(YmeshXY),
        np.max(YmeshXY),
    )

    if plaschke_g:
        proxy.append(mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[7]]))
        proxy_labs.append("$P_\\mathrm{dyn,x}>0.25 P_\\mathrm{dyn,sw}$")
    if ~(jet_mask == 0).all():
        proxy.append(mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[3]]))
        proxy_labs.append(
            "$P_\\mathrm{dyn} \\geq 2 \\langle P_\\mathrm{dyn} \\rangle_\\mathrm{3min}$"
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
    if draw_qperp:
        proxy.append(
            mpatches.Patch(
                fc=CB_color_cycle[6],
                # color="black",
                # fill=True,
                hatch=r"++++",
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
    if leg_g:
        ax.legend(
            proxy,
            proxy_labs,
            frameon=True,
            numpoints=1,
            markerscale=1 * highres_g,
            loc="lower left",
            fontsize=5 * highres_g,
        )

    global gprox, gprox_labs

    gprox = proxy
    gprox_labs = proxy_labs


def get_jets(runid, min_duration=0, minsize=0):
    non_ids = []

    # singular_counter = 0

    for n1 in range(6000):
        try:
            props = PropReader(str(n1).zfill(5), runid, transient="jet")
        except:
            continue

        if props.read("at_bow_shock")[0] != 1:
            continue

        if props.read("time")[-1] - props.read("time")[0] + 0.5 < min_duration:
            continue

        if max(props.read("Nr_cells")) < minsize:
            continue

        if np.sqrt(props.read("x_mean") ** 2 + props.read("y_mean") ** 2)[0] < 8:
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


def fig1(runid, panel_nums=True, vmax=1.0):
    var = "proton/vg_Pdyn"
    vscale = 1e9
    # vmax = 1.0
    # vmax = 1.2
    runids = ["AGF", "AIA", "AIC"]

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g, start_points, drawBy0, plaschke_g, linsg, draw_qperp, leg_g
    runid_g = runid
    Blines_g = True
    drawBy0 = True
    plaschke_g = False
    linsg = False
    draw_qperp = True
    leg_g = True

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

    pdmax = [1.5, 1.5, 1.5][runids.index(runid)]
    sw_pars = [
        [1e6, 750e3, 3e-9, 0.5e6],
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
    annot_lab = ["RD", "Jets", "Fast-mode\npulse", "MP\ndeformed"]
    annot_pan = ["a", "b", "c", "d"]
    annot_xy = [(16, 9), (13.5, -3), (11.5, 1), (8, 1)]
    annot_xytext = [(12, 12), (16, -9), (6, 9), (1, -7)]
    arrowc = ["k", "k", "k", "w"]

    for idx, fnr in enumerate([820, 880, 900, 1190]):
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
        ax_list[idx].annotate(
            annot_lab[idx],
            xy=annot_xy[idx],
            xytext=annot_xytext[idx],
            fontsize=10,
            arrowprops=dict(
                facecolor=arrowc[idx],
                ec=arrowc[idx],
                shrink=0.1,
                width=1,
                headwidth=3,
            ),
            bbox=dict(
                boxstyle="square,pad=0.15",
                fc="white",
                ec="k",
                lw=0.5,
            ),
        )
        if panel_nums:
            ax_list[idx].annotate(
                annot_pan[idx],
                (0.05, 0.90),
                xycoords="axes fraction",
                fontsize=12,
                bbox=dict(
                    boxstyle="square,pad=0.15",
                    fc="white",
                    ec="k",
                    lw=0.5,
                ),
            )
        if not legon[idx]:
            ax_list[idx].get_legend().remove()
        # else:
        #     ax_list[idx].get_legend().set(fontsize=12)

    for ax in ax_list:
        ax.label_outer()

    fig.savefig(outputdir + "fig1.pdf")
    fig.savefig(outputdir + "../fig1.png", dpi=300)
    plt.close(fig)


def fig1_new(
    runid,
    var="proton/vg_Pdyn",
    op=None,
    boxre=[-10, 20, -20, 20],
    tickint=5.0,
    blines=False,
    vscale=1e9,
    nstp=40,
    pdynmax=1.5,
    pdynmin=0.1,
    outdir="cmaps",
    cmap="batlow",
    pointsx=[],
    pointsy=[],
    fsaved=None,
    lin=1,
    By0=True,
    leg=True,
    track_jets=True,
    qperp=False,
    linestartstop=[],
    magten=False,
    usesci=0,
    magtenvec=False,
    pt_blines=False,
    min_duration=0,
    minsize=0,
    highres=None,
    plot_fluxfunc=True,
    draw_ch=True,
    draw_bs=False,
    draw_mms=True,
):

    if magten:
        vscale = 1
        expression = expr_magten
        usesci = 1
    else:
        vscale = vscale
        expression = None
        usesci = 0

    vmax = pdynmax
    runids = ["AGF", "AIA", "AIC"]

    if len(pointsx) != len(pointsy):
        print("x and y must have same length!")
        return 1

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g, start_points, drawBy0, plaschke_g, leg_g, draw_qperp, vobj, umagten_g, chg, highres_g, bsg, mmsg
    umagten_g = magtenvec
    runid_g = runid
    Blines_g = blines
    drawBy0 = By0
    plaschke_g = False
    leg_g = leg
    draw_qperp = qperp
    chg = draw_ch
    highres_g = highres
    bsg = draw_bs
    mmsg = draw_mms

    global xg, yg, linsg, lineg
    xg = pointsx
    yg = pointsy
    linsg, lineg = None, None
    if len(linestartstop) == 2:
        linsg = linestartstop[0]
        lineg = linestartstop[1]

    # nstp = 40
    start_points = np.array(
        [
            np.ones(nstp) * boxre[1] - 1,
            np.linspace(boxre[2] + 0.1, boxre[3] - 0.1, nstp),
        ]
        # [
        #     np.linspace(boxre[0] + 0.1, boxre[1] - 0.1, nstp),
        #     np.ones(nstp) * (boxre[2] + 1),
        # ]
    ).T

    bulkpath = find_bulkpath(runid)

    if track_jets:
        non_ids = get_jets(runid, min_duration=min_duration, minsize=minsize)
    else:
        non_ids = []

    sj_ids_g = []
    non_ids_g = non_ids

    sw_pars = [
        [1e6, 750e3, 3e-9, 0.5e6],
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

    if pt_blines:
        streamlines = "vg_b_vol"
    else:
        streamlines = None

    annot_pan = ["a", "b", "c", "d"]
    nodrawcb = [True, True, True, False]
    drawleg = [True, False, False, False]

    fig, ax_list = plt.subplots(2, 2, figsize=(9, 10), constrained_layout=True)
    ax_flat = ax_list.flatten()

    for idx, fnr in enumerate([781, 880, 1060, 1392]):
        filenr_g = fnr

        fname = "bulk.{}.vlsv".format(str(int(fnr)).zfill(7))

        vobj = pt.vlsvfile.VlsvReader(bulkpath + fname)
        leg_g = drawleg[idx]

        if plot_fluxfunc:
            fluxfile = vlasdir + "/2D/AIC/fluxfunction/" + fname + ".bin"
            fluxdir = None
            flux_levels = None
            fluxthick = 0.5
            fluxlines = 5
        else:
            fluxfile = None
            fluxdir = None
            flux_levels = None
            fluxthick = 1.0
            fluxlines = 1

        pt.plot.plot_colormap(
            axes=ax_flat[idx],
            vlsvobj=vobj,
            var=var,
            op=op,
            vmin=pdynmin,
            # vmax=1,
            vmax=vmax,
            vscale=vscale,
            # cbtitle="",
            # cbtitle="",
            usesci=usesci,
            # scale=3,
            title="Run: {}$~$t = {}s".format(runid, float(fnr) / 2.0),
            boxre=boxre,
            internalcb=True,
            lin=lin,
            highres=highres,
            colormap=cmap,
            tickinterval=tickint,
            fsaved=fsaved,
            useimshow=True,
            external=ext_jet,
            nocb=nodrawcb[idx],
            expression=expression,
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
            streamlines=streamlines,
            streamlinedensity=0.3,
            streamlinecolor="white",
            streamlinethick=0.8,
            fluxfile=fluxfile,
            fluxdir=fluxdir,
            flux_levels=flux_levels,
            fluxthick=fluxthick,
            fluxlines=fluxlines,
        )

        ax_flat[idx].annotate(
            annot_pan[idx],
            (0.05, 0.95),
            xycoords="axes fraction",
            fontsize=12,
            bbox=dict(
                boxstyle="square,pad=0.15",
                fc="white",
                ec="k",
                lw=0.5,
            ),
        )

    fig.savefig(wrkdir_DNR + "Figs/fig1_new.pdf", dpi=300)
    plt.close(fig)


def v5_plotter(
    runid,
    start,
    stop,
    var="proton/vg_Pdyn",
    op=None,
    boxre=None,
    tickint=5.0,
    blines=False,
    vscale=1e9,
    nstp=40,
    pdynmax=1.5,
    pdynmin=0.1,
    outdir="cmaps",
    cmap="batlow",
    pointsx=[],
    pointsy=[],
    fsaved=None,
    lin=1,
    By0=True,
    leg=True,
    track_jets=True,
    qperp=False,
    linestartstop=[],
    magten=False,
    usesci=0,
    magtenvec=False,
    pt_blines=False,
    min_duration=0,
    minsize=0,
    highres=None,
    plot_fluxfunc=False,
    draw_ch=False,
    draw_bs=True,
    draw_mms=False,
    draw_archer=True,
    draw_xo=False,
    fluxlines=5,
):

    if magten:
        vscale = 1
        expression = expr_magten
        usesci = 1
    else:
        vscale = vscale
        expression = None
        usesci = 0

    vmax = pdynmax
    runids = ["AGF", "AIA", "AIC"]

    if len(pointsx) != len(pointsy):
        print("x and y must have same length!")
        return 1

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g, start_points, drawBy0, plaschke_g, leg_g, draw_qperp, vobj, umagten_g, chg, highres_g, bsg, mmsg, archer_g, xo_g
    xo_g = draw_xo
    umagten_g = magtenvec
    runid_g = runid
    Blines_g = blines
    drawBy0 = By0
    plaschke_g = False
    leg_g = leg
    draw_qperp = qperp
    chg = draw_ch
    highres_g = highres
    bsg = draw_bs
    mmsg = draw_mms
    archer_g = draw_archer

    global xg, yg, linsg, lineg
    xg = pointsx
    yg = pointsy
    linsg, lineg = None, None
    if len(linestartstop) == 2:
        linsg = linestartstop[0]
        lineg = linestartstop[1]

    # nstp = 40
    # start_points = np.array(
    #     [
    #         np.ones(nstp) * boxre[1] - 1,
    #         np.linspace(boxre[2] + 0.1, boxre[3] - 0.1, nstp),
    #     ]
    #     # [
    #     #     np.linspace(boxre[0] + 0.1, boxre[1] - 0.1, nstp),
    #     #     np.ones(nstp) * (boxre[2] + 1),
    #     # ]
    # ).T

    bulkpath = find_bulkpath(runid)

    if track_jets:
        non_ids = get_jets(runid, min_duration=min_duration, minsize=minsize)
    else:
        non_ids = []

    sj_ids_g = []
    non_ids_g = non_ids

    pdmax = [1.5, 1.5, 1.5][runids.index(runid)]
    sw_pars = [
        [1e6, 750e3, 3e-9, 0.5e6],
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

    if pt_blines:
        streamlines = "vg_b_vol"
    else:
        streamlines = None

    for fnr in range(start, stop + 1):
        filenr_g = fnr

        fname = "bulk.{}.vlsv".format(str(int(fnr)).zfill(7))

        vobj = pt.vlsvfile.VlsvReader(bulkpath + fname)

        if plot_fluxfunc:
            fluxfile = vlasdir + "/2D/AIC/fluxfunction/" + fname + ".bin"
            fluxdir = None
            flux_levels = None
            fluxthick = 0.5
            fluxlines = fluxlines
        else:
            fluxfile = None
            fluxdir = None
            flux_levels = None
            fluxthick = 1.0
            fluxlines = 1

        pt.plot.plot_colormap(
            vlsvobj=vobj,
            outputfile=outputdir + "pdyn_{}.png".format(str(fnr).zfill(7)),
            var=var,
            op=op,
            vmin=pdynmin,
            # vmax=1,
            vmax=vmax,
            vscale=vscale,
            # cbtitle="",
            # cbtitle="",
            usesci=usesci,
            # scale=3,
            title="Run: {}$~$t = {}s".format(runid, float(fnr) / 2.0),
            boxre=boxre,
            internalcb=False,
            lin=lin,
            highres=highres,
            colormap=cmap,
            tickinterval=tickint,
            fsaved=fsaved,
            # useimshow=True,
            external=ext_jet,
            expression=expression,
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
            streamlines=streamlines,
            streamlinedensity=0.3,
            streamlinecolor="white",
            streamlinethick=0.8,
            fluxfile=fluxfile,
            fluxdir=fluxdir,
            flux_levels=flux_levels,
            fluxthick=fluxthick,
            fluxlines=fluxlines,
        )


def magten_vec(outcells, B, v):

    origshape = outcells.shape
    outcells = outcells.flatten()
    # print(outcells.shape)
    # print(B.shape)
    B = np.reshape(B, (outcells.size, 3))
    v = np.reshape(v, (outcells.size, 3))
    # print(v.shape)
    vg_b_jacobian = make_vg_b_jacobian(vobj)
    # print(vg_b_jacobian.shape)
    vg_b_jacobian = vg_b_jacobian[(outcells - 1)]

    B_reshaped = np.rollaxis(np.array([B]), 1, 0)

    magten = np.rollaxis(B_reshaped @ vg_b_jacobian, 1, 0)[0] / mu0

    magten = np.reshape(magten, (origshape[0], origshape[1], 3))

    return magten


def expr_magten(pass_maps):

    outcells = pass_maps["CellID"]
    origshape = outcells.shape
    outcells = outcells.flatten()
    print(outcells.shape)
    B = pass_maps["vg_b_vol"]
    v = pass_maps["proton/vg_v"]
    print(B.shape)
    B = np.reshape(B, (outcells.size, 3))
    Bmag = np.linalg.norm(B, axis=-1)
    v = np.reshape(v, (outcells.size, 3))
    print(v.shape)
    vg_b_jacobian = make_vg_b_jacobian(vobj)
    print(vg_b_jacobian.shape)
    vg_b_jacobian = vg_b_jacobian[(outcells - 1)]

    B_reshaped = np.rollaxis(np.array([B]), 1, 0)

    magten = np.rollaxis(B_reshaped @ vg_b_jacobian, 1, 0)[0] / mu0

    # magten = (magten.T / Bmag / Bmag).T
    # magten = (magten.T / np.linalg.norm(magten, axis=-1)).T
    magten = (magten * v).sum(axis=-1)  # * 1e-3
    print(magten.shape)
    magten = np.reshape(magten, (origshape[0], origshape[1]))

    # magten = np.reshape(magten, (origshape[0], origshape[1], 3))
    # magten = np.linalg.norm(magten, axis=-1)

    return magten


def VSC_time_Ecomponents(
    runid,
    x0,
    y0,
    t0,
    t1,
):
    bulkpath = find_bulkpath(runid)

    var_list = [
        "proton/vg_rho",
        "proton/vg_v",
        "vg_b_vol",
        "vg_e_vol",
        "vg_pressure",
        "vg_e_gradpe",
    ]

    ylabels = [
        "$v$ [km/s]",
        "$E_x$ [mV/m]",
        "$E_y$ [mV/m]",
        "$E_z$ [mV/m]",
    ]

    complabels = [
        "$-\\mathbf{v}\\times\\mathbf{B}$",
        "$\\mathbf{J}\\times\\mathbf{B}/ne$",
        "$\\mathbf{B}\\cdot\\nabla\\mathbf{B}/\\mu_0 ne$",
        "$-\\nabla(B^2)/2\\mu_0 ne$",
        "$-\\nabla(P_e)/ne$",
    ]

    t_arr = np.arange(t0, t1 + 0.1, 0.5)
    data_arr = np.zeros((len(complabels) + 1, t_arr.size, 3), dtype=float)

    for idx in range(t_arr.size):
        fnr = int(t_arr[idx] * 2)
        vlsvobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
        )

        rho, v, B, E, Pressure, EgradPe = [
            vlsvobj.read_interpolated_variable(var, [x0 * r_e, y0 * r_e, 0])
            for var in var_list
        ]

        data_arr[0, idx, :] = v * 1e-3
        data_arr[1, idx, :] = -np.cross(v, B) * 1e3
        data_arr[2, idx, :] = (
            (
                -pos_mag_gradient(vlsvobj, x0 * r_e, y0 * r_e)
                + pos_mag_tension(vlsvobj, x0 * r_e, y0 * r_e)
            )
            / q_p
            / rho
            * 1e3
        )
        data_arr[3, idx, :] = (
            pos_mag_tension(vlsvobj, x0 * r_e, y0 * r_e) / q_p / rho * 1e3
        )
        data_arr[4, idx, :] = (
            -pos_mag_gradient(vlsvobj, x0 * r_e, y0 * r_e) / q_p / rho * 1e3
        )
        data_arr[5, idx, :] = -EgradPe * 1e3

    fig, ax_list = plt.subplots(
        len(ylabels), 1, figsize=(9, 9), constrained_layout=True
    )

    ax_list[0].plot(t_arr, data_arr[0, :, 0], color=CB_color_cycle[0], label="x")
    ax_list[0].plot(t_arr, data_arr[0, :, 1], color=CB_color_cycle[1], label="y")
    ax_list[0].plot(t_arr, data_arr[0, :, 2], color=CB_color_cycle[2], label="z")
    ax_list[0].plot(
        t_arr, np.linalg.norm(data_arr[0, :, :], axis=-1), color="k", label="mag"
    )
    ax_list[0].legend()

    for idx in range(len(complabels)):
        ax_list[1].plot(
            t_arr,
            data_arr[idx + 1, :, 0],
            color=CB_color_cycle[idx],
            label=complabels[idx],
        )
        ax_list[2].plot(
            t_arr,
            data_arr[idx + 1, :, 1],
            color=CB_color_cycle[idx],
            label=complabels[idx],
        )
        ax_list[3].plot(
            t_arr,
            data_arr[idx + 1, :, 2],
            color=CB_color_cycle[idx],
            label=complabels[idx],
        )

    for idx, ax in enumerate(ax_list):
        ax.grid()
        ax.set_xlim(t_arr[0], t_arr[-1])
        ax.set_ylabel(ylabels[idx])

    ax_list[1].legend()
    ax_list[-1].set_xlabel("Simulation time [s]")

    figdir = wrkdir_DNR + "Figs/timeseries/"
    if not os.path.exists(figdir):
        try:
            os.makedirs(figdir)
        except OSError:
            pass

    fig.savefig(
        figdir
        + "Ecomps_{}_x{}_y{}_t0{}_t1{}.png".format(
            runid,
            x0,
            y0,
            t0,
            t1,
        ),
        dpi=300,
    )
    plt.close(fig)


def VSC_cut_Ecomponents(
    runid,
    x0,
    y0,
    x1,
    y1,
    dr,
    t0,
):
    bulkpath = find_bulkpath(runid)

    var_list = [
        "proton/vg_rho",
        "proton/vg_v",
        "vg_b_vol",
        "vg_e_vol",
        "vg_pressure",
        "vg_e_gradpe",
    ]

    ylabels = [
        "$v$ [km/s]",
        "$E_x$ [mV/m]",
        "$E_y$ [mV/m]",
        "$E_z$ [mV/m]",
    ]

    complabels = [
        "$-\\mathbf{v}\\times\\mathbf{B}$",
        "$\\mathbf{J}\\times\\mathbf{B}/ne$",
        "$\\mathbf{B}\\cdot\\nabla\\mathbf{B}/\\mu_0 ne$",
        "$-\\nabla(B^2)/2\\mu_0 ne$",
        "$-\\nabla(P_e)/ne$",
    ]

    alpha = np.arctan2(y1 - y0, x1 - x0)
    dx = dr * np.cos(alpha)
    nx = 1 + int((x1 - x0) / dx)
    x_arr = np.linspace(x0, x1, nx) * r_e
    y_arr = np.linspace(y0, y1, nx) * r_e
    n_arr = np.arange(x_arr.size)

    fnr0 = int(t0 * 2)
    data_arr = np.zeros((len(complabels) + 1, x_arr.size, 3), dtype=float)
    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
    )

    for idx in range(x_arr.size):

        rho, v, B, E, Pressure, EgradPe = [
            vlsvobj.read_interpolated_variable(var, [x_arr[idx], y_arr[idx], 0])
            for var in var_list
        ]

        data_arr[0, idx, :] = v * 1e-3
        data_arr[1, idx, :] = -np.cross(v, B) * 1e3
        data_arr[2, idx, :] = (
            (
                -pos_mag_gradient(vlsvobj, x_arr[idx], y_arr[idx])
                + pos_mag_tension(vlsvobj, x_arr[idx], y_arr[idx])
            )
            / q_p
            / rho
            * 1e3
        )
        data_arr[3, idx, :] = (
            pos_mag_tension(vlsvobj, x_arr[idx], y_arr[idx]) / q_p / rho * 1e3
        )
        data_arr[4, idx, :] = (
            -pos_mag_gradient(vlsvobj, x_arr[idx], y_arr[idx]) / q_p / rho * 1e3
        )
        data_arr[5, idx, :] = -EgradPe * 1e3

    fig, ax_list = plt.subplots(
        len(ylabels), 1, figsize=(9, 9), constrained_layout=True
    )

    ax_list[0].plot(n_arr, data_arr[0, :, 0], color=CB_color_cycle[0], label="x")
    ax_list[0].plot(n_arr, data_arr[0, :, 1], color=CB_color_cycle[1], label="y")
    ax_list[0].plot(n_arr, data_arr[0, :, 2], color=CB_color_cycle[2], label="z")
    ax_list[0].plot(
        n_arr, np.linalg.norm(data_arr[0, :, :], axis=-1), color="k", label="mag"
    )
    ax_list[0].legend()

    for idx in range(len(complabels)):
        ax_list[1].plot(
            n_arr,
            data_arr[idx + 1, :, 0],
            color=CB_color_cycle[idx],
            label=complabels[idx],
        )
        ax_list[2].plot(
            n_arr,
            data_arr[idx + 1, :, 1],
            color=CB_color_cycle[idx],
            label=complabels[idx],
        )
        ax_list[3].plot(
            n_arr,
            data_arr[idx + 1, :, 2],
            color=CB_color_cycle[idx],
            label=complabels[idx],
        )

    for idx, ax in enumerate(ax_list):
        ax.grid()
        ax.set_xlim(n_arr[0], n_arr[-1])
        ax.set_ylabel(ylabels[idx])

    ax_list[1].legend()
    ax_list[-1].set_xlabel("Point along cut")

    figdir = wrkdir_DNR + "Figs/cuts/"
    if not os.path.exists(figdir):
        try:
            os.makedirs(figdir)
        except OSError:
            pass

    fig.savefig(
        figdir
        + "Ecomps_{}_x{}_{}_y{}_{}_t0{}.png".format(
            runid,
            x0,
            x1,
            y0,
            y1,
            t0,
        ),
        dpi=300,
    )
    plt.close(fig)


def speiser(
    runid,
    x0,
    y0,
    x1,
    dr,
    t0,
    vdc=-85.3441844657656,
    polydeg=5,
    nsteps=1000,
    dt=0.01,
    xoffset=900,
    vx0=-500,
    diag=False,
):

    bulkpath = find_bulkpath(runid)
    var_list = [
        "vg_b_vol",
        "vg_b_vol",
        "vg_b_vol",
        "vg_e_vol",
        "vg_e_vol",
        "vg_e_vol",
    ]
    ops = [
        "x",
        "y",
        "z",
        "x",
        "y",
        "z",
    ]

    x_arr = np.linspace(x0, x1, int((x1 - x0) / dr) + 1) * r_e

    fnr0 = int(t0 * 2)
    data_arr = np.zeros((6, x_arr.size), dtype=float)
    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
    )
    for idx in range(x_arr.size):

        for idx2, var in enumerate(var_list):
            data_arr[idx2, idx] = vlsvobj.read_interpolated_variable(
                var, [x_arr[idx], y0 * r_e, 0], operator=ops[idx2]
            )

    polys = []

    for idx in range(len(var_list)):
        poly = np.polynomial.Polynomial.fit(x_arr, data_arr[idx, :], deg=polydeg)
        polys.append(poly)

    figdir = wrkdir_DNR + "Figs/speiser/"
    if not os.path.exists(figdir):
        try:
            os.makedirs(figdir)
        except OSError:
            pass

    if diag:
        fig, ax_list = plt.subplots(
            len(var_list), 1, figsize=(12, 12), constrained_layout=True
        )

        for idx in range(len(var_list)):
            ax_list[idx].grid()
            ax_list[idx].plot(x_arr, data_arr[idx, :], color="black")
            ax_list[idx].plot(
                x_arr, polys[idx](x_arr), color="black", linestyle="dashed"
            )
            ax_list[idx].set_xlim(x_arr[0], x_arr[-1])

        ax_list[0].set_title("Polynomial degree = {}".format(polydeg))

        fig.savefig(
            figdir
            + "diag_{}_x{}_{}_y{}_t0{}_polydeg{}.png".format(
                runid,
                x0,
                x1,
                y0,
                t0,
                polydeg,
            ),
            dpi=300,
        )
        plt.close(fig)

    time_arr = np.zeros(nsteps, dtype=float)
    time_arr[0] = t0
    t = t0

    xby0 = x_arr[np.argsort(np.abs(data_arr[1, :]))][0]
    xrdo = 0

    x, y, z = (xby0 + xoffset * 1e3, y0 * r_e, 0)
    xarr = np.zeros_like(time_arr)
    yarr = np.zeros_like(time_arr)
    zarr = np.zeros_like(time_arr)
    vxarr = np.zeros_like(time_arr)
    vyarr = np.zeros_like(time_arr)
    vzarr = np.zeros_like(time_arr)
    xarr[0] = x
    yarr[0] = y
    zarr[0] = z
    vx, vy, vz = (vx0 * 1e3, 0, 0)
    vxarr[0] = vx
    vyarr[0] = vy
    vzarr[0] = vz

    Barr = np.zeros((nsteps, 3), dtype=float)
    Barr[0, :] = np.array([polys[0](x - xrdo), polys[1](x - xrdo), polys[2](x - xrdo)])

    for n in range(1, nsteps):
        E = np.array([polys[3](x - xrdo), polys[4](x - xrdo), polys[5](x - xrdo)])
        B = np.array([polys[0](x - xrdo), polys[1](x - xrdo), polys[2](x - xrdo)])
        Omega = (q_p / m_p) * B
        v = np.array([vx, vy, vz])
        v1 = v + (q_p * dt / m_p / 2.0) * E
        v2 = (
            v * (1 - (np.linalg.norm(Omega) * dt / 2) ** 2)
            + np.cross(v1, Omega) * dt
            + 0.5 * (dt**2) * np.dot(v1, Omega) * Omega
        ) / (1 + (np.linalg.norm(Omega) * dt / 2) ** 2)
        v3 = v2 + (q_p * dt / m_p / 2.0) * E

        x = x + vx * dt
        y = y + vy * dt
        z = z + vz * dt
        vx = v3[0]
        vy = v3[1]
        vz = v3[2]
        t = t + dt
        xrdo = xrdo + vdc * 1e3 * dt

        xarr[n] = x
        yarr[n] = y
        zarr[n] = z

        vxarr[n] = vx
        vyarr[n] = vy
        vzarr[n] = vz

        time_arr[n] = t
        Barr[n, :] = np.array(
            [polys[0](x - xrdo), polys[1](x - xrdo), polys[2](x - xrdo)]
        )

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    ax[0].grid()
    ax[0].plot(xarr / r_e, yarr / r_e)
    ax[0].plot(xarr[0] / r_e, yarr[0] / r_e, "o", color="blue")
    ax[0].axvline(xby0 / r_e, linestyle="dashed", color="red")
    ax[0].set(xlabel="X [RE]", ylabel="Y [RE]")
    ax[1].grid()
    ax[1].plot(xarr / r_e, zarr / r_e)
    ax[1].plot(xarr[0] / r_e, zarr[0] / r_e, "o", color="blue")
    ax[1].axvline(xby0 / r_e, linestyle="dashed", color="red")
    ax[1].set(xlabel="X [RE]", ylabel="Z [RE]")

    fig.savefig(
        figdir
        + "xyz_{}_x{}_{}_y{}_t0{}_xoffset{}_vx0{}.png".format(
            runid,
            x0,
            x1,
            y0,
            t0,
            xoffset,
            vx0,
        ),
        dpi=300,
    )
    plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    ax[0].grid()
    ax[0].plot(time_arr, xarr / r_e, color=CB_color_cycle[0], label="x")
    ax[0].plot(time_arr, yarr / r_e, color=CB_color_cycle[1], label="y")
    ax[0].plot(time_arr, zarr / r_e, color=CB_color_cycle[2], label="z")
    ax[0].plot(
        time_arr,
        np.sqrt(xarr**2 + yarr**2 + zarr**2) / r_e,
        color="black",
        linestyle="dashed",
        label="r",
    )
    ax[0].legend()
    ax[0].set(xlabel="Time [s]", ylabel="r [RE]", xlim=(time_arr[0], time_arr[-1]))

    ax[1].grid()
    ax[1].plot(time_arr, vxarr / 1e3, color=CB_color_cycle[0], label="vx")
    ax[1].plot(time_arr, vyarr / 1e3, color=CB_color_cycle[1], label="vy")
    ax[1].plot(time_arr, vzarr / 1e3, color=CB_color_cycle[2], label="vz")
    ax[1].plot(
        time_arr,
        np.sqrt(vxarr**2 + vyarr**2 + vzarr**2) / 1e3,
        color="black",
        linestyle="dashed",
        label="v",
    )
    ax[1].legend()
    ax[1].set(xlabel="Time [s]", ylabel="v [km/s]", xlim=(time_arr[0], time_arr[-1]))

    vpararr = (np.array([vxarr, vyarr, vzarr]).T * Barr).sum(-1) / np.linalg.norm(
        Barr, axis=-1
    )
    vperparr = np.sqrt(vxarr**2 + vyarr**2 + vzarr**2 - vpararr**2)

    ax[2].grid()
    ax[2].plot(time_arr, vpararr / 1e3, color=CB_color_cycle[0], label="vpar")
    ax[2].plot(time_arr, vperparr / 1e3, color=CB_color_cycle[1], label="vperp")
    ax[2].plot(
        time_arr,
        np.sqrt(vxarr**2 + vyarr**2 + vzarr**2) / 1e3,
        color="black",
        linestyle="dashed",
        label="v",
    )
    ax[2].legend()
    ax[2].set(xlabel="Time [s]", ylabel="v [km/s]", xlim=(time_arr[0], time_arr[-1]))

    fig.savefig(
        figdir
        + "time_{}_x{}_{}_y{}_t0{}_xoffset{}_vx0{}.png".format(
            runid,
            x0,
            x1,
            y0,
            t0,
            xoffset,
            vx0,
        ),
        dpi=300,
    )
    plt.close(fig)


def VSC_cut_through(
    runid,
    x0,
    y0,
    x1,
    y1,
    dr,
    t0,
    pdx=False,
    vlines=[],
    fourier=None,
    pdavg=False,
    plot_gyro=False,
    dirprefix="",
):
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
        "$P_\\mathrm{dyn}$",
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
        "$\\rho~[\\mathrm{cm}^{-3}]$",
        "$v~[\\mathrm{km/s}]$",
        "$P_\\mathrm{dyn}~[\\mathrm{nPa}]$",
        "$B~[\\mathrm{nT}]$",
        "$E~[\\mathrm{mV/m}]$",
        "$T~[\\mathrm{MK}]$",
        # "$\\rho~[\\rho_\\mathrm{sw}]$",
        # "$v~[v_\\mathrm{sw}]$",
        # "$P_\\mathrm{dyn}~[P_\\mathrm{dyn,sw}]$",
        # "$B~[B_\\mathrm{IMF}]$",
        # "$E~[E_\\mathrm{sw}]$",
        # "$T~[T_\\mathrm{sw}]$",
    ]
    e_sw = 750e3 * 3e-9 * q_p / m_p * 1e3
    pdsw = m_p * 1e6 * 750e3 * 750e3 * 1e9
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

    alpha = np.arctan2(y1 - y0, x1 - x0)
    dx = dr * np.cos(alpha)
    nx = 1 + int((x1 - x0) / dx)
    x_arr = np.linspace(x0, x1, nx) * r_e
    y_arr = np.linspace(y0, y1, nx) * r_e
    n_arr = np.arange(x_arr.size)

    fnr0 = int(t0 * 2)
    data_arr = np.zeros((len(var_list) + 3, x_arr.size), dtype=float)
    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
    )
    cellids = np.array(
        [vlsvobj.get_cellid([x_arr[idx], y_arr[idx], 0]) for idx in range(len(x_arr))]
    )
    cellid_coords = np.array(
        [vlsvobj.get_cell_coordinates(cellid) for cellid in cellids]
    )
    try:
        pdavg_arr = np.loadtxt(tavgdir + runid + "/" + str(fnr0) + "_pdyn.tavg")[
            cellids - 1
        ]
        pdavg_arr_interp = np.interp(x_arr, cellid_coords[:, 0], pdavg_arr) * 1e9
    except:
        pdavg_arr = np.empty_like(x_arr)
        pdavg_arr[:] = np.nan
        pdavg_arr_interp = pdavg_arr * 1e9

    for idx in range(x_arr.size):

        for idx2, var in enumerate(var_list):
            data_arr[idx2, idx] = (
                vlsvobj.read_interpolated_variable(
                    var, [x_arr[idx], y_arr[idx], 0], operator=ops[idx2]
                )
                * scales[idx2]
            )
        data_arr[[idx2 + 1, idx2 + 2, idx2 + 3], idx] = 1e9 * (
            pos_mag_tension(vlsvobj, x_arr[idx], y_arr[idx])
        )

    fig, ax_list = plt.subplots(
        len(ylabels) + 1, 1, sharex=True, figsize=(6, 8), constrained_layout=True
    )
    ax_list[0].set_title(
        "Run: {}, $(x,y)_0$: {}, $(x,y)_1$: {}".format(runid, (x0, y0), (x1, y1))
    )
    for idx in range(len(var_list)):
        ax = ax_list[plot_index[idx]]
        for vline in vlines:
            ax.axvline(vline, linestyle="dashed", linewidth=0.6)
        ax.plot(n_arr, data_arr[idx], color=plot_colors[idx], label=plot_labels[idx])
        if idx == 5:
            if pdx:
                pdynx = (
                    m_p
                    * data_arr[0]
                    * 1e6
                    * data_arr[1]
                    * 1e3
                    * data_arr[1]
                    * 1e3
                    * 1e9
                )
                ax.plot(
                    n_arr,
                    pdynx,
                    color=CB_color_cycle[0],
                    label="$P_{\\mathrm{dyn},x}$",
                )
            if pdavg:
                ax.plot(
                    n_arr,
                    2 * pdavg_arr_interp,
                    color=CB_color_cycle[1],
                    linestyle="dashed",
                    label="$2\\langle P_\\mathrm{dyn} \\rangle$",
                )
                ax.axhline(
                    0.25 * pdsw,
                    linestyle="dotted",
                    color=CB_color_cycle[2],
                    label="$0.25P_\\mathrm{dyn,sw}$",
                )
                ax.axhline(
                    0.5 * pdsw,
                    linestyle="dotted",
                    color=CB_color_cycle[3],
                    label="$0.5P_\\mathrm{dyn,sw}$",
                )

        ax.set_xlim(n_arr[0], n_arr[-1])
        if draw_legend[idx]:
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    ylabels.append("$P_\\mathrm{dyn}$ contribution")

    avg_pdyn = np.nanmean(data_arr[5, :]) * 1e-9
    avg_rho = np.nanmean(data_arr[0, :]) * 1e6
    # avg_vx2 = np.nanmean(data_arr[1,:]*data_arr[1,:])*1e6
    # avg_vy2 = np.nanmean(data_arr[2,:]*data_arr[2,:])*1e6
    # avg_vz2 = np.nanmean(data_arr[3,:]*data_arr[3,:])*1e6
    avg_vt2 = np.nanmean(data_arr[4, :] * data_arr[4, :]) * 1e6

    rho_contrib = m_p * data_arr[0, :] * 1e6 * avg_vt2 / (avg_pdyn + 1e-27)
    vx2_contrib = (
        m_p * avg_rho * (data_arr[1, :] * data_arr[1, :]) * 1e6 / (avg_pdyn + 1e-27)
    )
    vy2_contrib = (
        m_p * avg_rho * (data_arr[2, :] * data_arr[2, :]) * 1e6 / (avg_pdyn + 1e-27)
    )
    vz2_contrib = (
        m_p * avg_rho * (data_arr[3, :] * data_arr[3, :]) * 1e6 / (avg_pdyn + 1e-27)
    )

    ax_list[-1].plot(
        n_arr,
        rho_contrib,
        color="k",
        # label="$\\nabla p$",
        label="$\\rho$",
    )
    ax_list[-1].plot(
        n_arr,
        vx2_contrib,
        color=CB_color_cycle[0],
        label="$v_x^2$",
    )
    ax_list[-1].plot(
        n_arr,
        vy2_contrib,
        color=CB_color_cycle[1],
        label="$v_y^2$",
    )
    ax_list[-1].plot(
        n_arr,
        vz2_contrib,
        color=CB_color_cycle[2],
        label="$v_z^2$",
    )
    ax_list[-1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    for vline in vlines:
        ax_list[-1].axvline(vline, linestyle="dashed", linewidth=0.6)
    ax_list[-1].set_xlim(n_arr[0], n_arr[-1])
    ax_list[-1].set_xlabel("Point along cut")
    for idx, ax in enumerate(ax_list):
        ax.grid()
        ax.set_ylabel(ylabels[idx])
        # ax.axvline(t0, linestyle="dashed")
    # plt.tight_layout()
    figdir = wrkdir_DNR + "Figs/cuts/{}".format(dirprefix)
    txtdir = wrkdir_DNR + "txts/cuts/{}".format(dirprefix)
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
        figdir
        + "{}_x{}_{}_y{}_{}_t0{}.png".format(
            runid,
            x0,
            x1,
            y0,
            y1,
            t0,
        ),
        dpi=300,
    )
    np.savetxt(
        txtdir
        + "{}_x{}_{}_y{}_{}_t0{}.txt".format(
            runid,
            x0,
            x1,
            y0,
            y1,
            t0,
        ),
        data_arr,
    )
    plt.close(fig)

    if plot_gyro:
        gyroperiod = 2 * np.pi / (q_p * data_arr[9, :] * 1e-9 / m_p)
        vel_magnitude = data_arr[4, :] * 1e3
        gyro_distance = gyroperiod * vel_magnitude

        fig, ax_list = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)

        ax_list[0].plot(n_arr, gyroperiod)
        ax_list[0].set_ylabel("Gyroperiod [s]")

        ax_list[1].plot(n_arr, vel_magnitude * 1e-3)
        ax_list[1].set_ylabel("Plasma speed [km/s]")

        ax_list[2].plot(n_arr, gyro_distance / r_e)
        ax_list[2].set_ylabel("Gyro distance [RE]")

        ax_list[-1].set_xlabel("Point along cut")
        ax_list[0].set_title(
            "Run: {}, $(x,y)_0$: {}, $(x,y)_1$: {}".format(runid, (x0, y0), (x1, y1))
        )

        for ax in ax_list:
            ax.set_xlim(n_arr[0], n_arr[-1])
            ax.grid()
            ax.label_outer()

        fig.savefig(
            figdir
            + "Gyro_{}_x{}_{}_y{}_{}_t0{}.png".format(
                runid,
                x0,
                x1,
                y0,
                y1,
                t0,
            ),
            dpi=300,
        )
        plt.close(fig)

    if fourier:
        fourier_var = data_arr[fourier - 1]
        fourier_var -= np.mean(fourier_var)
        r_arr = n_arr * dr
        N = r_arr.size
        T = dr
        yf = fft(fourier_var)
        xf = fftfreq(N, T)[: N // 2]
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        lin0 = ax[0].plot(xf, 2.0 / N * np.abs(yf[0 : N // 2]))
        ax[0].grid()
        ax[0].set_xlabel("k [1/RE]")
        xf0, yf0 = lin0[0].get_data()
        ax[0].set_title("k(max) = {}".format(xf0[yf0 == np.max(yf0)]))
        lin1 = ax[1].plot(1 / (xf[1:]), 2.0 / N * np.abs(yf[1 : N // 2]))
        ax[1].grid()
        ax[1].set_xlabel("$\\lambda$ [RE]")
        xf1, yf1 = lin1[0].get_data()
        ax[1].set_title("$\\lambda$(max) = {}".format(xf1[yf1 == np.max(yf1)]))
        fig.savefig(
            figdir
            + "{}_x{}_{}_y{}_{}_t0{}_fft_{}.png".format(
                runid, x0, x1, y0, y1, t0, fourier - 1
            ),
            dpi=300,
        )
        plt.close(fig)


def pos_mag_tension(vlsvobj, x, y, dx=300e3):
    """
    Calculate the magnetic tension force (B·∇)B/μ0 at a given point using finite differences.

    Parameters
    ----------
    vlsvobj : vlsvfile.VlsvReader object
        VLSV file object to read the magnetic field data from
    x : float
        X-coordinate in meters of the point where to calculate tension
    y : float
        Y-coordinate in meters of the point where to calculate tension
    dx : float, optional
        Step size in meters for finite difference calculation, default 300 km

    Returns
    -------
    ndarray
        3D vector of magnetic tension force components [x,y,z] in units of N/m^3

    Notes
    -----
    Uses central finite differences to calculate the gradient tensor. Only
    calculates gradients in x and y directions since simulation is 2D.
    """

    # Calculate B gradients in x and y directions using central differences
    dBxdx = (
        vlsvobj.read_interpolated_variable("vg_b_vol", [x + dx, y, 0], operator="x")
        - vlsvobj.read_interpolated_variable("vg_b_vol", [x - dx, y, 0], operator="x")
    ) / (2.0 * dx)
    dBydx = (
        vlsvobj.read_interpolated_variable("vg_b_vol", [x + dx, y, 0], operator="y")
        - vlsvobj.read_interpolated_variable("vg_b_vol", [x - dx, y, 0], operator="y")
    ) / (2.0 * dx)
    dBzdx = (
        vlsvobj.read_interpolated_variable("vg_b_vol", [x + dx, y, 0], operator="z")
        - vlsvobj.read_interpolated_variable("vg_b_vol", [x - dx, y, 0], operator="z")
    ) / (2.0 * dx)

    dBxdy = (
        vlsvobj.read_interpolated_variable("vg_b_vol", [x, y + dx, 0], operator="x")
        - vlsvobj.read_interpolated_variable("vg_b_vol", [x, y - dx, 0], operator="x")
    ) / (2.0 * dx)
    dBydy = (
        vlsvobj.read_interpolated_variable("vg_b_vol", [x, y + dx, 0], operator="y")
        - vlsvobj.read_interpolated_variable("vg_b_vol", [x, y - dx, 0], operator="y")
    ) / (2.0 * dx)
    dBzdy = (
        vlsvobj.read_interpolated_variable("vg_b_vol", [x, y + dx, 0], operator="z")
        - vlsvobj.read_interpolated_variable("vg_b_vol", [x, y - dx, 0], operator="z")
    ) / (2.0 * dx)

    # Construct the gradient tensor matrix (z-derivatives are 0 in 2D)
    B_jacobian = np.array([[dBxdx, dBxdy, 0], [dBydx, dBydy, 0], [0, 0, 0]]).T

    # Get magnetic field vector at the point
    B = vlsvobj.read_interpolated_variable("vg_b_vol", [x, y, 0])

    # Calculate B·∇B by matrix multiplication
    BdotJacobian = B @ B_jacobian

    # Return magnetic tension force divided by μ0
    return BdotJacobian / mu0


def pos_pressure_gradient(vlsvobj, x, y, dx=300e3):

    gradx = (
        vlsvobj.read_interpolated_variable("proton/vg_pressure", [x + dx, y, 0])
        - vlsvobj.read_interpolated_variable("proton/vg_pressure", [x - dx, y, 0])
    ) / (2.0 * dx)
    grady = (
        vlsvobj.read_interpolated_variable("proton/vg_pressure", [x, y + dx, 0])
        - vlsvobj.read_interpolated_variable("proton/vg_pressure", [x, y - dx, 0])
    ) / (2.0 * dx)

    return np.array([gradx, grady, 0])


def pos_mag_gradient(vlsvobj, x, y, dx=300e3):

    gradx = (
        vlsvobj.read_interpolated_variable(
            "vg_b_vol", [x + dx, y, 0], operator="magnitude"
        )
        * vlsvobj.read_interpolated_variable(
            "vg_b_vol", [x + dx, y, 0], operator="magnitude"
        )
        - vlsvobj.read_interpolated_variable(
            "vg_b_vol", [x - dx, y, 0], operator="magnitude"
        )
        * vlsvobj.read_interpolated_variable(
            "vg_b_vol", [x - dx, y, 0], operator="magnitude"
        )
    ) / (2 * mu0 * 2.0 * dx)
    grady = (
        vlsvobj.read_interpolated_variable(
            "vg_b_vol", [x, y + dx, 0], operator="magnitude"
        )
        * vlsvobj.read_interpolated_variable(
            "vg_b_vol", [x, y + dx, 0], operator="magnitude"
        )
        - vlsvobj.read_interpolated_variable(
            "vg_b_vol", [x, y - dx, 0], operator="magnitude"
        )
        * vlsvobj.read_interpolated_variable(
            "vg_b_vol", [x, y - dx, 0], operator="magnitude"
        )
    ) / (2 * mu0 * 2.0 * dx)

    return np.array([gradx, grady, 0])


def process_timestep_VSC_timeseries(args):
    """Helper function for parallel processing in VSC_timeseries"""
    (
        fnr,
        var_list,
        scales,
        bulkpath,
        tavgdir,
        cellid,
        ops,
        x0,
        y0,
        runid,
        pdavg,
    ) = args
    try:
        result = np.zeros(len(var_list) + 7, dtype=float)
        pdavg_result = None
        if pdavg:
            try:
                tavg_pdyn = np.loadtxt(
                    tavgdir + "/" + runid + "/" + str(fnr) + "_pdyn.tavg"
                )[int(cellid) - 1]
            except:
                tavg_pdyn = np.nan
            pdavg_result = tavg_pdyn * scales[5]
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
        return fnr, result, pdavg_result
    except Exception as e:
        print(f"Error processing timestep {fnr}: {str(e)}")
        return fnr, None, None


def mini_VSC(x0, y0, t0, t1):

    bulkpath = find_bulkpath("AIC")

    var_list = [
        "proton/vg_rho",
        "proton/vg_pdyn",
        "proton/vg_v",
        "proton/vg_v",
        "vg_vs",
        "vg_va",
        "vg_vms",
    ]
    scales = [1e-6, 1e9, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
    ops = ["pass", "pass", "magnitude", "y", "pass", "pass", "pass"]
    labs = ["$\\rho$", "$P_{dyn}$", "$v$", "$v_y$", "$v_s$", "$v_A$", "$v_{MS}$"]

    fig, ax = plt.subplots(len(var_list), 1, figsize=(12, 12), constrained_layout=True)
    fnr_arr = np.arange(t0 * 2, t1 * 2 + 1, dtype=int)
    t_arr = np.arange(t0, t1 + 0.1, 0.5)
    data_arr = np.zeros((len(var_list), fnr_arr.size), dtype=float)
    for idx in range(fnr_arr.size):
        vlsvobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr_arr[idx]).zfill(7))
        )
        for idx2, var in enumerate(var_list):
            data_arr[idx2, idx] = (
                vlsvobj.read_interpolated_variable(
                    var, [x0 * r_e, y0 * r_e, 0], operator=ops[idx2]
                )
                * scales[idx2]
            )

    for idx in range(len(var_list)):
        ax[idx].plot(t_arr, data_arr[idx])
        ax[idx].set_ylabel(labs[idx])
        ax[idx].grid()
        ax[idx].set_xlim(t_arr[0], t_arr[-1])
    ax[-1].set_xlabel("Time [s]")

    fig.savefig(wrkdir_DNR + "Figs/mini_VSC_diag.png", dpi=300)
    plt.close(fig)


def VSC_timeseries(
    runid,
    x0,
    y0,
    t0,
    t1,
    pdavg=False,
    pdx=False,
    delta=None,
    vlines=[],
    fmt="-",
    dirprefix="",
    skip=False,
    fromtxt=False,
    jett0=0.0,
    n_processes=1,
    draw=True,
):
    bulkpath = find_bulkpath(runid)

    figdir = wrkdir_DNR + "Figs/timeseries/{}".format(dirprefix)
    txtdir = wrkdir_DNR + "txts/timeseries/{}".format(dirprefix)
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
    if skip and os.path.isfile(
        figdir
        + "{}_x{:.3f}_y{:.3f}_t0{}_t1{}_delta{}.png".format(
            runid, x0, y0, t0, t1, delta
        )
    ):
        print("Skip is True and file already exists, exiting.")
        return None

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
        "$P_\\mathrm{dyn}$",
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
        "$\\rho~[\\mathrm{cm}^{-3}]$",
        "$v~[\\mathrm{km/s}]$",
        "$P_\\mathrm{dyn}~[\\mathrm{nPa}]$",
        "$B~[\\mathrm{nT}]$",
        "$E~[\\mathrm{mV/m}]$",
        "$T~[\\mathrm{MK}]$",
    ]
    if delta:
        for idx in range(len(ylabels)):
            ylabels[idx] = "$\\delta " + ylabels[idx][1:]
    e_sw = 750e3 * 3e-9 * q_p / m_p * 1e3
    pdsw_npa = m_p * 1e6 * 750e3 * 750e3 / 1e-9
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

    t_arr = np.arange(t0, t1 + 0.1, 0.5)
    fnr0 = int(t0 * 2)
    fnr_arr = np.arange(fnr0, int(t1 * 2) + 1, dtype=int)
    cellid = pt.vlsvfile.VlsvReader(
        bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
    ).get_cellid([x0 * r_e, y0 * r_e, 0 * r_e])
    data_arr = np.zeros((len(var_list) + 7, fnr_arr.size), dtype=float)
    tavg_arr = np.zeros(fnr_arr.size, dtype=float)

    if fromtxt:
        data_arr = np.loadtxt(
            txtdir
            + "{}_x{:.3f}_y{:.3f}_t0{}_t1{}_delta{}.txt".format(
                runid, x0, y0, t0, t1, delta
            )
        )
        tavg_arr = data_arr[-1, :]
    else:
        # Prepare arguments for parallel processing
        args_list = [
            (
                fnr,
                var_list,
                scales,
                bulkpath,
                tavgdir,
                cellid,
                ops,
                x0,
                y0,
                runid,
                pdavg,
            )
            for fnr in fnr_arr
        ]

        # Use multiprocessing Pool

        with Pool(processes=n_processes) as pool:
            results = pool.map(process_timestep_VSC_timeseries, args_list)

            # Process results
            for fnr, result, pdavg_result in results:
                if result is not None:
                    idx = np.where(fnr_arr == fnr)[0][0]
                    data_arr[:, idx] = result
                    if pdavg and pdavg_result is not None:
                        tavg_arr[idx] = pdavg_result

        # for idx, fnr in enumerate(fnr_arr):
        #     if pdavg:
        #         try:
        #             tavg_pdyn = np.loadtxt(
        #                 tavgdir + "/" + runid + "/" + str(fnr) + "_pdyn.tavg"
        #             )[int(cellid) - 1]
        #         except:
        #             tavg_pdyn = np.nan
        #         tavg_arr[idx] = tavg_pdyn * scales[5]
        #     vlsvobj = pt.vlsvfile.VlsvReader(
        #         bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
        #     )
        #     for idx2, var in enumerate(var_list):
        #         data_arr[idx2, idx] = (
        #             vlsvobj.read_interpolated_variable(
        #                 var, [x0 * r_e, y0 * r_e, 0], operator=ops[idx2]
        #             )
        #             * scales[idx2]
        #         )

    if draw:
        fig, ax_list = plt.subplots(
            len(ylabels) + 1, 1, sharex=True, figsize=(7, 9), constrained_layout=True
        )
        ax_list[0].set_title(
            "Run: {}, $x_0$: {:.3f}, $y_0$: {:.3f}".format(runid, x0, y0)
        )
        for idx in range(len(var_list)):
            ax = ax_list[plot_index[idx]]
            for vline in vlines:
                ax.axvline(vline, linestyle="dashed", linewidth=0.6)
            if delta:
                ax.plot(
                    t_arr,
                    data_arr[idx] - uniform_filter1d(data_arr[idx], size=delta),
                    fmt,
                    color=plot_colors[idx],
                    label=plot_labels[idx],
                )
            else:
                ax.plot(
                    t_arr,
                    data_arr[idx],
                    fmt,
                    color=plot_colors[idx],
                    label=plot_labels[idx],
                )
            if idx == 5 and pdavg and not delta:
                ax.plot(
                    t_arr,
                    2 * tavg_arr,
                    color=CB_color_cycle[1],
                    linestyle="dashed",
                    label="$2\\langle P_\\mathrm{dyn}\\rangle$",
                )
                ax.axhline(
                    0.5 * pdsw_npa,
                    color=CB_color_cycle[2],
                    linestyle="dotted",
                    label="$0.5P_\\mathrm{dyn,sw}$",
                )
                ax.axhline(
                    0.25 * pdsw_npa,
                    color=CB_color_cycle[3],
                    linestyle="dotted",
                    label="$0.25P_\\mathrm{dyn,sw}$",
                )
            if idx == 5 and pdx:
                pdynx = (
                    m_p
                    * data_arr[0]
                    * 1e6
                    * data_arr[1]
                    * 1e3
                    * data_arr[1]
                    * 1e3
                    * 1e9
                )
                if delta:
                    ax.plot(
                        t_arr,
                        pdynx - uniform_filter1d(pdynx, size=delta),
                        fmt,
                        color=CB_color_cycle[0],
                        label="$P_{\\mathrm{dyn},x}$",
                    )
                else:
                    ax.plot(
                        t_arr,
                        pdynx,
                        fmt,
                        color=CB_color_cycle[0],
                        label="$P_{\\mathrm{dyn},x}$",
                    )
            ax.set_xlim(t_arr[0], t_arr[-1])
            if draw_legend[idx]:
                ncols = 1
                if idx == 5:
                    ncols = 1
                ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncols=ncols)
        ylabels.append("$P_\\mathrm{dyn}$\ncontribution")

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

    if draw:
        ax_list[-1].plot(t_arr, rho_term, color="black", label="$\\rho$")
        ax_list[-1].plot(t_arr, vx_term, color=CB_color_cycle[0], label="$v_x^2$")
        ax_list[-1].plot(t_arr, vy_term, color=CB_color_cycle[1], label="$v_y^2$")
        ax_list[-1].plot(t_arr, vz_term, color=CB_color_cycle[2], label="$v_z^2$")

        ax_list[-1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), ncols=1)
        for vline in vlines:
            ax_list[-1].axvline(vline, linestyle="dashed", linewidth=0.6)
        ax_list[-1].set_xlim(t_arr[0], t_arr[-1])
        ax_list[-1].set_xlabel("Simulation time [s]")
        for idx, ax in enumerate(ax_list):
            ax.grid()
            ax.set_ylabel(ylabels[idx])
            ax.axvline(t0, linestyle="dashed")
            if pdavg:
                ax.fill_between(
                    t_arr,
                    0,
                    1,
                    where=data_arr[5, :] > 2 * tavg_arr,
                    color="red",
                    alpha=0.2,
                    transform=ax.get_xaxis_transform(),
                    linewidth=0,
                )

        fig.savefig(
            figdir
            + "{}_x{:.3f}_y{:.3f}_t0{}_t1{}_delta{}.png".format(
                runid, x0, y0, t0, t1, delta
            ),
            dpi=300,
        )
        plt.close(fig)
    np.savetxt(
        txtdir
        + "{}_x{:.3f}_y{:.3f}_t0{}_t1{}_delta{}.txt".format(
            runid, x0, y0, t0, t1, delta
        ),
        data_arr,
    )


def calc_cross_correlation(var1, var2):

    var1_standard = (var1 - np.nanmean(var1)) / (
        np.nanstd(var1, ddof=1) * (var1.size - 1)
    )
    var2_standard = (var2 - np.nanmean(var2)) / (np.nanstd(var2, ddof=1))

    return np.correlate(var1_standard, var2_standard, mode="valid")[0]


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
        "$P_\\mathrm{dyn}~[nPa]$",
        "$B_x~[nT]$",
        "$B_y~[nT]$",
        "$B_z~[nT]$",
        "$B~[nT]$",
        "$E_x~[mV/m]$",
        "$E_y~[mV/m]$",
        "$E_z~[mV/m]$",
        "$E~[mV/m]$",
        "$T_\\parallel~[MK]$",
        "$T_\\perp~[MK]$",
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
    vmsy = vms * -Bx / Bmag

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


def mini_jplots(
    x0,
    y0,
    t0,
    t1,
    xwidth=1,
    runid="AIC",
    bs_thresh=0.3,
    intpol=False,
    legsize=12,
    delta=None,
    folder_suffix="",
    skip=False,
):

    dr = 300e3 / r_e
    dr_km = 300

    figdir = wrkdir_DNR + "Figs/jmaps/{}".format(folder_suffix)
    if skip and os.path.isfile(
        figdir
        + "{}_x0_{}_y0_{}_t0_{}_t1_{}_delta_{}.png".format(runid, x0, y0, t0, t1, delta)
    ):
        print("Skip is true and file already exists, skipping.")
        return None

    varname_list = [
        "$v_x$ [km/s]",
        "$v_y$ [km/s]",
        "$v_z$ [km/s]",
        "$\\rho$ [cm$^{-3}$]",
        "$P_\\mathrm{dyn}$ [nPa]",
        "$B_x$ [nT]",
        "$B_y$ [nT]",
        "$B_z$ [nT]",
        "$T_\\perp$ [MK]",
        "$T_\\parallel$ [MK]",
    ]
    if delta:
        for idx in range(len(varname_list)):
            varname_list[idx] = "$\\delta " + varname_list[idx][1:]

    vars_list = [
        "proton/vg_v",
        "proton/vg_v",
        "proton/vg_v",
        "proton/vg_rho",
        "proton/vg_pdyn",
        "vg_b_vol",
        "vg_b_vol",
        "vg_b_vol",
        "proton/vg_t_perpendicular",
        "proton/vg_t_parallel",
        "proton/vg_beta_star",
        "proton/vg_t_thermal",
    ]
    ops_list = [
        "x",
        "y",
        "z",
        "pass",
        "pass",
        "x",
        "y",
        "z",
        "pass",
        "pass",
        "pass",
        "pass",
    ]
    scale_list = [
        1e-3,
        1e-3,
        1e-3,
        1e-6,
        1e9,
        1e9,
        1e9,
        1e9,
        1e-6,
        1e-6,
        1,
        1,
    ]
    # vmin = [1, -250, 0, 5, 0]
    # vmax = [5, 0, 0.3, 40, 4]
    vmin = [-750, -500, -250, 1, 0.5, -10, -10, -10, 5, 5]
    vmax = [750, 500, 250, 4, 2, 10, 10, 10, 20, 20]
    Tsw = 500e3
    # if delta:
    #     vmin = [-1, -100, -0.25, -7.5, -1]
    #     vmax = [1, 100, 0.25, 7.5, 1]

    bulkpath = find_bulkpath(runid)

    fnr0 = int(t0 * 2)
    fnr1 = int(t1 * 2)

    fnr_range = np.arange(fnr0, fnr1 + 1, 1, dtype=int)
    t_range = np.arange(t0, t1 + 0.1, 0.5)

    npoints = int((2 * xwidth) / dr) + 1

    xlist = np.linspace(x0 - xwidth, x0 + xwidth, npoints)

    if intpol:
        coords = [[xlist[idx] * r_e, y0 * r_e, 0] for idx in range(xlist.size)]

    fobj = pt.vlsvfile.VlsvReader(bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7)))

    cellids = [
        int(fobj.get_cellid([xlist[idx] * r_e, y0 * r_e, 0]))
        for idx in range(xlist.size)
    ]
    xplot_list = xlist
    xlab = "$X~[R_\\mathrm{E}]$"
    XmeshXY, YmeshXY = np.meshgrid(xlist, t_range)

    data_arr = np.zeros((len(vars_list), xplot_list.size, t_range.size), dtype=float)

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

    if delta:
        for idx in range(len(varname_list)):
            for idx2 in range(xplot_list.size):
                # data_arr[idx, idx2, :] = sosfilt(sos, data_arr[idx, idx2, :])
                data_arr[idx, idx2, :] = data_arr[idx, idx2, :] - uniform_filter1d(
                    data_arr[idx, idx2, :], size=delta
                )

    # data_arr = [rho_arr, v_arr, pdyn_arr, B_arr, T_arr]
    cmap = [
        "vik",
        "vik",
        "vik",
        "Blues_r",
        "Blues_r",
        "vik",
        "vik",
        "vik",
        "Blues_r",
        "Blues_r",
    ]
    if delta:
        cmap = [
            "vik",
            "vik",
            "vik",
            "vik",
            "vik",
            "vik",
            "vik",
            "vik",
            "vik",
            "vik",
        ]
    annot = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
    ]

    fig, ax_list = plt.subplots(
        2,
        5,
        figsize=(20, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    ax_list = ax_list.flatten()
    im_list = []
    cb_list = []
    fig.suptitle(
        "Run: {}, x0: {}, y0: {}".format(runid, x0, y0),
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
        cb_list.append(fig.colorbar(im_list[-1], ax=ax))
        cb_list[-1].ax.tick_params(labelsize=20)
        # ax.contour(
        #     XmeshXY, YmeshXY, data_arr[-2].T, [bs_thresh], colors=[CB_color_cycle[1]]
        # )
        ax.contour(
            XmeshXY, YmeshXY, data_arr[-1].T, [3 * Tsw], colors=[CB_color_cycle[1]]
        )
        # ax.plot([1, 2], [0, 1], color="k", label="$\\beta^*=$ {}".format(bs_thresh))

        ax.set_title(varname_list[idx], fontsize=24, pad=10)
        ax.set_xlim(xplot_list[0], xplot_list[-1])
        ax.set_ylim(t_range[0], t_range[-1])
        if idx in [5, 6, 7, 8, 9]:
            ax.set_xlabel(xlab, fontsize=24, labelpad=10)
        ax.annotate(
            annot[idx],
            (0.05, 0.90),
            xycoords="axes fraction",
            fontsize=24,
            bbox=dict(
                boxstyle="square,pad=0.15",
                fc="white",
                ec="k",
                lw=0.5,
            ),
        )
    ax_list[0].set_ylabel("Simulation time [s]", fontsize=28, labelpad=10)
    ax_list[5].set_ylabel("Simulation time [s]", fontsize=28, labelpad=10)
    # ax_list[0].legend(fontsize=legsize, loc="lower left", ncols=2)
    if not os.path.exists(figdir):
        try:
            os.makedirs(figdir)
        except OSError:
            pass

    fig.savefig(
        figdir
        + "{}_x0_{}_y0_{}_t0_{}_t1_{}_delta_{}.png".format(
            runid, x0, y0, t0, t1, delta
        ),
        dpi=300,
    )
    plt.close(fig)


def process_timestep_jplots(args):
    """Helper function for parallel processing in jplots"""
    (
        fnr,
        vars_list,
        ops_list,
        scale_list,
        intpol,
        coords,
        cellids,
        xlist,
        pdavg,
        runid,
        tavgdir,
        cellid_coords,
    ) = args
    try:
        result = np.zeros((len(vars_list), xlist.size), dtype=float)
        bulkpath = find_bulkpath(runid)
        vlsvobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
        )
        if pdavg:
            try:
                pdavg_arr = np.loadtxt(tavgdir + runid + "/" + str(fnr) + "_pdyn.tavg")[
                    cellids - 1
                ]
                if intpol:
                    if xlist[-1] != xlist[0]:
                        pdavg_arr_interp = (
                            np.interp(coords[:, 0], cellid_coords[:, 0], pdavg_arr)
                            * 1e9
                        )
                    else:
                        pdavg_arr_interp = (
                            np.interp(coords[:, 1], cellid_coords[:, 1], pdavg_arr)
                            * 1e9
                        )
                else:
                    pdavg_arr_interp = pdavg_arr * 1e9
            except:
                pdavg_arr_interp = np.ones(xlist.size) * np.nan
        else:
            pdavg_arr_interp = None

        for idx2 in range(len(vars_list)):
            if intpol:
                result[idx2, :] = [
                    vlsvobj.read_interpolated_variable(
                        vars_list[idx2], coords[idx3], operator=ops_list[idx2]
                    )
                    * scale_list[idx2]
                    for idx3 in range(xlist.size)
                ]
            else:
                result[idx2, :] = (
                    vlsvobj.read_variable(
                        vars_list[idx2], operator=ops_list[idx2], cellids=cellids
                    )
                    * scale_list[idx2]
                )
        return fnr, result, pdavg_arr_interp
    except Exception as e:
        print(f"Error processing timestep {fnr}: {str(e)}")
        return fnr, None, None


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
    vars_to_plot=[0, 1, 2, 3, 4],
    vels_to_plot=[0, 1, 2, 3, 4, 5, 6],
    legsize=12,
    filt=False,
    pdavg=False,
    n_processes=None,
):
    dr = 300e3 / r_e
    dr_km = 300
    varname_list = [
        "$\\rho$ [cm$^{-3}$]",
        "$v_x$ [km/s]",
        "$P_\\mathrm{dyn}$ [nPa]",
        "$B$ [nT]",
        # "$T$ [MK]",
        "$E$ [mV/m]",
    ]
    if filt:
        for idx in range(len(varname_list)):
            varname_list[idx] = "$\\delta " + varname_list[idx][1:]
    vars_list = [
        "proton/vg_rho",
        "proton/vg_v",
        "proton/vg_pdyn",
        "vg_b_vol",
        # "proton/vg_temperature",
        "vg_e_vol",
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
        # "pass",
        "magnitude",
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
        # 1e-6,
        1e3,
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
    runid_list = ["AGF", "AIA", "AIB", "AIC"]
    runids_paper = ["RDC", "RDC2", "RDC3", "RDC4"]
    sw_pars = [
        [1.0e6, 750.0e3, 3.0e-9, 0.5e6],
        [1.0e6, 750.0e3, 3.0e-9, 0.5e6],
        [1.0e6, 750.0e3, 3.0e-9, 0.5e6],
        [1.0e6, 750.0e3, 3.0e-9, 0.5e6],
    ]

    vmin = [1, -250, 0, 5, 0]
    vmax = [5, 0, 0.3, 40, 4]
    if filt:
        vmin = [-1, -100, -0.25, -7.5, -1]
        vmax = [1, 100, 0.25, 7.5, 1]

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

    cellids = np.array(
        [
            int(fobj.get_cellid([xlist[idx] * r_e, ylist[idx] * r_e, 0]))
            for idx in range(xlist.size)
        ]
    )

    if xlist[-1] != xlist[0]:
        xplot_list = xlist
        xlab = "$X~[R_\\mathrm{E}]$"
        XmeshXY, YmeshXY = np.meshgrid(xlist, t_range)
    else:
        xplot_list = ylist
        xlab = "$Y~[R_\\mathrm{E}]$"
        XmeshXY, YmeshXY = np.meshgrid(ylist, t_range)

    data_arr = np.zeros((len(vars_list), xplot_list.size, t_range.size), dtype=float)
    vt_arr = np.ones((xplot_list.size, t_range.size), dtype=float)
    dx_arr = (x1 - x0) * vt_arr
    dy_arr = (y1 - y0) * vt_arr

    figdir = wrkdir_DNR + "Figs/jmaps/"
    txtdir = wrkdir_DNR + "txts/jmaps/"

    cellid_coords = np.array([fobj.get_cell_coordinates(cellid) for cellid in cellids])
    pdavg_arr_interp = np.ones((xplot_list.size, t_range.size), dtype=float) * np.nan

    if txt:
        data_arr = np.load(
            txtdir
            + "{}_x0_{}_y0_{}_x1_{}_y1_{}_t0_{}_t1_{}.npy".format(
                runid, x0, y0, x1, y1, t0, t1
            )
        )
    else:
        # Initialize arrays
        data_arr = np.zeros(
            (len(vars_list), xplot_list.size, t_range.size), dtype=float
        )
        pdavg_arr_interp = (
            np.ones((xplot_list.size, t_range.size), dtype=float) * np.nan
        )

        # Prepare arguments for parallel processing
        args_list = [
            (
                fnr,
                vars_list,
                ops_list,
                scale_list,
                intpol,
                coords if intpol else None,
                cellids,
                xlist,
                pdavg,
                runid,
                tavgdir,
                cellid_coords,
            )
            for fnr in fnr_range
        ]

        # Use multiprocessing Pool

        with Pool(processes=n_processes) as pool:
            results = pool.map(process_timestep_jplots, args_list)

            # Process results
            for fnr, result, pdavg_result in results:
                if result is not None:
                    idx = np.where(fnr_range == fnr)[0][0]
                    data_arr[:, :, idx] = result
                    if pdavg and pdavg_result is not None:
                        pdavg_arr_interp[:, idx] = pdavg_result

    jetmask = (data_arr[2, :, :] > 2 * pdavg_arr_interp).astype(int)

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

    if filt:
        for idx in range(5):
            for idx2 in range(xplot_list.size):
                data_arr[idx, idx2, :] = data_arr[idx, idx2, :] - uniform_filter1d(
                    data_arr[idx, idx2, :], size=60
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

    cmap = ["Blues_r", "Blues_r", "Blues_r", "Blues_r", "Blues_r"]
    if filt:
        cmap = ["vik", "vik", "vik", "vik", "vik"]
    annot = ["a", "b", "c", "d", "e"]

    figh = 10
    if len(vels_to_plot) < 4:
        figh = 8
    if draw:
        fig, ax_list = plt.subplots(
            1,
            len(vars_to_plot),
            figsize=(6 * len(vars_to_plot), figh),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        if len(vars_to_plot) > 1:
            ax_list = ax_list.flatten()
        else:
            ax_list = [ax_list]
        im_list = []
        cb_list = []
        fig.suptitle(
            "Run: {}, x0: {}, y0: {}, x1: {}, y1: {}".format(runid, x0, y0, x1, y1),
            fontsize=28,
        )
        ax_idx = 0
        for idx in range(len(varname_list)):
            if idx not in vars_to_plot:
                continue
            ax = ax_list[ax_idx]
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
            cb_list.append(fig.colorbar(im_list[-1], ax=ax))
            cb_list[-1].ax.tick_params(labelsize=20)
            ax.contour(XmeshXY, YmeshXY, data_arr[5].T, [bs_thresh], colors=["k"])
            if pdavg:
                ax.contour(
                    XmeshXY,
                    YmeshXY,
                    jetmask.T,
                    [0.5],
                    colors=[CB_color_cycle[2]],
                    linestyles=["dotted"],
                    linewidths=[1.2],
                )
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
                    if itr not in vels_to_plot:
                        continue
                    vel_masked = vel.T
                    vt_masked = vt_arr.T

                    vel_masked[
                        np.logical_or(
                            XmeshXY < wavefan[0] - 2, XmeshXY > wavefan[0] + 2
                        )
                    ] = np.nan
                    vel_masked[
                        np.logical_or(YmeshXY < wavefan[1], YmeshXY > wavefan[1] + 40)
                    ] = np.nan

                    vt_masked[
                        np.logical_or(
                            XmeshXY < wavefan[0] - 2, XmeshXY > wavefan[0] + 2
                        )
                    ] = np.nan
                    vt_masked[
                        np.logical_or(YmeshXY < wavefan[1], YmeshXY > wavefan[1] + 40)
                    ] = np.nan

                    ax.streamplot(
                        XmeshXY,
                        YmeshXY,
                        vel_masked / 6371,
                        vt_masked,
                        # vel.T / 6371,
                        # vt_arr.T,
                        arrowstyle="-",
                        broken_streamlines=True,
                        color=CB_color_cycle[::-1][itr + 1],
                        linewidth=2,
                        # minlength=4,
                        maxlength=1,
                        integration_direction="forward",
                        density=35,
                        start_points=np.array([wavefan]),
                    )
                    ax.plot(
                        [1, 2],
                        [0, 1],
                        color=CB_color_cycle[::-1][itr + 1],
                        label=vels_labels[itr],
                    )
            ax.set_title(varname_list[idx], fontsize=24, pad=10)
            ax.set_xlim(xplot_list[0], xplot_list[-1])
            ax.set_ylim(t_range[0], t_range[-1])
            ax.set_xlabel(xlab, fontsize=24, labelpad=10)
            ax.annotate(
                annot[idx],
                (0.05, 0.90),
                xycoords="axes fraction",
                fontsize=24,
                bbox=dict(
                    boxstyle="square,pad=0.15",
                    fc="white",
                    ec="k",
                    lw=0.5,
                ),
            )
            ax_idx += 1
        ax_list[0].set_ylabel("Simulation time [s]", fontsize=28, labelpad=10)
        if len(vels_to_plot) >= 4:
            ax_list[0].legend(
                fontsize=legsize,
                bbox_to_anchor=(0.5, -0.12),
                loc="upper center",
                ncols=2,
            )
        else:
            ax_list[0].legend(fontsize=legsize, loc="lower left", ncols=2)
        if not os.path.exists(figdir):
            try:
                os.makedirs(figdir)
            except OSError:
                pass

        fig.savefig(
            figdir
            + "{}_x0_{}_y0_{}_x1_{}_y1_{}_t0_{}_t1_{}_filt_{}.png".format(
                runid, x0, y0, x1, y1, t0, t1, filt
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
            + "{}_x0_{}_y0_{}_x1_{}_y1_{}_t0_{}_t1_{}_filt_{}.npy".format(
                runid, x0, y0, x1, y1, t0, t1, filt
            ),
            data_arr,
        )


def get_contour_cells(
    vlsvobj, boxre, thresh=1, var="proton/vg_mmsx", op="pass", lt=True
):

    restricted_cells = restrict_area(vlsvobj, boxre)
    coords = np.array([vlsvobj.get_cell_coordinates(cell) for cell in restricted_cells])
    vals = vlsvobj.read_variable(var, operator=op, cellids=restricted_cells)

    y_unique = np.unique(coords[:, 1])
    xlist = []
    cell_list = []
    for yun in y_unique:
        x_at_y = coords[coords[:, 1] == yun, 0]
        val_at_y = vals[coords[:, 1] == yun]
        if lt:
            x_at_y_thresh = x_at_y[val_at_y <= thresh]
        else:
            x_at_y_thresh = x_at_y[val_at_y >= thresh]
        xright = np.max(x_at_y_thresh)
        xlist.append(xright)
        cell_list.append(
            restricted_cells[(coords[:, 0] == xright) & (coords[:, 1] == yun)]
        )

    xlist = np.array(xlist)
    cell_list = np.array(cell_list).flatten()

    return (cell_list, xlist, y_unique)


def process_fourier_timestep(args):
    """Helper function for parallel processing in contour_fourier_timeseries"""
    t, runid, boxre, filt, var, op, lt, thresh = args
    try:
        return t, plot_vars_on_contour(
            runid, t, boxre, filt=filt, draw=False, var=var, op=op, lt=lt, thresh=thresh
        )
    except Exception as e:
        print(f"Error processing timestep {t}: {str(e)}")
        return t, None


def contour_fourier_timeseries(
    runid,
    t0,
    t1,
    boxre,
    filt=10,
    n_processes=None,
    var="proton/vg_mmsx",
    op="pass",
    lt=True,
    thresh=0.3,
):

    figdir = wrkdir_DNR + "Figs/plots_on_cont/"
    if not os.path.exists(figdir):
        try:
            os.makedirs(figdir)
        except OSError:
            pass

    t_range = np.arange(t0, t1 + 0.1, 0.5)

    # Get sizes from first timestep
    lbd1, xfft, rhofft, vxfft = plot_vars_on_contour(
        runid, t0, boxre, filt=filt, draw=False
    )

    data_arr = np.zeros((3, lbd1.size, t_range.size), dtype=float)

    # Prepare arguments for parallel processing
    args_list = [(t, runid, boxre, filt, var, op, lt, thresh) for t in t_range]

    # Use multiprocessing Pool
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_fourier_timestep, args_list)

        # Process results
        for t, result in results:
            if result is not None:
                lbd1, xfft, rhofft, vxfft = result
                idx = np.where(t_range == t)[0][0]
                data_arr[0, :, idx] = xfft
                data_arr[1, :, idx] = rhofft
                data_arr[2, :, idx] = vxfft

    fig, ax_list = plt.subplots(1, 3, figsize=(16, 9))

    title_list = [r"$\delta X$", r"$\delta\rho$", r"$\delta v_x$"]

    for idx in range(ax_list.size):
        ax = ax_list[idx]
        ax.pcolormesh(lbd1, t_range, data_arr[idx].T, shading="nearest", cmap="batlow")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda$ [RE]")
        ax.set_ylabel("Time [s]")
        ax.set_title(title_list[idx])

    fig.savefig(
        figdir + "t0_{}_t1_{}_bs_contour_filt{}_fft_jplot.png".format(t0, t1, filt),
        dpi=300,
    )
    plt.close(fig)


def plot_vars_on_contour(
    runid,
    t0,
    boxre,
    filt=None,
    draw=True,
    var="proton/vg_mmsx",
    op="pass",
    lt=True,
    thresh=1,
):

    bulkpath = find_bulkpath(runid)
    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath + "bulk.{}.vlsv".format(str(int(t0 * 2)).zfill(7))
    )
    if filt:
        ylabels = [
            r"$\delta X~[R_\mathrm{E}]$",
            r"$\delta\rho~[\mathrm{cm}^{-3}]$",
            r"$\delta v_x~[\mathrm{km/s}]$",
        ]
    else:
        ylabels = [
            r"$X~[R_\mathrm{E}]$",
            r"$\rho~[\mathrm{cm}^{-3}]$",
            r"$v_x~[\mathrm{km/s}]$",
        ]
    figdir = wrkdir_DNR + "Figs/plots_on_cont/"
    if not os.path.exists(figdir):
        try:
            os.makedirs(figdir)
        except OSError:
            pass

    cont_cells, cont_x, cont_y = get_contour_cells(
        vlsvobj, boxre, thresh=thresh, var=var, op=op, lt=lt
    )
    cont_x /= r_e
    cont_y /= r_e

    rho = (
        vlsvobj.read_variable("proton/vg_rho", operator="pass", cellids=cont_cells)
        / 1e6
    )
    vx = vlsvobj.read_variable("proton/vg_v", operator="x", cellids=cont_cells) / 1e3

    if filt:
        cont_x = cont_x - uniform_filter1d(cont_x, size=filt)
        rho = rho - uniform_filter1d(rho, size=filt)
        vx = vx - uniform_filter1d(vx, size=filt)
    if draw:
        fig, ax_list = plt.subplots(3, 1, figsize=(8, 9))

        for ax in ax_list:
            ax.set_xlim(cont_y[0], cont_y[-1])
            ax.grid()

        ax_list[0].plot(cont_y, cont_x)
        ax_list[0].set_ylabel(ylabels[0])
        ax_list[0].set_title("t0 = {}".format(t0))

        ax_list[1].plot(cont_y, rho)
        ax_list[1].set_ylabel(ylabels[1])

        ax_list[2].plot(cont_y, vx)
        ax_list[2].set_ylabel(ylabels[2])
        ax_list[2].set_xlabel(r"$Y~[R_\mathrm{E}]$")

        fig.savefig(figdir + "t0_{}_bs_contour_filt{}.png".format(t0, filt), dpi=300)
        plt.close(fig)

    if filt:
        N = cont_y.size
        T = cont_y[1] - cont_y[0]
        yf1 = fft(cont_x)
        yf2 = fft(rho)
        yf3 = fft(vx)
        xf = fftfreq(N, T)[: N // 2]
        if draw:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
            ax[0].grid()
            ax[0].plot(1 / (xf[1:]), 2.0 / N * np.abs(yf1[1 : N // 2]))
            ax[0].set_xlabel("$\\lambda$ [RE]")
            xf1, yf1 = ax[0].get_lines()[0].get_data()
            ax[0].set_title("X: $\\lambda$(max) = {}".format(xf1[yf1 == np.max(yf1)]))

            ax[1].grid()
            ax[1].plot(1 / (xf[1:]), 2.0 / N * np.abs(yf2[1 : N // 2]))
            ax[1].set_xlabel("$\\lambda$ [RE]")
            xf2, yf2 = ax[1].get_lines()[0].get_data()
            ax[1].set_title("rho: $\\lambda$(max) = {}".format(xf2[yf2 == np.max(yf2)]))

            ax[2].grid()
            ax[2].plot(1 / (xf[1:]), 2.0 / N * np.abs(yf3[1 : N // 2]))
            ax[2].set_xlabel("$\\lambda$ [RE]")
            xf3, yf3 = ax[2].get_lines()[0].get_data()
            ax[2].set_title("vx: $\\lambda$(max) = {}".format(xf3[yf3 == np.max(yf3)]))

            for a in ax:
                a.set_xlim(2 * T / r_e, 2)
                a.set_xscale("log")

            fig.savefig(
                figdir + "t0_{}_bs_contour_filt{}_fft.png".format(t0, filt), dpi=300
            )
            plt.close(fig)

        return (
            1 / (xf[1:]),
            2.0 / N * np.abs(yf1[1 : N // 2]),
            2.0 / N * np.abs(yf2[1 : N // 2]),
            2.0 / N * np.abs(yf3[1 : N // 2]),
        )


def make_vg_b_jacobian(vobj):

    B = vobj.read_variable("vg_b_vol")
    ci = vobj.read_variable("CellID")
    B_sorted = B[np.argsort(ci)]

    meshshape = vobj.get_spatial_mesh_size()[:-1]

    Bx, By, Bz = B_sorted.T

    Bx_reshaped = np.reshape(Bx, meshshape)
    By_reshaped = np.reshape(By, meshshape)
    Bz_reshaped = np.reshape(Bz, meshshape)

    dx = vobj.get_fsgrid_cell_size()[:-1]

    dFx_dx, dFx_dy = np.gradient(Bx_reshaped[:, :], *dx)
    dFy_dx, dFy_dy = np.gradient(By_reshaped[:, :], *dx)
    dFz_dx, dFz_dy = np.gradient(Bz_reshaped[:, :], *dx)

    return np.rollaxis(
        np.array(
            [
                # [dFx_dx.flatten(), dFx_dy.flatten(), np.zeros_like(dFx_dx).flatten()],
                # [dFy_dx.flatten(), dFy_dy.flatten(), np.zeros_like(dFx_dx).flatten()],
                # [dFz_dx.flatten(), dFz_dy.flatten(), np.zeros_like(dFx_dx).flatten()],
                # [dFx_dx.flatten(), dFy_dx.flatten(), dFz_dx.flatten()],
                # [dFx_dy.flatten(), dFy_dy.flatten(), dFz_dy.flatten()],
                # [
                #     np.zeros_like(dFx_dx).flatten(),
                #     np.zeros_like(dFx_dx).flatten(),
                #     np.zeros_like(dFx_dx).flatten(),
                # ],
                [dFx_dx.flatten(), dFy_dx.flatten(), np.zeros_like(dFx_dx).flatten()],
                [dFx_dy.flatten(), dFy_dy.flatten(), np.zeros_like(dFx_dx).flatten()],
                [
                    np.zeros_like(dFx_dx).flatten(),
                    np.zeros_like(dFx_dx).flatten(),
                    np.zeros_like(dFx_dx).flatten(),
                ],
            ]
        ),
        2,
        0,
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


def vspace_reducer(
    vlsvobj,
    cellid,
    operator,
    dv=31e3,
    vmin=None,
    vmax=None,
    b=None,
    binw=31e3,
    fmin=1e-15,
):
    """
    Function for reducing a 3D VDF to 1D
    (object) vlsvobj = Analysator VLSV file object
    (int) cellid = ID of cell whose VDF you want
    (str) operator = "x", "y", or "z", which velocity component to retain after reduction, or "magnitude" to get the distribution of speeds (untested)
    (float) dv = Velocity space resolution in m/s
    """

    # List of valid operators from which to get an index
    op_list = ["x", "y", "z"]

    # Read velocity cell keys and values from vlsv file
    velcels = vlsvobj.read_velocity_cells(cellid)
    vc_coords = vlsvobj.get_velocity_cell_coordinates(list(velcels.keys()))
    vc_vals = np.array(list(velcels.values()))

    ii_fm = np.where(vc_vals >= fmin)
    vc_vals = vc_vals[ii_fm]
    vc_coords = vc_coords[ii_fm, :][0, :, :]

    # Select coordinates of chosen velocity component
    if operator in op_list:
        vc_coord_arr = vc_coords[:, op_list.index(operator)]
    elif operator == "magnitude":
        vc_coord_arr = np.sqrt(
            vc_coords[:, 0] ** 2 + vc_coords[:, 1] ** 2 + vc_coords[:, 2] ** 2
        )
    elif operator == "par":
        # print("par")
        vc_coord_arr = np.dot(vc_coords, b)
    elif operator == "perp":
        # print("perp")
        xvec = np.array([1, 0, 0])
        bxx = np.cross(b, xvec)
        bxbxx = np.cross(b, bxx)
        vc_coord_perp1 = np.dot(vc_coords, bxx)
        vc_coord_perp2 = np.dot(vc_coords, bxbxx)
        vc_coord_arr = np.sqrt(vc_coord_perp1**2 + vc_coord_perp2**2)
        # vc_coord_arr = np.sqrt(
        #     vc_coords[:, 0] ** 2
        #     + vc_coords[:, 1] ** 2
        #     + vc_coords[:, 2] ** 2
        #     - np.dot(vc_coords, b) ** 2
        # )
    elif operator == "cosmu":
        # print("cosmu")
        vc_coord_arr = np.dot(vc_coords, b) / (
            np.sqrt(vc_coords[:, 0] ** 2 + vc_coords[:, 1] ** 2 + vc_coords[:, 2] ** 2)
            + 1e-27
        )

    # Create histogram bins, one for each unique coordinate of the chosen velocity component
    # if bool(vmin or vmax):
    #     vbins = np.arange(vmin, vmax, dv)
    # else:
    # vbins = np.sort(np.unique(vc_coord_arr))
    # vbins = np.append(vbins - dv / 2, vbins[-1] + dv / 2)
    # if operator == "cosmu":
    #     vbins = np.sort(np.unique(vc_coord_arr))
    #     dcosmu = np.max(np.ediff1d(vbins))
    #     vbins = np.arange(-1, 1 + dcosmu / 2, dcosmu)
    vbins = np.sort(np.unique(vc_coord_arr))
    # dbins = np.max(np.ediff1d(vbins))
    # print("dbins = {}".format(dbins))
    vbins = np.arange(
        np.min(vbins) - binw / 2, np.max(vbins) + binw / 2 + binw / 4, binw
    )

    # if rotatetob or operator in ["par", "perp", "cosmu", "magnitude"]:
    #     cellsperbin = np.ones(vbins.size - 1, dtype=int)
    #     for idx in range(cellsperbin.size):
    #         # print("idx = {}".format(idx))
    #         cellsperbin[idx] = np.logical_and(
    #             vc_coord_arr > vbins[idx], vc_coord_arr <= vbins[idx + 1]
    #         ).sum()
    #     for idx in range(cellsperbin.size):
    #         # print("idx = {}".format(idx))
    #         vc_vals[
    #             np.logical_and(
    #                 vc_coord_arr > vbins[idx], vc_coord_arr <= vbins[idx + 1]
    #             )
    #         ] /= cellsperbin[idx] / np.mean(cellsperbin)

    # vbins = np.append(vbins - dbins / 2, vbins[-1] + dbins / 2)
    # if operator == "magnitude":
    #     vbins = vbins * 4

    # Create weights, <3D VDF value>*<vspace cell side area>, so that the histogram binning essentially performs an integration
    # if operator in op_list:
    #     vweights = vc_vals * dv * dv
    # elif operator == "magnitude":
    #     vweights = vc_vals * 4 * np.pi * vc_coord_arr**2
    vweights = vc_vals * dv * dv

    # Integrate over the perpendicular directions
    # if operator in ["magnitude", "par", "perp"]:
    #     # vbins = np.histogram_bin_edges(vc_coord_arr, bins="auto", weights=vweights)
    #     vbins = vbins * np.sqrt(3)
    #     hist, bin_edges = np.histogram(vc_coord_arr, bins=vbins, weights=vweights)
    # else:
    #     hist, bin_edges = np.histogram(vc_coord_arr, bins=vbins, weights=vweights)
    hist, bin_edges = np.histogram(vc_coord_arr, bins=vbins, weights=vweights)

    # Return the 1D VDF values in units of s/m^4 as well as the bin edges to assist in plotting
    return (hist, bin_edges)


def pos_vdf_1d_spectrogram(
    runid,
    x,
    y,
    t0,
    t1,
    vmin,
    vmax,
    dv=31e3,
    overplot_v=False,
    parperp=False,
    logcb=False,
    rotatetob=False,
    filtsize=None,
    clevels=10,
):
    runids = ["AGF", "AIA", "AIB", "AIC"]
    bulkpath = find_bulkpath(runid)

    if parperp:
        dv = [0.02, dv, dv]
        vmin = [-1, vmin, 0]
        vmax = [1, vmax, vmax]
        scales = [1, 1e-3, 1e-3]
    else:
        dv = [dv, dv, dv]
        vmin = [vmin, vmin, vmin]
        vmax = [vmax, vmax, vmax]
        scales = [1e-3, 1e-3, 1e-3]

    global xg, yg

    xg = []
    yg = []

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g, x0, y0, drawBy0, plaschke_g
    runid_g = runid
    Blines_g = False
    drawBy0 = False
    plaschke_g = False

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
        [1e6, 750e3, 3e-9, 0.5e6],
        [1e6, 750e3, 3e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runids.index(runid)]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw

    v_arr = [np.arange(vmin[idx], vmax[idx], dv[idx]) for idx in range(3)]
    t_arr = np.arange(t0, t1 + 0.1, 0.5)

    vx_arr = np.zeros((v_arr[0].size, t_arr.size), dtype=float)
    vy_arr = np.zeros((v_arr[1].size, t_arr.size), dtype=float)
    vz_arr = np.zeros((v_arr[2].size, t_arr.size), dtype=float)

    b_arr = np.zeros((t_arr.size, 3), dtype=float)

    if overplot_v:
        vmean_arr = np.zeros((3, t_arr.size), dtype=float)

    fig, ax_list = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)

    for idx, t in enumerate(np.arange(t0, t1 + 0.1, 0.5)):

        fnr = int(t * 2)
        filenr_g = fnr
        vobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
        )

        cellid = vobj.get_cellid([x * r_e, y * r_e, 0 * r_e])
        vdf_cellid = getNearestCellWithVspace(vobj, cellid)

        b = vobj.read_variable("vg_b_vol", cellids=vdf_cellid)
        b = b / np.linalg.norm(b)
        b_arr[idx] = b

    if filtsize:
        b_arr = uniform_filter1d(b_arr, size=filtsize)

    for idx, t in enumerate(np.arange(t0, t1 + 0.1, 0.5)):
        fnr = int(t * 2)
        filenr_g = fnr
        vobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
        )
        cellid = vobj.get_cellid([x * r_e, y * r_e, 0 * r_e])
        vdf_cellid = getNearestCellWithVspace(vobj, cellid)

        # b = vobj.read_variable("vg_b_vol", cellids=vdf_cellid)
        # b = b / np.linalg.norm(b)

        x_re, y_re, z_re = vobj.get_cell_coordinates(vdf_cellid) / r_e
        if parperp:
            xhist, xbin_edges = vspace_reducer(
                vobj, vdf_cellid, operator="cosmu", b=b_arr[idx], binw=dv[0]
            )
            yhist, ybin_edges = vspace_reducer(
                vobj, vdf_cellid, operator="par", b=b_arr[idx], binw=dv[1]
            )
            zhist, zbin_edges = vspace_reducer(
                vobj, vdf_cellid, operator="perp", b=b_arr[idx], binw=dv[2]
            )
        else:
            xhist, xbin_edges = vspace_reducer(
                vobj,
                vdf_cellid,
                operator="x",
                b=b_arr[idx],
                rotatetob=rotatetob,
                binw=dv[0],
            )
            yhist, ybin_edges = vspace_reducer(
                vobj,
                vdf_cellid,
                operator="y",
                b=b_arr[idx],
                rotatetob=rotatetob,
                binw=dv[1],
            )
            zhist, zbin_edges = vspace_reducer(
                vobj,
                vdf_cellid,
                operator="z",
                b=b_arr[idx],
                rotatetob=rotatetob,
                binw=dv[2],
            )
        xbin_centers = xbin_edges[:-1] + 0.5 * (xbin_edges[1] - xbin_edges[0])
        ybin_centers = ybin_edges[:-1] + 0.5 * (ybin_edges[1] - ybin_edges[0])
        zbin_centers = zbin_edges[:-1] + 0.5 * (zbin_edges[1] - zbin_edges[0])

        x0 = x_re
        y0 = y_re

        if overplot_v and not parperp:
            vmean_arr[:, idx] = (
                vobj.read_variable("proton/vg_v", cellids=vdf_cellid) * 1e-3
            )

        xhist_interp = np.interp(
            v_arr[0], xbin_centers, xhist, left=np.nan, right=np.nan
        )
        yhist_interp = np.interp(
            v_arr[1], ybin_centers, yhist, left=np.nan, right=np.nan
        )
        zhist_interp = np.interp(
            v_arr[2], zbin_centers, zhist, left=np.nan, right=np.nan
        )

        vx_arr[:, idx] = xhist_interp
        vy_arr[:, idx] = yhist_interp
        vz_arr[:, idx] = zhist_interp

    if logcb:
        norm = "log"
        vmin = 1e-2
    else:
        norm = None
        vmin = None

    pcx = ax_list[0].pcolormesh(
        t_arr,
        v_arr[0] * scales[0],
        vx_arr,
        shading="nearest",
        cmap="batlow",
        norm=norm,
        vmin=vmin,
    )
    cox = ax_list[0].contour(
        t_arr,
        v_arr[0] * scales[0],
        vx_arr,
        levels=clevels,
        colors="white",
        norm=norm,
        alpha=0.7,
        vmin=vmin,
    )

    pcy = ax_list[1].pcolormesh(
        t_arr,
        v_arr[1] * scales[1],
        vy_arr,
        shading="nearest",
        cmap="batlow",
        norm=norm,
        vmin=vmin,
    )
    coy = ax_list[1].contour(
        t_arr,
        v_arr[1] * scales[1],
        vy_arr,
        levels=clevels,
        colors="white",
        norm=norm,
        alpha=0.7,
        vmin=vmin,
    )

    pcz = ax_list[2].pcolormesh(
        t_arr,
        v_arr[2] * scales[2],
        vz_arr,
        shading="nearest",
        cmap="batlow",
        norm=norm,
        vmin=vmin,
    )
    coz = ax_list[2].contour(
        t_arr,
        v_arr[2] * scales[2],
        vz_arr,
        levels=clevels,
        colors="white",
        norm=norm,
        alpha=0.7,
        vmin=vmin,
    )

    if overplot_v:
        ax_list[0].plot(t_arr, vmean_arr[0, :])
        ax_list[1].plot(t_arr, vmean_arr[1, :])
        ax_list[2].plot(t_arr, vmean_arr[2, :])

    cbx = plt.colorbar(pcx, ax=ax_list[0])
    cby = plt.colorbar(pcy, ax=ax_list[1])
    cbz = plt.colorbar(pcz, ax=ax_list[2])

    ax_list[-1].set_xlabel("Simulation time [s]", fontsize=24, labelpad=10)
    ax_list[0].set_title(
        "Run = {}, x = {:.3f} $R_E$, y = {:.3f} $R_E$".format(runid, x0, y0),
        fontsize=28,
        pad=10,
    )

    labels = ["$V_x$ [km/s]", "$V_y$ [km/s]", "$V_z$ [km/s]"]
    if parperp:
        labels = ["$\\cos\\alpha$", "$V_\\parallel$ [km/s]", "$V_\\perp$ [km/s]"]

    for idx2, ax in enumerate(ax_list):
        ax.set(
            xlim=(t_arr[0], t_arr[-1]),
            ylim=(v_arr[idx2][0] * scales[idx2], v_arr[idx2][-1] * scales[idx2]),
        )
        ax.set_ylabel(labels[idx2], fontsize=24, labelpad=10)
        ax.label_outer()
        ax.tick_params(labelsize=20)
        cbax = [cbx, cby, cbz][idx2]
        cbax.ax.tick_params(labelsize=20)
        cbax.set_label("$f~[s/m^4]$", fontsize=24)

    outdir = wrkdir_DNR + "Figs/1d_vdf_spectrogram"

    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            pass
    fig.savefig(
        outdir
        + "/{}_x{:.3f}_y{:.3f}_t0{}_t1{}_parperp{}.png".format(
            runid, x0, y0, t0, t1, parperp
        )
    )
    plt.close(fig)


def pos_vdf_energy_spectrogram(runid, x, y, t0, t1, emin, emax, enum=10, fluxout=True):
    runids = ["AGF", "AIA", "AIC"]
    pdmax = [1.0, 1.0, 1.0][runids.index(runid)]
    bulkpath = find_bulkpath(runid)

    global xg, yg

    xg = []
    yg = []

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g, x0, y0, drawBy0, plaschke_g
    runid_g = runid
    Blines_g = False
    drawBy0 = False
    plaschke_g = False

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
        [1e6, 750e3, 3e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runids.index(runid)]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw

    t_arr = np.arange(t0, t1 + 0.1, 0.5)
    e_arr = np.zeros(enum, dtype=float)

    # vx_arr = np.zeros((v_arr.size, t_arr.size), dtype=float)
    # vy_arr = np.zeros((v_arr.size, t_arr.size), dtype=float)
    # vz_arr = np.zeros((v_arr.size, t_arr.size), dtype=float)

    data_arr = np.zeros((e_arr.size, t_arr.size), dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)

    for idx, t in enumerate(np.arange(t0, t1 + 0.1, 0.5)):
        fnr = int(t * 2)
        filenr_g = fnr
        vobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
        )
        cellid = vobj.get_cellid([x * r_e, y * r_e, 0 * r_e])
        vdf_cellid = getNearestCellWithVspace(vobj, cellid)

        x_re, y_re, z_re = vobj.get_cell_coordinates(vdf_cellid) / r_e

        success, bin_centers, bin_values = pt.plot.energy_spectrum_jetstyle(
            vobj, vdf_cellid, "proton", emin, emax, enum=enum, fluxout=fluxout
        )
        e_arr = bin_centers

        x0 = x_re
        y0 = y_re

        data_arr[:, idx] = bin_values

    pcm = ax.pcolormesh(
        t_arr,
        e_arr,
        np.log10(data_arr),
        shading="nearest",
        cmap="hot_desaturated",
        vmin=3.5,
        vmax=8,
        # norm=colors.LogNorm(vmin=10**4, vmax=10**8),
    )
    ax.tick_params(labelsize=20)
    ax.set_yscale("log")

    # cbx = plt.colorbar(pcx, ax=ax_list[0])
    # cby = plt.colorbar(pcy, ax=ax_list[1])
    # cbz = plt.colorbar(pcz, ax=ax_list[2])

    cbm = plt.colorbar(pcm, ax=ax)

    ax.set_xlabel("Simulation time [s]", fontsize=24, labelpad=10)
    ax.set_title(
        "Run = {}, x = {:.3f} $R_E$, y = {:.3f} $R_E$".format(runid, x0, y0),
        fontsize=28,
        pad=10,
    )

    ax.set(xlim=(t_arr[0], t_arr[-1]), ylim=(e_arr[0], e_arr[-1]))
    ax.set_ylabel("$E$ [eV]", fontsize=24, labelpad=10)
    ax.label_outer()
    ax.tick_params(labelsize=20)
    cbax = cbm
    cbax.ax.tick_params(labelsize=20)
    if fluxout:
        cbax.set_label("Flux [keV/(cm$^2$ s sr keV)]", fontsize=24)
    else:
        cbax.set_label("PSD", fontsize=24)

    # for idx2, ax in enumerate(ax_list):
    #     ax.set(xlim=(t_arr[0], t_arr[-1]), ylim=(v_arr[0] * 1e-3, v_arr[-1] * 1e-3))
    #     ax.set_ylabel(labels[idx2], fontsize=24, labelpad=10)
    #     ax.label_outer()
    #     ax.tick_params(labelsize=20)
    #     cbax = [cbx, cby, cbz][idx2]
    #     cbax.ax.tick_params(labelsize=20)
    #     cbax.set_label("$f~[s/m^4]$", fontsize=24)

    outdir = wrkdir_DNR + "Figs/1d_vdf_spectrogram"

    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            pass
    fig.savefig(
        outdir
        + "/{}_x{:.3f}_y{:.3f}_t0{}_t1{}_energy.png".format(runid, x0, y0, t0, t1)
    )
    plt.close(fig)


def pos_vdf_profile_plotter(runid, x, y, t0, t1, vmin=None, vmax=None):
    runids = ["AGF", "AIA"]
    pdmax = [1.0, 1.0][runids.index(runid)]
    bulkpath = find_bulkpath(runid)

    global xg, yg

    xg = []
    yg = []

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g, x0, y0, drawBy0, plaschke_g
    runid_g = runid
    Blines_g = False
    drawBy0 = False
    plaschke_g = False

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

        plotbin_centers = np.arange(vmin, vmax, (xbin_edges[1] - xbin_edges[0]))

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
            cbtitle="$P_\\mathrm{dyn}$ [nPa]",
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

        ax_list[1].step(
            plotbin_centers * 1e-3,
            np.interp(plotbin_centers, xbin_centers, xhist),
            "k",
            label="vx",
        )
        ax_list[1].step(
            plotbin_centers * 1e-3,
            np.interp(plotbin_centers, ybin_centers, yhist),
            "r",
            label="vy",
        )
        ax_list[1].step(
            plotbin_centers * 1e-3,
            np.interp(plotbin_centers, zbin_centers, zhist),
            "b",
            label="vz",
        )
        ax_list[1].legend(loc="upper right")
        ax_list[1].set_xlim(-2000, 2000)
        ax_list[1].set_xlabel("$v~[\\mathrm{kms}^{-1}]$")
        ax_list[1].set_ylabel("$f(v)~[\\mathrm{sm}^{-4}]$")

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


def vdf_along_fieldline(
    runid,
    x,
    y,
    t0,
    t1,
    skip=False,
    xyz=False,
    boxwidth=2000e3,
    pdmax=1.0,
    ncont=5,
    margin=1,
    fmin=1e-10,
    fmax=1e-4,
    npoints=5,
    max_dist=0.1,
    direction="forward",
    dr=0.1,
    justline=False,
):

    runids = ["AGF", "AIA", "AIC"]
    # pdmax = [1.0, 1.0, 1.0][runids.index(runid)]
    bulkpath = find_bulkpath(runid)

    global xg, yg

    xg = []
    yg = []

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g, x0, y0, plaschke_g, drawBy0, linsg, draw_qperp, leg_g
    drawBy0 = True
    draw_qperp = False
    plaschke_g = False
    runid_g = runid

    linsg = False
    leg_g = False
    Blines_g = False

    non_ids = []
    sj_ids = []

    sj_ids_g = sj_ids
    non_ids_g = non_ids

    sw_pars = [
        [1e6, 750e3, 3e-9, 0.5e6],
        [1e6, 750e3, 3e-9, 0.5e6],
        [1e6, 750e3, 3e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runids.index(runid)]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw

    vobj = pt.vlsvfile.VlsvReader(
        bulkpath + "bulk.{}.vlsv".format(str(int(t0 * 2)).zfill(7))
    )
    cellid = vobj.get_cellid([x * r_e, y * r_e, 0 * r_e])
    vdf_cellid = getNearestCellWithVspace(vobj, cellid)

    x_re, y_re, z_re = vobj.get_cell_coordinates(vdf_cellid) / r_e

    outdir = wrkdir_DNR + "VDFs/Balong_{}/x_{:.3f}_y_{:.3f}_t0_{}_t1_{}_xyz{}".format(
        runid, x_re, y_re, t0, t1, xyz
    )

    along_cellids = [vdf_cellid]
    along_coords = [np.array([x_re, y_re, z_re])]
    traveled_dist = 0.0
    xcurr, ycurr = (x_re, y_re)

    if direction == "backward":
        dr_sgn = -1
    else:
        dr_sgn = 1

    if not justline:

        while len(along_cellids) < npoints:
            Bcurr = vobj.read_interpolated_variable(
                "vg_b_vol", [xcurr * r_e, ycurr * r_e, 0]
            )
            bx, by = Bcurr[:2] / np.linalg.norm(Bcurr[:2])
            xnew, ynew = (xcurr + bx * dr * dr_sgn, ycurr + by * dr * dr_sgn)
            curr_cell = vobj.get_cellid([xnew * r_e, ynew * r_e, 0])
            closest_vdf_cell = getNearestCellWithVspace(vobj, curr_cell)
            dist_to_vdf = (
                np.linalg.norm(
                    vobj.get_cell_coordinates(curr_cell)
                    - vobj.get_cell_coordinates(closest_vdf_cell)
                )
                / r_e
            )
            # print(dist_to_vdf)
            if dist_to_vdf < max_dist and closest_vdf_cell not in along_cellids:
                along_cellids.append(closest_vdf_cell)
                along_coords.append(vobj.get_cell_coordinates(closest_vdf_cell) / r_e)
            traveled_dist += dr
            xcurr, ycurr = (xnew, ynew)
            # print(xnew, ynew)
            print(traveled_dist)
            if traveled_dist > 20:
                break
    else:
        for idx in range(1, npoints):
            curr_coord = np.array([x_re, y_re + dr_sgn * idx * 25 * 300e3 / r_e, z_re])
            curr_cell = vobj.get_cellid(curr_coord * r_e)
            curr_vdf_cell = getNearestCellWithVspace(vobj, curr_cell)
            along_coords.append(vobj.get_cell_coordinates(curr_vdf_cell) / r_e)
            along_cellids.append(curr_vdf_cell)

    along_coords = np.array(along_coords)
    print(along_cellids)

    if xyz:
        bpara = [None, None, None]
        bpara1 = [None, None, None]
        bperp = [None, None, None]
        xy = [1, None, None]
        xz = [None, 1, None]
        yz = [None, None, 1]
        bvector = [1, 1, 1]
    else:
        bpara = [1, None, None]
        bpara1 = [None, 1, None]
        bperp = [None, None, 1]
        xy = [None, None, None]
        xz = [None, None, None]
        yz = [None, None, None]
        bvector = [None, None, None]

    for t in np.arange(t0, t1 + 0.1, 0.5):
        print("t = {}s".format(t))
        fnr = int(t * 2)
        if skip and os.path.isfile(outdir + "/{}.png".format(fnr)):
            continue
        filenr_g = fnr
        vobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
        )
        cellid = vobj.get_cellid([x * r_e, y * r_e, 0 * r_e])
        vdf_cellid = getNearestCellWithVspace(vobj, cellid)

        v = vobj.read_variable("proton/vg_v", cellids=vdf_cellid) * 1e-3
        vth = vobj.read_variable("proton/vg_thermalvelocity", cellids=vdf_cellid) * 1e-3

        if not xyz:
            B = vobj.read_variable("vg_b_vol", cellids=vdf_cellid)
            b = B / np.linalg.norm(B)
            vpar = np.dot(v, b)
            vperp1 = np.cross(b, v)
            vperp2 = np.cross(b, np.cross(b, v))

        x_re, y_re, z_re = vobj.get_cell_coordinates(vdf_cellid) / r_e

        x0 = x_re
        y0 = y_re

        boxre = [
            np.min(along_coords[:, 0]) - margin,
            np.max(along_coords[:, 0]) + margin,
            np.min(along_coords[:, 1]) - margin,
            np.max(along_coords[:, 1]) + margin,
        ]

        # fig, ax_list = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
        fig = plt.figure(
            figsize=(12 + 4 * len(along_cellids), 12), constrained_layout=True
        )

        gs = fig.add_gridspec(9, 9 + 1 + 3 * len(along_cellids))

        cmap_ax = fig.add_subplot(gs[0:9, 0:9])
        # cmap_cb_ax = fig.add_subplot(gs[0:9, 9])
        vdf_ax_list = np.empty((3, len(along_cellids)), dtype=object)
        for row_idx in range(3):
            for col_idx in range(len(along_cellids)):
                vdf_ax_list[row_idx, col_idx] = fig.add_subplot(
                    gs[
                        3 * row_idx : 3 * row_idx + 3,
                        9 + 3 * col_idx : 9 + 3 * col_idx + 3,
                    ]
                )
        vdf_cb_ax = fig.add_subplot(gs[0:9, -1])

        pt.plot.plot_colormap(
            axes=cmap_ax,
            # cbaxes=cmap_cb_ax,
            vlsvobj=vobj,
            var="proton/vg_Pdyn",
            vmin=0.01,
            vmax=pdmax,
            vscale=1e9,
            cbtitle="$P_\\mathrm{dyn}$ [nPa]",
            usesci=0,
            boxre=boxre,
            # internalcb=True,
            # lin=1,
            colormap="batlow",
            scale=2,
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
            streamlines="vg_b_vol",
            streamlinedensity=0.4,
            streamlinecolor="white",
            streamlinethick=1,
            streamlinestartpoints=np.array([[x0, y0]]),
        )
        # cmap_ax.plot(
        #     along_coords[:, 0],
        #     along_coords[:, 1],
        #     linestyle="",
        #     marker=["$" + str(iii + 1) + "$" for iii in range(len(along_cellids))],
        #     color="red",
        #     markersize=10,
        # )
        for col_idx in range(len(along_cellids)):
            cmap_ax.plot(
                along_coords[col_idx, 0],
                along_coords[col_idx, 1],
                linestyle="",
                marker="$" + str(col_idx + 1) + "$",
                markersize=10,
                color="red",
            )

        for row_idx in range(3):
            for col_idx in range(len(along_cellids)):
                if row_idx == 0 and col_idx == 0:
                    cbaxes = vdf_cb_ax
                    nocb = None
                else:
                    cbaxes = None
                    nocb = True
                pt.plot.plot_vdf(
                    axes=vdf_ax_list[row_idx, col_idx],
                    vlsvobj=vobj,
                    cellids=[along_cellids[col_idx]],
                    colormap="batlow",
                    bvector=bvector[row_idx],
                    xy=xy[row_idx],
                    xz=xz[row_idx],
                    yz=yz[row_idx],
                    bpara=bpara[row_idx],
                    bpara1=bpara1[row_idx],
                    bperp=bperp[row_idx],
                    slicethick=0,
                    box=[-boxwidth, boxwidth, -boxwidth, boxwidth],
                    # internalcb=True,
                    setThreshold=1e-15,
                    scale=2,
                    fmin=fmin,
                    fmax=fmax,
                    contours=ncont,
                    cbaxes=cbaxes,
                    nocb=nocb,
                    title="",
                )
                # if row_idx != 2:
                #     vdf_ax_list[row_idx, col_idx].xaxis.set_ticklabels([])
                #     vdf_ax_list[row_idx, col_idx].set_xlabel("")
                # if col_idx != 0:
                #     vdf_ax_list[row_idx, col_idx].yaxis.set_ticklabels([])
                #     vdf_ax_list[row_idx, col_idx].set_ylabel("")
                if row_idx == 0:
                    vdf_ax_list[row_idx, col_idx].set_title(
                        "{}".format(col_idx + 1), fontsize=20
                    )

        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except OSError:
                pass

        fig.suptitle(
            "Run: {}, x: {:.3f}, y: {:.3f}, Time: {}s".format(runid, x_re, y_re, t),
            fontsize=20,
        )
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except OSError:
                pass
        fig.savefig(outdir + "/{}.png".format(fnr))
        plt.close(fig)

    return None


def plot_jet_formation_postime(
    runid,
    ymin,
    ymax,
    tmin,
    tmax,
    minduration=0.0,
    minsize=0,
    cmap="lipari",
    s=1,
):

    bulkpath = find_bulkpath(runid)
    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath + "bulk.{}.vlsv".format(str(int(tmin * 2)).zfill(7))
    )

    y_values = []
    t_values = []
    maxsize_values = []
    # duration_values = []
    ymax_values = []
    ymin_values = []

    y_arr = np.array([])
    t_arr = np.array([])
    maxsize_arr = np.array([])

    for n1 in range(6000):
        try:
            props = PropReader(str(n1).zfill(5), runid, transient="jet")
        except:
            continue

        print("Current jet ID = {}".format(n1))

        if props.read("at_bow_shock")[0] != 1:
            continue

        # if "splinter" in props.meta:
        #     continue

        t = np.array(props.get_times())
        # isnotmerger = (props.read("is_merger") == 0).astype(bool)
        isnotmerger = np.ones_like(t).astype(bool)
        xmean = props.read("x_mean")
        ymean = props.read("y_mean")

        x0, y0 = (xmean[0], ymean[0])

        t0 = t[0]
        duration = t[-1] - t[0] + 0.5
        maxsize = max(props.read("Nr_cells"))

        if np.sqrt(x0**2 + y0**2) < 8:
            continue
        if duration < minduration:
            continue
        if maxsize < minsize:
            continue
        if t0 < tmin:
            continue
        if t0 > tmax:
            continue
        if y0 < ymin:
            continue
        if y0 > ymax:
            continue

        t = t[isnotmerger]
        xmean = xmean[isnotmerger]
        ymean = ymean[isnotmerger]

        cell_list = props.get_cells()
        cell_list = [cell for cell in cell_list if isnotmerger[cell_list.index(cell)]]
        ymins = np.array(
            [vlsvobj.get_cell_coordinates(min(cell))[1] / r_e for cell in cell_list]
        )
        ymaxs = np.array(
            [vlsvobj.get_cell_coordinates(max(cell))[1] / r_e for cell in cell_list]
        )

        # y_values.append(y0)
        # t_values.append(t0)
        # maxsize_values.append(maxsize)
        # duration_values.append(duration)

        y_arr = np.append(y_arr, ymean)
        t_arr = np.append(t_arr, t)
        maxsize_arr = np.append(maxsize_arr, props.read("Nr_cells")[isnotmerger])
        y_values.append(ymean)
        t_values.append(t)
        maxsize_values.append(props.read("Nr_cells")[isnotmerger])
        ymin_values.append(ymins)
        ymax_values.append(ymaxs)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)
    ax.grid(zorder=2.5)

    # ax.plot(t_values, y_values, "o", color=CB_color_cycle[0])
    # ax.scatter(
    #     t_values,
    #     y_values,
    #     s=maxsize_values,
    #     c=duration_values,
    #     cmap=cmap,
    #     marker="o",
    #     norm="log",
    #     alpha=0.5,
    #     edgecolors="k",
    #     zorder=4.5,
    # )
    ax.scatter(
        t_arr,
        y_arr,
        c=np.log(maxsize_arr),
        cmap=cmap,
        zorder=4.5,
        marker="o",
        edgecolors="none",
        # alpha=0.5,
        s=s,
        rasterized=True,
    )
    for idx in range(len(y_values)):
        ax.plot(
            t_values[idx],
            y_values[idx],
            color="k",
            linewidth=0.2,
            zorder=4,
            rasterized=True,
        )
        # ax.fill_between(
        #     t_values[idx],
        #     ymin_values[idx],
        #     ymax_values[idx],
        #     # alpha=0.5,
        #     zorder=5.5,
        #     edgecolor=CB_color_cycle[0],
        #     linewidth=0.2,
        #     facecolor="none",
        #     # facecolor=CB_color_cycle[0],
        #     hatch="///",
        # )
    ax.add_patch(
        mpatches.Rectangle(
            (391, 3),
            426 - 391,
            17 - (3),
            color=CB_color_cycle[0],
            label="Dusk $Q\\parallel$",
            fill=False,
            linestyle="dashed",
            linewidth=1.5,
            zorder=3.5,
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (430, 3),
            470 - 430,
            17 - (3),
            color=CB_color_cycle[1],
            label="Dusk FB",
            fill=False,
            linestyle="dashed",
            linewidth=1.5,
            zorder=3.5,
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (430, -17),
            470 - 430,
            0 - (-17),
            color=CB_color_cycle[2],
            label="Dawn RD",
            fill=False,
            linestyle="dashed",
            linewidth=1.5,
            zorder=3.5,
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (509, -17),
            600 - 509,
            -3 - (-17),
            color=CB_color_cycle[3],
            label="Dawn, young FS",
            fill=False,
            linestyle="dashed",
            linewidth=1.5,
            zorder=3.5,
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (600, -17),
            800 - 600,
            -3 - (-17),
            color=CB_color_cycle[5],
            label="Dawn $Q\\parallel$",
            fill=False,
            linestyle="dashed",
            linewidth=1.5,
            zorder=3.5,
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (470, 3),
            800 - 470,
            17 - (3),
            color=CB_color_cycle[4],
            label="Dusk $Q\\perp$",
            fill=False,
            linestyle="dashed",
            linewidth=1.5,
            zorder=3.5,
        )
    )
    # ax.add_patch(
    #     mpatches.Rectangle(
    #         (391, -17),
    #         800 - 391,
    #         17 - (-17),
    #         color=CB_color_cycle[6],
    #         label="All",
    #         fill=False,
    #         linestyle="dashed",
    #         linewidth=1.5,
    #         zorder=3.5,
    #     )
    # )

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(tmin, tmax)
    ax.set_ylabel("$Y~[R_\\mathrm{E}]$", fontsize=20, labelpad=10)
    ax.set_xlabel("$t~[\\mathrm{s}]$", fontsize=20, labelpad=10)
    ax.tick_params(labelsize=16)
    ax.legend(loc="upper right", fontsize=16)

    figdir = wrkdir_DNR + "Figs/"

    fig.savefig(figdir + "formation_postime.png", dpi=300)
    fig.savefig(figdir + "formation_postime.pdf", dpi=300)
    plt.close(fig)


def all_cats_properties_script():

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
        "jets_qpar_before",
        "jets_qpar_after",
        "jets_qpar_fb",
        "jets_qperp_rd",
        "jets_all",
        "jets_qperp_after",
        "jets_qperp_inter",
    ]
    tmins = [391, 470, 430, 430, 391, 600, 509]
    tmaxs = [426, 800, 470, 470, 800, 800, 600]

    for idx in range(len(folder_suffixes)):
        get_jet_category_properties(
            "AIC",
            boxre=boxres[idx],
            tmin=tmins[idx],
            tmax=tmaxs[idx],
            folder_suffix=folder_suffixes[idx],
            minduration=1,
            minsize=4,
        )


def get_jet_category_properties(
    runid,
    boxre=None,
    tmin=None,
    tmax=None,
    folder_suffix="jets",
    minduration=0,
    minsize=0,
):

    txtdir = wrkdir_DNR + "jet_categories/"

    if not os.path.exists(txtdir):
        try:
            os.makedirs(txtdir)
        except OSError:
            pass

    jet_ids = []
    durs = []
    maxs = []
    rpen = []
    Dn = []
    Dpd = []
    DTPar = []
    DTPerp = []

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

        rmean = props.read("r_mean")
        maxsize = max(props.read("A"))

        jet_ids.append(n1)
        durs.append(duration)
        maxs.append(maxsize)
        rpen.append(rmean[-1] - rmean[0])
        Dn.append(props.read("Dn")[0])
        Dpd.append(props.read("Dpd")[0])
        DTPar.append(props.read("DTPar")[0])
        DTPerp.append(props.read("DTPerp")[0])

    outarr = np.array([jet_ids, durs, maxs, rpen, Dn, Dpd, DTPar, DTPerp], dtype=float)

    np.savetxt(txtdir + "{}.txt".format(folder_suffix), outarr)


def plot_category_props(
    folder_suffixes=[
        "jets_qpar_before",
        "jets_qpar_fb",
        "jets_qperp_rd",
        "jets_qperp_after",
        "jets_qperp_inter",
        "jets_qpar_after",
    ],
    aspect=0.5,
    avg=False,
):
    sfx_valid = [
        "jets_qpar_before",
        "jets_qpar_fb",
        "jets_qperp_rd",
        "jets_qperp_after",
        "jets_qperp_inter",
        "jets_qpar_after",
        "jets_all",
    ]
    sfx_labels = [
        "Dusk $Q_\\parallel$",
        "Dusk FB",
        "Dawn RD",
        "Dawn $Q_\\parallel$",
        "Dawn young FS",
        "Dusk $Q_\\perp$",
        "All",
    ]

    prop_labels = [
        "Duration [s]",
        "Max. area\n[$R_\\mathrm{E}^2$]",
        "Radial depth\n[$R_\\mathrm{E}$]",
        "Number of\njets",
    ]

    txtdir = wrkdir_DNR + "jet_categories/"

    categories_list = []

    for sfx in folder_suffixes:
        jetids, durs, maxs, rpens, Dn, Dpd, DTPar, DTPerp = np.loadtxt(
            txtdir + "{}.txt".format(sfx)
        )
        njets = np.ones_like(durs) * jetids.size
        categories_list.append([durs, maxs, -rpens, njets])

    carr = np.ones((len(folder_suffixes), len(prop_labels)), dtype=float) * np.nan

    fig, ax = plt.subplots()
    im = ax.imshow(carr, aspect=aspect)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(prop_labels)), labels=prop_labels)
    ax.set_yticks(np.arange(len(sfx_labels)), labels=sfx_labels)

    if avg:
        reducer = lambda x: np.nanmean(x)
    else:
        reducer = lambda x: np.nanmedian(x)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(folder_suffixes)):
        for j in range(len(prop_labels)):
            if j == 3:
                text = ax.text(
                    j,
                    i,
                    "{:d}".format(int(reducer(categories_list[i][j]))),
                    ha="center",
                    va="center",
                    color="k",
                )
            else:
                text = ax.text(
                    j,
                    i,
                    "{:.3f}".format(reducer(categories_list[i][j])),
                    ha="center",
                    va="center",
                    color="k",
                )

    for i in range(len(sfx_labels) - 1):
        ax.axhline(i + 0.5, color="k")
    for j in range(len(prop_labels) - 1):
        ax.axvline(j + 0.5, color="k")

    if avg:
        ax.set_title("Average properties")
    else:
        ax.set_title("Median properties")
    # ax.spines[:].set_visible(False)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    fig.tight_layout()
    figdir = wrkdir_DNR + "Figs/"
    if avg:
        fig.savefig(
            figdir + "jet_prop_averages.png",
            dpi=300,
        )
    else:
        fig.savefig(
            figdir + "jet_prop_medians.png",
            dpi=300,
        )
    plt.close(fig)


def plot_category_histograms(
    folder_suffixes=[
        "jets_qpar_before",
        "jets_qpar_fb",
        "jets_qperp_rd",
        "jets_qperp_after",
        "jets_qperp_inter",
        "jets_qpar_after",
    ]
):

    sfx_valid = [
        "jets_qpar_before",
        "jets_qpar_fb",
        "jets_qperp_rd",
        "jets_qperp_after",
        "jets_qperp_inter",
        "jets_qpar_after",
    ]
    sfx_labels = [
        "Dusk $Q_\\parallel$",
        "Dusk FB",
        "Dawn RD",
        "Dawn $Q_\\parallel$",
        "Dawn young FS",
        "Dusk $Q_\\perp$",
    ]

    prop_labels = [
        "Duration [s]",
        "Max. area [$R_\\mathrm{E}^2$]",
        "$r(t_0)-r(t_\\mathrm{last})$ [$R_\\mathrm{E}$]",
        # "$\\delta n~[\\mathrm{cm}^{-3}]$",
        # "$\\delta P_\\mathrm{dyn}$ [nPa]",
        # "$\\delta T_\\parallel$ [MK]",
        # "$\\delta T_\\perp$ [MK]",
    ]

    panel_labs = ["(a)", "(b)", "(c)"]

    txtdir = wrkdir_DNR + "jet_categories/"

    jetids_all, durs_all, maxs_all, rpen_all, Dn_all, Dpd_all, DTPar_all, DTPerp_all = (
        np.loadtxt(txtdir + "{}.txt".format("jets_all"))
    )
    all_arrs = [
        durs_all,
        maxs_all,
        -rpen_all[-rpen_all >= 0],
        Dn_all,
        Dpd_all,
        DTPar_all,
        DTPerp_all,
    ]

    bin_edges = [
        10 ** np.histogram_bin_edges(np.log10(durs_all)),
        10 ** np.histogram_bin_edges(np.log10(maxs_all)),
        # np.histogram_bin_edges(durs_all),
        # np.histogram_bin_edges(maxs_all),
        np.histogram_bin_edges(-rpen_all[-rpen_all >= 0]),
        np.histogram_bin_edges(Dn_all),
        np.histogram_bin_edges(Dpd_all),
        np.histogram_bin_edges(DTPar_all),
        np.histogram_bin_edges(DTPerp_all),
    ]

    categories_list = []

    for sfx in folder_suffixes:
        jetids, durs, maxs, rpens, Dn, Dpd, DTPar, DTPerp = np.loadtxt(
            txtdir + "{}.txt".format(sfx)
        )
        categories_list.append([durs, maxs, -rpens, Dn, Dpd, DTPar, DTPerp])

    fig, ax_list = plt.subplots(
        1, len(prop_labels), figsize=(4 * len(prop_labels), 5), constrained_layout=True
    )

    for idx in range(len(prop_labels)):
        ax = ax_list[idx]
        ax.tick_params(labelsize=12)
        ax.set_xlabel(prop_labels[idx], fontsize=16, labelpad=10)
        ax.grid()
        # if idx == 2:
        #     cumul = True
        #     stacked = True
        #     histtype = "barstacked"
        if idx in [0, 1, 2]:
            cumul = -1
            stacked = True
            histtype = "barstacked"
        else:
            cumul = False
            stacked = False
            histtype = "step"
        # for idx2 in range(len(folder_suffixes)):
        #     ax.hist(
        #         categories_list[idx2][idx],
        #         bins=bin_edges[idx],
        #         # density=True,
        #         weights=1.0
        #         / (
        #             np.ones(len(categories_list[idx2][idx]), dtype=float)
        #             * len(categories_list[idx2][idx])
        #         ),
        #         label=sfx_labels[idx2],
        #         color=CB_color_cycle[idx2],
        #         histtype="step",
        #         alpha=0.7,
        #         cumulative=cumul,
        #     )
        idx2_range = range(len(folder_suffixes))
        ax.hist(
            [categories_list[idx2][idx] for idx2 in idx2_range],
            bins=bin_edges[idx],
            # weights=[
            #     np.ones(len(categories_list[idx2][idx]), dtype=float)
            #     / (
            #         np.ones(len(categories_list[idx2][idx]), dtype=float)
            #         * len(categories_list[idx2][idx])
            #     )
            #     for idx2 in idx2_range
            # ],
            label=[sfx_labels[idx2] for idx2 in idx2_range],
            color=[CB_color_cycle[idx2] for idx2 in idx2_range],
            histtype=histtype,
            # alpha=0.7,
            cumulative=cumul,
            stacked=stacked,
        )
        ax.set_xlim(bin_edges[idx][0], bin_edges[idx][-1])
        # ax.hist(
        #     all_arrs[idx],
        #     bins=bin_edges[idx],
        #     density=True,
        #     label="All",
        #     color="k",
        #     histtype="step",
        #     alpha=0.7,
        #     cumulative=cumul,
        # )
        if idx == 0:
            ax.set_ylabel("Cumulative\nnumber of jets", fontsize=16, labelpad=10)
            ax.legend(fontsize=12)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ScalarFormatter())
        if idx == 1:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ScalarFormatter())
        # ax.set_yscale("log")
        ax.set_ylim(0.01, None)

        ax.annotate(
            panel_labs[idx], xy=(0.15, 0.9), xycoords="axes fraction", fontsize=16
        )

    figdir = wrkdir_DNR + "Figs/"

    fig.savefig(figdir + "jet_comp_hist.png", dpi=300)
    fig.savefig(figdir + "jet_comp_hist.pdf", dpi=300)

    plt.close(fig)


def archerplot_4():

    valid_cats = [
        "jets_qpar_before",
        "jets_qpar_fb",
        "jets_qperp_rd",
        "jets_qperp_after",
        "jets_qperp_inter",
        "jets_qpar_after",
    ]
    cat_names = [
        "Dusk $Q_\\parallel$",
        "Dusk FB",
        "Dawn RD",
        "Dawn $Q_\\parallel$",
        "Dawn young FS",
        "Dusk $Q_\\perp$",
    ]
    markers = ["x", "x", "o", "x", "x", "o"]
    # pair_markers = ["x", "o", "x", "x", "o", "o"]
    # pair_colors = ["k", "red", "k", "k", "red", "red"]
    # pair_ax_idx = [1, 1, 3, 2, 2, 3]
    colors = [
        "k",
        CB_color_cycle[3],
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        CB_color_cycle[4],
    ]
    pair_markers = ["x", "x", "o", "x", "o", "^"]
    pair_colors = [
        "k",
        "k",
        CB_color_cycle[3],
        "k",
        CB_color_cycle[3],
        CB_color_cycle[0],
    ]
    pair_ax_idx = [1, 2, 2, 3, 3, 3]
    panel_labs = ["(a)", "(b)", "(c)", "(d)"]

    fig, ax_list = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)
    ax_flat = ax_list.flatten()
    avgs = []
    meds = []
    xall = []
    yall = []

    for idx in range(len(valid_cats)):
        ax = ax_flat[pair_ax_idx[idx]]
        folder_suffix = valid_cats[idx]
        filenames = os.listdir(wrkdir_DNR + "txts/timeseries/" + folder_suffix)
        filenames = [fname for fname in filenames if "corr" not in fname]

        xvals = []
        yvals = []

        for idx2, fn in enumerate(filenames):
            data_arr = np.loadtxt(
                wrkdir_DNR + "txts/timeseries/" + folder_suffix + "/" + fn
            )
            pdyn = data_arr[5, :]
            v = data_arr[4, :]
            rho = data_arr[0, :]

            rhocontrib = (
                rho[pdyn == max(pdyn)][0] - np.nanmean(rho[:20])
            ) / np.nanmean(rho[:20])
            vcontrib = (
                (v**2)[pdyn == max(pdyn)][0] - np.nanmean((v**2)[:20])
            ) / np.nanmean((v**2)[:20])
            pdyncontrib = (max(pdyn) - np.nanmean(pdyn[:20])) / np.nanmean(pdyn[:20])

            xvals.append(rhocontrib / pdyncontrib)
            yvals.append(vcontrib / pdyncontrib)
            xall.append(rhocontrib / pdyncontrib)
            yall.append(vcontrib / pdyncontrib)

            if (
                rhocontrib / pdyncontrib > 2.5
                or vcontrib / pdyncontrib > 2.5
                or rhocontrib / pdyncontrib < -1
                or vcontrib / pdyncontrib < -1
            ):
                print(
                    "Jet of type {} has values outside of limits: ({:.2f},{:.2f})".format(
                        cat_names[idx], rhocontrib / pdyncontrib, vcontrib / pdyncontrib
                    )
                )

            if idx2 == 0:
                ax_flat[0].plot(
                    rhocontrib / pdyncontrib,
                    vcontrib / pdyncontrib,
                    markers[idx],
                    color=colors[idx],
                    label=cat_names[idx],
                    markersize=8,
                    fillstyle="none",
                )
                ax.plot(
                    rhocontrib / pdyncontrib,
                    vcontrib / pdyncontrib,
                    pair_markers[idx],
                    color=pair_colors[idx],
                    label=cat_names[idx],
                    markersize=8,
                    fillstyle="none",
                )

            else:
                ax_flat[0].plot(
                    rhocontrib / pdyncontrib,
                    vcontrib / pdyncontrib,
                    markers[idx],
                    color=colors[idx],
                    markersize=8,
                    fillstyle="none",
                )
                ax.plot(
                    rhocontrib / pdyncontrib,
                    vcontrib / pdyncontrib,
                    pair_markers[idx],
                    color=pair_colors[idx],
                    markersize=8,
                    fillstyle="none",
                )

        avgs.append([np.nanmean(xvals), np.nanmean(yvals)])
        meds.append([np.nanmedian(xvals), np.nanmedian(yvals)])

    for idx, ax in enumerate(ax_flat):
        ax.set_xlabel(
            "$\\frac{\\delta\\rho(P_\\mathrm{dyn,max})}{\\langle \\rho \\rangle_\\mathrm{pre-jet}} / \\frac{\\delta P_\\mathrm{dyn} (P_\\mathrm{dyn,max})}{\\langle P_\\mathrm{dyn} \\rangle_\\mathrm{pre-jet}}$",
            fontsize=24,
            labelpad=10,
        )
        ax.set_ylabel(
            "$\\frac{\\delta v^2 (P_\\mathrm{dyn,max})}{\\langle v^2 \\rangle_\\mathrm{pre-jet}} / \\frac{\\delta P_\\mathrm{dyn} (P_\\mathrm{dyn,max})}{\\langle P_\\mathrm{dyn} \\rangle_\\mathrm{pre-jet}}$",
            fontsize=24,
            labelpad=10,
        )
        ax.axvline(0, linestyle="dashed", linewidth=0.6)
        ax.axhline(0, linestyle="dashed", linewidth=0.6)
        ax.grid()
        ax.legend(fontsize=14)
        ax.set_xlim(-1, 2.5)
        ax.set_ylim(-1, 2.5)
        ax.label_outer()
        ax.tick_params(labelsize=12)
        ax.annotate(
            panel_labs[idx], xy=(0.05, 0.95), xycoords="axes fraction", fontsize=20
        )

    # for ax in ax_flat:
    #     ax.legend()
    #     handles, labels = ax.get_legend_handles_labels()
    #     for idx in range(len(labels)):
    #         labels[idx] = labels[idx] + ", med: ({:.2f}, {:.2f})".format(
    #             meds[idx][0], meds[idx][1]
    #         )
    #     ax.legend(handles, labels, fontsize=14)

    handles, labels = ax_flat[0].get_legend_handles_labels()
    for idx in range(len(labels)):
        labels[idx] = labels[idx] + ", med: ({:.2f}, {:.2f})".format(
            meds[idx][0], meds[idx][1]
        )
    ax_flat[0].legend(handles, labels, fontsize=14)

    fig.savefig(wrkdir_DNR + "Figs/archerplot_4.pdf", dpi=300)
    plt.close(fig)


def archerplot():

    valid_cats = [
        "jets_qpar_before",
        "jets_qpar_fb",
        "jets_qperp_rd",
        "jets_qperp_after",
        "jets_qperp_inter",
        "jets_qpar_after",
    ]
    cat_names = [
        "Dusk $Q_\\parallel$",
        "Dusk FB",
        "Dawn RD",
        "Dawn $Q_\\parallel$",
        "Dawn young FS",
        "Dusk $Q_\\perp$",
    ]
    markers = ["o", "o", "v", "o", "o", "^"]

    fig, ax_list = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    avgs = []
    meds = []
    xall = []
    yall = []

    ax = ax_list[1]

    for idx in range(len(valid_cats)):
        folder_suffix = valid_cats[idx]
        filenames = os.listdir(wrkdir_DNR + "txts/timeseries/" + folder_suffix)
        filenames = [fname for fname in filenames if "corr" not in fname]

        xvals = []
        yvals = []

        for idx2, fn in enumerate(filenames):
            data_arr = np.loadtxt(
                wrkdir_DNR + "txts/timeseries/" + folder_suffix + "/" + fn
            )
            pdyn = data_arr[5, :]
            v = data_arr[4, :]
            rho = data_arr[0, :]

            rhocontrib = (
                rho[pdyn == max(pdyn)][0] - np.nanmean(rho[:20])
            ) / np.nanmean(rho[:20])
            vcontrib = (
                (v**2)[pdyn == max(pdyn)][0] - np.nanmean((v**2)[:20])
            ) / np.nanmean((v**2)[:20])
            pdyncontrib = (max(pdyn) - np.nanmean(pdyn[:20])) / np.nanmean(pdyn[:20])

            # rhocontrib = (rho[20] - np.nanmean(rho[:20])) / np.nanmean(rho[:20])
            # vcontrib = ((v**2)[20] - np.nanmean((v**2)[:20])) / np.nanmean((v**2)[:20])
            # pdyncontrib = (pdyn[20] - np.nanmean(pdyn[:20])) / np.nanmean(pdyn[:20])

            xvals.append(rhocontrib / pdyncontrib)
            yvals.append(vcontrib / pdyncontrib)
            xall.append(rhocontrib / pdyncontrib)
            yall.append(vcontrib / pdyncontrib)

            if (
                rhocontrib / pdyncontrib > 2.5
                or vcontrib / pdyncontrib > 2.5
                or rhocontrib / pdyncontrib < -1
                or vcontrib / pdyncontrib < -1
            ):
                print(
                    "Jet of type {} has values outside of limits: ({:.2f},{:.2f})".format(
                        cat_names[idx], rhocontrib / pdyncontrib, vcontrib / pdyncontrib
                    )
                )

            if idx2 == 0:
                ax.plot(
                    rhocontrib / pdyncontrib,
                    vcontrib / pdyncontrib,
                    markers[idx],
                    color=CB_color_cycle[idx],
                    label=cat_names[idx],
                    alpha=0.5,
                    markeredgecolor="none",
                    markersize=6,
                )
            else:
                ax.plot(
                    rhocontrib / pdyncontrib,
                    vcontrib / pdyncontrib,
                    markers[idx],
                    color=CB_color_cycle[idx],
                    alpha=0.5,
                    markeredgecolor="none",
                    markersize=6,
                )

        avgs.append([np.nanmean(xvals), np.nanmean(yvals)])
        meds.append([np.nanmedian(xvals), np.nanmedian(yvals)])

    ax.set_xlabel(
        "$\\frac{\\delta\\rho(P_\\mathrm{dyn,max})}{\\langle \\rho \\rangle_\\mathrm{pre-jet}} / \\frac{\\delta P_\\mathrm{dyn} (P_\\mathrm{dyn,max})}{\\langle P_\\mathrm{dyn} \\rangle_\\mathrm{pre-jet}}$",
        fontsize=24,
        labelpad=10,
    )
    ax.set_ylabel(
        "$\\frac{\\delta v^2 (P_\\mathrm{dyn,max})}{\\langle v^2 \\rangle_\\mathrm{pre-jet}} / \\frac{\\delta P_\\mathrm{dyn} (P_\\mathrm{dyn,max})}{\\langle P_\\mathrm{dyn} \\rangle_\\mathrm{pre-jet}}$",
        fontsize=24,
        labelpad=10,
    )
    ax.legend()
    ax.axvline(0, linestyle="dashed", linewidth=0.6)
    ax.axhline(0, linestyle="dashed", linewidth=0.6)
    ax.grid()
    ax.set_xlim(-1, 2.5)
    ax.set_ylim(-1, 2.5)
    ax.label_outer()
    ax.tick_params(labelsize=12)
    ax.annotate("(b)", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=20)

    handles, labels = ax.get_legend_handles_labels()
    for idx in range(len(labels)):
        labels[idx] = labels[idx] + ", med: ({:.2f}, {:.2f})".format(
            meds[idx][0], meds[idx][1]
        )
    ax.legend(handles, labels, fontsize=14)

    # fig.savefig(wrkdir_DNR + "Figs/archerplot.pdf", dpi=300)
    # plt.close(fig)

    # fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
    ax = ax_list[0]
    im = ax.hist2d(
        xall,
        yall,
        bins=[np.arange(-1, 2.55, 0.1), np.arange(-1, 2.55, 0.1)],
        cmap="batlow",
        cmin=1,
    )
    cb = fig.colorbar(im[3], ax=ax, pad=0.01)
    cb.set_label("Number of jets", fontsize=16, labelpad=5)
    cb.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    cb.ax.tick_params(labelsize=12)
    ax.set_xlabel(
        "$\\frac{\\delta\\rho(P_\\mathrm{dyn,max})}{\\langle \\rho \\rangle_\\mathrm{pre-jet}} / \\frac{\\delta P_\\mathrm{dyn} (P_\\mathrm{dyn,max})}{\\langle P_\\mathrm{dyn} \\rangle_\\mathrm{pre-jet}}$",
        fontsize=24,
        labelpad=10,
    )
    ax.set_ylabel(
        "$\\frac{\\delta v^2 (P_\\mathrm{dyn,max})}{\\langle v^2 \\rangle_\\mathrm{pre-jet}} / \\frac{\\delta P_\\mathrm{dyn} (P_\\mathrm{dyn,max})}{\\langle P_\\mathrm{dyn} \\rangle_\\mathrm{pre-jet}}$",
        fontsize=24,
        labelpad=10,
    )
    ax.axvline(0, linestyle="dashed", linewidth=0.6)
    ax.axhline(0, linestyle="dashed", linewidth=0.6)
    ax.set_xlim(-1, 2.5)
    ax.set_ylim(-1, 2.5)
    ax.tick_params(labelsize=12)
    ax.label_outer()
    ax.grid()
    ax.annotate("(a)", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=20)
    # fig.savefig(wrkdir_DNR + "Figs/archerplot_hist2d.pdf", dpi=300)
    fig.savefig(wrkdir_DNR + "Figs/archerplot_full.pdf", dpi=300)
    plt.close(fig)


def plot_SEA_three(paper=True):

    if paper:
        valid_cats = [
            "jets_qperp_after",
            "jets_qpar_after",
            "jets_qperp_rd",
        ]
        cat_names = [
            "Dawn $Q\\parallel$",
            "Dusk $Q\\perp$",
            "Dawn RD",
        ]
    else:
        valid_cats = [
            "jets_qpar_before",
            "jets_qpar_fb",
            "jets_qperp_inter",
        ]
        cat_names = [
            "Dusk $Q\\parallel$",
            "Dusk FB",
            "Dawn young FS",
        ]

    plot_labels = [
        "$\\rho$",
        "$|v|$",
        "$|v_x|$",
        "$|v_y|$",
        # "$v_z$",
        "$P_\\mathrm{dyn}$",
        "$|B|$",
        "$B_x$",
        "$B_y$",
        "$B_z$",
        "$E_x$",
        "$E_y$",
        "$E_z$",
        "$T_\\parallel$",
        "$T_\\perp$",
        "$T_\\perp/T_\\parallel$",
        # "$\\rho$",
        # "$v_x^2$",
        # "$v_y^2$",
        # "$v_z^2$",
    ]
    draw_legend = [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        True,
        False,
        # False,
        # False,
        # True,
    ]
    ylabels = [
        "$\\rho$\n$[\\rho_\\mathrm{pre-jet}]$",
        "$|v|$\n$[|v|_\\mathrm{pre-jet}]$",
        "$v$\n$[|v|_\\mathrm{pre-jet}]$",
        "$P_\\mathrm{dyn}$\n$[P_\\mathrm{dyn,pre-jet}]$",
        "$|B|$\n$[|B|_\\mathrm{pre-jet}]$",
        "$B$\n$[|B|_\\mathrm{pre-jet}]$",
        "$E$\n$[|E|_\\mathrm{pre-jet}]$",
        # "$T~[T_\\mathrm{pre-jet}]$",
        "$T$\n$[MK]$",
        # "$P_\\mathrm{dyn}$\ncontribution",
        "$T_\\perp/T_\\parallel$",
    ]
    plot_index = [0, 1, 2, 2, 3, 4, 5, 5, 5, 6, 6, 6, 7, 7, 8]
    plot_colors = [
        "k",
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        # CB_color_cycle[2],
        "k",
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        CB_color_cycle[0],
        CB_color_cycle[1],
        "k",
        # CB_color_cycle[0],
        # CB_color_cycle[1],
        # CB_color_cycle[2],
    ]

    fig, ax_list_list = plt.subplots(
        len(ylabels),
        3,
        figsize=(16, 10),
        constrained_layout=True,
        sharex=True,
        sharey="row",
    )
    sea_t_arr = np.arange(-10, 10 + 0.1, 0.5)

    for idx3, folder_suffix in enumerate(valid_cats):
        filenames = os.listdir(wrkdir_DNR + "txts/timeseries/" + folder_suffix)
        filenames = [fname for fname in filenames if "corr" not in fname]

        test_data = np.loadtxt(
            wrkdir_DNR + "txts/timeseries/" + folder_suffix + "/" + filenames[0]
        )

        data_arr = np.zeros(
            (len(filenames), test_data.shape[0], test_data.shape[1]), dtype=float
        )
        for idx, fn in enumerate(filenames):
            data_arr[idx, :, :] = np.loadtxt(
                wrkdir_DNR + "txts/timeseries/" + folder_suffix + "/" + fn
            )

        data_arr2 = np.zeros((len(filenames), 15, test_data.shape[1]), dtype=float)
        prejet_avg_arr = np.zeros((len(filenames), 15), dtype=float)
        for idx, fn in enumerate(filenames):
            data_arr2[idx, 0, :] = data_arr[idx, 0, :]  # Density
            data_arr2[idx, 1, :] = data_arr[idx, 4, :]  # Velocity magnitude
            data_arr2[idx, [2, 3], :] = np.abs(data_arr[idx, [1, 2], :])  # Velocity
            data_arr2[idx, 4, :] = data_arr[idx, 5, :]  # Dynamic pressure
            data_arr2[idx, 5, :] = data_arr[idx, 9, :]  # Magnetic field magnitude
            data_arr2[idx, [6, 7, 8], :] = data_arr[idx, [6, 7, 8], :]  # Magnetic field
            data_arr2[idx, [9, 10, 11], :] = data_arr[
                idx, [10, 11, 12], :
            ]  # Electric field
            data_arr2[idx, [12, 13], :] = data_arr[idx, [14, 15], :]  # Temperature
            data_arr2[idx, 14, :] = (
                data_arr[idx, 15, :] / data_arr[idx, 14, :]
            )  # Temperature aniostropy
            # data_arr2[idx, [15, 16, 17, 18], :] = data_arr[
            #     idx, [16, 17, 18, 19], :
            # ]  # Pdyn contribution
            prejet_avg_arr[idx, 0] = np.nanmean(data_arr[idx, 0, :20])
            prejet_avg_arr[idx, 1] = np.nanmean(data_arr[idx, 4, :20])
            prejet_avg_arr[idx, 2] = np.nanmean(data_arr[idx, 4, :20])
            prejet_avg_arr[idx, 3] = np.nanmean(data_arr[idx, 4, :20])
            prejet_avg_arr[idx, 4] = np.nanmean(data_arr[idx, 5, :20])
            prejet_avg_arr[idx, 5] = np.nanmean(data_arr[idx, 9, :20])
            prejet_avg_arr[idx, 6] = np.nanmean(data_arr[idx, 9, :20])
            prejet_avg_arr[idx, 7] = np.nanmean(data_arr[idx, 9, :20])
            prejet_avg_arr[idx, 8] = np.nanmean(data_arr[idx, 9, :20])
            prejet_avg_arr[idx, 9] = np.nanmean(data_arr[idx, 13, :20])
            prejet_avg_arr[idx, 10] = np.nanmean(data_arr[idx, 13, :20])
            prejet_avg_arr[idx, 11] = np.nanmean(data_arr[idx, 13, :20])
            prejet_avg_arr[idx, 12] = 1
            prejet_avg_arr[idx, 13] = 1
            prejet_avg_arr[idx, 14] = 1

        for idx in range(len(filenames)):
            for idx2 in range(len(plot_index)):
                data_arr2[idx, idx2, :] /= prejet_avg_arr[idx, idx2]

        cat_avgs = np.nanmean(data_arr2, axis=0)
        cat_meds = np.nanmedian(data_arr2, axis=0)
        cat_25 = np.percentile(data_arr2, 25, axis=0)
        cat_75 = np.percentile(data_arr2, 75, axis=0)

        # print("\n" + cat_names[valid_cats.index(folder_suffix)])
        ax_list = ax_list_list[:, idx3]
        for idx2 in range(len(plot_index)):
            ax = ax_list[plot_index[idx2]]
            # print("\n{}".format(plot_labels[idx2]))
            # print("Prejet avg: {}".format(np.nanmean(prejet_avg_arr[:, idx2])))
            # print("Begin: {}".format(cat_avgs[idx2, 0] / np.nanmean(cat_avgs[idx2, :20])))
            # print("Form: {}".format(cat_avgs[idx2, 20] / np.nanmean(cat_avgs[idx2, :20])))
            # print("End: {}".format(cat_avgs[idx2, -1] / np.nanmean(cat_avgs[idx2, :20])))
            ax.plot(
                sea_t_arr,
                cat_avgs[idx2, :],
                color=plot_colors[idx2],
                label=plot_labels[idx2],
                linewidth=1.2,
                zorder=2,
            )
            ax.fill_between(
                sea_t_arr,
                cat_25[idx2],
                cat_75[idx2],
                facecolor=plot_colors[idx2],
                alpha=0.2,
                zorder=1,
            )
            if draw_legend[idx2] and idx3 == 0:
                ax.legend(loc="upper right")

        for idx, ax in enumerate(ax_list):
            ax.grid(zorder=0)
            ax.set_xlim(sea_t_arr[0], sea_t_arr[-1])
            ax.set_ylabel(ylabels[idx], fontsize=12, labelpad=10)
            ax.label_outer()
        ax_list[-1].set_xlabel("Epoch time [s]", fontsize=16, labelpad=10)
        ax_list[0].set_title(
            cat_names[valid_cats.index(folder_suffix)]
            + ", N = {}".format(len(filenames)),
            fontsize=16,
            pad=10,
        )

    if paper:
        fig.savefig(wrkdir_DNR + "Figs/SEA_new_three.png", dpi=300)
        fig.savefig(wrkdir_DNR + "Figs/SEA_new_three.pdf", dpi=300)
    else:
        fig.savefig(wrkdir_DNR + "Figs/SEA_new_three_supp.png", dpi=300)
        fig.savefig(wrkdir_DNR + "Figs/SEA_new_three_supp.pdf", dpi=300)

    plt.close(fig)


def plot_category_SEA_new(folder_suffix="jets"):

    valid_cats = [
        "jets_all",
        "jets_qpar_before",
        "jets_qpar_after",
        "jets_qpar_fb",
        "jets_qperp_rd",
        "jets_qperp_after",
        "jets_qperp_inter",
    ]
    cat_names = [
        "All",
        "Dusk $Q\\parallel$",
        "Dusk $Q\\perp$",
        "Dusk FB",
        "Dawn RD",
        "Dawn $Q\\parallel$",
        "Dawn young FS",
    ]

    plot_labels = [
        "$\\rho$",
        "$|v|$",
        "$|v_x|$",
        "$|v_y|$",
        # "$v_z$",
        "$P_\\mathrm{dyn}$",
        "$|B|$",
        "$B_x$",
        "$B_y$",
        "$B_z$",
        "$E_x$",
        "$E_y$",
        "$E_z$",
        "$T_\\parallel$",
        "$T_\\perp$",
        "$T_\\perp/T_\\parallel$",
        # "$\\rho$",
        # "$v_x^2$",
        # "$v_y^2$",
        # "$v_z^2$",
    ]
    draw_legend = [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        True,
        False,
        # False,
        # False,
        # True,
    ]
    ylabels = [
        "$\\rho~[\\rho_\\mathrm{pre-jet}]$",
        "$|v|~[|v|_\\mathrm{pre-jet}]$",
        "$v~[|v|_\\mathrm{pre-jet}]$",
        "$P_\\mathrm{dyn}$\n$[P_\\mathrm{dyn,pre-jet}]$",
        "$|B|~[|B|_\\mathrm{pre-jet}]$",
        "$B~[|B|_\\mathrm{pre-jet}]$",
        "$E~[|E|_\\mathrm{pre-jet}]$",
        # "$T~[T_\\mathrm{pre-jet}]$",
        "$T~[MK]$",
        # "$P_\\mathrm{dyn}$\ncontribution",
        "$T_\\perp/T_\\parallel$",
    ]
    plot_index = [0, 1, 2, 2, 3, 4, 5, 5, 5, 6, 6, 6, 7, 7, 8]
    plot_colors = [
        "k",
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        # CB_color_cycle[2],
        "k",
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        CB_color_cycle[0],
        CB_color_cycle[1],
        "k",
        # CB_color_cycle[0],
        # CB_color_cycle[1],
        # CB_color_cycle[2],
    ]

    filenames = os.listdir(wrkdir_DNR + "txts/timeseries/" + folder_suffix)
    filenames = [fname for fname in filenames if "corr" not in fname]

    test_data = np.loadtxt(
        wrkdir_DNR + "txts/timeseries/" + folder_suffix + "/" + filenames[0]
    )

    data_arr = np.zeros(
        (len(filenames), test_data.shape[0], test_data.shape[1]), dtype=float
    )
    for idx, fn in enumerate(filenames):
        data_arr[idx, :, :] = np.loadtxt(
            wrkdir_DNR + "txts/timeseries/" + folder_suffix + "/" + fn
        )

    data_arr2 = np.zeros((len(filenames), 15, test_data.shape[1]), dtype=float)
    prejet_avg_arr = np.zeros((len(filenames), 15), dtype=float)
    for idx, fn in enumerate(filenames):
        data_arr2[idx, 0, :] = data_arr[idx, 0, :]  # Density
        data_arr2[idx, 1, :] = data_arr[idx, 4, :]  # Velocity magnitude
        data_arr2[idx, [2, 3], :] = np.abs(data_arr[idx, [1, 2], :])  # Velocity
        data_arr2[idx, 4, :] = data_arr[idx, 5, :]  # Dynamic pressure
        data_arr2[idx, 5, :] = data_arr[idx, 9, :]  # Magnetic field magnitude
        data_arr2[idx, [6, 7, 8], :] = data_arr[idx, [6, 7, 8], :]  # Magnetic field
        data_arr2[idx, [9, 10, 11], :] = data_arr[
            idx, [10, 11, 12], :
        ]  # Electric field
        data_arr2[idx, [12, 13], :] = data_arr[idx, [14, 15], :]  # Temperature
        data_arr2[idx, 14, :] = (
            data_arr[idx, 15, :] / data_arr[idx, 14, :]
        )  # Temperature aniostropy
        # data_arr2[idx, [15, 16, 17, 18], :] = data_arr[
        #     idx, [16, 17, 18, 19], :
        # ]  # Pdyn contribution
        prejet_avg_arr[idx, 0] = np.nanmean(data_arr[idx, 0, :20])
        prejet_avg_arr[idx, 1] = np.nanmean(data_arr[idx, 4, :20])
        prejet_avg_arr[idx, 2] = np.nanmean(data_arr[idx, 4, :20])
        prejet_avg_arr[idx, 3] = np.nanmean(data_arr[idx, 4, :20])
        prejet_avg_arr[idx, 4] = np.nanmean(data_arr[idx, 5, :20])
        prejet_avg_arr[idx, 5] = np.nanmean(data_arr[idx, 9, :20])
        prejet_avg_arr[idx, 6] = np.nanmean(data_arr[idx, 9, :20])
        prejet_avg_arr[idx, 7] = np.nanmean(data_arr[idx, 9, :20])
        prejet_avg_arr[idx, 8] = np.nanmean(data_arr[idx, 9, :20])
        prejet_avg_arr[idx, 9] = np.nanmean(data_arr[idx, 13, :20])
        prejet_avg_arr[idx, 10] = np.nanmean(data_arr[idx, 13, :20])
        prejet_avg_arr[idx, 11] = np.nanmean(data_arr[idx, 13, :20])
        prejet_avg_arr[idx, 12] = 1
        prejet_avg_arr[idx, 13] = 1
        prejet_avg_arr[idx, 14] = 1

    sea_t_arr = np.arange(-10, 10 + 0.1, 0.5)

    for idx in range(len(filenames)):
        for idx2 in range(len(plot_index)):
            data_arr2[idx, idx2, :] /= prejet_avg_arr[idx, idx2]

    cat_avgs = np.nanmean(data_arr2, axis=0)
    cat_meds = np.nanmedian(data_arr2, axis=0)
    cat_25 = np.percentile(data_arr2, 25, axis=0)
    cat_75 = np.percentile(data_arr2, 75, axis=0)

    fig, ax_list = plt.subplots(
        len(ylabels), 1, figsize=(8, 10), constrained_layout=True
    )

    print("\n" + cat_names[valid_cats.index(folder_suffix)])
    for idx2 in range(len(plot_index)):
        ax = ax_list[plot_index[idx2]]
        print("\n{}".format(plot_labels[idx2]))
        print("Prejet avg: {}".format(np.nanmean(prejet_avg_arr[:, idx2])))
        print("Begin: {}".format(cat_avgs[idx2, 0] / np.nanmean(cat_avgs[idx2, :20])))
        print("Form: {}".format(cat_avgs[idx2, 20] / np.nanmean(cat_avgs[idx2, :20])))
        print("End: {}".format(cat_avgs[idx2, -1] / np.nanmean(cat_avgs[idx2, :20])))
        ax.plot(
            sea_t_arr,
            cat_avgs[idx2, :],
            color=plot_colors[idx2],
            label=plot_labels[idx2],
            linewidth=1.2,
            zorder=2,
        )
        ax.fill_between(
            sea_t_arr,
            cat_25[idx2],
            cat_75[idx2],
            facecolor=plot_colors[idx2],
            alpha=0.2,
            zorder=1,
        )
        if draw_legend[idx2]:
            ax.legend(loc="upper right")

    for idx, ax in enumerate(ax_list):
        ax.grid(zorder=0)
        ax.set_xlim(sea_t_arr[0], sea_t_arr[-1])
        ax.set_ylabel(ylabels[idx])
        ax.label_outer()
    ax_list[-1].set_xlabel("Epoch time [s]")
    ax_list[0].set_title(
        cat_names[valid_cats.index(folder_suffix)] + ", N = {}".format(len(filenames))
    )

    fig.savefig(wrkdir_DNR + "Figs/SEA_new_{}.png".format(folder_suffix), dpi=300)

    plt.close(fig)


def plot_colormap_cut(x0, y0, t0):

    bulkpath = find_bulkpath("AIC")

    rax_labs = [
        "$\\rho$\n$[\\mathrm{cm}^{-3}]$",
        "$v$\n$[\\mathrm{km/s}]$",
        "$P_\\mathrm{dyn}$\n$[\\mathrm{nPa}]$",
        "$B$\n$[\\mathrm{nT}]$",
        "$E$\n$[\\mathrm{mV/m}]$",
        "$T$\n$[\\mathrm{MK}]$",
        "$T_\\perp/T_\\parallel$",
    ]
    var_pars = [
        ["$\\rho$", "proton/vg_rho", "pass", 1e-6, 0, False, "k"],
        ["$v_x$", "proton/vg_v", "x", 1e-3, 1, False, CB_color_cycle[0]],
        ["$v_y$", "proton/vg_v", "y", 1e-3, 1, False, CB_color_cycle[1]],
        ["$v_z$", "proton/vg_v", "z", 1e-3, 1, False, CB_color_cycle[2]],
        ["$|v|$", "proton/vg_v", "magnitude", 1e-3, 1, True, "k"],
        ["$P_\\mathrm{dyn}$", "proton/vg_pdyn", "pass", 1e9, 2, False, "k"],
        [
            "$P_{\\mathrm{dyn},x}$",
            "proton/vg_pdynx",
            "pass",
            1e9,
            2,
            True,
            CB_color_cycle[0],
        ],
        ["$B_x$", "vg_b_vol", "x", 1e9, 3, False, CB_color_cycle[0]],
        ["$B_y$", "vg_b_vol", "y", 1e9, 3, False, CB_color_cycle[1]],
        ["$B_z$", "vg_b_vol", "z", 1e9, 3, False, CB_color_cycle[2]],
        ["$|B|$", "vg_b_vol", "magnitude", 1e9, 3, True, "k"],
        ["$E_x$", "vg_e_vol", "x", 1e3, 4, False, CB_color_cycle[0]],
        ["$E_y$", "vg_e_vol", "y", 1e3, 4, False, CB_color_cycle[1]],
        ["$E_z$", "vg_e_vol", "z", 1e3, 4, False, CB_color_cycle[2]],
        ["$|E|$", "vg_e_vol", "magnitude", 1e3, 4, True, "k"],
        [
            "$T_\\parallel$",
            "proton/vg_t_parallel",
            "pass",
            1e-6,
            5,
            False,
            CB_color_cycle[0],
        ],
        [
            "$T_\\perp$",
            "proton/vg_t_perpendicular",
            "pass",
            1e-6,
            5,
            True,
            CB_color_cycle[1],
        ],
        ["$T_\\perp/T_\\parallel$", "proton/vg_t_anisotropy", "pass", 1, 6, False, "k"],
    ]

    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    # fig = plt.figure(figsize=(16, 8), layout="constrained")

    # gs = fig.add_gridspec(7, 20)

    # ax1 = fig.add_subplot(gs[0:7, 0:9])
    # ax2 = fig.add_subplot(gs[0:7, 9:10])
    # rax_list = [fig.add_subplot(gs[idx : idx + 1, 11:20]) for idx in range(7)]
    fig2, rax_list = plt.subplots(7, 1, figsize=(8, 10), constrained_layout=True)

    fnr0 = int(t0 * 2)
    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
    )
    x_arr = np.arange(x0 * r_e - 30 * 300e3, x0 * r_e + 30 * 300e3 + 1, 300e3)
    data_arr = np.zeros((len(var_pars), x_arr.size), dtype=float)
    for idx in range(x_arr.size):
        for idx2 in range(len(var_pars)):
            data_arr[idx2, idx] = (
                vlsvobj.read_interpolated_variable(
                    var_pars[idx2][1],
                    [x_arr[idx], y0 * r_e, 0],
                    operator=var_pars[idx2][2],
                )
                * var_pars[idx2][3]
            )

    vscale = 1e9
    expression = None
    usesci = 0

    vmax = 1.1

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g, start_points, drawBy0, plaschke_g, leg_g, draw_qperp, vobj, umagten_g, chg, highres_g, bsg, mmsg
    umagten_g = False
    runid_g = "AIC"
    Blines_g = False
    drawBy0 = False
    plaschke_g = False
    leg_g = True
    draw_qperp = False
    chg = True
    highres_g = 3
    bsg = False
    mmsg = True

    global xg, yg, linsg, lineg
    xg = []
    yg = []
    linsg, lineg = None, None

    non_ids = get_jets("AIC", min_duration=1, minsize=4)

    sj_ids_g = []
    non_ids_g = non_ids

    sw_pars = [
        [1e6, 750e3, 3e-9, 0.5e6],
        [1e6, 750e3, 3e-9, 0.5e6],
        [1e6, 750e3, 3e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[2]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw

    streamlines = None

    fluxfile = (
        vlasdir
        + "/2D/AIC/fluxfunction/"
        + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
        + ".bin"
    )
    fluxdir = None
    flux_levels = None
    fluxthick = 0.5
    fluxlines = 5
    filenr_g = fnr0

    pt.plot.plot_colormap(
        axes=ax1,
        # cbaxes=ax1,
        vlsvobj=vlsvobj,
        var="proton/vg_pdyn",
        op=None,
        vmin=0.01,
        vmax=1.2,
        vscale=vscale,
        # cbtitle="",
        # cbtitle="",
        usesci=usesci,
        # scale=3,
        title="t = {} s".format(t0),
        boxre=[0, 20, -10, 10],
        highres=highres_g,
        colormap="grayC",
        tickinterval=4,
        fsaved=False,
        useimshow=True,
        internalcb=True,
        external=ext_jet,
        expression=expression,
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
        streamlines=streamlines,
        streamlinedensity=0.3,
        streamlinecolor="white",
        streamlinethick=0.8,
        fluxfile=fluxfile,
        fluxdir=fluxdir,
        flux_levels=flux_levels,
        fluxthick=fluxthick,
        fluxlines=fluxlines,
    )
    ax1.plot(
        [x_arr[0] / r_e, x_arr[-1] / r_e],
        [y0, y0],
        color="red",
        linestyle="dashed",
        linewidth=1.5,
    )

    for idx, ax in enumerate(var_pars):
        ax = rax_list[var_pars[idx][4]]
        ax.plot(
            x_arr / r_e,
            data_arr[idx, :],
            color=var_pars[idx][6],
            linewidth=1.2,
            label=var_pars[idx][0],
        )
        if var_pars[idx][5]:
            ax.legend(loc="upper right", fontsize=12)

    for idx, ax in enumerate(rax_list):
        ax.set_xlim(x_arr[0] / r_e, x_arr[-1] / r_e)
        ax.set_ylabel(rax_labs[idx], labelpad=10, fontsize=20)
        ax.grid()
        ax.tick_params(labelsize=12)
        ax.label_outer()
    rax_list[-1].set_xlabel("x~[$R_\\mathrm{E}$]", labelpad=10, fontsize=20)
    rax_list[0].set_title(
        "t = {} s, y = {}".format(t0, y0) + " $R_\\mathrm{E}$", pad=10, fontsize=20
    )

    for idx, fig in enumerate([fig1, fig2]):
        fig.savefig(wrkdir_DNR + "Figs/colormap_cut_{}.pdf".format(idx + 1), dpi=300)
        # fig.savefig(wrkdir_DNR + "Figs/colormap_cut_{}.png".format(idx + 1), dpi=300)
        plt.close(fig)


def plot_category_SEA(runid="AIC", folder_suffix="jets", delta=False):

    valid_cats = [
        "jets_all",
        "jets_qpar_before",
        "jets_qpar_after",
        "jets_qpar_fb",
        "jets_qperp_rd",
        "jets_qperp_after",
        "jets_qperp_inter",
    ]
    cat_names = [
        "All",
        "Dusk $Q\\parallel$",
        "Dusk $Q\\perp$",
        "Dusk FB",
        "Dawn RD",
        "Dawn $Q\\parallel$",
        "Dawn young FS",
    ]

    plot_labels = [
        None,
        "$v_x$",
        "$v_y$",
        "$v_z$",
        "$|v|$",
        "$P_\\mathrm{dyn}$",
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
        "$\\rho$",
        "$v_x^2$",
        "$v_y^2$",
        "$v_z^2$",
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
        False,
        False,
        False,
        True,
    ]
    ylabels = [
        # "$\\rho~[\\mathrm{cm}^{-3}]$",
        # "$v~[\\mathrm{km/s}]$",
        # "$P_\\mathrm{dyn}~[\\mathrm{nPa}]$",
        # "$B~[\\mathrm{nT}]$",
        # "$E~[\\mathrm{mV/m}]$",
        # "$T~[\\mathrm{MK}]$",
        "$\\rho~[\\rho_\\mathrm{pre-jet}]$",
        "$v~[v_\\mathrm{pre-jet}]$",
        "$P_\\mathrm{dyn}~[P_{dyn,\\mathrm{pre-jet}}]$",
        "$B~[B_\\mathrm{pre-jet}]$",
        "$E~[E_\\mathrm{pre-jet}]$",
        "$T~[T_\\mathrm{pre-jet}]$",
        "$P_\\mathrm{dyn}$\ncontribution",
    ]
    if delta:
        for idx in range(len(ylabels)):
            ylabels[idx] = "$\\delta " + ylabels[idx][1:]
        for idx in range(len(plot_labels)):
            if plot_labels[idx]:
                plot_labels[idx] = "$\\delta " + plot_labels[idx][1:]
    plot_index = [0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6]
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
        "k",
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
    ]

    filenames = os.listdir(wrkdir_DNR + "txts/timeseries/" + folder_suffix)
    filenames = [fname for fname in filenames if "corr" not in fname]

    test_data = np.loadtxt(
        wrkdir_DNR + "txts/timeseries/" + folder_suffix + "/" + filenames[0]
    )

    data_arr = np.zeros(
        (len(filenames), test_data.shape[0], test_data.shape[1]), dtype=float
    )
    for idx, fn in enumerate(filenames):
        data_arr[idx, :, :] = np.loadtxt(
            wrkdir_DNR + "txts/timeseries/" + folder_suffix + "/" + fn
        )

    sea_t_arr = np.arange(-10, 10 + 0.1, 0.5)

    ts_avgs = np.nanmean(data_arr, axis=2)

    for idx in range(len(filenames)):
        for idx2 in range(len(plot_index)):
            if idx2 in [1, 2, 3]:
                prejet_avg = np.nanmean(data_arr[idx, 4, :20])
            elif idx2 in [6, 7, 8]:
                prejet_avg = np.nanmean(data_arr[idx, 9, :20])
            elif idx2 in [10, 11, 12]:
                prejet_avg = np.nanmean(data_arr[idx, 13, :20])
            elif idx2 in [14, 15]:
                prejet_avg = np.nanmean(
                    data_arr[idx, 14, :20] + 2 * data_arr[idx, 15, :20]
                )
            elif idx2 in [16, 17, 18, 19]:
                prejet_avg = 1
            else:
                prejet_avg = np.nanmean(data_arr[idx, idx2, :20])
            # data_arr[idx, idx2, :] -= ts_avgs[idx, idx2]
            if delta:
                data_arr[idx, idx2, :] -= np.nanmean(data_arr[idx, idx2, :20])
            data_arr[idx, idx2, :] /= prejet_avg

    cat_avgs = np.nanmean(data_arr, axis=0)
    cat_meds = np.nanmedian(data_arr, axis=0)
    cat_25 = np.percentile(data_arr, 25, axis=0)
    cat_75 = np.percentile(data_arr, 75, axis=0)

    fig, ax_list = plt.subplots(
        len(ylabels), 1, figsize=(7, 9), constrained_layout=True
    )

    for idx2 in range(len(plot_index)):
        ax = ax_list[plot_index[idx2]]
        # for idx in range(len(filenames)):
        #     ax.plot(
        #         sea_t_arr,
        #         data_arr[idx, idx2, :],
        #         color=plot_colors[idx2],
        #         alpha=0.2,
        #         linewidth=0.4,
        #         zorder=0,
        #     )
        ax.plot(
            sea_t_arr,
            cat_avgs[idx2, :],
            color=plot_colors[idx2],
            label=plot_labels[idx2],
            linewidth=1.2,
            zorder=2,
        )
        ax.fill_between(
            sea_t_arr,
            cat_25[idx2],
            cat_75[idx2],
            facecolor=plot_colors[idx2],
            alpha=0.2,
            zorder=1,
        )
        if draw_legend[idx2]:
            ax.legend(loc="upper right")

    for idx, ax in enumerate(ax_list):
        ax.grid(zorder=0)
        ax.set_xlim(sea_t_arr[0], sea_t_arr[-1])
        ax.set_ylabel(ylabels[idx])
    ax_list[-1].set_xlabel("Epoch time [s]")
    ax_list[0].set_title(
        # folder_suffix.replace("_", " ").title() + ", N = {}".format(len(filenames))
        cat_names[valid_cats.index(folder_suffix)]
        + ", N = {}".format(len(filenames))
    )

    fig.savefig(
        wrkdir_DNR + "Figs/SEA_{}_delta{}.png".format(folder_suffix, delta), dpi=300
    )

    plt.close(fig)


def plot_category_correlation(runid, folder_suffix="jets"):

    corr_labels = ["$P_\\mathrm{dyn}$", "$\\rho$", "$v_x^2$", "$v_y^2$", "$v_z^2$"]
    # corr_vars = [pd_lp, rho_lp, vx_lp**2, vy_lp**2, vz_lp**2]
    filenames = os.listdir(wrkdir_DNR + "txts/timeseries/" + folder_suffix)
    filenames = [fname for fname in filenames if "corr" in fname]
    corr_mat = np.zeros(
        (len(corr_labels), len(corr_labels), len(filenames)), dtype=float
    )
    for idx, fn in enumerate(filenames):
        corr_mat[:, :, idx] = np.loadtxt(
            wrkdir_DNR + "txts/timeseries/" + folder_suffix + "/" + fn
        )

    corr_meds = np.median(corr_mat, axis=-1)
    corr_25 = np.percentile(corr_mat, 25, axis=-1)
    corr_75 = np.percentile(corr_mat, 75, axis=-1)

    fig, ax = plt.subplots()
    im = ax.imshow(corr_meds, cmap="vik", vmin=-1, vmax=1)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(corr_labels)), labels=corr_labels)
    ax.set_yticks(np.arange(len(corr_labels)), labels=corr_labels)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(corr_labels)):
        for j in range(len(corr_labels)):
            textstr = "{vala}_{{-{valm}}}^{{+{valp}}}".format(
                vala=round(corr_meds[i, j], 2),
                valm=round(corr_meds[i, j] - corr_25[i, j], 2),
                valp=round(corr_75[i, j] - corr_meds[i, j], 2),
            )
            text = ax.text(
                j,
                i,
                "$" + textstr + "$",
                ha="center",
                va="center",
                color="w",
            )
            # text = ax.text(
            #     j,
            #     i,
            #     round(corr_mat[i, j], 2),
            #     ha="center",
            #     va="center",
            #     color="w",
            # )

    ax.set_title("Variable cross-correlation, N = {}".format(len(filenames)))
    ax.spines[:].set_visible(False)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    fig.tight_layout()
    figdir = wrkdir_DNR + "Figs/"
    fig.savefig(
        figdir + "jet_correlation_{}_{}.png".format(runid, folder_suffix),
        dpi=300,
    )
    plt.close(fig)


def all_cats_timeseries_script(n_processes=1, draw=True):

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
        "jets_qpar_before",
        "jets_qpar_after",
        "jets_qpar_fb",
        "jets_qperp_rd",
        "jets_all",
        "jets_qperp_after",
        "jets_qperp_inter",
    ]
    tmins = [391, 470, 430, 430, 391, 600, 509]
    tmaxs = [426, 800, 470, 470, 800, 800, 600]

    for idx in range(len(folder_suffixes)):
        plot_timeseries_at_jets(
            "AIC",
            boxre=boxres[idx],
            tmin=tmins[idx],
            tmax=tmaxs[idx],
            folder_suffix=folder_suffixes[idx],
            skip=False,
            minduration=1,
            minsize=4,
            pdavg=False,
            n_processes=n_processes,
            draw=draw,
        )

    # plot_timeseries_at_jets(
    #     "AIC",
    #     boxre=[8, 16, 3, 17],
    #     tmin=391,
    #     tmax=426,
    #     folder_suffix="jets_qpar_before",
    #     skip=False,
    #     minduration=1,
    #     minsize=4,
    #     pdavg=False,
    # )
    # plot_timeseries_at_jets(
    #     "AIC",
    #     boxre=[8, 20, 3, 17],
    #     tmin=470,
    #     tmax=800,
    #     folder_suffix="jets_qpar_after",
    #     skip=False,
    #     minduration=1,
    #     minsize=4,
    #     pdavg=False,
    # )
    # plot_timeseries_at_jets(
    #     "AIC",
    #     boxre=[8, 16, 3, 17],
    #     tmin=430,
    #     tmax=470,
    #     folder_suffix="jets_qpar_fb",
    #     skip=False,
    #     minduration=1,
    #     minsize=4,
    #     pdavg=False,
    # )
    # plot_timeseries_at_jets(
    #     "AIC",
    #     boxre=[8, 16, -17, 0],
    #     tmin=430,
    #     tmax=470,
    #     folder_suffix="jets_qperp_rd",
    #     skip=False,
    #     minduration=1,
    #     minsize=4,
    #     pdavg=False,
    # )
    # plot_timeseries_at_jets(
    #     "AIC",
    #     boxre=[8, 20, -17, 17],
    #     tmin=391,
    #     tmax=800,
    #     folder_suffix="jets_all",
    #     skip=False,
    #     minduration=1,
    #     minsize=4,
    #     pdavg=False,
    # )
    # plot_timeseries_at_jets(
    #     "AIC",
    #     boxre=[8, 20, -17, -3],
    #     tmin=600,
    #     tmax=800,
    #     folder_suffix="jets_qperp_after",
    #     skip=False,
    #     minduration=1,
    #     minsize=4,
    #     pdavg=False,
    # )
    # plot_timeseries_at_jets(
    #     "AIC",
    #     boxre=[8, 16, -17, -3],
    #     tmin=509,
    #     tmax=600,
    #     folder_suffix="jets_qperp_inter",
    #     skip=False,
    #     minduration=1,
    #     minsize=4,
    #     pdavg=False,
    # )


def plot_timeseries_at_jets(
    runid,
    boxre=None,
    tmin=None,
    tmax=None,
    folder_suffix="jets",
    skip=False,
    minduration=0,
    minsize=0,
    pdavg=True,
    n_processes=1,
    draw=True,
):

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

        if pdavg:
            plott0 = max(391, t0 - 10)
            plott1 = min(1000, t0 + 10)
        else:
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
            pdavg=pdavg,
            pdx=True,
            # prefix="{}".format(n1),
            dirprefix="{}/".format(folder_suffix),
            skip=skip,
            jett0=t0,
            n_processes=n_processes,
            draw=draw,
        )


def plot_vdf_at_jets(runid, boxre=None, skip=False, pdmin=0.01):

    # non_ids = []

    bulkpath = find_bulkpath(runid)
    vobj = pt.vlsvfile.VlsvReader(
        bulkpath + "bulk.{}.vlsv".format(str(int(401 * 2)).zfill(7))
    )
    ci = vobj.read_variable("CellID")
    fsaved = vobj.read_variable("vg_f_saved")
    vdf_cells = ci[fsaved == 1]
    if boxre:
        restr_ci = restrict_area(vobj, boxre)
        restr_vdf_ci = np.intersect1d(restr_ci, vdf_cells)
    else:
        restr_vdf_ci = vdf_cells

    for n1 in range(6000):
        try:
            props = PropReader(str(n1).zfill(5), runid, transient="jet")
        except:
            continue

        jet_times = props.get_times()
        jet_cells = props.get_cells()
        jet_is_on_vdf = [np.in1d(c, restr_vdf_ci).any() for c in jet_cells]
        xmean = props.read("x_mean")
        ymean = props.read("y_mean")

        vdf_times = np.array(jet_times)[jet_is_on_vdf]
        if vdf_times.size > 0:
            first_vdf_time = vdf_times[0]
            last_vdf_time = vdf_times[-1]
            x0 = xmean[jet_is_on_vdf][0]
            y0 = ymean[jet_is_on_vdf][0]

            if last_vdf_time < 400 or first_vdf_time > 1000:
                continue

            print(
                "Plotting VDF at ({:.3f},{:.3f}) from t = {} to {} s, jet ID = {}".format(
                    x0,
                    y0,
                    max(400, first_vdf_time - 10),
                    min(1000, last_vdf_time + 10),
                    n1,
                )
            )

            pos_vdf_plotter(
                runid,
                x0,
                y0,
                max(400, first_vdf_time - 10),
                min(1000, last_vdf_time + 10),
                xyz=True,
                boxwidth=3000e3,
                rboxw=3,
                pdmax=2.0,
                prefix="jets/{}/".format(n1),
                print_unicorn=True,
                skip=skip,
                pdmin=pdmin,
            )

        # non_ids.append(n1)


def pos_vdf_plotter(
    runid,
    x,
    y,
    t0,
    t1,
    skip=False,
    xyz=False,
    boxwidth=2000e3,
    pdmax=1.0,
    pdmin=0.01,
    ncont=5,
    rboxw=2,
    boxre=None,
    fmin=1e-10,
    fmax=1e-4,
    prefix="",
    print_unicorn=False,
):
    runids = ["AGF", "AIA", "AIC"]
    # pdmax = [1.0, 1.0, 1.0][runids.index(runid)]
    bulkpath = find_bulkpath(runid)

    global xg, yg

    xg = []
    yg = []

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g, x0, y0, plaschke_g, drawBy0, linsg, draw_qperp, leg_g
    drawBy0 = True
    draw_qperp = False
    plaschke_g = False
    runid_g = runid

    linsg = False
    leg_g = False
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
        [1e6, 750e3, 3e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runids.index(runid)]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw

    vobj = pt.vlsvfile.VlsvReader(
        bulkpath + "bulk.{}.vlsv".format(str(int(t0 * 2)).zfill(7))
    )
    cellid = vobj.get_cellid([x * r_e, y * r_e, 0 * r_e])
    vdf_cellid = getNearestCellWithVspace(vobj, cellid)

    x_re, y_re, z_re = vobj.get_cell_coordinates(vdf_cellid) / r_e

    outdir = wrkdir_DNR + "VDFs/{}/{}x_{:.3f}_y_{:.3f}_t0_{}_t1_{}_xyz{}".format(
        runid, prefix, x_re, y_re, t0, t1, xyz
    )

    for t in np.arange(t0, t1 + 0.1, 0.5):
        print("t = {}s".format(t))
        fnr = int(t * 2)
        if skip and os.path.isfile(outdir + "/{}.png".format(fnr)):
            continue
        filenr_g = fnr
        vobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
        )
        cellid = vobj.get_cellid([x * r_e, y * r_e, 0 * r_e])
        vdf_cellid = getNearestCellWithVspace(vobj, cellid)
        v = vobj.read_variable("proton/vg_v", cellids=vdf_cellid) * 1e-3
        vth = vobj.read_variable("proton/vg_thermalvelocity", cellids=vdf_cellid) * 1e-3

        if not xyz:
            B = vobj.read_variable("vg_b_vol", cellids=vdf_cellid)
            b = B / np.linalg.norm(B)
            vpar = np.dot(v, b)
            vperp1 = np.cross(b, v)
            vperp2 = np.cross(b, np.cross(b, v))

        x_re, y_re, z_re = vobj.get_cell_coordinates(vdf_cellid) / r_e

        x0 = x_re
        y0 = y_re

        if type(boxre) is not list:
            boxre = [x_re - rboxw, x_re + rboxw, y_re - rboxw, y_re + rboxw]

        fig, ax_list = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)

        pt.plot.plot_colormap(
            axes=ax_list[0][0],
            vlsvobj=vobj,
            var="proton/vg_Pdyn",
            vmin=pdmin,
            vmax=pdmax,
            vscale=1e9,
            cbtitle="$P_\\mathrm{dyn}$ [nPa]",
            usesci=0,
            boxre=boxre,
            # internalcb=True,
            # lin=1,
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
            streamlines="vg_b_vol",
            streamlinedensity=0.4,
            streamlinecolor="white",
            streamlinethick=1,
            streamlinestartpoints=np.array([[x0, y0]]),
        )
        ax_list[0][0].axhline(y_re, linestyle="dashed", linewidth=0.6, color="k")
        ax_list[0][0].axvline(x_re, linestyle="dashed", linewidth=0.6, color="k")

        if xyz:
            pt.plot.plot_vdf(
                axes=ax_list[0][1],
                vlsvobj=vobj,
                cellids=[vdf_cellid],
                colormap="batlow",
                bvector=1,
                xy=1,
                # bpara=1,
                slicethick=0,
                box=[-boxwidth, boxwidth, -boxwidth, boxwidth],
                # internalcb=True,
                setThreshold=1e-15,
                scale=1.3,
                fmin=fmin,
                fmax=fmax,
                contours=ncont,
            )
            ax_list[0][1].plot(v[0], v[1], "x", color="red")
            ax_list[0][1].add_patch(
                plt.Circle(
                    (v[0], v[1]), radius=vth, fill=False, ec="red", linestyle="dashed"
                )
            )
            pt.plot.plot_vdf(
                axes=ax_list[1][0],
                vlsvobj=vobj,
                cellids=[vdf_cellid],
                colormap="batlow",
                bvector=1,
                xz=1,
                # bpara1=1,
                slicethick=0,
                box=[-boxwidth, boxwidth, -boxwidth, boxwidth],
                # internalcb=True,
                setThreshold=1e-15,
                scale=1.3,
                fmin=fmin,
                fmax=fmax,
                contours=ncont,
            )
            ax_list[1][0].plot(v[0], v[2], "x", color="red")
            ax_list[1][0].add_patch(
                plt.Circle(
                    (v[0], v[2]), radius=vth, fill=False, ec="red", linestyle="dashed"
                )
            )
            pt.plot.plot_vdf(
                axes=ax_list[1][1],
                vlsvobj=vobj,
                cellids=[vdf_cellid],
                colormap="batlow",
                bvector=1,
                yz=1,
                # bperp=1,
                slicethick=0,
                box=[-boxwidth, boxwidth, -boxwidth, boxwidth],
                # internalcb=True,
                setThreshold=1e-15,
                scale=1.3,
                fmin=fmin,
                fmax=fmax,
                contours=ncont,
            )
            ax_list[1][1].plot(v[1], v[2], "x", color="red")
            ax_list[1][1].add_patch(
                plt.Circle(
                    (v[1], v[2]), radius=vth, fill=False, ec="red", linestyle="dashed"
                )
            )
        else:
            pt.plot.plot_vdf(
                axes=ax_list[0][1],
                vlsvobj=vobj,
                cellids=[vdf_cellid],
                colormap="batlow",
                # bvector=1,
                # xy=1,
                bpara=1,
                slicethick=0,
                box=[-boxwidth, boxwidth, -boxwidth, boxwidth],
                # internalcb=True,
                setThreshold=1e-15,
                scale=1.3,
                fmin=1e-10,
                fmax=1e-4,
                contours=ncont,
            )
            pt.plot.plot_vdf(
                axes=ax_list[1][0],
                vlsvobj=vobj,
                cellids=[vdf_cellid],
                colormap="batlow",
                # bvector=1,
                # xz=1,
                bpara1=1,
                slicethick=0,
                box=[-boxwidth, boxwidth, -boxwidth, boxwidth],
                # internalcb=True,
                setThreshold=1e-15,
                scale=1.3,
                fmin=1e-10,
                fmax=1e-4,
                contours=ncont,
            )
            pt.plot.plot_vdf(
                axes=ax_list[1][1],
                vlsvobj=vobj,
                cellids=[vdf_cellid],
                colormap="batlow",
                # bvector=1,
                # yz=1,
                bperp=1,
                slicethick=0,
                box=[-boxwidth, boxwidth, -boxwidth, boxwidth],
                # internalcb=True,
                setThreshold=1e-15,
                scale=1.3,
                fmin=1e-10,
                fmax=1e-4,
                contours=ncont,
            )

        # plt.subplots_adjust(wspace=1, hspace=1)

        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except OSError:
                pass

        fig.suptitle(
            "Run: {}, x: {:.3f}, y: {:.3f}, Time: {}s".format(runid, x_re, y_re, t)
        )
        if print_unicorn:
            events = eventfile_read(runid, fnr)
            for event in events:
                if np.in1d(event, [vdf_cellid]).any():
                    fig.suptitle(
                        "Run: {}, x: {:.3f}, y: {:.3f}, Time: {}s, UNICORN".format(
                            runid, x_re, y_re, t
                        )
                    )
                    break

        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except OSError:
                pass
        fig.savefig(outdir + "/{}.png".format(fnr))
        plt.close(fig)

    return None
