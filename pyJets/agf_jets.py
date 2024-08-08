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
    By0=True,
    leg=True,
    track_jets=True,
    qperp=True,
    linestartstop=[],
):
    var = "proton/vg_Pdyn"
    vscale = 1e9
    vmax = pdynmax
    runids = ["AGF", "AIA", "AIC"]

    if len(pointsx) != len(pointsy):
        print("x and y must have same length!")
        return 1

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g, start_points, drawBy0, plaschke_g, leg_g, draw_qperp
    runid_g = runid
    Blines_g = blines
    drawBy0 = By0
    plaschke_g = False
    leg_g = leg
    draw_qperp = qperp

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
        non_ids = get_jets(runid)
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

    alpha = np.arctan2(y1 - y0, x1 - x0)
    dx = dr * np.cos(alpha)
    nx = 1 + int((x1 - x0) / dx)
    x_arr = np.linspace(x0, x1, nx) * r_e
    y_arr = np.linspace(y0, y1, nx) * r_e
    n_arr = np.arange(x_arr.size)

    fnr0 = int(t0 * 2)
    data_arr = np.zeros((len(var_list), x_arr.size), dtype=float)
    vlsvobj = pt.vlsvfile.VlsvReader(
        bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
    )

    for idx in range(x_arr.size):

        for idx2, var in enumerate(var_list):
            data_arr[idx2, idx] = (
                vlsvobj.read_interpolated_variable(
                    var, [x_arr[idx], y_arr[idx], 0], operator=ops[idx2]
                )
                * scales[idx2]
            )

    fig, ax_list = plt.subplots(
        len(ylabels), 1, sharex=True, figsize=(6, 8), constrained_layout=True
    )
    ax_list[0].set_title(
        "Run: {}, $(x,y)_0$: {}, $(x,y)_1$: {}".format(runid, (x0, y0), (x1, y1))
    )
    for idx in range(len(var_list)):
        ax = ax_list[plot_index[idx]]
        for vline in vlines:
            ax.axvline(vline, linestyle="dashed", linewidth=0.6)
        ax.plot(n_arr, data_arr[idx], color=plot_colors[idx], label=plot_labels[idx])
        if idx == 5 and pdx:
            pdynx = (
                m_p * data_arr[0] * 1e6 * data_arr[1] * 1e3 * data_arr[1] * 1e3 * 1e9
            )
            ax.plot(
                n_arr,
                pdynx,
                color=CB_color_cycle[0],
                label="$P_{\mathrm{dyn},x}$",
            )
        ax.set_xlim(n_arr[0], n_arr[-1])
        if draw_legend[idx]:
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax_list[-1].set_xlabel("Point along cut")
    for idx, ax in enumerate(ax_list):
        ax.grid()
        ax.set_ylabel(ylabels[idx])
        ax.axvline(t0, linestyle="dashed")
    # plt.tight_layout()
    figdir = wrkdir_DNR + "Figs/cuts/"
    txtdir = wrkdir_DNR + "txts/cuts/"
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


def VSC_timeseries(
    runid,
    x0,
    y0,
    t0,
    t1,
    pdavg=False,
    filt=None,
    pdx=False,
    delta=None,
    vlines=[],
    mva=False,
    mva_diag=False,
    grain=1,
    maxwidth=None,
    cutoff=0.9,
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
    if delta:
        for idx in range(len(ylabels)):
            ylabels[idx] = "$\\delta " + ylabels[idx][1:]
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
    if filt:
        sos = butter(10, filt, "lowpass", fs=2, output="sos")

    t_arr = np.arange(t0, t1 + 0.1, 0.5)
    fnr0 = int(t0 * 2)
    fnr_arr = np.arange(fnr0, int(t1 * 2) + 1, dtype=int)
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
        # try:
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
        # except:
        #     print("Something went wrong!")
        #     data_arr[:, idx] = np.nan

    if filt:
        for idx in range(len(var_list)):
            data_arr[idx, :] = sosfilt(sos, data_arr[idx, :])

    if mva:
        Bdata = deepcopy(data_arr[6:9, :])
        vdata = deepcopy(data_arr[1:4, :])
        Edata = deepcopy(data_arr[10:13, :])
        eigenvecs = MVA(Bdata)
        print(np.linalg.norm(data_arr[6:9, :].T[0]))
        print(Bdata.shape)
        print("Minimum Variance direction: {}".format(eigenvecs[0]))
        for idx in range(3):
            data_arr[idx + 6, :] = np.dot(Bdata.T, eigenvecs[idx, :])
            data_arr[idx + 1, :] = np.dot(vdata.T, eigenvecs[idx, :])
            data_arr[idx + 10, :] = np.dot(Edata.T, eigenvecs[idx, :])
        print(np.linalg.norm(data_arr[6:9, :].T[0]))
        plot_labels[1:4] = ["$v_N$", "$v_M$", "$v_L$"]
        plot_labels[6:9] = ["$B_N$", "$B_M$", "$B_L$"]
        plot_labels[10:13] = ["$E_N$", "$E_M$", "$E_L$"]

    fig, ax_list = plt.subplots(
        len(ylabels), 1, sharex=True, figsize=(6, 8), constrained_layout=True
    )
    ax_list[0].set_title("Run: {}, $x_0$: {}, $y_0$: {}".format(runid, x0, y0))
    for idx in range(len(var_list)):
        ax = ax_list[plot_index[idx]]
        for vline in vlines:
            ax.axvline(vline, linestyle="dashed", linewidth=0.6)
        if delta:
            ax.plot(
                t_arr,
                data_arr[idx] - uniform_filter1d(data_arr[idx], size=delta),
                color=plot_colors[idx],
                label=plot_labels[idx],
            )
        else:
            ax.plot(
                t_arr, data_arr[idx], color=plot_colors[idx], label=plot_labels[idx]
            )
        if idx == 5 and pdavg and not delta:
            ax.plot(
                t_arr,
                2 * tavg_arr,
                color=CB_color_cycle[0],
                linestyle="dashed",
                label="$2\\langle P_\mathrm{dyn}\\rangle$",
            )
        elif idx == 5 and pdx:
            pdynx = (
                m_p * data_arr[0] * 1e6 * data_arr[1] * 1e3 * data_arr[1] * 1e3 * 1e9
            )
            if delta:
                ax.plot(
                    t_arr,
                    pdynx - uniform_filter1d(pdynx, size=delta),
                    color=CB_color_cycle[0],
                    label="$P_{\mathrm{dyn},x}$",
                )
            else:
                ax.plot(
                    t_arr,
                    pdynx,
                    color=CB_color_cycle[0],
                    label="$P_{\mathrm{dyn},x}$",
                )
        ax.set_xlim(t_arr[0], t_arr[-1])
        if draw_legend[idx]:
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
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
        figdir
        + "{}_x{}_y{}_t0{}_t1{}_delta{}_mva{}.png".format(
            runid, x0, y0, t0, t1, delta, mva
        ),
        dpi=300,
    )
    np.savetxt(
        txtdir
        + "{}_x{}_y{}_t0{}_t1{}_delta{}_mva{}.txt".format(
            runid, x0, y0, t0, t1, delta, mva
        ),
        data_arr,
    )
    plt.close(fig)

    if mva_diag:
        Bdata = deepcopy(data_arr[6:9, :])
        dt = 0.5
        window_center = np.arange(0, t_arr.size, grain, dtype=int)
        window_halfwidth = np.arange(10, int(t_arr.size / 2), grain, dtype=int)
        window_size = (window_halfwidth * 2 * dt).astype(int)
        print(
            "Window center size: {}, window halfwidth size: {}, Time arr grain size: {}".format(
                window_center.size, window_halfwidth.size, t_arr[0::grain].size
            )
        )
        diag_data = np.empty((window_center.size, window_halfwidth.size), dtype=float)
        diag_vec_data = np.empty(
            (window_center.size, window_halfwidth.size, 3), dtype=float
        )
        diag_maxvec_data = np.empty(
            (window_center.size, window_halfwidth.size, 3), dtype=float
        )
        diag2_data = np.empty((window_center.size, window_halfwidth.size), dtype=float)

        for idx2 in range(window_center.size):
            for idx3 in range(window_halfwidth.size):
                start_id = max(window_center[idx2] - window_halfwidth[idx3], 0)
                stop_id = min(
                    window_center[idx2] + window_halfwidth[idx3] + 1, t_arr.size
                )
                print(
                    "Window center: {}, window halfwidth: {}, start id: {}, stop id: {}".format(
                        window_center[idx2], window_halfwidth[idx3], start_id, stop_id
                    )
                )
                eigvals, eigvecs = MVA(
                    Bdata[:, start_id:stop_id], eigvals=True, prnt=False
                )
                diag_data[idx2, idx3] = eigvals[2] - eigvals[0]
                diag_vec_data[idx2, idx3, :] = eigvecs[0] * np.sign(eigvecs[0][0])
                diag_maxvec_data[idx2, idx3, :] = eigvecs[2]
                diag2_data[idx2, idx3] = eigvals[2] - eigvals[1]

        fig, ax = plt.subplots(5, 1, figsize=(8, 15), constrained_layout=True)
        im = ax[0].pcolormesh(
            t_arr[0::grain],
            window_size,
            diag_data.T,
            shading="gouraud",
            cmap="hot_desaturated",
        )
        mva_cutoff = cutoff * np.max(diag_data)
        ax[0].contour(
            t_arr[0::grain], window_size, diag_data.T, [mva_cutoff], colors=["k"]
        )
        plt.colorbar(im, ax=ax[0])
        ax[0].set_ylabel("Window width [s]")
        ax[0].set_title("$\\lambda_3-\\lambda_1$")

        im = ax[1].pcolormesh(
            t_arr[0::grain],
            window_size,
            diag2_data.T,
            shading="gouraud",
            cmap="hot_desaturated",
        )
        ax[1].contour(
            t_arr[0::grain], window_size, diag_data.T, [mva_cutoff], colors=["k"]
        )
        plt.colorbar(im, ax=ax[1])
        ax[1].set_ylabel("Window width [s]")
        ax[1].set_title("$\\lambda_3-\\lambda_2$")

        for idx in range(3):
            im = ax[idx + 2].pcolormesh(
                t_arr[0::grain],
                window_size,
                diag_vec_data[:, :, idx].T,
                shading="gouraud",
                cmap="vik",
                vmin=-1,
                vmax=1,
            )
            ax[idx + 2].contour(
                t_arr[0::grain], window_size, diag_data.T, [mva_cutoff], colors=["k"]
            )
            plt.colorbar(im, ax=ax[idx + 2])
            ax[idx + 2].set_ylabel("Window width [s]")
            ax[idx + 2].set_title(["$n_x$", "$n_y$", "$n_z$"][idx])

        ax[-1].set_xlabel("Window center")
        fig.savefig(
            wrkdir_DNR
            + "Figs/vlas_mva_diag/{}_x{}_y{}_diag_mva.png".format(runid, x0, y0),
            dpi=150,
        )
        plt.close(fig)

        CmeshCW, WmeshCW = np.meshgrid(window_center, window_size)

        if maxwidth:
            indcs = np.where(
                np.logical_and(WmeshCW < maxwidth, diag_data.T >= mva_cutoff)
            )
        else:
            indcs = np.where(diag_data.T == np.max(diag_data.T))
        print(indcs)
        if indcs[0].size == 1:
            i, j = np.array(indcs).flatten()
        else:
            i, j = np.array(indcs).T[0]

        print("\nMaxvec: {}\n".format(diag_maxvec_data[j, i, :]))

        txtarr = [
            x0,
            y0,
            diag_vec_data[j, i, 0],
            diag_vec_data[j, i, 1],
            diag_vec_data[j, i, 2],
        ]

        np.savetxt(
            wrkdir_DNR + "vlas_pos_mva/{}_x{}_y{}.txt".format(runid, x0, y0), txtarr
        )

        return (diag_vec_data[j, i, :], t_arr[0::grain][j], window_size[i])


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


def plot_fft(
    x0,
    x1,
    y0,
    y1,
    t0,
    t1,
    runid="AGF",
    txt=False,
    draw=True,
    intpol=False,
):
    dr = 300e3 / r_e
    dr_km = 300

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
    t_real_range = np.zeros_like(t_range)
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

    data_arr = np.zeros((xplot_list.size, t_range.size), dtype=float)
    vt_arr = np.ones((xplot_list.size, t_range.size), dtype=float)
    dx_arr = (x1 - x0) * vt_arr
    dy_arr = (y1 - y0) * vt_arr

    figdir = wrkdir_DNR + "Figs/fft/"
    txtdir = wrkdir_DNR + "txts/fft/"

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
            treal = vlsvobj.read_parameter("t")
            t_real_range[idx] = treal
            if intpol:
                data_arr[:, idx] = [
                    vlsvobj.read_interpolated_variable("proton/vg_pdyn", coords[idx3])
                    for idx3 in range(xlist.size)
                ]
            else:
                data_arr[:, idx] = vlsvobj.read_variable(
                    "proton/vg_pdyn", cellids=cellids
                )

    for idx2 in range(len(xplot_list)):
        data_arr[idx2, :] = np.interp(t_range, t_real_range, data_arr[idx2, :])

    fft_arr = np.real(rfft2(data_arr))

    if draw:
        fig, ax = plt.subplots(
            1, 1, figsize=(10, 10), sharex=True, sharey=True, constrained_layout=True
        )
        fig.suptitle(
            "Run: {}, x0: {}, y0: {}, x1: {}, y1: {}".format(runid, x0, y0, x1, y1),
            fontsize=28,
        )
        ax.tick_params(labelsize=20)
        im = ax.pcolormesh(
            # XmeshXY,
            # YmeshXY,
            fft_arr.T,
            shading="nearest",
            cmap="batlow",
            rasterized=True,
        )
        # if idx == 1:
        #     cb_list.append(fig.colorbar(im_list[idx], ax=ax, extend="max"))
        #     cb_list[idx].cmap.set_over("red")
        # else:
        cb = fig.colorbar(im, ax=ax)
        # cb_list.append(fig.colorbar(im_list[idx], ax=ax))
        cb.ax.tick_params(labelsize=20)
        # ax.contour(XmeshXY, YmeshXY, Tcore_arr, [3], colors=[CB_color_cycle[1]])
        # ax.contour(XmeshXY, YmeshXY, mmsx_arr, [1.0], colors=[CB_color_cycle[4]])
        ax.set_xlim(data_arr[0, 0], data_arr[-1, 0])
        ax.set_ylim(data_arr[0, 0], data_arr[0, -1])
        # ax.set_xlabel(xlab, fontsize=24, labelpad=10)
        # ax.axhline(t0, linestyle="dashed", linewidth=0.6)
        # ax.axvline(x0, linestyle="dashed", linewidth=0.6)
        # ax.annotate(annot[idx], (0.05, 0.90), xycoords="axes fraction", fontsize=24)
        # ax.set_ylabel("Simulation time [s]", fontsize=28, labelpad=10)
        # ax.legend(fontsize=12, bbox_to_anchor=(0.5, -0.12), loc="upper center", ncols=2)
        # ax_list[int(np.ceil(len(varname_list) / 2.0))].set_ylabel(
        #     "Simulation time [s]", fontsize=28, labelpad=10
        # )
        # ax_list[-1].set_axis_off()

        # Save figure
        # plt.tight_layout()

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


def msheath_pdyn_hist(x0, x1, y0, y1, t0, t1):
    dr = 300e3 / r_e
    dr_km = 300

    var_list = [
        "proton/vg_pdyn",
        "proton/vg_rho",
        "proton/vg_v",
        "proton/vg_v",
        "proton/vg_v",
        "proton/vg_v",
        "vg_b_vol",
        "vg_b_vol",
        "vg_b_vol",
        "vg_b_vol",
        "proton/vg_t_parallel",
        "proton/vg_t_perpendicular",
    ]

    varlab_list = [
        "Pdyn",
        "Rho",
        "Vx",
        "Vy",
        "Vz",
        "Vmag",
        "Bx",
        "By",
        "Bz",
        "Bmag",
        "TPar",
        "TPerp",
    ]

    # runids = ["AGF", "AIA", "AIB", "AIC"]
    runids = ["AGF", "AIC"]
    runids_paper = ["RDC", "RDC2", "RDC3"]
    sw_pars = [
        [1.0e6, 750.0e3, 3.0e-9, 0.5e6],
        [1.0e6, 750.0e3, 3.0e-9, 0.5e6],
        [1.0e6, 750.0e3, 3.0e-9, 0.5e6],
    ]
    n_sw, v_sw, B_sw, T_sw = sw_pars[runids.index("AGF")]
    pdyn_sw_AGF = m_p * n_sw * v_sw * v_sw

    # n_sw, v_sw, B_sw, T_sw = sw_pars[runids.index("AIA")]
    # pdyn_sw_AIA = m_p * n_sw * v_sw * v_sw

    # n_sw, v_sw, B_sw, T_sw = sw_pars[runids.index("AIB")]
    # pdyn_sw_AIB = m_p * n_sw * v_sw * v_sw

    norm_list = [
        pdyn_sw_AGF,
        n_sw,
        v_sw,
        v_sw,
        v_sw,
        v_sw,
        B_sw,
        B_sw,
        B_sw,
        B_sw,
        T_sw,
        T_sw,
    ]
    op_list = [
        "pass",
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

    # Path to vlsv files for current run
    # bulkpath_AGF = find_bulkpath("AGF")
    # bulkpath_AIA = find_bulkpath("AIA")
    # bulkpath_AIB = find_bulkpath("AIB")
    # bulkpath_SIB = find_bulkpath("AIC")

    bulkpaths = [find_bulkpath(runid) for runid in runids]
    color_list = [
        CB_color_cycle[0],
        CB_color_cycle[1],
        CB_color_cycle[2],
        CB_color_cycle[3],
    ]

    fnr0 = int(t0 * 2)
    fnr1 = int(t1 * 2)

    fnr_range = np.arange(fnr0, fnr1 + 1, 1, dtype=int)
    t_range = np.arange(t0, t1 + 0.1, 0.5)

    fobjs = [
        pt.vlsvfile.VlsvReader(
            bulkpaths[idx] + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
        )
        for idx in range(len(runids))
    ]

    # fobj_AGF = pt.vlsvfile.VlsvReader(
    #     bulkpath_AGF + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
    # )
    # fobj_AIA = pt.vlsvfile.VlsvReader(
    #     bulkpath_AIA + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
    # )
    # fobj_AIB = pt.vlsvfile.VlsvReader(
    #     bulkpath_AIB + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
    # )

    xmesh, ymesh = np.meshgrid(
        np.arange(x0, x1 + 0.001, dr), np.arange(y0, y1 + 0.001, dr)
    )

    xlist = xmesh.flatten()
    ylist = ymesh.flatten()

    # cellids_AGF = [
    #     int(fobj_AGF.get_cellid([xlist[idx] * r_e, ylist[idx] * r_e, 0]))
    #     for idx in range(xlist.size)
    # ]
    # cellids_AIA = [
    #     int(fobj_AIA.get_cellid([xlist[idx] * r_e, ylist[idx] * r_e, 0]))
    #     for idx in range(xlist.size)
    # ]
    # cellids_AIB = [
    #     int(fobj_AIB.get_cellid([xlist[idx] * r_e, ylist[idx] * r_e, 0]))
    #     for idx in range(xlist.size)
    # ]

    cellids = [
        [
            int(fobjs[idx2].get_cellid([xlist[idx] * r_e, ylist[idx] * r_e, 0]))
            for idx in range(xlist.size)
        ]
        for idx2 in range(len(runids))
    ]

    # data_arr_AGF = np.zeros((len(var_list), xlist.size, t_range.size), dtype=float)
    # data_arr_AIA = np.zeros_like(data_arr_AGF)
    # data_arr_AIB = np.zeros_like(data_arr_AGF)

    data_arr = np.zeros(
        (len(runids), len(var_list), xlist.size, t_range.size), dtype=float
    )

    for idx3 in range(len(runids)):
        for idx in range(fnr_range.size):
            fnr = fnr_range[idx]
            # vlsvobj_AGF = pt.vlsvfile.VlsvReader(
            #     bulkpath_AGF + "bulk.{}.vlsv".format(str(fnr).zfill(7))
            # )
            # vlsvobj_AIA = pt.vlsvfile.VlsvReader(
            #     bulkpath_AIA + "bulk.{}.vlsv".format(str(fnr).zfill(7))
            # )
            # vlsvobj_AIB = pt.vlsvfile.VlsvReader(
            #     bulkpath_AIB + "bulk.{}.vlsv".format(str(fnr).zfill(7))
            # )
            vlsvobj = pt.vlsvfile.VlsvReader(
                bulkpaths[idx3] + "bulk.{}.vlsv".format(str(fnr).zfill(7))
            )

            for idx2 in range(len(var_list)):
                # data_arr_AGF[idx2, :, idx] = (
                #     vlsvobj_AGF.read_variable(
                #         var_list[idx2], operator=op_list[idx2], cellids=cellids_AGF
                #     )
                #     / norm_list[idx2]
                # )
                # data_arr_AIA[idx2, :, idx] = (
                #     vlsvobj_AIA.read_variable(
                #         var_list[idx2], operator=op_list[idx2], cellids=cellids_AIA
                #     )
                #     / norm_list[idx2]
                # )
                # data_arr_AIB[idx2, :, idx] = (
                #     vlsvobj_AIB.read_variable(
                #         var_list[idx2], operator=op_list[idx2], cellids=cellids_AIB
                #     )
                #     / norm_list[idx2]
                # )
                data_arr[idx3, idx2, :, idx] = (
                    vlsvobj.read_variable(
                        var_list[idx2], operator=op_list[idx2], cellids=cellids[idx3]
                    )
                    / norm_list[idx2]
                )

    fig, ax_list = plt.subplots(2, 6, figsize=(24, 12))

    for idx in range(len(var_list)):
        for idx2 in range(len(runids)):
            # ax_list.flatten()[idx].hist(
            #     data_arr_AGF[idx, :, :].flatten(),
            #     bins="fd",
            #     color="black",
            #     alpha=0.3,
            #     label="AGF",
            # )
            # ax_list.flatten()[idx].hist(
            #     data_arr_AIA[idx, :, :].flatten(),
            #     bins="fd",
            #     color="blue",
            #     alpha=0.3,
            #     label="AIA",
            # )
            # ax_list.flatten()[idx].hist(
            #     data_arr_AIB[idx, :, :].flatten(),
            #     bins="fd",
            #     color="red",
            #     alpha=0.3,
            #     label="AIB",
            # )
            ax_list.flatten()[idx].hist(
                data_arr[idx2, idx, :, :].flatten(),
                bins="fd",
                color=color_list[idx2],
                alpha=0.3,
                label=runids[idx2],
            )
            # ax_list.flatten()[idx].set(
            #     title=varlab_list[idx],
            #     # xlabel="$P_\mathrm{dyn}$ [$P_\mathrm{dyn,sw}$]",
            #     xlim=(
            #         np.min(
            #             [
            #                 np.min(data_arr_AGF[idx, :, :]),
            #                 np.min(data_arr_AIA[idx, :, :]),
            #                 np.min(data_arr_AIB[idx, :, :]),
            #             ]
            #         ),
            #         np.max(
            #             [
            #                 np.max(data_arr_AGF[idx, :, :]),
            #                 np.max(data_arr_AIA[idx, :, :]),
            #                 np.max(data_arr_AIB[idx, :, :]),
            #             ]
            #         ),
            #     ),
            # )
            ax_list.flatten()[idx].set(
                title=varlab_list[idx],
                # xlabel="$P_\mathrm{dyn}$ [$P_\mathrm{dyn,sw}$]",
                xlim=(
                    np.min(data_arr[:, idx, :, :]),
                    np.max(data_arr[:, idx, :, :]),
                ),
            )
    ax_list.flatten()[0].legend(loc="upper right")

    figdir = wrkdir_DNR + "Figs/histograms/"
    plt.tight_layout()

    if not os.path.exists(figdir):
        try:
            os.makedirs(figdir)
        except OSError:
            pass

    fig.savefig(
        figdir
        + "msheath_hist_x0_{}_y0_{}_x1_{}_y1_{}_t0_{}_t1_{}.png".format(
            x0, y0, x1, y1, t0, t1
        ),
        dpi=300,
    )
    plt.close(fig)


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
):
    dr = 300e3 / r_e
    dr_km = 300
    varname_list = [
        "$\\rho$ [cm$^{-3}$]",
        "$v_x$ [km/s]",
        "$P_\mathrm{dyn}$ [nPa]",
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
    n_sw, v_sw, B_sw, T_sw = sw_pars[runid_list.index(runid)]
    pdyn_sw = m_p * n_sw * v_sw * v_sw

    # vmin_norm = [1.0 / 2, -2.0, 1.0 / 6, 1.0 / 2, 1.0]
    # vmax_norm = [6.0, 2.0, 2.0, 6.0, 36.0]
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

    cellids = [
        int(fobj.get_cellid([xlist[idx] * r_e, ylist[idx] * r_e, 0]))
        for idx in range(xlist.size)
    ]
    cellnr = range(xlist.size)
    if xlist[-1] != xlist[0]:
        xplot_list = xlist
        xlab = "$X~[R_\mathrm{E}]$"
        XmeshXY, YmeshXY = np.meshgrid(xlist, t_range)
    else:
        xplot_list = ylist
        xlab = "$Y~[R_\mathrm{E}]$"
        XmeshXY, YmeshXY = np.meshgrid(ylist, t_range)

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

    if filt:
        for idx in range(5):
            for idx2 in range(xplot_list.size):
                # data_arr[idx, idx2, :] = sosfilt(sos, data_arr[idx, idx2, :])
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

    # data_arr = [rho_arr, v_arr, pdyn_arr, B_arr, T_arr]
    cmap = ["Blues_r", "Blues_r", "Blues_r", "Blues_r", "Blues_r"]
    if filt:
        cmap = ["vik", "vik", "vik", "vik", "vik"]
    annot = ["a", "b", "c", "d", "e"]

    # fig, ax_list = plt.subplots(
    #     1, len(varname_list), figsize=(20, 5), sharex=True, sharey=True
    # )
    figh = 10
    if len(vels_to_plot) < 4:
        figh = 8
    if draw:
        fig, ax_list = plt.subplots(
            1,
            # len(varname_list),
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
            # if idx == 1:
            #     cb_list.append(fig.colorbar(im_list[idx], ax=ax, extend="max"))
            #     cb_list[idx].cmap.set_over("red")
            # else:
            cb_list.append(fig.colorbar(im_list[-1], ax=ax))
            # cb_list.append(fig.colorbar(im_list[idx], ax=ax))
            cb_list[-1].ax.tick_params(labelsize=20)
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
            # ax.contour(XmeshXY, YmeshXY, Tcore_arr, [3], colors=[CB_color_cycle[1]])
            # ax.contour(XmeshXY, YmeshXY, mmsx_arr, [1.0], colors=[CB_color_cycle[4]])
            ax.set_title(varname_list[idx], fontsize=24, pad=10)
            ax.set_xlim(xplot_list[0], xplot_list[-1])
            ax.set_ylim(t_range[0], t_range[-1])
            ax.set_xlabel(xlab, fontsize=24, labelpad=10)
            # ax.axhline(t0, linestyle="dashed", linewidth=0.6)
            # ax.axvline(x0, linestyle="dashed", linewidth=0.6)
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
        # ax_list[int(np.ceil(len(varname_list) / 2.0))].set_ylabel(
        #     "Simulation time [s]", fontsize=28, labelpad=10
        # )
        # ax_list[-1].set_axis_off()

        # Save figure
        # plt.tight_layout()

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
    fmin=1e-15,
    rotatetob=False,
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

    if rotatetob:
        vc_coords_new = np.array([rotateVectorToVector_X(c, b) for c in vc_coords])
        vc_coords = vc_coords_new

    # Select coordinates of chosen velocity component
    if operator in op_list:
        vc_coord_arr = vc_coords[:, op_list.index(operator)]
    elif operator == "magnitude":
        vc_coord_arr = np.sqrt(
            vc_coords[:, 0] ** 2 + vc_coords[:, 1] ** 2 + vc_coords[:, 2] ** 2
        )
    elif operator == "par":
        vc_coord_arr = np.dot(vc_coords, b)
    elif operator == "perp":
        vc_coord_arr = np.sqrt(
            vc_coords[:, 0] ** 2
            + vc_coords[:, 1] ** 2
            + vc_coords[:, 2] ** 2
            - np.dot(vc_coords, b) ** 2
        )
    elif operator == "cosmu":
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
    dbins = np.max(np.ediff1d(vbins)) * np.sqrt(3)
    vbins = np.arange(
        np.min(vbins) - dbins / 2, np.max(vbins) + dbins / 2 + dbins / 4, dbins
    )
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


def rotateVectorToVector_X(vector1, vector2):
    """Applies rotation matrix that would rotate vector2 to x-axis on vector1 and then returns the rotated vector1

    :param vector1        Vector to be rotated
    :param vector2        Vector for creating the rotation matrix
    :returns rotated vector1 vector

    .. note::

       vector1 and vector2 must be 3d vectors
    """
    vector_u = np.cross(vector2, np.array([1, 0, 0]))
    if np.linalg.norm(vector_u) == 0.0:
        return vector1
    else:
        vector_u = vector_u / np.linalg.norm(vector_u)
        angle = np.arccos(vector2.dot(np.array([1, 0, 0])) / np.linalg.norm(vector2))
        # A unit vector version of the given vector
        R = rotation_matrix(vector_u, angle)
        # Rotate vector
        vector_rotated = R.dot(vector1.transpose()).transpose()
        return vector_rotated


def rotation_matrix(vector, angle):
    """Creates a rotation matrix that rotates around a given vector by a given angle
    :param vector        Some unit vector
    :param angle         Some angle
    :returns a rotation matrix
    """
    v = vector
    t = angle
    cost = np.cos(t)
    sint = np.sin(t)
    unitymcost = -cost + 1
    m = np.array(
        [
            [
                cost + v[0] ** 2 * unitymcost,
                v[0] * v[1] * unitymcost - v[2] * sint,
                v[0] * v[2] * unitymcost + v[1] * sint,
            ],
            [
                v[0] * v[1] * unitymcost + v[2] * sint,
                cost + v[1] ** 2 * unitymcost,
                v[1] * v[2] * unitymcost - v[0] * sint,
            ],
            [
                v[0] * v[2] * unitymcost - v[1] * sint,
                v[2] * v[1] * unitymcost + v[0] * sint,
                cost + v[2] ** 2 * unitymcost,
            ],
        ]
    )
    return m


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

    b_arr = uniform_filter1d(b_arr, size=60)

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
                vobj, vdf_cellid, operator="cosmu", b=b_arr[idx]
            )
            yhist, ybin_edges = vspace_reducer(
                vobj, vdf_cellid, operator="par", b=b_arr[idx]
            )
            zhist, zbin_edges = vspace_reducer(
                vobj, vdf_cellid, operator="perp", b=b_arr[idx]
            )
        else:
            xhist, xbin_edges = vspace_reducer(
                vobj, vdf_cellid, operator="x", b=b_arr[idx]
            )
            yhist, ybin_edges = vspace_reducer(
                vobj, vdf_cellid, operator="y", b=b_arr[idx]
            )
            zhist, zbin_edges = vspace_reducer(
                vobj, vdf_cellid, operator="z", b=b_arr[idx]
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
    else:
        norm = None

    pcx = ax_list[0].pcolormesh(
        t_arr, v_arr[0] * scales[0], vx_arr, shading="nearest", cmap="batlow", norm=norm
    )
    pcy = ax_list[1].pcolormesh(
        t_arr, v_arr[1] * scales[1], vy_arr, shading="nearest", cmap="batlow", norm=norm
    )
    pcz = ax_list[2].pcolormesh(
        t_arr, v_arr[2] * scales[2], vz_arr, shading="nearest", cmap="batlow", norm=norm
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
        labels = ["$\\cos\\mu$", "$V_\\parallel$ [km/s]", "$V_\\perp$ [km/s]"]

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
    ncont=5,
    rboxw=2,
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

    outdir = wrkdir_DNR + "VDFs/{}/x_{:.3f}_y_{:.3f}_t0_{}_t1_{}_xyz{}".format(
        runid, x_re, y_re, t0, t1, xyz
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

        fig, ax_list = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)

        pt.plot.plot_colormap(
            axes=ax_list[0][0],
            vlsvobj=vobj,
            var="proton/vg_Pdyn",
            vmin=0.01,
            vmax=pdmax,
            vscale=1e9,
            cbtitle="$P_\mathrm{dyn}$ [nPa]",
            usesci=0,
            boxre=[x_re - rboxw, x_re + rboxw, y_re - rboxw, y_re + rboxw],
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
                fmin=1e-10,
                fmax=1e-4,
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
                fmin=1e-10,
                fmax=1e-4,
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
                fmin=1e-10,
                fmax=1e-4,
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


def jet_vdf_profile_plotter(runid, skip=[], vmin=None, vmax=None):
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


def run_comp_plotter(
    start,
    stop,
    boxre=[-10, 20, -20, 20],
    tickint=5.0,
    blines=False,
    nstp=40,
    pdynmax=1.5,
    pdynmin=0.1,
    outdir="comps",
    pointsx=[],
    pointsy=[],
    fsaved=None,
    lin=1,
):
    var = "proton/vg_Pdyn"
    vscale = 1e9
    vmax = pdynmax
    runids = ["AGF", "AIA", "AIB"]

    if len(pointsx) != len(pointsy):
        print("x and y must have same length!")
        return 1

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g, start_points, drawBy0, ax_g, linestyle_g, idx_g
    global Bmag_g, ax3_g
    runid_g = "AGF"
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

    bulkpath_AGF = find_bulkpath("AGF")
    bulkpath_AIA = find_bulkpath("AIA")
    bulkpath_AIB = find_bulkpath("AIB")

    bulkpaths = [bulkpath_AGF, bulkpath_AIA, bulkpath_AIB]

    non_ids = []

    sj_ids_g = []
    non_ids_g = non_ids

    pdmax = [1.5, 1.5][runids.index("AGF")]
    sw_pars = [
        [1e6, 750e3, 3e-9, 0.5e6],
        [1e6, 750e3, 3e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runids.index("AGF")]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw

    outputdir = wrkdir_DNR + "Figs/{}/".format(outdir)
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    if not os.path.exists(outputdir + "../fluxf"):
        try:
            os.makedirs(outputdir + "../fluxf")
        except OSError:
            pass

    if not os.path.exists(outputdir + "../dipcomp"):
        try:
            os.makedirs(outputdir + "../dipcomp")
        except OSError:
            pass

    # global x0, y0
    # props = jio.PropReader(str(jetid).zfill(5), runid, transient="jet")
    # t0 = props.read("time")[0]
    # x0 = props.read("x_wmean")[0]
    # y0 = props.read("y_wmean")[0]
    # fnr0 = int(t0 * 2)
    linestyles = ["solid", "dashed", "dashdot"]

    for fnr in range(start, stop + 1):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        fig2, ax2 = plt.subplots(2, 2, figsize=(10, 10))
        fig3, ax3 = plt.subplots(2, 2, figsize=(10, 10))
        ax_g = ax
        filenr_g = fnr

        fname = "bulk.{}.vlsv".format(str(int(fnr)).zfill(7))

        vobj = pt.vlsvfile.VlsvReader(bulkpaths[0] + fname)
        Bmag_AGF = vobj.read_variable("vg_b_vol", operator="magnitude")
        cellids_AGF = vobj.read_variable("CellID")
        Bmag_g = Bmag_AGF[np.argsort(cellids_AGF)]

        ax2[1, 1].axis("off")
        ax3[1, 1].axis("off")

        for idx, bulkpath in enumerate(bulkpaths):
            ax3_g = ax3.flatten()[idx]
            idx_g = idx
            linestyle_g = linestyles[idx]
            pt.plot.plot_colormap(
                axes=ax2.flatten()[idx],
                filename=bulkpath + fname,
                outputfile=outputdir
                + "debug/{}_pdyn_{}.png".format(runids[idx], str(fnr).zfill(7)),
                var="vg_b_vol",
                vmin=0.1,
                # vmax=1,
                vmax=10,
                # vscale=1e9,
                # cbtitle="",
                # cbtitle="",
                usesci=0,
                # scale=3,
                title="Run = {}, t = {}s".format(runids[idx], float(fnr) / 2.0),
                cbtitle="$B/B_{AGF}$",
                boxre=boxre,
                internalcb=False,
                # lin=10,
                colormap="vik",
                tickinterval=tickint,
                fsaved=fsaved,
                # useimshow=True,
                external=ext_bs_mp,
                expression=expr_Bratio,
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
                # fluxdir="/wrk-vakka/group/spacephysics/vlasiator/2D/{}/fluxfunction".format(
                #     runids[idx]
                # ),
                fluxfile=bulkpath + "../fluxfunction/" + fname + ".bin",
                fluxlines=10,
            )
            ax3_g.set_title("Run = {}, t = {}s".format(runids[idx], float(fnr) / 2.0))

        fig.savefig(outputdir + "pdyn_{}.png".format(str(fnr).zfill(7)), dpi=300)
        plt.close(fig)

        fig2.savefig(outputdir + "../fluxf/{}.png".format(str(fnr).zfill(7)), dpi=300)
        plt.close(fig2)

        fig3.savefig(outputdir + "../dipcomp/{}.png".format(str(fnr).zfill(7)), dpi=300)
        plt.close(fig3)


def expr_Bratio(exprmaps, requestvariables=False):
    B = exprmaps["vg_b_vol"]
    ci = exprmaps["CellID"].flatten()
    Bmag = np.linalg.norm(B, axis=-1)
    Bref = np.reshape(Bmag_g[ci - 1], Bmag.shape)

    return Bmag / (Bref + 1.0e-30)


def ext_bs_mp(ax, XmeshXY, YmeshXY, pass_maps):
    B = pass_maps["vg_b_vol"]
    # rho = pass_maps["proton/vg_rho"]
    # cellids = pass_maps["CellID"]
    mmsx = pass_maps["proton/vg_mmsx"]
    # core_heating = pass_maps["proton/vg_core_heating"]
    Bmag = np.linalg.norm(B, axis=-1)
    # Pdyn = pass_maps["proton/vg_Pdyn"]
    # Pdynx = pass_maps["proton/vg_Pdynx"]
    beta_star = pass_maps["proton/vg_beta_star"]
    # By = B[:, :, 1]
    Bmag = np.linalg.norm(B, axis=-1)

    RmeshXY = np.sqrt(XmeshXY**2 + YmeshXY**2)
    B_dipole = 3.12e-5 / (RmeshXY**3 + 1e-30)

    dcplot = ax3_g.pcolormesh(
        XmeshXY,
        YmeshXY,
        Bmag / B_dipole,
        cmap="vik",
        # norm=colors.LogNorm(vmin=0.5, vmax=2),
        vmin=0,
        vmax=2,
    )
    dccb = plt.colorbar(dcplot, ax=ax3_g)
    dccb.set_label("Dipole compression $f$", rotation=270, labelpad=10)

    # try:
    #     slams_cells = np.loadtxt(
    #         "/wrk-vakka/users/jesuni/foreshock_bubble/working/SLAMS/Masks/{}/{}.mask".format(
    #             runid_g, int(filenr_g)
    #         )
    #     ).astype(int)
    # except:
    #     slams_cells = []
    # try:
    #     jet_cells = np.loadtxt(
    #         "/wrk-vakka/users/jesuni/foreshock_bubble/working/jets/Masks/{}/{}.mask".format(
    #             runid_g, int(filenr_g)
    #         )
    #     ).astype(int)
    # except:
    #     jet_cells = []

    # sj_jetobs = [
    #     PropReader(str(int(sj_id)).zfill(5), runid_g, transient="jet")
    #     for sj_id in sj_ids_g
    # ]
    # non_sjobs = [
    #     PropReader(str(int(non_id)).zfill(5), runid_g, transient="jet")
    #     for non_id in non_ids_g
    # ]

    sj_xlist = []
    sj_ylist = []
    non_xlist = []
    non_ylist = []

    # for jetobj in sj_jetobs:
    #     if filenr_g / 2.0 in jetobj.read("time"):
    #         sj_xlist.append(jetobj.read_at_time("x_wmean", filenr_g / 2.0))
    #         sj_ylist.append(jetobj.read_at_time("y_wmean", filenr_g / 2.0))
    # for jetobj in non_sjobs:
    #     if filenr_g / 2.0 in jetobj.read("time"):
    #         non_xlist.append(jetobj.read_at_time("x_wmean", filenr_g / 2.0))
    #         non_ylist.append(jetobj.read_at_time("y_wmean", filenr_g / 2.0))

    # for idx in range(len(xg)):
    #     ax.plot(xg[idx], yg[idx], "x", color=CB_color_cycle[idx])

    # slams_mask = np.in1d(cellids, slams_cells).astype(int)
    # slams_mask = np.reshape(slams_mask, cellids.shape)

    # jet_mask = np.in1d(cellids, jet_cells).astype(int)
    # jet_mask = np.reshape(jet_mask, cellids.shape)

    # ch_mask = (core_heating > 3 * T_sw).astype(int)
    mach_mask = (mmsx < 1).astype(int)
    # rho_mask = (rho > 2 * rho_sw).astype(int)

    # plaschke_mask = (Pdynx > 0.25 * Pdyn_sw).astype(int)
    # plaschke_mask[core_heating < 3 * T_sw] = 0

    # cav_shfa_mask = (Bmag < 0.8 * B_sw).astype(int)
    # cav_shfa_mask[rho >= 0.8 * rho_sw] = 0

    # diamag_mask = (Pdyn >= 1.2 * Pdyn_sw).astype(int)
    # diamag_mask[Bmag > B_sw] = 0

    # CB_color_cycle

    # start_points = np.array(
    #     [np.ones(20) * x0 + 0.5, np.linspace(y0 - 0.9, y0 + 0.9, 20)]
    # ).T
    # nstp = 40
    # start_points = np.array([np.ones(nstp) * 17, np.linspace(-20, 20, nstp)]).T

    # if Blines_g:
    #     blines_bx = np.copy(B[:, :, 0])
    #     blines_by = np.copy(B[:, :, 1])
    #     blines_bx[core_heating > 3 * T_sw] = np.nan
    #     blines_by[core_heating > 3 * T_sw] = np.nan
    #     stream = ax.streamplot(
    #         XmeshXY,
    #         YmeshXY,
    #         blines_bx,
    #         blines_by,
    #         arrowstyle="-",
    #         broken_streamlines=False,
    #         color="k",
    #         linewidth=0.4,
    #         # minlength=4,
    #         density=35,
    #         start_points=start_points,
    #     )

    lws = 1.0
    mrks = 2
    mews = 0.4

    # if drawBy0:
    #     by_mask = np.ones_like(By, dtype=int)
    #     by_mask[np.logical_and(By > 0, YmeshXY < 0)] = 0
    #     by_mask[np.logical_and(By < 0, YmeshXY > 0)] = 0

    #     # by_mask[YmeshXY < 0] = 0
    #     by_mask[beta_star < 0.3] = -1
    #     by_mask[core_heating < 3 * T_sw] = -1

    #     by_cont = ax.contourf(
    #         XmeshXY,
    #         YmeshXY,
    #         by_mask,
    #         [-0.5, 0.5],
    #         # linewidths=lws,
    #         colors=[CB_color_cycle[6], CB_color_cycle[8]],
    #         # linestyles=["dashed"],
    #         hatches=["++", "/"],
    #         alpha=0.3,
    #     )

    #     by0_cont = ax.contour(
    #         XmeshXY,
    #         YmeshXY,
    #         By,
    #         [0],
    #         linewidths=lws,
    #         colors="red",
    #         linestyles=["dashed"],
    #     )

    # jet_cont = ax.contour(
    #     XmeshXY,
    #     YmeshXY,
    #     jet_mask,
    #     [0.5],
    #     linewidths=lws,
    #     colors=CB_color_cycle[2],
    #     linestyles=["solid"],
    # )

    # ch_cont = ax_g.contour(
    #     XmeshXY,
    #     YmeshXY,
    #     ch_mask,
    #     [0.5],
    #     linewidths=lws,
    #     colors=CB_color_cycle[0],
    #     linestyles=linestyle_g,
    #     zorder=3,
    # )
    mms_cont = ax_g.contour(
        XmeshXY,
        YmeshXY,
        mach_mask,
        [0.5],
        linewidths=lws,
        colors=CB_color_cycle[0],
        linestyles=linestyle_g,
        zorder=3,
    )
    bs_cont = ax_g.contour(
        XmeshXY,
        YmeshXY,
        beta_star,
        [0.3],
        linewidths=lws,
        colors=CB_color_cycle[1],
        linestyles=linestyle_g,
        zorder=3,
    )

    # if plaschke_g:
    #     plaschke_cont = ax.contour(
    #         XmeshXY,
    #         YmeshXY,
    #         plaschke_mask,
    #         [0.5],
    #         linewidths=lws,
    #         colors=CB_color_cycle[7],
    #         linestyles=["solid"],
    #     )

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

    # (non_pos,) = ax.plot(
    #     non_xlist,
    #     non_ylist,
    #     "o",
    #     color="black",
    #     markersize=mrks,
    #     markeredgecolor="white",
    #     fillstyle="full",
    #     mew=mews,
    #     label="Tracked jet",
    # )
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

    # itr_jumbled = [1, 1, 4, 2, 7]
    # itr_jumbled = [1, 7, 4, 2, 7]

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

    # proxy_labs = [
    #     # "$n=2n_\mathrm{sw}$",
    #     # "$T_\mathrm{core}=3T_\mathrm{sw}$",
    #     "$\\beta^* = 0.3$",
    #     # "$M_{\mathrm{MS},x}=1$",
    #     # "$P_\mathrm{dyn,x}>0.25 P_\mathrm{dyn,sw}$",
    # ]

    # proxy = [
    #     mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[itr]])
    #     for itr in range(len(proxy_labs))
    # ]

    # xmin, xmax, ymin, ymax = (
    #     np.min(XmeshXY),
    #     np.max(XmeshXY),
    #     np.min(YmeshXY),
    #     np.max(YmeshXY),
    # )

    # if plaschke_g:
    #     proxy.append(mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[7]]))
    #     proxy_labs.append("$P_\mathrm{dyn,x}>0.25 P_\mathrm{dyn,sw}$")
    # if ~(jet_mask == 0).all():
    #     proxy.append(mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[3]]))
    #     proxy_labs.append(
    #         "$P_\mathrm{dyn} \geq 2 \\langle P_\mathrm{dyn} \\rangle_\mathrm{3min}$"
    #     )
    # # if ~(slams_mask == 0).all():
    # #     proxy.append(mlines.Line2D([], [], color=CB_color_cycle[itr_jumbled[4]]))
    # #     proxy_labs.append("FCS")
    # if Blines_g:
    #     proxy.append(mlines.Line2D([], [], color="k"))
    #     proxy_labs.append("$B$")
    # if drawBy0:
    #     proxy.append(mlines.Line2D([], [], color="red", linestyle="dashed"))
    #     proxy_labs.append("$B_y=0$")
    #     proxy.append(
    #         mpatches.Patch(
    #             fc=CB_color_cycle[6],
    #             # color="black",
    #             # fill=True,
    #             hatch=r"++++",
    #             alpha=0.3,
    #         )
    #     )
    #     proxy_labs.append("Q$\\perp$ sheath")
    # if np.logical_and(
    #     np.logical_and(non_xlist >= xmin, non_xlist <= xmax),
    #     np.logical_and(non_ylist >= ymin, non_ylist <= ymax),
    # ).any():
    #     proxy.append(non_pos)
    #     proxy_labs.append("Tracked jet")
    # # if np.logical_and(
    # #     np.logical_and(sj_xlist >= xmin, sj_xlist <= xmax),
    # #     np.logical_and(sj_ylist >= ymin, sj_ylist <= ymax),
    # # ).any():
    # #     proxy.append(sj_pos)
    # #     proxy_labs.append("FCS-jet")

    # ax.legend(
    #     proxy,
    #     proxy_labs,
    #     frameon=True,
    #     numpoints=1,
    #     markerscale=1,
    #     loc="lower left",
    #     fontsize=5,
    # )

    # global gprox, gprox_labs

    # gprox = proxy
    # gprox_labs = proxy_labs


def early_bulkpath(runid):
    if runid == "static_IB_test":
        return "/wrk-vakka/users/jesuni/static_IB_B_test_fullres/bulk_before_300/"
    elif runid == "AGF":
        return "/wrk-vakka/group/spacephysics/vlasiator/2D/AGF/run_300s_steady/"
    elif runid == "AIA":
        return "/wrk-vakka/group/spacephysics/vlasiator/2D/AIA/bulk/"
    elif runid == "AIB":
        return "/wrk-vakka/group/spacephysics/vlasiator/2D/AIB/bulk_before_300/"


def run_comp_plotter_early(
    start,
    stop,
    boxre=[-10, 20, -20, 20],
    tickint=5.0,
    blines=False,
    nstp=40,
    pdynmax=1.5,
    pdynmin=0.1,
    outdir="early_comps",
    pointsx=[],
    pointsy=[],
    fsaved=None,
    lin=1,
):
    var = "proton/vg_Pdyn"
    vscale = 1e9
    vmax = pdynmax
    runids = ["AGF", "static_IB_test", "AIA", "AIB"]

    if len(pointsx) != len(pointsy):
        print("x and y must have same length!")
        return 1

    global runid_g, sj_ids_g, non_ids_g, filenr_g, Blines_g, start_points, drawBy0, ax_g, linestyle_g, idx_g
    global Bmag_g, ax3_g
    runid_g = "AGF"
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

    bulkpath_AGF = find_bulkpath("AGF")
    bulkpath_AIA = find_bulkpath("AIA")
    bulkpath_AIB = find_bulkpath("AIB")

    bulkpaths = [
        early_bulkpath(runid) for runid in ["AGF", "static_IB_test", "AIA", "AIB"]
    ]

    non_ids = []

    sj_ids_g = []
    non_ids_g = non_ids

    pdmax = [1.5, 1.5][runids.index("AGF")]
    sw_pars = [
        [1e6, 750e3, 3e-9, 0.5e6],
        [1e6, 750e3, 3e-9, 0.5e6],
        [1e6, 750e3, 3e-9, 0.5e6],
        [1e6, 750e3, 3e-9, 0.5e6],
    ]
    global rho_sw, v_sw, B_sw, T_sw, Pdyn_sw
    rho_sw, v_sw, B_sw, T_sw = sw_pars[runids.index("AGF")]
    Pdyn_sw = m_p * rho_sw * v_sw * v_sw

    outputdir = wrkdir_DNR + "Figs/{}/".format(outdir)
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    if not os.path.exists(outputdir + "../early_fluxf"):
        try:
            os.makedirs(outputdir + "../early_fluxf")
        except OSError:
            pass

    if not os.path.exists(outputdir + "../early_dipcomp"):
        try:
            os.makedirs(outputdir + "../early_dipcomp")
        except OSError:
            pass

    # global x0, y0
    # props = jio.PropReader(str(jetid).zfill(5), runid, transient="jet")
    # t0 = props.read("time")[0]
    # x0 = props.read("x_wmean")[0]
    # y0 = props.read("y_wmean")[0]
    # fnr0 = int(t0 * 2)
    linestyles = ["solid", "dashed", "dashdot", "dotted"]

    for fnr in range(start, stop + 1):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        fig2, ax2 = plt.subplots(2, 2, figsize=(10, 10))
        fig3, ax3 = plt.subplots(2, 2, figsize=(10, 10))
        ax_g = ax
        filenr_g = fnr

        fname = "bulk.{}.vlsv".format(str(int(fnr)).zfill(7))

        vobj = pt.vlsvfile.VlsvReader(bulkpaths[0] + fname)
        Bmag_AGF = vobj.read_variable("vg_b_vol", operator="magnitude")
        cellids_AGF = vobj.read_variable("CellID")
        Bmag_g = Bmag_AGF[np.argsort(cellids_AGF)]

        # ax2[1, 1].axis("off")
        # ax3[1, 1].axis("off")

        for idx, bulkpath in enumerate(bulkpaths):
            ax.plot(
                [-1, -1],
                [-1, -1],
                linestyle=linestyles[idx],
                color=CB_color_cycle[0],
                label=runids[idx],
            )
            ax3_g = ax3.flatten()[idx]
            idx_g = idx
            linestyle_g = linestyles[idx]
            pt.plot.plot_colormap(
                axes=ax2.flatten()[idx],
                filename=bulkpath + fname,
                outputfile=outputdir
                + "debug/{}_pdyn_{}.png".format(runids[idx], str(fnr).zfill(7)),
                var="vg_b_vol",
                vmin=0.1,
                # vmax=1,
                vmax=10,
                # vscale=1e9,
                # cbtitle="",
                # cbtitle="",
                usesci=0,
                # scale=3,
                title="Run = {}, t = {}s".format(runids[idx], float(fnr) * 10),
                cbtitle="$B/B_{AGF}$",
                boxre=boxre,
                internalcb=False,
                # lin=10,
                colormap="vik",
                tickinterval=tickint,
                fsaved=fsaved,
                # useimshow=True,
                external=ext_bs_mp,
                expression=expr_Bratio,
                pass_vars=[
                    # "proton/vg_rho_thermal",
                    # "proton/vg_rho_nonthermal",
                    # "proton/vg_ptensor_thermal_diagonal",
                    "vg_b_vol",
                    "proton/vg_v",
                    "proton/vg_rho",
                    # "proton/vg_core_heating",
                    "CellID",
                    "proton/vg_mmsx",
                    "proton/vg_Pdyn",
                    "proton/vg_Pdynx",
                    "proton/vg_beta_star",
                ],
                # fluxdir="/wrk-vakka/group/spacephysics/vlasiator/2D/{}/fluxfunction".format(
                #     runids[idx]
                # ),
                # fluxfile=bulkpath + "../fluxfunction/" + fname + ".bin",
                # fluxlines=10,
            )
            ax3_g.set_title("Run = {}, t = {}s".format(runids[idx], float(fnr) * 10))
        ax.legend(loc="lower right")
        ax.set_title("MP and BS position, t = {}s".format(float(fnr) * 10))
        ax.set_xlabel("X [$R_E$]")
        ax.set_ylabel("Y [$R_E$]")
        ax.set(xlim=(boxre[0], boxre[1]), ylim=(boxre[2], boxre[3]))

        fig.savefig(outputdir + "pdyn_{}.png".format(str(fnr).zfill(7)), dpi=300)
        plt.close(fig)

        fig2.savefig(
            outputdir + "../early_fluxf/{}.png".format(str(fnr).zfill(7)), dpi=300
        )
        plt.close(fig2)

        fig3.savefig(
            outputdir + "../early_dipcomp/{}.png".format(str(fnr).zfill(7)), dpi=300
        )
        plt.close(fig3)


def hodogram(runid, x0, y0, t0, t1, electric=False, filt=None):

    runids_list = ["AGF", "AIA", "AIC"]

    bulkpath = find_bulkpath(runid)

    t_range = np.arange(t0, t1 + 0.01, 0.5)
    fnr_range = (t_range * 2).astype(int)

    data = np.zeros((3, t_range.size), dtype=float)
    if electric:
        var = "vg_e_vol"
        scale = 1e3
        labels = ["$E_x$", "$E_y$", "$E_z$"]
    else:
        var = "vg_b_vol"
        scale = 1e9
        labels = ["$B_x$", "$B_y$", "$B_z$"]
    op_list = ["x", "y", "z"]

    for idx2, fnr in enumerate(fnr_range):
        vobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
        )

        for idx in range(3):
            data[idx, idx2] = (
                vobj.read_interpolated_variable(
                    var, [x0 * r_e, y0 * r_e, 0], operator=op_list[idx]
                )
                * scale
            )

    if filt:
        for idx in range(3):
            data[idx] = data[idx] - uniform_filter1d(data[idx], size=filt)

    fig, ax_list = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)

    for idx in range(3):
        ax = ax_list[idx]
        ax.grid()
        idx_x = int(idx % 3)
        idx_y = int((idx + 1) % 3)
        ax.quiver(
            data[idx_x, :-1],
            data[idx_y, :-1],
            data[idx_x, 1:] - data[idx_x, :-1],
            data[idx_y, 1:] - data[idx_y, :-1],
            scale_units="xy",
            angles="xy",
            scale=1,
        )
        ax.set_xlabel(labels[idx_x])
        ax.set_ylabel(labels[idx_y])
        ax.set_aspect("equal")

    outdir = wrkdir_DNR + "Figs/hodograms/"
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    fig.savefig(
        outdir
        + "{}_x{}_y{}_t0{}_t1{}_var_{}_filt{}.png".format(
            runid, x0, y0, t0, t1, var, filt
        )
    )
    plt.close(fig)


def wavelet_analysis(runid, x0, y0, t0, t1, var, op="x"):

    dt = 0.5
    fs = 1 / dt

    t_arr = np.arange(t0, t1 + dt / 2, dt)
    fnr_arr = (t_arr * 2).astype(int)

    freq_min = int(np.log2(2.0 / (t1 - t0)))
    freq_max = int(np.log2(fs / 2))

    # freq = np.linspace(2.0 / (t1 - t0), fs / 2, 200)
    freq = np.logspace(freq_min, freq_max, 100, base=2.0)
    w = 6.0
    widths = w * fs / (2 * freq * np.pi)
    bulkpath = find_bulkpath(runid)

    tmeshtf, fmeshtf = np.meshgrid(t_arr, freq)

    data = np.zeros(t_arr.size)

    for idx, fnr in enumerate(fnr_arr):

        vobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr).zfill(7))
        )

        data[idx] = vobj.read_interpolated_variable(
            var, [x0 * r_e, y0 * r_e, 0], operator=op
        )

    cwtm = cwt(data, morlet2, widths, w=w)

    fig, ax = plt.subplots(1, 1, figsize=(15, 6), constrained_layout=True)

    pcm = ax.pcolormesh(
        tmeshtf,
        fmeshtf,
        np.abs(cwtm),
        cmap="hot_desaturated",
        shading="gouraud",
        norm=colors.LogNorm(vmin=1e-10),
    )
    plt.colorbar(pcm, ax=ax)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("f [Hz]")
    ax.set_yscale("log", base=2)

    persmesh = 1.0 / fmeshtf

    nogo_cont = ax.contourf(
        tmeshtf,
        fmeshtf,
        np.logical_or(tmeshtf < t0 + persmesh, tmeshtf > t1 - persmesh),
        [0.5, 1.5],
        # linewidths=lws,
        colors=[CB_color_cycle[6], CB_color_cycle[8]],
        # linestyles=["dashed"],
        hatches=["xx", "/"],
        alpha=0.3,
    )

    outdir = wrkdir_DNR + "Figs/wavelet/"
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    fig.savefig(
        outdir
        + "{}_x{}_y{}_t0{}_t1{}_var_{}_{}.png".format(runid, x0, y0, t0, t1, var, op)
    )
    plt.close(fig)


def MVA(data, eigvals=False, prnt=True):

    M = np.zeros((3, 3), dtype=float)

    for i in range(3):
        for j in range(3):
            M[i, j] = np.nanmean(data[i] * data[j]) - np.nanmean(data[i]) * np.nanmean(
                data[j]
            )

    eigenval, eigenvec = np.linalg.eig(M)
    eigenvec = eigenvec.T

    # for idx in range(3):
    #     if eigenvec[0][0] > 0:
    #         eigenvec[idx] *= -1

    if prnt:
        print("\n")
        print("Eigenvalues: ", np.sort(eigenval))
        print("Eigenvectors: ", eigenvec[np.argsort(eigenval)])

    # return (np.sort(eigenval),eigenvec[np.argsort(eigenval)])
    if eigvals:
        return (np.sort(eigenval), eigenvec[np.argsort(eigenval), :])
    else:
        return eigenvec[np.argsort(eigenval), :]


def cut_animation(runid, x0, x1, y0, t0, t1, intpol=False):

    dr = 300e3
    global bulkpath, var_list, plot_labels, scales, draw_legend, ylabels, norm, ops, plot_index, plot_colors, coords_arr, data_arr, x_arr, ax_list, fnr_arr, min_arr, max_arr
    bulkpath = find_bulkpath(runid)

    x_arr = np.arange(x0 * r_e, x1 * r_e + dr / 2.0, dr)
    coords_arr = np.array([[x, y0 * r_e, 0] for x in x_arr])

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

    t_arr = np.arange(t0, t1 + 0.1, 0.5)
    fnr0 = int(t0 * 2)
    fnr_arr = np.arange(fnr0, int(t1 * 2) + 1, dtype=int)
    cellid = pt.vlsvfile.VlsvReader(
        bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
    ).get_cellid([x0 * r_e, y0 * r_e, 0 * r_e])
    if not intpol:
        vobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
        )
        cellids = [vobj.get_cellid(coord) for coord in coords_arr]
    data_arr = np.zeros((fnr_arr.size, len(var_list), x_arr.size), dtype=float)

    for idx3 in range(fnr_arr.size):
        vlsvobj = pt.vlsvfile.VlsvReader(
            bulkpath + "bulk.{}.vlsv".format(str(fnr_arr[idx3]).zfill(7))
        )
        for idx2 in range(len(var_list)):
            if intpol:
                for idx in range(x_arr.size):

                    data_arr[idx3, idx2, idx] = (
                        vlsvobj.read_interpolated_variable(
                            var_list[idx2], coords_arr[idx], operator=ops[idx2]
                        )
                        * scales[idx2]
                    )
            else:
                data_arr[idx3, idx2, :] = (
                    vlsvobj.read_variable(
                        var_list[idx2],
                        operator=ops[idx2],
                        cellids=cellids,
                    )
                    * scales[idx2]
                )

    min_arr = [
        0.95 * np.min(data_arr[:, 0, :]),
        -1.05 * np.max(np.abs(data_arr[:, 1:5, :])),
        0.95 * np.min(data_arr[:, 5, :]),
        -1.05 * np.max(np.abs(data_arr[:, 6:10, :])),
        -1.05 * np.max(np.abs(data_arr[:, 10:14, :])),
        0.95 * np.min(data_arr[:, 14:, :]),
    ]
    max_arr = [
        1.05 * np.max(data_arr[:, 0, :]),
        1.05 * np.max(np.abs(data_arr[:, 1:5, :])),
        1.05 * np.max(data_arr[:, 5, :]),
        1.05 * np.max(np.abs(data_arr[:, 6:10, :])),
        1.05 * np.max(np.abs(data_arr[:, 10:14, :])),
        1.05 * np.max(data_arr[:, 14:, :]),
    ]

    fig, ax_list = plt.subplots(
        len(ylabels), 1, sharex=True, figsize=(6, 8), constrained_layout=True
    )

    figdir = wrkdir_DNR + "Figs/cut_anims/"
    if not os.path.exists(figdir):
        try:
            os.makedirs(figdir)
        except OSError:
            pass

    ani = FuncAnimation(fig, cut_update, frames=range(fnr_arr.size), blit=False)
    ani.save(
        figdir + "{}_x_{}_{}_y_{}_t_{}_{}.mp4".format(runid, x0, x1, y0, t0, t1),
        fps=5,
        dpi=150,
        bitrate=1000,
    )
    # print("Saved animation")
    plt.close(fig)


def cut_update(idx3):

    fnr = fnr_arr[idx3]

    # for idx in range(x_arr.size):
    #     for idx2 in range(len(var_list)):
    #         data_arr[idx2, idx] = (
    #             vlsvobj.read_interpolated_variable(
    #                 var_list[idx2], coords_arr[idx], operator=ops[idx2]
    #             )
    #             * scales[idx2]
    #         )

    for ax in ax_list:
        ax.clear()

    for idx in range(len(var_list)):
        ax = ax_list[plot_index[idx]]
        # for vline in vlines:
        #     ax.axvline(vline, linestyle="dashed", linewidth=0.6)
        ax.plot(
            x_arr / r_e,
            data_arr[idx3, idx],
            color=plot_colors[idx],
            label=plot_labels[idx],
        )
        if idx == 5:
            pdynx = (
                m_p
                * data_arr[idx3, 0]
                * 1e6
                * data_arr[idx3, 1]
                * 1e3
                * data_arr[idx3, 1]
                * 1e3
                * 1e9
            )
            ax.plot(
                x_arr / r_e,
                pdynx,
                color=CB_color_cycle[0],
                label="$P_{\mathrm{dyn},x}$",
            )

        if draw_legend[idx]:
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax_list[0].set_title("t = {}s".format(fnr / 2.0))
    ax_list[-1].set_xlabel("X [$R_\mathrm{E}$]")
    for idx, ax in enumerate(ax_list):
        ax.grid()
        ax.set_ylabel(ylabels[idx])
        ax.set_xlim(x_arr[0] / r_e, x_arr[-1] / r_e)
        ax.set_ylim(min_arr[idx], max_arr[idx])


def plot_vsc_tangents(t=600):

    T_sw = 0.5e6

    fnr = int(t * 2)
    bulkpath = find_bulkpath("AIC")
    bulkname = "bulk.{}.vlsv".format(str(fnr).zfill(7))
    vlsvobj = pt.vlsvfile.VlsvReader(bulkpath + bulkname)

    beta_star = vlsvobj.read_variable("proton/vg_beta_star")
    core_heating = vlsvobj.read_variable("proton/vg_core_heating")
    mmsx = vlsvobj.read_variable("proton/vg_mmsx")
    cellids = vlsvobj.read_variable("CellID")

    bs_cells = cellids[core_heating >= 3 * T_sw]
    # bs_cells = cellids[mmsx <= 1]
    mp_cells = cellids[beta_star <= 0.3]

    bs_coords = []
    mp_coords = []

    temp_coords = []

    for c in bs_cells:
        coords = vlsvobj.get_cell_coordinates(c) / r_e
        temp_coords.append(coords[:2])
    temp_coords = np.array(temp_coords)
    for yuni in np.unique(temp_coords[:, 1]):
        xclip = temp_coords[:, 0][temp_coords[:, 1] == yuni]
        bs_coords.append([max(xclip), yuni])

    temp_coords = []

    for c in mp_cells:
        coords = vlsvobj.get_cell_coordinates(c) / r_e
        temp_coords.append(coords[:2])
    temp_coords = np.array(temp_coords)
    for yuni in np.unique(temp_coords[:, 1]):
        xclip = temp_coords[:, 0][temp_coords[:, 1] == yuni]
        mp_coords.append([max(xclip), yuni])

    bs_coords = np.array(bs_coords)
    mp_coords = np.array(mp_coords)

    # bs_coords = bs_coords[np.argsort(bs_coords[:, 1])]
    # mp_coords = mp_coords[np.argsort(mp_coords[:, 1])]

    x_mp, y_mp = mp_coords.T
    x_bs, y_bs = bs_coords.T

    # x_mp, y_mp = MP_xy(m_p * 1e6 * 750e3 * 750e3 * 1e9, 0.0, thetaminmax=[-90.25, 90])
    # x_bs, y_bs = BS_xy(1, 750, 11.5, thetaminmax=[-90.25, 90])

    fig, ax = plt.subplots(1, 1, figsize=(12, 12), constrained_layout=True)

    for idx, fname in enumerate(os.listdir(wrkdir_DNR + "vlas_pos_mva")):
        x0, y0, nx, ny, nz = np.loadtxt(wrkdir_DNR + "vlas_pos_mva/" + fname)
        nvec = np.array([nx, ny, nz])
        ortho_vector = np.cross(nvec, [0, 0, 1])
        ortho_vector = ortho_vector / np.linalg.norm(ortho_vector)
        ax.plot(x0, y0, "*", color=CB_color_cycle[idx])
        ax.plot(
            [
                x0 - 2 * ortho_vector[0],
                x0 + 2 * ortho_vector[0],
            ],
            [
                y0 - 2 * ortho_vector[1],
                y0 + 2 * ortho_vector[1],
            ],
            color=CB_color_cycle[idx],
        )

    ax.set_ylabel("Y [RE]")
    ax.grid()
    ax.plot(x_bs, y_bs, color="k", zorder=0)
    ax.plot(x_mp, y_mp, color="k", zorder=0)
    ax.plot(
        np.cos(np.arange(0, 2 * np.pi + 0.02, 0.05)),
        np.sin(np.arange(0, 2 * np.pi + 0.02, 0.05)),
        color="k",
        zorder=0,
    )
    ax.set_aspect("equal")
    ax.set_xlim(left=0)
    ax.set_ylim(-20, 20)
    fig.savefig(wrkdir_DNR + "Figs/vlas_pos_mva.png", dpi=150)
    plt.close(fig)


def AGF_AIC_comp(x0, y0, t0, t1):

    bulkpath_AGF = find_bulkpath("AGF")
    bulkpath_AIC = find_bulkpath("AIC")

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
    plot_labels = [
        "$\\rho~[\mathrm{cm}^{-3}]$",
        "$v_x~[\mathrm{km/s}]$",
        "$v_y~[\mathrm{km/s}]$",
        "$v_z~[\mathrm{km/s}]$",
        "$|v|~[\mathrm{km/s}]$",
        "$P_\mathrm{dyn}~[\mathrm{nPa}]$",
        "$B_x~[\mathrm{nT}]$",
        "$B_y~[\mathrm{nT}]$",
        "$B_z~[\mathrm{nT}]$",
        "$|B|~[\mathrm{nT}]$",
        "$E_x~[\mathrm{mV/m}]$",
        "$E_y~[\mathrm{mV/m}]$",
        "$E_z~[\mathrm{mV/m}]$",
        "$|E|~[\mathrm{mV/m}]$",
        "$T_\\parallel~[\mathrm{MK}]$",
        "$T_\\perp~[\mathrm{MK}]$",
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

    t_arr = np.arange(t0, t1 + 0.1, 0.5)
    fnr0 = int(t0 * 2)
    fnr_arr = np.arange(fnr0, int(t1 * 2) + 1, dtype=int)
    cellid_AGF = pt.vlsvfile.VlsvReader(
        bulkpath_AGF + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
    ).get_cellid([x0 * r_e, y0 * r_e, 0 * r_e])
    cellid_AIC = pt.vlsvfile.VlsvReader(
        bulkpath_AIC + "bulk.{}.vlsv".format(str(fnr0).zfill(7))
    ).get_cellid([x0 * r_e, y0 * r_e, 0 * r_e])
    data_arr_AGF = np.zeros((len(var_list), fnr_arr.size), dtype=float)
    data_arr_AIC = np.zeros((len(var_list), fnr_arr.size), dtype=float)

    for idx, fnr in enumerate(fnr_arr):
        vlsvobj_AGF = pt.vlsvfile.VlsvReader(
            bulkpath_AGF + "bulk.{}.vlsv".format(str(fnr).zfill(7))
        )
        vlsvobj_AIC = pt.vlsvfile.VlsvReader(
            bulkpath_AIC + "bulk.{}.vlsv".format(str(fnr).zfill(7))
        )
        for idx2, var in enumerate(var_list):
            data_arr_AGF[idx2, idx] = (
                vlsvobj_AGF.read_variable(var, operator=ops[idx2], cellids=cellid_AGF)
                * scales[idx2]
            )
            data_arr_AIC[idx2, idx] = (
                vlsvobj_AIC.read_variable(var, operator=ops[idx2], cellids=cellid_AIC)
                * scales[idx2]
            )

    fig, ax_list = plt.subplots(
        len(plot_labels), 1, sharex=True, figsize=(6, 16), constrained_layout=True
    )
    ax_list[0].set_title("Run: AGF vs. AIC, $x_0$: {}, $y_0$: {}".format(x0, y0))
    for idx in range(len(var_list)):
        ax = ax_list[idx]
        ax.plot(t_arr, data_arr_AGF[idx], color="k", linestyle="solid")
        ax.plot(t_arr, data_arr_AIC[idx], color="k", linestyle="dashed")
        ax.set_xlim(t_arr[0], t_arr[-1])
    ax_list[-1].set_xlabel("Simulation time [s]")
    for idx, ax in enumerate(ax_list):
        ax.grid()
        ax.set_ylabel(plot_labels[idx])
    figdir = wrkdir_DNR + "Figs/timeseries/"
    if not os.path.exists(figdir):
        try:
            os.makedirs(figdir)
        except OSError:
            pass

    fig.savefig(
        figdir + "AGFvsAIC_x{}_y{}_t0{}_t1{}.png".format(x0, y0, t0, t1),
        dpi=300,
    )
    plt.close(fig)
