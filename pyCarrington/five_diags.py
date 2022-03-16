import numpy as np
import pytools as pt
from scipy import interpolate
import matplotlib.pyplot as plt

r_e = 6.371e6
mu_0 = 4 * np.pi * 1e-7
m_p = 1.672621898e-27

r_geo = 35.786e6 / r_e + 1
r_galileo = 23.222e6 / r_e + 1
r_gps = 20.180e6 / r_e + 1
r_glonass = 19.130e6 / r_e + 1

r_sats = [r_geo, r_galileo, r_gps, r_glonass]
name_sats = ["GEO", "Galileo", "GPS", "GLONASS"]
offset_sats = [-0.005, -0.005, -0.005, -0.025]


def map_surface_to_ib(theta, ib):

    return np.arcsin(np.sqrt(ib * np.sin(theta) * np.sin(theta) / r_e))


def colormap_diff_precip(run="BGF"):

    if run == "BGD":
        vlsvobj = pt.vlsvfile.VlsvReader(
            "/wrk/group/spacephysics/vlasiator/2D/BGD/bulk/bulk.0000450.vlsv"
        )
        step = 450

    elif run == "BGF":
        vlsvobj = pt.vlsvfile.VlsvReader(
            "/wrk/group/spacephysics/vlasiator/2D/BGF/extendvspace_restart229/bulk.0000450.vlsv"
        )
        step = 450
    elif run == "BGG":
        vlsvobj = pt.vlsvfile.VlsvReader(
            "/wrk/group/spacephysics/vlasiator/2D/BGG/denseIono_restart81/bulk/bulk.0000239.vlsv"
        )
        step = 239

    for k in range(16):
        pt.plot.plot_colormap(
            vlsvobj=vlsvobj,
            outputdir="/wrk/users/jesuni/Figures/carrington/",
            var="proton/vg_precipitationdifferentialflux",
            vmin=1,
            vmax=10 ** 5,
            operator="{}".format(k),
            run=run,
            step=step,
            boxre=[-10, 10, -10, 10],
            Earth=1,
        )


def plot_precip(plot_diff=False, min_energy=None):

    rho_arr = np.array([3.3, 7, 20])
    v_arr = np.array([600, 1000, 1500])
    pdyn_arr = m_p * rho_arr * 1e6 * v_arr * v_arr * 1e6 * 1e9
    B_arr = np.array([10, 20, 30])

    plt.ioff()

    (
        theta,
        precip_bgd,
        meanenergy_bgd,
        difflux_bgd,
        FAC_bgd,
        binedges_bgd,
        x_bgd,
        z_bgd,
        energybins,
        start_B_bgd,
        end_B_bgd,
    ) = precipitation_diag("BGD")
    (
        theta,
        precip_bgf,
        meanenergy_bgf,
        difflux_bgf,
        FAC_bgf,
        binedges_bgf,
        x_bgf,
        z_bgf,
        energybins,
        start_B_bgf,
        end_B_bgf,
    ) = precipitation_diag("BGF")
    (
        theta,
        precip_bgg,
        meanenergy_bgg,
        difflux_bgg,
        FAC_bgg,
        binedges_bgg,
        x_bgg,
        z_bgg,
        energybins,
        start_B_bgg,
        end_B_bgg,
    ) = precipitation_diag("BGG")

    deltaE_bgd = binedges_bgd[1:] - binedges_bgd[:-1]
    if min_energy:
        precip_bgd = np.zeros_like(theta)
        precip_bgf = np.zeros_like(theta)
        precip_bgg = np.zeros_like(theta)
        idx_list = np.arange(deltaE_bgd.size)
        idx_list = idx_list[binedges_bgd[:-1] >= min_energy]
        for idx in idx_list:
            precip_bgd += difflux_bgd[:, idx] * deltaE_bgd[idx] * energybins[idx] / 1e3
            precip_bgf += difflux_bgf[:, idx] * deltaE_bgd[idx] * energybins[idx] / 1e3
            precip_bgg += difflux_bgg[:, idx] * deltaE_bgd[idx] * energybins[idx] / 1e3
        precip_bgd[precip_bgd <= 0] = np.nan
        precip_bgf[precip_bgf <= 0] = np.nan
        precip_bgg[precip_bgg <= 0] = np.nan

    fig, ax = plt.subplots(1, 1)

    ax.grid()
    ax.semilogy(theta, FAC_bgd / 1e-9, label="Normal")
    ax.semilogy(theta, FAC_bgf / 1e-9, label="Moderate")
    ax.semilogy(theta, FAC_bgg / 1e-9, label="Strong")
    ax.legend(fontsize=14)

    ax.set_xlim(60, 120)
    ax.invert_xaxis()

    ax.set_xlabel("$\\theta$ [$^\\circ$]", fontsize=14)
    ax.set_ylabel(
        "FAC [nA/m$^2$]",
        fontsize=14,
    )

    plt.tight_layout()
    fig.savefig("/wrk/users/jesuni/Figures/carrington/FAC.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)

    ax.grid()
    ax.semilogy(theta, precip_bgd, label="Normal")
    ax.semilogy(theta, precip_bgf, label="Moderate")
    ax.semilogy(theta, precip_bgg, label="Strong")
    ax.legend(fontsize=14)

    ax.set_xlim(60, 120)
    ax.set_ylim(10 ** 0, 10 ** 12)
    ax.invert_xaxis()

    ax.set_xlabel("$\\theta$ [$^\\circ$]", fontsize=14)
    ax.set_ylabel(
        "Precipitation integral energy flux [$\mathrm{keV}\mathrm{cm}^{-2}\mathrm{s}^{-1}\mathrm{sr}^{-1}$]",
        fontsize=12,
    )
    if min_energy:
        ax.set_title("$>${:n} eV".format(min_energy), fontsize=14)

    plt.tight_layout()
    if min_energy:
        fig.savefig(
            "/wrk/users/jesuni/Figures/carrington/precipitation_integralflux_{}.png".format(
                min_energy
            )
        )
    else:
        fig.savefig(
            "/wrk/users/jesuni/Figures/carrington/precipitation_integralflux.png"
        )
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)

    ax.grid()
    ax.semilogy(theta, meanenergy_bgd, label="Normal")
    ax.semilogy(theta, meanenergy_bgf, label="Moderate")
    ax.semilogy(theta, meanenergy_bgg, label="Strong")
    ax.legend(fontsize=14)

    ax.set_xlim(60, 120)
    # ax.set_ylim(10 ** 0, 10 ** 10)
    ax.invert_xaxis()

    ax.set_xlabel("$\\theta$ [$^\\circ$]", fontsize=14)
    ax.set_ylabel("Precipitation mean energy [keV]", fontsize=14)

    plt.tight_layout()
    fig.savefig("/wrk/users/jesuni/Figures/carrington/precipitation_meanenergy.png")
    plt.close(fig)

    if plot_diff:
        for k in range(16):
            fig, ax = plt.subplots(1, 1)

            ax.grid()
            ax.semilogy(theta, difflux_bgd[:, k], label="Normal")
            ax.semilogy(theta, difflux_bgf[:, k], label="Moderate")
            ax.semilogy(theta, difflux_bgg[:, k], label="Strong")
            ax.legend(fontsize=14)

            ax.set_xlim(60, 120)
            # ax.set_ylim(10 ** 0, 10 ** 10)
            ax.invert_xaxis()

            ax.set_title(
                "{:n} - {:n} eV".format(binedges_bgd[k], binedges_bgd[k + 1]),
                fontsize=14,
            )

            ax.set_xlabel("$\\theta$ [$^\\circ$]", fontsize=14)
            ax.set_ylabel(
                "Precipitation diff number flux [$\mathrm{cm}^{-2}\mathrm{s}^{-1}\mathrm{sr}^{-1}$]",
                fontsize=12,
            )

            plt.tight_layout()
            fig.savefig(
                "/wrk/users/jesuni/Figures/carrington/precipitation_diffflux{}.png".format(
                    k
                )
            )
            plt.close(fig)

    fig, ax = plt.subplots(1, 1)

    ax.grid()
    ax.plot(theta, x_bgd / r_e, "o", label="Normal")
    ax.plot(theta, x_bgf / r_e, "o", label="Moderate")
    ax.plot(theta, x_bgg / r_e, "o", label="Strong")
    ax.legend(fontsize=14)

    ax.set_xlim(60, 120)
    # ax.set_ylim(10 ** 0, 10 ** 10)
    ax.invert_xaxis()

    ax.set_xlabel("$\\theta$ [$^\\circ$]", fontsize=14)
    ax.set_ylabel("X [RE]", fontsize=14)

    plt.tight_layout()
    fig.savefig("/wrk/users/jesuni/Figures/carrington/precip_x.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)

    ax.grid()
    ax.plot(theta, z_bgd / r_e, "o", label="Normal")
    ax.plot(theta, z_bgf / r_e, "o", label="Moderate")
    ax.plot(theta, z_bgg / r_e, "o", label="Strong")
    ax.legend(fontsize=14)

    ax.set_xlim(60, 120)
    # ax.set_ylim(10 ** 0, 10 ** 10)
    ax.invert_xaxis()

    ax.set_xlabel("$\\theta$ [$^\\circ$]", fontsize=14)
    ax.set_ylabel("Z [RE]", fontsize=14)

    plt.tight_layout()
    fig.savefig("/wrk/users/jesuni/Figures/carrington/precip_z.png")
    plt.close(fig)

    num_arr = [1, 2, 3]
    max_precip_arr = [
        np.nanmax(precip_bgd * start_B_bgd / end_B_bgd),
        np.nanmax(precip_bgf * start_B_bgf / end_B_bgf),
        np.nanmax(precip_bgg * start_B_bgg / end_B_bgg),
    ]

    fig, ax = plt.subplots(1, 1)

    ax.grid()
    ax.semilogy(pdyn_arr[0], max_precip_arr[0], "o", label="Normal", color="black")
    ax.semilogy(pdyn_arr[1], max_precip_arr[1], "o", label="Moderate", color="black")
    ax.semilogy(pdyn_arr[2], max_precip_arr[2], "o", label="Strong", color="black")
    # ax.set_xticks([1, 2, 3])
    # ax.set_xticklabels(["Normal", "Moderate", "Strong"])
    ax.set_ylabel(
        "Maximum Precipitation\nintegral energy flux [$\mathrm{keV}\mathrm{cm}^{-2}\mathrm{s}^{-1}\mathrm{sr}^{-1}$]",
        fontsize=12,
    )
    ax.set_xlabel(
        "$P_\mathrm{dyn,sw}$ [nPa]",
        fontsize=14,
    )
    # ax.legend(fontsize=14)
    if min_energy:
        ax.set_title("$>${:n} eV".format(min_energy), fontsize=14)

    plt.tight_layout()
    if min_energy:
        fig.savefig(
            "/wrk/users/jesuni/Figures/carrington/max_integralflux_{}.png".format(
                min_energy
            )
        )
    else:
        fig.savefig(
            "/wrk/users/jesuni/Figures/carrington/max_integralflux.png", dpi=300
        )
        fig.savefig("/wrk/users/jesuni/Figures/carrington/max_integralflux.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)

    max_fac_arr = [
        np.nanmax(FAC_bgd * start_B_bgd / end_B_bgd),
        np.nanmax(FAC_bgf * start_B_bgf / end_B_bgf),
        np.nanmax(FAC_bgg * start_B_bgg / end_B_bgg),
    ]

    ax.grid()
    ax.semilogy(pdyn_arr[0], max_fac_arr[0] / 1e-9, "o", label="Normal", color="black")
    ax.semilogy(
        pdyn_arr[1], max_fac_arr[1] / 1e-9, "o", label="Moderate", color="black"
    )
    ax.semilogy(pdyn_arr[2], max_fac_arr[2] / 1e-9, "o", label="Strong", color="black")
    # ax.set_xticks([1, 2, 3])
    # ax.set_xticklabels(["Normal", "Moderate", "Strong"])
    ax.set_ylabel(
        "Maximum FAC [nA/m$^2$]",
        fontsize=14,
    )
    ax.set_xlabel(
        "$P_\mathrm{dyn,sw}$ [nPa]",
        fontsize=14,
    )

    plt.tight_layout()
    fig.savefig("/wrk/users/jesuni/Figures/carrington/max_fac.png", dpi=300)
    fig.savefig("/wrk/users/jesuni/Figures/carrington/max_fac.pdf")
    plt.close(fig)

    plt.ion()


def precipitation_diag(run):

    D = -126.2e6

    if run == "BGD":
        vlsvobj = pt.vlsvfile.VlsvReader(
            "/wrk/group/spacephysics/vlasiator/2D/BGD/bulk/bulk.0000450.vlsv"
        )

        r_stop = 19.1e6
        ds = 500e3
    elif run == "BGF":
        vlsvobj = pt.vlsvfile.VlsvReader(
            "/wrk/group/spacephysics/vlasiator/2D/BGF/extendvspace_restart229/bulk.0000450.vlsv"
        )
        r_stop = 18.1e6
        ds = 500e3
    elif run == "BGG":
        vlsvobj = pt.vlsvfile.VlsvReader(
            "/wrk/group/spacephysics/vlasiator/2D/BGG/denseIono_restart81/bulk/bulk.0000239.vlsv"
        )
        r_stop = 13.6e6
        ds = 250e3

    dib = 4 * ds

    theta_arr = np.linspace(120, 60, 60)
    precip_arr = np.zeros_like(theta_arr)
    meanenergy_arr = np.zeros_like(theta_arr)
    diffprecip_arr = np.zeros((theta_arr.size, 16), dtype=float)
    x_arr = np.zeros_like(theta_arr)
    z_arr = np.zeros_like(theta_arr)
    FAC_arr = np.zeros_like(theta_arr)
    start_B = np.zeros_like(theta_arr)
    end_B = np.zeros_like(theta_arr)

    energybins = np.array(
        [
            vlsvobj.read_parameter("proton_PrecipitationCentreEnergy{}".format(i))
            for i in range(16)
        ]
    )
    dlogener = np.log(energybins[1]) - np.log(energybins[0])
    Ebinedges = np.zeros(len(energybins) + 1)
    Ebinedges[1:-1] = np.sqrt(energybins[1:] * energybins[:-1])
    Ebinedges[0] = np.exp(np.log(energybins[0]) - dlogener)
    Ebinedges[-1] = np.exp(np.log(energybins[-1]) + dlogener)
    deltaE = Ebinedges[1:] - Ebinedges[:-1]

    for itr, theta in enumerate(theta_arr):
        start_coords = [
            r_e * np.cos(np.deg2rad(theta)),
            0,
            r_e * np.sin(np.deg2rad(theta)),
        ]
        ib_coords = trace_b_good(
            start_coords,
            kind="linedipole",
            r_stop=r_stop,
            ds=100e3,
            direction=-1,
            iter_max=10000,
            trace_full=False,
            run=run,
        )
        if np.linalg.norm(ib_coords) >= (r_stop - 500e3):
            end_coords = trace_b_good(
                ib_coords,
                vlsvobj=vlsvobj,
                kind="vg_b_vol",
                r_stop=r_stop + dib,
                ds=ds,
                direction=-1,
                iter_max=10000,
                trace_full=False,
                run=run,
            )
            if np.linalg.norm(end_coords) >= (r_stop + dib - ds):
                ci = vlsvobj.get_cellid(end_coords)
                Bmag = vlsvobj.read_variable(
                    "proton/vg_b_vol", operator="magnitude", cellids=[int(ci), 1]
                )[0]
                precip = vlsvobj.read_variable(
                    "proton/vg_precipitationintegralenergyflux", cellids=[int(ci), 1]
                )[0]
                precip_arr[itr] = precip

                meanenergy = vlsvobj.read_variable(
                    "proton/vg_precipitationmeanenergy", cellids=[int(ci), 1]
                )[0]
                meanenergy_arr[itr] = meanenergy

                diffprecip = vlsvobj.read_variable(
                    "proton/vg_precipitationdifferentialflux", cellids=[int(ci), 1]
                )[0]
                diffprecip_arr[itr] = diffprecip
                FAC = calc_FAC(vlsvobj, [end_coords[0], end_coords[1]], dr=ds)
                FAC_arr[itr] = FAC
                end_B[itr] = Bmag
                start_B[itr] = D / np.linalg.norm(start_coords) ** 2

                coords_ci = vlsvobj.get_cell_coordinates(ci)
                x_arr[itr] = coords_ci[0]
                z_arr[itr] = coords_ci[2]
            else:
                start_B[itr] = D / np.linalg.norm(start_coords) ** 2
                end_B[itr] = 999999
                precip_arr[itr] = np.nan
                meanenergy_arr[itr] = np.nan
                x_arr[itr] = np.nan
                z_arr[itr] = np.nan
                FAC_arr[itr] = np.nan
                for k in range(16):
                    diffprecip_arr[itr][k] = np.nan
        else:
            start_B[itr] = D / np.linalg.norm(start_coords) ** 2
            end_B[itr] = 999999
            precip_arr[itr] = np.nan
            meanenergy_arr[itr] = np.nan
            x_arr[itr] = np.nan
            z_arr[itr] = np.nan
            FAC_arr[itr] = np.nan
            for k in range(16):
                diffprecip_arr[itr][k] = np.nan

    return (
        theta_arr,
        precip_arr,
        meanenergy_arr,
        diffprecip_arr,
        FAC_arr,
        Ebinedges,
        x_arr,
        z_arr,
        energybins,
        start_B,
        end_B,
    )


def plot_driving_MP_theta():

    mp_standoff_bgd, theta_mp_bgd = dayside_MP(7.0 * r_e, 8.0 * r_e, 500e3, run="BGD")
    mp_standoff_bgf, theta_mp_bgf = dayside_MP(4.0 * r_e, 5.0 * r_e, 500e3, run="BGF")
    mp_standoff_bgg, theta_mp_bgg = dayside_MP(13.6e6, 3.0 * r_e, 250e3, run="BGG")

    standoff_arr = np.array([mp_standoff_bgd, mp_standoff_bgf, mp_standoff_bgg])
    theta_arr = np.array([theta_mp_bgd, theta_mp_bgf, theta_mp_bgg])

    rho_arr = np.array([3.3, 7, 20])
    v_arr = np.array([600, 1000, 1500])
    pdyn_arr = m_p * rho_arr * 1e6 * v_arr * v_arr * 1e6 * 1e9
    B_arr = np.array([10, 20, 30])

    rho_3d = np.array([4, 7])
    v_3d = np.array([750, 1000])
    pdyn_3d = m_p * rho_3d * 1e6 * v_3d * v_3d * 1e6 * 1e9
    B_3d = np.array([10, 20])

    standoff_3d = np.array([8.01, 6.29])

    driving_arr = np.array([rho_arr, v_arr, pdyn_arr, B_arr])
    driving_3d = np.array([rho_3d, v_3d, pdyn_3d, B_3d])

    xlabel_arr = [
        "$n_\mathrm{sw}~[\mathrm{cm}^{-3}]$",
        "$v_\mathrm{sw}~[\mathrm{km/s}]$",
        "$P_\mathrm{dyn,sw}~[\mathrm{nPa}]$",
        "$-B_{\mathrm{IMF},z}~[\mathrm{nT}]$",
    ]
    outname_arr = ["n", "v", "pdyn", "B"]

    for n1 in range(4):

        fig, ax = plt.subplots(1, 1)

        ax.set_ylim(2, 10)

        ax.set_ylabel(
            "Magnetopause standoff [$R_\mathrm{E}$]",
            fontsize=14,
        )
        ax.set_xlabel(xlabel_arr[n1], fontsize=14)
        ax.grid()
        ax.plot(driving_arr[n1], standoff_arr, "o", label="Line dipole")

        ax.plot(
            driving_arr[n1],
            100 ** (1.0 / 6) * standoff_arr ** (2 / 3),
            "o",
            label="Estimated 3D",
        )
        ax.plot(driving_3d[n1], standoff_3d, "o", label="3D run")

        for idx in range(len(r_sats)):
            ax.axhline(r_sats[idx], linewidth=0.6, linestyle="dashed", color="red")
            ax.annotate(
                name_sats[idx],
                (1.01, r_sats[idx] / 8.0 - 1.0 / 4 + offset_sats[idx]),
                xycoords="axes fraction",
                color="red",
                fontsize=14,
            )

        ax.legend()

        plt.tight_layout()
        fig.savefig(
            "/wrk/users/jesuni/Figures/carrington/mp_standoff_{}.png".format(
                outname_arr[n1]
            ),
            dpi=300,
        )
        fig.savefig(
            "/wrk/users/jesuni/Figures/carrington/mp_standoff_{}.pdf".format(
                outname_arr[n1]
            )
        )
        plt.close(fig)


def plot_MP_theta():

    mp_standoff_bgd, theta_mp_bgd = dayside_MP(7.0 * r_e, 8.0 * r_e, 500e3, run="BGD")
    mp_standoff_bgf, theta_mp_bgf = dayside_MP(4.0 * r_e, 5.0 * r_e, 500e3, run="BGF")
    mp_standoff_bgg, theta_mp_bgg = dayside_MP(13.6e6, 3.0 * r_e, 250e3, run="BGG")

    fig, ax = plt.subplots(1, 1)

    ax.grid()
    ax.plot(1, mp_standoff_bgd, "o", label="Line dipole", color="black")
    ax.plot(2, mp_standoff_bgf, "o", color="black")
    ax.plot(3, mp_standoff_bgg, "o", color="black")
    ax.plot(
        1,
        100 ** (1.0 / 6) * mp_standoff_bgd ** (2.0 / 3),
        "o",
        label="3D dipole\nestimate",
        color="C1",
    )
    ax.plot(2, 100 ** (1.0 / 6) * mp_standoff_bgf ** (2.0 / 3), "o", color="C1")
    ax.plot(3, 100 ** (1.0 / 6) * mp_standoff_bgg ** (2.0 / 3), "o", color="C1")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Normal", "Moderate", "Strong"])
    ax.legend(fontsize=14)

    ax.set_xlim(0, 4)
    ax.set_ylim(2, 10)

    for idx in range(len(r_sats)):
        ax.axhline(r_sats[idx], linewidth=0.6, linestyle="dashed", color="red")
        ax.annotate(
            name_sats[idx],
            (1.01, r_sats[idx] / 8.0 - 1.0 / 4 + offset_sats[idx]),
            xycoords="axes fraction",
            color="red",
            fontsize=14,
        )

    ax.set_ylabel(
        "Magnetopause standoff [$R_\mathrm{E}$]",
        fontsize=14,
    )
    ax.set_xlabel(
        "Driving conditions",
        fontsize=14,
    )

    plt.tight_layout()
    fig.savefig("/wrk/users/jesuni/Figures/carrington/mp_standoff.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)

    ax.grid()
    ax.plot(1, theta_mp_bgd, "o", label="Normal", color="black")
    ax.plot(2, theta_mp_bgf, "o", label="Moderate", color="black")
    ax.plot(3, theta_mp_bgg, "o", label="Strong", color="black")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Normal", "Moderate", "Strong"])
    # ax.legend(fontsize=14)

    ax.set_xlim(0, 4)
    ax.set_ylim(70, 90)

    ax.set_ylabel(
        "Dayside polar cap boundary [$^\circ$]",
        fontsize=14,
    )
    ax.set_xlabel(
        "Driving conditions",
        fontsize=14,
    )

    plt.tight_layout()
    fig.savefig("/wrk/users/jesuni/Figures/carrington/mp_theta.png")
    plt.close(fig)


def dayside_MP(xstart, xstop, dx, run="BGD"):

    if run == "BGD":
        vlsvobj = pt.vlsvfile.VlsvReader(
            "/wrk/group/spacephysics/vlasiator/2D/BGD/bulk/bulk.0000500.vlsv"
        )

        r_stop = 19.1e6
        ds = 500e3
    elif run == "BGF":
        vlsvobj = pt.vlsvfile.VlsvReader(
            "/wrk/group/spacephysics/vlasiator/2D/BGF/extendvspace_restart229/bulk.0000470.vlsv"
        )
        r_stop = 18.1e6
        ds = 500e3
    elif run == "BGG":
        vlsvobj = pt.vlsvfile.VlsvReader(
            "/wrk/group/spacephysics/vlasiator/2D/BGG/denseIono_restart81/bulk/bulk.0000239.vlsv"
        )
        r_stop = 13.6e6
        ds = 250e3

    x_range = np.arange(xstart, xstop, dx)
    is_closed = np.zeros_like(x_range).astype(bool)
    for itr, x in enumerate(x_range):
        end_coord = trace_b_good(
            [x, 0, 0],
            vlsvobj=vlsvobj,
            kind="vg_b_vol",
            r_stop=r_stop,
            ds=ds,
            direction=1,
            iter_max=10000,
            run=run,
        )
        if end_coord is None:
            is_closed[itr] = False
        elif np.linalg.norm(end_coord) <= (r_stop + ds):
            is_closed[itr] = True
        else:
            is_closed[itr] = False
        # print("x = {} m, field line closed: {}".format(x, is_closed[itr]))

    xlast = x_range[is_closed][-1]
    print("Run is {}".format(run))
    print("Last closed field line at x = {} Re".format(xlast / r_e))

    ib_coords = trace_b_good(
        [xlast, 0, 0],
        vlsvobj=vlsvobj,
        kind="vg_b_vol",
        r_stop=r_stop,
        ds=ds,
        direction=1,
        iter_max=10000,
        trace_full=False,
        run=run,
    )
    surface_coords = trace_b_good(
        ib_coords,
        kind="linedipole",
        r_stop=6.371e6,
        ds=100e3,
        direction=1,
        iter_max=10000,
        trace_full=False,
        run=run,
    )

    print("Surface coords are {}".format(surface_coords / r_e))
    theta = np.rad2deg(np.arctan(surface_coords[2] / surface_coords[0]))
    print("Theta is {}".format(theta))

    return (xlast / r_e, theta)

    # return trace_b_good(
    #     [xlast, 0, 0],
    #     vlsvobj=vlsvobj,
    #     kind="vg_b_vol",
    #     r_stop=19.1e6,
    #     ds=100e3,
    #     direction=1,
    #     iter_max=10000,
    #     trace_full=True,
    # )


def trace_b_good(
    start_coords,
    vlsvobj=None,
    kind="dipole",
    r_stop=10 * r_e,
    ds=500e3,
    direction=1,
    iter_max=1000,
    trace_full=False,
    run="BGD",
):

    D = -126.2e6
    m = -8e15
    if trace_full:
        coordlist = []
    coords = np.array(start_coords, ndmin=1)
    if trace_full:
        coordlist.append(coords)

    if run == "BGG":
        X = np.arange(-125e6, 125e6, 250e3) + 125e3
        Z = np.arange(-150e6, 150e6, 250e3) + 125e3
    else:
        X = np.arange(-200e6, 200e6, 500e3) + 250e3
        Z = np.arange(-200e6, 200e6, 500e3) + 250e3

    if vlsvobj:
        cellids = vlsvobj.read_variable("CellID")
        BXint, BYint, BZint = vlsvobj.read_variable("vg_b_vol").T
        BXint = np.reshape(BXint[np.argsort(cellids)], (Z.size, X.size)).T
        BZint = np.reshape(BZint[np.argsort(cellids)], (Z.size, X.size)).T

        Bx_interpolator = interpolate.RectBivariateSpline(X, Z, BXint)
        Bz_interpolator = interpolate.RectBivariateSpline(X, Z, BZint)

    for iter in range(iter_max):
        r = np.linalg.norm(coords)
        if kind == "dipole":
            Bx = 3 * coords[0] * coords[2] * m / r ** 5
            Bz = (3 * coords[2] * coords[2] * m - m * r ** 2) / r ** 5
        elif kind == "linedipole":
            Bx = 2 * D * coords[0] * coords[2] / r ** 4
            Bz = D * (coords[2] * coords[2] - coords[0] * coords[0]) / r ** 4
        elif kind == "moddipole":
            Bx = 3 * coords[0] * coords[2] * m / r ** 5
            Bz = (3 * coords[2] * coords[2] * m - 2 * m * r ** 2) / r ** 5
        elif kind == "fg_b":
            B = vlsvobj.read_interpolated_fsgrid_variable("fg_b", coordinates=coords)
            if B is None:
                if trace_full:
                    return np.array(coordlist, ndmin=2).T
                else:
                    return None
            Bx = B[0]
            Bz = B[2]
        elif kind == "vg_b_vol":
            # B = vlsvobj.read_interpolated_variable("vg_b_vol", coordinates=coords)
            # if B is None:
            #    if trace_full:
            #        return (np.array(xlist)[:-1], np.array(zlist)[:-1])
            #    else:
            #        return None
            # Bx = B[0]
            # Bz = B[2]
            Bx = Bx_interpolator(coords[0], coords[2])
            Bz = Bz_interpolator(coords[0], coords[2])
        else:
            raise Exception

        if vlsvobj:
            Bmag = np.sqrt(Bx[0][0] ** 2 + Bz[0][0] ** 2)
            dx = Bx[0][0] / Bmag
            dz = Bz[0][0] / Bmag
        else:
            Bmag = np.sqrt(Bx ** 2 + Bz ** 2)
            dx = Bx / Bmag
            dz = Bz / Bmag
        dcoords = np.array([direction * ds * dx, 0, direction * ds * dz])

        coords = coords + dcoords
        if trace_full:
            coordlist.append(coords)

        if np.abs(np.linalg.norm(coords) - r_stop) < ds:
            break
        if np.linalg.norm(coords) <= 1.0 * r_e:
            break
        if (
            coords[0] <= np.min(X)
            or coords[0] >= np.max(X)
            or coords[2] <= np.min(Z)
            or coords[2] >= np.max(Z)
        ):
            break

    if trace_full:
        return np.array(coordlist).T
    else:
        return coords


def trace_b_xz(
    vlsvobj,
    x0,
    z0,
    ds=500e3,
    direction=-1,
    iter_max=1000,
    r_max=5.5 * r_e,
    r_min=1.0 * r_e,
):

    xlist = [x0]
    zlist = [z0]

    for iter in range(iter_max):
        # print(xlist[-1])
        # print(zlist[-1])

        # b = vlsvobj.read_interpolated_variable(
        #     "vg_b_vol", coordinates=[xlist[-1], 0, zlist[-1]]
        # )
        ci = vlsvobj.get_cellid([xlist[-1], 0, zlist[-1]])
        if np.isnan(ci):
            break
        b = vlsvobj.read_variable("vg_b_vol", cellids=int(ci))

        bmag = np.linalg.norm(b)
        dx = b[0] / bmag
        dz = b[2] / bmag

        xnew = xlist[-1] + direction * ds * dx
        znew = zlist[-1] + direction * ds * dz

        if np.isnan(xnew) or np.isnan(znew):
            break

        xlist.append(xnew)
        zlist.append(znew)

        if (np.linalg.norm([xnew, znew]) >= r_max) or (
            np.linalg.norm([xnew, znew]) <= r_min
        ):
            break

    return (np.array(xlist), np.array(zlist))


def trace_test(x0, z0):
    # x0 = np.cos(np.pi / 4) * 20e6
    # z0 = np.sin(np.pi / 4) * 20e6
    vlsvobj = pt.vlsvfile.VlsvReader(
        "/wrk/group/spacephysics/vlasiator/2D/BGD/bulk/bulk.0000400.vlsv"
    )

    tracexz = trace_b_xz(vlsvobj, x0, z0, r_max=25e6)
    B0 = vlsvobj.read_variable("vg_b_vol", cellids=int(vlsvobj.get_cellid([x0, 0, z0])))

    xlast = tracexz[0][-1]
    zlast = tracexz[1][-1]

    B = vlsvobj.read_variable(
        "vg_b_vol", cellids=int(vlsvobj.get_cellid([xlast, 0, zlast]))
    )
    b = B / np.linalg.norm(B)

    J = calc_J(vlsvobj, [xlast, zlast]) * np.linalg.norm(B0) / np.linalg.norm(B)

    return np.dot(J, b)


def calc_J(vlsvobj, coords_xz, dr=500e3):

    Bxp = vlsvobj.read_variable(
        "vg_b_vol",
        cellids=int(vlsvobj.get_cellid([coords_xz[0] + dr, 0, coords_xz[1]])),
    )
    Bxm = vlsvobj.read_variable(
        "vg_b_vol",
        cellids=int(vlsvobj.get_cellid([coords_xz[0] - dr, 0, coords_xz[1]])),
    )
    Bzp = vlsvobj.read_variable(
        "vg_b_vol",
        cellids=int(vlsvobj.get_cellid([coords_xz[0], 0, coords_xz[1] + dr])),
    )
    Bzm = vlsvobj.read_variable(
        "vg_b_vol",
        cellids=int(vlsvobj.get_cellid([coords_xz[0], 0, coords_xz[1] - dr])),
    )

    B = vlsvobj.read_variable(
        "vg_b_vol", cellids=int(vlsvobj.get_cellid([coords_xz[0], 0, coords_xz[1]]))
    )
    b = B / np.linalg.norm(B)

    dBxdz = (Bzp[0] - Bzm[0]) / 2.0 / dr
    dBydx = (Bxp[1] - Bxm[1]) / 2.0 / dr
    dBydz = (Bzp[1] - Bzm[1]) / 2.0 / dr
    dBzdx = (Bxp[2] - Bxm[2]) / 2.0 / dr

    return np.array([-dBydz, dBxdz - dBzdx, dBydx]) / mu_0


def calc_FAC(vlsvobj, coords_xz, dr=500e3):

    Bxp = vlsvobj.read_variable(
        "vg_b_vol",
        cellids=int(vlsvobj.get_cellid([coords_xz[0] + dr, 0, coords_xz[1]])),
    )
    Bxm = vlsvobj.read_variable(
        "vg_b_vol",
        cellids=int(vlsvobj.get_cellid([coords_xz[0] - dr, 0, coords_xz[1]])),
    )
    Bzp = vlsvobj.read_variable(
        "vg_b_vol",
        cellids=int(vlsvobj.get_cellid([coords_xz[0], 0, coords_xz[1] + dr])),
    )
    Bzm = vlsvobj.read_variable(
        "vg_b_vol",
        cellids=int(vlsvobj.get_cellid([coords_xz[0], 0, coords_xz[1] - dr])),
    )

    B = vlsvobj.read_variable(
        "vg_b_vol", cellids=int(vlsvobj.get_cellid([coords_xz[0], 0, coords_xz[1]]))
    )
    b = B / (np.linalg.norm(B) + 1e-27)

    dBxdz = (Bzp[0] - Bzm[0]) / 2.0 / dr
    dBydx = (Bxp[1] - Bxm[1]) / 2.0 / dr
    dBydz = (Bzp[1] - Bzm[1]) / 2.0 / dr
    dBzdx = (Bxp[2] - Bxm[2]) / 2.0 / dr

    return np.dot(b, np.array([-dBydz, dBxdz - dBzdx, dBydx]) / mu_0)
