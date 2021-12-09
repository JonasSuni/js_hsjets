import numpy as np
import pytools as pt
from scipy import interpolate
import matplotlib.pyplot as plt

r_e = 6.371e6
mu_0 = 4 * np.pi * 1e-7


def map_surface_to_ib(theta, ib):

    return np.arcsin(np.sqrt(ib * np.sin(theta) * np.sin(theta) / r_e))


def colormap_diff_precip(run="BGF"):

    if run == "BGD":
        vlsvobj = pt.vlsvfile.VlsvReader(
            "/wrk/group/spacephysics/vlasiator/2D/BGD/bulk/bulk.0000450.vlsv"
        )

    elif run == "BGF":
        vlsvobj = pt.vlsvfile.VlsvReader(
            "/wrk/group/spacephysics/vlasiator/2D/BGF/extendvspace_restart229/bulk.0000450.vlsv"
        )
    elif run == "BGG":
        vlsvobj = pt.vlsvfile.VlsvReader(
            "/wrk/group/spacephysics/vlasiator/2D/BGG/denseIono_restart81/bulk/bulk.0000154.vlsv"
        )

    for k in range(16):
        pt.plot.plot_colormap(
            vlsvobj=vlsvobj,
            outputdir="/wrk/users/jesuni/Figures/carrington/",
            var="proton/vg_precipitationdifferentialflux",
            vmin=1,
            vmax=10 ** 5,
            operator="{}".format(k),
            run=run,
            step=450,
            boxre=[-10, 10, -10, 10],
            Earth=1,
        )


def plot_precip(plot_diff=False):

    plt.ioff()

    (
        theta,
        precip_bgd,
        meanenergy_bgd,
        difflux_bgd,
        binedges_bgd,
        x_bgd,
        z_bgd,
    ) = precipitation_diag("BGD")
    (
        theta,
        precip_bgf,
        meanenergy_bgf,
        difflux_bgf,
        binedges_bgf,
        x_bgf,
        z_bgf,
    ) = precipitation_diag("BGF")
    (
        theta,
        precip_bgg,
        meanenergy_bgg,
        difflux_bgg,
        binedges_bgg,
        x_bgg,
        z_bgg,
    ) = precipitation_diag("BGG")

    fig, ax = plt.subplots(1, 1)

    ax.grid()
    ax.semilogy(theta, precip_bgd, label="Normal")
    ax.semilogy(theta, precip_bgf, label="Moderate")
    ax.semilogy(theta, precip_bgg, label="Strong")
    ax.legend(fontsize=14)

    ax.set_xlim(60, 120)
    ax.set_ylim(10 ** 0, 10 ** 10)
    ax.invert_xaxis()

    ax.set_xlabel("$\\theta$ [$^\\circ$]", fontsize=14)
    ax.set_ylabel(
        "Precipitation integral energy flux [$\mathrm{keV},\mathrm{cm}^{-2},\mathrm{s}^{-1},\mathrm{sr}^{-1}$]",
        fontsize=12,
    )

    plt.tight_layout()
    fig.savefig("/wrk/users/jesuni/Figures/carrington/precipitation_integralflux.png")
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
                "Precipitation diff number flux [$\mathrm{cm}^{-2},\mathrm{s}^{-1},\mathrm{sr}^{-1}$]",
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

    plt.ion()


def precipitation_diag(run):

    if run == "BGD":
        vlsvobj = pt.vlsvfile.VlsvReader(
            "/wrk/group/spacephysics/vlasiator/2D/BGD/bulk/bulk.0000450.vlsv"
        )

        r_stop = 19.1e6
    elif run == "BGF":
        vlsvobj = pt.vlsvfile.VlsvReader(
            "/wrk/group/spacephysics/vlasiator/2D/BGF/extendvspace_restart229/bulk.0000450.vlsv"
        )
        r_stop = 18.1e6
    elif run == "BGG":
        vlsvobj = pt.vlsvfile.VlsvReader(
            "/wrk/group/spacephysics/vlasiator/2D/BGG/denseIono_restart81/bulk/bulk.0000154.vlsv"
        )
        r_stop = 14.6e6

    theta_arr = np.linspace(120, 60, 60)
    precip_arr = np.zeros_like(theta_arr)
    meanenergy_arr = np.zeros_like(theta_arr)
    diffprecip_arr = np.zeros((theta_arr.size, 16), dtype=float)
    x_arr = np.zeros_like(theta_arr)
    z_arr = np.zeros_like(theta_arr)

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
                r_stop=r_stop + 2000e3,
                ds=100e3,
                direction=-1,
                iter_max=10000,
                trace_full=False,
                run=run,
            )
            if np.linalg.norm(end_coords) >= (r_stop + 2000e3 - 500e3):
                ci = vlsvobj.get_cellid(end_coords)
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

                coords_ci = vlsvobj.get_cell_coordinates(ci)
                x_arr[itr] = coords_ci[0]
                z_arr[itr] = coords_ci[2]
            else:
                precip_arr[itr] = np.nan
                meanenergy_arr[itr] = np.nan
                x_arr[itr] = np.nan
                z_arr[itr] = np.nan
        else:
            precip_arr[itr] = np.nan
            meanenergy_arr[itr] = np.nan
            x_arr[itr] = np.nan
            z_arr[itr] = np.nan
            for k in range(16):
                diffprecip_arr[itr][k] = np.nan

    return (
        theta_arr,
        precip_arr,
        meanenergy_arr,
        diffprecip_arr,
        Ebinedges,
        x_arr,
        z_arr,
    )


def plot_MP_theta():

    mp_standoff_bgd, theta_mp_bgd = dayside_MP(7.0 * r_e, 8.0 * r_e, 500e3, run="BGD")
    mp_standoff_bgf, theta_mp_bgf = dayside_MP(4.0 * r_e, 5.0 * r_e, 500e3, run="BGF")
    mp_standoff_bgg, theta_mp_bgg = dayside_MP(14.6e6, 3.0 * r_e, 250e3, run="BGG")

    fig, ax = plt.subplots(1, 1)

    ax.grid()
    ax.plot(1, mp_standoff_bgd, "o", label="Normal")
    ax.plot(2, mp_standoff_bgf, "o", label="Moderate")
    ax.plot(3, mp_standoff_bgg, "o", label="Strong")
    ax.legend(fontsize=14)

    ax.set_xlim(0, 4)
    ax.set_ylim(2, 10)

    ax.set_ylabel(
        "Magnetopause standoff [$R_\mathrm{E}$]",
        fontsize=14,
    )

    plt.tight_layout()
    fig.savefig("/wrk/users/jesuni/Figures/carrington/mp_standoff.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)

    ax.grid()
    ax.plot(1, theta_mp_bgd, "o", label="Normal")
    ax.plot(2, theta_mp_bgf, "o", label="Moderate")
    ax.plot(3, theta_mp_bgg, "o", label="Strong")
    ax.legend(fontsize=14)

    ax.set_xlim(0, 4)
    ax.set_ylim(70, 90)

    ax.set_ylabel(
        "Dayside polar cap boundary [$^\circ$]",
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
    elif run == "BGF":
        vlsvobj = pt.vlsvfile.VlsvReader(
            "/wrk/group/spacephysics/vlasiator/2D/BGF/extendvspace_restart229/bulk.0000470.vlsv"
        )
        r_stop = 18.1e6
    elif run == "BGG":
        vlsvobj = pt.vlsvfile.VlsvReader(
            "/wrk/group/spacephysics/vlasiator/2D/BGG/denseIono_restart81/bulk/bulk.0000154.vlsv"
        )
        r_stop = 14.6e6

    x_range = np.arange(xstart, xstop, dx)
    is_closed = np.zeros_like(x_range).astype(bool)
    for itr, x in enumerate(x_range):
        end_coord = trace_b_good(
            [x, 0, 0],
            vlsvobj=vlsvobj,
            kind="vg_b_vol",
            r_stop=r_stop,
            ds=100e3,
            direction=1,
            iter_max=10000,
            run=run,
        )
        if end_coord is None:
            is_closed[itr] = False
        elif np.linalg.norm(end_coord) <= (r_stop + 500e3):
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
        ds=100e3,
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
    ds=100e3,
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
        print(X.shape)
        print(Z.shape)
    else:
        X = np.arange(-200e6, 200e6, 500e3) + 250e3
        Z = np.arange(-200e6, 200e6, 500e3) + 250e3
        print(X.shape)
        print(Z.shape)

    if vlsvobj:
        cellids = vlsvobj.read_variable("CellID")
        BXint, BYint, BZint = vlsvobj.read_variable("vg_b_vol").T
        BXint = np.reshape(BXint[np.argsort(cellids)], (X.size, Z.size)).T
        BZint = np.reshape(BZint[np.argsort(cellids)], (X.size, Z.size)).T

        print(BXint.shape)
        print(BZint.shape)

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

    dBxdz = (Bzp[0] - Bzm[0]) / 2.0 / dr
    dBydx = (Bxp[1] - Bxm[1]) / 2.0 / dr
    dBydz = (Bzp[1] - Bzm[1]) / 2.0 / dr
    dBzdx = (Bxp[2] - Bxm[2]) / 2.0 / dr

    return np.array([-dBydz, dBxdz - dBzdx, dBydx]) / mu_0
