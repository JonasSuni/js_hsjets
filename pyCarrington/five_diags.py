import numpy as np
import pytools as pt

r_e = 6.371e6
mu_0 = 4 * np.pi * 1e-7


def map_surface_to_ib(theta, ib):

    return np.arcsin(np.sqrt(ib * np.sin(theta) * np.sin(theta) / r_e))


def dayside_MP(xstart, xstop):

    vlsvobj = pt.vlsvfile.VlsvReader(
        "/wrk/group/spacephysics/vlasiator/2D/BGD/bulk/bulk.0000500.vlsv"
    )

    r_stop = 19.1e6

    x_range = np.arange(xstart, xstop, 500e3)
    is_closed = np.zeros_like(x_range).astype(bool)
    for itr, x in enumerate(x_range):
        end_coord = trace_b_good(
            [x, 0, 0],
            vlsvobj=vlsvobj,
            kind="vg_b_vol",
            r_stop=r_stop,
            ds=100e3,
            direction=-1,
            iter_max=10000,
        )
        if end_coord is None:
            is_closed[itr] = False
        elif np.linalg.norm(end_coord) <= (r_stop + 500e3):
            is_closed[itr] = True
        else:
            is_closed[itr] = False
        print("x = {}, field line closed: {}".format(x / r_e, is_closed[itr]))

    xlast = x_range[is_closed][-1]
    print("Last closed field line at x = {}".format(xlast))

    return trace_b_good(
        [xlast, 0, 0],
        vlsvobj=vlsvobj,
        kind="vg_b_vol",
        r_stop=19.1e6,
        ds=100e3,
        direction=-1,
        iter_max=10000,
        trace_full=True,
    )


def trace_b_good(
    start_coords,
    vlsvobj=None,
    kind="dipole",
    r_stop=10 * r_e,
    ds=500e3,
    direction=1,
    iter_max=1000,
    trace_full=False,
):

    D = -126.2e6
    m = -8e15
    if trace_full:
        xlist = []
        zlist = []
    coords = np.array(start_coords, ndmin=1)
    if trace_full:
        xlist.append(coords[0])
        zlist.append(coords[2])

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
                    return (np.array(xlist)[:-1], np.array(zlist)[:-1])
                else:
                    return None
            Bx = B[0]
            Bz = B[2]
        elif kind == "vg_b_vol":
            B = vlsvobj.read_interpolated_variable("vg_b_vol", coordinates=coords)
            if B is None:
                if trace_full:
                    return (np.array(xlist)[:-1], np.array(zlist)[:-1])
                else:
                    return None
            Bx = B[0]
            Bz = B[2]
        else:
            raise Exception

        Bmag = np.sqrt(Bx ** 2 + Bz ** 2)
        dx = Bx / Bmag
        dz = Bz / Bmag
        dcoords = np.array([direction * ds * dx, 0, direction * ds * dz])

        coords = coords + dcoords
        if trace_full:
            xlist.append(coords[0])
            zlist.append(coords[2])

        if np.abs(np.linalg.norm(coords) - r_stop) < ds:
            break
        if np.linalg.norm(coords) <= 1.0 * r_e:
            break

    if trace_full:
        return (np.array(xlist), np.array(zlist))
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
