import numpy as np
import pytools as pt

r_e = 6.371e6


def map_surface_to_ib(theta, ib):

    return np.arcsin(np.sqrt(ib * np.sin(theta) * np.sin(theta) / r_e))


def trace_b_xz(
    vlsvobj, x0, z0, ds=500e3, direction=-1, iter_max=1000, r_trace=5.5 * r_e
):

    xlist = [x0]
    zlist = [z0]

    for iter in range(iter_max):

        b = vlsvobj.read_interpolated_variable(
            "vg_b_vol", coordinates=[xlist[-1], 0, zlist[-1]]
        )

        if np.isnan(b[0]):
            break

        bmag = np.linalg.norm(b)
        dx = b[0] / bmag
        dz = b[2] / bmag

        xnew = xlist[-1] + direction * ds * dx
        znew = zlist[-1] + direction * ds * dz

        xlist.append(xnew)
        zlist.append(znew)

        if np.linalg.norm([xnew, znew]) >= r_trace:
            break

    return (np.array(xlist), np.array(zlist))
