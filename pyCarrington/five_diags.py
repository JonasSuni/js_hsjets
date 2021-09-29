import numpy as np
import pytools as pt

r_e = 6.371e6
mu_0 = 4*np.pi*1e-7

def map_surface_to_ib(theta, ib):

    return np.arcsin(np.sqrt(ib * np.sin(theta) * np.sin(theta) / r_e))


def trace_b_xz(
    vlsvobj, x0, z0, ds=500e3, direction=-1, iter_max=1000, r_trace=5.5 * r_e
):

    xlist = [x0]
    zlist = [z0]

    for iter in range(iter_max):
        #print(xlist[-1])
        #print(zlist[-1])

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

        if np.linalg.norm([xnew, znew]) >= r_trace:
            break

    return (np.array(xlist), np.array(zlist))


def trace_test(x0,z0):
    #x0 = np.cos(np.pi / 4) * 20e6
    #z0 = np.sin(np.pi / 4) * 20e6
    vlsvobj = pt.vlsvfile.VlsvReader(
        "/wrk/group/spacephysics/vlasiator/2D/BGD/bulk/bulk.0000400.vlsv"
    )

    tracexz = trace_b_xz(vlsvobj, x0, z0, r_trace=25e6)
    B0 = vlsvobj.read_variable("vg_b_vol", cellids=int(vlsvobj.get_cellid([x0, 0, z0])))

    xlast = tracexz[0][-1]
    zlast = tracexz[1][-1]

    B = vlsvobj.read_variable("vg_b_vol", cellids=int(vlsvobj.get_cellid([xlast, 0, zlast])))
    b = B/np.linalg.norm(B)

    J = calc_J(vlsvobj,[xlast,zlast])*np.linalg.norm(B0)/np.linalg.norm(B)

    return np.dot(J,b)

def calc_J(vlsvobj,coords_xz,dr=500e3):

    Bxp = vlsvobj.read_variable("vg_b_vol", cellids=int(vlsvobj.get_cellid([coords_xz[0]+dr, 0, coords_xz[1]])))
    Bxm = vlsvobj.read_variable("vg_b_vol", cellids=int(vlsvobj.get_cellid([coords_xz[0]-dr, 0, coords_xz[1]])))
    Bzp = vlsvobj.read_variable("vg_b_vol", cellids=int(vlsvobj.get_cellid([coords_xz[0], 0, coords_xz[1]+dr])))
    Bzm = vlsvobj.read_variable("vg_b_vol", cellids=int(vlsvobj.get_cellid([coords_xz[0], 0, coords_xz[1]-dr])))

    dBxdz = (Bzp[0]-Bzm[0])/2.0/dr
    dBydx = (Bxp[1]-Bxm[1])/2.0/dr
    dBydz = (Bzp[1]-Bzm[1])/2.0/dr
    dBzdx = (Bxp[2]-Bxm[2])/2.0/dr

    return np.array([-dBydz,dBxdz-dBzdx,dBydx])/mu_0