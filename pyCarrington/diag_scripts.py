import pytools as pt
import numpy as np


def check_wall_hit(vlsvobj):

    cellids = vlsvobj.read_variable("CellID").astype(int)
    cellids.sort()

    max_vcoords = []
    wall_hit_cells = []

    for ci in cellids:
        if (ci - 1) % 1000 == 0:
            print(ci)
        velocity_cell_map = vlsvobj.read_velocity_cells(ci)
        velocity_cell_ids = np.array([*velocity_cell_map])
        velocity_cell_coordinates = vlsvobj.get_velocity_cell_coordinates(
            velocity_cell_ids
        )
        try:
            maxv = np.max(np.abs(velocity_cell_coordinates))
            max_vcoords.append(maxv)
            if maxv >= 3960e3:
                print("WALL HIT")
                wall_hit_cells.append(ci)
        except:
            continue

    if np.max(max_vcoords) >= 3960e3:
        print("WALL HIT")
    else:
        print("Wall not hit")

    return np.array(max_vcoords)
