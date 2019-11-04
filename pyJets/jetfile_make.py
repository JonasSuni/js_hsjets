import pytools as pt
import numpy as np
import scipy as sc
import jet_analyser as ja

m_p = 1.672621898e-27
r_e = 6.371e+6

def pahkmake(file_number,runid,halftimewidth):
    # creates temporary vlsv file with new variables: x-directional dynamic pressure/solar wind
    # dynamic pressure, dynamic pressure/time averaged dynamic pressure, density/time averaged
    # density

    # find correct file based on file number and run id
    file_nr = str(file_number).zfill(7)
    file_path = "/proj/vlasov/2D/"+runid+"/bulk/bulk."+file_nr+".vlsv"

    # open vlsv files for reading and writing
    vlsvreader = pt.vlsvfile.VlsvReader(file_path)
    vlsvwriter = pt.vlsvfile.VlsvWriter(vlsvReader=vlsvreader,file_name="VLSV/temp_all.vlsv")

    rho = vlsvreader.read_variable("rho")
    v = vlsvreader.read_variable("v")

    origid = vlsvreader.read_variable("CellID")
    sorigid = vlsvreader.read_variable("CellID")
    sorigid.sort()

    # calculate the dynamic pressure and the x-direction dynamic pressure
    pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)
    pdynx = m_p*rho*(v[:,0]**2)

    timerange = range(file_number-halftimewidth,file_number+halftimewidth+1)

    # initialise the time average of the dynamic pressures and densities
    tpdynavg = np.zeros(len(pdyn))
    tpdynxavg = np.zeros(len(pdynx))
    trhoavg = np.zeros(len(rho))

    for n in timerange:

        if n == file_number:
            continue

        # initialise new variables for current iteration (not necessarily necessary)
        f = 0
        trho = 0
        tv = 0
        tpdyn = 0

        # find correct file for current time step
        tfile_nr = str(n).zfill(7)
        tfile_p = "/proj/vlasov/2D/"+runid+"/bulk/bulk."+tfile_nr+".vlsv"

        # open file for current time step
        f = pt.vlsvfile.VlsvReader(tfile_p)

        trho = f.read_variable("rho")
        #tv = f.read_variable("v").transpose()
        tv = f.read_variable("v")

        # read cellids for current time step
        cellids = f.read_variable("CellID")

        # dynamic pressure for current time step
        tpdyn = m_p*trho*(np.linalg.norm(tv,axis=-1)**2)
        tpdynx = m_p*trho*(tv[:,0]**2)

        # sort dynamic pressures
        otpdyn = tpdyn[cellids.argsort()]
        otpdynx = tpdynx[cellids.argsort()]
        otrho = trho[cellids.argsort()]

        # prevent divide by zero errors
        otpdyn[otpdyn == 0.0] = 1.0e-27
        otpdynx[otpdynx == 0.0] = 1.0e-27
        otrho[otrho == 0.0] = 1.0e-27

        trhoavg += otrho
        tpdynavg += otpdyn
        tpdynxavg += otpdynx

    # calculate the time averages of the dynamic pressures and densities
    tpdynavg /= len(timerange)-1
    tpdynxavg /= len(timerange)-1
    trhoavg /= len(timerange)-1

    # sort dynamic pressure, x-directional dynamic pressure and density
    spdyn = pdyn[origid.argsort()]
    spdynx = pdynx[origid.argsort()]
    srho = rho[origid.argsort()]

    sw_pars = ja.sw_par_dict("ABA")
    pdyn_sw = sw_pars[3]
    rho_sw = sw_pars[0]

    # calculate ratios
    npdynx = spdynx/pdyn_sw
    npdyn = spdyn/pdyn_sw
    nrho = srho/rho_sw
    tapdyn = spdyn/tpdynavg
    tarho = srho/trhoavg
    tapdynx = spdynx/tpdynxavg

    # write the new variables to the writer file
    vlsvwriter.write(data=sorigid,name="CellID",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=srho,name="srho",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=spdyn,name="spdyn",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=trhoavg,name="trhoavg",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=tpdynavg,name="tpdynavg",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=spdynx,name="spdynx",tag="VARIABLE",mesh="SpatialGrid")

    # copy variables from reader file to writer file
    vlsvwriter.copy_variables(vlsvreader)

    vlsvwriter.close()

    return None
