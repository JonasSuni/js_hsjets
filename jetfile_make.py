import pytools as pt
import numpy as np
import scipy as sc
import jet_analyser as ja

m_p = 1.672621898e-27
r_e = 6.371e+6

def custmake(runid,filenumber,outputfilename):

    # find correct bulk path for run
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    # find correct file name for run
    if runid == "AED":
        bulkname = "bulk.old."+str(filenumber).zfill(7)+".vlsv"
    else:
        bulkname = "bulk."+str(filenumber).zfill(7)+".vlsv"

    vlsvreader = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
    #open("/wrk/sunijona/VLSV/"+outputfilename,"w").close()
    vlsvwriter = pt.vlsvfile.VlsvWriter(vlsvReader=vlsvreader,file_name="/wrk/sunijona/VLSV/"+outputfilename)

    # if file has separate populations, read the proton population
    rho = vlsvreader.read_variable("rho")
    if type(rho) is not np.ndarray:
        rho = vlsvreader.read_variable("proton/rho")
        v = vlsvreader.read_variable("proton/V")
    else:
        v = vlsvreader.read_variable("v")
    
    origid = vlsvreader.read_variable("CellID")
    sorigid = vlsvreader.read_variable("CellID")
    sorigid.sort()

    # calculate the dynamic pressure and the x-direction dynamic pressure
    pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)
    pdynx = m_p*rho*(v[:,0]**2)

    timerange = xrange(filenumber-180,filenumber+180+1)

    # initialise the time average of the dynamic pressures and densities
    tpdynavg = np.zeros(pdyn.shape)

    for n_t in timerange:

        if n_t == filenumber:
            continue
        
        # find correct file for current time step
        if runid == "AED":
            tfile_name = "bulk.old."+str(n_t).zfill(7)+".vlsv"
        else:
            tfile_name = "bulk."+str(n_t).zfill(7)+".vlsv"

        # open file for current time step
        f = pt.vlsvfile.VlsvReader(bulkpath+tfile_name)
        
        # if file has separate populations, read the proton population
        trho = f.read_variable("rho")
        if type(trho) is not np.ndarray:
            trho = f.read_variable("proton/rho")
            tv = f.read_variable("proton/V")
        else:
            tv = f.read_variable("v")

        # read cellids for current time step
        cellids = f.read_variable("CellID")
        
        # dynamic pressure for current time step
        tpdyn = m_p*trho*(np.linalg.norm(tv,axis=-1)**2)

        # sort dynamic pressures
        otpdyn = tpdyn[cellids.argsort()]

        tpdynavg = np.add(tpdynavg,otpdyn)

    tpdynavg /= len(timerange)-1

    # prevent divide-by-0 errors
    tpdynavg[tpdynavg == 0.0] = 1.0e-27

    # sort dynamic pressure, x-directional dynamic pressure and density
    spdyn = pdyn[origid.argsort()]
    spdynx = pdynx[origid.argsort()]
    srho = rho[origid.argsort()]

    # pull values of pdyn and rho in boxre area [14,16,-4,4]
    spdyn_sw,srho_sw = ja.ci2vars_nofile([spdyn,srho],sorigid,ja.restrict_area(vlsvreader,[14,16],[-4,4]))

    # calculate averages of said pdyn and rho to use as solar wind parameters
    pdyn_sw = np.mean(spdyn_sw)
    rho_sw = np.mean(srho_sw)

    # calculate ratios
    npdynx = spdynx/pdyn_sw
    nrho = srho/rho_sw
    tapdyn = np.divide(spdyn,tpdynavg)

    # density to 1/cm^3
    srho /= 1.0e+6

    # dynamic pressure to nPa
    spdyn /= 1.0e-9
    tpdynavg /= 1.0e-9

    # write the new variables to the writer file 
    vlsvwriter.write(data=npdynx,name="npdynx",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=nrho,name="nrho",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=tapdyn,name="tapdyn",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=spdyn,name="spdyn",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=sorigid,name="CellID",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=srho,name="rho",tag="VARIABLE",mesh="SpatialGrid")

    # copy variables from reader file to writer file
    #vlsvwriter.copy_variables(vlsvreader)
    
    vlsvwriter.close()
    
    return None

def pahkmake(file_number,runid,halftimewidth,sw_params=[1.0e+6,750.0e+3]):
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

    # solar wind density and dynamic pressure
    rho_sw = sw_params[0]
    vx_sw = sw_params[1]
    pdyn_sw = m_p*rho_sw*(vx_sw**2)

    timerange = xrange(file_number-halftimewidth,file_number+halftimewidth+1)

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

    spdynx_sw,srho_sw = ja.ci2vars_nofile([spdynx,srho],sorigid,ja.restrict_area(vlsvreader,[14,16],[-4,4]))

    pdyn_sw = np.mean(spdynx_sw)
    rho_sw = np.mean(srho_sw)

    # calculate ratios
    npdynx = spdynx/pdyn_sw
    npdyn = spdyn/pdyn_sw
    nrho = srho/rho_sw
    tapdyn = spdyn/tpdynavg
    tarho = srho/trhoavg
    tapdynx = spdynx/tpdynxavg

    # density to 1/cm^3
    srho /= 1.0e+6

    # dynamic pressure to nPa
    spdyn /= 1.0e-9
    spdynx /= 1.0e-9

    # time averages to 1/cm^3 and nPa
    trhoavg /= 1.0e+6
    tpdynavg /= 1.0e-9
    tpdynxavg /= 1.0e-9

    runid_map = map(ord,runid)
    identifiers = np.array([runid_map[0],runid_map[1],runid_map[2],file_number,halftimewidth])
    identifiers = np.pad(identifiers,(0,5*(rho.size-1)),"wrap")
    identifiers = np.reshape(identifiers,(rho.size,5))

    # write the new variables to the writer file 
    vlsvwriter.write(data=npdyn,name="npdyn",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=npdynx,name="npdynx",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=nrho,name="nrho",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=tapdyn,name="tapdyn",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=tarho,name="tarho",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=sorigid,name="CellID",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=srho,name="srho",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=identifiers,name="identifiers",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=spdyn,name="spdyn",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=trhoavg,name="trhoavg",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=tpdynavg,name="tpdynavg",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=tpdynxavg,name="tpdynxavg",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=spdynx,name="spdynx",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=tapdynx,name="tapdynx",tag="VARIABLE",mesh="SpatialGrid")

    # copy variables from reader file to writer file
    vlsvwriter.copy_variables(vlsvreader)
    
    vlsvwriter.close()
    
    return None


def pfmake(file_number,runid,sw_params=[1.0e+6,750.0e+3],test_bool=False):
    # creates temporary vlsv file with new variable: ratio of x-directional dynamic pressure
    # and solar wind dynamic pressure
    
    # find correct file based on file number and run id
    file_nr = str(file_number).zfill(7)
    file_path = "/proj/vlasov/2D/"+runid+"/bulk/bulk."+file_nr+".vlsv"

    # open vlsv files for reading and writing
    vlsvreader = pt.vlsvfile.VlsvReader(file_path)
    vlsvwriter = pt.vlsvfile.VlsvWriter(vlsvReader=vlsvreader,file_name="VLSV/temp_plaschke.vlsv")
    
    rho = vlsvreader.read_variable("rho")
    v = vlsvreader.read_variable("v")
    cellids = vlsvreader.read_variable("CellID")

    # calculate the dynamic pressure and the x-direction dynamic pressure
    pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)
    pdynx = m_p*rho*(v[:,0]**2)

    # solar wind density and dynamic pressure
    rho_sw = sw_params[0]
    vx_sw = sw_params[1]
    pdyn_sw = m_p*rho_sw*(vx_sw**2)

    # calculate the ratios of the dynamic pressures and density with respective solar wind values
    npdyn = pdyn/pdyn_sw
    npdynx = pdynx/pdyn_sw
    nrho = rho/rho_sw

    if test_bool:

        pdyn[np.where(np.in1d(cellids,ja.restrict_area(vlsvreader,[14,16],[-4,4])))] = 42

    # write the new variables to the writer file
    vlsvwriter.write(data=npdyn,name="npdyn",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=npdynx,name="npdynx",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=nrho,name="nrho",tag="VARIABLE",mesh="SpatialGrid")
    vlsvwriter.write(data=pdyn,name="pdyn",tag="VARIABLE",mesh="SpatialGrid")

    # copy variables from reader file to writer file
    vlsvwriter.copy_variables(vlsvreader)
    
    vlsvwriter.close()
    
    return None