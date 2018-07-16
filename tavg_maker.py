import numpy as np
import pytools as pt
import linecache as lc
import pandas as pd
import os

m_p = 1.672621898e-27

def avg_maker_slow(runid,start,stop):

    # Creates files for 3-minute time averages of dynamic pressure and density

    outputdir = "/wrk/sunijona/DONOTREMOVE/tavg/"+runid+"/"

    # make outputdir if it doesn't already exist
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    for n in xrange(start,stop+1):

        # Initialise dynamic pressure and density arrays
        pdyn_arr = np.array([])
        rho_arr = np.array([])

        missing_file_counter = 0

        if "bulk."+str(n).zfill(7)+".vlsv" not in os.listdir(bulkpath):
            print("Bulk file "+str(n)+" not found, continuing")
            continue

        for t in xrange(n-180,n+180+1):

            if t == n:
                continue

            # find correct file for current time step
            if runid == "AED":
                bulkname = "bulk.old."+str(t).zfill(7)+".vlsv"
            else:
                bulkname = "bulk."+str(t).zfill(7)+".vlsv"

            if bulkname not in os.listdir(bulkpath):
                print("Bulk file "+str(t)+" not found, continuing")
                missing_file_counter += 1
                continue

            # Open file and read cell IDs
            vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
            cellids = vlsvobj.read_variable("CellID")

            # If file has separate populations, read the proton population
            if vlsvobj.check_variable("rho"):
                rho = vlsvobj.read_variable("rho")[cellids.argsort()]
                v = vlsvobj.read_variable("v")[cellids.argsort()]
            else:
                rho = vlsvobj.read_variable("proton/rho")[cellids.argsort()]
                v = vlsvobj.read_variable("proton/V")[cellids.argsort()]

            # Dynamic pressure for current time step
            pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

            # Add to time average
            if t == n-180:
                rho_arr = rho
                pdyn_arr = pdyn
            else:
                rho_arr += rho
                pdyn_arr += pdyn

        # Calculate time average
        rho_arr /= 360 - missing_file_counter
        pdyn_arr /= 360 - missing_file_counter

        # Save time averages to files
        np.savetxt(outputdir+str(n)+"_rho.tavg",rho_arr)
        np.savetxt(outputdir+str(n)+"_pdyn.tavg",pdyn_arr)

    return None

def avg_maker(runid,start,stop):
    # Creates files for 3-minute time averages of dynamic pressure and density

    outputdir = "/wrk/sunijona/DONOTREMOVE/tavg/"+runid+"/"

    # make outputdir if it doesn't already exist
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    # Initialise dynamic pressure and density arrays
    pdyn_arr = np.array([])
    rho_arr = np.array([])

    for n in xrange(start-180,stop+1+180):

        # find correct file for current time step
        if runid == "AED":
            bulkname = "bulk.old."+str(n).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(n).zfill(7)+".vlsv"

        # Open file and read cell IDs
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
        cellids = vlsvobj.read_variable("CellID")

        # If file has separate populations, read the proton population
        if vlsvobj.check_variable("rho"):
            rho = vlsvobj.read_variable("rho")[cellids.argsort()]
            v = vlsvobj.read_variable("v")[cellids.argsort()]
        else:
            rho = vlsvobj.read_variable("proton/rho")[cellids.argsort()]
            v = vlsvobj.read_variable("proton/V")[cellids.argsort()]

        # Dynamic pressure for current time step
        pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

        # Append dynamic pressure and density to their respective arrays
        if pdyn_arr.size == 0:
            rho_arr = np.append(rho_arr,rho)
            pdyn_arr = np.append(pdyn_arr,pdyn)
        else:
            rho_arr = np.vstack((rho_arr,rho))
            pdyn_arr = np.vstack((pdyn_arr,pdyn))

    # Initialise time average arrays
    tpdyn_arr = np.array([])
    trho_arr = np.array([])

    for itr in xrange(start,stop+1):

        # Calculate time average of pdyn and rho for current time step
        trho = (np.sum(rho_arr[itr-start:itr+361-start],axis=0)-rho_arr[itr+180-start])/360
        tpdyn = (np.sum(pdyn_arr[itr-start:itr+361-start],axis=0)-pdyn_arr[itr+180-start])/360

        # Save time averages to files
        np.savetxt(outputdir+str(itr)+"_rho.tavg",trho)
        np.savetxt(outputdir+str(itr)+"_pdyn.tavg",tpdyn)

    return None

def TP_maker(runid,start,stop):
    # Create files for parallel and perpendicular temperature

    outputdir = "/wrk/sunijona/DONOTREMOVE/TP/"+runid+"/"

    # make outputdir if it doesn't already exist
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    for n in xrange(start,stop+1):

        # find correct file for current time step
        if runid == "AED":
            bulkname = "bulk.old."+str(n).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(n).zfill(7)+".vlsv"

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file "+str(n)+" not found, continuing.")
            continue

        # Open file and read cell IDs
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
        cellids = vlsvobj.read_variable("CellID")

        # If file has separate populations, read the proton population
        if vlsvobj.check_variable("rho"):
            TPar = vlsvobj.read_variable("TParallel")
            TPerp = vlsvobj.read_variable("TPerpendicular")
        else:
            TPar = vlsvobj.read_variable("proton/TParallel")
            TPerp = vlsvobj.read_variable("proton/TPerpendicular")

        # Sort the temperatures
        TPar = TPar[cellids.argsort()]
        TPerp = TPerp[cellids.argsort()]

        # Save to file
        np.savetxt(outputdir+str(n)+".tpar",TPar)
        np.savetxt(outputdir+str(n)+".tperp",TPerp)

    return None