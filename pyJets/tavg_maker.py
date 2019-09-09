import numpy as np
import pytools as pt
import linecache as lc
import pandas as pd
import os

m_p = 1.672621898e-27

wrkdir_DNR = "/wrk/sunijona/DONOTREMOVE/"

def avg_maker_slow(runid,start,stop):

    # Creates files for 3-minute time averages of dynamic pressure and density

    outputdir = wrkdir_DNR+"tavg/"+runid+"/"

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

    # range from first filenumber to last
    for n in xrange(start,stop+1):
        print("n = {}".format(n))

        # Initialise dynamic pressure and density arrays
        pdyn_arr = np.array([])
        rho_arr = np.array([])
        B_arr = np.array([])
        T_arr = np.array([])
        TPar_arr = np.array([])
        TPerp_arr = np.array([])
        v2_arr = np.array([])

        # initialise correction for missing files
        missing_file_counter = 0

        # skip process if corresponding file doesn't exist
        if "bulk."+str(n).zfill(7)+".vlsv" not in os.listdir(bulkpath):
            print("Bulk file "+str(n)+" not found, continuing")
            continue

        # range from current filenumber-180 to current filenumber+180
        for t in xrange(n-180,n+180+1):

            # exclude current filenumber from time average
            if t == n:
                continue

            # find correct file for current time step
            if runid == "AED":
                bulkname = "bulk.old."+str(t).zfill(7)+".vlsv"
            else:
                bulkname = "bulk."+str(t).zfill(7)+".vlsv"

            # iterate correction and skip processs if corresponding file doesn't exist
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
                B = vlsvobj.read_variable("B")[cellids.argsort()]
                T,TPar,TPerp = [vlsvobj.read_variable("Temperature")[cellids.argsort()],vlsvobj.read_variable("TParallel")[cellids.argsort()],vlsvobj.read_variable("TPerpendicular")[cellids.argsort()]]
            else:
                rho = vlsvobj.read_variable("proton/rho")[cellids.argsort()]
                v = vlsvobj.read_variable("proton/V")[cellids.argsort()]
                B = vlsvobj.read_variable("B")[cellids.argsort()]
                T,TPar,TPerp = [vlsvobj.read_variable("Temperature")[cellids.argsort()],vlsvobj.read_variable("TParallel")[cellids.argsort()],vlsvobj.read_variable("TPerpendicular")[cellids.argsort()]]

            # Dynamic pressure for current time step
            pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)
            Bmag = np.linalg.norm(B,axis=-1)

            # Add to time average
            if rho_arr.size == 0:
                rho_arr = rho
                pdyn_arr = pdyn
                B_arr = Bmag
                T_arr = T
                TPar_arr = TPar
                TPerp_arr = TPerp
                T_arr = np.linalg.norm(v,axis=-1)**2
            else:
                rho_arr += rho
                pdyn_arr += pdyn
                B_arr += Bmag
                T_arr += T
                TPar_arr += TPar
                TPerp_arr += TPerp
                T_arr += np.linalg.norm(v,axis=-1)**2

        # Calculate time average
        rho_arr /= (360 - missing_file_counter)
        pdyn_arr /= (360 - missing_file_counter)
        B_arr /= (360 - missing_file_counter)
        T_arr /= (360 - missing_file_counter)
        TPar_arr /= (360 - missing_file_counter)
        TPerp_arr /= (360 - missing_file_counter)
        v2_arr /= (360 - missing_file_counter)

        # Save time averages to files
        np.savetxt(outputdir+str(n)+"_rho.tavg",rho_arr)
        np.savetxt(outputdir+str(n)+"_pdyn.tavg",pdyn_arr)
        np.savetxt(outputdir+str(n)+"_B.tavg",B_arr)
        np.savetxt(outputdir+str(n)+"_T.tavg",T_arr)
        np.savetxt(outputdir+str(n)+"_TPar.tavg",TPar_arr)
        np.savetxt(outputdir+str(n)+"_TPerp.tavg",TPerp_arr)
        np.savetxt(outputdir+str(n)+"_v2.tavg",v2_arr)

    return None

def TP_maker(runid,start,stop):
    # Create files for parallel and perpendicular temperature

    outputdir = wrkdir_DNR+"TP/"+runid+"/"

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

    # range from first filenumber to last
    for n in xrange(start,stop+1):

        # find correct file for current time step
        if runid == "AED":
            bulkname = "bulk.old."+str(n).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(n).zfill(7)+".vlsv"

        # stop process for current filenumber if corresponding file doesn't exist
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