import numpy as np
import pytools as pt
import scipy
import pandas as pd
import jet_scripts as js
import jet_analyser as ja
import jetfile_make as jfm
import os
import jet_scripts as js
import copy
import matplotlib.pyplot as plt
import plot_contours as pc
import scipy.constants as sc
import jet_io as jio

m_p = 1.672621898e-27
r_e = 6.371e+6

'''
This file should contain all functions and scripts related to finding, sorting, tracking and statistically analysing SLAMS.
'''

def visual_slams_finder(runid,filenumber,boxre=[6,18,-8,6],vmax=1.5,plaschke=1.0,sw=1.0):

    if runid in ["AEA","AEC"]:
        B_sw = 10.0e-9
    else:
        B_sw = 5.0e-9

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    if runid == "AED":
        bulkname = "bulk.old."+str(filenumber).zfill(7)+".vlsv"
    else:
        bulkname = "bulk."+str(filenumber).zfill(7)+".vlsv"

    if runid == "BFD":
        pass_vars=["proton/rho","proton/V","CellID","B"]
    else:
        pass_vars=["rho","v","CellID","B"]

    pt.plot.plot_colormap(filename=bulkpath+bulkname,draw=1,usesci=0,lin=1,cbtitle="",boxre=boxre,colormap="parula",vmin=0,vmax=vmax,expression=pc.expr_pdyn,external=ext_slams,pass_vars=pass_vars,ext_pars=[runid,filenumber,plaschke,sw,B_sw])

def ext_slams(ax,XmeshPass,YmeshPass,extmaps,ext_pars):

    rho = extmaps[0]
    v = extmaps[1]
    cellids = extmaps[2]
    B = extmaps[3]

    vx = v[:,:,0]
    vmag = np.linalg.norm(v,axis=-1)

    Bmag = np.linalg.norm(B,axis=-1)

    shp = rho.shape

    runid = ext_pars[0]
    filenumber = ext_pars[1]

    sw_pars = ja.sw_par_dict()[runid]

    tpdynavg = np.loadtxt("/wrk/sunijona/DONOTREMOVE/tavg/"+runid+"/"+str(filenumber)+"_pdyn.tavg")

    vmag = vmag.flatten()
    vx = vx.flatten()
    rho = rho.flatten()
    cellids = cellids.flatten()
    Bmag = Bmag.flatten()

    pdyn = m_p*rho*(vmag**2)
    pdyn_x = m_p*rho*(vx**2)

    spdyn = pdyn[np.argsort(cellids)]
    spdyn_x = pdyn_x[np.argsort(cellids)]
    srho = rho[np.argsort(cellids)]
    sBmag = Bmag[np.argsort(cellids)]

    pdyn_sw = m_p*sw_pars[0]*(sw_pars[1]**2)

    slams = np.ma.masked_greater(sBmag,ext_pars[3]*ext_pars[4])
    slams.mask[spdyn < ext_pars[2]*pdyn_sw] = False
    slams.mask[srho > 3*sw_pars[0]] = False
    slams.fill_value = 0
    slams[slams.mask == False] = 1

    contour = ax.contour(XmeshPass,YmeshPass,np.reshape(slams.filled(),shp),[0.5],linewidths=1.0, colors="black")

    return None

def make_slams_mask(filenumber,runid,boxre=[6,18,-8,6]):
    # finds cellids of cells that fulfill the specified criterion and the specified
    # X,Y-limits

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    if runid == "AED":
        bulkname = "bulk.old."+str(filenumber).zfill(7)+".vlsv"
    else:
        bulkname = "bulk."+str(filenumber).zfill(7)+".vlsv"

    if bulkname not in os.listdir(bulkpath):
        print("Bulk file "+str(filenumber)+" not found, exiting.")
        return 1

    # open vlsv file for reading
    vlsvreader = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

    origid = vlsvreader.read_variable("CellID")
    sorigid = origid[np.argsort(origid)]

    # if file has separate populations, read proton population
    if type(vlsvreader.read_variable("rho")) is not np.ndarray:
        rho = vlsvreader.read_variable("proton/rho")[np.argsort(origid)]
        v = vlsvreader.read_variable("proton/V")[np.argsort(origid)]
    else:
        rho = vlsvreader.read_variable("rho")[np.argsort(origid)]
        v = vlsvreader.read_variable("v")[np.argsort(origid)]

    B = vlsvreader.read_variable("B")[np.argsort(origid)]
    Bmag = np.linalg.norm(B,axis=-1)

    # dynamic pressure
    pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

    sw_pars = sw_par_dict()[runid]
    rho_sw = sw_pars[0]
    v_sw = sw_pars[1]
    pdyn_sw = m_p*rho_sw*(v_sw**2)
    if runid in ["AEA","AEC"]:
        B_sw = 10.0e-9
    else:
        B_sw = 5.0e-9

    # make custom SLAMS mask
    slams = np.ma.masked_greater(Bmag,1.25*B_sw)
    slams.mask[pdyn < 1.25*pdyn_sw] = False
    slams.mask[rho > 3*rho_sw] = False

    # discard unmasked cellids
    masked_ci = np.ma.array(sorigid,mask=~slams.mask).compressed()

    if not os.path.exists("SLAMS/masks/"+runid+"/"):
        try:
            os.makedirs("SLAMS/masks/"+runid+"/")
        except OSError:
            pass

    # if boundaries have been set, discard cellids outside boundaries
    if not not boxre:
        masked_ci = np.intersect1d(masked_ci,restrict_area(vlsvreader,boxre[0:2],boxre[2:4]))
        np.savetxt("SLAMS/masks/"+runid+"/"+str(filenumber)+".mask",masked_ci)
        return masked_ci
    else:
        np.savetxt("SLAMS/masks/"+runid+"/"+str(filenumber)+".mask",masked_ci)
        return masked_ci

def sort_slams(vlsvobj,cells,min_size=0,max_size=3000,neighborhood_reach=[1,1]):
    # sort masked cells into events based on proximity in X,Y-space

    # initialise list of events and current event
    events = []
    curr_event = np.array([],dtype=int)

    for cell in cells:

        # check if cell already in list of events
        bl_a = False
        for event in events:
            if cell in event:
                bl_a = True
        if bl_a:
            continue

        # number of times to search for more neighbors
        it_range = xrange(200)

        # initialise current event
        curr_event = np.array([cell])

        for n in it_range:

            curr_event_size = curr_event.size

            # find neighbors within the confines of the mask
            curr_event = np.unique(np.append(curr_event,np.intersect1d(cells,ja.get_neighbors(vlsvobj,curr_event,neighborhood_reach))))

            # exit loop if all valid neighbors found
            if curr_event_size == curr_event.size:
                break

        # cast cellids of current event to int and append to list of events
        curr_event = curr_event.astype(int)
        events.append(curr_event)

    # remove events smaller than the minimum size and larger than maximum size
    events_culled = [slams for slams in events if slams.size >= min_size and slams.size <= max_size]

    return events_culled

def slams_maker(runid,start,stop,boxre=[6,18,-8,6],maskfile=False):

    outputdir = "/homeappl/home/sunijona/SLAMS/events/"+runid+"/"

    # make outputdir if it doesn't already exist
    if not os.path.exists(outputdir):
        try:
            os.makedirs(outputdir)
        except OSError:
            pass

    for file_nr in xrange(start,stop+1):

        # find correct file based on file number and run id
        if runid in ["AEC","AEF","BEA","BEB"]:
            bulkpath = "/proj/vlasov/2D/"+runid+"/"
        elif runid == "AEA":
            bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
        else:
            bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

        if runid == "AED":
            bulkname = "bulk.old."+str(file_nr).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(file_nr).zfill(7)+".vlsv"

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file "+str(file_nr)+" not found, continuing")
            continue

        # open vlsv file for reading
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

        # create mask
        if maskfile:
            msk = np.loadtxt("SLAMS/masks/"+runid+"/"+str(file_nr)+".mask").astype(int)
        else:
            msk = make_slams_mask(file_nr,runid,boxre)

        print(len(msk))
        print("Current file number is " + str(file_nr))

        # sort jets
        slams = sort_slams(vlsvobj,msk,10,1000,[2,2])

        # erase contents of output file
        open(outputdir+str(file_nr)+".events","w").close()

        # open output file
        fileobj = open(outputdir+str(file_nr)+".events","a")

        # write jets to outputfile
        for slam in slams:

            fileobj.write(",".join(map(str,slam))+"\n")

        fileobj.close()

    return None

def track_slams(runid,start,stop,threshold=0.5):

    # find correct file based on file number and run id
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    if runid == "AED":
        bulkname = "bulk.old."+str(start).zfill(7)+".vlsv"
    else:
        bulkname = "bulk."+str(start).zfill(7)+".vlsv"

    if bulkname not in os.listdir(bulkpath):
        print("Bulk file "+str(start)+" not found, exiting")
        return 1

    # Create outputdir if it doesn't already exist
    if not os.path.exists("/homeappl/home/sunijona/SLAMS/slams/"+runid):
        try:
            os.makedirs("/homeappl/home/sunijona/SLAMS/slams/"+runid)
        except OSError:
            pass

    # Get solar wind parameters
    sw_pars = ja.sw_par_dict()[runid]
    rho_sw = sw_pars[0]
    v_sw = sw_pars[1]

    # Open file, get Cell IDs and sort them
    vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
    sorigid = vlsvobj.read_variable("CellID")
    sorigid = sorigid[sorigid.argsort()]
    
    # Find bow shock cells and area of one cell
    bs_cells = ja.bow_shock_finder(vlsvobj,rho_sw,v_sw)
    dA = ja.get_cell_area(vlsvobj)

    # Read initial event files
    events_old = eventfile_read(runid,start)
    events = eventfile_read(runid,start+1)

    # do nothing
    bs_events = []
    for old_event in events_old:
        bs_events.append(old_event)

    # Initialise list of jet objects
    jetobj_list = []
    dead_jetobj_list = []

    # Initialise unique ID counter
    counter = 1

    # Print current time
    print("t = "+str(float(start+1)/2)+"s")

    # Look for slams
    for event in events:

        for bs_event in bs_events:

            if np.intersect1d(bs_event,event).size > threshold*len(event):

                # Create unique ID
                curr_id = str(counter).zfill(5)

                # Create new jet object
                jetobj_list.append(jio.Jet(curr_id,runid,float(start)/2))

                # Append current events to jet object properties
                jetobj_list[-1].cellids.append(bs_event)
                jetobj_list[-1].cellids.append(event)
                jetobj_list[-1].times.append(float(start+1)/2)

                # Iterate counter
                counter += 1

                break

    # Track jets
    for n in xrange(start+2,stop+1):

        for jetobj in jetobj_list:
            if float(n)/2 - jetobj.times[-1] > 5:
                dead_jetobj_list.append(jetobj)
                jetobj_list.remove(jetobj)

        # Print  current time
        print("t = "+str(float(n)/2)+"s")

        # Find correct bulkname
        if runid == "AED":
            bulkname = "bulk.old."+str(n).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(n).zfill(7)+".vlsv"

        if bulkname not in os.listdir(bulkpath):
            print("Bulk file "+str(n)+" not found, continuing")
            events = []
            continue

        # Open bulkfile and get bow shock cells
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
        bs_cells = ja.bow_shock_finder(vlsvobj,rho_sw,v_sw)

        # List of old events
        bs_events = []
        for old_event in events:
            bs_events.append(old_event)

        # Initialise flags for finding splintering jets
        flags = []

        # Read event file for current time step
        events = eventfile_read(runid,n)
        events.sort(key=len)
        events = events[::-1]

        # Iniatilise list of cells currently being tracked
        curr_jet_temp_list = []

        # Update existing jets
        for event in events:

            for jetobj in jetobj_list:

                if jetobj.ID in flags:
                    
                    if np.intersect1d(jetobj.cellids[-2],event).size > threshold*len(event):

                        curr_id = str(counter).zfill(5)

                        # Create new jet
                        jetobj_new = jio.Jet(curr_id,runid,float(n)/2)
                        jetobj_new.cellids.append(event)
                        jetobj_list.append(jetobj_new)
                        curr_jet_temp_list.append(event)

                        #Clone jet
                        #jetobj_new = copy.deepcopy(jetobj)
                        #jetobj_new.ID = str(counter).zfill(5)
                        #print("Cloned jet to new one with ID "+jetobj_new.ID)
                        #jetobj_new.cellids = jetobj_new.cellids[:-1]
                        #jetobj_new.cellids.append(event)
                        #jetobj_list.append(jetobj_new)
                        #curr_jet_temp_list.append(event)

                        # Iterate counter
                        counter += 1

                        break

                else:

                    if np.intersect1d(jetobj.cellids[-1],event).size > threshold*len(event):

                        # Append event to jet object properties
                        jetobj.cellids.append(event)
                        jetobj.times.append(float(n)/2)
                        print("Updated jet with ID "+jetobj.ID)

                        # Flag jet object
                        flags.append(jetobj.ID)
                        curr_jet_temp_list.append(event)

                        break

        # Look for new jets at bow shock
        for event in events:

            if event not in curr_jet_temp_list:

                for bs_event in bs_events:

                    if np.intersect1d(bs_event,event).size > threshold*len(event):

                        # Create unique ID
                        curr_id = str(counter).zfill(5)

                        # Create new jet object
                        jetobj_list.append(jio.Jet(curr_id,runid,float(n-1)/2))

                        # Append current events to jet object properties
                        jetobj_list[-1].cellids.append(bs_event)
                        jetobj_list[-1].cellids.append(event)
                        jetobj_list[-1].times.append(float(n)/2)

                        # Iterate counter
                        counter += 1

                        break

    jetobj_list = jetobj_list + dead_jetobj_list

    for jetobj in jetobj_list:

        # Write jet object cellids and times to files
        jetfile = open("/homeappl/home/sunijona/SLAMS/slams/"+jetobj.runid+"/"+str(start)+"."+jetobj.ID+".slams","w")
        timefile = open("/homeappl/home/sunijona/SLAMS/slams/"+jetobj.runid+"/"+str(start)+"."+jetobj.ID+".times","w")

        jetfile.write(jetobj.return_cellid_string())
        timefile.write(jetobj.return_time_string())

        jetfile.close()
        timefile.close()

    return None