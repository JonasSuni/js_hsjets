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

m_p = 1.672621898e-27
r_e = 6.371e+6

class Jet:
    # Class for identifying and handling individual jets and their properties

    def __init__(self,ID,runid,birthday):

        self.ID = ID
        self.runid = runid
        self.birthday = birthday
        self.cellids = []
        self.times = [birthday]

        print("Created jet with ID "+self.ID)

    def return_cellid_string(self):

        return "\n".join([",".join(map(str,l)) for l in self.cellids])

    def return_time_string(self):

        return "\n".join(map(str,self.times))

def jet_maker(runid,start,stop,boxre=[6,16,-8,6],maskfile=False,avgfile=False):

    outputdir = "/homeappl/home/sunijona/events/"+runid+"/"

    # make outputdir if it doesn't already exist
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

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

        # open vlsv file for reading
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

        # create mask
        if maskfile:
            msk = np.loadtxt("Masks/"+runid+"/"+str(file_nr)+".mask").astype(int)
        else:
            msk = ja.make_cust_mask(file_nr,runid,180,boxre,avgfile)

        print(len(msk))
        print("Current file number is " + str(file_nr))

        # sort jets
        jets = ja.sort_jets(vlsvobj,msk,25,4500,[2,2])

        # erase contents of output file
        open(outputdir+str(file_nr)+".events","w").close()

        # open output file
        fileobj = open(outputdir+str(file_nr)+".events","a")

        # write jets to outputfile
        for jet in jets:

            fileobj.write(",".join(map(str,jet))+"\n")

        fileobj.close()

    return None

def timefile_read(runid,filenr,key):
    # Read array of times from file

    tf = open("/homeappl/home/sunijona/jets/"+runid+"/"+str(filenr)+"."+key+".times","r")
    contents = tf.read().split("\n")
    tf.close()

    return map(float,contents)

def jetfile_read(runid,filenr,key):
    # Read array of cellids from file

    outputlist = []

    jf = open("/homeappl/home/sunijona/jets/"+runid+"/"+str(filenr)+"."+key+".jet","r")
    contents = jf.read()
    lines = contents.split("\n")

    for line in lines:

        outputlist.append(map(int,line.split(",")))

    return outputlist

def eventfile_read(runid,filenr):
    # Read array of arrays of cellids from file

    outputlist = []

    ef = open("/homeappl/home/sunijona/events/"+runid+"/"+str(filenr)+".events","r")
    contents = ef.read().strip("\n")
    lines = contents.split("\n")

    for line in lines:

        outputlist.append(map(int,line.split(",")))

    return outputlist

def propfile_write(runid,filenr,key,props):
    # Write jet properties to file

    open("/homeappl/home/sunijona/jets/"+runid+"/"+str(filenr)+"."+key+".props","w").close()
    pf = open("/homeappl/home/sunijona/jets/"+runid+"/"+str(filenr)+"."+key+".props","a")
    pf.write("time [s],x_mean [R_e],y_mean [R_e],z_mean [R_e],A [R_e^2],Nr_cells,r_mean [R_e],theta_mean [deg],phi_mean [deg],size_rad [R_e],size_tan [R_e],x_max [R_e],y_max [R_e],z_max [R_e]"+"\n")
    pf.write("\n".join([",".join(map(str,line)) for line in props]))
    pf.close()
    print("Wrote to /homeappl/home/sunijona/jets/"+runid+"/"+str(filenr)+"."+key+".props")

def jio_figmake(runid,start,jetid,figname):
    # Create time series figures of specified jet

    props = calc_jet_properties(runid,start,jetid)

    if type(props) is not np.ndarray:
        return 1
    else:
        jetsize_fig(runid,start,jetid,figname=figname,props_arr=props)

def figmake_script(runid,start,ids):

    for ID in ids:
        jio_figmake(runid,start,ID,figname=ID)

def plotmake_script(runid,start,stop,vmax=1.5,boxre=[6,16,-8,6]):
    # Create plots of the dynamic pressure with contours of jets as well as their geometric centers

    if not os.path.exists("Contours/jetfigs/"+runid):
        os.makedirs("Contours/jetfigs/"+runid)

    # Find names of property files
    filenames = os.listdir("jets/"+runid)
    prop_fns = []
    for filename in filenames:
        if ".props" in filename:
            prop_fns.append(filename)
    prop_fns.sort()

    # Create dictionaries of jet positions with their times as keys
    tpos_dict_list = []
    for fname in prop_fns:
        props = pd.read_csv("/homeappl/home/sunijona/jets/"+runid+"/"+fname,index_col=False).as_matrix()
        t=props[:,0]
        X=props[:,1]
        Y=props[:,2]
        xmax=props[:,11]
        ymax=props[:,12]
        tpos_dict_list.append(dict(zip(t,np.array([X,Y,xmax,ymax]).T)))

    # Find names of event files
    filenames = os.listdir("events/"+runid)
    filenames.sort()

    # Create list of arrays of cellids to use as contour mask
    cells_list = []
    for filename in filenames:

        fileobj = open("events/"+runid+"/"+filename,"r")
        contents = fileobj.read()
        cells = map(int,contents.replace("\n",",").split(",")[:-1])
        cells_list.append(cells)

    fullmask_list = []
    for itr2 in xrange(start,stop+1):

        fullmask = np.loadtxt("Masks/"+runid+"/"+str(itr2)+".mask").astype(int)
        fullmask_list.append(fullmask)

    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    for itr in xrange(start,stop+1):

        # Find positions of all jets for current time step
        x_list = []
        y_list = []
        xmax_list = []
        ymax_list = []
        for tpos_dict in tpos_dict_list:
            if float(itr)/2 in tpos_dict:
                x_list.append(tpos_dict[float(itr)/2][0])
                y_list.append(tpos_dict[float(itr)/2][1])
                xmax_list.append(tpos_dict[float(itr)/2][2])
                ymax_list.append(tpos_dict[float(itr)/2][3])

        # Create plot
        pt.plot.plot_colormap(filename=bulkpath+"bulk."+str(itr).zfill(7)+".vlsv",outputdir="Contours/jetfigs/"+runid+"/",step=itr,run=runid,usesci=0,lin=1,boxre=boxre,vmin=0,vmax=vmax,colormap="parula",cbtitle="",external=pms_ext,expression=pc.expr_pdyn,pass_vars=["rho","v","CellID"],ext_pars=[x_list,y_list,cells_list[itr-start],fullmask_list[itr-start],xmax_list,ymax_list])


def pms_ext(ax,XmeshXY,YmeshXY,extmaps,ext_pars):

    rho,v,cellids = extmaps

    x_list,y_list,cells,fullmask,xmax_list,ymax_list = ext_pars

    # Create mask
    msk = np.in1d(cellids,cells).astype(int)
    msk = np.reshape(msk,rho.shape)

    fullmsk = np.in1d(cellids,fullmask).astype(int)
    fullmsk = np.reshape(fullmsk,rho.shape)

    # Draw contours
    fullcont = ax.contour(XmeshXY,YmeshXY,fullmsk,[0.5],linewidths=1.0,colors="magenta")
    cont = ax.contour(XmeshXY,YmeshXY,msk,[0.5],linewidths=1.0,colors="black")

    # Plot jet positions
    ax.plot(x_list,y_list,"o",color="red",markersize=4)
    ax.plot(xmax_list,ymax_list,"o",color="white",markersize=4)

def jetsize_fig(runid,start,jetid,figsize=(15,10),figname="sizefig",props_arr=None):
    # script for creating time series of jet linear sizes and area

    # Decide whether to read properties from file or input variable
    if props_arr == None:
        linsizes = pd.read_csv("/homeappl/home/sunijona/jets/"+runid+"/"+str(start)+"."+jetid+".props").as_matrix()
    else:
        linsizes = props_arr

    # Create variable value arrays
    time_arr = linsizes[:,0]
    area_arr = linsizes[:,4]
    rad_size_arr = linsizes[:,9]
    tan_size_arr = linsizes[:,10]
    x_arr = linsizes[:,1]
    y_arr = linsizes[:,2]
    z_arr = linsizes[:,3]

    # Minimum and maximum values
    minmax_list = [min(time_arr),max(time_arr),min(area_arr),max(area_arr),min(rad_size_arr),max(rad_size_arr),min(tan_size_arr),max(tan_size_arr),min(x_arr),max(x_arr),min(y_arr),max(y_arr),min(z_arr),max(z_arr)]

    for n in xrange(0,len(minmax_list),2):
        if np.abs((minmax_list[n]-minmax_list[n+1])/float(minmax_list[n])) < 1.0e-5:
            minmax_list[n+1] += 1
            minmax_list[n] -= 1

    tmin,tmax,Amin,Amax,rsmin,rsmax,psmin,psmax,xmin,xmax,ymin,ymax,zmin,zmax = minmax_list

    # Create figure
    plt.ion()
    fig = plt.figure(figsize=figsize)

    # Add subplots
    area_ax = fig.add_subplot(321)
    rad_size_ax = fig.add_subplot(323)
    tan_size_ax = fig.add_subplot(325)
    x_ax = fig.add_subplot(322)
    y_ax = fig.add_subplot(324)
    z_ax = fig.add_subplot(326)

    # Draw grids
    area_ax.grid()
    rad_size_ax.grid()
    tan_size_ax.grid()
    x_ax.grid()
    y_ax.grid()
    z_ax.grid()

    # Set x-limits
    area_ax.set_xlim(tmin,tmax)
    rad_size_ax.set_xlim(tmin,tmax)
    tan_size_ax.set_xlim(tmin,tmax)
    x_ax.set_xlim(tmin,tmax)
    y_ax.set_xlim(tmin,tmax)
    z_ax.set_xlim(tmin,tmax)

    # Set y-limits
    area_ax.set_ylim(Amin,Amax)
    rad_size_ax.set_ylim(rsmin,rsmax)
    tan_size_ax.set_ylim(psmin,psmax)
    x_ax.set_ylim(xmin,xmax)
    y_ax.set_ylim(ymin,ymax)
    z_ax.set_ylim(zmin,zmax)

    #area_ax.set_yticks([0.5,1,1.5,2,2.5])
    #rad_size_ax.set_yticks([0.8,1.2,1.6,2,2.4,2.8])
    #tan_size_ax.set_yticks([0.6,1,1.4])

    #area_ax.set_xticks([295,300,305,310,315,320])
    #rad_size_ax.set_xticks([295,300,305,310,315,320])
    #tan_size_ax.set_xticks([295,300,305,310,315,320])

    # Set x-ticklabels
    area_ax.set_xticklabels([])
    rad_size_ax.set_xticklabels([])
    x_ax.set_xticklabels([])
    y_ax.set_xticklabels([])

    # Set y-labels
    area_ax.set_ylabel("Area [R$_{e}^{2}$]",fontsize=20)
    rad_size_ax.set_ylabel("Radial size [R$_{e}$]",fontsize=20)
    tan_size_ax.set_ylabel("Tangential size [R$_{e}$]",fontsize=20)
    tan_size_ax.set_xlabel("Time [s]",fontsize=20)
    x_ax.set_ylabel("X [R$_{e}$]",fontsize=20)
    y_ax.set_ylabel("Y [R$_{e}$]",fontsize=20)
    z_ax.set_ylabel("Z [R$_{e}$]",fontsize=20)

    # Set tick label sizes
    area_ax.tick_params(labelsize=16)
    rad_size_ax.tick_params(labelsize=16)
    tan_size_ax.tick_params(labelsize=16)
    x_ax.tick_params(labelsize=16)
    y_ax.tick_params(labelsize=16)
    z_ax.tick_params(labelsize=16)

    # Plot variables
    area_ax.plot(time_arr,area_arr,color="black",linewidth=2)
    rad_size_ax.plot(time_arr,rad_size_arr,color="black",linewidth=2)
    tan_size_ax.plot(time_arr,tan_size_arr,color="black",linewidth=2)
    x_ax.plot(time_arr,x_arr,color="black",linewidth=2)
    y_ax.plot(time_arr,y_arr,color="black",linewidth=2)
    z_ax.plot(time_arr,z_arr,color="black",linewidth=2)

    plt.tight_layout()

    fig.show()

    # Create outputdir if it doesn't already exist
    if not os.path.exists("jet_sizes/"+runid):
        os.makedirs("jet_sizes/"+runid)

    # Save figure
    plt.savefig("jet_sizes/"+runid+"/"+figname+".png")
    print("jet_sizes/"+runid+"/"+figname+".png")

    return None

def calc_jet_properties(runid,start,jetid):

    # Read jet cellids and times
    jet_list = jetfile_read(runid,start,jetid)
    time_list = timefile_read(runid,start,jetid)

    # Discard jet if it has large gaps in the times
    if len(time_list) < 2:
        print("Jet not sufficiently long-lived, exiting.")
        return 1
    dt = (np.pad(np.array(time_list),(0,1),"constant")-np.pad(np.array(time_list),(1,0),"constant"))[1:-1]
    if max(dt) > 5:
        print("Jet not sufficiently continuous, exiting.")
        return 1

    # Find correct bulk path
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = "/proj/vlasov/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = "/proj/vlasov/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = "/proj/vlasov/2D/"+runid+"/bulk/"

    # Convert times to file numbers
    nr_list = [int(t*2) for t in time_list]

    # Initialise property array
    prop_arr = np.array([])

    for n in xrange(len(nr_list)):

        curr_list = jet_list[n]
        curr_list.sort()

        # Find correct file name
        if runid == "AED":
            bulkname = "bulk.old."+str(nr_list[n]).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(nr_list[n]).zfill(7)+".vlsv"

        # Open VLSV file
        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

        origid = vlsvobj.read_variable("CellID")

        # read variables
        if vlsvobj.check_variable("X"):
            X = vlsvobj.read_variable("X",cellids=curr_list)
            Y = vlsvobj.read_variable("Y",cellids=curr_list)
            Z = vlsvobj.read_variable("Z",cellids=curr_list)
        else:
            sorigid = origid[origid.argsort()]
            X,Y,Z = ja.ci2vars_nofile(ja.xyz_reconstruct(vlsvobj),sorigid,curr_list)

        # Calculate area of one cell
        if n == 0 and vlsvobj.check_variable("DX"):
            dA = vlsvobj.read_variable("DX")[0]*vlsvobj.read_variable("DY")[0]
        elif n == 0 and not vlsvobj.check_variable("DX"):
            dA = ja.get_cell_area(vlsvobj)

        # If file has more than one population, choose proton population
        var_list = ["rho","v","B","Temperature","va","vms","CellID","TParallel","TPerpendicular"]
        var_list_alt = ["proton/rho","proton/V","B","proton/Temperature","proton/va","proton/vms","CellID","proton/TParallel","proton/TPerpendicular"]
        if not vlsvobj.check_variable("rho"):
            var_list = var_list_alt

        v = vlsvobj.read_variable(var_list[1],cellids=curr_list)
        vmag = np.linalg.norm(v,axis=-1)

        # calculate geometric center of jet
        r = np.linalg.norm(np.array([X,Y,Z]),axis=0)
        theta = np.rad2deg(np.arccos(Z/r))
        phi = np.rad2deg(np.arctan(Y/X))

        r_mean = np.mean(r)/r_e
        theta_mean = np.mean(theta)
        phi_mean = np.mean(phi)

        x_mean = r_mean*np.sin(np.deg2rad(theta_mean))*np.cos(np.deg2rad(phi_mean))
        y_mean = r_mean*np.sin(np.deg2rad(theta_mean))*np.sin(np.deg2rad(phi_mean))
        z_mean = r_mean*np.cos(np.deg2rad(theta_mean))

        x_max = X[vmag==max(vmag)][0]/r_e
        y_max = Y[vmag==max(vmag)][0]/r_e
        z_max = Z[vmag==max(vmag)][0]/r_e

        #r_max = np.linalg.norm(np.array([x_mean,y_mean,z_mean]))
        #theta_max = np.rad2deg(np.arccos(z_mean/r_mean))
        #phi_max = np.rad2deg(np.arctan(y_mean/x_mean))

        # calculate jet size
        A = dA*len(curr_list)/(r_e**2)
        Nr_cells = len(curr_list)

        # calculate linear sizes of jet
        size_rad = (max(r)-min(r))/r_e
        size_tan = A/size_rad

        # current time
        time = time_list[n]

        # "time [s],x_mean [R_e],y_mean [R_e],z_mean [R_e],A [R_e^2],Nr_cells,r_mean [R_e],theta_mean [deg],phi_mean [deg],size_rad [R_e],size_tan [R_e],x_max [R_e],y_max [R_e],z_max [R_e]"
        temp_arr = [time,x_mean,y_mean,z_mean,A,Nr_cells,r_mean,theta_mean,phi_mean,size_rad,size_tan,x_max,y_max,z_max]

        # append properties to property array
        prop_arr = np.append(prop_arr,np.array(temp_arr))

    # reshape property array
    prop_arr = np.reshape(prop_arr,(len(nr_list),len(temp_arr)))

    # write property array to file
    propfile_write(runid,start,jetid,prop_arr)

    return prop_arr

def track_jets(runid,start,stop,threshold=0.3):

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

    # Create outputdir if it doesn't already exist
    if not os.path.exists("/homeappl/home/sunijona/jets/"+runid):
        os.makedirs("/homeappl/home/sunijona/jets/"+runid)

    sw_pars = ja.sw_par_dict()[runid]
    rho_sw = sw_pars[0]
    v_sw = sw_pars[1]

    # Open file to get
    vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
    sorigid = vlsvobj.read_variable("CellID")
    sorigid = sorigid[sorigid.argsort()]
    fX,fY,fZ = ja.xyz_reconstruct(vlsvobj)
    bs_cells = ja.bow_shock_finder(vlsvobj,rho_sw,v_sw)
    dA = ja.get_cell_area(vlsvobj)

    # Read initial event files
    events_old = eventfile_read(runid,start)
    events = eventfile_read(runid,start+1)

    # remove events that are not initially at the bow shock
    bs_events = []
    for old_event in events_old:
        if np.intersect1d(bs_cells,ja.get_neighbors(vlsvobj,old_event,[2,2])).size > 0:
            bs_events.append(old_event)

    # Initialise list of jet objects
    jetobj_list = []

    # Initialise unique ID counter
    counter = 1

    print("t = "+str(float(start+1)/2)+"s")

    # Look for jets at bow shock
    for event in events:

        for bs_event in bs_events:

            if np.intersect1d(bs_event,event).size > threshold*len(event):

                # Create unique ID
                curr_id = str(counter).zfill(5)

                # Create new jet object
                jetobj_list.append(Jet(curr_id,runid,float(start)/2))

                # Append current events to jet object properties
                jetobj_list[-1].cellids.append(bs_event)
                jetobj_list[-1].cellids.append(event)
                jetobj_list[-1].times.append(float(start+1)/2)

                # Iterate counter
                counter += 1

                break

    # Track jets
    for n in xrange(start+2,stop+1):

        print("t = "+str(float(n)/2)+"s")

        if runid == "AED":
            bulkname = "bulk.old."+str(n).zfill(7)+".vlsv"
        else:
            bulkname = "bulk."+str(n).zfill(7)+".vlsv"

        vlsvobj = pt.vlsvfile.VlsvReader(bulkpath+bulkname)
        bs_cells = ja.bow_shock_finder(vlsvobj,rho_sw,v_sw)

        # Filtered list of events that are at the bow shock at the current time
        bs_events = []
        for old_event in events:
            if np.intersect1d(bs_cells,ja.get_neighbors(vlsvobj,old_event,[2,2])).size > 0:
                bs_events.append(old_event)

        # Initialise flags for finding splintering jets
        flags = []

        # Read event file for current time step
        events = eventfile_read(runid,n)
        events.sort(key=len)
        events = events[::-1]

        curr_jet_temp_list = []

        # Update existing jets
        for event in events:

            for jetobj in jetobj_list:

                if jetobj.ID in flags:
                    
                    if np.intersect1d(jetobj.cellids[-2],event).size > threshold*len(event):

                        curr_id = str(counter).zfill(5)

                        # Create new jet
                        jetobj_new = Jet(curr_id,runid,float(n)/2)
                        #jetobj_new = copy.deepcopy(jetobj)
                        #jetobj_new.ID = str(counter).zfill(5)
                        #print("Cloned jet to new one with ID "+jetobj_new.ID)
                        #jetobj_new.cellids = jetobj_new.cellids[:-1]
                        #jetobj_new.cellids.append(jetobj.cellids[-2])
                        jetobj_new.cellids.append(event)
                        #jetobj_new.times = jetobj_new.times[:-1]
                        #jetobj_new.times.append(float(n)/2)
                        jetobj_list.append(jetobj_new)
                        curr_jet_temp_list.append(event)

                        # Iterate counter
                        counter += 1

                        # Alternative algorithm
                        #r_objevent = np.mean(np.linalg.norm(ja.ci2vars_nofile([fX,fY,fZ],sorigid,jetobj.cellids[-1]),axis=0))
                        #r_event = np.mean(np.linalg.norm(ja.ci2vars_nofile([fX,fY,fZ],sorigid,event),axis=0))
                        #linsizemin = np.min([np.sqrt(dA*len(jetobj.cellids[-1])),np.sqrt(dA*len(event))])

                        #if np.abs(r_objevent-r_event) < linsizemin:
                        #    jetobj.cellids[-1] += event
                        #    print("Updated jet with ID "+jetobj.ID)
                        #    curr_jet_temp_list.append(event)
                        #else:
                            # Clone previously existing jet object as a new unique jet
                        #    jetobj_new = copy.deepcopy(jetobj)
                        #    jetobj_new.ID = str(counter).zfill(5)
                        #    print("Cloned jet to new one with ID "+jetobj_new.ID)
                        #    jetobj_new.cellids = jetobj_new.cellids[:-1]
                        #    jetobj_new.cellids.append(event)
                        #    jetobj_new.times = jetobj_new.times[:-1]
                        #    jetobj_new.times.append(float(n)/2)
                        #    jetobj_list.append(jetobj_new)
                        #    curr_jet_temp_list.append(event)

                            # Iterate counter
                        #    counter += 1

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
                        jetobj_list.append(Jet(curr_id,runid,float(n-1)/2))

                        # Append current events to jet object properties
                        jetobj_list[-1].cellids.append(bs_event)
                        jetobj_list[-1].cellids.append(event)
                        jetobj_list[-1].times.append(float(n)/2)

                        # Iterate counter
                        counter += 1

                        break

    for jetobj in jetobj_list:

        # Write jet object cellids and times to files
        jetfile = open("/homeappl/home/sunijona/jets/"+jetobj.runid+"/"+str(start)+"."+jetobj.ID+".jet","w")
        timefile = open("/homeappl/home/sunijona/jets/"+jetobj.runid+"/"+str(start)+"."+jetobj.ID+".times","w")

        jetfile.write(jetobj.return_cellid_string())
        timefile.write(jetobj.return_time_string())

        jetfile.close()
        timefile.close()

    return None