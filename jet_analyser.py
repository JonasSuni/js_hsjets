import pytools as pt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

m_p = 1.672621898e-27
r_e = 6.371e+6

def v_decomp(B,v):

    Bmag = np.linalg.norm(B,axis=-1)

    Bhat = B
    Bhat[:,0] /= Bmag
    Bhat[:,1] /= Bmag
    Bhat[:,2] /= Bmag

    Bv_dot = np.sum(v*Bhat,axis=-1)

    v_par = Bhat
    v_par[:,0] *= Bv_dot
    v_par[:,1] *= Bv_dot
    v_par[:,2] *= Bv_dot

    v_perp = v-v_par

    v_par_mag = np.linalg.norm(v_par,axis=-1)
    v_perp_mag = np.linalg.norm(v_perp,axis=-1)
    
    v_decomps = np.array([v_par_mag,v_perp_mag]).T

    return v_decomps

def T_decomp(T,v_decomps,vmag):

    T_par = T*(v_decomps[:,0]/vmag)**2
    T_perp = T*(v_decomps[:,1]/vmag)**2

    T_decomps = np.array([T_par,T_perp]).T

    return T_decomps

def phi_hist(input_file):
    # creates normed histogram of the angle phi, weighted according to the number of cells

    # read variables from properties file
    phi = pd.read_csv(input_file).as_matrix()[:,20]
    nr_cells = pd.read_csv(input_file).as_matrix()[:,19]
    
    #create figure
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Angle [deg]")
    ax.set_ylabel("Probability density")

    # draw histogram
    phi_h = ax.hist(phi,bins=list(xrange(-40,41,5)),weights=nr_cells,normed=True)

    fig.show()

def y_hist(input_file):
    # creates normed histogram of the angle phi, weighted according to the number of cells

    # read variables from properties file
    y = pd.read_csv(input_file).as_matrix()[:,16]
    nr_cells = pd.read_csv(input_file).as_matrix()[:,19]
    
    # create figure
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("$Y_{vmax}$ $[R_e]$")
    ax.set_ylabel("Probability density")

    # draw histogram
    y_h = ax.hist(y,bins=list(xrange(-6,7)),weights=nr_cells,normed=True)

    fig.show()

def calc_props(vlsvobj,jets,runid,file_number,criterion,halftimewidth,freeform_file_id=""):
    # calculates certain properties for the jets

    # area of one cell
    dA = vlsvobj.read_variable("DX")[0]*vlsvobj.read_variable("DY")[0]

    # erase contents of file if it's not already empty
    open("Props/"+runid+"/props_"+runid+"_"+str(file_number)+"_"+str(halftimewidth)+freeform_file_id+".csv","w").close()

    # open csv file for writing
    outputfile = open("Props/"+runid+"/props_"+runid+"_"+str(file_number)+"_"+str(halftimewidth)+freeform_file_id+".csv","a")

    # write header to csv file
    outputfile.write("n_avg [cm^-3],n_med [cm^-3],n_max [cm^-3],v_avg [km/s],v_med [km/s],v_max [km/s],B_avg [nT],B_med [nT],B_max [nT],T_avg [MK],T_med [MK],T_max [MK],Tpar_avg [MK],Tpar_med [MK],Tpar_max [MK],Tperp_avg [MK],Tperp_med [MK],Tperp_max [MK],X_vmax [R_e],Y_vmax [R_e],Z_vmax [R_e],A [km^2],Nr_cells,phi [deg],r_d [R_e],mag_p_bool,size_x [R_e],size_y [R_e]")

    # initialise list of properties
    props_list = []

    # read variables from vlsv object
    rho,v,B,T,X,Y,Z,va,vms,cellids = read_mult_vars(vlsvobj,["rho","v","B","Temperature","X","Y","Z","va","vms","CellID"])

    # calculate magnitudes
    vmag = np.linalg.norm(v,axis=-1)
    Bmag = np.linalg.norm(B,axis=-1)

    T_decomps = T_decomp(T,v_decomp(B,v),vmag)
    Tpar = T_decomps[:,0]
    Tperp = T_decomps[:,1]

    for event in jets:

        outputfile.write("\n")

        # get the values of the variables corresponding to cellids in the current event
        jrho,jvmag,jBmag,jT,jX,jY,jZ,jva,jvms,jTpar,jTperp = ci2vars_nofile([rho,vmag,Bmag,T,X,Y,Z,va,vms,Tpar,Tperp],cellids,event)

        # calculate mean, maximum and median of density
        n_avg = np.mean(jrho)/1.0e+6
        n_max = max(jrho)/1.0e+6
        n_med = np.median(jrho)/1.0e+6

        # calculate mean, maximum and median of velocity
        v_avg = np.mean(jvmag)/1.0e+3
        v_max = max(jvmag)/1.0e+3
        v_med = np.median(jvmag)/1.0e+3

        # calculate mean, maximum and median of the magnetic field
        B_avg = np.mean(jBmag)/1.0e-9
        B_max = max(jBmag)/1.0e-9
        B_med = np.median(jBmag)/1.0e-9

        # calculate mean, maximum and median of temperatures
        T_avg = np.nanmean(jT)/1.0e+6
        T_max = max(jT)/1.0e+6
        T_med = np.median(jT)/1.0e+6

        Tpar_avg = np.nanmean(jTpar)/1.0e+6
        Tpar_max = max(jTpar)/1.0e+6
        Tpar_med = np.median(jTpar)/1.0e+6

        Tperp_avg = np.nanmean(jTperp)/1.0e+6
        Tperp_max = max(jTperp)/1.0e+6
        Tperp_med = np.median(jTperp)/1.0e+6

        # calculate the position of the cell that corresponds to maximum velocity
        X_vmax = jX[np.where(jvmag==max(jvmag))[0]][0]/r_e
        Y_vmax = jY[np.where(jvmag==max(jvmag))[0]][0]/r_e
        Z_vmax = jZ[np.where(jvmag==max(jvmag))[0]][0]/r_e

        # calculate area of current event
        A = dA*event.size/1.0e+6
        Nr_cells = event.size

        # calculate angular position of jet
        phi = np.arctan(Y_vmax/X_vmax)*360/(2*np.pi)

        # radial distance of jet
        r_d = ((X_vmax**2+Y_vmax**2)**0.5)

        # does jet reach magnetopause?
        mag_p_bool = 0.0
        if r_d < 8:
            mag_p_bool = 1.0

        x_size = (max(jX)-min(jX))/r_e
        y_size = (max(jY)-min(jY))/r_e

        # properties for current event
        temp_arr = [n_avg,n_med,n_max,v_avg,v_med,v_max,B_avg,B_med,B_max,T_avg,T_med,T_max,Tpar_avg,Tpar_med,Tpar_max,Tperp_avg,Tperp_med,Tperp_max,X_vmax,Y_vmax,Z_vmax,A,Nr_cells,phi,r_d,mag_p_bool,x_size,y_size]

        # write properties for current event to list of properties
        props_list.append(temp_arr)

        # write properties for current event to csv file
        outputfile.write(",".join(map(str,temp_arr)))

    outputfile.close()
    print("Props/"+runid+"/props_"+runid+"_"+str(file_number)+"_"+str(halftimewidth)+freeform_file_id+".csv")

    return np.asarray([np.asarray(prop) for prop in props_list])


def sort_jets(vlsvobj,cells,min_size=0,max_size=3000,neighborhood_reach=[1,1]):
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
        it_range = xrange(100)

        # initialise current event
        curr_event = np.array([cell])

        for n in it_range:

            curr_event_temp = curr_event

            # find neighbors within the confines of the mask
            curr_event = np.unique(np.append(curr_event,np.intersect1d(cells,get_neighbors(vlsvobj,curr_event,neighborhood_reach))))

            # exit loop if all valid neighbors found
            if curr_event_temp.size == curr_event.size:
                break

        # cast cellids of current event to int and append to list of events
        curr_event = curr_event.astype(int)
        events.append(curr_event)

    # remove events smaller than the minimum size and larger than maximum size
    events_culled = [jet for jet in events if jet.size>=min_size and jet.size<=max_size]

    return events_culled


def jet_script(filenumber,runid,halftimewidth=180,criterion="AH",boxre=[8,16,-6,6],min_size=0,max_size=3000,neighborhood_reach=[1,1],freeform_file_id=""):

    file_nr = str(filenumber).zfill(7)
    file_path = "/proj/vlasov/2D/"+runid+"/bulk/bulk."+file_nr+".vlsv"

    vlsvobj=pt.vlsvfile.VlsvReader(file_path)

    msk = make_mask(filenumber,runid,halftimewidth,criterion,boxre)
    jets = sort_jets(vlsvobj,msk,min_size,max_size,neighborhood_reach)
    props = calc_props(vlsvobj,jets,runid,filenumber,criterion,halftimewidth,freeform_file_id)

    return props

def jet_script_cust(filenumber,runid,halftimewidth=180,boxre=[8,16,-6,6],min_size=0,max_size=3000,neighborhood_reach=[1,1],freeform_file_id=""):

    file_nr = str(filenumber).zfill(7)
    file_path = "/proj/vlasov/2D/"+runid+"/bulk/bulk."+file_nr+".vlsv"

    vlsvobj=pt.vlsvfile.VlsvReader(file_path)

    msk = make_cust_mask(filenumber,runid,halftimewidth,boxre)
    jets = sort_jets(vlsvobj,msk,min_size,max_size,neighborhood_reach)
    props = calc_props(vlsvobj,jets,runid,filenumber,"CUST",halftimewidth,freeform_file_id)

    return props


def give_file():

    f = pt.vlsvfile.VlsvReader("/proj/vlasov/2D/ABA/bulk/bulk.0000611.vlsv")

    return f


def get_neighbors(vlsvobj,c_i,neighborhood_reach=[1,1]):
    # finds the neighbors of the specified cells within the maximum offsets in neighborhood_reach

    # initialise array of neighbors
    neighbors = np.array([],dtype=int)

    # range of offsets to take into account
    x_r = xrange(-1*neighborhood_reach[0],neighborhood_reach[0]+1)
    y_r = xrange(-1*neighborhood_reach[1],neighborhood_reach[1]+1)

    for n in c_i:

        # append cellids of neighbors, cast as int, to array of neighbors
        for a in x_r:
            for b in y_r:
                neighbors = np.append(neighbors,int(vlsvobj.get_cell_neighbor(cellid=n,offset=[a,b,0],periodic=[0,0,0])))

    # discard invalid cellids
    neighbors = neighbors[neighbors != 0]

    # discard duplicate cellids
    neighbors = np.unique(neighbors)

    return neighbors


def ci2vars(vlsvobj,input_vars,cells):
    # find the values for the input variables corresponding to the masked cellids.
    # reads the specified variables from the vlsvobject

    # initialise list of output vars
    output_vars = []

    cellids = vlsvobj.read_variable("CellID")

    # find the indices of the masked cells
    n_i = np.where(np.in1d(cellids,cells))[0]

    for input_var in input_vars:

        variable = vlsvobj.read_variable(input_var)

        # find values of the variable for the masked cells
        n_var = variable[n_i]

        # append variable values to list of output vars
        output_vars.append(n_var)

    return output_vars


def ci2vars_nofile(input_vars,cellids,cells):
    # find the values for the input variables corresponding to the masked cellids.
    # uses variables that have been read from vlsvobject beforehand

    # initialise list of output vars
    output_vars = []

    # find the indices of the masked cells
    n_i = np.where(np.in1d(cellids,cells))[0]

    for input_var in input_vars:

        # find values of the variable for the masked cells
        n_var = input_var[n_i]

        # append variable values to list of output vars
        output_vars.append(n_var)

    return output_vars


def read_mult_vars(vlsvobj,input_vars):
    # reads multiple variables from vlsvobject

    # initialise list of output vars
    output_vars = []

    for input_var in input_vars:

        # read variable from vlsvobject and append it to list of output_vars
        variable = vlsvobj.read_variable(input_var)
        output_vars.append(variable)

    return output_vars


def restrict_area(vlsvobj,xlim,ylim):
    # find cellids of cells that correspond to X,Y-positions within the specified limits

    # read variables from vlsvobject
    X = vlsvobj.read_variable("X")
    Y = vlsvobj.read_variable("Y")
    cellids = vlsvobj.read_variable("CellID")

    # mask the cellids within the specified limits
    msk = np.ma.masked_greater_equal(X,xlim[0]*r_e)
    msk.mask[X > xlim[1]*r_e] = False
    msk.mask[Y < ylim[0]*r_e] = False
    msk.mask[Y > ylim[1]*r_e] = False

    # discard unmasked cellids
    masked_ci = cellids[msk.mask]

    return masked_ci


def make_cust_mask(filenumber,runid,halftimewidth,boxre=[8,16,-6,6]):

    # finds cellids of cells that fulfill the specified criterion and the specified
    # X,Y-limits

    # find correct file based on file number and run id
    file_nr = str(filenumber).zfill(7)
    file_path = "/proj/vlasov/2D/"+runid+"/bulk/bulk."+file_nr+".vlsv"

    # open vlsv file for reading
    vlsvreader = pt.vlsvfile.VlsvReader(file_path)

    origid = vlsvreader.read_variable("CellID")
    sorigid = origid[np.argsort(origid)]

    # solar wind parameters
    rho_sw = 1.0e+6
    vx_sw = 750.0e+3

    rho = vlsvreader.read_variable("rho")[np.argsort(origid)]
    v = vlsvreader.read_variable("v")[np.argsort(origid)]

    # ratio of x-directional dynamic pressure and solar wind dynamic pressure
    npdynx = rho*(v[:,0]**2)/(rho_sw*(vx_sw**2))
    nrho = rho/rho_sw

    # dynamic pressure
    pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

    # initialise time average of dynamic pressure
    tpdynavg = np.zeros(np.shape(pdyn))

    # range of timesteps to calculate average of
    timerange = xrange(filenumber-halftimewidth,filenumber+halftimewidth+1)

    for n in timerange:

        # exclude the main timestep
        if n == filenumber:
            continue

        # find correct file path for current time step
        tfile_nr = str(n).zfill(7)
        tfile_p = "/proj/vlasov/2D/"+runid+"/bulk/bulk."+tfile_nr+".vlsv"

        # open file for current time step
        f = pt.vlsvfile.VlsvReader(tfile_p)
        
        trho = f.read_variable("rho")
        tv = f.read_variable("v")

        # read cellids for current time step
        cellids = f.read_variable("CellID")
        
        # dynamic pressure for current time step
        tpdyn = m_p*trho*(np.linalg.norm(tv,axis=-1)**2)

        # sort dynamic pressures
        otpdyn = tpdyn[cellids.argsort()]

        # prevent divide by zero errors
        otpdyn[otpdyn == 0.0] = 1.0e-27
        
        tpdynavg += otpdyn

    # calculate time average of dynamic pressure
    tpdynavg /= len(timerange)-1

    # ratio of dynamic pressure to its time average
    tapdyn = np.divide(pdyn,tpdynavg)

    # make plaschke mask
    jet_p = np.ma.masked_greater(npdynx,0.25)
    jet_p.mask[nrho < 3] = False

    # make archer&horbury mask
    jet_ah = np.ma.masked_greater(tapdyn,2)

    jet_cust = jet_ah
    jet_cust.mask = np.logical_or(jet_cust.mask,jet_p.mask)

    # discard unmasked cellids
    masked_ci = sorigid[jet_cust.mask]

    # if boundaries have been set, discard cellids outside boundaries
    if not not boxre:
        return np.intersect1d(masked_ci,restrict_area(vlsvreader,boxre[0:2],boxre[2:4]))
    else:
        return masked_ci

def make_mask(timestep,runid,halftimewidth,criterion,boxre=[8,16,-6,6]):
    # finds cellids of cells that fulfill the specified criterion and the specified
    # X,Y-limits

    # find correct file based on file number and run id
    file_nr = str(timestep).zfill(7)
    file_path = "/proj/vlasov/2D/"+runid+"/bulk/bulk."+file_nr+".vlsv"

    # open vlsv file for reading
    vlsvreader = pt.vlsvfile.VlsvReader(file_path)

    origid = vlsvreader.read_variable("CellID")
    sorigid = origid[np.argsort(origid)]

    # solar wind parameters
    rho_sw = 1.0e+6
    vx_sw = 750.0e+3

    rho = vlsvreader.read_variable("rho")[np.argsort(origid)]
    v = vlsvreader.read_variable("v")[np.argsort(origid)]

    if criterion == "P":
        # mask according to Plaschke criterion

        # ratio of x-directional dynamic pressure and solar wind dynamic pressure
        npdynx = rho*(v[:,0]**2)/(rho_sw*(vx_sw**2))
        nrho = rho/rho_sw

        # mask according to the criterion
        jet = np.ma.masked_greater(npdynx,0.25)
        jet.mask[nrho < 2] = False

        # discard unmasked cellids
        masked_ci = sorigid[jet.mask]

        # if boundaries have been set, discard cellids outside boundaries
        if not not boxre:
            return np.intersect1d(masked_ci,restrict_area(vlsvreader,boxre[0:2],boxre[2:4]))
        else:
            return masked_ci

    elif criterion == "AH":
        # mask according to Archer&Horbury criterion

        # calculate dynamic pressure
        pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

        timerange = xrange(timestep-halftimewidth,timestep+halftimewidth+1)

        # initialise time average of dynamic pressure
        tpdynavg = np.zeros(np.shape(pdyn))

        for n in timerange:

            # exclude the main timestep
            if n == timestep:
                continue

            # find correct file path for current time step
            tfile_nr = str(n).zfill(7)
            tfile_p = "/proj/vlasov/2D/"+runid+"/bulk/bulk."+tfile_nr+".vlsv"

            # open file for current time step
            f = pt.vlsvfile.VlsvReader(tfile_p)
        
            trho = f.read_variable("rho")
            tv = f.read_variable("v")

            # read cellids for current time step
            cellids = f.read_variable("CellID")
        
            # dynamic pressure for current time step
            tpdyn = m_p*trho*(np.linalg.norm(tv,axis=-1)**2)

            # sort dynamic pressures
            otpdyn = tpdyn[cellids.argsort()]

            # prevent divide by zero errors
            otpdyn[otpdyn == 0.0] = 1.0e-27
        
            tpdynavg += otpdyn

        # calculate time average of dynamic pressure
        tpdynavg /= len(timerange)-1

        # ratio of dynamic pressure to its time average
        tapdyn = np.divide(pdyn,tpdynavg)

        # mask according to criterion
        jet = np.ma.masked_greater(tapdyn,2)

        # discard unmasked cellids
        masked_ci = sorigid[jet.mask]

        # if boundaries have been set, discard cellids outside boundaries
        if not not boxre:
            return np.intersect1d(masked_ci,restrict_area(vlsvreader,boxre[0:2],boxre[2:5]))
        else:
            return masked_ci

    elif criterion == "K":
        # mask according to Karlsson criterion

        timerange = xrange(timestep-halftimewidth,timestep+halftimewidth+1)

        # initialise time average of density
        trhoavg = np.zeros(np.shape(rho))

        for n in timerange:

            # exclude main timestep
            if n == timestep:
                continue

            # find correct file path for current time step
            tfile_nr = str(n).zfill(7)
            tfile_p = "/proj/vlasov/2D/"+runid+"/bulk/bulk."+tfile_nr+".vlsv"

            # open file for current time step
            f = pt.vlsvfile.VlsvReader(tfile_p)
        
            trho = f.read_variable("rho")

            # read cellids for current time step
            cellids = f.read_variable("CellID")

            # sort dynamic pressures
            otrho = trho[cellids.argsort()]

            # prevent divide by zero errors
            otrho[otrho == 0.0] = 1.0e-27
        
            trhoavg += otrho

        # calculate time average of density
        trhoavg /= len(timerange)-1

        # ratio of density to its time average
        tarho = np.divide(rho,trhoavg)

        # mask according to criterion
        jet = np.ma.masked_greater(tarho,1.5)

        # exclude unmasked cellids
        masked_ci = sorigid[jet.mask]

        # if boundaries have been set, discard cellids outside boundaries
        if not not boxre:
            return np.intersect1d(masked_ci,restrict_area(vlsvreader,boxre[0:2],boxre[2:5]))
        else:
            return masked_ci

    else:
        print("criterion must be either P, AH or K")
        return None