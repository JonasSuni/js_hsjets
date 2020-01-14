def calc_jet_properties(runid,start,jetid,tp_files=False,transient="jet"):

    # Check transient type
    if transient == "jet":
        inputdir = wrkdir_DNR+"working/jets"
        extension = "jet"
    elif transient == "slamsjet":
        inputdir = wrkdir_DNR+"working/SLAMSJETS/slamsjets"
        extension = "slamsjet"

    # Check if transient with specified ID exists
    if "{}.{}.{}".format(str(start),jetid,extension) not in os.listdir("{}/{}".format(inputdir,runid)):
        print("Transient with ID "+jetid+" does not exist, exiting.")
        return 1

    # Read jet cellids and times
    jet_list = jetfile_read(runid,start,jetid,transient)
    time_list = timefile_read(runid,start,jetid,transient)

    # Discard jet if it's very short-lived
    if time_list[-1] - time_list[0] + 0.5 < 5:
        print("Transient not sufficiently long-lived, exiting.")
        return 1

    # Find correct bulk path
    if runid in ["AEC","AEF","BEA","BEB"]:
        bulkpath = vlasdir+"/2D/"+runid+"/"
    elif runid == "AEA":
        bulkpath = vlasdir+"/2D/"+runid+"/round_3_boundary_sw/"
    else:
        bulkpath = vlasdir+"/2D/"+runid+"/bulk/"

    # Convert times to file numbers
    nr_list = [int(t*2) for t in time_list]

    # Initialise property array
    prop_arr = np.array([])

    for n in range(len(nr_list)):

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
        sorigid = origid[origid.argsort()]

        # read variables
        if vlsvobj.check_variable("X"):
            X = np.array(vlsvobj.read_variable("X",cellids=curr_list),ndmin=1)
            Y = np.array(vlsvobj.read_variable("Y",cellids=curr_list),ndmin=1)
            Z = np.array(vlsvobj.read_variable("Z",cellids=curr_list),ndmin=1)
        else:
            X,Y,Z = ja.xyz_reconstruct(vlsvobj,cellids=curr_list)
            X = np.array(X,ndmin=1)
            Y = np.array(Y,ndmin=1)
            Z = np.array(Z,ndmin=1)

        # Calculate area of one cell
        if n == 0 and vlsvobj.check_variable("DX"):
            dA = vlsvobj.read_variable("DX")[0]*vlsvobj.read_variable("DY")[0]
        elif n == 0 and not vlsvobj.check_variable("DX"):
            dA = ja.get_cell_volume(vlsvobj)

        # If file has more than one population, choose proton population
        var_list = ["rho","v","B","Temperature","CellID","beta","TParallel","TPerpendicular"]
        var_list_alt = ["proton/rho","proton/V","B","proton/Temperature","CellID","proton/beta","proton/TParallel","proton/TPerpendicular"]
        if vlsvobj.check_population("proton"):
            try:
                rho,v,B,T,cellids,beta,TParallel,TPerpendicular = [np.array(vlsvobj.read_variable(s,cellids=curr_list),ndmin=1) for s in var_list_alt]
            except:
                rho,v,B,T,cellids,beta,TParallel,TPerpendicular = [np.array(vlsvobj.read_variable(s,cellids=curr_list),ndmin=1) for s in var_list]
        else:
            rho,v,B,T,cellids,beta,TParallel,TPerpendicular = [np.array(vlsvobj.read_variable(s,cellids=curr_list),ndmin=1) for s in var_list]

        pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

        # Scale variables
        rho /= 1.0e+6
        v /= 1.0e+3
        B /= 1.0e-9
        pdyn /= 1.0e-9
        T /= 1.0e+6
        TParallel /= 1.0e+6
        TPerpendicular /= 1.0e+6

        # Calculate magnitudes of v and B
        vmag = np.linalg.norm(v,axis=-1)
        Bmag = np.linalg.norm(B,axis=-1)

        if type(vmag) == float:
            vmag = np.array(vmag)
        if type(Bmag) == float:
            Bmag = np.array(Bmag)

        # Calculate means, medians and maximums for rho,vmag,Bmag,Pdyn,T,TParallel,TPerpendicular,beta
        n_avg = np.nanmean(rho)
        n_med = np.median(rho)
        n_max = np.max(rho)

        v_avg = np.nanmean(vmag)
        v_med = np.median(vmag)
        v_max = np.max(vmag)

        B_avg = np.nanmean(Bmag)
        B_med = np.median(Bmag)
        B_max = np.max(Bmag)

        pd_avg = np.nanmean(pdyn)
        pd_med = np.median(pdyn)
        pd_max = np.max(pdyn)

        T_avg = np.nanmean(T)
        T_med = np.median(T)
        T_max = np.max(T)

        TPar_avg = np.nanmean(TParallel)
        TPar_med = np.median(TParallel)
        TPar_max = np.max(TParallel)

        TPerp_avg = np.nanmean(TPerpendicular)
        TPerp_med = np.median(TPerpendicular)
        TPerp_max = np.max(TPerpendicular)

        beta_avg = np.nanmean(beta)
        beta_med = np.median(beta)
        beta_max = np.max(beta)

        # Convert X,Y,Z to spherical coordinates
        r = np.linalg.norm(np.array([X,Y,Z]),axis=0)
        theta = np.rad2deg(np.arccos(Z/r))
        phi = np.rad2deg(np.arctan(Y/X))

        # calculate geometric center of jet
        r_mean = np.mean(r)/r_e
        theta_mean = np.mean(theta)
        phi_mean = np.mean(phi)

        # Geometric center of jet in cartesian coordinates
        x_mean = r_mean*np.sin(np.deg2rad(theta_mean))*np.cos(np.deg2rad(phi_mean))
        y_mean = r_mean*np.sin(np.deg2rad(theta_mean))*np.sin(np.deg2rad(phi_mean))
        z_mean = r_mean*np.cos(np.deg2rad(theta_mean))

        # Position of maximum velocity in cartesian coordinates
        x_max = X[vmag==max(vmag)][0]/r_e
        y_max = Y[vmag==max(vmag)][0]/r_e
        z_max = Z[vmag==max(vmag)][0]/r_e

        # Minimum x and density at maximum velocity
        x_min = min(X)/r_e
        rho_vmax = rho[vmag==max(vmag)]
        b_vmax = beta[vmag==max(vmag)]

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

        '''
        0: time [s],
        1: x_mean [R_e],        2: y_mean [R_e],        3: z_mean [R_e],
        4: A [R_e^2],           5: Nr_cells,
        6: r_mean [R_e],        7: theta_mean [deg],    8: phi_mean [deg],
        9: size_rad [R_e],      10: size_tan [R_e],
        11: x_max [R_e],        12: y_max [R_e],        13: z_max [R_e],
        14: n_avg [1/cm^3],     15: n_med [1/cm^3],     16: n_max [1/cm^3],
        17: v_avg [km/s],       18: v_med [km/s],       19: v_max [km/s],
        20: B_avg [nT],         21: B_med [nT],         22: B_max [nT],
        23: T_avg [MK],         24: T_med [MK],         25: T_max [MK],
        26: TPar_avg [MK],      27: TPar_med [MK],      28: TPar_max [MK],
        29: TPerp_avg [MK],     30: TPerp_med [MK],     31: TPerp_max [MK],
        32: beta_avg,           33: beta_med,           34: beta_max
        35: x_min [R_e],        36: rho_vmax [1/cm^3],  37: b_vmax
        38: pd_avg [nPa],       39: pd_med [nPa],       40: pd_max [nPa]
        '''

        # Create temporary property array
        temp_arr = [time,x_mean,y_mean,z_mean,A,Nr_cells,r_mean,theta_mean,phi_mean,size_rad,size_tan,x_max,y_max,z_max,n_avg,n_med,n_max,v_avg,v_med,v_max,B_avg,B_med,B_max,T_avg,T_med,T_max,TPar_avg,TPar_med,TPar_max,TPerp_avg,TPerp_med,TPerp_max,beta_avg,beta_med,beta_max,x_min,rho_vmax,b_vmax,pd_avg,pd_med,pd_max]

        # append properties to property array
        prop_arr = np.append(prop_arr,np.array(temp_arr))

    # reshape property array
    prop_arr = np.reshape(prop_arr,(len(nr_list),len(temp_arr)))

    # write property array to file
    propfile_write(runid,start,jetid,prop_arr,transient)

    return prop_arr

def sort_jets_new(vlsvobj,cells,min_size=0,max_size=3000,neighborhood_reach=[1,1,0]):
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
        it_range = range(200)

        # initialise current event
        curr_event = np.array([cell])

        for n in it_range:

            curr_event_size = curr_event.size

            # find neighbors within the confines of the mask
            curr_event = np.unique(np.append(curr_event,np.intersect1d(cells,get_neighbors(vlsvobj,curr_event,neighborhood_reach))))

            # exit loop if all valid neighbors found
            if curr_event_size == curr_event.size:
                break

        # cast cellids of current event to int and append to list of events
        curr_event = curr_event.astype(int)
        events.append(curr_event)

    # remove events smaller than the minimum size and larger than maximum size
    events_culled = [jet for jet in events if jet.size>=min_size and jet.size<=max_size]

    props = [calc_event_props(vlsvobj,event) for event in events_culled]

    return [events_culled,props]
