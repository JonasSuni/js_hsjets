def make_cust_mask_opt_new(filenumber,runid,halftimewidth=180,boxre=[6,18,-8,6],avgfile=False,transient="jet"):
    # finds cellids of cells that fulfill the specified criterion and the specified
    # X,Y-limits

    if transient == "jet":
        trans_folder = "jets/"
    elif transient == "slams":
        trans_folder = "SLAMS/"
    elif transient == "slamsjet":
        trans_folder = "SLAMSJETS/"

    bulkpath = find_bulkpath(runid)

    bulkname = "bulk."+str(filenumber).zfill(7)+".vlsv"

    if bulkname not in os.listdir(bulkpath):
        print("Bulk file "+str(filenumber)+" not found, exiting.")
        return 1

    # open vlsv file for reading
    vlsvreader = pt.vlsvfile.VlsvReader(bulkpath+bulkname)

    if not not boxre:
        cells = restrict_area(vlsvreader,boxre)
        cells.sort()
    else:
        cells = -1

    # if file has separate populations, read proton population
    if type(vlsvreader.read_variable("rho")) is not np.ndarray:
        rho = vlsvreader.read_variable("proton/rho",cellids=cells)
        v = vlsvreader.read_variable("proton/V",cellids=cells)
        B = vlsvreader.read_variable("B",cellids=cells)
        tvars = ["proton/rho","proton/V"]
    else:
        rho = vlsvreader.read_variable("rho",cellids=cells)
        v = vlsvreader.read_variable("v",cellids=cells)
        B = vlsvreader.read_variable("B",cellids=cells)
        tvars = ["rho","v"]

    if vlsvreader.check_variable("X"):
        X,Y,Z = [vlsvreader.read_variable("X",cellids=cells),vlsvreader.read_variable("Y",cellids=cells),vlsvreader.read_variable("Z",cellids=cells)]
    else:
        X,Y,Z = xyz_reconstruct(vlsvreader,cellids=cells)

    X,Y,Z = [X/r_e,Y/r_e,Z/r_e]

    x_res = spat_res(runid)

    p = bow_shock_markus(runid,filenumber) #PLACEHOLDER

    bs_cond = X-np.polyval(p[::-1],Y)

    # x-directional dynamic pressure
    spdynx = m_p*rho*(v[:,0]**2)

    # dynamic pressure
    pdyn = m_p*rho*(np.linalg.norm(v,axis=-1)**2)

    Bmag = np.linalg.norm(B,axis=-1)

    sw_pars = sw_par_dict(runid)
    rho_sw = sw_pars[0]
    pdyn_sw = sw_pars[3]
    B_sw = sw_pars[2]

    npdyn = pdyn/pdyn_sw
    nrho = rho/rho_sw

    # initialise time average of dynamic pressure
    tpdynavg = np.zeros(pdyn.shape)

    # range of timesteps to calculate average of
    timerange = range(filenumber-halftimewidth,filenumber+halftimewidth+1)

    missing_file_counter = 0

    vlsvobj_list = []

    if avgfile:
        tpdynavg = np.load(tavgdir+"/"+runid+"/"+str(filenumber)+"_pdyn.npy")
        tpdynavg = tpdynavg[cells-1]
    else:

        for n_t in timerange:

            # exclude the main timestep
            if n_t == filenumber:
                continue

            # find correct file path for current time step
            tfile_name = "bulk."+str(n_t).zfill(7)+".vlsv"

            if tfile_name not in os.listdir(bulkpath):
                print("Bulk file "+str(n_t)+" not found, continuing")
                continue

            # open file for current time step
            vlsvobj_list.append(pt.vlsvfile.VlsvReader(bulkpath+tfile_name))

        for f in vlsvobj_list:

            f.optimize_open_file()

            trho = f.read_variable(tvars[0],cellids = cells)
            tv = f.read_variable(tvars[1],cellids = cells)

            # dynamic pressure for current time step
            tpdyn = m_p*trho*(np.linalg.norm(tv,axis=-1)**2)

            tpdynavg = np.add(tpdynavg,tpdyn)

            f.optimize_clear_fileindex_for_cellid()
            f.optimize_close_file()

        # calculate time average of dynamic pressure
        tpdynavg /= len(vlsvobj_list)

    # prevent divide by zero errors
    tpdynavg[tpdynavg == 0.0] = 1.0e-27

    # ratio of dynamic pressure to its time average
    tapdyn = pdyn/tpdynavg

    # make custom jet mask
    if transient == "jet":
        jet = np.ma.masked_greater_equal(tapdyn,2.0)
        jet.mask[bs_cond-2*x_res-0.5 > 0] = False
    elif transient == "slams":
        jet = np.ma.masked_greater_equal(Bmag,1.5*B_sw)
        jet.mask[bs_cond+2*x_res+0.5 < 0] = False
        jet.mask[nrho < 1.5] = False
        jet.mask[npdyn < 1.25] = False
    elif transient == "slamsjet":
        jet1 = np.ma.masked_greater_equal(Bmag,1.5*B_sw)
        jet1.mask[bs_cond+2*x_res+0.5 < 0] = False
        jet1.mask[nrho < 1.5] = False
        jet1.mask[npdyn < 1.25] = False
        jet2 = np.ma.masked_greater_equal(tapdyn,2.0)
        jet2.mask[bs_cond-2*x_res-0.5 > 0] = False
        jet = np.logical_or(jet1,jet2)

    # discard unmasked cellids
    masked_ci = np.ma.array(cells,mask=~jet.mask).compressed()

    if not os.path.exists("{}working/{}Masks/{}/".format(wrkdir_DNR,trans_folder,runid)):
        os.makedirs("{}working/{}Masks/{}/".format(wrkdir_DNR,trans_folder,runid))

    print("Writing to "+"{}working/{}Masks/{}/".format(wrkdir_DNR,trans_folder,runid)+str(filenumber)+".mask")
    #print(masked_ci[69])

    np.savetxt("{}working/{}Masks/{}/".format(wrkdir_DNR,trans_folder,runid)+str(filenumber)+".mask",masked_ci)
    return masked_ci
