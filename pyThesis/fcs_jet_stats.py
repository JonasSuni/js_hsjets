import pytools as pt
import jet_jh2020 as jh20
import jet_io as jio
import jet_analyser as ja

def fcs_jet_histogram():

    runids = ["ABA","ABC","AEA","AEC"]
    fcs_counts = [0,0,0,0]
    jet_counts = [0,0,0,0]

    fcs_Dn = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]
    fcs_Dv = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]
    fcs_DB = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]
    fcs_DT = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]

    fcs_dur = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]
    fcs_tsiz = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]
    fcs_srat = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]

    jet_Dn = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]
    jet_Dv = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]
    jet_DB = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]
    jet_DT = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]

    jet_dur = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]
    jet_tsiz = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]
    jet_srat = [np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)]

    for idx,runid in enumerate(runids):
        sj_ids,non_sj_ids,fcs_ids = jh20.separate_jets_god(runid,False)
        jet_ids = np.union1d(sj_ids,non_sj_ids)
        fcs_counts[idx] = fcs_ids.size
        fcs_counts[idx] = fcs_ids.size
        for fcs_id in fcs_ids:
            trans_obj = jio.PropReader(str(fcs_id).zfill(5),runid,580,transient="slams")
            fcs_Dn[idx] = np.append(fcs_Dn[idx],trans_obj.read_at_randt("Dn")/ja.sw_normalisation(runid,"Dn"))
            fcs_Dv[idx] = np.append(fcs_Dn[idx],trans_obj.read_at_randt("Dv")/ja.sw_normalisation(runid,"Dv"))
            fcs_DB[idx] = np.append(fcs_Dn[idx],trans_obj.read_at_randt("DB")/ja.sw_normalisation(runid,"DB"))
            fcs_Dt[idx] = np.append(fcs_Dn[idx],trans_obj.read_at_randt("Dt")/ja.sw_normalisation(runid,"Dt"))

            fcs_dur[idx] = np.append(fcs_dur[idx],trans_obj.read_at_randt("duration"))
            fcs_tsiz[idx] = np.append(fcs_tsiz[idx],trans_obj.read_at_randt("size_tan"))
            fcs_srat[idx] = np.append(fcs_srat[idx],trans_obj.read_at_randt("size_ratio"))

        for jet_id in jet_ids:
            trans_obj = jio.PropReader(str(jet_id).zfill(5),runid,580,transient="jet")
            jet_Dn[idx] = np.append(jet_Dn[idx],trans_obj.read_at_randt("Dn")/ja.sw_normalisation(runid,"Dn"))
            jet_Dv[idx] = np.append(jet_Dn[idx],trans_obj.read_at_randt("Dv")/ja.sw_normalisation(runid,"Dv"))
            jet_DB[idx] = np.append(jet_Dn[idx],trans_obj.read_at_randt("DB")/ja.sw_normalisation(runid,"DB"))
            jet_Dt[idx] = np.append(jet_Dn[idx],trans_obj.read_at_randt("Dt")/ja.sw_normalisation(runid,"Dt"))

            jet_dur[idx] = np.append(jet_dur[idx],trans_obj.read_at_randt("duration"))
            jet_tsiz[idx] = np.append(jet_tsiz[idx],trans_obj.read_at_randt("size_tan"))
            jet_srat[idx] = np.append(jet_srat[idx],trans_obj.read_at_randt("size_ratio"))
