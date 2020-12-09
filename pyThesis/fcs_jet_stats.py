import pytools as pt
import jet_jh2020 as jh20

def fcs_jet_histogram(transient="jet"):

    runids = ["ABA","ABC","AEA","AEC"]
    transient_id_list = []
    for runid in runids:
        sj_ids,non_sj_ids,fcs_ids = jh20.separate_jets_god(runid,False)
        if transient == "jet":
            transient_id_list.append(np.union1d(sj_ids,non_sj_ids))
        else:
            transient_id_list.append(fcs_ids)
