"""
Imports all the modules in the pyJets folder
"""

from jet_analyser import (
    ci2vars,
    ci2vars_nofile,
    read_mult_vars,
    xyz_reconstruct,
    restrict_area,
    bow_shock_markus,
    sw_normalisation,
    bs_mp_fit,
    make_cust_mask_opt,
)

# from jet_contours import *
from jetfile_make import pahkmake
from jet_io import (
    PropReader,
    jet_maker,
    track_jets,
    calc_event_props,
    sort_jets_2,
    eventfile_read,
    get_sheath_cells,
    get_neighbors,
    jet_creator,
    jet_tracker,
    get_neighbors_asym,
    eventprop_read,
)
from jet_scripts import (
    jet_pos_graph,
    jet_paper_pos,
    slamjet_plotter,
    jet_paper_counter,
    jet_plotter,
    MMSReader,
    hack_2019_fig4,
    jetcand_vdf,
    hack_2019_fig6,
    read_mult_runs,
    DT_comparison,
    hack_2019_fig1,
    hack_2019_fig2,
    DT_mach_comparison,
    hack_2019_fig6_alt,
    hack_2019_fig78,
    hack_2019_fig9,
    MMSJet,
    get_timeseries,
    hack_2019_fig35,
    slams_jet_counter,
    make_transient_timeseries,
    h19_extra_1,
    get_SEA,
    plot_new_sj,
    h19_movie,
    jet_maxdiff_counter,
    rev1_jetcone,
    rev1_jetpath,
    rev1_deflection,
    rev1_defplot,
    rev1_jetcone_all,
)
from plot_contours import expr_pdyn_gen, expr_pdyn, expr_srho
from tavg_maker import avg_maker_slow, TP_maker, v_avg_maker, testplot_vavg

# from vspacecraft import jet_sc,jet_spacecrafts,slams_spacecraft,wave_spacecraft
from jet_aux import (
    BS_xy,
    MP_xy,
    get_cell_coordinates,
    make_bs_fit,
    transfer_tavg,
    get_neighs_asym,
    get_neighs,
    bow_shock_jonas,
    ext_magpause,
)
from jet_jh2020 import (
    jh2020_movie,
    get_timeseries_data,
    get_cut_through,
    jh2020_fig2_mesh,
    jh2020_cut_plot,
    sj_non_counter,
    mag_thresh_plot,
    find_one_jet,
    event_424_cut,
    pendep_hist,
    separate_jets_god,
    find_markus_FCS,
)

from jet_21_scripts import (
    make_plots,
    tail_sheet_jplot,
    make_flap_plots,
    tail_sheet_jplot_y,
)
