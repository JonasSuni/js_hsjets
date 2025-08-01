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
from tavg_maker import (
    avg_maker_slow,
    TP_maker,
    v_avg_maker,
    testplot_vavg,
    extract_var,
    tavg_maker_2023,
)

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
    bs_mp_fit,
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
    vfield3_grad,
    vfield3_dot,
    vfield3_matder,
    plot_ballooning,
    tail_sheet_jplot_balloon,
    vfield3_curl,
    vfield3_normalise,
    fac_migration_plot,
    plot_residual_bz,
    maxime2_diff,
)

from papu_2 import (
    sj_non_timeseries,
    SEA_plots,
    fcs_non_jet_hist,
    colormap_with_contours,
    papu22_mov_script,
    non_jet_jplots,
    SEA_types,
    non_type_hist,
    types_jplot_SEA,
    jet_pos_plot,
    P_jplots,
    types_P_jplot_SEA,
    fcs_jet_jplot_txtonly,
    foreshock_jplot_SEA,
    jet_vdf_plotter,
    vdf_plotter,
    jet_animator,
    kind_animations,
    kind_timeseries,
    jet_avg_std,
    kind_SEA_timeseries,
    trifecta,
    SEA_trifecta,
    non_jet_omni,
    jmap_SEA_comp,
    SEA_timeseries_comp,
    timing_comp,
    kinds_pca,
    print_means_max,
    jet_var_plotter,
    vz_timeseries,
    # quadfecta,
    # fecta_9,
    weighted_propagation_velocity,
    auto_classifier,
    fig0,
    jet_counter,
    jet_vdf_profile_plotter,
    filter_fcs_jets,
    kind_size_hist,
    fig02_alt,
    fig07_alt,
    SEA_timeseries_comp_violin,
    clock_angle_comp,
)

from jet_23_scripts import (
    ani_timeseries,
    multi_VSC_timeseries,
    fincospar_plots,
)

import agf_jets

import jets_2025

import foreshock_3d

try:
    import satellite
except:
    print("Did not import satellite")