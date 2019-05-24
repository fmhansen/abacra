"""
Example script for running the model
"""

import os
import abacra
import abacra.network_creation
import platform
import pandas as pd
import networkx as nx

verbosity = 0


wd = os.getcwd()


plot_path = os.path.join(wd, "figures")
if not os.path.isdir(plot_path):
    os.makedirs(plot_path)

data_path = os.path.join(wd, "data")
if not os.path.isdir(data_path):
    os.makedirs(data_path)

network_file_path = os.path.join(wd, "spatial_network_data")
if not os.path.isdir(network_file_path):
    os.makedirs(network_file_path)

os.chdir(wd)

print("Running application of module in {}".format(wd))
print("\nSaving plots in {}".format(plot_path))

# ========== initialize and run single model ===================

# parameters

pars = dict()

# exemplary setting of parameters
pars["price_elasticity_of_demand"] = 1.
pars["imitation_rate"] = 0.1

run_single_model = True
plot_single_run = True

# set up network creation from car data

car_data_path = "./data/"

centroid_shp_filepath = os.path.join(car_data_path, "centroids_study_region1car_no_overlap.shp")
underlying_map_path = os.path.join(car_data_path, "study_region1car_remove_overlap.shp")
neighborhood_table_path = os.path.join(car_data_path, "distance_matrix_linear_500nearest_meters_feature_id.csv")
attribute_table_path = os.path.join(car_data_path, "car_with_nearest_city+road+slaughterhouse.csv")

initial_pasture = "2016"
if initial_pasture is "2000":
    pars["initial_pasture_key"] = "deforested2000"
elif initial_pasture is "2016":
    pars["initial_pasture_key"] = "deforested2016"
else:
    print("using default inital conditions for pasture")
    pars["initial_pasture_key"] = None

# parameters for network

teleconnection_share = 0.0
rng_seed = 0
rewiring_type = 'rewiring1'
network_model = 'uniform'

network_file = (network_file_path + "/network_studyregion1_"
                + network_model
                + "_telecs" + str(teleconnection_share).replace(".", "o")
                + "_" + rewiring_type
                + "_rngs" + str(rng_seed))

if not os.path.isfile(network_file + ".gml"):
    print("creating network with teleconnection share {}...".format(teleconnection_share))
    g = abacra.network_creation.generate_spatial_network(neighborhood_table_path,
                                                         attribute_table_path,
                                                         centroid_shp_filepath,
                                                         rng_seed=rng_seed,
                                                         frac_rewire=teleconnection_share,
                                                         rewiring_type=rewiring_type,
                                                         network_model=network_model,
                                                         save_filename=(network_file + ".gml"),
                                                         source_type="qgis_neartable",
                                                         compute_measures=True,
                                                         max_distance=10000,
                                                         verbosity=verbosity)

    # do not use target keys with special characters like "_" because saving in gml does not work with them
    g = abacra.network_creation.add_attribute_from_csv(g, os.path.join(car_data_path,
                                                                       "Deforestation_ha_Prodes_on_car_properties_1997-2014_studyregion1.csv"),
                                                       key_to_load="2000_tot", target_key="deforested2000")
    g = abacra.network_creation.add_attribute_from_csv(g, os.path.join(car_data_path,
                                                                       "Deforestation_ha_Prodes_on_car_properties_2012-2016_studyregion1.csv"),
                                                       key_to_load="2016_tot", target_key="deforested2016")

    nx.write_gml(g, network_file + '.gml')

    abacra.network_creation.plot_network_geometric_space(g, network_file + "_geometric_space.pdf",
                                                         underlying_map_path)
    print("plotted network and saved at " + network_file + "_geometric_space.pdf")


if run_single_model:

    par_file = "./abacra/default_parametrized.par"
    # use network_type="gml" to load graph
    model1 = abacra.Model(verbosity=verbosity, network_type="gml", par_dict=pars, par_file="default",
                          network_path=(network_file + ".gml"), rng_seed=10)

    # model1.print_network_properties(shorten=True)

    model1.run(t_max=100)
    print("calculated single model trajectory successfully")

    # stats = model1.compute_stats(stats="all", weighted_qv=True)
    # print(stats)
    # print("successfully computed the stats")

    model1.save_stats(os.path.join(wd, "statistics.csv"), stats="all")
    print("Saved stats to {}".format(os.path.join(wd, "statistics.csv")))

    # save model to pickle file
    # model1.pickle(data_path + "test_pickle.pkl")

    # save all single trajectories to a csv file
    # model1.save_trajectory(data_path + "test_saving_traj.csv")

# plotting
# ------------

if plot_single_run:

    for i in range(1, 50):
        model1.plot_single_agent(os.path.join(plot_path, "single_ranch" + str(i) + ".png"), node_id=i)
        model1.plot_single_agent(os.path.join(plot_path, "single_ranch" + str(i) + "_contr_aux.png"),
                                 node_id=i, plot_controls=True, plot_aux=True)

    # plot network snapshots
    plot_network_snapshots = True
    if plot_network_snapshots:
        for t_plot in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 99]:
            print("Plotting network snapshot for t={}".format(t_plot))
            print("# intensive: {} of {}".format(sum(model1.strategy_traj[t_plot]), model1.no_agents))
            abacra.plotting.plot_pie_network(model1,
                                             os.path.join(plot_path,"plot_network_snapshot_t" + str(t_plot) + ".png"),
                                             t=t_plot, show_landcover_pies=False, annotation="(b)",
                                             xylim=[[-56.4, -54.1],[-9.5, -6.0]])

    model1.plot_stat(os.path.join(plot_path, "plot_mean_with_controls.png"), measure='mean',
                     bounds=None, plot_controls=True, plot_price=True,
                     plot_strategy=True)
    # for preliminary plots in paper
    model1.plot_stat(os.path.join(plot_path, "plot_mean.png"), measure='mean',
                     bounds=None, plot_price=True,
                     plot_strategy=True)
    model1.plot_stat(os.path.join(plot_path, "plot_median.png"), measure='median',
                     bounds="percentiles", percentiles=[25, 75],
                     plot_controls=True, plot_price=True)

# ========== initialize and run model ensemble ===========================

run_ensemble = False
plot_ensemble = False


saving_dir = "./test_ensemble"
model_ensemble1 = abacra.ModelEnsemble(t_max=100, n_runs=5, save_traj=True,
                                       plot_traj=True, initial_rng_seed=0,
                                       saving_dir=saving_dir, verbosity=2)

if run_ensemble:

    model_ensemble1.loop(verbosity=1, network_type="gml", pars=pars,
                         network_path=network_file)


if plot_ensemble:

    model_ensemble1.load_ensemble_stats_csv()

#    model_ensemble1.load_ensemble_stats_pkl()

    # print(model_ensemble1.stats_panel)

    model_ensemble1.plot_ensemble_stats(
        saving_dir + "/ensemble_mean_minmax_price.png",
        ensemble_measure='mean',
        ensemble_bounds='minmax',
        measure="mean", plot_qk=True, plot_controls=True,
        plot_price=False)

quit()