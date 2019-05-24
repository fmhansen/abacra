"""
Example script for running the model several times using the package pymofa and aggregating the results
Settings in the beginning of the script can be used to generate the figures from the publication Müller-Hansen et al., Ecol. Econ. 2019.

install of pymofa required (pymofa needs ubuntu package libopenmpi-dev)
for install instructions, see https://github.com/wbarfuss/pymofa

"""

import os
import abacra
import abacra.network_creation
import abacra.plotting
from pymofa.experiment_handling import experiment_handling as eh
import itertools as it
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import platform
import random
import sys
import networkx as nx
random.seed(0)  # for reproducibility

print("Running experiment handling with pymofa")
print("on {}\n".format(sys.version))

# 2D parameter plots
# settings for imitationr_elasticity or teleconnection_imitationr plots or additional parameter runs (choose only one!)
imitationr_elasticity_runs = False
if imitationr_elasticity_runs:
    print("Using setup for imitation rate - elasticity plots")

teleconnection_imitationr_runs = False
if teleconnection_imitationr_runs:
    print("Using setup for imitation rate - teleconnection share plots")

additional_parameter_runs = False
additional_parameter_name = "maximally convertible fraction of property"
# additional_parameter_name = "intensification_credit_limit_per_ha"
if additional_parameter_runs:
    print("Using setup for plots imitation rate - additional parameter: {}".format(additional_parameter_name))
else:
    additional_parameter_name = None

sampled_sensitivity_runs = False
if sampled_sensitivity_runs:
    randomized_parnames = ["savings_rate", "intensification_cost", "def_cost"]
    print("Using randomization of parameters")

sensitivity_trajectories = True
if sensitivity_trajectories:
    #sensitivity_parameter_key = "intensification_credit_limit_per_ha"
    #sensitivity_parameter_symbol = "$k_{min}$"
    #sensitivity_parameter_values = [0, 200, 400]

    #sensitivity_parameter_key = "elasticity"
    #sensitivity_parameter_symbol = "$\epsilon$"
    #sensitivity_parameter_values = [1., 10.,  100.]

    #sensitivity_parameter_key = "def_cost"
    #sensitivity_parameter_symbol = "$c_D$"
    #sensitivity_parameter_values = [1000, 1500, 2000, 3000]

    #sensitivity_parameter_key = "intensification_cost"
    #sensitivity_parameter_symbol = "$c_I$"
    #sensitivity_parameter_values = [300, 500, 800, 1000]

    #sensitivity_parameter_key = "imitation_rate"
    #sensitivity_parameter_symbol = "$\lambda$"
    #sensitivity_parameter_values = [0.1, 1, 10]
    ## sensitivity_parameter_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    sensitivity_parameter_key = "teleconnection_share"
    sensitivity_parameter_symbol = "$\alpha$"
    sensitivity_parameter_values = [0.0, 0.02, 0.1]
    ##sensitivity_parameter_values = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]


default_teleconnection_share = 0.02
default_elasticity = 100.
default_imitation_rate = 1.
t_max = 100

verbosity = 0

if platform.node().startswith("cs-") or platform.node().startswith("login"):
    print("Using settings for cluster")
    on_cluster = True
else:
    on_cluster = False
    print("Using settings for local machine")

if on_cluster:
    # to solve issues with displaying
    plt.switch_backend('agg')

    wd = os.getcwd()
    test = False
    single_runs = False
    single_agent_paperplot = False
    sampling_resolution = "high"
    no_single_agent_trajectories = 10
    SAVE_PATH_EXPERIMENTS = wd + "/experiment_data"
    SAVE_PATH_RES = wd + "/parameter_data"
    SAVE_PATH_FIGS = wd + "/figures"
    single_agent_figure_path = os.path.join(wd, "single_agent_figures")
    network_file_path = wd + "/../spatial_network_data"
    car_data_path = "./data/"

else:
    wd = os.getcwd()
    save_dir = "/media/Data/Projects/Amazon_ABM/Local_runs_with_pymofa/run_with_pymofa_new_single_ranch_plots"
    test = True
    single_runs = False
    single_agent_paperplot = True
    sampling_resolution = "test"
    no_single_agent_trajectories = 200

    SAVE_PATH_EXPERIMENTS = save_dir + "/experiment_data"
    SAVE_PATH_RES = save_dir + "/parameter_data"
    SAVE_PATH_FIGS = save_dir + "/figures"
    single_agent_figure_path = os.path.join(save_dir, "single_agent_figures")
    network_file_path = "./Local_runs_with_pymofa/spatial_network_data"
    car_data_path = "/media/Data/Projects/Amazon_ABM/CAR_data/Studyregion1/"

    np.seterr(all='raise', under='warn')

par_file = os.path.join(wd, "default_parametrized.par")

if not os.path.isfile(par_file):
    print("falling back to parameter file in abacra package")
    package_path = os.path.dirname(abacra.__file__)
    par_file = os.path.join(package_path, "../default_parametrized.par")
    assert os.path.isfile(par_file)

try:
    os.makedirs(SAVE_PATH_FIGS)
    print("created directory for figures: {}".format(SAVE_PATH_FIGS))
except FileExistsError:
    print("Directory for saving figures {} already exists".format(SAVE_PATH_FIGS))

try:
    os.makedirs(single_agent_figure_path)
    print("created directory for figures: {}".format(single_agent_figure_path))
except FileExistsError:
    print("Directory for saving figures {} already exists".format(single_agent_figure_path))

try:
    os.makedirs(network_file_path)
    print("created directory for figures: {}".format(network_file_path))
except FileExistsError:
    print("Directory for saving figures {} already exists".format(network_file_path))

# statistical measure to use for ensemble averages
ensemble_stat_measure = 'mean'


# ==================================================================================================

# copied and modified from pymofa.experiment_handling.py
def get_pymofa_id(parameter_combination):
    """
    Get a unique ID for a `parameter_combination` and ensemble index `i`.

    ID is of the form 'parameterID_index_ID.pkl'

    Parameters
    ----------
    parameter_combination : tuple
        The combination of Parameters

    Returns
    -------
    ID : string
        unique ID or pattern plus the ".pkl" ending
    """
    res = str(parameter_combination)  # convert to sting
    res = res[1:-1]  # delete brackets
    res = res.replace(", ", "-")  # remove ", " with "-"
    res = res.replace(".", "o")  # replace dots with an "o"
    res = res.replace("'", "")  # remove 's from values of string variables
    # Remove all the other left over mean
    # characters that might fuck with you
    # bash scripting or wild card usage.
    for mean_character in "[]()^ #%&!@:+={}'~":
        res = res.replace(mean_character, "")
    return res

# ==================================================================================================

# Parameter combinations to investigate

if sampling_resolution is "high":
    imitation_rates = np.logspace(-3, 1, num=21)  # (high - low) * sample_density + 1
    elasticities = np.logspace(-1, 3, num=21)
    teleconnection_shares = np.linspace(0.0, 0.1, num=11)
elif sampling_resolution is "low":
    imitation_rates = np.logspace(-3, 1, num=9)
    elasticities = np.logspace(-1, 3, num=9)
    teleconnection_shares = np.linspace(0.0, 0.1, num=6)
else:
    print("resolution by default set to 'low'")
    imitation_rates = np.logspace(-3, 1, num=9)
    elasticities = np.logspace(-1, 3, num=9)
    teleconnection_shares = np.linspace(0.0, 0.1, num=6)

if imitationr_elasticity_runs:
    teleconnection_shares = [default_teleconnection_share]

if teleconnection_imitationr_runs:
    elasticities = [default_elasticity]

if additional_parameter_runs:
    if additional_parameter_name is "D":
        additional_parameters = np.linspace(0.01, 0.1, num=10)
    elif additional_parameter_name is "intensification_credit_limit_per_ha":
        additional_parameters = np.linspace(0, 500, num=6)
    else:
        additional_parameters = np.linspace(0, 1., num=6)

    elasticities = [default_elasticity]
    teleconnection_shares = [default_teleconnection_share]
else:
    additional_parameters = [0]

if sensitivity_trajectories:
    if sensitivity_parameter_key == "elasticity":
        teleconnection_shares = [default_teleconnection_share]
        imitation_rates = [default_imitation_rate]
        elasticities = sensitivity_parameter_values
    elif sensitivity_parameter_key == "imitation_rate":
        teleconnection_shares = [default_teleconnection_share]
        elasticities = [default_elasticity]
        imitation_rates = sensitivity_parameter_values
    elif sensitivity_parameter_key == "teleconnection_share":
        teleconnection_shares = sensitivity_parameter_values
        elasticities = [default_elasticity]
        imitation_rates = [default_imitation_rate]
    else:
        teleconnection_shares = [default_teleconnection_share]
        elasticities = [default_elasticity]
        imitation_rates = [default_imitation_rate]
        additional_parameter_name = sensitivity_parameter_key
        additional_parameters = sensitivity_parameter_values

if test and not sensitivity_trajectories:
    imitation_rates = [0.001]
    elasticities = [100.]
    teleconnection_shares = [0.02]
    additional_parameters = [0.05]

elif single_runs:
    imitation_rates = [0.1, 1.]
    elasticities = [1., 100.]
    teleconnection_shares = [0.02]
    additional_parameters = [0.05]

PARAM_COMBS = list(it.product(imitation_rates, elasticities, teleconnection_shares, additional_parameters))

# Sample Size
if test:
    SAMPLE_SIZE = 2
elif single_runs:
    SAMPLE_SIZE = 1000
else:
    if sampling_resolution is "high":
        SAMPLE_SIZE = 100
    elif sampling_resolution is "low":
        SAMPLE_SIZE = 5
    else:
        SAMPLE_SIZE = 2

# ==================================================================================================

# Defining the experiment execution function
# it gets parameter you want to investigate, plus `filename` as the last parameter
def run_func(imitation_rate, elasticity, teleconnection_share, additional_parameter, filename):
    """Run func."""

    # ========== initialize and run single model ===================

    pars = dict()

    # pars["price_feedback"] = True
    pars["price_elasticity_of_demand"] = elasticity

    # pars["initial_strategy_distance_threshold"] = 100000.  # in meters
    pars["intensification_cost"] = 500.
    pars["intensification_credit_limit_per_ha"] = 200.0

    # pars["available_area_frac"] = 0.2
    # pars["D_ext"] = pars["D_int"] = 0.25
    # pars["A_int"] = pars["A_ext"] = 0.25
    # pars["R_int"] = pars["R_ext"] = 0.25

    # pars["beta"] = 0.2
    # pars["income_investment_comp_factor_ext"] = 4

    # switch for relative vs. absolute areas
    # pars["relative_area"] = False

    # for no regeneration of secondary vegetation to primary forest
    # model1.pars["r_N"] = 0.

    pars["m"] = 1.55
    pars["man_cost"] = 100

    pars["imitation_rate"] = imitation_rate

    # additional parameter settings

    if additional_parameter_name is not None:
        if additional_parameter_name is "maximally convertible fraction of property":
            pars["D_ext"] = pars["D_int"] = additional_parameter
            pars["A_int"] = pars["A_ext"] = additional_parameter
            pars["R_int"] = pars["R_ext"] = additional_parameter
        else:
            pars[additional_parameter_name] = additional_parameter

    rng_seed = random.randint(0, 2**32-1)

    # settings for network generation
    source_type = "qgis_neartable"  # alternatively = "arcgis_neartable"
    data_set_name = "studyregion1"

    centroid_shp_filepath = os.path.join(car_data_path, "centroids_study_region1car_no_overlap.shp")
    underlying_map_path = os.path.join(car_data_path, "study_region1car_remove_overlap.shp")
    neighborhood_table_path = os.path.join(car_data_path, "distance_matrix_linear_500nearest_meters_feature_id.csv")
    attribute_table_path = os.path.join(car_data_path, "car_with_nearest_city+road+slaughterhouse.csv")

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
                                                             source_type="qgis_neartable",
                                                             compute_measures=True,
                                                             max_distance=10000,
                                                             spatial_decay_range=10000,
                                                             verbosity=verbosity)

        # do not use target keys with special characters like "_" because saving in gml does not work with them
        g = abacra.network_creation.add_attribute_from_csv(g, os.path.join(car_data_path,
                                                 "Deforestation_ha_Prodes_on_car_properties_1997-2014_studyregion1.csv"),
                                                           key_to_load="2000_tot", target_key="deforested2000")
        g = abacra.network_creation.add_attribute_from_csv(g, os.path.join(car_data_path,
                                                 "Deforestation_ha_Prodes_on_car_properties_2012-2016_studyregion1.csv"),
                                                           key_to_load="2016_tot", target_key="deforested2016")

        nx.write_gml(g, network_file + '.gml')

        abacra.network_creation.plot_network_geometric_space(g, network_file + "_geometric_space.png",
                                                             underlying_map_path)
        print("plotted network and saved at " + network_file + "_geometric_space.png")

    initial_pasture = "2000"
    if initial_pasture is "2000":
        pars["initial_pasture_key"] = "deforested2000"
    elif initial_pasture is "2016":
        pars["initial_pasture_key"] = "deforested2016"
    else:
        print("using default inital conditions for pasture")
        pars["initial_pasture_key"] = None

    # initializing model
    model1 = abacra.Model(verbosity=verbosity, network_type="gml",
                          par_file=par_file, par_dict=pars,
                          network_path=network_file + ".gml", rng_seed=rng_seed)

    print(model1.pars)

    if sampled_sensitivity_runs:
        randomized_pars = {}

        for parameter in randomized_parnames:
            randomized_pars[parameter] = random.uniform(model1.pars[parameter + "_rmin"],
                                                        model1.pars[parameter + "_rmax"])

        print("Randomized parameters: {}".format(randomized_pars))

        model1.pars.update(randomized_pars)

    # model1.print_network_properties(shorten=True)

    # running the model
    model1.run(t_max=t_max)
    print("calculated single model trajectory successfully")

    if os.path.splitext(filename)[0].endswith("_s0"):
        file_id = os.path.splitext(os.path.basename(filename))[0]
        fig_path = os.path.join(single_agent_figure_path, file_id)

        try:
            os.makedirs(fig_path)
        except FileExistsError:
            pass

        fig_path = os.path.join(single_agent_figure_path, file_id)
        model1.plot_pars["dpi_saving"] = 200

        model1.plot_stat(os.path.join(fig_path, 'model_stat_' + file_id + '.png'),
                         plot_controls=True,
                         plot_strategy=True,
                         plot_qk=True,
                         plot_price=True,
                         secveg_dynamics=True,
                         plot_active_ranches=True,
                         t_min=1)

        for agent_id in range(no_single_agent_trajectories):
            if single_agent_paperplot:

                abacra.plotting.plot_single_agent(model1, os.path.join(fig_path,
                                                  'single_agent' + str(agent_id) + '_' + file_id + '_paperplot.png'),
                                                  node_id=agent_id, secveg_dynamics=True)
                #abacra.plotting.plot_single_agent(model1, os.path.join(fig_path,
                #                                  'single_agent' + str(agent_id) + '_' + file_id + '_presiplot.png'),
                #                                  node_id=agent_id, secveg_dynamics=True, plot_qk=False)

            abacra.plotting.plot_single_agent(model1, os.path.join(fig_path,
                                              'single_agent' + str(agent_id) + '_' + file_id + '_controlplot.png'),
                                              node_id=agent_id, plot_controls=True, plot_aux=True, secveg_dynamics=True)

        model1.print_settings_to_file(os.path.join(fig_path, file_id + "_settings.txt"))

    model1.save_stats(filename, stats="all")
    print("saved stats")

    # RUN_FUNC needs to return exit_status (< failed, > passed)
    return 1


# ====================================================================================================


# INDEX -> dict with argument number and argument names for post-processing
INDEX = {i: run_func.__code__.co_varnames[i] for i in range(run_func.__code__.co_argcount-1)}
# print(INDEX)

# initiate handle instance with experiment variables
handle = eh(SAMPLE_SIZE, PARAM_COMBS, INDEX, SAVE_PATH_EXPERIMENTS,
            path_res=SAVE_PATH_RES)

# Compute experiments raw data
handle.compute(run_func)

# range for average
t_min_avg = 0
t_max_avg = 50


def all_ensemble_stats(filenames):
    """
    evaluate ensemble statistics
    """

    return abacra.ensemble.ensemble_stats_from_csv(filenames, verbosity=verbosity)


def min_forest(filenames):

    _ensemble_stats_df = abacra.ensemble.ensemble_stats_from_csv(filenames,
                                                                 ensemble_stat_measures=ensemble_stat_measure,
                                                                 verbosity=verbosity)

    min_def = _ensemble_stats_df['F', 'mean'][ensemble_stat_measure][t_min_avg:t_max_avg].min()

    return min_def


def t_min_forest(filenames):
    _ensemble_stats_df = abacra.ensemble.ensemble_stats_from_csv(filenames,
                                                                 ensemble_stat_measures=ensemble_stat_measure,
                                                                 verbosity=verbosity)

    t_min_def = _ensemble_stats_df['F', 'mean'][ensemble_stat_measure][t_min_avg:t_max_avg].idxmin()

    return t_min_def


def time_avg_deforestation(filenames):
    _ensemble_stats_df = abacra.ensemble.ensemble_stats_from_csv(filenames,
                                                                 ensemble_stat_measures=ensemble_stat_measure,
                                                                 verbosity=verbosity)

    time_avg_def = _ensemble_stats_df['d', 'mean'][ensemble_stat_measure][t_min_avg:t_max_avg].mean()

    return time_avg_def


def avg_delta_forest(filenames):
    _ensemble_stats_df = abacra.ensemble.ensemble_stats_from_csv(filenames,
                                                                 ensemble_stat_measures=ensemble_stat_measure,
                                                                 verbosity=verbosity)

    avg_delta_f = _ensemble_stats_df['F', 'mean'][ensemble_stat_measure][t_min_avg:t_max_avg].diff().mean() * (-1.)

    return avg_delta_f


def avg_cattle_production(filenames):

    _ensemble_stats_df = abacra.ensemble.ensemble_stats_from_csv(filenames,
                                                                 ensemble_stat_measures=ensemble_stat_measure,
                                                                 verbosity=verbosity)

    avg_cattle_production_value = _ensemble_stats_df['cattle_quantity', 'mean'][ensemble_stat_measure][t_min_avg:t_max_avg].mean()

    return avg_cattle_production_value


def max_ratio_intensive(filenames):

    _ensemble_stats_df = abacra.ensemble.ensemble_stats_from_csv(filenames,
                                                                 ensemble_stat_measures=ensemble_stat_measure,
                                                                 verbosity=verbosity)

    max_ratio_intensive_value = _ensemble_stats_df['strategy', 'mean'][ensemble_stat_measure][t_min_avg:t_max_avg].max()

    return max_ratio_intensive_value


def avg_consumption(filenames):

    _ensemble_stats_df = abacra.ensemble.ensemble_stats_from_csv(filenames,
                                                                 ensemble_stat_measures=ensemble_stat_measure,
                                                                 verbosity=verbosity)

    avg_consumption_value = _ensemble_stats_df['I', 'mean'][ensemble_stat_measure][t_min_avg:t_max_avg].mean()

    return avg_consumption_value


recalculate = True
filename_parameters = "stateval_results.pkl"

print("evaluating different parameter settings...")
# callables can be functions, lambda expressions etc...
EVAL = {"min_forest": min_forest, "t_min_forest": t_min_forest, "time_avg_def": time_avg_deforestation,
        "avg_delta_f": avg_delta_forest, "avg_cattle_production": avg_cattle_production,
        "max_ratio_intensive": max_ratio_intensive, "avg_consumption": avg_consumption}

z_vars = EVAL.keys()
z_var_labels = ["min forest [ha]", "t at min forest [yr]", "mean deforestation rate [ha/a]", "<d F> [ha/a]",
                "<Y> [head/a]", r"max $N_I/N$", "<C> [$R]"]
vmins = [None, None,  0, None, None, None, None]
vmaxs = [None, None,  9, None, None, None, None]

# for all time series:
# EVAL = {"ensemble_stats": ensemble_stats}
if recalculate:
    handle.resave(EVAL, filename_parameters)


# ===================================================================================================================
# ====================               plot parameter dependencies in 2D     ==========================================
# ===================================================================================================================

if imitationr_elasticity_runs or teleconnection_imitationr_runs or additional_parameter_runs:

    data = pd.read_pickle(os.path.join(SAVE_PATH_RES, filename_parameters))

    if verbosity > 0:
        print("Data for plotting:")
        print(data.head())

    # 0: imitation_rates, 1: elasticities, 2: teleconnection_shares
    x_id = 0

    if imitationr_elasticity_runs:
        y_id = 1
    elif teleconnection_imitationr_runs:
        y_id = 2
    elif additional_parameter_runs:
        y_id = 3
    else:
        y_id = 3

    for z_var_index, z_var in enumerate(z_vars):

        X = data.index.levels[x_id].values
        xname = data.index.levels[x_id].name

        Y = data.index.levels[y_id].values
        if additional_parameter_runs:
            yname = additional_parameter_name
        else:
            yname = data.index.levels[y_id].name

        if imitationr_elasticity_runs:
            Z = data[z_var].xs((teleconnection_shares[0], additional_parameters[0]),
                               level=['teleconnection_share', 'additional_parameter']).values.reshape([X.size, Y.size]).T
            yname = "price_elasticity"
        elif teleconnection_imitationr_runs:
            Z = data[z_var].xs((elasticities[0], additional_parameters[0]),
                               level=['elasticity', 'additional_parameter']).values.reshape([X.size, Y.size]).T
        elif additional_parameter_runs:
            Z = data[z_var].xs((elasticities[0], teleconnection_shares[0]),
                               level=['elasticity', 'teleconnection_share']).values.reshape([X.size, Y.size]).T

        zname = z_var_labels[z_var_index]

        Xi, Yi = np.meshgrid(X, Y)

        if verbosity > 1:
            print(X)
            print(Y)
            print(Z)

        # plot contour
        fig, ax = plt.subplots()

        ax.set_xscale("log")
        if imitationr_elasticity_runs:
            ax.set_yscale("log")
        ax.set_xlabel(xname.replace("_", " "), fontsize=16)
        ax.set_ylabel(yname.replace("_", " "), fontsize=16)

        cs = ax.contourf(Xi, Yi, Z, alpha=1)  # cmap=plt.cm.jet)

        cbar = fig.colorbar(cs)
        cbar.set_label(label=zname, fontsize=16)

        plt.tight_layout()

        plt.savefig(SAVE_PATH_FIGS + "/2Dcplot_" + xname + "-" + yname + "-" + z_var + ".png")

        plt.clf()

        # plot pcolormesh
        fig, ax = plt.subplots()

        ax.set_xscale("log")
        if imitationr_elasticity_runs:
            ax.set_yscale("log")
        ax.set_xlabel(xname.replace("_", " "), fontsize=16)
        ax.set_ylabel(yname.replace("_", " "), fontsize=16)

        cs = ax.pcolormesh(Xi, Yi, Z, alpha=1, vmin=vmins[z_var_index], vmax=vmaxs[z_var_index])  # cmap=plt.cm.jet)

        cbar = fig.colorbar(cs)
        cbar.set_label(label=zname, fontsize=16)

        plt.tight_layout()

        plt.savefig(SAVE_PATH_FIGS + "/2Dpcolor_" + xname + "-" + yname + "-" + z_var + ".png")

        plt.clf()


# ===================================================================================================================
# ====================               plot parameter dependencies in trajectories     ================================
# ===================================================================================================================

if sensitivity_trajectories:

    no_plots = 5
    fig, ax = plt.subplots(no_plots, sharex='all', figsize=(6, no_plots * 2 + 1))

    fig2 = plt.figure(figsize=(6, 3))
    axf = plt.gca()

    lw = [2., 2., 2., 2., 1., 1., 1., 1., 3., 3., 3., 3.]
    ls = ['--', '-', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.']

    for i, parameters in enumerate(PARAM_COMBS):
        # parameter_str = "-".join([str(i) for i in parameters]).replace(".", "o")

        parameter_str = get_pymofa_id(parameters)

        fnames = []
        for file in os.listdir(SAVE_PATH_EXPERIMENTS):
            if file.startswith(parameter_str + "_s") and file.endswith('.pkl'):
                fnames.append(os.path.join(SAVE_PATH_EXPERIMENTS, file))

        if fnames:

            if verbosity > 0:
                print("loading files to calculate ensemble statistics (average over {} runs)".format(len(fnames)))

            if verbosity > 1:
                print("file list: {}".format(', '.join(fnames)))

            percentiles = [5, 95]
            ensemble_stats_df = abacra.ensemble.ensemble_stats_from_csv(fnames, ensemble_stat_measures='all',
                                                                        percentiles=percentiles)

            # unstack converts level of one axis to another
            ensemble_stats_df = ensemble_stats_df.unstack(level='ensemble_stat_measure')

            # slicing data frame
            ensemble_stats_df = ensemble_stats_df.xs('mean', level=1, axis=1, drop_level=True)
            # level=1 selects the level with the statistics of individual runs

            t = ensemble_stats_df.index.values
            t_min = 0
            if t_min is None:
                t_min = min(t)
            t_max = max(t)

            t = t[t_min:t_max]

            if sensitivity_parameter_key == "imitation_rate":
                ll = r"$\lambda$ = {0:g}".format(parameters[0])
            elif sensitivity_parameter_key == "elasticity":
                ll = r"$\epsilon$ = {0:g}".format(parameters[1])
            elif sensitivity_parameter_key == "teleconnection_share":
                ll = r"$\alpha$ = {0:g}".format(parameters[2])
            else:
                ll = r"{0} = {1:g}".format(sensitivity_parameter_symbol, parameters[3])

            lb_label = "perc" + str(percentiles[0])
            ub_label = "perc" + str(percentiles[1])
            range_alpha = 0.2

            axf.plot(t, ensemble_stats_df['F', 'median'][t_min:t_max], color='darkgreen',
                     linewidth=lw[i], linestyle=ls[i], label=ll)

            ax[0].plot(t, ensemble_stats_df['F', 'median'][t_min:t_max], color='darkgreen',
                       linewidth=lw[i], linestyle=ls[i], label=ll)
            ax[1].plot(t, ensemble_stats_df['P', 'median'][t_min:t_max], color='lawngreen',
                       linewidth=lw[i], linestyle=ls[i], label=ll)
            ax[2].plot(t, ensemble_stats_df['q', 'median'][t_min:t_max], color='saddlebrown',
                       linewidth=lw[i], linestyle=ls[i], label=ll)
            ax[3].plot(t, ensemble_stats_df['strategy', 'mean'][t_min:t_max], color='k',
                       linewidth=lw[i], linestyle=ls[i], label=ll)
            ax[4].plot(t, ensemble_stats_df['cattle_quantity', 'median'][t_min:t_max]/1000000, color='r',
                       linewidth=lw[i], linestyle=ls[i], label=ll)

            plot_ranges = False
            if plot_ranges:
                axf.fill_between(t, ensemble_stats_df["F", lb_label][t_min:t_max],
                                 ensemble_stats_df["F", ub_label][t_min:t_max], color='darkgreen', alpha=range_alpha)

                ax[0].fill_between(t, ensemble_stats_df["F", lb_label][t_min:t_max],
                                   ensemble_stats_df["F", ub_label][t_min:t_max], color='darkgreen', alpha=range_alpha)
                ax[1].fill_between(t, ensemble_stats_df["P", lb_label][t_min:t_max],
                                   ensemble_stats_df["P", ub_label][t_min:t_max], color='lawngreen', alpha=range_alpha)
                ax[2].fill_between(t, ensemble_stats_df["q", lb_label][t_min:t_max],
                                   ensemble_stats_df["q", ub_label][t_min:t_max], color='saddlebrown', alpha=range_alpha)
                ax[3].fill_between(t, ensemble_stats_df["strategy", lb_label][t_min:t_max],
                                   ensemble_stats_df["strategy", ub_label][t_min:t_max], color='k', alpha=range_alpha)
                ax[4].fill_between(t, ensemble_stats_df["cattle_quantity", lb_label][t_min:t_max]/1000000,
                                   ensemble_stats_df["cattle_quantity", ub_label][t_min:t_max]/1000000, color='r', alpha=range_alpha)

        else:
            print("No files for parameters {} found".format(parameter_str))

    axf.set_xlim([t_min, t_max])
    axf.legend(loc='upper right')

    plot_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    annotation_pos = (-0.17, 1.1)
    annotation_offset = (1, -1)
    axis_label_fontsize = 14

    for i, a in enumerate(ax):
        a.set_xlim([t_min, t_max])
        a.legend(loc='upper right')
        a.annotate(plot_labels[i], xy=annotation_pos, xycoords='axes fraction',
                    fontsize=axis_label_fontsize,
                    xytext=annotation_offset, textcoords='offset points',
                    horizontalalignment='left', verticalalignment='top')

    axf.set_ylim([0, 500])
    ax[0].set_ylim([0, 500])
    ax[1].set_ylim([0, 500])
    ax[2].set_ylim([0, 1])
    ax[3].set_ylim([0, 1])

    #ax[3].set_ylim([t_min, t_max])

    axis_label_fontsize = 13
    axf.set_ylabel('Forest [ha]', fontsize=axis_label_fontsize)
    ax[0].set_ylabel('Forest [ha]', fontsize=axis_label_fontsize)
    ax[1].set_ylabel('Pasture [ha]', fontsize=axis_label_fontsize)
    ax[2].set_ylabel('Pasture produc-\ntivity [a.u.]', fontsize=axis_label_fontsize, multialignment='center')
    ax[3].set_ylabel('Intensification', fontsize=axis_label_fontsize)
    ax[4].set_ylabel('Cattle production\n[million heads]', fontsize=axis_label_fontsize, multialignment='center')

    axf.set_xlabel(r'$t$', fontsize=axis_label_fontsize)
    ax[4].set_xlabel(r'$t$', fontsize=axis_label_fontsize)

    fig_file = SAVE_PATH_FIGS + "/sensitivity_" + sensitivity_parameter_key + "_median_percentiles.png"
    fig_file2 = SAVE_PATH_FIGS + "/sensitivity_" + sensitivity_parameter_key + "_median_percentiles_onlyF.png"

    fig.tight_layout()
    fig.savefig(fig_file, dpi=150)

    fig2.tight_layout()
    fig2.savefig(fig_file2, dpi=150)

    plt.clf()

    if verbosity > 0:
        print("saved figure to {}".format(fig_file))


# ===================================================================================================================
# ====================               plot ensemble averages              ============================================
# ===================================================================================================================

plot_single_ensemble = False
plot_all_ensembles = True

chosen_parameters = []

if plot_single_ensemble:
    if verbosity > 0:
        print("plotting single ensemble...")
    chosen_parameters = [[0.05, 1.]]

if plot_all_ensembles:
    if verbosity > 0:
        print("plotting all ensembles...")
    chosen_parameters = PARAM_COMBS

if plot_all_ensembles or plot_single_ensemble:

    for parameters in chosen_parameters:
        # parameter_str = "-".join([str(i) for i in parameters]).replace(".", "o")

        parameter_str = get_pymofa_id(parameters)

        fnames = []
        for file in os.listdir(SAVE_PATH_EXPERIMENTS):
            if file.startswith(parameter_str + "_s") and file.endswith('.pkl'):
                fnames.append(os.path.join(SAVE_PATH_EXPERIMENTS, file))

        if fnames:

            if verbosity > 0:
                print("loading files to calculate ensemble statistics (average over {} runs)".format(len(fnames)))

            if verbosity > 1:
                print("file list: {}".format(', '.join(fnames)))

            percentiles = [5, 95]
            ensemble_stats_df = abacra.ensemble.ensemble_stats_from_csv(fnames, ensemble_stat_measures='all',
                                                                        percentiles=percentiles)

            if verbosity > 1:
                print(ensemble_stats_df.head())

            fig_file = SAVE_PATH_FIGS + "/ensemble_mean_minmax_" + parameter_str + ".png"
            abacra.plotting.plot_ensemble_stats(ensemble_stats_df, fig_file,
                                                ensemble_stat_measure='mean',
                                                individual_run_measure='mean',
                                                bounds='minmax',
                                                plot_strategy=True,
                                                plot_price=True, secveg_dynamics=True,
                                                t_min=0)

            fig_file = SAVE_PATH_FIGS + "/ensemble_median_5-95percentile_" + parameter_str + ".png"
            abacra.plotting.plot_ensemble_stats(ensemble_stats_df, fig_file,
                                                ensemble_stat_measure='median',
                                                individual_run_measure='mean',
                                                bounds='percentiles', percentiles=percentiles,
                                                plot_strategy=True,
                                                plot_price=True, secveg_dynamics=True,
                                                t_min=0)

            if verbosity > 0:
                print("saved figure to {}".format(fig_file))
        else:
            print("No files for parameters {} found".format(parameter_str))

