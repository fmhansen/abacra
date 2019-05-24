# -*- coding: utf-8 -*-
"""
This is the main class for the abacra model
"""

# enable for python2 execution
# from __future__ import print_function, division, absolute_import

import matplotlib.pylab as plt
import networkx as nx
import numpy as np
import pandas as pd
import time
import os
import pickle
import abacra.network_creation

# setting for printout of larger tables
pd.set_option("display.max_columns",200)

# setting all errors in numpy
# np.seterr(all='raise')

# ============================================================================
# =============== base class =================================================
# ============================================================================


class Model(object):
    """
    The base class of the abacra model
    """

    def __init__(self, par_dict=None, par_file="default", verbosity=0,
                 initial_conditions="default", network_type='grid_2d',
                 network_path=None, rng_seed=0):

        print("Initializing model ...")

        self.setup_time = time.process_time()

        self.verbosity = verbosity
        self.regenerate_network = False
        self.rng_seed = "random"
        self.t_max = 0
        self.t = [0]
        self.state_variables = ["P", "q", "k", "F", "S", "v", "y", "I", "C"]
        self.sv_position = {"P": 0, "q": 1, "k": 2, "F": 3, "S": 4, "v": 5, "y": 6, "I": 7, "C": 8}
        self.dim = len(self.state_variables)
        self.no_aux_vars = 3
        self.control_variables = ["d", "a", "r", "l", "m"]
        self.cv_position = {"d": 0, "a": 1, "r": 2, "l": 3, "m": 4}
        self.no_controls = len(self.control_variables)

        self.plot_pars = {"dpi_saving": 150}

        if rng_seed is not None:
            self.rng_seed = rng_seed

        # load parameter file
        if par_file is "default":
            self.load_parfile(os.path.dirname(__file__) + "/../default_parametrized.par")
        elif par_file is not None:
            self.load_parfile(par_file)

        assert self.pars["S_0"] <= 0

        if type(par_dict) is dict:
            self.pars.update(par_dict)
            print("Modified parameters:")
            print(par_dict)

        #  ============   initialize network structure ==================================

        self.G, self.node_pos = self._return_network(network_type=network_type, network_path=network_path)

        if self.verbosity > 1:
            self.print_network_properties()

        if not "pie_radius" in self.plot_pars:
            self.plot_pars["pie_radius"] = 0.3

        self.adj_matrix = nx.adjacency_matrix(self.G, weight=None)
        self.no_agents = self.adj_matrix.shape[0]
        self.pars["network_type"] = network_type

        self.network_type = network_type
        self.network_path = network_path

        # ============== ========================== ======================================
        # ============== setting initial conditions ======================================

        print("Setting initial conditions...")

        self.control_vec = np.zeros(shape=[self.no_agents, self.no_controls])
        #self.state_vec = np.zeros(shape=[self.no_agents, self.dim])
        self.state_vec_new = np.zeros(shape=[self.no_agents, self.dim])

        np.random.seed(self.rng_seed)

        # state_vec = np.zeros(shape=[no_agents, dim])
        # or better in a one-dim array: p_1, p_2, ..., p_n,
        # q_1, ..., q_2, ..., q_n ?

        # randomization of initial soil quality
        if self.pars["randomize_initial_soil_quality"] is "random_uniform":
            initial_qp = np.random.uniform(self.pars["q_0_mean"] - self.pars["q_0_dev"],
                                          self.pars["q_0_mean"] + self.pars["q_0_dev"], self.no_agents)
        else:
            initial_qp = self.pars["q_0_mean"] * np.ones(self.no_agents)

        # randomization of initial savings
        if self.pars["randomize_initial_savings"] is "random_uniform":
            initial_savings = np.random.uniform(self.pars["k_0_mean"] - self.pars["k_0_dev"],
                                                self.pars["k_0_mean"] + self.pars["k_0_dev"], self.no_agents)
        elif self.pars["randomize_initial_savings"] is "random_pareto":
            initial_savings = np.random.pareto(self.pars["k_0_pareto_shape"], size=self.no_agents)
        elif self.pars["randomize_initial_savings"] is "random_lognormal":
            mu = np.log(self.pars["k_0_mean"] / np.sqrt(1 + self.pars["k_0_std"]**2 / (self.pars["k_0_mean"] ** 2)))
            sigma = np.sqrt(np.log(1 + self.pars["k_0_std"]**2 / (self.pars["k_0_mean"] ** 2)))
            initial_savings = np.random.lognormal(mean=mu, sigma=sigma, size=self.no_agents)
        else:
            initial_savings = self.pars["k_0_mean"] * np.ones(self.no_agents)

        initial_qs = np.ones(self.no_agents)

        # initialization:
        if self.pars["absolute_area"]:

            areas_dict = dict(nx.get_node_attributes(self.G, "area"))

            if not areas_dict:
                print("network does not have node attribute 'area'")
                print("using default instead")
                areas = self.pars["default_property_area"] * np.ones(self.no_agents)
            else:
                areas = np.zeros(self.no_agents)
                for i, area in areas_dict.items():
                    if areas_dict[i] >= 0:
                        areas[i] = areas_dict[i]
                    else:
                        print("Warning: Area smaller than 0, using 0 instead.")
                        areas[i] = 0

            self.total_area = sum(areas_dict.values())
            print("total area is {}.".format(self.total_area))
            if verbosity > 1:
                print("areas:")
                print(areas_dict)

            if self.pars["k_0_prop_to_area"]:
                initial_savings = initial_savings * areas

            initial_pasture_dict = dict(nx.get_node_attributes(self.G, self.pars["initial_pasture_key"]))

            if not initial_pasture_dict:
                print("network does not have node attribute {}".format(self.pars["initial_pasture_key"]))

            if not initial_pasture_dict or self.pars["initial_pasture_key"] is None:
                print("using default (relative pasture area = {})".format(self.pars["P_0"]))

                initial_pasture = self.pars["P_0"] * areas * self.pars["available_area_frac"]
                initial_forest = self.pars["F_0"] * areas * self.pars["available_area_frac"]
                initial_secondary_vegetation = self.pars["S_0"] * areas * self.pars["available_area_frac"]

            else:
                print("using initial pasture values from network attributes")
                initial_pasture = np.zeros(self.no_agents)
                initial_forest = np.zeros(self.no_agents)
                initial_secondary_vegetation = np.zeros(self.no_agents)

                negatives_nodes = []
                deviations = []
                for i, area in areas_dict.items():
                    remaining_area = area * self.pars["available_area_frac"] - initial_pasture_dict[i]
                    if remaining_area < 0:
                        negatives_nodes.append(i)
                        if area > 0:
                            deviations.append(remaining_area / (area * self.pars["available_area_frac"]))
                        else:
                            "Warning: area is <= 0"
                            deviations.append("NaN")

                        initial_pasture[i] = area * self.pars["available_area_frac"]
                        initial_forest[i] = 0
                        initial_secondary_vegetation[i] = 0

                    else:
                        initial_pasture[i] = initial_pasture_dict[i]
                        initial_forest[i] = remaining_area * (1.-self.pars["S_0"])
                        initial_secondary_vegetation[i] = remaining_area*self.pars["S_0"]

                if negatives_nodes:
                    if max(deviations) < 0.001 or self.pars["available_area_frac"] < 1:
                        print("Warning: remaining area on {} nodes is negative.".format(len(negatives_nodes)))
                        print("Maximal deviation: {} percent".format(max(deviations)*100))
                        print("Using all area as pasture on these nodes.")
                    else:
                        print("Warning: remaining area on {} nodes is negative.".format(len(negatives_nodes)))
                        print("Deviations bigger than 0.1 percent:\n{}".format(deviations))
                        print("On nodes: {}".format(negatives_nodes))
                        print("Using all area as pasture on these nodes.")

                print("total pasture area is {}".format(sum(initial_pasture)))

                if verbosity > 1:
                    print("initial pasture values:")
                    print(initial_pasture)

        # relative areas
        else:
            initial_pasture = self.pars["P_0"] * np.ones(self.no_agents)
            initial_forest = self.pars["F_0"] * np.ones(self.no_agents)
            initial_secondary_vegetation = self.pars["S_0"] * np.ones(self.no_agents)
            areas = np.ones(self.no_agents)

        # set initial pasture productivity to 0 if there is no pasture
        for i in range(len(initial_qp)):
            if initial_pasture[i] <= 0:
                initial_qp[i] = 0

        state_vec = np.array([initial_pasture, initial_qp, initial_savings,
                              initial_forest, initial_secondary_vegetation, initial_qs]).T

        self.state_vec = np.concatenate((state_vec, np.zeros((self.no_agents, self.no_aux_vars))), axis=1)

        if self.verbosity > 1:
            print("initial state vector")
            print(self.state_vec)

        print(self.no_agents)
        print(self.dim)
        assert self.state_vec.shape == (self.no_agents, self.dim),\
            "shape of state_vec = {} not matching".format(self.state_vec.shape)

        self.interaction_vars = [0, 0]

        if self.pars["initial_strategy"] is "random":
            # initialization of strategies with fraction alpha_0 in strategy 2
            self.strategy_vec = np.zeros(self.no_agents)
            no_strat2 = int(round(self.pars["initial_strategy_prob"] * self.no_agents))
            self.strategy_vec[0:no_strat2] = 1
            # print(sum(strategy))
            # randomize position of different strategies
            np.random.shuffle(self.strategy_vec)

        elif self.pars["initial_strategy"] is "near_city":
            # if property is less than distance x away from a city, it is intensive (with certain probability)

            distance_to_city = np.array(list(nx.get_node_attributes(self.G, "dcity").values()))
            if len(distance_to_city) == 0:
                raise InputError("distance to city attribute in graph input missing")

            self.strategy_vec = np.greater(self.pars["initial_strategy_prob"]
                                           * heaviside(self.pars["initial_strategy_distance_threshold"]
                                                       - distance_to_city),
                                           np.random.rand(self.no_agents)).astype(int)
            # print("'Warning: attribute distance to city not available for initialization of strategies")

        elif self.pars["initial_strategy"] is "near_road":
            # if property is less than distance x away from a city, it is intensive (with certain probability)

            distance_to_road = np.array(list(nx.get_node_attributes(self.G, "droad").values()))
            if len(distance_to_road) == 0:
                raise InputError("distance to road attribute in graph input missing")

            self.strategy_vec = np.greater(self.pars["initial_strategy_prob"]
                                           * heaviside(self.pars["initial_strategy_distance_threshold"]
                                                       - distance_to_road),
                                           np.random.rand(self.no_agents)).astype(int)
            # print("'Warning: attribute distance to city not available for initialization of strategies")

        if self.verbosity > 1:
            print("Initial strategy vector:")
            print(self.strategy_vec)

        # fcattle_to_sell is not used because the mean of q_0 should be used to make functions comparable
        # with different random initial conditions
        # calculate average stocking rate
        average_stocking = ((1. - sum(self.strategy_vec)/self.no_agents) * self.pars["l_ext"]
                            + sum(self.strategy_vec)/self.no_agents * self.pars["l_int"])

        self.pars["initial_cattle_quantity"] = fcattle_to_sell(np.sum(self.state_vec.T[0]),  # pasture area
                                                               self.pars["q_0_mean"],  # pasture productivity
                                                               average_stocking,
                                                               self.pars)

        if verbosity > 0:
            pass
        print("average initial stocking: {}".format(average_stocking))
        print("initial cattle quantity demand curve parameter: {}".format(self.pars["initial_cattle_quantity"]))


        # initialize auxiliary variables:
        for j in range(self.no_agents):
            self.state_vec[j][6] = fcattle_to_sell(self.state_vec[j][0], self.state_vec[j][1],
                                                    fl(self.state_vec[j], self.strategy_vec[j], self.pars), self.pars)  # y

        actual_cattle_quantity = np.sum(self.state_vec.T[6])

        for j in range(self.no_agents):
            # I
            self.state_vec[j][7] = (frevenue(self.state_vec[j][6], fcattle_price(actual_cattle_quantity, self.pars))
                                    - self.pars["man_cost"]
                                    * fl(self.state_vec[j], self.strategy_vec[j], self.pars)
                                    * self.state_vec[j][0])
            # C
            self.state_vec[j][8] = (1 - self.pars["savings_rate"]) * self.state_vec[j][7]

        if "intensification_credit_limit_per_ha" in self.pars:
            self.credit_limit = self.pars["intensification_credit_limit_per_ha"] * areas
        elif "intensification_credit_limit_abs" in self.pars:
            self.credit_limit = self.pars["intensification_credit_limit_abs"] * np.ones(self.no_agents)
        else:
            print("credit limit set to 0")
            self.credit_limit = np.zeros(self.no_agents)

        # Todo: implement other ways of setting the initial strategies

        # variables for post-processing
        self.stats_df = pd.DataFrame()

        # timing
        self.end_setup_time = time.process_time()
        if self.verbosity > 0:
            print("Time for initialization: " + str(time.process_time() - self.setup_time))

# ============== method for loading the parameter file ==============================

    def load_parfile(self, par_file):

        try:
            file = open(par_file)
        except:
            print("Warning: kwarg 'par_file' is not a path to a parameter file nor a file in the module")
            raise IOError

        try:
            txt = file.read()
            self.pars = eval(txt)
            file.close()
            if self.verbosity > 1:
                print("Loaded the following parameters from {}:".format(par_file))
                print(self.pars)
            else:
                print("Used parameters from {}".format(par_file))

        except:
            print("Warning: Parameter file not valid.")
            raise IOError

        return 0

# ============== return a network for initialization ================================

    def _return_network(self, network_type="grid_2d", network_path=None):

        # 2D grid
        if network_type is 'grid_2d':

            if self.verbosity > 0:
                print("Generating 2D grid network ...")

            self.pars["n_x"] = 2
            self.pars["n_y"] = 2
            g = nx.grid_2d_graph(self.pars["n_x"], self.pars["n_y"], periodic=True)

            node_pos = {}
            for index in g.nodes():
                node_pos[index] = index

        # Watts-Strogatz graph
        elif network_type is 'ws':
            if self.verbosity > 0:
                print("Generating Watts-Strogatz network ...")

            N = 100
            no_nearest_neighbors = 4
            rewiring_prob = 0.2
            g = nx.watts_strogatz_graph(N, no_nearest_neighbors, rewiring_prob)
            self.plot_pars["nx_scale"] = g.number_of_nodes() / (2. * np.pi)
            node_pos = nx.circular_layout(g, scale=self.plot_pars["nx_scale"])

            self.plot_pars["pie_radius"] = 0.45

        # random geometric graph
        elif network_type is 'geometric':
            if self.verbosity > 0:
                print("Generating random geometric network ...")

            n_x = 10
            n_y = 5
            radius = 1.5

            node_pos = {i: (np.squeeze([n_x * np.random.random(1),
                        n_y * np.random.random(1)])) for i in range(n_x * n_y)}

            g = nx.random_geometric_graph(n_x * n_y, radius, pos=node_pos, dim=2)
            nx.set_node_attributes(g, "position", node_pos)

        # =====================================================================
        # Generators for geometric graphs.
        # random_geometric_graph(n, radius[, dim, pos])
        #    Return the random geometric graph in the unit cube.
        # geographical_threshold_graph(n, theta[, ...])
        # Return a geographical threshold graph.
        # waxman_graph(n[, alpha, beta, L, domain])
        # Return a Waxman random graph
        # navigable_small_world_graph(n[, p, q, r, ...])
        # Return a navigable small-world graph.
        # =====================================================================

        # graph from gml file
        elif network_type is 'gml':

            # try to read file for 100 times
            g = None
            n = 0
            while g is None and n < 100:
                try:
                    n += 1
                    g = nx.read_gml(network_path)
                except nx.exception.NetworkXError:
                    time.sleep(np.random.uniform(low=0.0001, high=0.1))
                    pass

            g = nx.convert_node_labels_to_integers(g)

            node_pos = dict((key, (nx.get_node_attributes(g, 'lat')[key],
                            nx.get_node_attributes(g, 'long')[key])) for key in nx.nodes(g))

            print("Loaded network from {}".format(network_path))
            print("with {} nodes".format(g.number_of_nodes()))

            if self.verbosity > 1:
                print("Nodes:")
                print(g.nodes())

            self.plot_pars["pie_radius"] = 0.01

        elif network_type is 'pickle':

            if self.verbosity > 0:
                print("Loading network from {}".format(network_path))

            g = nx.read_gpickle(network_path)

            node_pos = dict((key, (nx.get_node_attributes(g, 'lat')[key],
                            nx.get_node_attributes(g, 'long')[key])) for key in nx.nodes(g))

            self.plot_pars["pie_radius"] = 0.01

        # network generation and randomization
        elif network_type is 'from_distance_matrix':

            filename = ""
            network_file = os.path.join(network_path, filename)

            print("Generating network from distance matrix at {}".format(network_path))

            g, node_pos = abacra.network_creation.generate_spatial_network(network_path)

        else:
            print("network_type not given or available")
            raise InputError

        return g, node_pos

    # =================== run the system ====================

    def run(self, *, t_max):

        self.t_max = t_max
        self.t = np.arange(0, t_max)

        self.sv_traj = np.ndarray(shape=[t_max, self.dim * self.no_agents])
        self.cv_traj = np.ndarray(shape=[t_max, self.no_controls * self.no_agents])
        self.strategy_traj = np.ndarray(shape=[t_max, self.no_agents])
        self.interaction_traj = np.ndarray(shape=[t_max, 2])

        self.sv_traj[0] = self.state_vec.flatten()
        self.cv_traj[0] = self.control_vec.flatten()
        self.strategy_traj[0] = self.strategy_vec

        initial_cattle = np.sum(self.state_vec.T[6])

        print("initial cattle (including variation): {}".format(initial_cattle))

        self.interaction_traj[0] = np.array([initial_cattle,
                                             fcattle_price(initial_cattle, self.pars)])

        start_run_time = time.process_time()

        for i in np.arange(1, t_max):
            self.iterate()

            self.sv_traj[i] = self.state_vec.flatten()
            self.cv_traj[i] = self.control_vec.flatten()
            self.strategy_traj[i] = self.strategy_vec
            self.interaction_traj[i] = self.interaction_vars

        if self.verbosity > 0:
            print("Process time since beginning of initialization: "
                  + str(time.process_time() - self.setup_time))
            print("Process time since start of main computation: "
                  + str(time.process_time() - self.end_setup_time))
        return 0

    # iterate one time step
    def iterate(self):
        # ================= single farm dynamics ===========================

        for j in range(self.no_agents):
            # bring control and ecological dynamics together
            # first without savings evolution

            # 1) agents decide on controls
            self.control_vec[j] = np.array(np.append(land_conversion_decisions(self.state_vec[j], self.strategy_vec[j],
                                                                               self.interaction_vars, self.pars),
                                [fl(self.state_vec[j], self.strategy_vec[j], self.pars),
                                 fm(self.state_vec[j], self.strategy_vec[j], self.pars)]))

            assert self.control_vec[j][0] >= 0
            assert self.control_vec[j][1] >= 0
            assert self.control_vec[j][2] >= 0

            # 2) evolution of environmental variables
            self.state_vec_new[j][:6] =\
                np.array([fpasture(self.state_vec[j], self.control_vec[j], self.pars),                            # P
                          fqp(self.state_vec[j], self.control_vec[j], self.pars),                                 # q
                          self.state_vec[j][2],                          # no dynamics                            # k
                          fforest(self.state_vec[j], self.control_vec[j], self.pars),                             # F
                          fsecondary_vegetation(self.state_vec[j], self.control_vec[j], self.pars),               # S
                          fqs(self.state_vec[j], self.control_vec[j], self.pars)])                                # v

            # y
            self.state_vec_new[j][6] = fcattle_to_sell(self.state_vec_new[j][0], self.state_vec_new[j][1],
                                                       self.control_vec[j][3], self.pars)

            assert self.state_vec_new[j][0] >= 0  # P
            assert self.state_vec_new[j][1] >= 0  # q
            # k
            assert self.state_vec_new[j][3] >= 0  # F
            assert self.state_vec_new[j][4] >= 0, "SV:{} at node {}".format(self.state_vec_new[j][4], j)  # S
            assert self.state_vec_new[j][5] >= 0  # v
            assert self.state_vec_new[j][6] >= 0  # y

        sum_cattle_quantity = np.sum(self.state_vec_new.T[6])  # sum cattle quantity y
        self.interaction_vars[0] = sum_cattle_quantity
        self.interaction_vars[1] = fcattle_price(self.interaction_vars[0], self.pars)

        # 3) outcomes for agents
        for j in range(self.no_agents):

            self.state_vec_new[j][7] = (frevenue(fcattle_price(sum_cattle_quantity, self.pars),
                                                 self.state_vec_new[j][6])
                                        - self.pars["man_cost"] * self.control_vec[j][4] * self.state_vec_new[j][0])

            self.state_vec_new[j][2] = fk(self.state_vec_new[j], self.control_vec[j],
                                          self.interaction_vars, self.strategy_vec[j], self.pars)

            self.state_vec_new[j][8] = fconsumption(self.state_vec_new[j][7], self.pars)

        self.state_vec = self.state_vec_new

        # ================ dynamics on network ================================

        # poisson distribution p(k, lam) = lam^k exp(-lam) / k!
        no_events = np.random.poisson(lam=self.pars["imitation_rate"] * self.no_agents)

        # print("events of updating: {}".format(no_events))

        # vector for agent choices
        agent_choice = np.random.randint(0, high=self.no_agents, size=no_events)

        # update strategies

        if self.pars["imitation_setting"] == "savings":

            for j in agent_choice:
                # choose agent neighbor
                if self.G.neighbors(j):  # implicitly checking if the neighbor list is empty
                    neighbor_id = np.random.choice(list(self.G.neighbors(j)))
                    # alternatively choose edge + direction?

                    # calculate relative success (with respect to savings)
                    diff_success = self.state_vec[neighbor_id][2] - self.state_vec[j][2]

                    # do strategy update with prob depending on difference of savings
                    if np.random.random() < 0.5 * (np.tanh(diff_success) + 1.):
                        if self.strategy_vec[neighbor_id] == 1:
                            # only if there are enough savings for intensification
                            if ((self.state_vec[j][2] + self.credit_limit[j] >= self.state_vec[j][0] * self.pars["intensification_cost"])
                                and self.strategy_vec[j] == 0):
                                self.strategy_vec[j] = 1
                                self.state_vec[j][2] -= self.state_vec[j][0] * self.pars["intensification_cost"]
                                self.state_vec[j][1] = 1
                        else:
                            self.strategy_vec[j] = self.strategy_vec[neighbor_id]

        elif self.pars["imitation_setting"] == "consumption_absolute":
            for j in agent_choice:
                # choose agent neighbor
                if self.G.neighbors(j):  # implicitly checking if the neighbor list is empty
                    neighbor_id = np.random.choice(list(self.G.neighbors(j)))
                    # alternatively choose edge + direction?

                    # calculate relative success (with respect to income)
                    own_income = frevenue(fcattle_price(sum_cattle_quantity, self.pars),
                                          fcattle_to_sell(self.state_vec[j][0],
                                                          self.state_vec[j][1],
                                                          self.control_vec[j][3],
                                                          self.pars))
                    own_consumption = fconsumption(own_income, self.pars)
                    neighbor_income = frevenue(fcattle_price(sum_cattle_quantity, self.pars),
                                          fcattle_to_sell(self.state_vec[neighbor_id][0],
                                                          self.state_vec[neighbor_id][1],
                                                          self.control_vec[neighbor_id][3],
                                                          self.pars))
                    neighbor_consumption = fconsumption(neighbor_income, self.pars)
                    diff_success = neighbor_consumption - own_consumption

                    # do strategy update with prob depending on difference of consumption
                    if np.random.random() < 0.5 * (np.tanh(diff_success) + 1.):
                        if self.strategy_vec[neighbor_id] == 1:
                            # only if there are enough savings for intensification
                            if ((self.state_vec[j][2] + self.credit_limit[j] >= self.state_vec[j][0] * self.pars["intensification_cost"])
                                and self.strategy_vec[j] == 0):
                                self.strategy_vec[j] = 1
                                self.state_vec[j][2] -= self.state_vec[j][0] * self.pars["intensification_cost"]
                                self.state_vec[j][1] = 1
                        else:
                            self.strategy_vec[j] = self.strategy_vec[neighbor_id]

        elif self.pars["imitation_setting"] == "consumption_relative":
            for j in agent_choice:
                # choose agent neighbor
                if self.G.neighbors(j):  # implicitly checking if the neighbor list is empty
                    neighbor_id = np.random.choice(list(self.G.neighbors(j)))
                    # alternatively choose edge + direction?

                    # calculate relative success (with respect to income)
                    own_income = frevenue(fcattle_price(sum_cattle_quantity, self.pars),
                                          fcattle_to_sell(self.state_vec[j][0],
                                                          self.state_vec[j][1],
                                                          self.control_vec[j][3],
                                                          self.pars))
                    own_consumption = fconsumption(own_income, self.pars)
                    neighbor_income = frevenue(fcattle_price(sum_cattle_quantity, self.pars),
                                          fcattle_to_sell(self.state_vec[neighbor_id][0],
                                                          self.state_vec[neighbor_id][1],
                                                          self.control_vec[neighbor_id][3],
                                                          self.pars))
                    neighbor_consumption = fconsumption(neighbor_income, self.pars)
                    rel_success = (neighbor_consumption - own_consumption)/(neighbor_consumption + own_consumption)

                    # do strategy update with prob depending on difference of consumption
                    if np.random.random() < 0.5 * (rel_success + 1.):
                        if self.strategy_vec[neighbor_id] == 1:
                            # only if there are enough savings for intensification
                            if ((self.state_vec[j][2] + self.credit_limit[j] >= self.state_vec[j][0] * self.pars["intensification_cost"])
                                and self.strategy_vec[j] == 0):
                                self.strategy_vec[j] = 1
                                self.state_vec[j][2] -= self.state_vec[j][0] * self.pars["intensification_cost"]
                                self.state_vec[j][1] = 1
                        else:
                            self.strategy_vec[j] = self.strategy_vec[neighbor_id]


        else:
            print("No imitation enabled.")

        return 0

    # ============================= check input and output ====================

    def check_output(self):

        np.set_printoptions(threshold=10000)
        print(self.strategy_traj)
        print(self.interaction_traj)
        return 0

    def print_network_properties(self, shorten=True):
        print(nx.info(self.G))
        print("Network nodes:")
        if shorten:
            print(self.G.nodes())
        else:
            print(self.G.nodes(1))
        return 0

    def compute_stats(self, stats, percentiles=None, recalc=False, weighted_qv=True):

        # check if stats have already been computed:
        if (not self.stats_df.empty) and (not recalc):
            print("stats_df has already been computed")
            return self.stats_df

        # initialize data frame

        sv_traj = self.sv_traj.T
        cv_traj = self.cv_traj.T

        other_variables = ["cattle_price", "strategy", "cattle_quantity", "active_ranches"]

        if stats is "all":
            stats_to_compute = ["mean", "std", "median", "min", "max",
                                "percentiles", "gini"]
            if percentiles is None:
                percentiles = [25, 75]
        else:
            stats_to_compute = stats
            if "percentiles" in stats and percentiles is None:
                percentiles = [25, 75]

        stats_names = list(stats_to_compute)

        if "percentiles" in stats_to_compute:
            stats_names.remove("percentiles")
            for p in percentiles:
                stats_names.append("perc" + str(p))

        col_index = pd.MultiIndex.from_product([self.state_variables + self.control_variables
                                                + other_variables, stats_names], names=['variables', 'statistics'])
        stats_df = pd.DataFrame(data=None, index=self.t, columns=col_index)

        if self.verbosity > 1:
            print("Using the following MultiIndex:")
            print(col_index)

        def measure_fct(x, *, stat_name, percentile_value=None, avg_weights=None):
            if stat_name == "mean":
                return np.average(x, axis=0, weights=avg_weights)
            if stat_name == "std":
                return np.std(x, axis=0)
            if stat_name == "median":
                return np.median(x, axis=0)
            if stat_name == "min":
                return np.min(x, axis=0)
            if stat_name == 'max':
                return np.max(x, axis=0)
            if stat_name == "percentile":
                assert type(percentile_value) in [float, int]
                return np.percentile(x, percentile_value, axis=0)
            if stat_name == "gini":
                return gini_array(x, axis=0)

        for stat in stats_to_compute:
            for sv in self.state_variables:

                if self.verbosity > 1:
                    print("computing {} for {} ...".format(stat, sv))

                if sv is "q" and weighted_qv:
                    weights = sv_traj[self.sv_position["P"]::self.dim] + 1e-9
                elif sv is "v" and weighted_qv:
                    weights = sv_traj[self.sv_position["S"]::self.dim] + 1e-9
                else:
                    weights = None

                if stat is "percentiles":
                    for percentile in percentiles:
                        stats_df[sv, "perc"+str(percentile)] = \
                            measure_fct(sv_traj[self.sv_position[sv]::self.dim],
                                        stat_name="percentile",
                                        percentile_value=percentile,
                                        avg_weights=weights)

                else:
                    stats_df[sv, stat] = \
                        measure_fct(sv_traj[self.sv_position[sv]::self.dim], stat_name=stat, avg_weights=weights)

        for stat in stats_to_compute:
            for cv in self.control_variables:

                if self.verbosity > 1:
                    print("computing {} for {} ...".format(stat, cv))

                if stat is "percentiles":
                    for percentile in percentiles:
                        stats_df[cv, "perc"+str(percentile)] = \
                            measure_fct(cv_traj[self.cv_position[cv]::self.no_controls],
                                        stat_name="percentile",
                                        percentile_value=percentile)

                else:
                    stats_df[cv, stat] = \
                        measure_fct(cv_traj[self.cv_position[cv]::self.no_controls], stat_name=stat)

        stats_df["strategy", "mean"] = np.mean(self.strategy_traj, axis=1)
        stats_df["cattle_price", "mean"] = self.interaction_traj.T[1]
        stats_df["cattle_quantity", "mean"] = self.interaction_traj.T[0]

        not_active = np.logical_and(sv_traj[::self.dim]/(sv_traj[::self.dim] + sv_traj[3::self.dim]
                                    + sv_traj[4::self.dim]) < self.pars["A_int"], sv_traj[1::self.dim] < 0.05)

        stats_df["active_ranches", "mean"] = 1 - np.mean(not_active, axis=0)

        if self.verbosity > 1:
            print("Agents not active:")
            print(np.mean(not_active, axis=0))
            print(not_active.shape)

        if self.verbosity > 1:
            print(stats_df)

        self.stats_df = stats_df

        return stats_df

    # ========== save and load data as csv ====================================

    def save_trajectory(self, path):
        print("writing trajectories to csv file {}".format(path))
        file = open(path, 'w')

        file.write(self.settings_string())

        col_index = pd.MultiIndex.from_product([range(self.no_agents), self.state_variables],
                                               names=['agent_number','variable'])

        pd.DataFrame(self.sv_traj, index=self.t, columns=col_index).to_csv(file, sep=',')
        file.close()
        return 0

    def save_stats(self, path, *, stats):
        print("writing stats to csv file {}".format(path))
        file = open(path, 'w')
        file.write(self.settings_string())

        stats_df = self.compute_stats(stats, weighted_qv=True)
        stats_df.to_csv(file, sep=',')
        file.close()

        if self.verbosity > 1:
            print("Saved the following table:")
            print(stats_df)

        return 0

    def settings_string(self):
        settings_str =\
                "# trajectory of acabra model with the following settings\n" \
                + str(self.pars) + "\n" \
                + "network: " + str(self.pars["network_type"]) \
                + ", path: " + str(self.network_path) \
                + ", rng seed: " + str(self.rng_seed) + "\n"
        return settings_str

    def load_trajectory(self, path):
        try:
            file = open(path, 'r')
        except IOError:
            print("Wasn't able to open file.")
            return 1
        file.readline()  # reading the first line
        pars = eval(file.readline())  # parameter settings in second line
        file.close()

        if self.verbosity > 0:
            print("Loaded options:")
            print(pars)  # parameter settings in second line

        df = pd.read_csv(path, sep=',', header=[3, 4], index_col=0)
        return df, pars

    def load_stats(self, path):
        df, pars = load_stats_csv(path)
        self.pars = pars
        return df

    # ========== save and load data as pickle =================================

    def pickle(self, path):
        file = open(path, 'wb')
        print("dumping model settings and calculations to {}".format(path))
        pickle.dump(self, file)
        file.close()
        return 0

    def unpickle(self, filepath):
        file = open(filepath, 'rb')
        print("loading model settings and calculations from pickle file {}".format(filepath))
        self = pickle.load(file)
        file.close()
        return self

    # =========================================================================
    # ============= plotting ==================================================
    # =========================================================================

    # ============= plotting routine single ranch =============================

    def plot_single_agent(self, path, **kwargs):
        import abacra.plotting
        abacra.plotting.plot_single_agent(self, path, **kwargs)
        return 0

    # ============= plotting routine aggregate ================================

    def plot_stat(
            self,
            path,
            percentiles=None,
            measure='mean',
            **kwargs):

        import abacra.plotting

        if percentiles is None:
            percentiles = [25, 75]

        if self.verbosity > 0:
            print("Plotting " + measure + " ...")
        if self.verbosity > 1:
            print("kwargs:" + str(kwargs))
            print("percentiles: " + str(percentiles))

        stats_df = self.compute_stats(stats="all", percentiles=percentiles, weighted_qv=True)

        abacra.plotting.plot_trajectory_stat(stats_df, path, percentiles=percentiles,
                             dpi=self.plot_pars["dpi_saving"], measure=measure, **kwargs)
        return 0

    def plot_sample_of_trajectories(self, path, agent_ids=range(100)):
        import abacra.plotting
        abacra.plotting.sample_of_trajectories(self, path, agent_ids=agent_ids)
        return 0

    def print_settings_to_file(self, filename):
        f = open(filename, 'w')

        for k, v in self.pars.items():
            f.write("{}: {}\n".format(k, v))

        f.close()

        return 0


# =========== set up auxiliary functions =================

# auxiliary heaviside function
def heaviside(x):
    return np.piecewise(x, [x < 0, x >= 0], [0, 1])


# ================ functions for strategies ======================
#  0: extensive, 1: intensive

# state variables P = sv[0], q = sv[1], k = sv[2], F = sv[3], S = sv[4]
# control variables cv[0] = d, cv[1] = a, cv[2] = r, cv[3] = l, cv[4] = m 

# deforestation: old implementation
def fd(sv, strategy, interaction_vars, pars):
    if strategy == 0:
        def_amount = pars["D_ext"] * (sv[0] + sv[3] + sv[4])
        cattle_price_old = interaction_vars[1]
        income_increase = frevenue(cattle_price_old, fcattle_to_sell(def_amount, pars["q_d"],
                                                                     fl(sv, strategy, pars), pars))
        return (def_amount  # amount of deforestation
                # * heaviside(pars["q_thrD"] - sv[1])  # only deforestation, if land quality is below threshold
                * heaviside(sv[3] - def_amount)      # only deforestation, if there is enough forest left
                * heaviside(sv[2] - def_amount * pars["def_cost"])  # only deforestation, if there are enough savings
                * heaviside(pars["income_investment_comp_factor_ext"] * income_increase - def_amount * pars["def_cost"])
                # only deforestation if the expected increase in income is x times bigger than investment cost
                * heaviside(pars["P_max_ext"] * (sv[0] + sv[3] + sv[4]) - sv[0])
                )

    if strategy == 1:
        def_amount = pars["D_int"] * (sv[0] + sv[3] + sv[4])
        cattle_price_old = interaction_vars[1]
        income_increase = (frevenue(cattle_price_old, fcattle_to_sell(def_amount, pars["q_d"],
                                                                      fl(sv, strategy, pars), pars))
                           - pars["man_cost"] * def_amount * pars["m"])
        costs = def_amount * (pars["def_cost"] + pars["intensification_cost"])
        return (def_amount  # amount of deforestation
                * heaviside(sv[2] - costs)       # only deforestation if there are enough savings
                * heaviside(sv[3] - def_amount)  # only deforestation if there is enough forest left
                # * heaviside(def_amount - sv[4])  # only deforestation if secondary vegetation is already converted
                * heaviside(pars["income_investment_comp_factor_int"] * income_increase - costs)
                # only deforestation if the expected increase in income is x times bigger than investment cost
                )


# reuse: old implementation
def fr(sv, strategy, interaction_vars, pars):
    if strategy == 0:
        reuse_amount = pars["R_ext"] * (sv[0] + sv[3] + sv[4])
        cattle_price_old = interaction_vars[1]
        income_increase = frevenue(cattle_price_old,
                                   fcattle_to_sell(reuse_amount, sv[5], fl(sv, strategy, pars), pars))
        return (reuse_amount
                * heaviside(sv[2] - reuse_amount * pars["reuse_cost"])  # only reuse, if there are enough savings
                # * heaviside(pars["q_thrR"] - sv[1])  # reuse if soil quality is below threshold
                * heaviside(pars["income_investment_comp_factor_ext"] * income_increase - reuse_amount * pars["reuse_cost"])
                * heaviside(reuse_amount - sv[3])  # reuse if too little forest left
                * heaviside(sv[4] - reuse_amount)  # reuse if there is enough secondary vegetation
                * heaviside(pars["P_max_ext"] * (sv[0] + sv[3] + sv[4]) - sv[0])
                )

    if strategy == 1:
        reuse_amount = pars["R_int"] * (sv[0] + sv[3] + sv[4])

        cattle_price_old = interaction_vars[1]
        income_increase = (frevenue(cattle_price_old,
                                    fcattle_to_sell(reuse_amount, sv[5], fl(sv, strategy, pars), pars))
                           - pars["man_cost"] * reuse_amount * pars["m"])
        costs = reuse_amount * (pars["reuse_cost"] + pars["intensification_cost"])
        return (reuse_amount
                * heaviside(sv[2] - costs)  # only reuse, if there are enough savings
                * heaviside(pars["income_investment_comp_factor_int"] * income_increase - costs)
                # only reuse if the expected increase in income is x times bigger than investment cost
                * heaviside(sv[4] - reuse_amount)  # reuse if there is enough secondary vegetation
                )


# abandonment: old implementation
def fa(sv, strategy, interaction_vars, pars):
    if strategy == 0:
        abd_amount = pars["A_ext"] * (sv[0] + sv[3] + sv[4])
        return (abd_amount
                * heaviside(sv[0] - abd_amount)   # abandon if there is enough pasture
                * heaviside(pars["q_thrA"] - sv[1])  # and the pasture productivity is low
                )
    if strategy == 1:
        abd_amount = pars["A_int"] * (sv[0] + sv[3] + sv[4])

        cattle_price_old = interaction_vars[1]
        exp_income = (frevenue(cattle_price_old, fcattle_to_sell(sv[0], sv[1], fl(sv, strategy, pars), pars))
                      - pars["man_cost"] * sv[0] * pars["m"])
        return (abd_amount
                * heaviside((-1.)*exp_income)      # abandon if expected income is negative
                * heaviside(sv[0] - abd_amount)    # abandon if there is enough pasture
                )


def land_conversion_decisions_old(sv, strategy, interaction_vars, pars):
    d = fd(sv, strategy, interaction_vars, pars)
    a = fa(sv, strategy, interaction_vars, pars)
    r = fr(sv, strategy, interaction_vars, pars)
    return [d, a, r]


def land_conversion_decisions(sv, strategy, interaction_vars, pars):

    cattle_price_old = interaction_vars[1]

    if strategy == 0:
        def_amount = pars["D_ext"] * (sv[0] + sv[3] + sv[4])
        income_increase_d = frevenue(cattle_price_old, fcattle_to_sell(def_amount, pars["q_d"],
                                                                     fl(sv, strategy, pars), pars))
        d = (def_amount  # amount of deforestation
             # * heaviside(pars["q_thrD"] - sv[1])  # only deforestation, if land quality is below threshold
             * heaviside(sv[3] - def_amount)  # only deforestation, if there is enough forest left
             * heaviside(sv[2] - def_amount * pars["def_cost"])  # only deforestation, if there are enough savings
             * heaviside(pars["income_investment_comp_factor_ext"] * income_increase_d - def_amount * pars["def_cost"])
             # only deforestation if the expected increase in income is x times bigger than investment cost
             * heaviside(pars["P_max_ext"] * (sv[0] + sv[3] + sv[4]) - sv[0])
             )

        abd_amount = pars["A_ext"] * (sv[0] + sv[3] + sv[4])
        a = (abd_amount
                * heaviside(sv[0] - abd_amount)  # abandon if there is enough pasture
                * heaviside(pars["q_thrA"] - sv[1])  # and the pasture productivity is low
                )

        if abd_amount > sv[0] and sv[1] < pars["q_thrA"]:
            a = sv[0]

        reuse_amount = pars["R_ext"] * (sv[0] + sv[3] + sv[4])
        income_increase_r = frevenue(cattle_price_old,
                                   fcattle_to_sell(reuse_amount, sv[5], fl(sv, strategy, pars), pars))
        r = (reuse_amount
             * heaviside(sv[2] - reuse_amount * pars["reuse_cost"])  # only reuse, if there are enough savings
             # * heaviside(pars["q_thrR"] - sv[1])  # reuse if soil quality is below threshold
             * heaviside(pars["income_investment_comp_factor_ext"] * income_increase_r - reuse_amount * pars["reuse_cost"])
             * heaviside(reuse_amount - sv[3])  # reuse if too little forest left
             * heaviside(sv[4] - reuse_amount)  # reuse if there is enough secondary vegetation
             * heaviside(pars["P_max_ext"] * (sv[0] + sv[3] + sv[4]) - sv[0])
             )

        if r > 0 and d > 0:
            # print("ruling out simultaneous reuse and deforestation")
            if income_increase_r > income_increase_d:
                d = 0
            else:
                r = 0

    else:
        def_amount = pars["D_int"] * (sv[0] + sv[3] + sv[4])
        income_increase_d = (frevenue(cattle_price_old, fcattle_to_sell(def_amount, pars["q_d"],
                                                                      fl(sv, strategy, pars), pars))
                           - pars["man_cost"] * def_amount * pars["m"])
        costs = def_amount * (pars["def_cost"] + pars["intensification_cost"])
        d = (def_amount  # amount of deforestation
             * heaviside(sv[2] - costs)  # only deforestation if there are enough savings
             * heaviside(sv[3] - def_amount)  # only deforestation if there is enough forest left
             * heaviside(pars["income_investment_comp_factor_int"] * income_increase_d - costs)
             # only deforestation if the expected increase in income is x times bigger than investment cost
             )

        abd_amount = pars["A_int"] * (sv[0] + sv[3] + sv[4])
        exp_income = (frevenue(cattle_price_old, fcattle_to_sell(sv[0], sv[1], fl(sv, strategy, pars), pars))
                      - pars["man_cost"] * sv[0] * pars["m"])
        a = (abd_amount
             * heaviside((-1.) * exp_income)  # abandon if expected income is negative
             * heaviside(sv[0] - abd_amount)  # abandon if there is enough pasture
             )

        if abd_amount > sv[0] and exp_income <= 0:
            a = sv[0]

        reuse_amount = pars["R_int"] * (sv[0] + sv[3] + sv[4])
        income_increase_r = (frevenue(cattle_price_old,
                                    fcattle_to_sell(reuse_amount, sv[5], fl(sv, strategy, pars), pars))
                           - pars["man_cost"] * reuse_amount * pars["m"])
        costs = reuse_amount * (pars["reuse_cost"] + pars["intensification_cost"])
        r = (reuse_amount
             * heaviside(sv[2] - costs)  # only reuse, if there are enough savings
             * heaviside(pars["income_investment_comp_factor_int"] * income_increase_r - costs)
             # only reuse if the expected increase in income is x times bigger than investment cost
             * heaviside(sv[4] - reuse_amount)  # reuse if there is enough secondary vegetation
             )

        if r > 0 and d > 0:
            # print("ruling out simultaneous reuse and deforestation")
            if income_increase_r > income_increase_d:
                d = 0
            else:
                r = 0

    return [d, a, r]


# stocking_density
def fl(sv, strategy, pars):
    if strategy == 0:
        return pars["l_ext"]
    if strategy == 1:
        return pars["l_int"]


# management
def fm(sv, strategy, pars):
    if strategy == 0:
        return 0.  # no management
    if strategy == 1:
        return (pars["m"]
                # * heaviside(pars["q_thrm"] - sv[1])  # do management if soil quality is below threshold
                # * heaviside(sv[2] - pars["m"] * sv[0] * pars["man_cost"])  # only do management if sufficient capital is available
                )


def fconsumption(income, pars):
    # note: could also depend on savings
    return (1 - pars["savings_rate"]) * income + pars["const_consumption"]


# =========== land succession dynamics ==================================
# state variables P = sv[0], q = sv[1], k = sv[2], F = sv[3], S = sv[4], v = sv[5]
# control variables cv[0] = d, cv[1] = a, cv[2] = r, cv[3] = l, cv[4] = m 

# dynamics of pasture area
def fpasture(sv, cv, pars):
    return sv[0] + (cv[0] - cv[1] + cv[2])


# dynamics of forest
def fforest(sv, cv, pars):
    return sv[3] - cv[0] + pars["full_regeneration_rate"] * sv[5] * (sv[4] - cv[2] + cv[1])


# dynamics of secondary vegetation
def fsecondary_vegetation(sv, cv, pars):
    return (1 - pars["full_regeneration_rate"] * sv[5]) * (sv[4] - cv[2] + cv[1])


# ============ dynamics of pasture productivity ==========================

def fqp(sv, cv, pars):
    # for P = 0 problem with definition, therefore testing
    if abs(sv[0] + cv[0] - cv[1] + cv[2]) <= 1e-9:
        return 0
    else:
        return ((sv[1] - (pars["beta"] * (cv[3] - cv[4]) * sv[1])
                 + pars["pasture_recovery_rate"] * (sv[1] - pars["q_max_natural"]))
                * (sv[0] - cv[1]) + pars["q_d"] * cv[0] + sv[5] * cv[2]) / (sv[0] + cv[0] - cv[1] + cv[2])

    # OLD IMPLEMENTATION, CHANGED 16/02/2018
    # return ((sv[1] - (pars["beta"] * cv[3] * sv[1]) + cv[4] *
    #         (pars["q_m"] - sv[1]) + pars["gamma"] * (sv[1] - pars["q_n"]))
    #        * (sv[0] - cv[1]) + pars["q_d"] * cv[0] + pars["q_r"] * cv[2]) / (sv[0] + cv[0] - cv[1] + cv[2])


# ============ dynamics of secondary vegetation productivity ==============

def fqs(sv, cv, pars):
    # for S = 0 problem with definition, therefore testing
    if abs(sv[4] + cv[1] - cv[2]) <= 1e-9:
        return 1
    else:
        # print("v: {}, r_S: {}, S: {}, r: {}, a: {}, q: {}".format(sv[5], pars["r_S"], sv[4], cv[2], cv[1], sv[1]))
        #        (v     + r_S         +  (1 - v) )    * ( S    - r)     + a     * q
        new_value = (((sv[5] + pars["r_S"] * (1. - sv[5])) * (sv[4] - cv[2]) + cv[1] * sv[1])
                     / (sv[4] - cv[2] + cv[1]))
        # print(new_value)
        assert new_value <= 1
        return new_value


# =========== dynamics of savings/technological resources ================

def fk(sv, cv, interaction_vars, strategy, pars):

    interest = (heaviside(sv[2]) * sv[2] * pars["interest_rate_on_savings"]
               + heaviside((-1) *sv[2]) * sv[2] * pars["interest_rate_on_credit"])

    revenue = frevenue(fcattle_price(interaction_vars[0], pars), fcattle_to_sell(sv[0], sv[1], cv[3], pars))

    #                               c_m *  m_t    * P_t
    income = revenue - pars["man_cost"] * cv[4] * sv[0] + interest

    # print("income: {}, state_vec[7]: {}".format(income, sv[7]))

    if strategy == 0:
        return (sv[2] + income - fconsumption(income, pars) - pars["def_cost"] * cv[0]
                - pars["reuse_cost"] * cv[2])
    if strategy == 1:
        return (sv[2] + income - fconsumption(income, pars) - (pars["def_cost"] + pars["intensification_cost"]) * cv[0]
                - (pars["reuse_cost"] + pars["intensification_cost"]) * cv[2])


# cattle demand function determining the price of cattle
# given a supplied number of heads
def fcattle_price(cattle_quantity, pars):
    # assert pars["initial_cattle_quantity"] > 0
    # assert pars["price_elasticity_of_demand"] > 0

    if pars["price_feedback"]:
        return pars["initial_cattle_price"] * \
               np.power(cattle_quantity / pars["initial_cattle_quantity"], (-1./pars["price_elasticity_of_demand"]))
    else:
        return pars["fixed_cattle_price"] * np.ones(len(cattle_quantity))


# calculate revenue
def frevenue(cattle_price, cattle_quantity):
    return cattle_price * cattle_quantity


# calculate cattle quantity to sell
def fcattle_to_sell(pasture_area, pasture_productivity, stocking_rate, pars):
    return pasture_area * pasture_productivity * stocking_rate / pars["years_on_pasture"] # P * q * l / yrs

# ============================================================================

#  loading data from files


def load_stats_csv(path, verbosity=0):
    try:
        file = open(path, 'r')
    except IOError:
        print("Wasn't able to open file.")
        return 1
    file.readline() # reading the first line
    pars = eval(file.readline()) # parameter settings in second line
    file.close()

    if verbosity > 0:
        print("Loaded options:")
        print(pars) # parameter settings in second line

    df = pd.read_csv(path, sep=',', header=[3,4], index_col=0)
    return df, pars

# =============================================================================
# calculate gini coefficient


def gini(x):
    # from https://pysal.readthedocs.io/en/latest/_modules/pysal/inequality/gini.html#Gini
    # see also: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    n = len(x)
    try:
        x_sum = x.sum()
    except AttributeError:
        x = np.asarray(x)
        x_sum = x.sum()
    n_x_sum = n * x_sum
    r_x = (2. * np.arange(1, len(x)+1) * x[np.argsort(x)]).sum()
    return (r_x - n_x_sum - x_sum) / n_x_sum


def gini_array(x, axis=0):
    try:
        n = x.shape[axis]
    except AttributeError:
        x = np.asarray(x)
        n = x.shape[axis]
    m = x.shape[(axis + 1) % 2]
    x_sum = x.sum(axis=axis)
    n_x_sum = n * x_sum
    r_x = np.zeros(m)
    for i in range(m):
        if axis == 1:
            y = x[i]
        else:
            y = x.T[i]
        r_x[i] = (2. * np.arange(1, n + 1) * y[np.argsort(y)]).sum()

        if (r_x[i] - n_x_sum[i] - x_sum[i]) == 0 and n_x_sum[i] == 0:
            n_x_sum[i] = 1

    return (r_x - n_x_sum - x_sum) / n_x_sum


class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        msg  -- explanation of the error
    """

    def __init__(self, msg):
        self.msg = msg

