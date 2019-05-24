"""
network creation functions for the abacra model
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pylab as plt
import os
import shapefile as shp


# main function for generating a spatial network from a list of neighborhood data#
# or a distance matrix
def generate_spatial_network(neighborhood_table_path, attribute_table_path, centroid_shp_filepath,
                             distance_matrix_file=None, verbosity=1, source_type="qgis_neartable",
                             save_filename=None, frac_rewire=0.1, compute_measures=False,
                             save_centrality_histograms=False,
                             rewiring_type="rewiring1", rng_seed=0,
                             network_model="uniform", max_distance=5000, spatial_decay_range=5000):

    if save_filename:
        file_path = os.path.dirname(save_filename)
        if not os.path.isdir(file_path):
            try:
                os.makedirs(file_path)
            except FileExistsError:
                pass

    if not distance_matrix_file:
        distance_matrix_file = os.path.splitext(neighborhood_table_path)[0] + "_distance_matrix.pkl"

    coord_dict = coord_dict_from_shpfile(centroid_shp_filepath, polygon_code_field_name="COD_IMOVEL")
    no_nodes = len(coord_dict)

    # if the file is already generated, use it
    if os.path.isfile(distance_matrix_file):
        print("loading pickle file instead of neighborhood table from " + distance_matrix_file)
        # distance_matrix = pd.read_hdf(distance_matrix_file)
        distance_matrix = pd.read_pickle(distance_matrix_file)

    # if not, generate it and use the obtained distance matrix
    else:
        distance_matrix = distance_matrix_from_neighborhood_table(neighborhood_table_path, no_nodes=no_nodes,
                                                                  source_type=source_type)

        # distance_matrix.to_hdf(distance_matrix_file, 'distance_matrix_from_neighborhood_table')
        distance_matrix.to_pickle(distance_matrix_file)

        print("wrote data of distance matrix to " + distance_matrix_file)

    if verbosity > 1:
        print("distance matrix:")
        print(distance_matrix.head())

    # generate network from distance matrix using a specific network model
    g = network_from_distance_matrix(distance_matrix, network_model=network_model,
                                     max_distance=max_distance, spatial_decay_range=spatial_decay_range,
                                     rng_seed=rng_seed)

    if frac_rewire > 0.:
        # rewire links (adding teleconnections)
        g = random_rewiring(g, rewiring_type=rewiring_type, frac_rewire=frac_rewire, rng_seed=(rng_seed+1))

    add_attributes(g, attribute_table_path, coord_dict, verbosity=verbosity)

    if compute_measures:
        if save_filename:
            output_file = os.path.splitext(save_filename)[0] + "_network_measures.txt"
        else:
            output_file = None
        # quantify measures of the obtained networks
        # with verbosity > 0, all measures are printed to the std output
        compute_network_measures(g, verbosity=verbosity, all_global_measures=False, centrality_measures=False,
                                 write_output_file=output_file)

    if save_filename:
        # write network data to file

        nx.write_gml(g, os.path.splitext(save_filename)[0] + '.gml')
        # nx.write_gpickle(g, os.path.splitext(save_filename)[0] + '.pkl')

    if save_centrality_histograms and save_filename:
        plot_network_centrality_histograms(g, os.path.splitext(save_filename)[0] + "_centrality_histograms_lin.pdf")
        plot_network_centrality_histograms(g, os.path.splitext(save_filename)[0] + "_centrality_histograms_log.pdf", log=True)

    print("Network successfully created.")

    return g


# =============================================================================================

def distance_matrix_from_neighborhood_table(neighborhood_table_path, no_nodes=None, source_type="qgis_neartable"):
    # for extracting neighborhood information from near table generated
    # with ArcMap Proximity -> Generate with Near Table tool or
    # equivalent QGIS tool

    adj_df = pd.read_csv(neighborhood_table_path)

    counter = 0
    distance_matrix_df = pd.DataFrame(index=range(no_nodes), columns=range(no_nodes))

    if source_type is "arcgis_neartable":
        for row_index, row in adj_df.iterrows():
            # print row
            i = int(row['IN_FID'])
            j = int(row['NEAR_FID'])
            distance_matrix_df.at[i, j] = float(row['NEAR_DIST'])
            counter += 1

    if source_type is "qgis_neartable":
        for row_index, row in adj_df.iterrows():
            # print row
            i = int(row['InputID'])
            j = int(row['TargetID'])
            distance_matrix_df.at[i, j] = float(row['Distance'])
            counter += 1

    # for double checking
    links = counter / 2
    print("From table file, generated distance matrix with " + str(links) + " links")
    fully_connected = no_nodes * (no_nodes - 1) / 2
    print("(corresponding to a density of " + str(links / fully_connected))

    distance_matrix_df.fillna(1e31, inplace=True)

    return distance_matrix_df

# =============================================================================================


# apply network model to adjacency matrix
def network_from_distance_matrix(distance_matrix, network_model="uniform", max_distance=10000,
                                 spatial_decay_range=10000, rng_seed=0, verbosity=1):
    # choose average link density or prob. scale
    linkdensity = 50
    scale = 1

    if type(distance_matrix) is pd.DataFrame:
        distance_matrix = distance_matrix.as_matrix()

    # choose probability model
    if network_model is "uniform":

        def p_of_d(d):
            return scale * heaviside(max_distance - d)

    elif network_model is "waxman":
        # proble: large d produce underflow

        def p_of_d(d):
            return scale * np.exp((-1) * d / spatial_decay_range)
    else:
        print("Network model not chosen or available")
        raise KeyError

    np.random.seed(rng_seed)

    dim = distance_matrix.shape[0]
    adj_matrix = np.greater(p_of_d(distance_matrix), np.random.rand(dim, dim)).astype(int)
    no_polygons = distance_matrix.shape[0]
    adj_matrix_df = pd.DataFrame(np.triu(adj_matrix), index=range(no_polygons), columns=range(no_polygons))

    if verbosity > 0:
        # check minimal and maximal values
        print("maximal value is " + str(adj_matrix_df.max().max()))
        print("minimal value is " + str(adj_matrix_df.min().min()))
        # check sum of matrix
        print("sum of the matrix is " + str(adj_matrix_df.sum().sum()))

    # generate networkx object from adjacency matrix
    adj_matrix = adj_matrix_df.as_matrix()

    adj_matrix = np.triu(adj_matrix)

    g = nx.from_numpy_matrix(adj_matrix, create_using=nx.Graph())

    return g


# =============================================================================================

def random_rewiring(G, rewiring_type="rewiring1", frac_rewire=0.1, rng_seed=0):

    G_new = G.copy()
    no_links = G.number_of_edges()
    no_nodes = G.number_of_nodes()

    np.random.seed(rng_seed)

    # rewire a proportion alpha of links randomly while leaving one node fixed
    if rewiring_type is "rewiring1":
        no_rewire = round(frac_rewire * no_links)

        link_list = np.random.permutation(no_links)

        link_choice = link_list[:int(no_rewire)]

        edge_list = list(G_new.edges())

        for i in link_choice:
            edge = edge_list[i]
            G_new.remove_edge(*edge)
            G_new.add_edge(edge[np.random.randint(0, high=2)], np.random.randint(0, high=no_nodes))

        print("rewired {} links".format(no_rewire))

    # rewire a proportion alpha of links randomly
    elif rewiring_type is "rewiring2":
        no_rewire = round(frac_rewire * no_links)

        link_list = np.random.permutation(no_links)

        link_choice = link_list[:int(no_rewire)]

        # remove random edges
        edge_list = G_new.edges()
        edges = [edge_list[i] for i in link_choice]
        G_new.remove_edges_from(edges)

        # add random edges
        node_list1 = np.random.randint(0, high=no_nodes, size=no_rewire)
        node_list2 = np.random.randint(0, high=no_nodes, size=no_rewire)

        edges = np.array([node_list1, node_list2]).T

        G_new.add_edges_from(edges)

        while G_new.number_of_edges() < no_links:
            G_new.add_edges_from(np.random.choice(no_nodes, 2, replace=False))

        print("rewired {} links".format(no_rewire))

    return G_new


# =============================================================================================

def add_attributes(G, attribute_table_path, coord_dict, verbosity=0):

    attr_df = pd.read_csv(attribute_table_path)

    if verbosity > 1:
        print("attribute data frame:")
        print(attr_df.head())

    # add coordinates of nodes if given
    if coord_dict is not None:
        lat_coord_dict = {}
        long_coord_dict = {}
        i = 0
        for fid in range(G.number_of_nodes()):
            try:
                property_code = attr_df['COD_IMOVEL'][fid]
            except KeyError:
                print("Did not find code for fid " + str(fid))

            try:
                lat_coord_dict[i] = coord_dict[property_code][0]
                long_coord_dict[i] = coord_dict[property_code][1]
            except KeyError:
                print("Missing coordinates")
                lat_coord_dict[i] = 0
                long_coord_dict[i] = 0

            i += 1

        nx.set_node_attributes(G, lat_coord_dict, name='lat')
        nx.set_node_attributes(G, long_coord_dict,  name='long')

    # add metadata to nodes
    if coord_dict is not None:
        area_dict = {}
        property_code_dict = {}
        i = 0
        try:
            area_dict = attr_df['NUM_AREA'].to_dict()
            dist_city = attr_df['dist_city'].to_dict()
            dist_road = attr_df['dist_road'].to_dict()
            dist_slaug = attr_df['dist_slaug'].to_dict()
            property_code_dict = attr_df['COD_IMOVEL'].to_dict()

            nx.set_node_attributes(G, area_dict, name='area')
            nx.set_node_attributes(G, property_code_dict, name='propertycode')
            nx.set_node_attributes(G, dist_city, name='dcity')
            nx.set_node_attributes(G, dist_road, name='droad')
            nx.set_node_attributes(G, dist_slaug, name='dslaug')

        except KeyError:
            print("Did not find attributes")

    return G


# load dataframe with initial conditions and add them to the network
def add_attribute_from_csv(g, initial_conditions_filepath, key_to_load=None, target_key=None):

    assert isinstance(key_to_load, str), "key_to_load has to be a valid string"
    assert isinstance(target_key, str), "target_key has to be a valid string"

    df = pd.read_csv(initial_conditions_filepath, index_col=0)

    initial_conditions = df[key_to_load]

    property_ids = nx.get_node_attributes(g, name='propertycode')

    idict = {}

    for node_id, property_id in property_ids.items():

        try:
            idict[node_id] = initial_conditions[property_id]
        except KeyError:
            print("No initial condition given for property {}".format(property_id))

    nx.set_node_attributes(g, idict, name=target_key)

    return g


# =============================================================================================


# make dictionary of coordinates from centroid file
def coord_dict_from_shpfile(centroid_shapefile, polygon_code_field_name="id"):

    coord_dict = dict()
    sf_coord = shp.Reader(centroid_shapefile)
    features = sf_coord.shapeRecords()

    i_index = None

    for i in range(len(sf_coord.fields)):
        if sf_coord.fields[i][0] == polygon_code_field_name:
            i_index = i - 1

    if i_index is None:
        print("unable to find the correct field in the shapefile.")
        return 0

    else:
        for feature in features:
            coordinates = np.array(feature.shape.points[0])
            subregion = str(feature.record[i_index])
            coord_dict[subregion] = coordinates

        return coord_dict


# =============================================================================================

# functions to compute and visualize network measures
def compute_network_measures(G, all_global_measures=True, centrality_measures=False,
                             write_output_file=None, verbosity=1):

    print("computing network measures...")
    no_nodes = G.number_of_nodes()
    no_edges = G.number_of_edges()
    link_density = no_edges / no_nodes
    nx.set_node_attributes(G, dict(nx.degree(G)), 'degree')
    mean_degree = np.mean(list(nx.get_node_attributes(G, 'degree').values()))

    ccs = nx.clustering(G)
    mean_clustering_coefficient = sum(list(ccs.values())) / no_nodes

    list_cc = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    nodes_in_largest_component = list_cc[0]
    # make copy to separate largest component of the graph
    G_p = G.copy()
    largest_component = sorted(nx.connected_components(G_p),
                               key=len, reverse=True)[0]
    node_list_old = list(G_p.nodes())
    for node in node_list_old:
        if not (node in largest_component):
            G_p.remove_node(node)

    if all_global_measures:
        print("calculating diameter...")
        diameter_largest_component = nx.diameter(G_p)
        print("calculating radius...")
        radius_largest_component = nx.radius(G_p)

    if centrality_measures:
        print("calculating betweenness centrality...")
        # betweenness centrality
        bnc = nx.betweenness_centrality(G_p)
        assert isinstance(bnc, dict)
        nx.set_node_attributes(G, bnc, 'betweennessc_largest_component')
        mean_betweennessc_largest_component = np.mean(list(bnc.values()))

        print("calculating eigenvector centrality...")
        # eigenvector centrality
        evc = nx.eigenvector_centrality(G_p)
        assert isinstance(evc, dict)
        nx.set_node_attributes(G, evc, 'eigenvectorc_largest_component')
        mean_eigenvectorc_largest_component = np.mean(list(evc.values()))

        print("calculating closeness centrality...")
        # closeness centrality
        clc = nx.closeness_centrality(G_p)
        assert isinstance(clc, dict)
        nx.set_node_attributes(G, clc, 'closenessc_largest_component')
        mean_closenessc_largest_component = np.mean(list(clc.values()))

    if verbosity > 0:
        print("Number of nodes: " + str(no_nodes))
        print("Number of edges: " + str(no_edges))
        print("Link density: " + str(link_density))
        print("Mean degree: " + str(mean_degree))
        print("Mean clustering coefficient: " + str(mean_clustering_coefficient))
        print("Connected components:", list_cc)

        if all_global_measures:
            print("Diameter of largest component: " + str(diameter_largest_component))
            print("Radius of largest component: " + str(radius_largest_component))

        if centrality_measures:
            print("Mean betweenness centrality (of largest component): "
                  + str(mean_betweennessc_largest_component))

            print("Mean eigenvector centrality (of largest component): "
                  + str(mean_eigenvectorc_largest_component))

            print("Mean closeness centrality (of largest component) "
                  + str(mean_closenessc_largest_component))

    if write_output_file:
        f = open(write_output_file, "w")
        f.write("Network measures for spatial graph\n")
        f.write("\nNumber of nodes: " + str(no_nodes))
        f.write("\nNumber of edges: " + str(no_edges))
        f.write("\nLink density: " + str(link_density))
        f.write("\nMean degree: " + str(mean_degree))
        f.write("\nMean clustering coefficient: " + str(mean_clustering_coefficient))
        f.write("\nConnected components: " + str(list_cc))

        if all_global_measures:
            f.write("\nDiameter of largest component: " + str(diameter_largest_component))
            f.write("\nRadius of largest component: " + str(radius_largest_component))

        if centrality_measures:
            f.write("\nMean betweenness centrality (of largest component): "
                    + str(mean_betweennessc_largest_component))

            f.write("\nMean eigenvector centrality (of largest component): "
                    + str(mean_eigenvectorc_largest_component))

            f.write("\nMean closeness centrality (of largest component) "
                    + str(mean_closenessc_largest_component))

        f.close()

    return 0


def plot_network_centrality_histograms(G, figure_file_path, log=False, normalized=True, no_bins=50):

    if 'betweennessc_largest_component' not in G.node[1]:
        compute_network_measures(G, centrality_measures=True, all_global_measures=False, verbosity=0)

    fig, ax = plt.subplots(4, figsize=(6, 8))

    degrees = dict(nx.get_node_attributes(G, 'degree'))
    # show degree distribution
    ax[0].hist(list(degrees.values()), bins=no_bins, normed=normalized)
    ax[0].set_title("degree histogram")

    betweennessc = dict(nx.get_node_attributes(G, 'betweennessc_largest_component'))
    ax[1].hist(list(betweennessc.values()), bins=no_bins, normed=normalized)
    ax[1].set_title("betweenness centrality histrogram")

    eigenvectorc = dict(nx.get_node_attributes(G, 'eigenvectorc_largest_component'))
    ax[2].hist(list(eigenvectorc.values()), bins=no_bins, normed=normalized)
    ax[2].set_title("eigenvector centrality histogram")

    closenessc = dict(nx.get_node_attributes(G, 'eigenvectorc_largest_component'))
    ax[3].hist(list(closenessc.values()), bins=no_bins, normed=normalized)
    ax[3].set_title("closeness centrality histrogram")

    if log:
        for axis in ax:
            axis.set_yscale('log', nonposy='clip')

    plt.tight_layout()
    fig.savefig(figure_file_path)
    print("saved figure to " + figure_file_path)


# plotting routine for network
def plot_network_geometric_space(G, figure_file_path, underlying_map_path):
    fig = plt.figure(figsize=(15, 20))

    pos = np.array([list(nx.get_node_attributes(G, 'lat').values()),
                   list(nx.get_node_attributes(G, 'long').values())]).T

    # draw network
    nx.draw_networkx_nodes(G, pos, node_size=70, alpha=0.5, node_color='blue')
    nx.draw_networkx_edges(G, pos, alpha=0.2)

    # draw borders of subregions
    if underlying_map_path != "":
        sf_coord = shp.Reader(underlying_map_path)
        shapeIter = sf_coord.iterShapes()
        sum_area = 0
        for shape in shapeIter:
            coordinates = np.array(shape.points).T
            plt.plot(*coordinates, color='g', linestyle='-', alpha = 0.2)

    plt.xlim(-57.2, -54.3)
    plt.ylim(-9.5, -6.0)

    #plt.axis('off')
    plt.tight_layout()
    plt.savefig(figure_file_path)
    print("saved figure to " + figure_file_path)
    return 0


# auxiliary heaviside function
def heaviside(x):
    return np.piecewise(x, [x < 0, x >= 0], [0, 1])