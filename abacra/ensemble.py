# -*- coding: utf-8 -*-
"""
Class for the abacra model main module that helps to aggregate results over several model runs
"""

import pandas as pd
import os
import abacra

# ======== class for running system several times with different rng seeds ===


class ModelEnsemble(object):

    def __init__(self, *, t_max, n_runs, save_traj=False, save_stats=True,
                 plot_traj=False, initial_rng_seed=0, saving_dir=None,
                 series_name=None, verbosity=0):
        self.t_max = t_max
        self.n_runs = n_runs
        self.save_traj = save_traj
        self.save_stats = save_stats
        self.plot_traj = plot_traj
        self.rng_seed = initial_rng_seed
        print("using inital rng seed %d" % initial_rng_seed)

        self.verbosity = verbosity

        self.ensemble_df = None

        if saving_dir is not None:
            self.saving_dir = saving_dir
            if not os.path.isdir(saving_dir):
                os.mkdir(saving_dir)
        else:
            self.saving_dir = ""

        if series_name is None:
            self.series_name = "test"
        else:
            self.series_name = series_name

    def loop(self, **kwargs):

        stats_df_dict = dict()

        for n in range(self.n_runs):

            # set new random seed and initial conditions
            model = abacra.Model(rng_seed=self.rng_seed, **kwargs)
            # run with new setup
            model.run(t_max=self.t_max)

            file_str = os.path.join(self.saving_dir, self.series_name + "_")
            # save trajectories and/or compute statistics
            if self.save_traj:
                model.pickle(file_str + "traj_rng" +
                             str(self.rng_seed) + ".pkl")

            if self.save_stats:
                model.save_stats(file_str + "stats_rng" +
                                 str(self.rng_seed) + ".csv", stats="all")
            if self.plot_traj:
                model.plot_stat(file_str + "plot_median_rng" +
                                           str(self.rng_seed) + ".png",
                                           measure="median",
                                           bounds='percentiles')

                model.plot_stat(file_str + "plot_mean_rng" +
                                           str(self.rng_seed) + ".png",
                                           measure="mean")

            stats_df_dict[self.rng_seed] = model.compute_stats(stats="all", weighted_qv=True)

            del model

            self.rng_seed += 1

        self.ensemble_df = pd.concat(stats_df_dict)

        return self

    def load_ensemble_stats_csv(self):

        print("Loading ensemble statistics...")
        import re

        stats_df_dict = dict()
        for file in os.listdir(self.saving_dir):
            if self.series_name + "_stats_rng" in file:
                print("loading " + file)
                stats_df = pd.read_csv(os.path.join(self.saving_dir, file), header=[3,4],
                                       index_col=0)
                rng = re.findall(r'\d+', file)[-1]
                stats_df_dict[rng] = stats_df

        self.ensemble_df = pd.concat(stats_df_dict)

        if self.verbosity > 0:
            print("Loaded ensemble dataframe:")
            print(self.ensemble_df)

        return self


def ensemble_stats_from_csv(paths, ensemble_stat_measures=['mean', 'median', 'min', 'max'],
                            percentiles=None, verbosity=0):

    stats_df_dict = dict()

    check_pars = None

    for i, path in enumerate(paths):
        stats_df_dict[i], pars = abacra.load_stats_csv(path, verbosity=verbosity)
        if i > 0:
            if check_pars != pars:
                print("Warning: Parameters not matching:")
                print(check_pars)
                print(pars)
                print("")
        check_pars = pars
        stats_df_dict[i].name = str(i)

    ensemble_df = pd.concat(stats_df_dict, names=['run_no', 'time'])

    if ensemble_stat_measures is 'all' and percentiles is None:
        ensemble_stat_measures = ['mean', 'median', 'min', 'max', 'std']
    elif ensemble_stat_measures is 'all':
        ensemble_stat_measures = ['mean', 'median', 'min', 'max', 'std', 'percentiles']

    if type(ensemble_stat_measures) is str:
        ensemble_stat_measures = [ensemble_stat_measures]

    ensemble_stat_df_dict = {}

    for ensemble_stat_measure in ensemble_stat_measures:

        if ensemble_stat_measure == 'mean':
            ensemble_stat_df_dict['mean'] = ensemble_df.groupby(level='time').mean()

        if ensemble_stat_measure == 'median':
            ensemble_stat_df_dict['median'] = ensemble_df.groupby(level='time').median()

        if ensemble_stat_measure == 'min':
            ensemble_stat_df_dict['min'] = ensemble_df.groupby(level='time').min()

        if ensemble_stat_measure == 'max':
            ensemble_stat_df_dict['max'] = ensemble_df.groupby(level='time').max()

        if ensemble_stat_measure == 'std':
            ensemble_stat_df_dict['std'] = ensemble_df.groupby(level='time').std()

        if ensemble_stat_measure == 'percentiles':
            for percentile in percentiles:
                ensemble_stat_df_dict['perc' + str(percentile)] = \
                    ensemble_df.groupby(level='time').quantile(q=percentile/100)

    ensemble_stat_df = pd.concat(ensemble_stat_df_dict, names=['ensemble_stat_measure', 'time'])

    if verbosity > 1:
        print("ensemble statistics dataframe:")
        print(ensemble_stat_df)

    return ensemble_stat_df

