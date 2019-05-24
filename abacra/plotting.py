"""
plotting functions for the abacra model
"""

import matplotlib
import matplotlib.pylab as plt
import networkx as nx
import numpy as np

import matplotlib.patches
import matplotlib.colors

# to get rid of max open warnings
matplotlib.rcParams.update({'figure.max_open_warning': 0})


# new plotting routine for pandas data structure of stats
def plot_trajectory_stat(
        stats_df,
        path,
        measure="median",
        bounds=None,
        percentiles=None,
        plot_areas=True,
        plot_controls=False,
        plot_strategy=True,
        plot_active_ranches=False,
        plot_qk=True,
        plot_price=False,
        plot_gini=False,
        secveg_dynamics=False,
        t_min=None,
        dpi=150,
        ensemble_stats=False):

    axis_label_fontsize = 14
    # show also standard deviation/percentiles of aggregate variables
    # with the following alpha
    range_alpha = 0.2
    t = stats_df.index.values
    if t_min is None:
        t_min = min(t)
    t_max = max(t)

    t = t[t_min:t_max]

    no_plots = sum([plot_areas, plot_controls, plot_strategy, plot_qk, plot_price, plot_gini])

    fig, ax = plt.subplots(no_plots, sharex='all', figsize=(6, no_plots * 2 + 1))

    # plot labels
    plot_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    annotation_pos = (-0.19, 1)
    annotation_offset = (1, -1)

    # current axis
    axis_counter = 0

    if bounds is 'percentiles':
        if type(percentiles) is not list:
            print("Error: Need percentiles for plotting!")
            return 1

        lb_label = "perc" + str(percentiles[0])
        ub_label = "perc" + str(percentiles[1])

    elif bounds is 'minmax':
        lb_label = 'min'
        ub_label = 'max'

    elif bounds is 'std' or bounds is None:
        pass
    else:
        print("No valid bounds chosen for plotting")

    if plot_areas:
        cax = ax[axis_counter]

        if bounds is 'std':
            cax.fill_between(t, stats_df['F', measure][t_min:t_max]
                             - stats_df['F', 'std'][t_min:t_max],
                             stats_df['F', measure][t_min:t_max]
                             + stats_df['F', 'std'][t_min:t_max],
                             color='darkgreen',
                             alpha=range_alpha)
            cax.fill_between(t, stats_df['P', measure][t_min:t_max]
                             - stats_df['P', 'std'][t_min:t_max],
                             stats_df['P', measure][t_min:t_max]
                             + stats_df['P', 'std'][t_min:t_max], color='lawngreen',
                             alpha=range_alpha)
            cax.fill_between(t, stats_df['S', measure][t_min:t_max]
                             - stats_df['S', 'std'][t_min:t_max],
                             stats_df['S', measure][t_min:t_max]
                             + stats_df['S', 'std'][t_min:t_max], color='m', alpha=range_alpha)

        elif bounds in ['percentiles', 'minmax']:

            cax.fill_between(t, stats_df['F', lb_label][t_min:t_max],
                             stats_df['F', ub_label][t_min:t_max], color='darkgreen',
                             alpha=range_alpha)
            cax.fill_between(t,  stats_df['P', lb_label][t_min:t_max],
                             stats_df['P', ub_label][t_min:t_max], color='lawngreen',
                             alpha=range_alpha)
            cax.fill_between(t,  stats_df['S', lb_label][t_min:t_max],
                             stats_df['S', ub_label][t_min:t_max], color='m', alpha=range_alpha)

        cax.plot(t, stats_df['F', measure][t_min:t_max], color='darkgreen',
                 linewidth=2., label='Forest F')
        cax.plot(t, stats_df['P', measure][t_min:t_max], color='lawngreen',
                 linewidth=2., label='Pasture P')
        cax.plot(t, stats_df['S', measure][t_min:t_max], color='m',
                 linewidth=2., label='Sec. vegetation S')

        cax.set_ylabel(r'$F, P, S$ [ha]', fontsize=axis_label_fontsize)

        if stats_df['F', measure].max() < 1:
            cax.set_ylim([-0.1, 1.1])
        cax.set_xlim([t_min, t_max])
        cax.legend(loc='right')

        cax.annotate(plot_labels[axis_counter], xy=annotation_pos, xycoords='axes fraction', fontsize=axis_label_fontsize,
                     xytext=annotation_offset, textcoords='offset points',
                     horizontalalignment='left', verticalalignment='top')
        axis_counter += 1

    if plot_qk:
        cax = ax[axis_counter]
        cax.plot(t, stats_df['q', measure][t_min:t_max], color='saddlebrown',
                 linewidth=2., label='pasture productivity q')

        if bounds is 'std':
            cax.fill_between(t, stats_df['q', measure][t_min:t_max]
                             - stats_df['q', 'std'][t_min:t_max],
                             stats_df['q', measure][t_min:t_max]
                             + stats_df['q', 'std'][t_min:t_max], color='saddlebrown',
                             alpha=range_alpha)
        elif bounds in ['percentiles', 'minmax']:
            cax.fill_between(t,  stats_df['q', lb_label][t_min:t_max],
                             stats_df['q', ub_label][t_min:t_max], color='saddlebrown',
                             alpha=range_alpha)

        if secveg_dynamics:
            cax.plot(t, stats_df['v', measure][t_min:t_max], color='m',
                     linewidth=2., label='soil quality on S v')

            if bounds is 'std':
                cax.fill_between(t, stats_df['v', measure][t_min:t_max]
                                 - stats_df['v', 'std'][t_min:t_max],
                                 stats_df['v', measure][t_min:t_max]
                                 + stats_df['v', 'std'][t_min:t_max], color='m',
                                 alpha=range_alpha)
            elif bounds in ['percentiles', 'minmax']:
                cax.fill_between(t, stats_df['v', lb_label][t_min:t_max],
                                 stats_df['v', ub_label][t_min:t_max], color='m',
                                 alpha=range_alpha)

            cax.set_ylabel(r'$q$, $v$ [a.u.]',
                           fontsize=axis_label_fontsize, color='k')

        else:
            cax.set_ylabel(r'$q$ [a.u.]', fontsize=axis_label_fontsize,
                           color='saddlebrown')

        if not secveg_dynamics:
            for tl in cax.get_yticklabels():
                tl.set_color('saddlebrown')

        # plot k

        if max(stats_df['k', measure][t_min:t_max]) > 500000:
            rescale_k = 1000000
            rescale_k_label = "million "
        elif max(stats_df['k', measure][t_min:t_max]) > 1000:
            rescale_k = 1000
            rescale_k_label = "1000 "
        else:
            rescale_k = 1
            rescale_k_label = ""

        cax2 = cax.twinx()
        cax2.set_xlim([t_min, t_max])
        cax2.plot(t, stats_df['k', measure][t_min:t_max]/rescale_k, 'b', linewidth=2., label=r'savings $k$')
        # , marker='o', fillstyle = 'none')


        label_str = 'savings\n[{}BRL]'.format(rescale_k_label)
        cax2.set_ylabel(label_str, color='b', fontsize=axis_label_fontsize)

        if bounds is 'std':
            cax2.fill_between(t, (stats_df['k', measure][t_min:t_max]
                                  - stats_df['k', 'std'][t_min:t_max])/rescale_k,
                              (stats_df['k', measure][t_min:t_max]
                               + stats_df['k', 'std'][t_min:t_max])/rescale_k,
                              color='b', alpha=range_alpha)
        elif bounds in ['percentiles', 'minmax']:
            cax2.fill_between(t,  stats_df['k', lb_label][t_min:t_max]/rescale_k,
                              stats_df['k', ub_label][t_min:t_max]/rescale_k, color='b', alpha=range_alpha)

        for tick_label in cax2.get_yticklabels():
            tick_label.set_color('b')

        if secveg_dynamics:
            cax.legend(loc='lower right')  # loc='upper right')

        cax.annotate(plot_labels[axis_counter], xy=annotation_pos, xycoords='axes fraction', fontsize=axis_label_fontsize,
                     xytext=annotation_offset, textcoords='offset points',
                     horizontalalignment='left', verticalalignment='top')
        axis_counter += 1

    if plot_strategy:
        cax = ax[axis_counter]

        if ensemble_stats:

            if bounds is 'std':
                cax.fill_between(t,  stats_df["strategy", measure][t_min:t_max]
                                 - stats_df["strategy", "std"][t_min:t_max],
                                 stats_df["strategy", measure][t_min:t_max]
                                 + stats_df["strategy", "std"][t_min:t_max],
                                 color='k', alpha=range_alpha)
            elif bounds in ['percentiles', 'minmax']:
                cax.fill_between(t,  stats_df["strategy", lb_label][t_min:t_max],
                                 stats_df["strategy", ub_label][t_min:t_max], color='k', alpha=range_alpha)

            cax.plot(t, stats_df["strategy", measure][t_min:t_max], linewidth=2., color='k')

        else:
            cax.plot(t, stats_df["strategy", "mean"][t_min:t_max], linewidth=2., color='k')

        cax.set_ylim([0., 1.])
        cax.set_ylabel(r'intensification', fontsize=axis_label_fontsize)

        if plot_active_ranches:
            cax.plot(t, stats_df["active_ranches", "mean"][t_min:t_max], linewidth=2., color='b')

        cax.annotate(plot_labels[axis_counter], xy=annotation_pos, xycoords='axes fraction', fontsize=axis_label_fontsize,
                     xytext=annotation_offset, textcoords='offset points',
                     horizontalalignment='left', verticalalignment='top')

        axis_counter += 1

    if plot_price:
        cax = ax[axis_counter]

        if ensemble_stats:

            if bounds is 'std':
                cax.fill_between(t,  (stats_df["cattle_price", measure][t_min:t_max]
                                 - stats_df["cattle_price", "std"][t_min:t_max])/1000,
                                 (stats_df["cattle_price", measure][t_min:t_max]
                                 + stats_df["cattle_price", "std"][t_min:t_max])/1000,
                                 color='k', alpha=range_alpha)
            elif bounds in ['percentiles', 'minmax']:
                cax.fill_between(t,  stats_df["cattle_price", lb_label][t_min:t_max]/1000,
                                 stats_df["cattle_price", ub_label][t_min:t_max]/1000, color='k', alpha=range_alpha)

            cax.plot(t, stats_df["cattle_price", measure][t_min:t_max]/1000, linewidth=2., color='k')

        else:
            cax.plot(t, stats_df["cattle_price", "mean"][t_min:t_max]/1000, linewidth=2., color='k')

        cax.set_ylabel("cattle price\n[1000 BRL]", fontsize=axis_label_fontsize)
        cax.set_ylim(bottom=0.9, top=3.7)

        cax2 = cax.twinx()

        if stats_df["cattle_quantity", measure][t_min:t_max].max() > 500000:
            rescale_cattle_quantity = 1000000
            print_rescale_cattle_quantity = "million"
        elif stats_df["cattle_quantity", measure][t_min:t_max].max() > 1000:
            rescale_cattle_quantity = 1000
            print_rescale_cattle_quantity = "1000"
        else:
            rescale_cattle_quantity = 1
            print_rescale_cattle_quantity = ""

        if ensemble_stats:

            if bounds is 'std':
                cax2.fill_between(t,  (stats_df["cattle_quantity", measure][t_min:t_max]
                                       - stats_df["cattle_quantity", "std"][t_min:t_max])
                                  / rescale_cattle_quantity,
                                  (stats_df["cattle_quantity", measure][t_min:t_max]
                                   + stats_df["cattle_quantity", "std"][t_min:t_max])
                                  / rescale_cattle_quantity,
                                  color='r', alpha=range_alpha)
            elif bounds in ['percentiles', 'minmax']:
                cax2.fill_between(t,  stats_df["cattle_quantity", lb_label][t_min:t_max]/rescale_cattle_quantity,
                                  stats_df["cattle_quantity", ub_label][t_min:t_max]/rescale_cattle_quantity,
                                  color='r', alpha=range_alpha)

            cax2.plot(t, stats_df["cattle_quantity", measure][t_min:t_max]/rescale_cattle_quantity, linewidth=2., color='r')

        else:
            cax2.plot(t, stats_df["cattle_quantity", "mean"][t_min:t_max], linewidth=2., color='r')

        if rescale_cattle_quantity == 1:
            cax2.set_ylabel('cattle production\n[heads]', fontsize=axis_label_fontsize, color='r')
        else:
            cax2.set_ylabel('cattle production\n[{} heads]'.format(print_rescale_cattle_quantity), fontsize=axis_label_fontsize, color='r')
        for tl in cax2.get_yticklabels():
            tl.set_color('r')

        cax.annotate(plot_labels[axis_counter], xy=annotation_pos, xycoords='axes fraction', fontsize=axis_label_fontsize,
                     xytext=annotation_offset, textcoords='offset points',
                     horizontalalignment='left', verticalalignment='top')
        axis_counter += 1

    if plot_controls:
        cax = ax[axis_counter]

        cax.plot(t, stats_df['d', measure][t_min:t_max],
                 label=r'$\left\langle d \right\rangle$',
                 color='b')
        cax.plot(t, stats_df['a', measure][t_min:t_max],
                 label=r'$\left\langle a \right\rangle$',
                 color='r')
        cax.plot(t, stats_df['r', measure][t_min:t_max],
                 label=r'$\left\langle r \right\rangle$',
                 color='g')
        # ax4.plot(t, plot_l, label=r'$\left\langle l \right\rangle$',
        # color='m')
        cax.plot(t, stats_df['m', measure][t_min:t_max],
                 label=r'$\left\langle m \right\rangle$',
                 color='c')
        cax.set_xlim([t_min, t_max])

        cax.legend(loc='upper right')

        cax.annotate(plot_labels[axis_counter], xy=annotation_pos, xycoords='axes fraction', fontsize=axis_label_fontsize,
                     xytext=annotation_offset, textcoords='offset points',
                     horizontalalignment='left', verticalalignment='top')
        axis_counter += 1

    cax.set_xlim([t_min, t_max])
    cax.set_xlabel(r'time [years]', fontsize=axis_label_fontsize)
    plt.tight_layout()

    if type(path) is not list:
        path = [path]

    for p in path:
        fig.savefig(p, dpi=dpi)

    fig.clf()
    plt.close(fig)

    return 0


def plot_ensemble_stats(ensemble_stats_df, path, ensemble_stat_measure='mean',
                        individual_run_measure='mean', bounds='minmax', **kwargs):

    # unstack converts level of one axis to another
    ensemble_stats_df = ensemble_stats_df.unstack(level='ensemble_stat_measure')

    # slicing data frame
    ensemble_stats_df = ensemble_stats_df.xs(individual_run_measure, level=1, axis=1, drop_level=True)
    # level=1 selects the level with the statistics of individual runs

    # select the appropriate indices
    plot_trajectory_stat(ensemble_stats_df, path, ensemble_stats=True,
                         measure=ensemble_stat_measure, bounds=bounds, **kwargs)

    return 0


# auxiliary function for plotting
def draw_pie(x, center=(0, 0), radius=1., colors=None, ax=None,
             width=None, startangle=None, counterclock=False):

    if sum(x) > 0:
        x = 360. * np.array(x) / np.sum(x)
    else:
        x = [360]
        colors = ['w']

    if startangle is None:
        theta1 = 0
    else:
        theta1 = startangle

    if ax is None:
        ax = plt.gca()

    if colors is None:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    for i in range(len(x)):
        if counterclock:
            theta2 = theta1 + x[i]
        else:
            theta2 = theta1 - x[i]

        # print(center, radius, theta1, theta2)
        wedge = matplotlib.patches.Wedge(center, radius, min(theta1, theta2),
                                         max(theta1, theta2), facecolor=colors[i % len(colors)], width=width)
        ax.add_patch(wedge)
        theta1 = theta2

    return 0


# ============= plotting routine single ranch =============================

def plot_single_agent(model, path, node_id=1,
                      plot_areas=True,
                      plot_qk=True,
                      plot_controls=False,
                      plot_aux=False,
                      secveg_dynamics=False,
                      ):

    axis_label_fontsize = 14

    # print(sv_traj[0:dim])
    plot_p, plot_qp, plot_k, plot_f, plot_s, plot_qs, *aux_vars = np.copy(model.sv_traj.T[node_id * model.dim:(node_id + 1) * model.dim])

    no_plots = sum([plot_areas, plot_qk,  plot_controls, plot_aux])

    fig, ax = plt.subplots(no_plots, sharex='all', figsize=(6, no_plots * 2 + 1))

    if no_plots == 1:
        ax = [ax]

    # current axis
    axis_counter = 0
    # plot labels
    plot_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    annotation_pos = (-0.15, 1)
    annotation_offset = (1, -1)

    if plot_areas:
        cax = ax[axis_counter]
        cax.plot(model.t, plot_f, color='darkgreen', linewidth=2., label='Forest F')
        cax.plot(model.t, plot_p, color='lawngreen', linewidth=2., label='Pasture P')
        cax.plot(model.t, plot_s, color='m', linewidth=2., label='Sec. Vegetation S')
        cax.set_ylabel(r'$F, P, S$ [ha]', fontsize=axis_label_fontsize)
        if not model.pars["absolute_area"]:
            cax.set_ylim([-0.1, 1.1])
        cax.legend(loc='right')

        cax.annotate(plot_labels[axis_counter], xy=annotation_pos, xycoords='axes fraction', fontsize=axis_label_fontsize,
                     xytext=annotation_offset, textcoords='offset points',
                     horizontalalignment='left', verticalalignment='top')
        axis_counter += 1

    if plot_qk:
        cax = ax[axis_counter]
        cax.plot(model.t, plot_qp, color='saddlebrown', linewidth=2., label='pasture productivity q')
        if secveg_dynamics:
            cax.plot(model.t, plot_qs, color='m', linewidth=2., label='soil quality (on S) v')

        if secveg_dynamics:
            cax.set_ylabel(r'$q, v$ [a.u.]', fontsize=axis_label_fontsize, color='k')
        else:
            cax.set_ylabel(r'$q$ [a.u.]', fontsize=axis_label_fontsize, color='saddlebrown')
            for tl in cax.get_yticklabels():
                tl.set_color('saddlebrown')

        #cax.set_zorder(100)

        axt = cax.twinx()
        if max(plot_k) > 10000000:
            rescale_k = 1000000
        elif max(plot_k) > 10000:
            rescale_k = 1000
        else:
            rescale_k = 1

        axt.plot(model.t, plot_k/rescale_k, 'b', linewidth=2., label=r'savings $k$')  # , marker='o', fillstyle = 'none')
        if rescale_k == 1:
            axt.set_ylabel(r'savings [BRL]', color='b', fontsize=axis_label_fontsize)
        else:
            axt.set_ylabel(r'savings [{} BRL]'.format(rescale_k), color='b', fontsize=axis_label_fontsize)
        for tick_label in axt.get_yticklabels():
            tick_label.set_color('b')

        if secveg_dynamics:
            cax.legend(loc='lower right')  # loc='upper/lower right'

        cax.annotate(plot_labels[axis_counter], xy=annotation_pos, xycoords='axes fraction', fontsize=axis_label_fontsize,
                     xytext=annotation_offset, textcoords='offset points',
                     horizontalalignment='left', verticalalignment='top')
        axis_counter += 1

    if plot_controls:
        cax = ax[axis_counter]
        plot_d, plot_a, plot_r, plot_l, plot_m = np.copy(model.cv_traj.T[node_id * model.no_controls:(node_id + 1)
                                                         * model.no_controls])

        cax.plot(model.t, plot_d, label='d', color='b')
        cax.plot(model.t, plot_a, label='a', color='r')
        cax.plot(model.t, plot_r, label='r', color='g')
        cax.plot(model.t, plot_l, label='l', color='m')
        cax.plot(model.t, plot_m, label='m', color='c')

        max_plot = np.array([plot_d, plot_a, plot_r, plot_l, plot_m]).max().max()
        cax.set_ylim([-0.2*max_plot, 1.1 * max_plot])

        # print(strategy_traj.T[inspect_id])
        # visualize strategy
        for i, strategy_t in enumerate(model.strategy_traj.T[node_id]):
            if strategy_t == 0:
                color = 'g'
            else:
                color = 'r'
            cax.plot((i - 0.5, i + 0.5), (-0.1*max_plot, -0.1*max_plot), linewidth=10, color=color, solid_capstyle='butt')

        cax.legend(loc='upper right')
        cax.text(-1., -0.1*max_plot, 'strategy', horizontalalignment='right',
                 verticalalignment='center')

        cax.annotate(plot_labels[axis_counter], xy=annotation_pos, xycoords='axes fraction', fontsize=axis_label_fontsize,
                     xytext=annotation_offset, textcoords='offset points',
                     horizontalalignment='left', verticalalignment='top')
        axis_counter += 1

    if plot_aux:
        cax = ax[axis_counter]
        for i, aux_var in enumerate(aux_vars):
            cax.plot(model.t, aux_var, label=model.state_variables[model.dim - model.no_aux_vars + i])

        cax.legend(loc='upper right')

        cax.annotate(plot_labels[axis_counter], xy=annotation_pos, xycoords='axes fraction', fontsize=axis_label_fontsize,
                     xytext=annotation_offset, textcoords='offset points',
                     horizontalalignment='left', verticalalignment='top')
        axis_counter += 1

    cax.set_xlabel(r'time [years]', fontsize=axis_label_fontsize)

    plt.tight_layout()

    fig.savefig(path, dpi=model.plot_pars["dpi_saving"])

    fig.clf()
    plt.close(fig)

    return 0


# ============= plotting routine all ranches ==============================
# incomplete
def sample_of_trajectories(model, path, agent_ids=range(100), secveg_dynamics=False):

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='all', figsize=(6, 6))

    ax1b = ax2.twinx()

    if agent_ids == "all":
        agent_ids = range(model.no_agents)

    for inspect_id in agent_ids:

        print("plotting trajectory %d" % inspect_id )
        # print(sv_traj[0:dim])
        plot_P, plot_qp, plot_k, plot_F, plot_S, plot_qs, *aux_vars = np.copy(
            model.sv_traj.T[inspect_id * model.dim:(inspect_id + 1) * model.dim])

        ax1.plot(model.t, plot_F, color='darkgreen', linewidth=.5, label='F', alpha=0.3)
        ax1.plot(model.t, plot_P, color='lawngreen', linewidth=.5, label='P', alpha=0.3)
        ax1.plot(model.t, plot_S, color='m', linewidth=.5, label='S', alpha=0.3)

        ax2.plot(model.t, plot_qp, color='saddlebrown', linewidth=.5, label='q', alpha=0.3)
        if secveg_dynamics:
            ax2.plot(model.t, plot_qs, color='m', linewidth=.5, label=r'qs', alpha=0.3)
        ax1b.plot(model.t, plot_k, 'b', linewidth=.5, label=r'$k$', alpha=0.3)  # , marker='o', fillstyle = 'none')

        plot_d, plot_a, plot_r, plot_l, plot_m = np.copy(
            model.cv_traj.T[inspect_id * model.no_controls:(inspect_id + 1) * model.no_controls])

        ax3.plot(model.t, plot_d, label='d', color='b', linewidth=.5, alpha=0.3)
        ax3.plot(model.t, plot_a, label='a', color='r', linewidth=.5, alpha=0.3)
        ax3.plot(model.t, plot_r, label='r', color='g', linewidth=.5, alpha=0.3)
        ax3.plot(model.t, plot_l, label='l', color='m', linewidth=.5, alpha=0.3)
        ax3.plot(model.t, plot_m, label='m', color='c', linewidth=.5, alpha=0.3)

    ax1.set_ylabel(r'$F, P, S$', fontsize=16)
    if not model.pars["absolute_area"]:
        ax1.set_ylim([-0.1, 1.1])

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[0:3], labels[0:3], loc='upper left')

    ax2.set_ylabel(r'$q$', fontsize=16, color='saddlebrown')
    for tl in ax2.get_yticklabels():
        tl.set_color('saddlebrown')

    ax1b.set_ylabel(r'$k$', color='b', fontsize=16)
    for tick_label in ax1b.get_yticklabels():
        tick_label.set_color('b')

    ax3.set_xlabel(r't', fontsize=16)

    ax3.set_ylim([-0.2, 1])

    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles[0:5], labels[0:5], loc='upper left')

    ax3.text(-1., -0.1, 'strategy', horizontalalignment='right',
             verticalalignment='center')

    plt.tight_layout()

    fig.savefig(path, dpi=model.plot_pars["dpi_saving"])

    fig.clf()
    plt.close(fig)

    return 0

# ================= spatial representation and animation ==================


def prepare_network_fig(model, network_type, xylim=None, gml_setting='studyregion1'):

    # instead of creating a new figure, use the old one and delete its contents
    fig = plt.gcf()
    fig.clf()

    if network_type is 'grid_2d':
        fig = plt.figure(figsize=(5, 5 * model.pars["N_y"] / model.pars["N_x"]))
        plt.axes([0, 0, 1, 1], aspect=1)

        plt.xlim([-0.5, model.pars["N_x"] - 0.5])
        plt.ylim([-0.5, model.pars["N_y"] - 0.5])
        node_size = 80

    elif network_type is 'ws':
        fig = plt.figure(figsize=(7, 7))

        plt.xlim([-model.plot_pars["nx_scale"] - 0.5, model.plot_pars["nx_scale"] + 0.5])
        plt.ylim([-model.plot_pars["nx_scale"] - 0.5, model.plot_pars["nx_scale"] + 0.5])
        node_size = 80

    elif network_type is 'geometric':
        fig = plt.figure(figsize=(5, 5 * model.pars["N_y"] / model.pars["N_x"]))

        plt.xlim([-0.5, model.pars["N_x"] + 0.5])
        plt.ylim([-0.5, model.pars["N_y"] + 0.5])
        node_size = 80

    # municipalities
    elif network_type is 'gml' and gml_setting is 'municipalities':
        fig_scale = 2
        fig = plt.figure(figsize=(fig_scale * 9, fig_scale * 7))
        plt.xlim(-76, -42)
        plt.ylim(-19, 5.5)
        node_size = 80

    # car_novo_progresso
    elif network_type is 'gml' and gml_setting is 'novo_progresso':
        fig = plt.figure(figsize=(15, 20))
        plt.xlim(-56.25, -54.75)
        plt.ylim(-8.5, -6.5)
        node_size = 50

    # studyregion1
    elif network_type is 'gml' and gml_setting is 'studyregion1':
        fig.set_size_inches(15, 20)
        plt.xlim(-57.2, -54.3)
        plt.ylim(-9.5, -6.0)
        node_size = 25

    else:
        print("no valid network_type chosen")
        return 1

    if model.verbosity > 1:
        print("prepared map for {}".format(network_type))
        print("node size: {}".format(node_size))

    if xylim is not None:
        assert np.array(xylim).shape == (2,2), "xylim has to be an array of the form [[xmin, xmax],[ymin, ymax]]"
        plt.xlim(xylim[0])
        plt.ylim(xylim[1])

    else:
        print("Using default xlim and ylim.")

    return fig, node_size


# ============ snapshot plot of the network with lu pies ================


def plot_pie_network(model, path, t=20, xylim=None, show_landcover_pies=True, annotation=None):

    fig, node_size = prepare_network_fig(model, model.pars["network_type"], xylim=xylim)

    P_traj = model.sv_traj.T[::model.dim]
    # q_traj = self.sv_traj.T[1::5]
    # k_traj = self.sv_traj.T[2::5]
    F_traj = model.sv_traj.T[3::model.dim]
    S_traj = model.sv_traj.T[4::model.dim]

    if show_landcover_pies:
        node_pos = list(model.node_pos.values())
        for node in range(model.no_agents):
            draw_pie([F_traj[node][t], P_traj[node][t], S_traj[node][t]], center=node_pos[node],
                     radius=model.plot_pars["pie_radius"], startangle=90, width=0.8 * model.plot_pars["pie_radius"],
                     colors=('darkgreen', 'lawngreen', 'm'))

    nx.draw_networkx_edges(model.G, model.node_pos, alpha=0.2)

    strategy_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', [(0, 'darkblue'), (1, 'darkred')])
    # because mapping does not work if there is no node with strategy 0:
    if sum(model.strategy_traj[t]) == model.no_agents:
        strategy_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', [(0, 'darkred'), (1, 'darkred')])

    nx.draw_networkx_nodes(model.G, model.node_pos, node_color=model.strategy_traj[t], node_size=node_size,
                           cmap=strategy_cmap)

    if annotation is not None:
        plt.annotate(annotation, xy=(0, 1), xycoords='axes fraction', fontsize=36,
                     xytext=(5, -5), textcoords='offset points',
                     horizontalalignment='left', verticalalignment='top')

    circle1 = plt.Line2D((0, 1), (0, 0), marker='o', linestyle='', markersize=12,
                         color="darkblue", label='Extensive Strategy')

    circle2 = plt.Line2D((0, 1), (0, 0), marker='o', linestyle='', markersize=12,
                         color="darkred", label='Intensive Strategy')

    legend_handles = [circle1, circle2]

    if show_landcover_pies:

        fpatch = matplotlib.patches.Patch(facecolor='darkgreen', edgecolor='k', label='Forest')
        ppatch = matplotlib.patches.Patch(facecolor='lawngreen', edgecolor='k', label='Pasture')
        svpatch = matplotlib.patches.Patch(facecolor='m', edgecolor='k', label='Sec. Vegetation')

        legend_handles.append([fpatch, ppatch, svpatch])

    plt.legend(handles=legend_handles, loc='lower left', prop={'size': 24})

    plt.tight_layout()
    plt.axis('off')

    plt.savefig(path, dpi=model.plot_pars["dpi_saving"])

    plt.clf()
    plt.close(fig)

    return 0


# ========== make pie plot animation ======================================


def make_pie_animation(model, path):

    import matplotlib.animation

    P_traj = model.sv_traj.T[::model.dim]
    # q_traj = self.sv_traj.T[1::model.dim]
    # k_traj = self.sv_traj.T[2::model.dim]
    F_traj = model.sv_traj.T[3::model.dim]
    S_traj = model.sv_traj.T[4::model.dim]

    metadata = dict(title='Amazonas ABM animation', artist='FMH', comment='test')
    writer = matplotlib.animation.FFMpegFileWriter(fps=10, metadata=metadata)

    print(writer.fps)

    fig = model.prepare_network_fig(model.pars['network_type'], model.G)
    strategy_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', [(0, 'darkblue'), (1, 'darkred')])

    with writer.saving(fig, path, 100):
        for t in range(model.t_max):
            plt.cla()
            nx.draw_networkx_edges(model.G, model.node_pos)

            for node in range(model.no_nodes):
                draw_pie([F_traj[node][t], P_traj[node][t], S_traj[node][t]],
                         center=model.node_pos.values()[node],
                         radius=0.45, startangle=90, width=0.35,
                         colors=('darkgreen', 'lawngreen', 'm'))

            nx.draw_networkx_nodes(model.G, model.node_pos, node_color=model.strategy_traj[t], node_size=80,
                                   cmap=strategy_cmap)

            plt.text(0.1, 0.1, "t = " + str(t), horizontalalignment='right', fontsize=30,
                     verticalalignment='center', transform=plt.gca().transAxes)
            plt.axis('off')
            if t == 0:
                plt.tight_layout()
            for i in range(2):
                writer.grab_frame()

    print(writer.fps)
    print(writer.frame_size)

    # alternative implementation
    # https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
    # writer.save('filename.mp4', fps=120, writer=animation.FFMpegFileWriter())

    return 0
