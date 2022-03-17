import matplotlib.pyplot as plt
import numpy as np
from slam.FactorGraphSimulator import read_factor_graph_from_file
from utils.Visualization import plot_2d_samples
from slam.Variables import Variable, VariableType
import os
from slam.RunBatch import group_nodes_factors_incrementally
from scipy import stats
import matplotlib
from factors.Factors import PriorFactor, SE2RelativeGaussianLikelihoodFactor
from geometry.TwoDimension import SE2Pose

matplotlib.rcParams.update({'font.size': 16})

if __name__ == '__main__':

    if_side_plots = False
    side_plot_type = "kde" # or "kde"
    targ_var_name = "L1"

    seed_dir = "res/seed0"
    case_list = [seed_dir+'/'+dir for dir in os.listdir(seed_dir) if os.path.isdir(seed_dir+'/'+dir)]

    plot_args = {'xlim': (-150, 400), 'ylim': (-150, 400), 'fig_size': (8, 8), 'truth_label_offset': (3, -3)}
    incremental_step = 1

    num_samples = 500

    kde_bw = 'silverman'

    for case_folder in case_list:
        gtsam_folder = "run6"
        fg_file = case_folder+"/factor_graph.fg"

        gtsam_dir = f"{case_folder}/{gtsam_folder}"

        nodes, truth, factors = read_factor_graph_from_file(fg_file)

        plot_dir = f"{gtsam_dir}/traj_video"
        if(os.path.exists(plot_dir)):
            pass
        else:
            os.mkdir(plot_dir)

        nodes_factors_by_step = group_nodes_factors_incrementally(
            nodes=nodes, factors=factors, incremental_step=incremental_step)

        rbt_vars = []
        var2pose = {}
        odom_x = []
        odom_y = []
        for step in range(len(nodes_factors_by_step)):
            step_nodes, step_factors = nodes_factors_by_step[step]
            for f in step_factors:
                if isinstance(f, PriorFactor):
                    rbt_vars.append(f.vars[0])
                    var2pose[f.vars[0]] = SE2Pose(*f.observation)
                    odom_y.append(var2pose[rbt_vars[-1]].y)
                    odom_x.append(var2pose[rbt_vars[-1]].x)
                elif isinstance(f, SE2RelativeGaussianLikelihoodFactor):
                    if f.var1 == rbt_vars[-1]:
                        var2pose[f.var2] = var2pose[f.var1] * SE2Pose(*f.observation)
                        rbt_vars.append(f.var2)
                        odom_y.append(var2pose[rbt_vars[-1]].y)
                        odom_x.append(var2pose[rbt_vars[-1]].x)

        cur_factors = []
        for step in range(len(nodes_factors_by_step)):
            step_file_prefix = f"{plot_dir}/step{step}"
            cur_sample = None
            step_nodes, step_factors = nodes_factors_by_step[step]

            recent_rbt_vars = []
            for var in step_nodes:
                if var.type == VariableType.Pose:
                    recent_rbt_vars.append(var)
            cur_factors += step_factors

            if gtsam_folder[:3] == "cae":
                sol_label = "mm-iSAM"
                order_file = f"{gtsam_dir}/batch{step+1}.ordering"
                sample_file = f"{gtsam_dir}/batch{step+1}"
            elif gtsam_folder[:3] == "gts":
                sol_label = "max-mixtures"
                step_offset = 0
                order_file = f"{gtsam_dir}/batch_{step}_ordering"
                sample_file= f"{gtsam_dir}/batch{step}"
            elif gtsam_folder[:3] == "dyn":
                sample_file = f"{gtsam_dir}/step{step}.sample"
                order_file = f"{gtsam_dir}/step{step}_ordering"
                step_offset = 0
                sol_label = "Nested sampling"
            else:
                order_file = f"{gtsam_dir}/step{step}_ordering"
                sample_file = f"{gtsam_dir}/step{step}"
                step_offset = 0
                sol_label = "NF-iSAM"

            if os.path.exists(sample_file):
                cur_sample = np.loadtxt(fname=sample_file)
                if cur_sample.shape[0] > num_samples:
                    cur_sample = cur_sample[np.random.choice(np.arange(len(cur_sample)), num_samples, False)]
                order = Variable.file2vars(order_file=order_file)
                if not if_side_plots:
                    fig, ax = plt.subplots(figsize=plot_args['fig_size'])
                    ax.plot(odom_x, odom_y, '-', c = '0.8')
                    plot_2d_samples(ax=ax, samples_array=cur_sample, variable_ordering=order,
                                    show_plot=False, equal_axis=False,
                                    # truth={variable: pose for variable, pose in
                                    #        truth.items() if variable in order},
                                    # truth_factors={factor for factor in cur_factors},
                                    truth=truth,
                                    truth_factors=factors,
                                    file_name=f"{step_file_prefix}.png", title=f'{sol_label} (step {step})',
                                    plot_all_meas = False,
                                    plot_meas_give_pose = recent_rbt_vars,
                                    rbt_traj_no_samples = True,
                                    truth_R2 = True,
                                    truth_SE2 = False,
                                    truth_odometry_color = 'k',
                                    truth_landmark_markersize = 15,
                                    **plot_args)
                    plt.close()
                else:
                    # start with a square Figure
                    fig = plt.figure(figsize=plot_args['fig_size'])
                    fig.suptitle(f'{sol_label} (step {step})')

                    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
                    # the size of the marginal axes and the main axes in both directions.
                    # Also adjust the subplot parameters for a square plot.
                    gs = fig.add_gridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5),
                                          left=0.15, right=0.95, bottom=0.1, top=0.9,
                                          wspace=0.05, hspace=0.05)

                    ax = fig.add_subplot(gs[1, 0])
                    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
                    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

                    ax_histx.tick_params(axis="x", labelbottom=False)
                    ax_histy.tick_params(axis="y", labelleft=False)

                    plot_2d_samples(ax=ax, samples_array=cur_sample, variable_ordering=order,
                                    show_plot=False, equal_axis=False,
                                    truth={variable: pose for variable, pose in
                                           truth.items() if variable in order},
                                    truth_factors={factor for factor in cur_factors},
                                    **plot_args)
                    # use the previously defined function
                    exist_names = [var.name for var in order]
                    if targ_var_name in set(exist_names):
                        targ_var = order[exist_names.index(targ_var_name)]
                        straight_x = np.linspace(truth[targ_var][1], plot_args['ylim'][1], 10)
                        straight_y = np.linspace(truth[targ_var][0], plot_args['xlim'][1], 10)
                        ax.plot(straight_y, truth[targ_var][1] * np.ones_like(straight_y), '--r')
                        ax.plot(truth[targ_var][0] * np.ones_like(straight_x), straight_x, '--r')

                        cur_dim = 0
                        for var in order:
                            if var.name == targ_var_name:
                                break
                            cur_dim += var.dim
                        x = cur_sample[:, cur_dim]
                        y = cur_sample[:, cur_dim+1]
                        if side_plot_type == "hist":
                            binwidth = 1.0
                            xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
                            lim = (int(xymax / binwidth) + 1) * binwidth
                            bins = np.arange(-lim, lim + binwidth, binwidth)
                            ax_histx.hist(x, bins=bins)
                            ax_histy.hist(y, bins=bins, orientation='horizontal')
                        else:
                            pts = np.linspace(plot_args['xlim'][0], plot_args['xlim'][1], 500)
                            x_kernel = stats.gaussian_kde(x, bw_method=kde_bw)
                            y_kernel = stats.gaussian_kde(y, bw_method=kde_bw)
                            ax_histx.plot(pts, x_kernel(pts), '-b', label=f'{targ_var_name}x')
                            ax_histx.legend(prop={'size': 9})
                            ax_histy.plot(y_kernel(pts), pts, '-b', label=f'{targ_var_name}y')
                            ax_histy.set_xlim([0, 1.1*max(y_kernel(pts))])
                            # ax_histy.invert_yaxis()
                            ax_histy.legend(prop={'size': 9})
                    else:
                        ax_histx.axis("off")
                        ax_histy.axis("off")
                    plt.savefig(f"{step_file_prefix}.png", dpi=300)
                    plt.show()
            else:
                if not if_side_plots:
                    plt.figure(figsize=plot_args['fig_size'])
                    plt.plot(0,0)
                    plt.xlim(plot_args['xlim'])
                    plt.ylim(plot_args['ylim'])
                    plt.title(f"{sol_label} (step {step})")
                    plt.xlabel('x(m)')
                    plt.ylabel('y(m)')
                    plt.savefig(f"{step_file_prefix}.png", dpi=300)
                    plt.show()
                    plt.close()
                else:
                    fig = plt.figure(figsize=plot_args['fig_size'])
                    fig.suptitle(f"{sol_label} (step {step})")

                    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
                    # the size of the marginal axes and the main axes in both directions.
                    # Also adjust the subplot parameters for a square plot.
                    gs = fig.add_gridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5),
                                          left=0.15, right=0.95, bottom=0.1, top=0.9,
                                          wspace=0.05, hspace=0.05)

                    ax = fig.add_subplot(gs[1, 0])
                    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
                    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
                    ax_histy.axis("off")
                    ax_histx.axis("off")

                    ax_histx.tick_params(axis="x", labelbottom=False)
                    ax_histy.tick_params(axis="y", labelleft=False)

                    ax.plot(0,0)
                    ax.set_xlim(plot_args['xlim'])
                    ax.set_ylim(plot_args['ylim'])
                    ax.set_xlabel('x (m)')
                    ax.set_ylabel('y (m)')
                    plt.savefig(f"{step_file_prefix}.png", dpi=300)
                    plt.show()
                    plt.close()