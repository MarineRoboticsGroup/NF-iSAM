import matplotlib.pyplot as plt
import numpy as np
from slam.FactorGraphSimulator import read_factor_graph_from_file
from utils.Functions import kabsch_umeyama
from utils.Visualization import plot_2d_samples
from slam.Variables import Variable, VariableType
import os
from slam.RunBatch import group_nodes_factors_incrementally
from scipy import stats
import matplotlib
from factors.Factors import PriorFactor, SE2RelativeGaussianLikelihoodFactor
from geometry.TwoDimension import SE2Pose
from example.slam.manhattan_world_with_range.mmd_rmse_time_da_plot_grid import reorder_samples

matplotlib.rcParams.update({'font.size': 16})

if __name__ == '__main__':
    seed_dir = "datasets/RangeOnlyDataset/Plaza1EFG"
    # case_list = [seed_dir+'/'+dir for dir in os.listdir(seed_dir) if os.path.isdir(seed_dir+'/'+dir)]
    case_list = [seed_dir]

    # plot_args = {'xlim': (-150, 400), 'ylim': (-150, 400), 'fig_size': (8, 8), 'truth_label_offset': (3, -3)}
    # plot_args = {'xlim': (-60, 30), 'ylim': (-20, 70), 'fig_size': (8, 8), 'truth_label_offset': (3, -3)}
    # plot_args = {'xlim': (-80, 20), 'ylim': (-20, 80), 'fig_size': (8, 8), 'truth_label_offset': (3, -3)}
    plot_args = {'fig_size': (8, 8), 'truth_label_offset': (3, -3)}
    incremental_step = 5

    num_samples = 500

    kde_bw = 'silverman'

    KU_align = True

    for case_folder in case_list:
        gtsam_folder = "run2"
        fg_file = case_folder+"/../../../RangeOnlyDataset/Plaza1EFG/factor_graph.fg"

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

        KU_t, KU_c, KU_R = None, None, None

        for step in range(len(nodes_factors_by_step)-1,-1,-1):
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
                order_file = f"{gtsam_dir}/step{step}_ordering"
                sample_file = f"{gtsam_dir}/step{step}"
            elif gtsam_folder[:3] == "gts":
                sol_label = "max-mixtures"
                step_offset = 0
                order_file = f"{gtsam_dir}/step{step}_ordering"
                sample_file= f"{gtsam_dir}/step{step}"
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
                order = Variable.file2vars(order_file=order_file)
                if cur_sample.shape[0] > num_samples:
                    cur_sample = cur_sample[np.random.choice(np.arange(len(cur_sample)), num_samples, False)]

                trans_sample = reorder_samples(ref_order=order, sample_order=order, samples=cur_sample)
                assert (trans_sample.shape[1] % 2 == 0)
                if KU_align:
                    if KU_c is None:
                        true_xy = []
                        for var in order:
                            true_xy.append(truth[var][0])
                            true_xy.append(truth[var][1])
                        true_xy = np.array(true_xy)

                        mean_xy = np.mean(trans_sample, axis=0)
                        true_xy = true_xy.reshape((-1, 2))
                        mean_xy = mean_xy.reshape((-1, 2))
                        KU_R, KU_c, KU_t = kabsch_umeyama(true_xy, mean_xy)
                    cur_dim = 0
                    for tmp_i, var in enumerate(order):
                        cur_sample[:, cur_dim:cur_dim+2] = KU_t + KU_c * trans_sample[:, tmp_i*2:tmp_i*2+2] @ KU_R.T
                        cur_dim += var.dim

                fig, ax = plt.subplots(figsize=plot_args['fig_size'])
                ax.plot(odom_x, odom_y, '-', c = '0.8')
                plot_2d_samples(ax=ax, samples_array=cur_sample, variable_ordering=order,
                                show_plot=False, equal_axis=True,
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