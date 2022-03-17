import os
import random
import numpy as np

from slam.NFiSAM import NFiSAM_empirial_study


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    knots = [9]
    iters = [2000]
    training_samples = [2000]
    learning_rates = [.025]
    hidden_dims = [8]

    run_file_dir = os.path.dirname(__file__)
    case_folder = 'journal_paper/case1'
    case_dir = os.path.join(run_file_dir, case_folder)
    data_file = 'factor_graph.fg'
    data_format = 'fg'
    NFiSAM_empirial_study(knots, iters, training_samples, learning_rates, hidden_dims, case_dir, data_file, data_format,
                          incremental_step=1, traj_plot=True,
                          plot_args={'xlim': (-100, 100), 'ylim': (-100, 100), 'fig_size': (8, 8),
                                     'truth_label_offset': (3, -3), 'show_plot': False}, check_root_transform=False, cuda_training=True,
                          elimination_method='pose_first', training_set_frac=1.0, loss_delta_tol=.01, posterior_sample_num=1000)