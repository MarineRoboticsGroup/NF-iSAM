import os
from slam.NFiSAM import NFiSAM_empirial_study

if __name__ == '__main__':
    # knots = [5,7,9,11,13]
    # hidden_dims = [4,6,8,10,12]
    knots = [9]
    hidden_dims = [8]
    iters = [2000]
    training_samples = [2000]
    learning_rates = [.02]
    run_file_dir = os.path.dirname(__file__)

    seed_dir = f'{run_file_dir}/res/seed1/'

    case_list = [seed_dir+'/pada0.4_r2_odom0.01_mada3']

    for case_folder in case_list:
        case_dir = os.path.join(run_file_dir, case_folder)
        data_file = 'factor_graph.fg'
        data_format = 'fg'
        NFiSAM_empirial_study(knots, iters, training_samples, learning_rates, hidden_dims, case_dir, data_file,
                              data_format, incremental_step=1, traj_plot=True,
                              plot_args={'xlim': (-150, 300), 'ylim': (-150, 300), 'fig_size': (8, 8),
                                         'truth_label_offset': (3, -3), 'show_plot': False}, cuda_training=True,
                              posterior_sample_num=1000, elimination_method='pose_first', training_set_frac=1.0,
                              loss_delta_tol=.01, average_window=50)