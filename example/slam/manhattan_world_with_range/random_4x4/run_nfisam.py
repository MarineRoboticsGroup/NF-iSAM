import os
from slam.NFiSAM import NFiSAM_empirial_study

if __name__ == '__main__':
    knots = [9]
    # knots = [5]
    iters = [2000]
    training_samples = [2000]
    # training_samples = [3000]
    learning_rates = [.01]
    hidden_dims = [8]
    run_file_dir = os.path.dirname(__file__)

    # df_pada = 0.4
    # df_rstd = 3
    # df_mada = 2
    # df_ostd = 0.01
    # p_adas = [0.0, 0.2, df_pada, 0.6, 0.8]
    # rstds = [1, 2, 4, 5]
    # ostds = [0.001, 0.005, 0.02, 0.03]
    # # ada_lmks = [2, 4]
    #
    # for seed in range(0, 1):
    #     for da in p_adas:
    #         case_list.append(f'seed{seed}/pada{da}_r{df_rstd}_odom{df_ostd}_mada{df_mada}')
    #     for std in rstds:
    #         case_list.append(f'seed{seed}/pada{df_pada}_r{std}_odom{df_ostd}_mada{df_mada}')
    #     for std in ostds:
    #         case_list.append(f'seed{seed}/pada{df_pada}_r{df_rstd}_odom{std}_mada{df_mada}')
    #     # for lmk in ada_lmks:
    #     #     case_list.append(f'seed{seed}/pada{df_pada}_r{df_rstd}_odom{df_ostd}_mada{lmk}')

    seed_dir = f'{run_file_dir}/res'
    case_list = [seed_dir+'/'+dir for dir in os.listdir(seed_dir) if os.path.isdir(seed_dir+'/'+dir)]

    # case_list = ["res/seed8"]
    nums = [3, 5, 6]
    case_list = [f"res/seed{i}" for i in nums]
    for case_folder in case_list:
        case_dir = os.path.join(run_file_dir, case_folder)
        data_file = 'factor_graph.fg'
        data_format = 'fg'
        NFiSAM_empirial_study(knots, iters, training_samples, learning_rates, hidden_dims, case_dir, data_file,
                              data_format, incremental_step=1, traj_plot=True,
                              plot_args={'xlim': (-150, 300), 'ylim': (-150, 300), 'fig_size': (8, 8),
                                         'truth_label_offset': (3, -3), 'show_plot': False}, cuda_training=True,
                              elimination_method='pose_first', training_set_frac=1.0, loss_delta_tol=.01,
                              average_window=50)