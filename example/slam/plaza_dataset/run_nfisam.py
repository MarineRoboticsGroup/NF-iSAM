import os
from slam.NFiSAM import NFiSAM_empirial_study

if __name__ == '__main__':
    knots = [9]
    iters = [2000]
    training_samples = [2000]
    learning_rates = [.01]
    hidden_dims = [8]

    # cases = ["Plaza1","Plaza2","Plaza1ADA0.6","Plaza2ADA0.6","Plaza1ADA0.4","Plaza2ADA0.4","Plaza1ADA0.2","Plaza2ADA0.2"]
    cases = ["Plaza1ADA0.6","Plaza2ADA0.6","Plaza1ADA0.4","Plaza2ADA0.4","Plaza1ADA0.2","Plaza2ADA0.2","Plaza1","Plaza2"]
    for case in cases:
        case_dir = f"RangeOnlyDataset/{case}EFG"
        data_file = 'factor_graph.fg'
        data_format = 'fg'
        NFiSAM_empirial_study(knots, iters, training_samples, learning_rates, hidden_dims, case_dir, data_file,
                              data_format, incremental_step=5, traj_plot=True,
                              plot_args={'truth_label_offset': (3, -3), 'show_plot': False}, cuda_training=True,
                              elimination_method='pose_first', training_set_frac=1.0, loss_delta_tol=.01,
                              average_window=50)