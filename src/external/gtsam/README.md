# 1.write your setting file YOUR_FILE
factor_graph_path [your blank]

output_dir [your blank]

incremental_step [your blank]

artificial_prior_sigma [your blank, float, negative means no artificial prior]

groud_truth_initialization [your blank, 1 indicates groundtruth initial values while 0 indicates bootstrapped initial values]
# 2.execute with your setting file
./gtsam_solution YOUR_FILE
