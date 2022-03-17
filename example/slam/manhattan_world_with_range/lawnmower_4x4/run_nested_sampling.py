import os
from sampler.NestedSampling import dynesty_run_batch

if __name__ == '__main__':
    run_file_dir = os.path.dirname(__file__)
    sampling_methods = ['rwalk']

    for seed in range(1):
        seed_dir = f'{run_file_dir}'
        case_dirs = [seed_dir+'/'+dir for dir in os.listdir(seed_dir) if os.path.isdir(seed_dir+'/'+dir)]
        for case_dir in case_dirs:
            data_file = 'factor_graph.fg'
            data_format = 'fg'
            for spl_mth in sampling_methods:
                try:
                    dynesty_run_batch(1000, case_dir, data_file, data_format, parallel_config={'cpu_frac': 1.0, 'queue_size': 12},
                                      incremental_step=16, prior_cov_scale=0.1,
                                      plot_args={'xlim': (-150,300), 'ylim': (-150,300), 'fig_size': (8, 8),
                                                 'truth_label_offset': (3, -3)},
                                      dynamic_ns=False,
                                      adapt_live_pt=False,
                                      dlogz=.1,
                                      sample=spl_mth,
                                      use_grad_u=False
                                      )
                except Exception as e:
                    print(str(e))