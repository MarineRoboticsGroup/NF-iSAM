import os
from sampler.NestedSampling import dynesty_run_batch

if __name__ == '__main__':
    run_file_dir = os.path.dirname(__file__)

    # seed_dir = f'../lawnmower_4x4/res/seed0'
    # case_dirs = [seed_dir+'/'+dir for dir in os.listdir(seed_dir) if os.path.isdir(seed_dir+'/'+dir)]
    case_dirs = []
    res_dir = run_file_dir+'/res'
    case_dirs2 = [res_dir+'/'+dir for dir in os.listdir(res_dir) if os.path.isdir(res_dir+'/'+dir)]

    case_dirs += case_dirs2

    sampling_methods = ['rwalk']
    # sampling_methods = ['auto']

    case_dirs = ['/home/chad/Research/optimalTransport/CouplingSLAM/example/slam/problems_for_paper/small_range_gaussian_problem/journal_paper/case1_da']
    for case_dir in case_dirs:
        data_file = 'factor_graph.fg'
        data_format = 'fg'
        for spl_mth in sampling_methods:
            try:
                dynesty_run_batch(500, case_dir, data_file, data_format,
                                  parallel_config={'cpu_frac': 1.0, 'queue_size': 12},
                                  incremental_step=1, prior_cov_scale=0.1,
                                  plot_args={'xlim': (-60, 60), 'ylim': (-60, 60), 'fig_size': (8, 8),
                                             'truth_label_offset': (3, -3)},
                                  dynamic_ns=False,
                                  adapt_live_pt=False,
                                  # dlogz=.1,
                                  dlogz=.05,
                                  # first_update={'min_eff': 10.0},
                                  sample=spl_mth,
                                  use_grad_u=False
                                  # dns_params = {'wt_kwargs': {'pfrac': 1.0},
                                  #               'dlogz_init': .1,
                                  #               'nlive_init': 1500,
                                  #               'maxbatch': 0,
                                  #               'use_stop': False}
                                  )
            except Exception as e:
                print(str(e))