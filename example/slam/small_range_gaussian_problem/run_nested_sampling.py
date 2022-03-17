import os
from sampler.NestedSampling import dynesty_run_batch

if __name__ == '__main__':
    run_file_dir = os.path.dirname(__file__)

    sampling_methods = ['auto']

    # case_dirs = ['/home/chad/Research/optimalTransport/CouplingSLAM/example/slam/problems_for_paper/small_range_gaussian_problem/journal_paper/case1']
    # parent_dir = "../../manhattan_world_with_range/3Poses1Lmk"
    parent_dir = "journal_paper/case1_da"
    case_dirs = [parent_dir]

    # case_dirs = []
    # hypo_num = [3,10,15]
    for case_dir in case_dirs:
        # for i in hypo_num:
        #     data_file = f'factor_graph{i}.fg'
        data_file = f'factor_graph.fg'
        data_format = 'fg'
        for spl_mth in sampling_methods:
            try:
                dynesty_run_batch(1000, case_dir, data_file, data_format,
                                  parallel_config={'cpu_frac': 1.0, 'queue_size': 12},
                                  incremental_step=1, prior_cov_scale=0.1,
                                  plot_args={'xlim': (-100, 100), 'ylim': (-100, 100), 'fig_size': (8, 8),
                                             'truth_label_offset': (3, -3)},
                                  dynamic_ns=False,
                                  adapt_live_pt=False,
                                  # dlogz=.1,
                                  dlogz=.1,
                                  # xlim=[-40, 55],
                                  # ylim=[-40, 55],
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