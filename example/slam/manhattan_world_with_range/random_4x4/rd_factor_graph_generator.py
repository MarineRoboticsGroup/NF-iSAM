import json

from manhattan_world_with_range.Simulator import *
from manhattan_world_with_range.Environment import *
from manhattan_world_with_range.Agent import *
import os

from slam.FactorGraphSimulator import factor_graph_to_string

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def batch_factor_graphs(subdir, seed, p_range=0.0, p_da=.0, p_nh=.0, step = 15,odom_std_scale = .01, range_std = 3.0, max_ada_lmk: int = 3, cell_scale = 20):
    robot_area = [(3,3),(6,6)]
    sim_env = ManhattanWaterworld(grid_vertices_shape = (10,10),
                 cell_scale = cell_scale,
                 robot_area = robot_area)
    cwd = os.path.dirname(os.path.abspath(__file__))
    rd_seed = seed
    random.seed(rd_seed)
    np.random.seed(rd_seed)
    # case_path = cwd+f"/{subdir}/step{step}_seed{seed}_range{p_range}_da{p_da}_nh{p_nh}"
    if not os.path.exists(subdir):
       os.mkdir(subdir)

    res_path = cwd+f"/{subdir}/seed"+str(seed)
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    case_path = res_path

    if not os.path.exists(case_path):
        os.mkdir(case_path)

    rbt = GridRobot(name='rbt', step_scale=cell_scale, range_std=range_std,
                    odom_cov=np.diag((cell_scale *np.array([odom_std_scale, odom_std_scale/5, odom_std_scale/10]))**2))
    # lmk1 = GridBeacon(name='L0')
    # lmk2 = GridBeacon(name='L1')
    # lmk3 = GridBeacon(name='L2')
    # lmk4 = GridBeacon(name='L3')
    # sim_env.add_landmark(lmk1,0,0)
    # sim_env.add_landmark(lmk2,7,0)
    # sim_env.add_landmark(lmk3,0,7)
    # sim_env.add_landmark(lmk4,7,7)

    lmks = []
    n_lmks = 3
    for i in range(n_lmks):
        lmks.append(GridBeacon(name=f'L{i}'))
    lmk_vertices = np.random.choice(np.arange(len(sim_env.landmark_feasible_vertices)),
                                    size=n_lmks,
                                    replace=False)
    lmk_vertices = sim_env.landmark_feasible_vertices[lmk_vertices]
    for i, lmk in enumerate(lmks):
        sim_env.add_landmark(lmk, *lmk_vertices[i])

    sim_env.add_robot(rbt,*random.choice(sim_env.robot_feasible_vertices))

    args = SimulationArgs(range_sensing_prob=p_range, ambiguous_data_association_prob=p_da, outlier_prob=p_nh, seed = seed,
                          range_std=range_std, max_da_lmk=max_ada_lmk)
    args_save = deepcopy(args.__dict__)
    args_save['odom_std_scale'] = odom_std_scale
    args_save['cell_scale'] = cell_scale
    with open(case_path+"/fg.config", 'w+') as f:
        # perform file operations
        f.write(json.dumps(args_save,cls=NumpyEncoder))

    sim = ManhattanSimulator(sim_env, args)

    gt_plot = case_path + '/gt_plot'
    if not os.path.exists(gt_plot):
        os.mkdir(gt_plot)

    rbt_vars, lmk_vars, factors, var2truth = sim.single_robot_range_slam_iterate(rbt,
                                        num_rand_waypoints = step,
                                        rbt_prefix = 'X',
                                        prior_pose_cov = np.diag([.0001,.000001,.00000001]),
                                        save_plot=gt_plot
                                        )
    vars = rbt_vars+lmk_vars
    fg = open(case_path+'/factor_graph.fg','w+')
    lines = factor_graph_to_string(vars,factors, var2truth)
    fg.write(lines)
    fg.close()

    del sim
    del sim_env

if __name__ == '__main__':
    res_folder = 'res'
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)
    for seed in range(0, 10):
        df_pada = 0.4
        df_rstd = 2
        df_mada = 3
        df_ostd = 0.01
        batch_factor_graphs(res_folder, seed, p_range=1, p_da=df_pada, p_nh=0,
                            step=15, range_std=df_rstd, odom_std_scale = df_ostd, max_ada_lmk=df_mada)