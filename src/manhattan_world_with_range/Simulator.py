import numpy as np

from factors.Factors import UnarySE2ApproximateGaussianPriorFactor, BinaryFactorWithNullHypo, \
    SE2R2RangeGaussianLikelihoodFactor, AmbiguousDataAssociationFactor, Factor, SE2RelativeGaussianLikelihoodFactor
from manhattan_world_with_range.Environment import *
import random

from slam.Variables import SE2Variable, R2Variable, VariableType, Variable
from utils.Visualization import plot_pose, plot_point, plot_likelihood_factor
import matplotlib.pyplot as plt

class SimulationArgs:
    def __init__(self,
                 range_sensing_prob = .5,
                 ambiguous_data_association_prob = 0,
                 outlier_prob = 0,
                 loop_closure_prob = 0,
                 loop_closure_radius = 0,
                 outlier_scale = 5,
                 outlier_weights = np.array([.5, .5]),
                 seed = -1,
                 range_std = 4,
                 max_da_lmk = 3):
        self.range_prob = range_sensing_prob
        self.lc_prob = loop_closure_prob
        self.lc_radius = loop_closure_radius

        # Apply to range measurements and loop closures
        self.ada_prob = ambiguous_data_association_prob
        self.outlier_prob = outlier_prob
        self.outlier_scale = outlier_scale
        self.outlier_weights = outlier_weights
        self.seed = seed

        self.range_std = range_std
        self.max_da_lmk = max_da_lmk

class ManhattanSimulator:
    """This class defines a simulator using Manhattan world-like environments
    :param
    env: simulated environment in which agents are added.
    """
    def __init__(self, env: ManhattanWaterworld,
                 args: SimulationArgs
                 ):
        self._env = env
        self._args = args
        self._rbt2gtpose = {}
        for rbt in env.robots:
            self._rbt2gtpose[rbt] = [env._rbt2pose[rbt]]

    def execute_waypoints(self, waypoints):
        raise NotADirectoryError

    def add_range_factors(self, cur_pose:SE2Pose,
                          rbt: GridRobot,
                          rbt_var: SE2Variable,
                          lmk_vars: List[R2Variable],
                          factors:List[Factor],
                          var2truth:[Variable, np.ndarray],
                          ax = None,
                          only_one_da = True,
                          max_ada_lmk = 3):
        env = self._env
        has_da = False
        for lmk in env.landmarks:
            lmk_pt = env._lmk2point[lmk]
            if random.random() < self._args.range_prob:
                r, _ = cur_pose.range_and_bearing(lmk_pt)
                var = R2Variable(name=lmk.name, variable_type=VariableType.Landmark)
                noisy_r = rbt.get_range_measurement(r)
                r_sigma = rbt._range_std
                odd = random.random()
                lmk_set = set(lmk_vars)
                if odd < self._args.outlier_prob:
                    if var not in lmk_set:
                        lmk_vars.append(var)
                        var2truth[var] = np.array([lmk_pt.x,lmk_pt.y])
                        if ax is not None:
                            plot_point(ax,point=lmk_pt, marker_size=40, color='blue', label=lmk.name, label_offset=(3,3))
                    outlier_r = noisy_r+self._args.outlier_scale*rbt._range_std
                    factors.append(BinaryFactorWithNullHypo(var1=rbt_var,
                                                            var2=var,
                                                            weights=self._args.outlier_weights,
                                                            binary_factor_class=SE2R2RangeGaussianLikelihoodFactor,
                                                            observation=outlier_r,
                                                            sigma=r_sigma,
                                                            null_sigma_scale=self._args.outlier_scale))
                elif odd < self._args.outlier_prob + self._args.ada_prob and var in lmk_set and len(lmk_vars) > 1:
                    lmk_num = len(lmk_vars)
                    create_da = False
                    if only_one_da and not has_da:
                        create_da = True
                    elif not only_one_da:
                        create_da = True
                    if create_da:
                        factors.append(AmbiguousDataAssociationFactor(observer_var=rbt_var,
                                                                      observed_vars=lmk_vars,
                                                                      weights=np.ones(lmk_num)/lmk_num,
                                                                      binary_factor_class=SE2R2RangeGaussianLikelihoodFactor,
                                                                      observation=noisy_r,
                                                                      sigma=r_sigma))
                        has_da = True
                else:
                    if var not in lmk_set:
                        lmk_vars.append(var)
                        var2truth[var] = np.array([lmk_pt.x,lmk_pt.y])
                        if ax is not None:
                            plot_point(ax,point=lmk_pt, marker_size=40, color='blue', label=lmk.name, label_offset=(3,3))
                    factors.append(SE2R2RangeGaussianLikelihoodFactor(var1=rbt_var,
                                                                      var2=var,
                                                                      observation=noisy_r,
                                                                      sigma=rbt._range_std))
                if ax is not None:
                    plot_likelihood_factor(ax, factor=factors[-1], var2truth=var2truth)

    def add_one_range_factor(self, cur_pose:SE2Pose,
                          rbt: GridRobot,
                          rbt_var: SE2Variable,
                          lmk_vars: List[R2Variable],
                          factors:List[Factor],
                          var2truth:[Variable, np.ndarray],
                          ax = None,
                          only_one_da = True):
        env = self._env
        has_da = False
        lmk = random.choice(env.landmarks)
        lmk_pt = env._lmk2point[lmk]
        if random.random() < self._args.range_prob:
            r, _ = cur_pose.range_and_bearing(lmk_pt)
            var = R2Variable(name=lmk.name, variable_type=VariableType.Landmark)
            noisy_r = rbt.get_range_measurement(r)
            r_sigma = rbt._range_std
            odd = random.random()
            lmk_set = set(lmk_vars)

            if len(lmk_vars) > self._args.max_da_lmk:
                wrong_da = list(lmk_set - {var})
                wrong_da_idx = np.arange(len(wrong_da))
                np.random.shuffle(wrong_da_idx)
                observed = [var] + [wrong_da[i] for i in wrong_da_idx[:self._args.max_da_lmk-1]]
            else:
                observed = [var] + list(lmk_set - {var})

            if odd < self._args.outlier_prob:
                if var not in lmk_set:
                    lmk_vars.append(var)
                    var2truth[var] = np.array([lmk_pt.x,lmk_pt.y])
                    if ax is not None:
                        plot_point(ax,point=lmk_pt, marker_size=40, color='blue', label=lmk.name, label_offset=(3,3))
                outlier_r = noisy_r+self._args.outlier_scale*rbt._range_std
                factors.append(BinaryFactorWithNullHypo(var1=rbt_var,
                                                        var2=var,
                                                        weights=self._args.outlier_weights,
                                                        binary_factor_class=SE2R2RangeGaussianLikelihoodFactor,
                                                        observation=outlier_r,
                                                        sigma=r_sigma,
                                                        null_sigma_scale=self._args.outlier_scale))
            elif odd < self._args.outlier_prob + self._args.ada_prob and var in lmk_set and len(lmk_vars) > 1:
                create_da = False
                if only_one_da and not has_da:
                    create_da = True
                elif not only_one_da:
                    create_da = True
                if create_da:
                    factors.append(AmbiguousDataAssociationFactor(observer_var=rbt_var,
                                                                  observed_vars=observed,
                                                                  weights=np.ones(len(observed))/len(observed),
                                                                  binary_factor_class=SE2R2RangeGaussianLikelihoodFactor,
                                                                  observation=noisy_r,
                                                                  sigma=r_sigma))
            else:
                if var not in lmk_set:
                    lmk_vars.append(var)
                    var2truth[var] = np.array([lmk_pt.x,lmk_pt.y])
                    if ax is not None:
                        plot_point(ax,point=lmk_pt, marker_size=40, color='blue', label=lmk.name, label_offset=(3,3))
                factors.append(SE2R2RangeGaussianLikelihoodFactor(var1=rbt_var,
                                                                  var2=var,
                                                                  observation=noisy_r,
                                                                  sigma=rbt._range_std))
            if ax is not None:
                plot_likelihood_factor(ax, factor=factors[-1], var2truth=var2truth)


    def single_robot_range_slam_iterate(self, rbt: GridRobot,
                                        num_rand_waypoints = 50,
                                        rbt_prefix = 'X',
                                        prior_pose_cov = np.diag([.1,.1,.02]),
                                        save_plot=None,
                                        ):
        ax = None
        if save_plot is not None:
            plt.ion()
            fig, ax = plt.subplots()

        rbt_vars = []
        lmk_vars = []
        var2truth = {}
        factors = []
        pose_id = 0
        env = self._env
        last_pose = env._rbt2pose[rbt]
        last_rbt_var = SE2Variable(rbt_prefix+str(pose_id))
        rbt_vars.append(last_rbt_var)
        var2truth[last_rbt_var] = np.array([last_pose.x, last_pose.y, last_pose.theta])

        if ax is not None:
            plot_pose(ax, last_pose, color='red')

        factors.append(UnarySE2ApproximateGaussianPriorFactor(var=last_rbt_var, prior_pose=last_pose, covariance=prior_pose_cov))
        self.add_one_range_factor(cur_pose=last_pose,rbt=rbt, rbt_var=last_rbt_var, lmk_vars=lmk_vars, factors=factors,
                               var2truth=var2truth, ax=ax)
        if save_plot is not None:
            plt.title(f'Step {pose_id}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(save_plot+f"/{pose_id}.png", dpi=300)

        for iter in range(num_rand_waypoints):
            goals = env.nearest_robot_vertex_coordinates(last_pose.x, last_pose.y)
            next_wp = rbt.select_goals(last_pose, goals)
            moves = rbt.local_path_planner(cur_pose=last_pose, goal=Point2(*next_wp))
            for move in moves:
                pose_id += 1
                rbt_var = SE2Variable(rbt_prefix+str(pose_id))
                rbt_vars.append(rbt_var)
                cur_pose = last_pose * move
                var2truth[rbt_var] = [cur_pose.x, cur_pose.y, cur_pose.theta]
                if ax is not None:
                    plot_pose(ax, pose=cur_pose, marker_size=40, color='red')

                env._rbt2pose[rbt] = cur_pose
                noisy_move = rbt.get_odom_measurement(move)
                factors.append(SE2RelativeGaussianLikelihoodFactor(var1=last_rbt_var,
                                                                   var2=rbt_var,
                                                                   observation=noisy_move,
                                                                   covariance=rbt._odom_cov))
                if ax is not None:
                    plot_likelihood_factor(ax,factors[-1],var2truth)

                self.add_one_range_factor(cur_pose=cur_pose, rbt=rbt, rbt_var=rbt_var, lmk_vars=lmk_vars,
                                       factors=factors,var2truth=var2truth, ax=ax)
                if save_plot is not None:
                    plt.title(f'Step {pose_id}')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.savefig(save_plot + f"/{pose_id}.png", dpi=300)
                last_pose = cur_pose
                last_rbt_var = rbt_var
        return rbt_vars, lmk_vars, factors, var2truth

    def single_robot_range_slam_given_waypoints(self, rbt: GridRobot,
                                        waypoints = List[Tuple],
                                        rbt_prefix = 'X',
                                        prior_pose_cov = np.diag([.1,.1,.02]),
                                        save_plot=None,
                                        ):
        ax = None
        if save_plot is not None:
            plt.ion()
            fig, ax = plt.subplots()

        rbt_vars = []
        lmk_vars = []
        var2truth = {}
        factors = []
        pose_id = 0
        env = self._env
        last_pose = env._rbt2pose[rbt]
        last_rbt_var = SE2Variable(rbt_prefix+str(pose_id))
        rbt_vars.append(last_rbt_var)
        var2truth[last_rbt_var] = np.array([last_pose.x, last_pose.y, last_pose.theta])

        if ax is not None:
            plot_pose(ax, last_pose, color='red')

        factors.append(UnarySE2ApproximateGaussianPriorFactor(var=last_rbt_var, prior_pose=last_pose, covariance=prior_pose_cov))
        self.add_one_range_factor(cur_pose=last_pose,rbt=rbt, rbt_var=last_rbt_var, lmk_vars=lmk_vars, factors=factors,
                               var2truth=var2truth, ax=ax)
        if save_plot is not None:
            plt.title(f'Step {pose_id}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(save_plot+f"/{pose_id}.png", dpi=300)

        for iter, next_wp in enumerate(waypoints):
            moves = rbt.local_path_planner(cur_pose=last_pose, goal=Point2(*self._env.vertex2coordinate(*next_wp)))
            for move in moves:
                pose_id += 1
                rbt_var = SE2Variable(rbt_prefix+str(pose_id))
                rbt_vars.append(rbt_var)
                cur_pose = last_pose * move
                var2truth[rbt_var] = [cur_pose.x, cur_pose.y, cur_pose.theta]
                if ax is not None:
                    plot_pose(ax, pose=cur_pose, marker_size=40, color='red')

                env._rbt2pose[rbt] = cur_pose
                noisy_move = rbt.get_odom_measurement(move)
                factors.append(SE2RelativeGaussianLikelihoodFactor(var1=last_rbt_var,
                                                                   var2=rbt_var,
                                                                   observation=noisy_move,
                                                                   covariance=rbt._odom_cov))
                if ax is not None:
                    plot_likelihood_factor(ax,factors[-1],var2truth)

                self.add_one_range_factor(cur_pose=cur_pose, rbt=rbt, rbt_var=rbt_var, lmk_vars=lmk_vars,
                                       factors=factors,var2truth=var2truth, ax=ax)
                if save_plot is not None:
                    plt.title(f'Step {pose_id}')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.savefig(save_plot + f"/{pose_id}.png", dpi=300)
                last_pose = cur_pose
                last_rbt_var = rbt_var
        return rbt_vars, lmk_vars, factors, var2truth