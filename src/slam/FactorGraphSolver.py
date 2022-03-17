import json
import os
import time
from copy import deepcopy

import TransportMaps.Distributions as dist
import TransportMaps.Likelihoods as like
from typing import List, Dict

from matplotlib import pyplot as plt

from factors.Factors import Factor, ExplicitPriorFactor, ImplicitPriorFactor, \
    LikelihoodFactor, BinaryFactorMixture, KWayFactor
from sampler.NestedSampling import GlobalNestedSampler
from sampler.SimulationBasedSampler import SimulationBasedSampler
from slam.Variables import Variable, VariableType
from slam.FactorGraph import FactorGraph
from slam.BayesTree import BayesTree, BayesTreeNode
import numpy as np
from sampler.sampler_utils import JointFactor

from utils.Functions import sort_pair_lists
from utils.Visualization import plot_2d_samples
from utils.Functions import sample_dict_to_array, array_order_to_dict


class SolverArgs:
    def __init__(self,
                 elimination_method: str = "natural",
                 posterior_sample_num: int = 500,
                 local_sample_num: int = 500,
                 store_clique_samples: bool = False,
                 local_sampling_method="direct",
                 adaptive_posterior_sampling=None,
                 *args, **kwargs
                 ):
        # graph-related and tree-related params
        self.elimination_method = elimination_method
        self.posterior_sample_num = posterior_sample_num
        self.store_clique_samples = store_clique_samples
        self.local_sampling_method = local_sampling_method
        self.local_sample_num = local_sample_num
        self.adaptive_posterior_sampling = adaptive_posterior_sampling

    def jsonStr(self):
        return json.dumps(self.__dict__)


class CliqueSeparatorFactor(ImplicitPriorFactor):
    def sample(self, num_samples: int, **kwargs):
        return NotImplementedError("implementation depends on density models")


class ConditionalSampler:
    def conditional_sample_given_observation(self, conditional_dim,
                                             obs_samples=None,
                                             sample_number=None):
        """
        This method returns samples with the dimension of conditional_dim.
        If sample_number is given, samples of the first conditional_dim variables are return.
        If obs_samples is given, samples of the first conditional_dim variables after
            the dimension of obs_samples will be returned. obs_samples.shape = (sample num, dim)
        Note that the dims here are of the vectorized point on manifolds not the dim of manifold.
        """
        raise NotImplementedError("Implementation depends on density estimation method.")


class FactorGraphSolver:
    """
    This is the abstract class of factor graph solvers.
    It mainly works as:
        1. the interface for users to define and solve factor graphs.
        2. the maintainer of factor graphs and Bayes tree for incremental inference
        3. fitting probabilistic models to the working part of factor graph and Bayes tree
        4. inference (sampling) on the entire Bayes tree
    The derived class may reply on different probabilistic modeling approaches.
    """

    def __init__(self, args: SolverArgs):
        """
        Parameters
        ----------
        elimination_method : string
            option of heuristics for variable elimination ordering.
            TODO: this can be a dynamic parameter when updating Bayes tree
        """
        self._args = args
        self._physical_graph = FactorGraph()
        self._working_graph = FactorGraph()
        self._physical_bayes_tree = None
        self._working_bayes_tree = None
        self._conditional_couplings = {}  # map from Bayes tree clique to flows
        self._implicit_factors = {}  # map from Bayes tree clique to factor
        self._samples = {}  # map from variable to samples
        self._new_nodes = []
        self._new_factors = []
        self._clique_samples = {}  # map from Bayes tree clique to samples
        self._clique_true_obs = {}  # map from Bayes tree clique to observations which augments flow models
        self._clique_density_model = {}  # map from Bayes tree clique to flow model
        # map from Bayes tree clique to variable pattern; (Separator,Frontal) in reverse elimination order
        self._clique_variable_pattern = {}
        self._elimination_ordering = []
        self._reverse_ordering_map = {}
        self._temp_training_loss = {}

    def set_args(self, args: SolverArgs):
        raise NotImplementedError("Implementation depends on probabilistic modeling approaches.")

    @property
    def elimination_method(self) -> str:
        return self._args.elimination_method

    @property
    def elimination_ordering(self) -> List[Variable]:
        return self._elimination_ordering

    @property
    def physical_vars(self) -> List[Variable]:
        return self._physical_graph.vars

    @property
    def new_vars(self) -> List[Variable]:
        return self._new_nodes

    @property
    def working_vars(self) -> List[Variable]:
        return self._working_graph.vars

    @property
    def physical_factors(self) -> List[Factor]:
        return self._physical_graph.factors

    @property
    def new_factors(self) -> List[Factor]:
        return self._new_factors

    @property
    def working_factors(self) -> List[Factor]:
        return self._working_graph.factors

    @property
    def working_factor_graph(self) -> FactorGraph:
        return self._working_graph

    @property
    def physical_factor_graph(self) -> FactorGraph:
        return self._physical_graph

    @property
    def working_bayes_tree(self) -> BayesTree:
        return self._working_bayes_tree

    @property
    def physical_bayes_tree(self) -> BayesTree:
        return self._physical_bayes_tree

    def generate_natural_ordering(self) -> None:
        """
        Generate the ordering by which nodes are added
        """
        self._elimination_ordering = self._physical_graph.vars + self._new_nodes

    def generate_pose_first_ordering(self) -> None:
        """
        Generate the ordering by which nodes are added and lmk eliminated later
        """
        natural_order = self._physical_graph.vars + self._new_nodes
        pose_list = []
        lmk_list = []
        for node in natural_order:
            if node._type == VariableType.Landmark:
                lmk_list.append(node)
            else:
                pose_list.append(node)
        self._elimination_ordering = pose_list + lmk_list

    def generate_ccolamd_ordering(self) -> None:
        """

        """
        physical_graph_ordering = [var for var in self._elimination_ordering if var not in self._working_graph.vars]
        working_graph_ordering = self._working_graph.analyze_elimination_ordering(
            method="ccolamd", last_vars=
            [[var for var in self._working_graph.vars if
              var.type == VariableType.Pose][-1]])
        self._elimination_ordering = physical_graph_ordering + working_graph_ordering

    def generate_ordering(self) -> None:
        """
        Generate the ordering by which Bayes tree should be generated
        """
        if self._args.elimination_method == "natural":
            self.generate_natural_ordering()
        elif self._args.elimination_method == "ccolamd":
            self.generate_ccolamd_ordering()
        elif self._args.elimination_method == "pose_first":
            self.generate_pose_first_ordering()

        self._reverse_ordering_map = {
            var: index for index, var in
            enumerate(self._elimination_ordering[::-1])}
        # TODO: Add other ordering methods

    def add_node(self, var: Variable = None, name: str = None,
                 dim: int = None) -> "FactorGraphSolver":
        """
        Add a new node
        The node has not been added to the physical or current factor graphs
        :param var:
        :param name: used only when variable is not specified
        :param dim: used only when variable is not specified
        :return: the current problem
        """
        if var:
            self._new_nodes.append(var)
        else:
            self._new_nodes.append(Variable(name, dim))
        return self

    def add_factor(self, factor: Factor) -> "FactorGraphSolver":
        """
        Add a prior factor to specified nodes
        The factor has not been added to physical or current factor graphs
        :param factor
        :return: the current problem
        """
        self._new_factors.append(factor)
        return self

    def add_prior_factor(self, vars: List[Variable],
                         distribution: dist.Distribution) -> "FactorGraphSolver":
        """
        Add a prior factor to specified nodes
        The factor has not been added to physical or current factor graphs
        :param vars
        :param distribution
        :return: the current problem
        """
        self._new_factors.append(ExplicitPriorFactor(
            vars=vars, distribution=distribution))
        return self

    def add_likelihood_factor(self, vars: List[Variable],
                              likelihood: like.LikelihoodBase) -> "FactorGraphSolver":
        """
        Add a likelihood factor to specified nodes
        The factor has not been added to physical or current factor graphs
        :param vars
        :param likelihood
        :return: the current problem
        """
        self._new_factors.append(LikelihoodFactor(
            vars=vars, log_likelihood=likelihood))
        return self

    def update_physical_and_working_graphs(self, timer: List[float] = None, device: str = "cpu"
                                           ) -> "FactorGraphSolver":
        """
        Add all new nodes and factors into the physical factor graph,
        retrieve the working factor graph, update Bayes trees
        :return: the current problem
        """
        start = time.time()

        # Determine the affected variables in the physical Bayes tree
        old_nodes = set(self.physical_vars)
        nodes_of_new_factors = set.union(*[set(factor.vars) for
                                           factor in self._new_factors])
        old_nodes_of_new_factors = set.intersection(old_nodes,
                                                    nodes_of_new_factors)

        # Get the working factor graph
        if self._physical_bayes_tree:  # if not first step, get sub graph
            affected_nodes, sub_bayes_trees = \
                self._physical_bayes_tree. \
                    get_affected_vars_and_partial_bayes_trees(
                    vars=old_nodes_of_new_factors)
            self._working_graph = self._physical_graph.get_sub_factor_graph_with_prior(
                variables=affected_nodes,
                sub_trees=sub_bayes_trees,
                clique_prior_dict=self._implicit_factors)
        else:
            sub_bayes_trees = set()
        for node in self._new_nodes:
            self._working_graph.add_node(node)
        for factor in self._new_factors:
            self._working_graph.add_factor(factor)

        # Get the working Bayes treeget_sub_factor_graph
        old_ordering = self._elimination_ordering
        self.generate_ordering()
        self._working_bayes_tree = self._working_graph.get_bayes_tree(
            ordering=[var for var in self._elimination_ordering
                      if var in set(self.working_vars)])

        # Update the physical factor graph
        for node in self._new_nodes:
            self._physical_graph.add_node(node)
        for factor in self._new_factors:
            self._physical_graph.add_factor(factor)

        # Update the physical Bayesian tree
        self._physical_bayes_tree = self._working_bayes_tree.__copy__()
        self._physical_bayes_tree.append_child_bayes_trees(sub_bayes_trees)

        # Delete legacy conditional samplers in the old tree and
        # convert the density model w/o separator at leaves to density model w/ separator.
        cliques_to_delete = set()
        for old_clique in set(self._clique_density_model.keys()).difference(self._physical_bayes_tree.clique_nodes):
            for new_clique in self._working_bayes_tree.clique_nodes:
                if old_clique.vars == new_clique.vars and [var for var in old_ordering if var in old_clique.vars] == \
                        [var for var in self._elimination_ordering if var in new_clique.vars]:
                    # This clique was the root in the old tree but is leaf in the new tree.
                    # If the ordering of variables remains the same, its density model can be re-used.
                    # Update the clique to density model dict
                    self._clique_true_obs[new_clique] = self._clique_true_obs[old_clique]
                    if old_clique in self._clique_variable_pattern:
                        self._clique_variable_pattern[new_clique] = self._clique_variable_pattern[old_clique]
                    if old_clique in self._clique_samples:
                        self._clique_samples[new_clique] = self._clique_samples[old_clique]

                    self._clique_density_model[new_clique] = \
                        self.root_clique_density_model_to_leaf(old_clique, new_clique, device)

                    # since new clique will be skipped, related factors shall be eliminated beforehand.
                    # TODO: update _clique_density_model.keys() in which some clique parents change
                    # TODO: this currently has no impact on results
                    # TODO: if we store all models or clique-depend values on cliques, this issue will disappear
                    new_separator_factor = None
                    if new_clique.separator:
                        # extract new factor over separator
                        separator_var_list = sorted(new_clique.separator, key=lambda x: self._reverse_ordering_map[x])
                        new_separator_factor = \
                            self.clique_density_to_separator_factor(separator_var_list,
                                                                    self._clique_density_model[new_clique],
                                                                    self._clique_true_obs[old_clique])
                        self._implicit_factors[new_clique] = new_separator_factor
                    self._working_graph = self._working_graph.eliminate_clique_variables(clique=new_clique,
                                                                                         new_factor=new_separator_factor)
                    break
            cliques_to_delete.add(old_clique)

        for old_clique in cliques_to_delete:
            del self._clique_density_model[old_clique]
            del self._clique_true_obs[old_clique]
            if old_clique in self._clique_variable_pattern:
                del self._clique_variable_pattern[old_clique]
            if old_clique in self._clique_samples:
                del self._clique_samples[old_clique]

        # Clear all newly added variables and factors
        self._new_nodes = []
        self._new_factors = []

        end = time.time()
        if timer is not None:
            timer.append(end - start)
        return self

    def root_clique_density_model_to_leaf(self,
                                          old_clique: BayesTreeNode,
                                          new_clique: BayesTreeNode,
                                          device) -> "ConditionalSampler":
        """
        when old clique and new clique have same variables but different division of frontal and separator vars,
        recycle the density model in the old clique and convert it to that in the new clique.
        """
        raise NotImplementedError("Implementation depends on probabilistic modeling")

    def clique_density_to_separator_factor(self,
                                           separator_var_list: List[Variable],
                                           density_model,
                                           true_obs: np.ndarray) -> CliqueSeparatorFactor:
        """
        extract marginal of separator variables from clique density as separator factor
        """
        raise NotImplementedError("Implementation depends on probabilistic modeling")

    def incremental_inference(self,
                              timer: List[float] = None,
                              clique_dim_timer: List[List[float]] = None,
                              *args, **kwargs
                              ):

        self.fit_tree_density_models(timer=timer,
                                     clique_dim_timer=clique_dim_timer,
                                     *args, **kwargs)
        if self._args.adaptive_posterior_sampling is None:
            self._samples = self.sample_posterior(timer=timer, *args, **kwargs)
        else:
            self._samples = self.adaptive_posterior(timer=timer, *args, **kwargs)
        return self._samples

    def fit_clique_density_model(self,
                                 clique,
                                 samples,
                                 var_ordering,
                                 timer,
                                 *args, **kwargs) -> "ConditionalSampler":
        raise NotImplementedError("Implementation depends on probabilistic modeling.")

    def adaptive_posterior(self, timer: List[float] = None, *args, **kwargs
                           ) -> Dict[Variable, np.ndarray]:
        """
        Generate samples for all variables
        """
        raise NotADirectoryError("implementation depends on density models.")

    def fit_tree_density_models(self,
                                timer: List[float] = None,
                                clique_dim_timer: List[List[float]] = None,
                                *args, **kwargs):
        """
        By the order of Bayes tree, perform local sampling and training
        on all cliques
        :return:
        """
        self._temp_training_loss = {}
        clique_ordering = self._working_bayes_tree.clique_ordering()
        total_clique_num = len(clique_ordering)
        clique_cnt = 1
        before_clique_time = time.time()
        while clique_ordering:
            start_clique_time = time.time()
            clique = clique_ordering.pop()
            if clique in self._clique_density_model:
                end_clique_time = time.time()
                print(f"\tTime for clique {clique_cnt}/{total_clique_num}: " + str(
                    end_clique_time - start_clique_time) + " sec, "
                                                           "total time elapsed: " + str(
                    end_clique_time - before_clique_time) + " sec")
                clique_cnt += 1
                if (clique_dim_timer is not None):
                    clique_dim_timer.append([clique.dim, end_clique_time - before_clique_time])
                continue

            # local sampling
            sampler_start = time.time()
            local_samples, sample_var_ordering, true_obs = \
                self.clique_training_sampler(clique,
                                             num_samples=self._args.local_sample_num,
                                             method=self._args.local_sampling_method)
            sampler_end = time.time()
            if timer is not None:
                timer.append(sampler_end - sampler_start)

            self._clique_true_obs[clique] = true_obs
            if self._args.store_clique_samples:
                self._clique_samples[clique] = local_samples

            local_density_model = \
                self.fit_clique_density_model(clique=clique,
                                              samples=local_samples,
                                              var_ordering=sample_var_ordering,
                                              timer=timer)
            self._clique_density_model[clique] = local_density_model
            new_separator_factor = None
            if clique.separator:
                # extract new factor over separator
                separator_list = sorted(clique.separator,
                                        key=lambda x:
                                        self._reverse_ordering_map[x])
                new_separator_factor = self.clique_density_to_separator_factor(separator_list,
                                                                               local_density_model,
                                                                               true_obs)
                self._implicit_factors[clique] = new_separator_factor
            self._working_graph = self._working_graph.eliminate_clique_variables(clique=clique,
                                                                                 new_factor=new_separator_factor)

            end_clique_time = time.time()
            print(f"\tTime for clique {clique_cnt}/{total_clique_num}: " + str(
                end_clique_time - start_clique_time) + " sec, "
                                                       "total time elapsed: " + str(
                end_clique_time - before_clique_time) + " sec" + ", clique_dim is " + str(clique.dim))
            if (clique_dim_timer is not None):
                clique_dim_timer.append([clique.dim, end_clique_time - before_clique_time])
            clique_cnt += 1

    def clique_training_sampler(self, clique: BayesTreeNode, num_samples: int, method: str):
        r""" This function returns training samples, simulated variables, and unused observations
        """
        graph = self._working_graph.get_clique_factor_graph(clique)
        variable_pattern = \
            self._working_bayes_tree.clique_variable_pattern(clique)
        if method == "direct":
            sampler = SimulationBasedSampler(factors=graph.factors, vars=variable_pattern)
            samples, var_list, unused_obs = sampler.sample(num_samples)
        elif method == "nested" or method == "dynamic nested":
            ns_sampler = GlobalNestedSampler(nodes=variable_pattern, factors=graph.factors)
            samples = ns_sampler.sample(live_points=num_samples, sampling_method=method)
            var_list = variable_pattern
            unused_obs = np.array([])
        else:
            raise ValueError("Unknown sampling method.")
        return samples, var_list, unused_obs

    def sample_posterior(self, timer: List[float] = None, *args, **kwargs
                         ) -> Dict[Variable, np.ndarray]:
        """
        Generate samples for all variables
        """
        num_samples = self._args.posterior_sample_num
        start = time.time()

        stack = [self._physical_bayes_tree.root]
        samples = {}

        while stack:
            # Retrieve the working clique
            clique = stack.pop()
            # Local sampling
            frontal_list = sorted(clique.frontal,
                                  key=lambda x: self._reverse_ordering_map[x])
            separator_list = sorted(clique.separator,
                                    key=lambda x: self._reverse_ordering_map[x])
            clique_density_model = self._clique_density_model[clique]

            obs = self._clique_true_obs[clique]

            aug_separator_samples = np.zeros(shape=(num_samples, 0))

            if len(obs) != 0:
                aug_separator_samples = np.tile(obs, (num_samples, 1))
            for var in separator_list:
                aug_separator_samples = np.hstack((aug_separator_samples,
                                                   samples[var]))

            if aug_separator_samples.shape[1] != 0:
                frontal_samples = clique_density_model. \
                    conditional_sample_given_observation(conditional_dim=clique.frontal_dim,
                                                         obs_samples=aug_separator_samples)
            else:  # the root clique
                frontal_samples = clique_density_model. \
                    conditional_sample_given_observation(conditional_dim=clique.frontal_dim,
                                                         sample_number=num_samples)
            # Dispatch samples
            cur_index = 0
            for var in frontal_list:
                samples[var] = frontal_samples[:,
                               cur_index: cur_index + var.dim]
                cur_index += var.dim
            if clique.children:
                for child in clique.children:
                    stack.append(child)

        end = time.time()
        if timer is not None:
            timer.append(end - start)

        return samples

    def plot2d_posterior(self, title: str = None, xlim=None, ylim=None,
                         marker_size: float = 1, if_legend: bool = False):
        # xlim and ylim are tuples
        vars = self._elimination_ordering
        # list(self._samples.keys())
        len_var = len(vars)

        for i in range(len_var):
            cur_sample = self._samples[vars[i]]
            plt.scatter(cur_sample[:, 0], cur_sample[:, 1], marker=".",
                        s=marker_size)
            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)
        if if_legend:
            plt.legend([var.name for var in vars])
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        if title is not None:
            plt.title(title)
        fig_handle = plt.gcf()
        plt.show()
        return fig_handle

    def results(self):
        return list(self._samples.values()), list(self._samples.keys())

    def plot2d_mean_points(self, title: str = None, xlim=None, ylim=None,
                           if_legend: bool = False):
        # xlim and ylim are tuples
        vars = self._elimination_ordering
        # list(self._samples.keys())
        len_var = len(vars)
        x_list = []
        y_list = []
        for i in range(len_var):
            cur_sample = self._samples[vars[i]]
            x = np.mean(cur_sample[:, 0])
            y = np.mean(cur_sample[:, 1])
            x_list.append(x)
            y_list.append(y)
            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)
        plt.plot(x_list, y_list)
        if if_legend:
            plt.legend([var.name for var in vars])
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        if title is not None:
            plt.title(title)
        fig_handle = plt.gcf()
        plt.show()
        return fig_handle

    def plot2d_mean_rbt_only(self, title: str = None, xlim=None, ylim=None,
                             if_legend: bool = False, fname=None, front_size=None, show_plot=False, **kwargs):
        # xlim and ylim are tuples
        vars = self._elimination_ordering
        # list(self._samples.keys())
        len_var = len(vars)
        x_list = []
        y_list = []
        lmk_list = []
        for i in range(len_var):
            if vars[i]._type == VariableType.Landmark:
                lmk_list.append(vars[i])
            else:
                cur_sample = self._samples[vars[i]]
                x = np.mean(cur_sample[:, 0])
                y = np.mean(cur_sample[:, 1])
                x_list.append(x)
                y_list.append(y)
                if xlim is not None:
                    plt.xlim(xlim)
                if ylim is not None:
                    plt.ylim(ylim)
        plt.plot(x_list, y_list)
        for var in lmk_list:
            cur_sample = self._samples[var]
            plt.scatter(cur_sample[:, 0], cur_sample[:, 1], label=var.name)
        if if_legend:
            if front_size is not None:
                plt.legend()
            else:
                plt.legend(fontsize=front_size)

        if front_size is not None:
            plt.xlabel('x (m)', fontsize=front_size)
            plt.ylabel('y (m)', fontsize=front_size)
        else:
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')

        if title is not None:
            if front_size is not None:
                plt.title(title, fontsize=front_size)
            else:
                plt.title(title)
        fig_handle = plt.gcf()
        if fname is not None:
            plt.savefig(fname)
        if show_plot:
            plt.show()
        return fig_handle

    def plot2d_MAP_rbt_only(self, title: str = None, xlim=None, ylim=None,
                             if_legend: bool = False, fname=None, front_size=None):
        # xlim and ylim are tuples
        vars = self._elimination_ordering
        jf = JointFactor(self.physical_factors, vars)
        # list(self._samples.keys())
        all_sample = sample_dict_to_array(self._samples, vars)
        log_pdf = jf.log_pdf(all_sample)
        
        max_idx = np.argmax(log_pdf)

        map_sample = all_sample[max_idx:max_idx+1]

        map_sample_dict = array_order_to_dict(map_sample, vars)
        len_var = len(vars)
        x_list = []
        y_list = []
        lmk_list = []
        for i in range(len_var):
            if vars[i]._type == VariableType.Landmark:
                lmk_list.append(vars[i])
            else:
                cur_sample = map_sample_dict[vars[i]]
                x = np.mean(cur_sample[:, 0])
                y = np.mean(cur_sample[:, 1])
                x_list.append(x)
                y_list.append(y)
                if xlim is not None:
                    plt.xlim(xlim)
                if ylim is not None:
                    plt.ylim(ylim)
        plt.plot(x_list, y_list)
        for var in lmk_list:
            cur_sample = map_sample_dict[var]
            plt.scatter(cur_sample[:, 0], cur_sample[:, 1], label=var.name)
        if if_legend:
            if front_size is not None:
                plt.legend()
            else:
                plt.legend(fontsize=front_size)

        if front_size is not None:
            plt.xlabel('x (m)', fontsize=front_size)
            plt.ylabel('y (m)', fontsize=front_size)
        else:
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')

        if title is not None:
            if front_size is not None:
                plt.title(title, fontsize=front_size)
            else:
                plt.title(title)
        fig_handle = plt.gcf()
        if fname is not None:
            plt.savefig(fname)
        plt.show()
        return fig_handle

    def plot2d_mean_poses(self, title: str = None, xlim=None, ylim=None,
                          width: float = 0.05, if_legend: bool = False):
        # xlim and ylim are tuples
        vars = self._elimination_ordering
        # list(self._samples.keys())
        len_var = len(vars)
        x_list = []
        y_list = []
        for i in range(len_var):
            cur_sample = self._samples[vars[i]]
            x = np.mean(cur_sample[:, 0])
            y = np.mean(cur_sample[:, 1])
            x_list.append(x)
            y_list.append(y)
            # th_mean = circmean(cur_sample[:,2])
            # dx, dy = np.cos(th_mean), np.sin(th_mean)
            # plt.arrow(x-dx/2, y-dy/2, dx, dy,
            #           head_width=4*width,
            #           width=0.05)
            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)
        plt.plot(x_list, y_list)
        if if_legend:
            plt.legend([var.name for var in vars])
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        if title is not None:
            plt.title(title)
        fig_handle = plt.gcf()
        plt.show()
        return fig_handle

    def plot_factor_graph(self):
        pass

    def plot_bayes_tree(self):
        pass


def run_incrementally(case_dir: str, solver: FactorGraphSolver, nodes_factors_by_step, truth=None, traj_plot=False,
                      plot_args=None, check_root_transform=False) -> None:
    run_count = 1
    while os.path.exists(f"{case_dir}/run{run_count}"):
        run_count += 1
    os.mkdir(f"{case_dir}/run{run_count}")
    run_dir = f"{case_dir}/run{run_count}"
    print("create run dir: " + run_dir)

    file = open(f"{run_dir}/parameters", "w+")
    params = solver._args.jsonStr()
    print(params)
    file.write(params)
    file.close()

    num_batches = len(nodes_factors_by_step)
    observed_nodes = []
    step_timer = []
    step_list = []

    posterior_sampling_timer = []
    fitting_timer = []

    mixture_factor2weights = {}

    show_plot = True
    if "show_plot" in plot_args and not plot_args["show_plot"]:
        show_plot = False

    for i in range(num_batches):
        step_nodes, step_factors = nodes_factors_by_step[i]
        for node in step_nodes:
            solver.add_node(node)
        for factor in step_factors:
            solver.add_factor(factor)
            if isinstance(factor, BinaryFactorMixture):
                mixture_factor2weights[factor] = []
        observed_nodes += step_nodes

        step_list.append(i)
        step_file_prefix = f"{run_dir}/step{i}"
        detailed_timer = []
        clique_dim_timer = []
        start = time.time()
        solver.update_physical_and_working_graphs(timer=detailed_timer)
        cur_sample = solver.incremental_inference(timer=detailed_timer, clique_dim_timer=clique_dim_timer)
        end = time.time()

        step_timer.append(end - start)
        print(f"step {i}/{num_batches} time: {step_timer[-1]} sec, "
              f"total time: {sum(step_timer)}")

        file = open(f"{step_file_prefix}_ordering", "w+")
        file.write(" ".join([var.name for var in solver.elimination_ordering]))
        file.close()

        file = open(f"{step_file_prefix}_split_timing", "w+")
        file.write(" ".join([str(t) for t in detailed_timer]))
        file.close()

        file = open(f"{step_file_prefix}_step_training_loss", "w+")
        last_training_loss = json.dumps(solver._temp_training_loss)
        file.write(last_training_loss)
        file.close()

        posterior_sampling_timer.append(detailed_timer[-1])
        fitting_timer.append(sum(detailed_timer[1:-1]))

        X = np.hstack([cur_sample[var] for var in solver.elimination_ordering])
        np.savetxt(fname=step_file_prefix, X=X)

        # check transformation
        if check_root_transform:
            root_clique = solver.physical_bayes_tree.root
            root_clique_model = solver._clique_density_model[root_clique]
            y = root_clique_model.prior.sample((3000,))
            tx = deepcopy(y)
            if hasattr(root_clique_model, "flows"):
                for f in root_clique_model.flows[::-1]:
                    tx = f.inverse_given_separator(tx, None)
            y = y.detach().numpy()
            tx = tx.detach().numpy()
            np.savetxt(fname=step_file_prefix + '_root_normal_data', X=y)
            np.savetxt(fname=step_file_prefix + '_root_transformed', X=tx)

            plt.figure()
            x_sort, tx_sort = sort_pair_lists(tx[:,0], y[:,0])
            plt.plot(x_sort, tx_sort)
            plt.ylabel("T(x)")
            plt.xlabel("x")
            plt.savefig(f"{step_file_prefix}_transform.png", bbox_inches="tight")
            if show_plot: plt.show()
            plt.close()

        # clique dim and timing
        np.savetxt(fname=step_file_prefix + '_dim_time', X=np.array(clique_dim_timer))

        if traj_plot:
            plot_2d_samples(samples_mapping=cur_sample,
                            equal_axis=True,
                            truth={variable: pose for variable, pose in
                                   truth.items() if variable in solver.physical_vars},
                            truth_factors={factor for factor in solver.physical_factors if
                                           set(factor.vars).issubset(solver.physical_vars)},
                            title=f'Step {i}',
                            plot_all_meas=False,
                            plot_meas_give_pose=[var for var in step_nodes if var.type == VariableType.Pose],
                            rbt_traj_no_samples=True,
                            truth_R2=True,
                            truth_SE2=False,
                            truth_odometry_color='k',
                            truth_landmark_markersize=10,
                            truth_landmark_marker='x',
                            file_name=f"{step_file_prefix}.png",
                            **plot_args)
        else:
            plot_2d_samples(samples_mapping=cur_sample,
                            equal_axis=True,
                            truth={variable: pose for variable, pose in
                                   truth.items() if variable in solver.physical_vars},
                            truth_factors={factor for factor in solver.physical_factors if
                                           set(factor.vars).issubset(solver.physical_vars)},
                            file_name=f"{step_file_prefix}.png", title=f'Step {i}',
                            **plot_args)
            solver.plot2d_mean_rbt_only(title=f"step {i} posterior", if_legend=False, fname=f"{step_file_prefix}.png", **plot_args)
            # solver.plot2d_MAP_rbt_only(title=f"step {i} posterior", if_legend=False, fname=f"{step_file_prefix}.png")

        file = open(f"{run_dir}/step_timing", "w+")
        file.write(" ".join(str(t) for t in step_timer))
        file.close()
        file = open(f"{run_dir}/step_list", "w+")
        file.write(" ".join(str(s) for s in step_list))
        file.close()

        file = open(f"{run_dir}/posterior_sampling_timer", "w+")
        file.write(" ".join(str(t) for t in posterior_sampling_timer))
        file.close()

        file = open(f"{run_dir}/fitting_timer", "w+")
        file.write(" ".join(str(t) for t in fitting_timer))
        file.close()

        plt.figure()
        plt.plot(np.array(step_list)*5+5, step_timer, 'go-', label='Total')
        plt.plot(np.array(step_list)*5+5, posterior_sampling_timer, 'ro-', label='Posterior sampling')
        plt.plot(np.array(step_list)*5+5, fitting_timer, 'bd-', label='Learning NF')
        plt.ylabel(f"Time (sec)")
        plt.xlabel(f"Key poses")
        plt.legend()
        plt.savefig(f"{run_dir}/step_timing.png", bbox_inches="tight")
        if show_plot: plt.show()
        plt.close()

        if mixture_factor2weights:
            # write updated hypothesis weights
            hypo_file = open(run_dir + f'/step{i}.hypoweights', 'w+')
            plt.figure()
            for factor, weights in mixture_factor2weights.items():
                hypo_weights = factor.posterior_weights(cur_sample)
                line = ' '.join([var.name for var in factor.vars]) + ' : ' + ','.join(
                    [str(w) for w in hypo_weights])
                hypo_file.writelines(line + '\n')
                weights.append(hypo_weights)
                for i_w in range(len(hypo_weights)):
                    plt.plot(np.arange(i + 1 - len(weights), i + 1), np.array(weights)[:, i_w], '-o',
                             label=f"H{i_w}at{factor.observer_var.name}" if not isinstance(factor, KWayFactor) else
                             f"{factor.observer_var.name} to {factor.observed_vars[i_w].name}")
            hypo_file.close()
            plt.legend()
            plt.xlabel('Step')
            plt.ylabel('Hypothesis weights')
            plt.savefig(run_dir + f'/step{i}_hypoweights.png', dpi=300)
            if show_plot: plt.show()
            plt.close()
