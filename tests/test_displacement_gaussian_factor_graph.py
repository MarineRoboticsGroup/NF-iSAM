import unittest
import numpy as np
import TransportMaps.Distributions as dist
import TransportMaps.Likelihoods as like
from slam.FactorGraph import FactorGraph
from factors.Factors import ExplicitPriorFactor, LikelihoodFactor
from utils.Statistics import \
    gaussian_displacement_factor_graph_with_equal_dim, mmd
from typing import Tuple, List, Dict
from slam.Variables import Variable
from TransportMaps.Distributions import GaussianDistribution
from slam.NFiSAM import NFiSAM


def verify_displacement_gaussian_factor_graph(variables: List[Variable],
        displacements: Dict[Tuple[Variable, Variable],
                            Tuple[np.ndarray, np.ndarray]],
        priors: Dict[Variable, Tuple[np.ndarray, np.ndarray]],
        num_samples: int) -> float:
    """
    Generate samples from the computed SLAM model and the analytic Gaussian
        graphical model, compare the results
    :param variables: variables in the distribution
    :param displacements: the displacements from variable to variable
        (variable_from, variable_to): (mean, sigma)
    :param priors: the priors of variables
        variable: (mean, sigma)
    :param num_samples
    :return: the MMD
    """
    # Generate analytic samples
    analytic_mean, analytic_cov = gaussian_displacement_factor_graph_with_equal_dim(
        variables, displacements, priors)
    analytic_model = GaussianDistribution(mu=analytic_mean, sigma=analytic_cov)
    samples_analytic = analytic_model.rvs(num_samples)
    dim_tot = analytic_mean.shape[0]

    # Generate computed samples
    model = NFiSAM()
    for var in variables:
        model.add_node(var)

    minus_mat = np.array([[-1.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]])
    for displacement in displacements:
        pass

    # Comparison
    return mmd(samples_analytic, samples_computed)


class GaussianFactorGraphTestCase1(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the following 2D Gaussian example:
            x2 -- (1, 1) -- x0 -- (0, -1) -- x3 -- (2, 1) -- x1
                -- (-2, -1) -- x4
        """
        cls.graph = FactorGraph()
        cls.graph.add_node(0, 2)
        cls.graph.add_node(1, 2)
        cls.graph.add_node(2, 2)
        cls.graph.add_node(3, 2)
        cls.graph.add_node(4, 2)
        cls.displacement20 = np.array([1.0, 1.0])
        cls.displacement03 = np.array([0.0, -1.0])
        cls.displacement31 = np.array([2.0, 1.0])
        cls.displacement14 = np.array([-2.0, -1.0])
        cls.factor2 = ExplicitPriorFactor(
            var_dim=[(2, 2)], distribution=dist.StandardNormalDistribution(2))
        minus_mat = np.array([[-1.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]])
        cls.factor20 = LikelihoodFactor(
            var_dim=[(2, 2), (0, 2)],
            log_likelihood=like.AdditiveLinearGaussianLogLikelihood(
                y=cls.displacement20, c=np.zeros(2), mu=np.zeros(2),
                precision=np.array([[1.0, 0.0], [0.0, 1.0]]), T=minus_mat))
        cls.factor03 = LikelihoodFactor(
            var_dim=[(0, 2), (3, 2)],
            log_likelihood=like.AdditiveLinearGaussianLogLikelihood(
                y=cls.displacement03, c=np.zeros(2), mu=np.zeros(2),
                precision=np.array([[0.25, 0.0], [0.0, 0.36]]), T=minus_mat))
        cls.factor31 = LikelihoodFactor(
            var_dim=[(3, 2), (1, 2)],
            log_likelihood=like.AdditiveLinearGaussianLogLikelihood(
                y=cls.displacement31, c=np.zeros(2), mu=np.zeros(2),
                precision=np.array([[0.81, 0.16], [0.16, 0.49]]), T=minus_mat))
        cls.factor14 = LikelihoodFactor(
            var_dim=[(1, 2), (4, 2)],
            log_likelihood=like.AdditiveLinearGaussianLogLikelihood(
                y=cls.displacement14, c=np.zeros(2), mu=np.zeros(2),
                precision=np.array([[1.69, 1.0], [1.0, 2.25]]), T=minus_mat))
        cls.graph.add_factor(cls.factor2)
        cls.graph.add_factor(cls.factor20)
        cls.graph.add_factor(cls.factor03)
        cls.graph.add_factor(cls.factor31)
        cls.graph.add_factor(cls.factor14)
        cls.bayes_tree = cls.graph.get_bayes_tree()
        cls.clique_ordering = []
        queue = [cls.bayes_tree.root]
        while queue:
            clique = queue.pop(0)
            cls.clique_ordering.append(clique)
            if clique.children:
                for child in clique.children:
                    queue.append(child)

    def test_clique_1(self) -> None:
        clique = self.clique_ordering.pop()
        self.graph.local_inference(clique, order=1, direct_quad_type=0, direct_quad_num=1000)

        # dim = 4
        # factor = JointFactor([self.factor2, self.factor20], [2, 0])
        # print(factor.log_pdf(np.array([[1,2,3,4],[-1,-2,-3,4]])))
        # map = tm.Default_IsotropicIntegratedSquaredTriangularTransportMap(dim=dim, order=1)
        # push = dist.PushForwardTransportMapDistribution(map, dist.StandardNormalDistribution(dim))
        # push.minimize_kl_divergence(tar=factor, qtype=0, qparams=1000)


        # self.graph.solve_for_global_transport_map()
        # self.assertSetEqual({1}, {1})


if __name__ == '__main__':
    unittest.main()
