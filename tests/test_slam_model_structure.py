import unittest
import numpy as np
from slam.Variables import Variable
from factors.Factors import ExplicitPriorFactor, LikelihoodFactor
from statistics.Distributions import \
    GaussianMixtureDistribution
import TransportMaps.Likelihoods as like
from slam.NFiSAM import NFiSAM


class TestStructure(unittest.TestCase):
    def test_1(self):
        node_x0 = Variable('X0', 2)
        node_x1 = Variable('X1', 2)
        node_x2 = Variable('X2', 2)
        node_l1 = Variable('L1', 2)
        node_l2 = Variable('L2', 2)
        disp_x0_l1 = np.array([5, 5])
        disp_l1_x1 = np.array([0, -10])
        disp_x0_x1 = np.array([5, -5])
        disp_x1_x2 = np.array([5, 5])
        disp_l2_x2 = np.array([0, -5])
        minus_mat = np.array([[-1.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]])
        prior_factor_l1 = ExplicitPriorFactor(
            vars=[node_l1],
            distribution=GaussianMixtureDistribution(
                weights=[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                means=[np.array([3.0, 3.0]), np.array([7.0, 3.0]),
                       np.array([5.0, 5.0])],
                sigmas=[np.identity(2) * 0.5, np.identity(2) * 0.5,
                        np.identity(2) * 0.5]))
        prior_factor_l2 = ExplicitPriorFactor(
            vars=[node_l2],
            distribution=GaussianMixtureDistribution(
                weights=[0.5, 0.5],
                means=[np.array([10.0, 5.0]), np.array([13, 7.0])],
                sigmas=[np.identity(2) * 0.5, np.identity(2) * 0.5]))
        llfactor_x0_l1 = LikelihoodFactor(
            vars=[node_x0, node_l1],
            log_likelihood=like.AdditiveLinearGaussianLogLikelihood(
                y=disp_x0_l1, c=np.zeros(2), mu=np.zeros(2),
                precision=np.array([[100, 0.0], [0.0, 100]]), T=minus_mat))
        llfactor_l1_x1 = LikelihoodFactor(
            vars=[node_l1, node_x1],
            log_likelihood=like.AdditiveLinearGaussianLogLikelihood(
                y=disp_l1_x1, c=np.zeros(2), mu=np.zeros(2),
                precision=np.array([[10000.0, 0.0], [0.0, 100]]), T=minus_mat))
        llfactor_x0_x1 = LikelihoodFactor(
            vars=[node_x0, node_x1],
            log_likelihood=like.AdditiveLinearGaussianLogLikelihood(
                y=disp_x0_x1, c=np.zeros(2), mu=np.zeros(2),
                precision=np.array([[100, 0], [0, 100]]), T=minus_mat))
        llfactor_x1_x2 = LikelihoodFactor(
            vars=[node_x1, node_x2],
            log_likelihood=like.AdditiveLinearGaussianLogLikelihood(
                y=disp_x1_x2, c=np.zeros(2), mu=np.zeros(2),
                precision=np.array([[100.0, 0], [0, 100.0]]), T=minus_mat))
        llfactor_l2_x2 = LikelihoodFactor(
            vars=[node_l2, node_x2],
            log_likelihood=like.AdditiveLinearGaussianLogLikelihood(
                y=disp_l2_x2, c=np.zeros(2), mu=np.zeros(2),
                precision=np.array([[10000.0, 0], [0, 100.0]]), T=minus_mat))

        model = NFiSAM()

        model.add_node(node_l1)
        model.add_node(node_l2)
        model.add_node(node_x0)
        model.add_node(node_x1)
        model.add_node(node_x2)

        model.add_factor(prior_factor_l1)
        model.add_factor(prior_factor_l2)
        model.add_factor(llfactor_x0_l1)
        model.add_factor(llfactor_x0_x1)
        model.add_factor(llfactor_x1_x2)
        model.add_factor(llfactor_l1_x1)
        model.add_factor(llfactor_l2_x2)

        model.update_physical_and_working_graphs()


if __name__ == '__main__':
    unittest.main()
