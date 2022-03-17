import unittest
import numpy as np
import TransportMaps.Distributions as dist
import TransportMaps.Likelihoods as like
from slam.FactorGraph import FactorGraph
from factors.Factors import ExplicitPriorFactor, LikelihoodFactor
from slam.Variables import Variable


class BeysTreeConstructionTestCase1(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the following 2D Gaussian example:
            x2 -- (1, 1) -- x0 -- (0, -1) -- x3 -- (2, 1) -- x1
                -- (-2, -1) -- x4
        """
        self.node0 = Variable("L1", 2)
        self.node1 = Variable("L2", 2)
        self.node2 = Variable("X0", 2)
        self.node3 = Variable("X1", 2)
        self.node4 = Variable("X2", 2)
        self.graph = FactorGraph()
        self.graph.add_node(self.node0)
        self.graph.add_node(self.node1)
        self.graph.add_node(self.node2)
        self.graph.add_node(self.node3)
        self.graph.add_node(self.node4)
        self.displacement20 = np.array([1.0, 1.0])
        self.displacement03 = np.array([0.0, -1.0])
        self.displacement31 = np.array([2.0, 1.0])
        self.displacement14 = np.array([-2.0, -1.0])
        self.factor2 = ExplicitPriorFactor(
            vars=[self.node2], distribution=dist.StandardNormalDistribution(2))
        minus_mat = np.array([[-1.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]])
        self.factor20 = LikelihoodFactor(
            vars=[self.node2, self.node0],
            log_likelihood=like.AdditiveLinearGaussianLogLikelihood(
                y=self.displacement20, c=np.zeros(2), mu=np.zeros(2),
                precision=np.array([[1.0, 0.0], [0.0, 1.0]]), T=minus_mat))
        self.factor03 = LikelihoodFactor(
            vars=[self.node0, self.node3],
            log_likelihood=like.AdditiveLinearGaussianLogLikelihood(
                y=self.displacement03, c=np.zeros(2), mu=np.zeros(2),
                precision=np.array([[0.25, 0.0], [0.0, 0.36]]), T=minus_mat))
        self.factor31 = LikelihoodFactor(
            vars=[self.node3, self.node1],
            log_likelihood=like.AdditiveLinearGaussianLogLikelihood(
                y=self.displacement31, c=np.zeros(2), mu=np.zeros(2),
                precision=np.array([[0.81, 0.16], [0.16, 0.49]]), T=minus_mat))
        self.factor14 = LikelihoodFactor(
            vars=[self.node1, self.node4],
            log_likelihood=like.AdditiveLinearGaussianLogLikelihood(
                y=self.displacement14, c=np.zeros(2), mu=np.zeros(2),
                precision=np.array([[1.69, 1.0], [1.0, 2.25]]), T=minus_mat))
        self.graph.add_factor(self.factor2)
        self.graph.add_factor(self.factor20)
        self.graph.add_factor(self.factor03)
        self.graph.add_factor(self.factor31)
        self.graph.add_factor(self.factor14)

    def test_before_elimination(self) -> None:
        graph = self.graph
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node0),
                            {self.node2, self.node3})
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node1),
                            {self.node3, self.node4})
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node2),
                            {self.node0})
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node3),
                            {self.node0, self.node1})
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node4),
                            {self.node1})
        self.assertSetEqual(graph.get_adjacent_factors_from_node(self.node0),
                            {self.factor20, self.factor03})
        self.assertSetEqual(graph.get_adjacent_factors_from_node(self.node1),
                            {self.factor31, self.factor14})
        self.assertSetEqual(graph.get_adjacent_factors_from_node(self.node2),
                            {self.factor2, self.factor20})
        self.assertSetEqual(graph.get_adjacent_factors_from_node(self.node3),
                            {self.factor03, self.factor31})
        self.assertSetEqual(graph.get_adjacent_factors_from_node(self.node4),
                            {self.factor14})
        self.assertSetEqual(graph.get_adjacent_nodes_from_factor(self.factor2),
                            {self.node2})
        self.assertSetEqual(graph.get_adjacent_nodes_from_factor(self.factor20),
                            {self.node2, self.node0})
        self.assertSetEqual(graph.get_adjacent_nodes_from_factor(self.factor03),
                            {self.node0, self.node3})
        self.assertSetEqual(graph.get_adjacent_nodes_from_factor(self.factor31),
                            {self.node3, self.node1})
        self.assertSetEqual(graph.get_adjacent_nodes_from_factor(self.factor14),
                            {self.node1, self.node4})
        self.assertRaises(KeyError,
                          lambda: graph.get_parents_in_bayesian_network(
                              self.node0))
        self.assertRaises(KeyError,
                          lambda: graph.get_parents_in_bayesian_network(
                              self.node1))
        self.assertRaises(KeyError,
                          lambda: graph.get_parents_in_bayesian_network(
                              self.node2))
        self.assertRaises(KeyError,
                          lambda: graph.get_parents_in_bayesian_network(
                              self.node3))
        self.assertRaises(KeyError,
                          lambda: graph.get_parents_in_bayesian_network(
                              self.node4))

    def test_elimination_step_1(self) -> None:
        graph = self.graph
        graph.eliminate_from_factor_graph_for_analysis(self.node0)
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node0),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node1),
                            {self.node3, self.node4})
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node2),
                            {self.node3})
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node3),
                            {self.node1, self.node2})
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node4),
                            {self.node1})
        self.assertSetEqual(graph.get_adjacent_factors_from_node(self.node1),
                            {self.factor31, self.factor14})
        self.assertSetEqual(graph.get_adjacent_factors_from_node(self.node4),
                            {self.factor14})
        self.assertSetEqual(graph.get_adjacent_nodes_from_factor(self.factor2),
                            {self.node2})
        self.assertSetEqual(graph.get_adjacent_nodes_from_factor(self.factor31),
                            {self.node3, self.node1})
        self.assertSetEqual(graph.get_adjacent_nodes_from_factor(self.factor14),
                            {self.node1, self.node4})
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node0),
                            {self.node2, self.node3})
        self.assertRaises(KeyError,
                          lambda: graph.get_parents_in_bayesian_network(
                              self.node1))
        self.assertRaises(KeyError,
                          lambda: graph.get_parents_in_bayesian_network(
                              self.node2))
        self.assertRaises(KeyError,
                          lambda: graph.get_parents_in_bayesian_network(
                              self.node3))
        self.assertRaises(KeyError,
                          lambda: graph.get_parents_in_bayesian_network(
                              self.node4))

    def test_elimination_step_2(self) -> None:
        graph = self.graph
        graph.eliminate_from_factor_graph_for_analysis(self.node0)
        graph.eliminate_from_factor_graph_for_analysis(self.node1)
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node0),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node1),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node2),
                            {self.node3})
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node3),
                            {self.node2, self.node4})
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node4),
                            {self.node3})
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node0),
                            {self.node2, self.node3})
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node0),
                            {self.node2, self.node3})
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node1),
                            {self.node3, self.node4})
        self.assertRaises(KeyError,
                          lambda: graph.get_parents_in_bayesian_network(
                              self.node2))
        self.assertRaises(KeyError,
                          lambda: graph.get_parents_in_bayesian_network(
                              self.node3))
        self.assertRaises(KeyError,
                          lambda: graph.get_parents_in_bayesian_network(
                              self.node4))

    def test_elimination_step_3(self) -> None:
        graph = self.graph
        graph.eliminate_from_factor_graph_for_analysis(self.node0)
        graph.eliminate_from_factor_graph_for_analysis(self.node1)
        graph.eliminate_from_factor_graph_for_analysis(self.node2)
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node0),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node1),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node2),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node3),
                            {self.node4})
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node4),
                            {self.node3})
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node0),
                            {self.node2, self.node3})
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node1),
                            {self.node3, self.node4})
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node2),
                            {self.node3})
        self.assertRaises(KeyError,
                          lambda: graph.get_parents_in_bayesian_network(
                              self.node3))
        self.assertRaises(KeyError,
                          lambda: graph.get_parents_in_bayesian_network(
                              self.node4))

    def test_elimination_step_4(self) -> None:
        graph = self.graph
        graph.eliminate_from_factor_graph_for_analysis(self.node0)
        graph.eliminate_from_factor_graph_for_analysis(self.node1)
        graph.eliminate_from_factor_graph_for_analysis(self.node2)
        graph.eliminate_from_factor_graph_for_analysis(self.node3)
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node0),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node1),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node2),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node3),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node4),
                            set())
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node0),
                            {self.node2, self.node3})
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node1),
                            {self.node3, self.node4})
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node2),
                            {self.node3})
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node3),
                            {self.node4})
        self.assertRaises(KeyError,
                          lambda: graph.get_parents_in_bayesian_network(
                              self.node4))

    def test_elimination_step_5(self) -> None:
        graph = self.graph
        graph.eliminate_from_factor_graph_for_analysis(self.node0)
        graph.eliminate_from_factor_graph_for_analysis(self.node1)
        graph.eliminate_from_factor_graph_for_analysis(self.node2)
        graph.eliminate_from_factor_graph_for_analysis(self.node3)
        graph.eliminate_from_factor_graph_for_analysis(self.node4)
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node0),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node1),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node2),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node3),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node4),
                            set())
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node0),
                            {self.node2, self.node3})
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node1),
                            {self.node3, self.node4})
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node2),
                            {self.node3})
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node3),
                            {self.node4})
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node4),
                            set())

    def test_chordalization(self) -> None:
        graph = self.graph
        graph.convert_to_bayesian_network_for_analysis([self.node0, self.node1,
                                                        self.node2, self.node3,
                                                        self.node4])
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node0),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node1),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node2),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node3),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node4),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node0),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node1),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node2),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node3),
                            set())
        self.assertSetEqual(graph.get_neighbors_in_factor_graph(self.node4),
                            set())
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node0),
                            {self.node2, self.node3})
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node1),
                            {self.node3, self.node4})
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node2),
                            {self.node3})
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node3),
                            {self.node4})
        self.assertSetEqual(graph.get_parents_in_bayesian_network(self.node4),
                            set())

    def test_bayes_tree(self) -> None:
        bayes_tree = self.graph.get_bayes_tree(
            ordering=[self.node0, self.node1, self.node2, self.node3,
                      self.node4])
        root = bayes_tree.root
        leaf = bayes_tree.leaves.pop()
        self.assertSetEqual(root.frontal, {self.node1, self.node3, self.node4})
        self.assertSetEqual(root.separator, set())
        self.assertSetEqual(leaf.frontal, {self.node0, self.node2})
        self.assertSetEqual(leaf.separator, {self.node3})


if __name__ == '__main__':
    unittest.main()
