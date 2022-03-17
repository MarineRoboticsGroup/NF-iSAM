import unittest
from slam.FactorGraphSimulator import G2oToroPoseGraphReader
import numpy as np
from slam.Variables import Variable

class TestCase(unittest.TestCase):
    def test_g2o(self) -> None:
        # constructor
        file_path ="/home/chad/Research/optimalTransport/CouplingSLAM/example/slam/MIT_g2o_dataset/trimmed_MIT.g2o"
        pg = G2oToroPoseGraphReader(file_path)
        self.assertEqual(pg.file_path, file_path)
        self.assertEqual(pg.file_type, "g2o")
        self.assertEqual(pg.node_head, "VERTEX_SE2")
        self.assertEqual(pg.factor_head, "EDGE_SE2")

        self.assertEqual(pg.node_list[0], Variable("0", 3))
        self.assertEqual(pg.node_list[-1], Variable("10", 3))

        first_factor = pg.factor_list[0]
        last_factor = pg.factor_list[-1]
        self.assertEqual(first_factor.vars[0], Variable("0", 3))
        self.assertEqual(first_factor.vars[1], Variable("1", 3))
        self.assertIsNone(
            np.testing.assert_almost_equal(first_factor.observation,
                                           np.array([2.039345, 0.003006, 0.014452])))
        info_mat = np.array([[1.778126, 0.026853,0.000000],
                             [0.026853, 3.846788,0.000000],
                             [0.000000, 0.000000,388.684289]])
        self.assertIsNone(
            np.testing.assert_almost_equal(first_factor.noise_cov,
                                           np.linalg.inv(info_mat)))
        self.assertEqual(last_factor.vars[0], Variable("9", 3))
        self.assertEqual(last_factor.vars[1], Variable("4", 3))
        obs = np.array([-7.500000, -8.000000, 1.570796])
        self.assertIsNone(
            np.testing.assert_almost_equal(last_factor.observation,obs))
        info_mat = np.array([[1.008417, -0.820651,0.000000],
                             [-0.820651, 0.902417,0.000000],
                             [0.000000, 0.000000,60.523586]])
        self.assertIsNone(
            np.testing.assert_almost_equal(last_factor.noise_cov,
                                           np.linalg.inv(info_mat)))

    def test_toro(self) -> None:
        # constructor
        file_path ="/home/chad/Research/optimalTransport/CouplingSLAM/example/slam/CSAIL_toro_dataset/CSAIL_P_toro.graph"
        pg = G2oToroPoseGraphReader(file_path)
        self.assertEqual(pg.file_path, file_path)
        self.assertEqual(pg.file_type, "graph")
        self.assertEqual(pg.node_head, "VERTEX2")
        self.assertEqual(pg.factor_head, "EDGE2")

        self.assertEqual(pg.node_list[0], Variable("0", 3))
        self.assertEqual(pg.node_list[-1], Variable("1044", 3))

        first_factor = pg.factor_list[0]
        last_factor = pg.factor_list[-1]
        self.assertEqual(first_factor.vars[0], Variable("0", 3))
        self.assertEqual(first_factor.vars[1], Variable("1", 3))
        obs = np.array([0.082760, 0.003050, 0.284020])
        self.assertIsNone(
            np.testing.assert_almost_equal(first_factor.observation,obs))

        info_mat = np.array([[123.488234, -2144.807885,0.000000],
                             [-2144.807885, 58242.575768,0.000000],
                             [0.000000, 0.000000,6065.357771]])
        self.assertIsNone(
            np.testing.assert_almost_equal(first_factor.noise_cov,
                                           np.linalg.inv(info_mat)))

        self.assertEqual(last_factor.vars[0], Variable("915", 3))
        self.assertEqual(last_factor.vars[1], Variable("1008", 3))
        obs = np.array([0.462450, 0.580490, 1.824460])
        self.assertIsNone(
            np.testing.assert_almost_equal(last_factor.observation,obs))
        info_mat = np.array([[461.494377, -332.244726,0.000000],
                             [-332.244726, 309.128718,0.000000],
                             [0.000000, 0.000000,1253.513867]])
        self.assertIsNone(
            np.testing.assert_almost_equal(last_factor.noise_cov,
                                           np.linalg.inv(info_mat)))

if __name__ == '__main__':
    unittest.main()
