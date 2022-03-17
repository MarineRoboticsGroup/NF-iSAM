import unittest
import numpy.testing as test

from sampler.sampler_utils import JointFactor
from factors.Factors import *


class GaussianFactorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.var1 = Variable("X1", 1)
        cls.var2 = Variable("X2", 2)
        cls.var3 = Variable("X3", 3)
        cls.var4 = Variable("X4", 1)
        cls.factor1dim = cls.var1.dim + cls.var2.dim
        cls.factor2dim = cls.var2.dim + cls.var3.dim + cls.var4.dim
        cls.dist1 = dist.StandardNormalDistribution(cls.factor1dim)
        cls.explicit_factor1 = ExplicitPriorFactor([cls.var1, cls.var2],
            dist.StandardNormalDistribution(cls.factor1dim))
        num_positions_1 = 1000
        cls.x1 = (np.random.rand(num_positions_1, cls.factor1dim)
                  - 0.5 + cls.dist1.mu)
        tmp1 = np.random.rand(cls.factor2dim)
        tmp2 = np.random.rand(cls.factor2dim, cls.factor2dim)
        cls.dist2 = dist.GaussianDistribution(mu=tmp1, precision=tmp2 @ tmp2.T)
        cls.explicit_factor2 = ExplicitPriorFactor(
            [cls.var2, cls.var3, cls.var4], cls.dist2)
        num_positions_2 = 1000
        cls.x2 = ((np.random.rand(num_positions_2, cls.factor2dim) - 0.5)
                  + cls.dist2.mu)
        cls.joint_factor = JointFactor([cls.explicit_factor1,
                                        cls.explicit_factor2],
                                       [cls.var3, cls.var1, cls.var2, cls.var4])
        cls.x = (np.random.rand(3000, 7) - 0.5) * 5

    def test_standard_gaussian_factor_pdf(self) -> None:
        test.assert_array_equal(self.dist1.pdf(self.x1),
                                self.explicit_factor1.pdf(self.x1))

    def test_standard_gaussian_factor_log_pdf(self) -> None:
        test.assert_array_equal(self.dist1.log_pdf(self.x1),
                                self.explicit_factor1.log_pdf(self.x1))

    def test_standard_gaussian_factor_grad_x_log_pdf(self) -> None:
        test.assert_array_equal(self.dist1.grad_x_log_pdf(self.x1),
                                self.explicit_factor1.grad_x_log_pdf(self.x1))

    def test_standard_gaussian_factor_hess_x_log_pdf(self) -> None:
        test.assert_array_equal(self.dist1.hess_x_log_pdf(self.x1),
                                self.explicit_factor1.hess_x_log_pdf(self.x1))

    def test_gaussian_factor_pdf(self) -> None:
        test.assert_array_equal(self.dist2.pdf(self.x2),
                                self.explicit_factor2.pdf(self.x2))

    def test_gaussian_factor_log_pdf(self) -> None:
        test.assert_array_equal(self.dist2.log_pdf(self.x2),
                                self.explicit_factor2.log_pdf(self.x2))

    def test_gaussian_factor_grad_x_log_pdf(self) -> None:
        test.assert_array_equal(self.dist2.grad_x_log_pdf(self.x2),
                                self.explicit_factor2.grad_x_log_pdf(self.x2))

    def test_gaussian_factor_hess_x_log_pdf(self) -> None:
        test.assert_array_equal(self.dist2.hess_x_log_pdf(self.x2),
                                self.explicit_factor2.hess_x_log_pdf(self.x2))

    def test_joint_factor_var_order(self) -> None:
        test.assert_array_equal([Variable("X3", 3), Variable("X1", 1),
                                 Variable("X2", 2), Variable("X4", 4)],
                                self.joint_factor.vars)
        dict_to_test = self.joint_factor.factor_to_indices
        test.assert_array_equal([3, 4, 5],
                                dict_to_test[self.explicit_factor1])
        test.assert_array_equal([4, 5, 0, 1, 2, 6],
                                dict_to_test[self.explicit_factor2])

    def test_joint_factor_pdf(self) -> None:
        test.assert_array_equal(self.dist1.pdf(self.x[:, [3, 4, 5]])
                                * self.dist2.pdf(self.x[:, [4, 5, 0, 1, 2, 6]]),
                                self.joint_factor.pdf(self.x))

    def test_joint_factor_log_pdf(self) -> None:
        test.assert_array_equal(self.dist1.log_pdf(self.x[:, [3, 4, 5]]) +
                                self.dist2.log_pdf(self.x[:, [4, 5, 0,
                                                              1, 2, 6]]),
                                self.joint_factor.log_pdf(self.x))

    def test_joint_factor_grad_x_log_pdf(self) -> None:
        grad = np.zeros((3000, 7))
        grad[:, [3, 4, 5]] = self.dist1.grad_x_log_pdf(self.x[:, [3, 4, 5]])
        grad[:, [4, 5, 0, 1, 2, 6]] += self.dist2.grad_x_log_pdf(
            self.x[:, [4, 5, 0, 1, 2, 6]])
        test.assert_array_equal(grad, self.joint_factor.grad_x_log_pdf(self.x))

    def test_joint_factor_hess_x_log_pdf(self) -> None:
        grad = np.zeros((3000, 7, 7))
        ind1 = [3, 4, 5]
        ind2 = [4, 5, 0, 1, 2, 6]
        grad[np.ix_(range(3000), ind1, ind1)] \
            = self.dist1.hess_x_log_pdf(self.x[:, ind1])
        grad[np.ix_(range(3000), ind2, ind2)] += self.dist2.hess_x_log_pdf(
            self.x[:, ind2])
        test.assert_array_equal(grad, self.joint_factor.hess_x_log_pdf(self.x))


if __name__ == '__main__':
    unittest.main()

