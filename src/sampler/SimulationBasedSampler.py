from typing import List

import numpy as np

from factors.Factors import Factor
from factors.utils import unpack_prior_binary_nh_da_factors
from slam.Variables import Variable


class SimulationBasedSampler:
    def __init__(self, factors: List[Factor], vars: List[Variable]):
        self.factors = factors
        self.vars = vars
    def sample(self, num_samples:int):
        # binary_factors are binary factors while da_factors connect with three or more variables
        prior_factors, binary_factors, nh_factors, da_factors  = \
            unpack_prior_binary_nh_da_factors(self.factors)
        var_sample_dict = {}
        # assuming no conflicts in prior factors
        for factor in prior_factors:
            factor_samples = factor.sample(num_samples, )
            cur_var = 0
            for var in factor.vars:
                # here we assume that var.dim = distribution.dim
                # this iteration is primarily for flowsPriorFactor
                var_sample_dict[var] = factor_samples[:,
                                       cur_var: cur_var + var.dim]
                cur_var = cur_var + var.dim
        # obs samples come first the vars
        local_samples = np.empty(shape=(num_samples, 0))
        # obs dim comes first then var dim
        # circular_dim_list = []
        # obs var comes first then physical var
        var_ordering = []
        # observation is in the shape of zeros(dim,)
        unused_obs = np.array([])

        added_nh_factors = False

        unresolved_binary_factors = []
        # sampling from binary likelihood factors
        while binary_factors or nh_factors:
            if not added_nh_factors and len(binary_factors) == 0:
                binary_factors = nh_factors
                added_nh_factors = True
            factor = binary_factors.pop(0)
            var_intersection = set.intersection(set(factor.vars),
                                                set(var_sample_dict.keys()))
            intersect_len = len(var_intersection)

            var1 = factor.vars[0]
            var2 = factor.vars[1]
            if intersect_len == 1:
                if next(iter(
                        var_intersection)) == var1:  # only var1 has been sampled
                    if var1.dim < var2.dim:
                        #prevent sampling a SE3 pose from a a R2 landmark
                        if len(binary_factors) == 0:
                            #the only remaining factor can't be sampled
                            #"The only remaining factor in this clique requires sampling from landmark to pose"
                            unresolved_binary_factors.append(factor)
                            continue
                        binary_factors.append(factor)
                        continue
                    else:
                        var_sample_dict[var2] = factor.sample(
                            var1=var_sample_dict[var1], var2=None)
                else:  # only var2 has been sampled
                    if var2.dim < var1.dim:
                        #prevent sampling a SE3 pose from a a R2 landmark
                        if len(binary_factors) == 0:
                            #the only remaining factor can't be sampled
                            #"The only remaining factor in this clique requires sampling from landmark to pose"
                            unresolved_binary_factors.append(factor)
                            continue
                        binary_factors.append(factor)
                        continue
                    else:
                        var_sample_dict[var1] = factor.sample(
                            var1=None, var2=var_sample_dict[var2])
            elif intersect_len == 2:  # both vars have been sampled
                unused_obs = np.hstack((unused_obs,
                                      factor.observation))
                obs_sample = factor.sample(var1=var_sample_dict[var1],
                                           var2=var_sample_dict[var2])
                local_samples = np.hstack((local_samples,
                                           obs_sample))
                # TODO: make the name of observation_var independent of end nodes since it is possible that two binary factors sharing the same vars
                var_ordering.append(factor.observation_var)
            elif intersect_len == 0:  # both vars have not been sampled so push this factor back
                binary_factors.append(factor)
                continue
            else:
                raise ValueError("Oops! The number of variable"
                                 " intersection is " + str(intersect_len))
        sampled_physical_vars = set(var_sample_dict.keys())
        for factor in da_factors:
            da_vars = set(factor.vars)
            if da_vars.issubset(sampled_physical_vars):
                unused_obs = np.hstack((unused_obs,
                                      factor.observation))
                var_samples = {var: var_sample_dict[var] for var in factor.vars}
                obs_sample = factor.sample_observations(var_samples = var_samples)
                local_samples = np.hstack((local_samples,
                                           obs_sample))
                var_ordering.append(factor.observation_var)
            else:
                unsampled_vars = da_vars - sampled_physical_vars
                if unsampled_vars == {factor.observer_var}:
                    var_sample_dict[factor.observer_var] = factor.sample_observer(var_sample_dict)
                else:
                    raise ValueError("Some variables of the data association have not been sampled: "+" ".join([var.name for var in unsampled_vars]))

        sampled_physical_vars = set(var_sample_dict.keys())
        for factor in unresolved_binary_factors:
            f_vars = set(factor.vars)
            if f_vars.issubset(sampled_physical_vars):
                unused_obs = np.hstack((unused_obs,
                                      factor.observation))
                obs_sample = factor.sample(var1=var_sample_dict[factor.var1],
                                           var2=var_sample_dict[factor.var2])
                local_samples = np.hstack((local_samples,
                                           obs_sample))
                var_ordering.append(factor.observation_var)
            else:
                unsampled_vars = f_vars - sampled_physical_vars
                raise ValueError("Some variables of the data association have not been sampled: "+" ".join([var.name for var in unsampled_vars])+
                                 '. Consider using a different variable elimination ordering.')

        for var in self.vars:
            local_samples = np.hstack((local_samples,
                                       var_sample_dict[var]))
            var_ordering.append(var)
        return local_samples, var_ordering, unused_obs