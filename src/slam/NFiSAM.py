import logging
import os

import matplotlib.pyplot as plt
import scipy
import torch
from scipy.stats import circmean
from flows.models import NormalizingFlowModel
from flows.prior_dist import *
from flows.flows import *
from slam.FactorGraphSolver import *
import torch.optim as optim

from slam.RunBatch import graph_file_parser, group_nodes_factors_incrementally
from slam.FactorGraphSolver import run_incrementally
from utils.Functions import theta_to_pipi

class NFiSAMArgs(SolverArgs):
    def __init__(self,
                 elimination_method: str = "natural",
                 posterior_sample_num: int = 500,
                 local_sample_num: int = 500,
                 store_clique_samples: bool = False,
                 local_sampling_method="direct",
                 learning_rate: float = 0.015,
                 flow_number: int = 1,
                 flow_type: str = "NSF_AR",
                 flow_iterations: int = 10,
                 num_knots: int = 12,
                 cuda_training: bool = False,
                 adaptive_flow_setup: bool = False,
                 hidden_dim: int = 8,
                 average_window=50,
                 loss_delta_tol = 1e-2,
                 training_set_frac = 1.0,
                 validation_interval = 10,
                 slower_stop_rate = 2.0,
                 data_parallel = False,
                 training_loss_dir = None,
                 *args, **kwargs
                 ):
        super().__init__(elimination_method=elimination_method,
                         posterior_sample_num=posterior_sample_num,
                         local_sample_num=local_sample_num,
                         store_clique_samples=store_clique_samples,
                         local_sampling_method=local_sampling_method,
                         *args, **kwargs)
        # normalizing flow parameters
        self.flow_number = flow_number
        self.flow_type = flow_type
        self.flow_iterations = flow_iterations
        self.num_knots = num_knots
        self.cuda_training = cuda_training
        self.learning_rate = learning_rate
        self.adaptive_flow_setup = adaptive_flow_setup
        self.hidden_dim = hidden_dim
        self.average_window = average_window
        self.loss_delta_tol = loss_delta_tol
        self.training_set_frac = training_set_frac
        self.validation_interval = validation_interval
        self.slower_stop_rate = slower_stop_rate
        self.data_parallel = data_parallel
        if training_loss_dir is not None and not os.path.exists(training_loss_dir):
            os.mkdir(training_loss_dir)
        self.training_loss_dir = training_loss_dir
        self.tl_cnt = 0

class NormalizingFlowModelWithSeparator(NormalizingFlowModel, ConditionalSampler):
    """
    This class uses normalizing flows to model the joint density of clique factors.

    Parameters
    __________
    flows: a list of normalizing flows modeling T between prior distribution and target density
    prior: a distribution over all variables
    separator_prior: a distribution over separator variables; S in T(S,F)
    frontal_prior: a distribution over frontal variables

    """
    def __init__(self, flows, prior: MultivariateNormalVonmises, separator_prior: MultivariateNormalVonmises,
                 circular_dim_list, samples_mean: "torch.Tensor" = None, samples_std: "torch.Tensor" = None):
        super().__init__(prior, flows)
        self.separator_prior = separator_prior
        if separator_prior is not None:
            self.separator_dim = separator_prior.dim
        else:
            self.separator_dim = 0
        self.samples_mean = samples_mean
        self.samples_std = samples_std
        self.circular_dim_list = circular_dim_list

    @property
    def dim(self):
        return len(self.circular_dim_list)

    def normalize_samples(self, samples, init_dim):
        circular_indices = np.where(self.circular_dim_list[init_dim:init_dim+samples.shape[-1]])[0]
        euclidean_indices = np.setdiff1d(np.arange(samples.shape[-1]), circular_indices)
        # regularize observation samples
        samples[:, circular_indices] = \
            theta_to_pipi(samples[:, circular_indices] - self.samples_mean[circular_indices+init_dim]) / \
            self.samples_std[circular_indices+init_dim]
        samples[:, euclidean_indices] = \
            (samples[:, euclidean_indices] - self.samples_mean[euclidean_indices+init_dim]) / \
            self.samples_std[euclidean_indices+init_dim]
        return samples

    def unnormalize_samples(self, normalized, init_dim):
        circular_indices = np.where(self.circular_dim_list[init_dim:init_dim+normalized.shape[-1]])[0]
        euclidean_indices = np.setdiff1d(np.arange(normalized.shape[-1]), circular_indices)
        # affine transform to the target space
        normalized[:, euclidean_indices] = \
            normalized[:, euclidean_indices] * self.samples_std[euclidean_indices+init_dim] + \
            self.samples_mean[euclidean_indices+init_dim]
        normalized[:, circular_indices] = theta_to_pipi(
            normalized[:, circular_indices] * self.samples_std[circular_indices+init_dim] + \
            self.samples_mean[circular_indices+init_dim])
        return normalized
    #get samples of C in P(C|O) where O is obs_samples
    def conditional_sample_given_observation(self,
                                             conditional_dim,
                                             obs_samples = None,
                                             sample_number = None)->"np.ndarray":
        if sample_number is None and obs_samples is not None:
            # given observation samples
            n_samples = obs_samples.shape[0]
            obs_dim = obs_samples.shape[1]
            x_s = obs_samples
        elif sample_number is not None:
            n_samples = sample_number
            x_s = None
            obs_dim = 0
        else:
            raise ValueError("must input one of obs_samples or sample_number")
        dim_sum = obs_dim+conditional_dim
        z = self.prior.sample((n_samples,))[:, obs_dim:dim_sum]
        conditional_samples = self.inverse_given_separator(z,x_s)
        return conditional_samples.detach().numpy()

    def inverse_given_separator(self, z, x_s = None):
        """
        z is the samples in the latent space and will be pulled back to the target space.
        x_s is the unnormalized samples of separator variables
        """
        if x_s is None:
            obs_dim = 0
            rectified_obs = None
        else:
            obs_dim = x_s.shape[1]
            rectified_obs = self.normalize_samples(torch.tensor(np.float32(x_s)), init_dim=0)
        for flow in self.flows[::-1]:
            z = flow.inverse_given_separator(z, rectified_obs)
        # after the for loop, z is normalized x in the target space
        z = self.unnormalize_samples(z, init_dim=obs_dim)
        return z

    def separator_forward(self,x):
        """
        x is samples of separator variables and will be pushed forward to the latent space.
        """
        m, d = x.shape
        assert d == self.separator_dim
        separator_log_det = torch.zeros(m)
        if x.is_cuda:
            separator_log_det = separator_log_det.cuda()

        x = self.normalize_samples(x, init_dim=0)
        for flow in self.flows:
            x, ld = flow.forward(x)
            separator_log_det += ld
        separator_prior_logprob = self.separator_prior.log_prob(x)
        # x is z in the latent space at this point
        return x, separator_prior_logprob, separator_log_det

    @property
    def is_cpu(self):
        #presume prior is a multinormal with mean
        return self.prior.is_cpu()
    def to_cpu(self):
        cpu_flows = self.flows.cpu()
        cpu_prior = self.prior.cpu()
        if self.separator_prior is None:
            cpu_sep = None
        else:
            cpu_sep = self.separator_prior.cpu()
        return NormalizingFlowModelWithSeparator(flows=cpu_flows, prior=cpu_prior, separator_prior=cpu_sep,
                                                 circular_dim_list=self.circular_dim_list,
                                                 samples_mean=self.samples_mean, samples_std=self.samples_std)

    def to(self, device: str):
        dev_flows = self.flows.to(device)
        dev_prior = self.prior.to(device)
        if self.separator_prior is None:
            dev_sep = None
        else:
            dev_sep = self.separator_prior.to(device)
        return NormalizingFlowModelWithSeparator(flows=dev_flows, prior=dev_prior, separator_prior=dev_sep,
                                                 circular_dim_list=self.circular_dim_list,
                                                 samples_mean=self.samples_mean, samples_std=self.samples_std)

# TODO: remove circular_dim_list in factor
class FlowsPriorFactor(CliqueSeparatorFactor):
    def __init__(self, vars: List[Variable], flow_model: NormalizingFlowModelWithSeparator, true_obs: np.ndarray,
                 circular_dim_list: List) -> None:
        """
        :param means, stds
        :warning: means and stds of separator samples
        :the circular_dim_list should be informed during local inference
        """
        super().__init__()
        self._vars = vars
        self._flow_model = flow_model
        self._is_gaussian = False
        self._true_obs = true_obs
        self._obs_dim = len(true_obs)
        self._circular_dim_list = circular_dim_list[:]
        assert self.dim == len(circular_dim_list)

    def append_obs_sample(self, x):
        """
        This method inserts true observations to x.
        Note that put columns of observations before physical variables, x,
            which is the convention of variable ordering in the flow model.
        """
        n, d = x.shape
        if self._obs_dim == 0:
            aug_sample = x
        else:
            obs_sample = np.tile(self._true_obs, (n, 1))
            aug_sample = np.concatenate( (obs_sample, x), axis=1)
        return aug_sample

    def log_pdf(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        This method return log_pdf of x up to a constant scale.
        The flow model can only return log_pdf of (obs, x) in which obs is fixed.
        """
        aug_sample = self.append_obs_sample(x)
        if ~(self._flow_model.is_cpu):
            z, separator_prior_logprob, separator_log_det = \
                self._flow_model.separator_forward(
                    torch.Tensor(aug_sample).cuda())
            tmp_tensor = separator_prior_logprob + separator_log_det
            ##return variable on cpu for sampling
            return tmp_tensor.cpu().detach().numpy()
        else:
            z, separator_prior_logprob, separator_log_det = \
                self._flow_model.separator_forward(torch.Tensor(aug_sample))
            tmp_tensor = separator_prior_logprob + separator_log_det
            ##return variable on cpu for sampling
            return tmp_tensor.detach().numpy()

    def grad_x_log_pdf(self, x: np.ndarray, **kwargs) -> np.ndarray:
        aug_sample = self.append_obs_sample(x)
        x_ts = torch.Tensor(aug_sample.astype(float))  # astype(np.float32)
        x_ts.requires_grad_(True)
        if ~(self._flow_model.is_cpu):
            z, separator_prior_logprob, separator_log_det = \
                self._flow_model.separator_forward(x_ts.cuda())
            tmp_tensor = separator_prior_logprob + separator_log_det
            tmp_tensor.backward()
            res_arr = x_ts.grad.cpu().detach().numpy()
        else:
            z, separator_prior_logprob, separator_log_det = \
                self._flow_model.separator_forward(x_ts)
            tmp_tensor = separator_prior_logprob + separator_log_det
            tmp_tensor.backward()
            res_arr = x_ts.grad.detach().numpy()
        return res_arr[:,self._obs_dim:self._obs_dim+x.shape[1]]

    def sample(self, num_samples: int, **kwargs) -> np.ndarray:
        """
        :param num_samples: number of samples
        :type: int
        :return: samples
        :rtype: numpy.ndarray
               each row is a sample
               the number of columns is the number of dim
        """
        if self._obs_dim == 0:
            return self._flow_model. \
                conditional_sample_given_observation(conditional_dim=self.dim,
                                                     sample_number=num_samples)
        else:
            obs_samples = np.tile(self._true_obs, (num_samples, 1))
            return self._flow_model. \
                conditional_sample_given_observation(conditional_dim=self.dim,
                                                     obs_samples=obs_samples)

    def unif_to_sample(self, u) -> np.ndarray:
        # For nested sampling
        # u is 1*D numpy array
        normal_var = np.array([scipy.stats.norm.ppf(u)]).astype(np.float32)  # convert to standard normal
        # normalize true obs
        if self._obs_dim == 0:
            # the return x has to been shifted by mean and scaled by sigma.
            x = self._flow_model.inverse_given_separator(z=torch.tensor(normal_var))
        else:
            obs_samples = np.tile(self._true_obs, (normal_var.shape[0], 1))
            # the return x has to been shifted by mean and scaled by sigma.
            x = self._flow_model.inverse_given_separator(z=torch.tensor(normal_var), x_s=obs_samples)
        # return 1*D array as well
        return x[0, :]

    @property
    def is_gaussian(self) -> bool:
        return self._is_gaussian

    @property
    def vars(self) -> List[Variable]:
        return self._vars

    @property
    def circular_dim_list(self) -> List[bool]:
        return self._circular_dim_list

class NFiSAM(FactorGraphSolver):
    def __init__(self, args: NFiSAMArgs = NFiSAMArgs()):
        super().__init__(args=args)
        # overiding args is just to get type hint when coding
        self._args = args

    def fit_clique_density_model(self,
                                 clique: BayesTreeNode,
                                 samples: np.ndarray,
                                 var_ordering: List[Variable],
                                 timer: List,
                                 *args, **kwargs)\
            -> NormalizingFlowModelWithSeparator:
        """
        :return: flow model
        """
        # hyperparameters for learning
        num_knots  = self._args.num_knots
        flow_iterations = self._args.flow_iterations
        flow_number = self._args.flow_number
        flow_type = self._args.flow_type
        learning_rate  = self._args.learning_rate
        hidden_dim = self._args.hidden_dim

        # clique variables are augmented by forcasted observations
        frontal_dim = clique.frontal_dim
        aug_separator_dim = samples.shape[-1] - frontal_dim
        aug_clique_dim = samples.shape[-1]

        circular_dim_list = []
        for var in var_ordering:
            circular_dim_list += var.circular_dim_list
        # CPU/GPU flow model
        if (self._args.cuda_training and not self._args.adaptive_flow_setup) or \
                (self._args.cuda_training and self._args.adaptive_flow_setup and
                 (aug_clique_dim * samples.shape[0] > 6000)):
            if torch.cuda.is_available():
                device = "cuda:0"
                print("\tcuda training")
            else:
                device = "cpu"
                print("\tcannot find cuda so training with cpu\n")
        else:
            device = "cpu"
            print("\tcpu training")

        print("Identify device: ", time.time())
        # if true_obs_dim > 0 and self._args.adaptive_flow_setup:
        #     # observation samples will be much more non-Gaussian
        #     num_knots = 2 * num_knots
        #     flow_iterations = 2 * flow_iterations

        # if aug_clique_dim != samples.shape[1] or aug_clique_dim != len(
        #         circular_dim_list):
        #     raise ValueError("Input sample dim is incorrect.")

        # train-test split
        train_size = min(int(samples.shape[0]*self._args.training_set_frac), samples.shape[0])
        np.random.shuffle(samples)
        train_samples, test_samples = samples[:train_size], samples[train_size:]
        print("Got training samples: ", time.time())
        # normalizing samples
        # the outputs are torch tensors
        training_data, means, stds = self.normalize_training_samples(train_samples, circular_dim_list, flow_type)
        if len(test_samples) > 0:
            testing_data, _, _ = self.normalize_training_samples(test_samples, circular_dim_list, flow_type)
        else:
            testing_data = None
        print("Normalized samples: ", time.time())
        # declare a flow type; add specific setting of cnn later
        flow = eval(flow_type)
        if flow_type == "NSF_AR":
            flows = [flow(dim=aug_clique_dim, K=num_knots, hidden_dim=hidden_dim).to(
                device) for _ in range(flow_number)]
            normal_clique = CustomMultivariateNormal(dim=aug_clique_dim,
                                                     device=device)
            if aug_separator_dim > 0:
                normal_separator = CustomMultivariateNormal(dim=aug_separator_dim,
                                                            device=device)
            else:
                normal_separator = None
        elif flow_type == "NSF_AR_CS":
            flows = [flow(dim=aug_clique_dim, K=num_knots, hidden_dim=hidden_dim,
                          circular_dim_list=circular_dim_list).to(
                device) for _ in range(flow_number)]
            normal_clique = MultivariateNormalVonmises(circular_dim_list,
                                                       device=device)
            if aug_separator_dim > 0:
                normal_separator = MultivariateNormalVonmises(
                    circular_dim_list[:aug_separator_dim],
                    device=device)
            else:
                normal_separator = None
        else:
            raise NotImplementedError("Unknown flow type for the pipeline")

        print("Declare flows: ", time.time())
        # adding circular list into flow models?
        # clique_density_model = NormalizingFlowModelWithSeparator(flows, normal_clique, normal_separator,
        #                                                          circular_dim_list, means, stds).to(device)

        clique_density_model = NormalizingFlowModelWithSeparator(flows, normal_clique, normal_separator,
                                                                 circular_dim_list, means, stds)
        print("Declare model: ", time.time())
        if device[:4] == "cuda" and self._args.data_parallel:
            clique_density_model = nn.DataParallel(clique_density_model)
            print("Parallelize model: ", time.time())

        optimizer = optim.Adam(clique_density_model.parameters(),
                               lr=learning_rate)

        # only training_data, normal_clique, and flows will be involved with training on GPU
        training_data = training_data.to(device)
        if testing_data is not None:
            testing_data = testing_data.to(device)

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("flows on clique")
        opt_start = time.time()

        # convergene criteria
        iter_loss = torch.zeros(flow_iterations).to(device)
        average_window = self._args.average_window
        loss_delta_tol = self._args.loss_delta_tol
        loss_avg = None

        # # validation set size
        # validation_frac = .4
        validation_interval = self._args.validation_interval
        last_validation_loss = float('inf')

        slower_stop_iter = None

        print("Start training: ", time.time())
        for i in range(flow_iterations):
            if slower_stop_iter is not None:
                if (i+1) >= slower_stop_iter:
                    logger.info(f"Slower stop at iter {i + 1}")
                    break
            elif testing_data is not None:
                if (i+1) % validation_interval == 0:
                    # using validation set
                    z, prior_logprob, log_det = clique_density_model(testing_data)
                    logprob = prior_logprob + log_det
                    loss = -torch.mean(logprob)
                    new_loss = loss.data
                    logger.info(f"Iter: {i + 1}\t, validation loss: {new_loss}")
                    if new_loss > last_validation_loss:
                        logger.info(f"Early stopping at iter {i + 1}")
                        slower_stop_iter = int(self._args.slower_stop_rate * (i+1) )
                    else:
                        last_validation_loss = new_loss
            optimizer.zero_grad()
            z, prior_logprob, log_det = clique_density_model(training_data)
            logprob = prior_logprob + log_det
            loss = -torch.mean(logprob)
            iter_loss[i] = loss.data
            loss.backward()
            optimizer.step()
            # if (i+1) % 100 == 0:
            #     logger.info(f"Iter: {i+1}\t" +
            #                 f"Logprob: {logprob.mean().data:.2f}\t" +
            #                 f"Prior: {prior_logprob.mean().data:.2f}\t" +
            #                 f"LogDet: {log_det.mean().data:.2f}")
            if testing_data is None:
                # simply using convergence as stopping criteria
                if (i+1) % average_window == 0:
                    new_loss = torch.mean(iter_loss[i-average_window+1:i+1])
                    if loss_avg is not None and loss_avg != 0.0:
                        delta = abs(1.0 - new_loss/loss_avg)
                        logger.info(f"Iter: {i+1}\t, last {average_window} steps loss change: {delta}")
                        if delta < loss_delta_tol:
                            logger.info(f"Early stopping at iter {i+1}")
                            break
                    loss_avg = new_loss
        opt_end = time.time()
        if timer is not None:
            timer.append(opt_end - opt_start)

        clique_name = ''.join([var.name for var in clique.vars])
        self._temp_training_loss[clique_name] = [step_loss for step_loss in np.array(iter_loss.to("cpu"), dtype=np.float64)]
        if self._args.training_loss_dir is not None and os.path.exists(self._args.training_loss_dir):
            plt.figure()
            plt.plot(iter_loss.to("cpu"))
            fig_name = f"{self._args.training_loss_dir}/{self._args.tl_cnt}.png"
            plt.savefig(fig_name, dpi=300, bbox_inches="tight")
            self._args.tl_cnt += 1
            plt.close()

        # if model trained on gpu, move it back to cpu for post-processing
        if device != "cpu":
            if self._args.data_parallel:
                return clique_density_model.module.to("cpu")
            else:
                return clique_density_model.to("cpu")
        else:
            return clique_density_model

    def normalize_training_samples(self, samples, circular_dim_list, flow_type: str):
        aug_clique_dim = samples.shape[-1]
        means = np.zeros(aug_clique_dim)
        stds = np.zeros(aug_clique_dim)
        circular_indices = np.where(circular_dim_list)[0]
        euclidean_indices = np.setdiff1d(np.arange(aug_clique_dim), circular_indices)

        if flow_type == "NSF_AR":
            means[circular_indices] = circmean(samples[:, circular_indices], high=np.pi, low=-np.pi, axis=0)
            # transform the data to [-pi, pi]
            shifted_sample = theta_to_pipi(samples[:, circular_indices] - means[circular_indices])
            stds[circular_indices] = np.std(
                shifted_sample, axis=0)  # approximiate std of circular quantity
            samples[:, circular_indices] = torch.tensor(shifted_sample)
        elif flow_type == "NSF_AR_CS":
            means[circular_indices] = circmean(samples[:, circular_indices], high=np.pi, low=-np.pi, axis=0)
            # transform the data to [-pi, pi]
            stds[circular_indices] = 1.0  # no scaling samples as it is circular spline flow fixed on [-pi,pi]
            # transform the data to [-pi, pi]
            samples[:, circular_indices] = theta_to_pipi(samples[:, circular_indices] -
                                                               means[circular_indices])
        else:
            raise NotImplementedError(
                "Unknown flow type for the pipeline")
        #Euclidean dim
        means[euclidean_indices] = np.mean(samples[:, euclidean_indices], axis=0)
        stds[euclidean_indices] = np.std(samples[:, euclidean_indices], axis=0)
        samples[:, euclidean_indices] = samples[:, euclidean_indices] - means[euclidean_indices]

        # this may cause error when samples of a variable get narrowed down to a point
        stds = np.clip(stds, a_min = 1e-5, a_max=None)
        samples = samples /  stds
        training_data = torch.Tensor(samples)
        return training_data, torch.Tensor(means), torch.Tensor(stds)

    def root_clique_density_model_to_leaf(self, old_clique, new_clique, device):
        """
        when old clique and new clique have same variables but different division of frontal and separator vars,
        recycle the density model in the old clique and convert it to that in the new clique.
        """
        old_flow = self._clique_density_model[old_clique]
        obs_dim = old_flow.dim - old_clique.dim
        separator_dim = new_clique.separator_dim + obs_dim
        if isinstance(old_flow.flows[0], NSF_AR):
            if separator_dim > 0:
                normal_separator = CustomMultivariateNormal(dim=separator_dim, device=device)
            else:
                normal_separator = None
        elif isinstance(old_flow.flows[0], NSF_AR_CS):
            if separator_dim > 0:
                normal_separator = MultivariateNormalVonmises(
                    old_flow.circular_dim_list[:separator_dim], device=device)
            else:
                normal_separator = None
        else:
            raise NotImplementedError("Unknown flow type for the pipeline")

        new_flow_model = NormalizingFlowModelWithSeparator(flows=old_flow.flows, prior=old_flow.prior,
                                                           separator_prior=normal_separator,
                                                           circular_dim_list=old_flow.circular_dim_list,
                                                           samples_mean=old_flow.samples_mean,
                                                           samples_std=old_flow.samples_std)
        return new_flow_model

    def clique_density_to_separator_factor(self, separator_var_list, density_model, true_obs):
        """
        true_obs is a 1D array that concatenates all observations in the clique
        """
        obs_dim = true_obs.shape[-1]
        obs_separator_dim = sum([var.dim for var in separator_var_list]) + obs_dim
        return FlowsPriorFactor(vars=separator_var_list, flow_model=density_model, true_obs=true_obs,
                                circular_dim_list=density_model.circular_dim_list[obs_dim: obs_separator_dim])


def NFiSAM_empirial_study(knots, iters, training_samples, learning_rates, hidden_dims, case_dir, data_file, data_format,
                          incremental_step=1, prior_cov_scale=0.1, traj_plot=False, plot_args=None, check_root_transform=False, **kwargs):
    data_dir = os.path.join(case_dir, data_file)
    nodes, truth, factors = graph_file_parser(data_file=data_dir, data_format=data_format, prior_cov_scale=prior_cov_scale)

    nodes_factors_by_step = group_nodes_factors_incrementally(
        nodes=nodes, factors=factors, incremental_step=incremental_step)

    for knt in knots:
        for iter in iters:
            for training_sample in training_samples:
                for lr in learning_rates:
                    for hidden_dim in hidden_dims:
                        args = NFiSAMArgs(num_knots=knt,
                                          flow_iterations=iter,
                                          local_sample_num=training_sample,
                                          learning_rate=lr,
                                          hidden_dim=hidden_dim,
                                          **kwargs)
                        solver = NFiSAM(args)
                        run_incrementally(case_dir, solver, nodes_factors_by_step, truth, traj_plot, plot_args,check_root_transform)