from torch.distributions import Normal, MultivariateNormal, VonMises
import torch


class CustomMultivariateNormal(MultivariateNormal):
    def __init__(self, dim: int, device: str = "cpu") -> None:
        self._dim = dim
        self._device = device
        self._loc = torch.zeros(dim).to(device)
        self._scale_tril = torch.eye(dim).to(device)
        super().__init__(self._loc,
                         scale_tril = self._scale_tril)
        # super().__init__(torch.zeros(dim),
        #                  torch.eye(dim)).to(device)
    def cpu(self):
        return CustomMultivariateNormal(dim=self._dim, device="cpu")

    def is_cpu(self):
        return self._device == "cpu"

    @property
    def dim(self) -> int:
        return self._dim

    def to(self, device: str):
        return CustomMultivariateNormal(dim=self._dim, device=device)


class MultivariateNormalVonmises(object):
    def __init__(self, circular_dim_list: "List[bool]", device="cpu") -> None:
        self._dist = []
        self._device = device
        self._circular_dim_list = circular_dim_list
        for circular_dim in circular_dim_list:
            if circular_dim:
                self._dist.append(VonMises(loc=torch.tensor([0.0]).to(device),
                                           concentration=torch.tensor([1.0]).to(device)))
            else:
                self._dist.append(Normal(loc=torch.tensor([0.0]).to(device),
                                         scale=torch.tensor([1.0]).to(device)))

    def sample(self, sample_shape: tuple):
        # expect sample_shape would be sth like (5,)
        samples = torch.empty((sample_shape[0], 0)).to(self._device)
        for dist in self._dist:
            tmp_samples = dist.sample((sample_shape[0],))
            samples = torch.cat((samples, tmp_samples), 1).to(self._device)
        return samples

    def log_prob(self, x: "torch.tensor"):
        assert len(self._dist) == x.shape[1]
        res = self._dist[0].log_prob(x[:, 0])
        #caution: this res is supposed to be a vector?
        for i in range(1, x.shape[1]):
            res = res + self._dist[i].log_prob(x[:, i])
        return res

    def cpu(self):
        return MultivariateNormalVonmises(circular_dim_list=self._circular_dim_list,
                                          device="cpu")

    def is_cpu(self):
        return self._device == "cpu"

    @property
    def dim(self) -> int:
        return len(self._circular_dim_list)

    def to(self, device: str):
        return CustomMultivariateNormal(self._circular_dim_list, device)