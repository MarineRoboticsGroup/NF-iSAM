import torch
import torch.nn as nn

class NormalizingFlowModel(nn.Module):
    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flows = nn.ModuleList(flows)
        self._prior_device_check = False

    def forward(self, x):
        if not self._prior_device_check:
            if self.prior._device != x.device.__str__():
                self.prior = self.prior.to(x.device.__str__())
            self._prior_device_check = True
        m, _ = x.shape
        log_det = torch.zeros(m)
        if x.is_cuda:
            log_det = log_det.cuda()
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def inverse(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        if z.is_cuda:
            log_det = log_det.cuda()
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples):
        z = self.prior.sample((n_samples,), )
        x, _ = self.inverse(z)
        return x

