import numpy as np
from theano import tensor as tt


class LogJF(tt.Op):
    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """
    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, Joint_Factor, dim):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        JF:
            A joint factor object

        """

        # add inputs as class attributes
        self.likelihood = Joint_Factor
        self.dim = dim

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = (self.likelihood.log_pdf(theta.reshape(1, self.dim)))[0]

        outputs[0][0] = np.array(logl)  # output the log-likelihood


class LogJFWithGrad(tt.Op):
    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, Joint_Factor, dim):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        Joint_Factor:
            A joint factor object
        dim:
            int
        """

        # add inputs as class attributes
        self.likelihood = Joint_Factor
        self.dim = dim

        # initialise the gradient Op (below)
        self.logpgrad = LogJFGrad(self.likelihood, self.dim)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = (self.likelihood.log_pdf(theta.reshape(1, self.dim)))[0]

        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        theta, = inputs  # our parameters
        return [g[0] * self.logpgrad(theta)]


class LogJFGrad(tt.Op):
    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, Joint_Factor, dim):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        Joint_Factor:
            A joint factor object
        """

        # add inputs as class attributes
        self.likelihood = Joint_Factor
        self.dim = dim

    def perform(self, node, inputs, outputs):
        theta, = inputs

        # define version of likelihood function to pass to derivative function

        # calculate gradients
        grads = self.likelihood.grad_x_log_pdf(theta.reshape(1, self.dim))

        outputs[0][0] = grads.flatten()