class Noise:
    """An implementation of Ornstein Uhlenbeck noise

    :param nactions: (int) action space size
    :param nsteps: (int) observations in an epoch * n epochs
    :param mu: (float) noise mean
    :param sigma: (float) noise std
    :param theta: (float) decay/growth rate
    """

    def __init__(self, nactions, nsteps, mu=0, sigma=.02, theta=1.1):
        self.nactions = nactions
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = np.mean(np.diff(np.linspace(0, 10, nsteps)))
        self.reset()

    def reset(self):
        self.noise = np.ones(self.nactions)

    def __call__(self):
        self.noise = self.noise + self.theta * (self.mu - self.noise) * self.dt \
            + self.sigma * np.random.normal() * np.sqrt(self.dt)
        return self.noise
