import numpy as np

class GaussianNoise():
    def __init__(self, size, mu, sigma, clip=None):
        self.noise = None
        self.size = size

        self.sigma = sigma
        self.mu = mu
        self.clip = clip

    def sample(self):
        noise = np.random.normal(size=self.size, loc=self.mu, scale=self.sigma)

        if self.clip is not None:
            noise = np.clip(noise, a_min=-self.clip, a_max=self.clip)

        return noise