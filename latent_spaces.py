from typing import Callable, List

import torch

from spaces import Space


class LatentSpace:
    """Combines a topological space with a marginal and conditional density to sample from."""

    def __init__(self, space: Space, sample_latent: Callable):
        self.space = space
        self._sample_latent = sample_latent

    @property
    def sample_latent(self):
        if self._sample_latent is None:
            raise RuntimeError("sample_marginal was not set")
        return lambda *args, **kwargs: self._sample_latent(self.space, *args, **kwargs)

    @sample_latent.setter
    def sample_latent(self, value: Callable):
        assert callable(value)
        self._sample_latent = value

    @property
    def dim(self):
        return self.space.dim


class ProductLatentSpace(LatentSpace):
    """A latent space which is the cartesian product of other latent spaces."""

    def __init__(self, spaces: List[LatentSpace]):
        self.spaces = spaces

    """
    def sample_conditional(self, z, size, **kwargs):
        x = []
        n = 0
        for s in self.spaces:
            if len(z.shape) == 1:
                z_s = z[n : n + s.space.n]
            else:
                z_s = z[:, n : n + s.space.n]
            n += s.space.n
            x.append(s.sample_conditional(z=z_s, size=size, **kwargs))

        return torch.cat(x, -1)
    """

    def sample_latent(self, size, **kwargs):
        x = [s.sample_latent(size=size, **kwargs) for s in self.spaces]

        return torch.cat(x, -1)

    @property
    def dim(self):
        return sum([s.dim for s in self.spaces])
