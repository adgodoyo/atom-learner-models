import numpy as np
import torch

from atom.utils.seeds import seed_everything, torch_generator


def test_seed_everything_makes_torch_reproducible():
    seed_everything(123)
    a = torch.rand(5)
    seed_everything(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_seed_everything_makes_numpy_reproducible():
    seed_everything(7)
    a = np.random.rand(5)
    seed_everything(7)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_different_seeds_differ():
    seed_everything(1)
    a = torch.rand(5)
    seed_everything(2)
    b = torch.rand(5)
    assert not torch.equal(a, b)


def test_torch_generator_is_seeded():
    g1 = torch_generator(99)
    g2 = torch_generator(99)
    assert torch.equal(torch.rand(3, generator=g1), torch.rand(3, generator=g2))
