#!/usr/bin/env python

# HFC experiments
import sys

sys.path.insert(0, "..")  ##############################################

from functools import singledispatchmethod  # noqa E402
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from radicalpy import data  # noqa E402
from radicalpy.utils import spherical_to_cartesian


def main():
    rng = np.random.default_rng()
    num_samples = 4000000
    num_molecules = 2
    num_dims = 3
    samples = np.random.multivariate_normal(
        mean=[0, 0],
        cov=np.diag([1, 2]),
        size=(num_dims, num_samples),
    )
    # axis, num_samples, num_molecules
    samples_molecule0 = samples[:, :, 0]
    samples_molecule1 = samples[:, :, 1]
    norm = np.linalg.norm(samples_molecule0, axis=0)
    print(f"{np.std(norm)=}")
    print(f"{np.var(norm)=}")

    p = np.array(
        [
            np.diag([1, 1, 1]),
            np.diag([2, 2, 2]),
            np.diag([3, 3, 3]),
        ]
    )
    print(f"{p.shape=}")
    print(f"{samples.shape=}")
    samples[:, 0, 0] = [100, 10, 1]
    prod = np.einsum("anm,axy->nmxy", samples, p)
    print(f"{prod.shape=}")
    print(prod[0, 0, :, :])


def dist():
    rng = np.random.default_rng()
    num_samples = 40000
    num_dims = 3
    samples = np.random.multivariate_normal(
        mean=[0, 0, 0],
        cov=np.diag([1, 2, 3]),
        size=(num_samples),
    )
    print(f"{samples.shape=}")
    norm = np.linalg.norm(samples, axis=1)
    print(f"{norm.shape=}")
    print(f"{np.std(norm)**2=}")
    print(f"{np.var(norm)**2=}")


def randang():
    from radicalpy.classical import new_random_theta_phi, random_theta_phi

    np.random.seed(42)
    ra = random_theta_phi()
    print(f"{ra=}")
    np.random.seed(420)
    ra = new_random_theta_phi(1000)
    print(f"{ra=}")
    t, p = new_random_theta_phi()
    print(f"{t,p=}")
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*spherical_to_cartesian(ra[0], ra[1]))

    old = []
    for _ in range(50):
        old.append(random_theta_phi())
    old = np.array(old).T
    # ax.scatter(*spherical_to_cartesian(old[0], old[1]))
    # fig.savefig("wip_experiment.png")
    plt.show()


def surface():
    n = 500
    samples = np.random.normal(loc=0, scale=1, size=(3, n))
    norm = np.linalg.norm(samples, axis=0)
    ns = samples / norm
    fig = plt.figure()
    # ax = fig.add_subplot(2, 2, 1, projection="3d")
    # ax.scatter(*ns)
    # ax = fig.add_subplot(2, 2, 3, projection="3d")
    # ax.scatter(*ns)
    # ax.scatter(*samples)
    ax = fig.add_subplot(2, 2, 2)
    ax.hist(norm, density=True)
    I = np.linspace(0, 5, 20)
    tau2 = 4
    fI = (tau2 / (4 * np.pi)) ** (3 / 2) * np.exp(-1 / 4 * I**2 * tau2)
    ax.plot(I, fI)
    plt.show()


def algebra():
    from sympy import exp, pi, sqrt, symbols
    from sympy.abc import T, h, l, n, sigma, tau, x

    pexpr = (tau**2 / (4 * pi)) ** (3 / 2) * exp(-1 / 4 * x**2 * tau**2)
    pexpr = pexpr.subs(x, h)
    print(pexpr)

    bexpr = (3 / (2 * pi * n * l**2)) ** (3 / 2) * exp(-(3 * h**2) / (2 * n * l**2))
    bexpr = bexpr.subs(n * l**2, 6 * tau**-2)
    print(bexpr)

    pb_delta = (pexpr - bexpr).simplify()
    print(f"{pb_delta=}")

    nexpr = 1 / ((sigma**2 * 2 * pi) ** (1 / 2)) * exp(-1 / 2 * x**2 / sigma**2)
    nexpr = nexpr.subs(sigma**2, 2 * T).subs(x, h)
    print(nexpr**3)
    pn_delta = (pexpr / nexpr**3).simplify().expand()
    print(f"{pn_delta=}")
    # 1/tau2 == n*l2 / 6


if __name__ == "__main__" or True:
    # main()
    # dist()
    # randang()
    # surface()
    algebra()
