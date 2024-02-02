#!/usr/bin/env python
import numpy as np


def semiclassical_calculate_tau(Is, HFCs):
    I = [Is * (Is + 1) for Is in Is]
    HFC = [HFCs**2 for HFCs in HFCs]
    return (sum(np.array(I) * np.array(HFC)) / 6) ** -0.5
def tau(Is, HFCs):
    return sum(I * (

def angle(num_samples):
    for i in range(num_samples):
        while True:
            theta_r = np.random.rand() * np.pi
            s_r = np.random.rand()
            if s_r < np.sin(theta_r):
                theta = theta_r
                break

        phi = 2 * np.pi * np.random.rand()
        yield theta, phi


if __name__ == "__main__":
    sample_number = 10
    np.random.seed(42)
    theta, phi = semiclassical_theta_phi(sample_number)
    np.random.seed(42)
    for i, (t, p, (tt, pp)) in enumerate(zip(theta, phi, angle(sample_number))):
        print(f"{t-tt=} {p-pp=} {tt=} {pp=}")
