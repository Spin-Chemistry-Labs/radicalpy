#! /usr/bin/env python

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import radicalpy as rp


def main():
    #def kinetics(time, initial_populations, states, rate_equations):
    #    shape = (len(states), len(states))
    #    arrange = [
    #        rate_equations[i][j] if (i in rate_equations and j in rate_equations[i]) else 0
    #        for i in states
    #        for j in states
    #    ]
    #    rates = np.reshape(arrange, shape)
    #    dt = time[1] - time[0]
    #    result = np.zeros([len(time), *rates[0].shape], dtype=float)
    #    propagator = sp.sparse.linalg.expm(sp.sparse.csc_matrix(rates) * dt)
    #    result[0] = initial_populations
    #    for t in range(1, len(time)):
    #        result[t] = propagator @ result[t - 1]
    #    return result

    # Kinetic simulation of time-resolved optical absorption of FAD at pH 2.3.
    # For FAD quenching: change the parameter `kq` from 0 to the commented value (`1e9`).

    # Rates
    kex = 1e4  # groundstate excitation
    kfl = 3.55e8  # fluorescence
    kic = 1.28e9  # internal conversion
    kisc = 3.64e8  # intersystem crossing
    khfc = 8e7  # spin-state mixing
    kd = 3e5  # protonated triplet to ground state
    k1 = 7e6  # protonated triplet to RP
    km1 = 2.7e9  # RP to protonated triplet
    krt = 1e9  # triplet state relaxation
    kq = 0  # 1e9  # quenching
    kp = 3.3e3  # free radical recombination
    Q = 3e-3  # quencher concentration
    kbet = 1.3e7  # singlet recombination
    kr = 1.7e6  # RP relaxation
    pH = 2.3
    Hp = 10 ** (-1 * pH)  # concentration of hydrogen ions

    # Rate equations
    base = {}
    base["S0"] = {
        "S0": -kex,
        "T+/-": kd,
        "T0": kd,
        "S": kbet,
        "S*": (kfl + kic),
        "FR": kp,
    }
    base["S*"] = {
        "S*": -(kfl + kic + 3 * kisc),
        "S0": kex,
    }
    base["T*+/-"] = {
        "T*+/-": -(kd + k1 + krt),
        "T+/-": (km1 * Hp),
        "T*0": (2 * krt),
        "S*": (2 * kisc),
    }
    base["T*0"] = {
        "T*0": -(kd + k1 + 2 * krt),
        "T0": (km1 * Hp),
        "T*+/-": krt,
        "S*": kisc,
    }
    base["FR"] = {
        "FR": -kp,
        "S": (kq * Q),
        "T+/-": (kq * Q),
        "T0": (kq * Q),
    }

    off = {}
    off["S"] = {
        "S": -(3 * khfc + kbet),
        "T+/-": khfc,
        "T0": khfc,
    }
    off["T+/-"] = {
        "T+/-": -(2 * khfc + km1 * Hp),
        "S": (2 * khfc),
        "T0": (2 * khfc),
        "T*+/-": k1,
    }
    off["T0"] = {
        "T0": -(3 * khfc + km1 * Hp),
        "S": khfc,
        "T+/-": khfc,
        "T*0": k1,
    }

    on = {}
    on["S"] = {
        "S": -(2 * kr + khfc + kbet),
        "T+/-": kr,
        "T0": khfc,
    }
    on["T+/-"] = {
        "T+/-": -(2 * kr + km1 * Hp),
        "S": (2 * kr),
        "T0": (2 * kr),
        "T*+/-": k1,
    }
    on["T0"] = {
        "T0": -(2 * kr + khfc + km1 * Hp),
        "S": khfc,
        "T+/-": kr,
        "T*0": k1,
    }

    my_states = ["S0", "S*", "T*+/-", "T*0", "S", "T+/-", "T0", "FR"]
    initial = [0, 0, 0, 0, 0, 2 / 3, 1 / 3, 0]
    time = np.linspace(0, 3.5e-6, 200)

    rates_off = {**base, **off}
    rates_on = {**base, **on}
    result_off = kinetics(time, initial, my_states, rates_off)
    result_on = kinetics(time, initial, my_states, rates_on)
    fac = 0.07

    triplet_off = result_off[:, 2] + result_off[:, 3]
    radical_pair_off = result_off[:, 4] + result_off[:, 5] + result_off[:, 6]
    free_radical_off = result_off[:, 7]
    field_off = fac * (radical_pair_off + 2 * triplet_off + free_radical_off)

    triplet_on = result_on[:, 2] + result_on[:, 3]
    radical_pair_on = result_on[:, 4] + result_on[:, 5] + result_on[:, 6]
    free_radical_on = result_on[:, 7]
    field_on = fac * (radical_pair_on + 2 * triplet_on + free_radical_on)

    delta_delta_A = field_on - field_off

    plt.clf()
    plt.grid(False)
    plt.axis("on")
    plt.rc("axes", edgecolor="k")
    scale = 1e6
    plt.plot(time * scale, field_off, color="red", linewidth=2)
    plt.plot(time * scale, field_on, color="blue", linewidth=2)
    plt.plot(time * scale, delta_delta_A, color="green", linewidth=2)
    plt.legend([r"$\Delta A (B_0 = 0)$", r"$\Delta A (B_0 \neq 0)$", r"$\Delta \Delta A$"])
    plt.xlabel("Time ($\mu s$)", size=14)
    plt.ylabel("$\Delta A$", size=14)
    plt.tick_params(labelsize=14)
    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path)

if __name__ == "__main__":
    main()
