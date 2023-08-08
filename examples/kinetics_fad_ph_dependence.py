#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from radicalpy.classical import Rate, RateEquations, latex_eqlist_to_align, latexify, reaction_scheme
from pathlib import Path

def create_equation(kr, krlx, pH):

    kex = Rate(1e4, "k_{ex}")  # groundstate excitation rate
    kfl = Rate(3.55e8, "k_{fl}")  # fluorescence rate
    kic = Rate(1.28e9, "k_{IC}")  # internal conversion rate
    kisc = Rate(3.64e8, "k_{ISC}")  # intersystem crossing rate
    kst = Rate(8e7, "k_{ST}")  # spin-state mixing rate
    kd = Rate(3e5, "k_d")  # protonated triplet to ground state
    k1 = Rate(7e6, "k_1")  # protonated triplet to RP
    km1 = Rate(2.7e9, "k_{-1}")  # RP to protonated triplet
    krt = Rate(1e9, "k^T_{Rlx}")  # triplet state relaxation rate

    # Quenching kinetic parameters
    kq = Rate(0, "k_q")  # 1e9  # quenching rate
    kfr = Rate(0, "k_{FR}")  # 3.3e3  # free radical recombination
    Q = Rate(0, "[Q]")  # 1e-3  # quencher concentration

    Hp = Rate(10 ** (-1 * pH), "H^+")  # concentration of hydrogen ions
    
    S0, S1, Trpm, Tr0, S, Tpm, T0, FR = "S_0", "S_1", "T^*_{+/-}", "T^*_0", "S", "T_{+/-}", "T_0", "FR"

    base = {}
    base[S0] = {S0: -kex, Trpm: kd, Tr0: kd, S: kr, S1: kfl + kic, FR: kfr}
    base[S1] = {S1: -(kfl + kic + 3 * kisc), S0: kex}
    base[Trpm] = {Trpm: -(kd + k1 + krt), Tpm: km1 * Hp, Tr0: 2 * krt, S1: 2 * kisc}
    base[Tr0] = {Tr0: -(kd + k1 + 2 * krt), T0: km1 * Hp, Trpm: krt, S1: kisc}
    base[FR] = {FR: -kfr, Trpm: kq * Q, Tr0: kq * Q}
    #base[FR] = {FR: -kfr, S: kq * Q, Tpm: kq * Q, T0: kq * Q}

    off = {}
    off[S] = {S: -(3 * kst + kr), Tpm: kst, T0: kst}
    off[Tpm] = {Tpm: -(2 * kst + km1 * Hp), S: 2 * kst, T0: 2 * kst, Trpm: k1}
    off[T0] = {T0: -(3 * kst + km1 * Hp), S: kst, Tpm: kst, Tr0: k1}

    on = {}
    on[S] = {S: -(2 * krlx + kst + kr), Tpm: krlx, T0: kst}
    on[Tpm] = {Tpm: -(2 * krlx + km1 * Hp), S: 2 * krlx, T0: 2 * krlx, Trpm: k1}
    on[T0] = {T0: -(2 * krlx + kst + km1 * Hp), S: kst, Tpm: krlx, Tr0: k1}

    return base, off, on

def main():
    # Kinetic simulation of FAD from pH 1.9 to 3.5.

    # Rate equations
    S0, S1, Trpm, Tr0, S, Tpm, T0, FR = "S_0", "S_1", "T^*_{+/-}", "T^*_0", "S", "T_{+/-}", "T_0", "FR"

    # FAD kinetic parameters
    # singlet recombination rate
    kr = [1e7, 1.1e7, 1.3e7, 1.8e7, 1.9e7, 2.3e7, 2.7e7, 3.3e7, 3.75e7] # "k_R"
    kr = [Rate(t, "k_R") for t in kr]

    # RP relaxation
    krlx = [1e6, 1.3e6, 1.7e6, 3.7e6, 4.1e6, 5.5e6, 6.8e6, 8.3e6, 10.8e6] # "k_{Rlx}"
    krlx = [Rate(t, "k_{Rlx}") for t in krlx]

    pH = [1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5] # "pH"

    initial_states = {Trpm: 2 / 3, Tr0: 1 / 3}
    time = np.linspace(0, 6e-6, 200)
    fac = 0.07
    scale = 1e6
    plt.clf()
    ax = plt.gca()

    for i in range(len(pH)):
        base, off, on = create_equation(kr[i], krlx[i], pH[i])
        result_off = RateEquations({**base, **off}, time, initial_states)
        result_on = RateEquations({**base, **on}, time, initial_states)

        keys = [S, Tpm, T0, FR] + 2 * [Trpm, Tr0]
        field_off = fac * result_off[keys]
        field_on = fac * result_on[keys]
        delta_delta_A = field_on - field_off

        plt.plot(time * scale, delta_delta_A, linewidth=2)
        ax.set_xlabel("Time ($\mu s$)", size=14)
        ax.set_ylabel("$\Delta \Delta A$", size=14)
        ax.tick_params(labelsize=14)
        plt.gcf().set_size_inches(10, 5)

    ax.legend([f"pH = {t}" for t in pH])
    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path)
    
    print(latex_eqlist_to_align(latexify(base)))
    print(latex_eqlist_to_align(latexify(off)))
    print(latex_eqlist_to_align(latexify(on)))
    #reaction_scheme(on)

if __name__ == "__main__":
    main()
