#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from radicalpy.classical import Rate, RateEquations, latex_eqlist_to_align, latexify, reaction_scheme
from pathlib import Path

def create_equation(Q):

    # FAD kinetic parameters  
    kex = Rate(1e4, "k_{ex}")  # groundstate excitation rate
    kfl = Rate(3.55e8, "k_{fl}")  # fluorescence rate
    kic = Rate(1.28e9, "k_{IC}")  # internal conversion rate
    kisc = Rate(3.64e8, "k_{ISC}")  # intersystem crossing rate
    kst = Rate(8e7, "k_{ST}")  # spin-state mixing rate
    kd = Rate(3e5, "k_d")  # protonated triplet to ground state
    k1 = Rate(7e6, "k_1")  # protonated triplet to RP
    km1 = Rate(2.7e9, "k_{-1}")  # RP to protonated triplet
    krt = Rate(1e9, "k^T_{Rlx}")  # triplet state relaxation rate
    kr = Rate(1.3e7, "k_R") # singlet recombination rate
    krlx = Rate(1.7e6, "k_{Rlx}") # RP relaxation
    pH = 2.3
    
    # Quenching kinetic parameters
    kq = Rate(1e9, "k_q") # quenching rate
    kp = Rate(3.3e3, "k_p") # free radical recombination

    Hp = Rate(10 ** (-1 * pH), "H^+")  # concentration of hydrogen ions
    
    S0, S1, Trpm, Tr0, S, Tpm, T0, Quencher = "S_0", "S_1", "T^*_{+/-}", "T^*_0", "S", "T_{+/-}", "T_0", "Quencher"

    base = {}
    base[S0] = {S0: -kex, Trpm: kd, Tr0: kd, S: kr, S1: kfl + kic, Quencher: kp}
    base[S1] = {S1: -(kfl + kic + 3 * kisc), S0: kex}
    base[Trpm] = {Trpm: -(kd + k1 + krt), Tpm: km1 * Hp, Tr0: 2 * krt, S1: 2 * kisc}
    base[Tr0] = {Tr0: -(kd + k1 + 2 * krt), T0: km1 * Hp, Trpm: krt, S1: kisc}
    base[Quencher] = {Quencher: -kp, Trpm: kq * Q, Tr0: kq * Q}    
    #base[Quencher] = {Quencher: -kp, S: kq * Q, Tpm: kq * Q, T0: kq * Q}

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
    # Kinetic simulation of FAD at pH 2.3 with Trp quencher.

    # Rate equations
    S0, S1, Trpm, Tr0, S, Tpm, T0, Quencher = "S_0", "S_1", "T^*_{+/-}", "T^*_0", "S", "T_{+/-}", "T_0", "Quencher"
    
    # quencher concentration    
    Q1 = [0, 0.1e-3, 0.5e-3, 1e-3, 1.5e-3]
    Q = [Rate(t, "[Q]") for t in Q1]


    initial_states = {Trpm: 2 / 3, Tr0: 1 / 3}
    time = np.linspace(0, 6e-6, 200)
    fac = 0.07
    scale = 1e6
    plt.clf()
    ax = plt.gca()

    for i in range(len(Q)):
        base, off, on = create_equation(Q[i])
        result_off = RateEquations({**base, **off}, time, initial_states)
        result_on = RateEquations({**base, **on}, time, initial_states)

        keys = [S, Tpm, T0, Quencher] + 2 * [Trpm, Tr0]
        field_off = fac * result_off[keys]
        field_on = fac * result_on[keys]
        delta_delta_A = field_on - field_off

        plt.plot(time * scale, delta_delta_A, linewidth=2)
        ax.set_xlabel("Time ($\mu s$)", size=14)
        ax.set_ylabel("$\Delta \Delta A$", size=14)
        ax.tick_params(labelsize=14)
        plt.gcf().set_size_inches(10, 5)

    ax.legend([f"[Trp] = {t*1000} mM" for t in Q1])
    path = __file__[:-3] + f"_{0}.png"
    plt.savefig(path)
    
    print(latex_eqlist_to_align(latexify(base)))
    print(latex_eqlist_to_align(latexify(off)))
    print(latex_eqlist_to_align(latexify(on)))
    #reaction_scheme(on)

if __name__ == "__main__":
    main()
    
