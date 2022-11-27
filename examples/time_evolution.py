#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import radicalpy as rp
from radicalpy.simulation import State


def main():
    flavin = rp.simulation.Molecule("flavin_anion", ["H25"])
    Z = rp.simulation.Molecule("Z")
    sim = rp.simulation.HilbertSimulation([flavin, Z])
    print(sim)
    #sim = rp.simulation.LiouvilleSimulation([flavin, Z])
    time = np.arange(0, 2e-6, 5e-9)
    Bs = np.arange(0, 1, 1)
    k = 3e6

    results = sim.MARY(
        init_state=State.SINGLET,
        obs_state=State.TRIPLET,
        time=time,
        B=Bs,
        D=0,
        J=0,
        kinetics=[rp.kinetics.Exponential(k)],
        #kinetics=[
         #   rp.kinetics.Haberkorn(k, State.SINGLET),
          #  rp.kinetics.HaberkornFree(k),
        #],
    )

    Bi = 0
    B = Bs[Bi]
    x = time * 1e6
    PY = results["product_yield_sums"]
    #print(results.keys())

    plt.plot(x, results["time_evolutions"][Bi], color="red", linewidth=2)
    plt.fill_between(x, results["product_yields"][Bi], color="blue", alpha=0.2)
    plt.xlabel("Time ($\mu s$)")
    plt.ylabel("Probability"); plt.ylim([0, 1])
    #plt.title(f"B={B} mT")
    plt.legend([r"$P_i(t)$", r"$\Phi_i$"])
    path = __file__[:-3] + f"_{B}.png"
    plt.savefig(path)
    #plt.show()
	
    print(f"Product yield = {PY}")


if __name__ == "__main__":
    main()
