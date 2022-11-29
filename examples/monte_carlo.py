#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import radicalpy as rp
from radicalpy import utils, classical

def main():
	n_steps = 4000
	r_min = 0.5e-9 / 2
	r_max = 2e-9 / 2
	r = (r_min) + np.random.sample() * ((r_max) - (r_min))
	x0, y0, z0 = r, 0, 0
	mut_D = 1e-6 / 10000
	del_T = 40e-12
	
	delta_r = classical.get_delta_r(mut_D, del_T)
	pos, dist, ang = classical.randomwalk_3d(n_steps, x0, y0, z0, delta_r, r_max)
	classical.plot_sphere(pos, r_max)
	
	path = __file__[:-3] + f"_{0}.png"
	plt.savefig(path)
  

if __name__ == "__main__":
    main()