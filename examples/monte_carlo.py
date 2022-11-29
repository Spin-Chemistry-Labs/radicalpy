#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import radicalpy as rp
from radicalpy import utils, classical

def main():
	n_steps = 200000
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
	
	t, r, J, D = classical.monte_carlo_exchange_dipolar(n_steps -1, r_min, del_T, pos[:,0], dist[0:-1], ang)
	
	t_convert = 1e3
	
	fig = plt.figure()
	ax = fig.add_axes([0, 0, 1, 1])
	ax.set_facecolor("none")
	ax.grid(False)
	plt.axis('on')
	plt.rc('axes',edgecolor='k')
	plt.plot(t / t_convert, r * 1e9, 'r')
	ax.set_title("Time evolution of radical pair separation", size=16)
	ax.set_xlabel('$t$ ($\mu s$)', size=14)
	ax.set_ylabel('$r$ (nm)', size=14)
	plt.tick_params(labelsize=14)
	path = __file__[:-3] + f"_{1}.png"
	plt.savefig(path)
	
	fig = plt.figure()
	ax = fig.add_axes([0, 0, 1, 1])
	ax.set_facecolor("none")
	ax.grid(False)
	plt.axis('on')
	plt.rc('axes',edgecolor='k')
	plt.plot(t / t_convert, J)
	ax.set_title("Time evolution of the exchange interaction", size=16)
	ax.set_xlabel('$t$ ($\mu s$)', size=14)
	ax.set_ylabel('$J$ (mT)', size=14)
	plt.tick_params(labelsize=14)
	path = __file__[:-3] + f"_{2}.png"
	plt.savefig(path)
	
	fig = plt.figure()
	ax = fig.add_axes([0, 0, 1, 1])
	ax.set_facecolor("none")
	ax.grid(False)
	plt.axis('on')
	plt.rc('axes',edgecolor='k')
	plt.plot(t / t_convert, D, 'g')
	ax.set_title("Time evolution of the dipolar interaction", size=16)
	ax.set_xlabel('$t$ ($\mu s$)', size=14)
	ax.set_ylabel('$D$ (mT)', size=14)
	plt.tick_params(labelsize=14)
	path = __file__[:-3] + f"_{3}.png"
	plt.savefig(path)
	
	acf_j = utils.autocorrelation(J, factor=2)
	acf_d = utils.autocorrelation(D, factor=2)
	
	t_tot = n_steps * del_T
	t = np.linspace(del_T, t_tot, len(acf_j))
	
	fig = plt.figure()
	ax = fig.add_axes([0, 0, 1, 1])
	ax.set_facecolor("none")
	ax.grid(False)
	plt.axis("on")
	plt.xscale("log")
	plt.rc("axes", edgecolor="k")
	plt.plot(t, acf_j, "b", label="J")
	ax.set_xlabel(r"$\tau$ (s)", size=14)
	ax.set_ylabel(r"$g_J(\tau)$", size=14)
	plt.tick_params(labelsize=14)
	ax.set_title("Autocorrelation: exchange interaction", size=16)
	path = __file__[:-3] + f"_{4}.png"
	plt.savefig(path)
	
	fig = plt.figure()
	ax = fig.add_axes([0, 0, 1, 1])
	ax.set_facecolor("none")
	ax.grid(False)
	plt.axis("on")
	plt.xscale("log")
	plt.rc("axes", edgecolor="k")
	plt.plot(t, acf_d, "g", label="D")
	ax.set_xlabel(r"$\tau$ (s)", size=14)
	ax.set_ylabel(r"$g_D(\tau)$", size=14)
	plt.tick_params(labelsize=14)
	ax.set_title("Autocorrelation: dipolar interaction", size=16)
	path = __file__[:-3] + f"_{5}.png"
	plt.savefig(path)
	
	
if __name__ == "__main__":
    main()