#!/usr/bin/env python


def steady_state_mary(sim, D, E):
    H = sim.zero_field_splitting_hamiltonian(D, E)
    return H
