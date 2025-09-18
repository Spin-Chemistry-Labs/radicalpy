#!/usr/bin/env python
"""
Classical kinetics, diffusion, and utility helpers.

This module provides small, focused tools for building and simulating classical
(first-order) kinetic schemes, along with diffusion step estimates, LaTeX
formatting helpers, and basic 3D random-walk sampling.

Contents
--------
- Diffusion:
  - get_delta_r(D, dt): RMS relative displacement √(6 D Δt).

- LaTeX utilities:
  - latexify(rate_equations): Build d[Xi]/dt = … strings from a rate map.
  - latex_eqlist_to_align(eqs): Wrap equations in an `align*` environment.

- Kinetics:
  - Rate: Numeric value + LaTeX label with operator overloading that carries
    symbolic expressions through +, −, ×, ÷.
  - RateEquations: Sparse representation and time evolution (matrix exponential)
    for first-order networks; returns EquationRateResult for easy access.
  - EquationRateResult: Convenience accessor to sum populations of one or more
    states over time.

- Visualization:
  - reaction_scheme(path, rate_equations): Generate a LaTeX (dot2tex) diagram
    of the reaction network.

- Random sampling:
  - random_theta_phi(n): Uniform sampling on the unit sphere.
  - randomwalk_3d(...): Monte-Carlo 3D random walk with min/ max-distance
    constraints (solution or spherical microreactor).

Key conventions
---------------
- Rate maps use:  sink_state -> { source_state: Rate|float, ... }
  The resulting sparse matrix M stores rates at (sink, source), so dP/dt = M P.

- Time evolution assumes uniform spacing in `time` and advances via the matrix
  exponential of the rate matrix over Δt.

Quick example
-------------
>>> # Define a simple A -> B with rate k
>>> k = Rate(1.0, "k_{AB}")
>>> req = {"A": {"A": -k}, "B": {"A": k}}
>>> net = RateEquations(req)
>>> t = np.linspace(0, 5.0, 501)
>>> res = net.time_evolution(t, {"A": 1.0, "B": 0.0})
>>> A_t, B_t = res["A"], res["B"]

Dependencies
------------
Relies on NumPy and SciPy (sparse) for numerics, graphviz + dot2tex for
LaTeX diagram generation.
"""

from pathlib import Path
from typing import Tuple

import dot2tex
import graphviz
import numpy as np
import scipy as sp
from numpy.typing import ArrayLike

from . import utils


def get_delta_r(mutual_diffusion: float, delta_T: float) -> float:
    """Root-mean-square step length for relative diffusion.

    Uses ⟨Δr²⟩ = 6 D Δt for three-dimensional relative motion of two
    particles (mutual diffusion).

    Args:
            mutual_diffusion (float): Mutual diffusion coefficient, D (m²/s).

            delta_T (float): Time interval, Δt (s).

    Returns:
            float: RMS displacement, √(6 D Δt) in meters (m).
    """
    return np.sqrt(6 * mutual_diffusion * delta_T)


def latexify(rate_equations: dict):
    """Convert rate equations into LaTeX strings.

    Produces a list of differential equations in LaTeX format,
    with terms written as sums over outgoing edges.

    Args:
            rate_equations (dict): Dictionary mapping state labels (LHS) to
                dictionaries of transitions (RHS). Each RHS dict maps target
                states to `Rate` objects.

    Returns:
        list[str]: List of LaTeX-formatted rate equations.
    """
    result = []
    for lhs_data, rhs_data in rate_equations.items():
        lhs = f"\\frac{{d[{lhs_data}]}}{{dt}} "
        rhs_list = [f"{edge.label} [{vertex}]" for vertex, edge in rhs_data.items()]
        rhs = " + ".join(rhs_list)
        result.append(f"{lhs} = {rhs}")
    return result


def latex_eqlist_to_align(eqs: list):
    """Format LaTeX equations for alignment.

    Wraps a list of LaTeX equations in an `align*` environment
    and inserts alignment markers before equal signs.

    Args:
        eqs (list[str]): List of LaTeX equation strings.

    Returns:
        str: A single LaTeX string with all equations aligned.
    """
    body = " \\\\\n".join(map(lambda t: t.replace("=", "&="), eqs))
    return f"\\begin{{align*}}\n{body}\n\\end{{align*}}"


class Rate:
    """Scalar rate value paired with a LaTeX-formatted symbolic label.

    Stores a numeric rate (or expression value) and a human-readable/LaTeX
    label that tracks algebraic manipulations via operator overloading.
    """

    value: float
    label: str
    """LaTeX representation of the rate constant or expression."""

    def __repr__(self):
        """Readable representation showing the label and numeric value.

        Returns:
            str: String of the form "<label> = <value>".
        """
        return f"{self.label} = {self.value}"

    def __init__(self, value: float, label: str):  # noqa D102
        """Create a new `Rate`.

        Args:
            value (float): Numeric value of the rate/expression.
            label (str): LaTeX or plain-text symbol (e.g., "k_1", "k_{AB}").
        """
        self.value = value
        self.label = label

    @staticmethod
    def _get_value_label(v):
        """Return numeric value and label for either `Rate` or scalar.

        Args:
            v (Rate | float): Operand to unpack.

        Returns:
            tuple[float, str]: `(value, label)` where `label` is the symbol
            if `v` is a `Rate`, otherwise a stringified scalar.
        """
        return (v.value, v.label) if isinstance(v, Rate) else (v, v)

    def __rmul__(self, v):
        """Left multiplication by scalar or `Rate`.

        Places the left operand's label before this label to keep
        expressions readable (e.g., "2 k" or "k_a k_b").

        Args:
            v (Rate | float): Left operand.

        Returns:
            Rate: Product with value `v * self.value` and label
            `"<v.label> <self.label>"` (or scalar string for `v`).
        """
        value, label = self._get_value_label(v)
        return Rate(self.value * value, f"{label} {self.label}")

    def __mul__(self, v):
        """Right multiplication by scalar or `Rate`.

        Mirrors `__rmul__` so that scalar–Rate and Rate–scalar both produce
        the same readable label ordering.

        Args:
            v (Rate | float): Right operand.

        Returns:
            Rate: Product `self * v` (delegates to `__rmul__`).
        """
        # When multiplying with a constant, this puts the constant
        # before the variable.
        return self.__rmul__(v)

    def __radd__(self, v):
        """Left addition with scalar or `Rate`.

        Args:
            v (Rate | float): Left operand.

        Returns:
            Rate: Sum with value `v + self.value` and label
            `"<v.label> + <self.label>"` (or scalar string for `v`).
        """
        value, label = self._get_value_label(v)
        return Rate(value + self.value, f"{label} + {self.label}")

    def __add__(self, v):
        """Right addition with scalar or `Rate`.

        Args:
            v (Rate | float): Right operand.

        Returns:
            Rate: Sum with value `self.value + v` and label
            `"<self.label> + <v.label>"` (or scalar string for `v`).
        """
        value, label = self._get_value_label(v)
        return Rate(self.value + value, f"{self.label} + {label}")

    def __neg__(self):
        """Unary negation.

        Returns:
            Rate: Negated value with label wrapped as `"-(<label>)"`.
        """
        return Rate(-self.value, f"-({self.label})")

    def __truediv__(self, v):
        """Division by scalar or `Rate`.

        Args:
            v (Rate | float): Denominator.

        Returns:
            Rate: Quotient with value `self.value / v` and label
            `"<self.label> / <v.label>"` (or scalar string for `v`).
        """
        value, label = self._get_value_label(v)
        return Rate(self.value / value, f"{self.label} / {label}")


class EquationRateResult:
    """Container for rate-equation time evolution results."""

    def __init__(self, result: np.ndarray, indices: dict):
        """Initialise result wrapper.

        Args:
            result (np.ndarray): Array of state populations with shape
                (n_times, n_states).

            indices (dict): Mapping of state labels to column indices.
        """
        self.result = result
        self.indices = indices

    def __getitem__(self, keys: list) -> np.ndarray:
        """Extract summed populations for one or more states.

        Args:
            keys (str | list[str]): One state label or a list of labels.

        Returns:
            np.ndarray: Population trajectory over time for the requested
            state(s), shape (n_times,).
        """
        ks = [keys] if isinstance(keys, str) else keys
        return np.sum([self.result[:, self.indices[k]] for k in ks], axis=0)


class RateEquations:
    """Representation and solver for first-order kinetic rate equations."""

    def __init__(self, rate_equations: dict):
        """Build a sparse matrix representation from a reaction network.

        Args:
            rate_equations (dict): Nested dictionary mapping sink states
                (outer keys) to dicts of source states (inner keys) with
                `Rate` objects or floats as values.
        """
        self.rate_equations = rate_equations
        inner_keys = [list(v.keys()) for v in rate_equations.values()]
        outer_keys = list(rate_equations.keys())
        if set(outer_keys) != set(sum(inner_keys, [])):
            print("WARNING:The set of sources and sinks is different!")
        # all_keys = list(set(sum(inner_keys, outer_keys)))
        self.indices = {k: i for i, k in enumerate(outer_keys)}
        self._construct_matrix()

    @property
    def all_keys(self) -> list:
        """List of all state labels in the network.

        Returns:
            list[str]: Names of all states included in the equations.
        """
        return list(self.indices.keys())

    def are_valid_keys(self, keys: dict) -> bool:
        """Check whether a set of state labels is valid.

        Args:
            keys (iterable[str]): Collection of state labels to check.

        Returns:
            bool: True if all labels are recognised states.
        """
        return set(keys).issubset(self.all_keys)

    def check_initial_states(self, initial_states: dict):
        """Validate an initial state population dictionary.

        Ensures all keys are valid states and populations sum to 1.

        Args:
            initial_states (dict): Mapping from state label to initial
                population (fractions).

        Raises:
            ValueError: If unknown state labels are included.

        Notes:
            Prints a warning if the populations do not sum to unity.
        """
        if not self.are_valid_keys(initial_states.keys()):
            raise ValueError("Unknown keys specified in `initial_states`")
        if sum(initial_states.values()) != 1:
            print("WARNING: Initial state values don't sum up to 1")

    def _construct_matrix(self):
        """Assemble the sparse transition-rate matrix.

        Builds a compressed sparse column (CSC) matrix where entry
        (i, j) contains the rate from state j → i.

        Returns:
            None: Populates `self.matrix` internally.
        """
        tmp = [
            (v.value if isinstance(v, Rate) else v, self.indices[i], self.indices[j])
            for i, d in self.rate_equations.items()
            for j, v in d.items()
        ]
        data, row_ind, col_ind = zip(*tmp)
        N = len(self.all_keys)
        self.matrix = sp.sparse.csc_matrix((data, (row_ind, col_ind)), (N, N))

    def time_evolution(
        self, time: np.ndarray, initial_states: dict
    ) -> EquationRateResult:
        """Evolve populations in time under the rate equations.

        Solves dP/dt = M P using matrix exponentials at fixed time steps.

        Args:
            time (np.ndarray): Array of time points (uniform spacing).
            initial_states (dict): Mapping of state → initial population.
                Must sum to 1.

        Returns:
            EquationRateResult: Object containing the full population
            time series and state index mapping.
        """
        self.check_initial_states(initial_states)
        dt = time[1] - time[0]
        propagator = sp.sparse.linalg.expm(self.matrix * dt)
        result = np.zeros([len(time), len(self.all_keys)], dtype=float)
        result[0] = [initial_states.get(k, 0) for k in self.all_keys]
        for t in range(1, len(time)):
            result[t] = propagator @ result[t - 1]
        return EquationRateResult(result, self.indices)


def reaction_scheme(path: str, rate_equations: dict):
    """Generate a LaTeX reaction scheme diagram using Graphviz + dot2tex.

    Builds a directed graph representation of the rate equations and
    writes the LaTeX TikZ code to a file.

    Args:
        path (str): Output file path. Appends `.tex` if not present.

        rate_equations (dict): Dictionary mapping states to outgoing
            transitions, with `Rate` objects providing LaTeX labels.

    Returns:
        None: Writes a `.tex` file containing the diagram source.
    """
    data = [
        (v1, v2, edge.label)
        for v1, rhs_data in rate_equations.items()
        for v2, edge in rhs_data.items()
    ]
    G = graphviz.Digraph("G")
    for v1, v2, edge in data:
        if not edge.startswith("-"):
            # TODO: add only if not present already
            # TODO: check position
            # TODO: use sympy (how to check negative)
            G.node(v1, texlbl=f"${v1}$")
            G.node(v2, texlbl=f"${v2}$")
            G.edge(v2, v1, edge, texlbl=f"${edge}$")

    if not path.endswith("tex"):
        path += ".tex"
    texcode = dot2tex.dot2tex(G.source)
    Path(path).write_text(texcode)


def random_theta_phi(num_samples: int = 1) -> ArrayLike:
    """Randomly sample polar (θ) and azimuthal (φ) angles.

    Samples directions uniformly over the unit sphere using inverse
    transform sampling.

    Args:
        num_samples (int, optional): Number of random angle pairs to
            generate. Defaults to 1.

    Returns:
        np.ndarray: Array of shape (2, num_samples) containing sampled
        angles in radians:
        - [0]: θ ∈ [0, π] (polar angle)
        - [1]: φ ∈ [0, 2π) (azimuthal angle)
    """
    phi = np.random.uniform(0, 2 * np.pi, size=num_samples)
    theta = np.arccos(np.random.uniform(-1, 1, size=num_samples))
    return np.array([theta, phi])


def randomwalk_3d(
    n_steps: int,
    x_0: float,
    y_0: float,
    z_0: float,
    delta_r: float,
    r_min: float,
    r_max: float = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a 3D Monte Carlo random walk of a radical pair.

    Models diffusion of a radical in solution (`r_max = 0`) or inside
    a spherical microreactor (`r_max > 0`), enforcing reflective
    boundaries and minimum approach distance.

    Args:

            n_steps (int): Number of simulation steps.

            x_0 (float): Initial x-coordinate (m).

            y_0 (float): Initial y-coordinate (m).

            z_0 (float): Initial z-coordinate (m).

            delta_r (float): Step length, e.g., from `get_delta_r` (m).

            r_min (float): Minimum allowed radical–radical distance (m).

            r_max (float, optional): Maximum radius of microreactor (m). Set
                to 0 for solution-based, and to a positive value for
                microreactor-based simulations.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - pos (ndarray): Radical positions at each step, shape (n_steps, 3).
            - dist (ndarray): Radical–radical distances at each step, shape (n_steps,).
            - angle (ndarray): Polar angles (θ) of displacements, shape (n_steps,).

    """
    if r_max != 0 and r_min > r_max:
        raise ValueError("r_min should be less than (or equal to) r_max.")
    if r_min < 0 or r_max < 0:
        raise ValueError("r_min and r_max should not be negative.")
    pos = np.zeros([n_steps, 3])
    dist = np.zeros(n_steps)
    angle = np.zeros(n_steps)

    pos[0] = np.array([x_0, y_0, z_0])
    dist[0] = np.linalg.norm(pos[0])
    if np.linalg.norm(pos[0]) <= r_min:
        raise ValueError("Molecule starting distance is needs to be > r_min.")

    for i in range(1, n_steps):
        theta, phi = random_theta_phi()
        new_pos = pos[i - 1] + delta_r * utils.spherical_to_cartesian(theta, phi)
        d = np.linalg.norm(new_pos)
        while (r_max > 0 and d >= r_max - r_min) or d <= r_min + r_min:
            theta, phi = random_theta_phi()
            new_pos = pos[i - 1] + delta_r * utils.spherical_to_cartesian(theta, phi)
            d = np.linalg.norm(new_pos)
        angle[i] = theta
        pos[i] = new_pos
        dist[i] = d
    return pos, dist, angle
