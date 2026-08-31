"""Structural identifiability analysis for radical-pair models.

RadicalPy can already *fit* parameters — `radicalpy.utils.Bhalf_fit` and
`Bhalf_LFEhalf_fit` wrap `scipy.optimize.curve_fit`, and
`radicalpy.estimations` fits correlation functions. This module answers the
prior question: **can the data determine those parameters at all?**

A fit returns numbers whatever the data contains. When two parameters enter a
model only through a fixed combination, `curve_fit` still converges, the
covariance is near-singular, and the reported values are arbitrary points along
a flat valley. Standard errors from ``sqrt(diag(pcov))`` diagnose this only
indirectly and can be misleading when the degeneracy is exact.

The method here is sensitivity-rank analysis. Build the Jacobian of the
observable with respect to the parameters by central finite differences, then
take its singular value decomposition. A singular value whose ratio to the
largest falls below `DEGENERACY_TOLERANCE` marks a direction in parameter space
that no measurement of that observable can constrain — a *structural*, not
merely practical, non-identifiability. The corresponding right singular vector
names the offending parameter combination.

The distinction matters and is standard in systems biology and chemical
kinetics, where these tools are mature (Bellu et al., 2007; Raue et al., 2009);
this module ports them to radical-pair observables.

Functions:
        - `finite_difference_jacobian`: Central-difference sensitivity matrix.
        - `log_sensitivity_jacobian`: Sensitivity to log-parameters, appropriate
          for rate constants.
        - `analyze_jacobian`: Rank analysis of a sensitivity matrix.
        - `analyze_model`: Convenience wrapper over a forward model.

Classes:
        - `IdentifiabilityReport`: Singular values, rank, and null directions.

Key concepts:
        - **Structural vs practical**: A structural degeneracy is exact — the
          singular value is zero to machine precision and no amount of data or
          precision recovers it. A practical one is merely ill-conditioned.
          `DEGENERACY_TOLERANCE` separates them.
        - **Null direction**: A right singular vector with a vanishing singular
          value, expressed as a signed combination of parameter names. This is
          the *diagnosis*: it says which combination is unconstrained.
        - **Log-parameterisation**: For rate constants spanning orders of
          magnitude, ``dy/dk`` is the wrong sensitivity. It saturates at a
          non-zero constant as ``k -> 0`` (because ``d/dk exp(-k t) -> -t``), so
          an absolute Jacobian can declare an unmeasurably slow rate
          identifiable. ``k dy/dk`` vanishes linearly in ``k`` instead.

Usage pattern:
        1) Write the forward model as a callable mapping a parameter vector to
           an observable vector.
        2) Call `analyze_model` at the parameter values of interest.
        3) Inspect `rank` and, if degenerate, `describe_direction`.

References:
        Bellu, G., Saccomani, M. P., Audoly, S. & D'Angiò, L. DAISY: A new
        software tool to test global identifiability of biological and
        physiological systems. *Comput. Methods Programs Biomed.* **88**,
        52–61 (2007).

        Raue, A. et al. Structural and practical identifiability analysis of
        partially observed dynamical models by exploiting the profile
        likelihood. *Bioinformatics* **25**, 1923–1929 (2009).
"""

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

DEGENERACY_TOLERANCE: float = 1e-10
"""Singular-value ratio below which a direction counts as exactly degenerate."""

RELATIVE_STEP: float = 1e-5
"""Default central-difference step, relative to each parameter's magnitude.

Chosen near the optimum for central differences, h ~ eps**(1/3) ~ 6e-6, where
truncation error O(h**2) and roundoff error O(eps/h) balance. This is not
cosmetic: it sets the smallest singular-value ratio the method can resolve, and
therefore whether `DEGENERACY_TOLERANCE` is reachable.

An exactly degenerate model whose degenerate parameters have unequal magnitudes
receives unequal absolute steps, so its identical Jacobian columns differ at
O(h**2) rather than cancelling. At rel_step = 1e-4 that floor is ~6e-10,
above the tolerance, and such a model is wrongly reported as identifiable.
"""


def finite_difference_jacobian(
    function: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    rel_step: float = RELATIVE_STEP,
) -> np.ndarray:
    """Central-difference Jacobian of a forward model.

    Args:
            function (Callable): Forward model mapping a 1-D parameter vector to
                an observable vector.
            params (np.ndarray): Parameter values at which to evaluate the
                sensitivity.
            rel_step (float): Step size relative to each parameter's magnitude.

    Returns:
            np.ndarray: Sensitivity matrix with one row per observable sample and
            one column per parameter.

    Examples:
            >>> import numpy as np
            >>> J = finite_difference_jacobian(lambda p: np.array([p[0] * p[1]]),
            ...                                np.array([3.0, 5.0]))
            >>> np.allclose(J, [[5.0, 3.0]])
            True
    """
    params = np.asarray(params, dtype=float)
    if params.ndim != 1 or params.size == 0:
        raise ValueError("params must be a non-empty 1-D vector")

    scale = float(np.max(np.abs(params))) if np.any(params) else 1.0
    columns = []
    for index in range(params.size):
        magnitude = abs(params[index]) if params[index] != 0.0 else scale
        step = rel_step * magnitude
        forward = params.copy()
        backward = params.copy()
        forward[index] += step
        backward[index] -= step
        columns.append(
            (np.asarray(function(forward)) - np.asarray(function(backward)))
            / (2.0 * step)
        )

    return np.column_stack(columns)


def log_sensitivity_jacobian(
    function: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    rel_step: float = RELATIVE_STEP,
) -> np.ndarray:
    """Jacobian with respect to log-parameters: columns are ``k * dy/dk``.

    Preferred for rate constants. See the module docstring on
    log-parameterisation for why the absolute Jacobian misleads.

    Args:
            function (Callable): Forward model mapping parameters to observables.
            params (np.ndarray): Strictly positive parameter values.
            rel_step (float): Step size relative to each parameter's magnitude.

    Returns:
            np.ndarray: Log-sensitivity matrix.

    Examples:
            >>> import numpy as np
            >>> J = log_sensitivity_jacobian(lambda p: np.array([np.log(p[0])]),
            ...                              np.array([7.0]))
            >>> bool(np.allclose(J, [[1.0]]))
            True
    """
    params = np.asarray(params, dtype=float)
    if np.any(params <= 0.0):
        raise ValueError("log-sensitivity requires strictly positive parameters")

    return finite_difference_jacobian(function, params, rel_step=rel_step) * params[
        np.newaxis, :
    ]


@dataclass(frozen=True, eq=False)
class IdentifiabilityReport:
    """Outcome of a rank analysis on one observable and one parameter set.

    Attributes:
            parameter_names (Tuple[str, ...]): Names, ordered as the Jacobian
                columns.
            singular_values (np.ndarray): One per parameter, zero-padded when the
                Jacobian has fewer rows than columns.
            singular_value_ratios (np.ndarray): Singular values divided by the
                largest.
            rank (int): Number of directions the observable constrains.
            degenerate_directions (Tuple[np.ndarray, ...]): Right singular vectors
                spanning the unconstrained subspace.
            tolerance (float): Ratio below which a direction counted as degenerate.
    """

    parameter_names: Tuple[str, ...]
    singular_values: np.ndarray
    singular_value_ratios: np.ndarray
    rank: int
    degenerate_directions: Tuple[np.ndarray, ...]
    tolerance: float

    @property
    def is_degenerate(self) -> bool:
        """bool: Whether any parameter direction is unconstrained."""
        return self.rank < len(self.parameter_names)

    def describe_direction(self, direction: np.ndarray) -> str:
        """Render a null direction as a signed combination of parameter names.

        Args:
                direction (np.ndarray): A right singular vector.

        Returns:
                str: Human-readable combination, e.g. ``+0.707*k_a -0.707*k_b``.
        """
        terms = [
            f"{value:+.3f}*{name}"
            for name, value in zip(self.parameter_names, direction)
            if abs(value) > 1e-8
        ]
        return " ".join(terms)


def analyze_jacobian(
    jacobian: np.ndarray,
    parameter_names: Sequence[str],
    tolerance: float = DEGENERACY_TOLERANCE,
) -> IdentifiabilityReport:
    """Rank analysis of a sensitivity matrix.

    ``full_matrices=True`` is required so that an underdetermined Jacobian —
    fewer observable samples than parameters — still reports its null
    directions. A 1x2 sensitivity matrix has a genuine second singular value of
    zero, and the corresponding right singular vector is the flat direction of
    interest; the economy-size decomposition omits it.

    Args:
            jacobian (np.ndarray): Sensitivity matrix, parameters along columns.
            parameter_names (Sequence[str]): One name per column.
            tolerance (float): Singular-value ratio below which a direction is
                treated as exactly degenerate.

    Returns:
            IdentifiabilityReport: Singular values, rank, and null directions.

    Examples:
            A model in which two parameters appear only as their sum is
            degenerate, and the null direction says so:

            >>> import numpy as np
            >>> t = np.linspace(0, 1, 50)
            >>> model = lambda p: np.exp(-(p[0] + p[1]) * t)
            >>> report = analyze_model(model, np.array([1.0, 2.0]), ("k_a", "k_b"))
            >>> report.rank
            1
            >>> report.is_degenerate
            True
            >>> report.describe_direction(report.degenerate_directions[0])
            '+0.707*k_a -0.707*k_b'
    """
    jacobian = np.atleast_2d(np.asarray(jacobian, dtype=float))
    names = tuple(parameter_names)
    parameter_count = jacobian.shape[1]
    if parameter_count != len(names):
        raise ValueError(
            f"jacobian has {parameter_count} columns but "
            f"{len(names)} parameter names were given"
        )

    _, computed, right_vectors = np.linalg.svd(jacobian, full_matrices=True)

    singular_values = np.zeros(parameter_count, dtype=float)
    singular_values[: computed.size] = computed

    largest = singular_values[0] if singular_values.size else 0.0
    ratios = (
        singular_values / largest if largest > 0.0 else np.zeros_like(singular_values)
    )
    degenerate = ratios < tolerance

    return IdentifiabilityReport(
        parameter_names=names,
        singular_values=singular_values,
        singular_value_ratios=ratios,
        rank=int(np.count_nonzero(~degenerate)),
        degenerate_directions=tuple(
            right_vectors[i] for i in np.flatnonzero(degenerate)
        ),
        tolerance=tolerance,
    )


def analyze_model(
    function: Callable[[np.ndarray], np.ndarray],
    params: np.ndarray,
    parameter_names: Sequence[str],
    tolerance: float = DEGENERACY_TOLERANCE,
    rel_step: float = RELATIVE_STEP,
    log_space: bool = False,
) -> IdentifiabilityReport:
    """Rank analysis of a forward model at given parameter values.

    Args:
            function (Callable): Forward model mapping parameters to observables.
            params (np.ndarray): Parameter values to analyse.
            parameter_names (Sequence[str]): One name per parameter.
            tolerance (float): Degeneracy threshold on singular-value ratios.
            rel_step (float): Relative central-difference step.
            log_space (bool): Analyse sensitivity to log-parameters, appropriate
                for rate constants.

    Returns:
            IdentifiabilityReport: Singular values, rank, and null directions.

    Examples:
            A well-posed two-parameter fit is full rank:

            >>> import numpy as np
            >>> t = np.linspace(0, 1, 50)
            >>> model = lambda p: p[0] * np.exp(-p[1] * t)
            >>> analyze_model(model, np.array([2.0, 3.0]), ("A", "k")).rank
            2

            A double-Lorentzian MARY fit is not, once the two components
            overlap. `radicalpy.utils.double_Lorentzian` combines them with
            opposite signs, so at equal half-fields it collapses to
            ``(LFE_amplitude - amplitude) * shape``: only the difference is
            determined and the common mode is invisible. The null direction
            reports exactly that, and note its sign is the one you would get
            wrong by inspection.

            This is the form `radicalpy.utils.Bhalf_LFEhalf_fit` passes to
            `curve_fit`, so the degeneracy is reachable in ordinary use whenever
            a low-field feature sits close to the main one:

            >>> from radicalpy.utils import double_Lorentzian
            >>> B = np.linspace(0.1, 10.0, 60)
            >>> def overlapping(p):
            ...     return double_Lorentzian(B, p[0], 2.0, p[1], 2.0)
            >>> report = analyze_model(overlapping, np.array([1.0, 0.5]),
            ...                        ("amplitude", "LFE_amplitude"))
            >>> report.rank
            1
            >>> report.describe_direction(report.degenerate_directions[0])
            '+0.707*amplitude +0.707*LFE_amplitude'

            Separating the half-fields restores identifiability:

            >>> def separated(p):
            ...     return double_Lorentzian(B, p[0], 5.0, p[1], 0.5)
            >>> analyze_model(separated, np.array([1.0, 0.5]),
            ...               ("amplitude", "LFE_amplitude")).rank
            2
    """
    builder = log_sensitivity_jacobian if log_space else finite_difference_jacobian
    return analyze_jacobian(
        builder(function, params, rel_step=rel_step),
        parameter_names,
        tolerance=tolerance,
    )
