"""Physics-guided generative optimization of launch-vehicle acoustic liners.

Reading order for a newcomer: `physics` (one liner's acoustics and the reduced-order oracle) -> `design`
(how a liner is described, constrained and mapped to coordinates) -> `models` (the LinerGen generator) ->
`optimization` (CMA-ES and the LinerGen-seeded hybrid) -> `evaluation` (conditions, budgets, the study and
its statistics) -> `fem` and `cfd` (higher-fidelity checks) -> `vehicles` (the two launch-vehicle
applications) -> `visualization` and `provenance`.

All results are simulation-based.
"""
__version__ = "1.0.0"
