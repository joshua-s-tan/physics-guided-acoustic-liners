"""The FEM toolchain (scikit-fem, gmsh, meshio), imported on first use, and the axisymmetric (m = 0)
r-weighted bilinear and linear forms shared by the mesh measures and the solver.
"""
from __future__ import annotations



def _lazy():
    import skfem
    from skfem import BilinearForm, LinearForm, FacetBasis
    from skfem.helpers import dot, grad
    from scipy.sparse.linalg import spsolve
    return skfem, BilinearForm, LinearForm, FacetBasis, dot, grad, spsolve


def _forms():
    skfem, BilinearForm, LinearForm, FacetBasis, dot, grad, spsolve = _lazy()

    @BilinearForm
    def axisym_stiffness(u, v, w):
        return w.x[0] * dot(grad(u), grad(v))

    @BilinearForm
    def axisym_mass(u, v, w):
        return w.x[0] * u * v

    return skfem, BilinearForm, LinearForm, FacetBasis, spsolve, axisym_stiffness, axisym_mass
