# (T u_3d, u_2d)_Gamma integrals

from dolfin import *
from xii import *
import numpy as np


def test_T(n, f3, f2):
    '''(T u_3d, u_2d)_Gamma.'''
    # True here is the reduced one
    omega = UnitCubeMesh(n, n, 2*n)

    # Coupling surface
    walls = ['near((x[0]-0.25)*(x[0]-0.75), 0) && ((0.25-tol) < x[1]) && ((0.75+tol) > x[1])',
             'near((x[1]-0.25)*(x[1]-0.75), 0) && ((0.25-tol) < x[0]) && ((0.75+tol) > x[0])']
    walls = ['( %s )' % w for w in walls]
    
    chi = CompiledSubDomain(' || '.join(walls), tol=1E-10)
    surfaces = MeshFunction('size_t', omega, 2, 0)
    chi.mark(surfaces, 1)

    gamma = EmbeddedMesh(surfaces, 1)

    V3 = FunctionSpace(omega, 'CG', 1)
    V2 = FunctionSpace(gamma, 'CG', 1)

    u3 = TrialFunction(V3)
    v2 = TestFunction(V2)

    dx_ = Measure('dx', domain=gamma)
    a = inner(Trace(u3, gamma), v2)*dx_

    A = ii_assemble(a)

    # Now check action
    f3h = interpolate(f3, V3)
    x = A*f3h.vector()

    f2h = interpolate(f2, V2)
    result = f2h.vector().inner(x)

    true = assemble(inner(f2, f3)*dx_)
    
    error = abs(true - result)

    return gamma.hmin(), error, Function(V2, x)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from test_surf_avg import analyze_convergence
    from functools import partial
    import sympy as sp

    ns = (4, 8, 16, 32)
    
    # Exact, exact
    f3, f2 = Constant(2), Constant(3)
    get_error = partial(test_T, f3=f3, f2=f2)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert error < 1E-12

    # Exact, exact
    f3 = Expression('x[0] - x[2] + 2*x[1]', degree=3)
    get_error = partial(test_T, f3=f3, f2=f3)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert error < 1E-12

    # Cvrg
    f3 = Expression('sin(pi*(x[0] - x[2] + 2*x[1]))', degree=3)
    get_error = partial(test_T, f3=f3, f2=f3)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert rate > 1

