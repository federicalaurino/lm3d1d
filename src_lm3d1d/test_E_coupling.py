# (E u_1d, u_2d)_Gamma integrals

from weak_bcs.burman.generation import line_mesh, line, StraightLineMesh
from dolfin import *
from xii import *
import numpy as np


def test_E(n, f2, f1):
    '''(E u_1d, u_2d)_Gamma.'''
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

    A = np.array([0.5, 0.5, 0.0])
    B = np.array([0.5, 0.5, 1.0])             
    lmbda = StraightLineMesh(A, B, 2*n)  # The curve

    V3 = FunctionSpace(omega, 'CG', 1)
    V2 = FunctionSpace(gamma, 'CG', 1)
    V1 = FunctionSpace(lmbda, 'CG', 1)
    
    u1 = TrialFunction(V1)
    v2 = TestFunction(V2)

    dx_ = Measure('dx', domain=gamma)
    a = inner(Extension(u1, gamma, 'uniform'), v2)*dx_

    A = ii_assemble(a)

    # Now check action
    f1h = interpolate(f1, V1)
    x = A*f1h.vector()

    f2h = interpolate(f2, V2)
    result = f2h.vector().inner(x)

    true = assemble(inner(f2, f1)*dx_)

    error = abs(true - result)

    return gamma.hmin(), error, Function(V2, x)

# --------------------------------------------------------------------

if __name__ == '__main__':
    from test_surf_avg import analyze_convergence
    from functools import partial
    import sympy as sp

    ns = (4, 8, 16, 32)
    
    # Exact, exact
    f2, f1 = Constant(2), Constant(3)
    get_error = partial(test_E, f2=f2, f1=f1)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert error < 1E-12

    # Exact, exact
    f2, f1 = Expression('x[0] - x[2] + 2*x[1]', degree=3), Expression('x[2]', degree=1)
    get_error = partial(test_E, f2=f2, f1=f1)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert error < 1E-12

    # # Convergence
    f2, f1 = Expression('x[0] - x[2] + 2*x[1]', degree=3), Expression('sin(2*pi*x[2])', degree=3)
    get_error = partial(test_E, f2=f2, f1=f1)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert rate > 1

    # Convergence
    f2, f1 = Expression('sin(pi*(x[0] - x[2] + 2*x[1]))', degree=3), Expression('sin(2*pi*x[2])', degree=3)
    get_error = partial(test_E, f2=f2, f1=f1)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert rate > 1
