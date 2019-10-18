# (Pi u_3d, u_1d)_Gamma integrals

from weak_bcs.burman.generation import line_mesh, line, StraightLineMesh
from dolfin import *
from xii import *
from xii.assembler.average_matrix import average_matrix
import numpy as np


def test_Pi(n, f3, f1, Pi_f3):
    '''(Pi u_3d, u_1d)_Gamma.'''
    # True here is the reduced one
    omega = UnitCubeMesh(n, n, 2*n)

    A = np.array([0.5, 0.5, 0.0])
    B = np.array([0.5, 0.5, 1.0])             
    gamma = StraightLineMesh(A, B, 2*n)

    V3 = FunctionSpace(omega, 'CG', 1)
    V1 = FunctionSpace(gamma, 'CG', 1)

    u3 = TrialFunction(V3)
    v1 = TestFunction(V1)

    shape = SquareRim(P=lambda x: np.array([0.25, 0.25, x[2]]),
                      degree=10)

    dx_ = Measure('dx', domain=gamma)
    a = inner(Average(u3, gamma, shape=shape), v1)*dx_

    A = ii_assemble(a)

    # Now check action
    f3h = interpolate(f3, V3)
    x = A*f3h.vector()

    f1h = interpolate(f1, V1)
    Pi_f = Function(V1, x)
    result = f1h.vector().inner(x)

    true = assemble(inner(Pi_f3, f1)*dx_)
    
    error = abs(true - result)

    return gamma.hmin(), error, Pi_f

# --------------------------------------------------------------------

if __name__ == '__main__':
    from test_surf_avg import analyze_convergence
    from functools import partial
    import sympy as sp

    ns = (4, 8, 16, 32)
    
    # Exact, exact
    f3, f1 = Constant(2), Constant(3)
    get_error = partial(test_Pi, f3=f3, f1=f1, Pi_f3=f3)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert error < 1E-12

    # Exact, exact
    f3, f1 = Expression('x[2]', degree=3), Expression('x[2]', degree=3)
    get_error = partial(test_Pi, f3=f3, f1=f1, Pi_f3=f3)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert error < 1E-12

    # Ecact, exact
    f3, f1 = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])*x[2]', degree=3), Expression('x[2]', degree=3)
    get_error = partial(test_Pi, f3=f3, f1=f1, Pi_f3=Constant(0))
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert error < 1E-12

    # Ecact, exact
    f3, f1 = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])*x[2]', degree=3), Expression('sin(pi*x[2])', degree=3)
    get_error = partial(test_Pi, f3=f3, f1=f1, Pi_f3=Constant(0))
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert error < 1E-12

    # Exact, exact
    f3, f1 = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])*sin(pi*x[2])', degree=3), Expression('sin(pi*x[2])', degree=3)
    get_error = partial(test_Pi, f3=f3, f1=f1, Pi_f3=Constant(0))
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert error < 1E-12

    # Convergence
    f3 = Expression('sin(pi*x[0])*sin(pi*x[1])', degree=5)

    x = sp.Symbol('x')
    side = sp.sin(sp.pi/4)*sp.integrate(sp.sin(sp.pi*x), (x, 0.25, 0.75))
    true = 4*side/2  # 2 for the circumnference
    Pi_f3 = Constant(true)

    f1 = Expression('sin(pi*x[2])', degree=3)
    
    get_error = partial(test_Pi, f3=f3, f1=f1, Pi_f3=Pi_f3)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert rate > 1

    # Convergence
    f3 = Expression('sin(pi*x[2])*sin(pi*x[0])*sin(pi*x[1])', degree=5)

    x = sp.Symbol('x')
    side = sp.sin(sp.pi/4)*sp.integrate(sp.sin(sp.pi*x), (x, 0.25, 0.75))
    true = 4*side/2  # 2 for the circumnference
    Pi_f3 = Expression('A*sin(pi*x[2])', A=float(true), degree=3)

    f1 = Expression('sin(pi*x[2])', degree=3)
    
    get_error = partial(test_Pi, f3=f3, f1=f1, Pi_f3=Pi_f3)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert rate > 1
