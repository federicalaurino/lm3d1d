# Gamma as straight line {0.5}x{0.5}x(0, 1)
# Omega as [0, 1]^2
#
# I consider q from the benchmark q = sin(pi*z/H)

from weak_bcs.burman.generation import line_mesh, line, StraightLineMesh
from dolfin import *
from xii import *
from xii.assembler.average_matrix import average_matrix
import numpy as np


def test_vol_approx(n, f, true=None):
    '''Volume integral (for benchmark problem)'''
    omega = UnitCubeMesh(n, n, 2*n)

    A = np.array([0.5, 0.5, 0.0])
    B = np.array([0.5, 0.5, 1.0])             
    gamma = StraightLineMesh(A, B, 2*n)

    V3 = FunctionSpace(omega, 'CG', 1)
    V1 = FunctionSpace(gamma, 'CG', 1)

    f3 = interpolate(f, V3)

    shape = Square(P=lambda x: np.array([0.25, 0.25, x[2]]),
                   degree=10)

    Pi = average_matrix(V3, V1, shape=shape)
    x = Pi*f3.vector()
    f1 = Function(V1, x)

    if true is None:
        true = f
    error = sqrt(abs(assemble(inner(true - f1, true - f1)*dx)))

    return gamma.hmin(), error, f1

# --------------------------------------------------------------------

if __name__ == '__main__':
    from test_surf_avg import analyze_convergence
    from functools import partial
    import sympy as sp

    ns = (4, 8, 16, 32)
    
    # Convergence to 'identity'
    f = Expression('sin(pi*x[2])', degree=5)
    get_error = partial(test_vol_approx, f=f, true=f)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert rate > 1
    
    # Should be exact right away
    f = Expression('2*x[2]', degree=5)
    get_error = partial(test_vol_approx, f=f, true=f)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert error < 1E-12

    # The exact solution in 3d has Avg(u) != 0
    f = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=5)

    x = sp.Symbol('x')
    side = sp.integrate(sp.sin(2*sp.pi*x), (x, 0.25, 0.75))
    truth = side*side/(0.25)

    get_error = partial(test_vol_approx, f=f, true=Constant(truth))
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert rate > 1

    # Close to
    f = Expression('sin(pi*x[0])', degree=5)

    x = sp.Symbol('x')
    side = sp.integrate(sp.sin(sp.pi*x), (x, 0.25, 0.75))
    truth = side*0.5/(0.25)

    true = Expression('A', A=float(truth), degree=5)
    get_error = partial(test_vol_approx, f=f, true=true)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert rate > 1

    # Close to
    f = Expression('x[2]*sin(pi*x[0])*sin(pi*x[1])', degree=5)

    x = sp.Symbol('x')
    side = sp.integrate(sp.sin(sp.pi*x), (x, 0.25, 0.75))
    truth = side*side/(0.25)

    true = Expression('A*x[2]', A=float(truth), degree=5)
    get_error = partial(test_vol_approx, f=f, true=true)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert rate > 1
