# Gamma as straight line {0.5}x{0.5}x(0, 1)
# Omega as [0, 1]^2
#
# I consider q from the benchmark q = sin(pi*z/H)

from weak_bcs.burman.generation import line_mesh, line, StraightLineMesh
from dolfin import *
from xii import *
from xii.assembler.average_matrix import average_matrix
import numpy as np
import warnings


def analyze_convergence(ns, get_error):
    '''get_error: n -> h, e'''
    her = []  # h error rate
    for n in ns:
        h, e, f = get_error(n)

        if her:
            h0, e0, _ = her[-1]
            try:
                rate = np.log(e/e0)/np.log(h/h0)
            except ZeroDivisionError:
                rate = np.nan
        else:
            rate = np.nan

        print 'n(%g) -> %g %g [%g]' % (n, h, e, rate)
        her.append([h, e, rate])
    # Final fit
    h, e, _ = np.array(her).T

    with warnings.catch_warnings('error'):
        rate, _ = np.polyfit(np.log(h), np.log(e), deg=1)
        print 'Final rate %g \n' % rate
    
    return her, f, rate, np.mean(e)


def test_surf_approx(n, f, true=None):
    '''Surface integral (for benchmark problem)'''
    omega = UnitCubeMesh(n, n, 2*n)

    A = np.array([0.5, 0.5, 0.0])
    B = np.array([0.5, 0.5, 1.0])             
    gamma = StraightLineMesh(A, B, 2*n)

    V3 = FunctionSpace(omega, 'CG', 1)
    V1 = FunctionSpace(gamma, 'CG', 1)

    f3 = interpolate(f, V3)

    shape = SquareRim(P=lambda x: np.array([0.25, 0.25, x[2]]),
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
    from functools import partial
    import sympy as sp

    ns = (4, 8, 16, 32)
    
    # Convergence to 'identity'
    f = Expression('sin(pi*x[2])', degree=5)
    get_error = partial(test_surf_approx, f=f, true=f)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert rate > 1
    
    # # Should be exact right away
    f = Expression('2*x[2]', degree=5)
    get_error = partial(test_surf_approx, f=f, true=f)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert error < 1E-12

    # The exact solution in 3d has Avg(u) = 0
    f = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=5)
    get_error = partial(test_surf_approx, f=f, true=Constant(0))
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert error < 1E-12
    
    # Something that is a bit harder
    f = Expression('sin(pi*x[0])*sin(pi*x[1])', degree=5)

    x = sp.Symbol('x')
    side = sp.sin(sp.pi/4)*sp.integrate(sp.sin(sp.pi*x), (x, 0.25, 0.75))
    true = Constant(4*side/2)  # 2 for the circumnference
    get_error = partial(test_surf_approx, f=f, true=true)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert rate > 1
    
    # Something that is a bit harder
    f = Expression('x[2]*sin(pi*x[0])*sin(pi*x[1])', degree=5)
    true = Expression('A*x[2]', degree=5, A=true(0))
    get_error = partial(test_surf_approx, f=f, true=true)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert rate > 1
