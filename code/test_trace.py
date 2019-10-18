# Gamma as straight line {0.5}x{0.5}x(0, 1)
# Omega as [0, 1]^2
#
# Things for formulation with the multiplier on surface


from dolfin import *
from xii import *
from xii.assembler.trace_matrix import trace_mat_no_restrict
import numpy as np

set_log_level(WARNING)


def test_trace(n, f, true=None):
    '''Surface integral (for benchmark problem)'''
    omega = UnitCubeMesh(n, n, n)

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

    # What we want to extend
    f3 = interpolate(f, V3)

    T = PETScMatrix(trace_mat_no_restrict(V3, V2))
    x = T*f3.vector()
    f2 = Function(V2, x)

    if true is None:
        true = f
    error = sqrt(abs(assemble(inner(true - f2, true - f2)*dx)))

    return gamma.hmin(), error, f2

# --------------------------------------------------------------------

if __name__ == '__main__':
    from test_surf_avg import analyze_convergence
    from functools import partial
    import sympy as sp

    ns = (4, 8, 16, 32)
    
    # Exact
    f = Constant(1)
    get_error = partial(test_trace, f=f, true=f)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert error < 1E-12

    # Exact as we have P1
    f = Expression('x[0] - x[1] + 2*x[2]', degree=2)
    get_error = partial(test_trace, f=f, true=f)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert error < 1E-12

    # Convergence
    f = Expression('sin(2*pi*x[2])', degree=2)
    get_error = partial(test_trace, f=f, true=f)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert rate > 1

    # Convergence
    f = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])*sin(pi*x[2])', degree=2)
    get_error = partial(test_trace, f=f, true=f)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert rate > 1
