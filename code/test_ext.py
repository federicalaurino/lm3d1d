# Gamma as straight line {0.5}x{0.5}x(0, 1)
# Omega as [0, 1]^2
#
# I consider q from the benchmark q = sin(pi*z/H)

from weak_bcs.burman.generation import line_mesh, line, StraightLineMesh
from dolfin import *
from xii import *
from xii.assembler.extension_matrix import uniform_extension_matrix
import numpy as np

set_log_level(WARNING)


def test_ext(n, f, true=None):
    '''Surface integral (for benchmark problem)'''
    omega = UnitCubeMesh(n, n, 2*n)

    A = np.array([0.5, 0.5, 0.0])
    B = np.array([0.5, 0.5, 1.0])             
    lmbda = StraightLineMesh(A, B, 2*n)  # The curve

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
    V1 = FunctionSpace(lmbda, 'CG', 1)

    # What we want to extend
    f1 = interpolate(f, V1)

    E = uniform_extension_matrix(V1, V2)
    x = E*f1.vector()
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
    get_error = partial(test_ext, f=f, true=f)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert error < 1E-12

    # Exact
    f = Expression('x[2]', degree=5)
    get_error = partial(test_ext, f=f, true=f)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert error < 1E-12

    # Convergence
    f = Expression('sin(pi*x[2])', degree=5)
    get_error = partial(test_ext, f=f, true=f)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert rate > 1
