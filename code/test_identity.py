# If I extend 1d to 2d (then 3d) and reduce back to 1d do I get the same

from weak_bcs.burman.generation import line_mesh, line, StraightLineMesh
from dolfin import *
from xii import *
from xii.assembler.extension_matrix import uniform_extension_matrix
from xii.assembler.average_matrix import average_matrix
import numpy as np

set_log_level(WARNING)


def test_id(n, f, align=True):
    '''Surface integral (for benchmark problem)'''
    omega = UnitCubeMesh(n, n, 2*n)

    A = np.array([0.5, 0.5, 0.0])
    B = np.array([0.5, 0.5, 1.0])

    lmbda = StraightLineMesh(A, B, 2*n if align else n)  # The curve

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
    # To 2d
    E = uniform_extension_matrix(V1, V2)
    x = E*f1.vector()
    f2 = Function(V2, x)
    # Extend to 3d by harmonic extension
    u, v = TrialFunction(V3), TestFunction(V3)
    a, L = inner(grad(u), grad(v))*dx, inner(Constant(0), v)*dx
    bcs = [DirichletBC(V3, f, 'on_boundary'),
           DirichletBC(V3, f, surfaces, 1)]

    A, b = assemble_system(a, L, bcs)
    f3 = Function(V3)
    
    solver = KrylovSolver('cg', 'hypre_amg')
    solver.parameters['relative_tolerance'] = 1E-13
    
    solver.solve(A, f3.vector(), b)

    # Come back
    shape = SquareRim(P=lambda x: np.array([0.25, 0.25, x[2]]),
                      degree=10)

    Pi = average_matrix(V3, V1, shape=shape)
    x = Pi*f3.vector()

    f10 = Function(V1, x)

    error = sqrt(abs(assemble(inner(f10 - f, f10 - f)*dx)))

    return gamma.hmin(), error, f10

# --------------------------------------------------------------------

if __name__ == '__main__':
    from test_surf_avg import analyze_convergence
    from functools import partial
    import sympy as sp

    ns = (4, 8, 16, 32)
    
    # Exact
    f = Constant(1)
    get_error = partial(test_id, f=f)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert error < 1E-12

    # Exact
    f = Expression('x[2]', degree=5)
    get_error = partial(test_id, f=f)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert error < 1E-12

    # Convergence
    f = Expression('sin(pi*x[2])', degree=5)
    get_error = partial(test_id, f=f)
    res, f, rate, error = analyze_convergence(ns, get_error)
    assert rate > 1

    # Convergence
    f = Expression('sin(2*pi*x[2])', degree=5)
    get_error = partial(test_id, f=f, align=False)
    res, f, rate, error = analyze_convergence(ns, get_error)
