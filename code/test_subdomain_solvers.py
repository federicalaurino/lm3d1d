from weak_bcs.burman.generation import StraightLineMesh
from dolfin import *
import numpy as np


def poisson_3d(n, u_true, f):
    '''3d part of benchmark'''
    mesh = UnitCubeMesh(n, n, n)

    V = FunctionSpace(mesh, 'CG', 1)
    bc = DirichletBC(V, u_true, 'on_boundary')
    
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    A, b = assemble_system(a, L, bc)

    solver = KrylovSolver('cg', 'amg')
    solver.parameters['relative_tolerance'] = 1E-13

    uh = Function(V)
    solver.solve(A, uh.vector(), b)
    
    return mesh.hmin(), errornorm(u_true, uh, 'H1'), uh


def poisson_1d(n, u_true, f):
    '''1d part of benchmark'''
    A, B = np.array([0.5, 0.5, 0]), np.array([0.5, 0.5, 1])
    mesh = StraightLineMesh(A, B, n)

    V = FunctionSpace(mesh, 'CG', 1)
    bc = DirichletBC(V, u_true, 'on_boundary')
    
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    A, b = assemble_system(a, L, bc)

    solver = KrylovSolver('cg', 'amg')
    solver.parameters['relative_tolerance'] = 1E-13

    uh = Function(V)
    solver.solve(A, uh.vector(), b)

    return mesh.hmin(), errornorm(u_true, uh, 'H1'), uh

# --------------------------------------------------------------------

if __name__ == '__main__':
    from test_surf_avg import analyze_convergence
    from functools import partial


    ns = (4, 8, 16, 32)
    # 3d
    u3d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=4)
    f3d = Expression('(8*pi*pi)*sin(2*pi*x[0])*sin(2*pi*x[1])', degree=4)    
    get_error = partial(poisson_3d, f=f3d, u_true=u3d)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert rate > 0.9
        
    ns = (128, 256, 512, 1024)
    # 1d
    f1d = Expression('pi*pi*sin(pi*x[2])', degree=4)
    u1d = Expression('sin(pi*x[2])', degree=4)    
    get_error = partial(poisson_1d, f=f1d, u_true=u1d)
    rate, error = analyze_convergence(ns, get_error)[2:]
    assert rate > 0.9
