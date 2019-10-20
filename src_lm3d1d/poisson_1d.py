from dolfin import *
from xii import *
from weak_bcs.utils import block_form, MMSData, H1_norm
from block import block_mat, block_vec
import numpy as np


def setup_mms(params):
    '''Manufactured solution'''
    
    u3d = Expression('sin(2*pi*x[0])', degree=4)
    f3d = Expression('(4*pi*pi)*sin(2*pi*x[0])', degree=4)    

    return MMSData(solution=[u3d], rhs=[f3d], subdomains=['on_boundary'],normals=None)


def setup_problem(n, mms, params):
    '''Poisson solver'''
    mesh = UnitIntervalMesh(n)
    
    V = FunctionSpace(mesh, 'CG', 1)

    # Just one
    boundaries = MeshFunction('size_t', mesh, 0, 0)
    for tag, subd in enumerate(mms.subdomains, 1):
        CompiledSubDomain(subd, tol=1E-10).mark(boundaries, tag)

    u_true,  = mms.solution
    bc = DirichletBC(V, u_true, boundaries, 1)

    f, = mms.rhs
    
    u, v = TrialFunction(V), TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    A, b = assemble_system(a, L, bc)

    return block_mat([[A]]), block_vec([b]), [V]


def setup_error_monitor(mms_data, params):
    '''We look at H1 norm'''
    # Error of the passed wh
    
    def get_error(wh, u_exact=mms_data.solution[0]):
        # We look at the H1 error for velocity
        uh, = wh

        V = uh.function_space()
        V = FunctionSpace(V.mesh(), 'CG', V.ufl_element().degree()+2)
        u, v = TrialFunction(V), TestFunction(V)
        a, L = inner(grad(u), grad(v))*dx, inner(Constant(0), v)*dx
        A, _ = assemble_system(a, L, bcs=DirichletBC(V, Constant(0), 'on_boundary'))

        e = interpolate(u_exact, V).vector()
        e.axpy(-1, interpolate(uh, V).vector())

        error = sqrt(e.inner(A*e))
        
        return (error, )
    # Pretty print
    error_types = ('|u|_1', )
    
    return get_error, error_types

# How is the problem parametrized
PARAMETERS = ('mu', )
