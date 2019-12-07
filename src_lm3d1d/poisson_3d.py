from dolfin import *
from xii import *
from weak_bcs.utils import block_form, MMSData, H1_norm
from block import block_mat, block_vec
from block.algebraic.petsc import AMG
import numpy as np


def setup_mms(params):
    '''Manufactured solution'''
    
    u3d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=4)
    f3d = Expression('(8*pi*pi)*sin(2*pi*x[0])*sin(2*pi*x[1])', degree=4)    

    return MMSData(solution=[u3d], rhs=[f3d], subdomains=['on_boundary'],normals=None)


def setup_problem(n, mms, params):
    '''Poisson solver'''
    mesh = UnitCubeMesh(n, n, n)
    
    V = FunctionSpace(mesh, 'CG', 1)

    # Just one
    boundaries = MeshFunction('size_t', mesh, 2, 0)
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

        return (H1_norm(u_exact, uh), )
    # Pretty print
    error_types = ('|u|_1', )
    
    return get_error, error_types

# How is the problem parametrized
PARAMETERS = ('mu', )

# --------------------------------------------------------------------

if __name__ == '__main__':
    from petsc4py import PETSc
    from IPython import embed
    
    for n in [4*2**i for i in range(6)]:
        mms = setup_mms(None)

        A, b, V = setup_problem(n, mms, params=None)
        A = as_backend_type(A[0][0]).mat()
        b = as_backend_type(b[0]).vec()
        V = V[0]

        ksp = PETSc.KSP().create()
        ksp.setType('cg')
        ksp.setOperators(A)
        ksp.setConvergenceHistory()
        ksp.setNormType(PETSc.KSP.NormType.NORM_PRECONDITIONED)

        pc = ksp.getPC()
        pc.setType('hypre')
        pc.setOperators(A, A)
        pc.setUp()
        
        # User configs
        opts = PETSc.Options()
        opts.setValue('ksp_rtol', 1E-8)
        opts.setValue('ksp_atol', 1E-20)
        opts.setValue('ksp_monitor_true_residual', None)
        ksp.setFromOptions()
        
        x = A.createVecLeft()
        dt = Timer('ksp')
        ksp.solve(b, x)
        dt = dt.stop()

        niters = ksp.getIterationNumber()

        print V.dim(), '%.2f' % dt, niters
