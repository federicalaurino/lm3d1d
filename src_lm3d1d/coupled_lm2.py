# Formulation with the multiplier on the surface
from dolfin import *
from xii import *
from weak_bcs.utils import block_form, MMSData, H1_norm, Hs0_norm, matrix_fromHs, L2_norm
import src_lm3d1d.coupled_lm1 as lm1
from hsmg.hseig import Hs0Norm
from block.algebraic.petsc import LU, AMG
from block import block_mat, block_vec
import numpy as np


def setup_problem(n, mms, params):
    '''Coupled 3d-1d with 2d Lagrange multiplier'''
    # NOTE: inborg distinguishes between parallel and not parallel hmin
    omega = UnitCubeMesh(n, n, n)

    size = 0.25
    # Coupling surface
    walls = ['near((x[0]-L)*(x[0]-H), 0) && ((L-tol) < x[1]) && ((H+tol) > x[1])',
             'near((x[1]-L)*(x[1]-H), 0) && ((L-tol) < x[0]) && ((H+tol) > x[0])']
    walls = ['( %s )' % w for w in walls]
    
    chi = CompiledSubDomain(' || '.join(walls), tol=1E-10, L=0.5-size, H=0.5+size)
    surfaces = MeshFunction('size_t', omega, 2, 0)
    chi.mark(surfaces, 1)

    gamma = EmbeddedMesh(surfaces, 1)

    # Centerline
    A, B = np.array([0.5, 0.5, 0]), np.array([0.5, 0.5, 1])
    lmbda = StraightLineMesh(A, B, n)  

    # The poisson problem
    V3 = FunctionSpace(omega, 'CG', 1)
    V1 = FunctionSpace(lmbda, 'CG', 1)
    Q = FunctionSpace(gamma, 'CG', 1)
    
    W = [V3, V1, Q]

    u3, u1, p = map(TrialFunction, W)
    v3, v1, q = map(TestFunction, W)

    Tu3, Tv3 = (Trace(f, gamma) for f in (u3, v3))
    Eu1, Ev1 = (Extension(f, gamma, type='uniform') for f in (u1, v1))
    
    # The size of averaging square
    alen = 2*size
    # Cell integral of Qspace
    dx_ = Measure('dx', domain=gamma)
    avg_len = Constant(4*alen)
    avg_area = Constant(alen**2)
    
    a = block_form(W, 2)
    # Subdomains
    a[0][0] = inner(grad(u3), grad(v3))*dx
    a[1][1] = avg_area*inner(grad(u1), grad(v1))*dx
    # Coupling
    a[0][2] = inner(Tv3, p)*dx_
    a[1][2] = -inner(Ev1, p)*dx_
    a[2][0] = inner(Tu3, q)*dx_
    a[2][1] = -inner(Eu1, q)*dx_

    # Right hand sides and dirichlet data for the unknowns
    (f3, f1, fI), (g3, g1, gI) = mms.rhs
    # Linear part
    L = block_form(W, 1)
    L[0] = inner(f3, v3)*dx

    dl = Measure('dx', domain=lmbda)
    # Account for reduction from volume to line
    avg_shape = Square(lambda x: np.array([size, size, x[-1]]), degree=10)
    # NOTE: Avg only works with functions so need to interpolate it
    V3h = FunctionSpace(omega, 'CG', 2)
    f1h = interpolate(f1, V3h)
    L[1] = avg_area*inner(Average(f1h, lmbda, avg_shape), v1)*dl

    L[2] = inner(fI, q)*dx_
    
    # Homog everywhere
    V3_bcs = [DirichletBC(V3, g3, 'on_boundary')]
    V1_bcs = [DirichletBC(V1, g1, 'on_boundary')]
    Q_bcs = [DirichletBC(Q, gI, 'on_boundary')]
    # Group
    bcs = [V3_bcs, V1_bcs, Q_bcs]
    # No bcs
    A, b = map(ii_assemble, (a, L))
    # With bcs
    A, b = apply_bc(A, b, bcs)

    return A, b, W


def setup_mms(params):
    '''Manufactured solution from notes'''
    
    u3d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=4)
    f3d = Expression('(8*pi*pi)*sin(2*pi*x[0])*sin(2*pi*x[1])', degree=4)

    u1d = Expression('sin(pi*x[2])', degree=4)
    f1d = Expression('pi*pi*sin(pi*x[2])', degree=4)

    p = Expression('0', degree=4)  # The multiplier
    fLM = Expression('sin(2*pi*x[0])*sin(2*pi*x[1]) - sin(pi*x[2])', degree=4)

    return MMSData(solution=[u3d, u1d, p],
                   rhs=[(f3d, f1d, fLM), (u3d, u1d, p)],
                   subdomains=[], normals=[])

def setup_error_monitor(mms_data, params):
    '''We look at H1, H1, Hs norm'''
    # Error of the passed wh
    
    def get_error(wh, mms=mms_data):
        # We look at the H1 error for velocity
        u3, u1, p = mms.solution
        u3h, u1h, ph = wh

        # X = ph.function_space()
        # bcs = DirichletBC(X, Constant(0), 'on_boundary').get_boundary_values().keys()
        # print np.linalg.norm(ph.vector().get_local()[bcs], np.inf)

        # Passing the HsNorm from preconditioner so that we don't compute
        # eigenvalues twice
        Hs0Norm = mms.normals.pop()
        # e*H*e ie (e, e)_H
        e = interpolate(p, ph.function_space()).vector()
        e.axpy(-1, ph.vector())

        e = np.sqrt(e.inner(Hs0Norm*e))
        # On big meshes constructing the error for interpolation space
        # is expensive so wi reduce order then
        degree_rise = 1 if u3h.function_space().dim() < 2E6 else 0
        # NOTE: the error is interpolated to P2 disc, or P1 disc
        return (H1_norm(u3, u3h, degree_rise=degree_rise),
                H1_norm(u1, u1h),  
                e,
                L2_norm(p, ph))
    # Pretty print
    error_types = ('|u3|_1', '|u1|_1', '|p|_{-1/2}', '|p|_0')
    
    return get_error, error_types


cannonical_inner_product = lm1.cannonical_inner_product
cannonical_riesz_map = lm1.cannonical_riesz_map

withL2_inner_product = lm1.withL2_inner_product
withL2_riesz_map = lm1.withL2_riesz_map

just_identity = lm1.just_identity

# --------------------------------------------------------------------

# The idea now that we register the inner product so that from outside
# of the module they are accessible without referring to them by name
W_INNER_PRODUCTS = {0: cannonical_inner_product,
                    1: withL2_inner_product,
                    2: just_identity}

# And we do the same for preconditioners / riesz maps
W_RIESZ_MAPS = {0: cannonical_riesz_map,
                1: withL2_riesz_map,
                2: just_identity}

# How is the problem parametrized
PARAMETERS = ('size', )
