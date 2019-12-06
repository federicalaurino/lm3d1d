# Formulation with the multiplier on the curve
from dolfin import *
from xii import *
from weak_bcs.utils import (block_form, MMSData, H1_norm, L2_norm, matrix_fromHs,
                            mat_add)

import src_lm3d1d.coupled_lm1 as lm1
from line_cut import curve_cover
from hsmg.hseig import Hs0Norm, HsNorm
from block.algebraic.petsc import LU, AMG
from block import block_mat, block_vec
import numpy as np


def setup_mms(params):
    '''Manufactured solution from notes'''
    
    u3d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=4)
    f3d = Expression('(8*pi*pi)*sin(2*pi*x[0])*sin(2*pi*x[1])', degree=4)

    f1d = Expression('pi*pi*sin(pi*x[2])', degree=4)
    u1d = Expression('sin(pi*x[2])', degree=4)

    p = Expression('0', degree=4)  # The multiplier

    return MMSData(solution=[u3d, u1d, p],
                   rhs=[(f3d, f1d, u1d), (u3d, u1d, p)],
                   subdomains=[], normals=[])


def setup_problem(n, mms, params):
    '''Coupled 3d-1d with 1d Lagrange multiplier'''
    # NOTE: inborg distinguishes between parallel and not parallel hmin
    omega = UnitCubeMesh(n, n, n)

    A, B = np.array([0.5, 0.5, 0]), np.array([0.5, 0.5, 1])
    gamma = StraightLineMesh(A, B, 3*n)

    cut_cells = curve_cover(gamma, omega, tol=1E-10)
    omega_subdomains = MeshFunction('size_t', omega, omega.topology().dim(), 0)
    omega_subdomains.array()[cut_cells] = 1

    lm_gamma = EmbeddedMesh(omega_subdomains, 1)
    File('x.pvd') << omega
    File('y.pvd') << lm_gamma
    
    # The poisson problem
    V3 = FunctionSpace(omega, 'CG', 1)
    V1 = FunctionSpace(gamma, 'CG', 1)
    Q = FunctionSpace(lm_gamma, 'DG', 0)
    
    W = [V3, V1, Q]

    u3, u1, p = map(TrialFunction, W)
    v3, v1, q = map(TestFunction, W)

    # The size of averaging square is 0.5
    alen = 0.5
    avg_shape = SquareRim(lambda x: np.array([0.25, 0.25, x[-1]]), degree=10)

    # Setup bounding curve
    Pi_u3, Pi_v3 = (Average(f, gamma, avg_shape) for f in (u3, v3))
    Tp, Tq = (Average(f, gamma, None) for f in (p, q))

    # Cell integral of Qspace
    dx_ = Measure('dx', domain=gamma)
    avg_len = Constant(4*alen)
    avg_area = Constant(alen**2)
    
    a = block_form(W, 2)
    # Subdomains
    a[0][0] = inner(grad(u3), grad(v3))*dx
    a[1][1] = avg_area*inner(grad(u1), grad(v1))*dx
    # Coupling
    a[0][2] = avg_len*inner(Pi_v3, Tp)*dx_
    a[1][2] = -avg_len*inner(v1, Tp)*dx_
    a[2][0] = avg_len*inner(Pi_u3, Tq)*dx_
    a[2][1] = -avg_len*inner(u1, Tq)*dx_

    hK = CellDiameter(lm_gamma)
    # Stabiliation
    # FIXME: should be only on the boundary?
    a[2][2] = -avg(hK)*inner(jump(p), jump(q))*dS# -hK*inner(p, q)*ds

    # Right hand sides and dirichlet data for the unknowns
    (f3, f1, fI), (g3, g1, gI) = mms.rhs
    # Linear part
    L = block_form(W, 1)
    L[0] = inner(f3, v3)*dx

    # Account for reduction from volume to line
    avg_shape = Square(lambda x: np.array([0.25, 0.25, x[-1]]), degree=10)
    # NOTE: Avg only works with functions so need to interpolate it
    V3h = FunctionSpace(omega, 'CG', 2)
    f1h = interpolate(f1, V3h)
    L[1] = avg_area*inner(Average(f1h, gamma, avg_shape), v1)*dx_

    L[2] = avg_len*inner(-fI, Tq)*dx_
    
    # Homog everywhere
    V3_bcs = [DirichletBC(V3, g3, 'on_boundary')]
    V1_bcs = [DirichletBC(V1, g1, 'on_boundary')]
    Q_bcs = []
    # Group
    bcs = [V3_bcs, V1_bcs, Q_bcs]
    # No bcs
    A, b = map(ii_assemble, (a, L))
    # With bcs
    A, b = apply_bc(A, b, bcs)

    # print np.sort(np.abs(np.linalg.eigvalsh(ii_convert(A).array())))

    return A, b, W


def setup_error_monitor(mms_data, params):
    '''We look at H1, H1, Hs norm'''
    # Error of the passed wh
    
    def get_error(wh, mms=mms_data):
        # We look at the H1 error for velocity
        u3, u1, p = mms.solution
        u3h, u1h, ph = wh

        degree_rise = 1 if u3h.function_space().dim() < 2E6 else 0
        # NOTE: the error is interpolated to P2 disc, or P1 disc
        return (H1_norm(u3, u3h, degree_rise=degree_rise),
                H1_norm(u1, u1h),  
                L2_norm(p, ph))
    # Pretty print
    error_types = ('|u3|_1', '|u1|_1', '|p|_0')
    
    return get_error, error_types


def cannonical_inner_product(W, mms, params, AA, Z=None):
    '''Inner product of alpha*H1 x beta*H1 x H-0.5'''
    V3, V1, Q = W
    # Extact Vi norms from system
    V3_inner = AA[0][0]
    V1_inner = AA[1][1]

    #foo = MeshFunction('size_t', Q.mesh(), 2, 0)
    #CompiledSubDomain('near(x[2]*(1-x[2]), 0)').mark(foo, 1)
    #assert np.any(foo.array())
    #bcs = foo
    
    p, q = TrialFunction(Q), TestFunction(Q)
    Q_inner = mat_add(ii_convert(-1*AA[2][2]),
                      assemble(inner(p, q)*dx))
                      
    #matrix_fromHs(HsNorm(Q, s=-0.5, bcs=bcs)))

    B = block_diag_mat([V3_inner, V1_inner, Q_inner])

    return B


def cannonical_riesz_map(W, mms, params, AA, Z):
    '''Riesz map wrt. inner product of alpha*H1 x beta*H1 x H-0.5'''
    B = cannonical_inner_product(W, mms, params, AA, Z)
    B = block_diag_mat([AMG(B[0][0]), AMG(B[1][1]), LU(B[2][2])])

    return B


def just_identity(W, mms, params, AA, Z):
    '''1'''
    return identity_matrix(W)


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
PARAMETERS = ()
