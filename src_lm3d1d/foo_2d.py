from line_cut import curve_cover
import numpy as np
from dolfin import *
from xii import *

n = 4

omega = UnitSquareMesh(n+1, n+1)

A, B = np.array([0.5, 0.0]), np.array([0.5, 1])
gamma = StraightLineMesh(A, B, 3*n)

cut_cells = curve_cover(gamma, omega, tol=1E-10)
omega_subdomains = MeshFunction('size_t', omega, omega.topology().dim(), 0)
omega_subdomains.array()[cut_cells] = 1

File('cut_2d.pvd') << omega_subdomains


lm_gamma = EmbeddedMesh(omega_subdomains, 1)

Q = FunctionSpace(lm_gamma, 'DG', 0)

p, q = TrialFunction(Q), TestFunction(Q)
Tp, Tq = Trace(p, gamma), Trace(q, gamma)

dx_ = Measure('dx', domain=gamma)
m = inner(Tp, Tq)*dx_

X, y, Z = map(ii_convert, ii_assemble(m).chain)

x = Function(Q).vector()
for i in range(Q.dim()):
    x.set_local(np.eye(Q.dim())[i])

    if (Z*x).norm('l2') < 1E-13:
        omega_subdomains[int(cut_cells[i])] = 2
        print '!'
    x *= 0.

File('cut_2d.pvd') << omega_subdomains
