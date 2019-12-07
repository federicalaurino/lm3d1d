from line_cut import curve_cover
import numpy as np
from dolfin import *
from xii import *

n = 4

omega = UnitCubeMesh(3, 3, n)

A, B = np.array([0.5, 0.5, 0]), np.array([0.5, 0.5, 1])
gamma = StraightLineMesh(A, B, 3*n+1)

cut_cells = curve_cover(gamma, omega, tol=1E-10)
omega_subdomains = MeshFunction('size_t', omega, omega.topology().dim(), 0)
omega_subdomains.array()[cut_cells] = 1

lm_gamma = EmbeddedMesh(omega_subdomains, 1)
File('omega_subdomains.pvd') << omega_subdomains

Q = FunctionSpace(lm_gamma, 'DG', 0)
print Q.dim()
p, q = TrialFunction(Q), TestFunction(Q)

circle = None#Circle(1E-10, 12)
Tp, Tq = Average(p, gamma, circle), Average(q, gamma, circle)

dx_ = Measure('dx', domain=gamma)
m = inner(Tp, Tq)*dx_

X, y, Z = map(ii_convert, ii_assemble(m).chain)

x = Function(Q).vector()
for i in range(Q.dim()):
    x.set_local(np.eye(Q.dim())[i])

    print (Z*x).norm('l2')
    #< 1E-13:
    #    omega_subdomains[int(cut_cells[i])] = 2
    #    print '!'
    #x *= 0.
print Z.array()

print np.sort(np.linalg.eigvalsh(y.array()))
print np.sort(np.linalg.eigvalsh(ii_convert(ii_assemble(m)).array()))
