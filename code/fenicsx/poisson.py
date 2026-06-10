from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import VTXWriter

import ufl
import numpy as np

# ============================================================
# Mesh
# ============================================================

nx = ny = 64

domain = mesh.create_unit_square(
    MPI.COMM_WORLD,
    nx,
    ny
)

# ============================================================
# Finite element space
# ============================================================

V = fem.functionspace(domain, ("Lagrange", 1))

# ============================================================
# Boundary conditions
# ============================================================

fdim = domain.topology.dim - 1

facets = mesh.locate_entities_boundary(
    domain,
    fdim,
    lambda x: np.full(x.shape[1], True)
)

dofs = fem.locate_dofs_topological(
    V,
    fdim,
    facets
)

u_D = fem.Constant(domain, PETSc.ScalarType(0.0))

bc = fem.dirichletbc(
    u_D,
    dofs,
    V
)

# ============================================================
# Exact solution
# ============================================================

x = ufl.SpatialCoordinate(domain)

u_exact_expr = (
    ufl.sin(ufl.pi * x[0])
    * ufl.sin(ufl.pi * x[1])
)

# ============================================================
# Right-hand side
# ============================================================

f = (
    2.0 * np.pi**2
    * ufl.sin(np.pi * x[0])
    * ufl.sin(np.pi * x[1])
)

# ============================================================
# Variational formulation
# ============================================================

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

L = f * v * ufl.dx

# ============================================================
# Solve
# ============================================================

problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options_prefix="poisson_",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu"
    }
)

uh = problem.solve()

uh.name = "solution"

# ============================================================
# Exact solution interpolation
# ============================================================

u_exact = fem.Function(V)
u_exact.interpolate(
    fem.Expression(
        u_exact_expr,
        V.element.interpolation_points
    )
)
# ============================================================
# L2 error
# ============================================================

error_form = fem.form(
    ufl.inner(
        uh - u_exact,
        uh - u_exact
    ) * ufl.dx
)

error_L2 = np.sqrt(
    domain.comm.allreduce(
        fem.assemble_scalar(error_form),
        op=MPI.SUM
    )
)

if domain.comm.rank == 0:
    print()
    print("===================================")
    print("Poisson problem")
    print("===================================")
    print(f"Mesh : {nx} x {ny}")
    print(f"L2 error = {error_L2:.6e}")
    print()

# ============================================================
# Output
# ============================================================

with VTXWriter(
    domain.comm,
    "results/vtk/poisson.bp",
    [uh]
) as vtx:
    vtx.write(0.0)

