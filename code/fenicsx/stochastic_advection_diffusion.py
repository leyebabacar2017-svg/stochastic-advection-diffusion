from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import VTXWriter

import ufl
import numpy as np

# ============================================================
# Time parameters
# ============================================================

T = 1.0
dt = 0.0125

num_steps = int(T / dt)

t = 0.0

# ============================================================
# Mesh
# ============================================================

nx = ny = 128

domain = mesh.create_unit_square(
    MPI.COMM_WORLD,
    nx,
    ny
)

# ============================================================
# Finite element space
# ============================================================

V = fem.functionspace(domain, ("Lagrange", 1))

b = ufl.as_vector([1.0, 0.0])

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

time = fem.Constant(domain, PETSc.ScalarType(0.0))

u_n = fem.Function(V)

u_n.interpolate(
    fem.Expression(
        ufl.sin(ufl.pi * x[0])
        * ufl.sin(ufl.pi * x[1]),
        V.element.interpolation_points
    )
)

# ============================================================
# Right-hand side
# ============================================================

kappa = 0.01
sigma = 0.1

f = (
    -ufl.exp(-time)
    * ufl.sin(np.pi * x[0])
    * ufl.sin(np.pi * x[1])

    +

    np.pi
    * ufl.exp(-time)
    * ufl.cos(np.pi * x[0])
    * ufl.sin(np.pi * x[1])

    +

    2.0
    * kappa
    * np.pi**2
    * ufl.exp(-time)
    * ufl.sin(np.pi * x[0])
    * ufl.sin(np.pi * x[1])
)

# ============================================================
# Variational formulation
# ============================================================

noise = fem.Constant(
    domain,
    PETSc.ScalarType(0.0)
)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = (
    (1.0 / dt) * u * v
    + kappa * ufl.inner(ufl.grad(u), ufl.grad(v))
    + ufl.inner(b, ufl.grad(u)) * v
) * ufl.dx

L = (
    (1.0 / dt) * u_n * v
    + f * v
    + noise * v
) * ufl.dx

# ============================================================
# Solve
# ============================================================

uh = fem.Function(V)

for n in range(num_steps):

    t += dt

    time.value = PETSc.ScalarType(t)
    
    dW = np.sqrt(dt) * np.random.normal()
    
    noise.value = PETSc.ScalarType(
          sigma * dW / dt
    )  

    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="advdiff_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu"
        }
    )

    uh = problem.solve()

    u_n.x.array[:] = uh.x.array

    if domain.comm.rank == 0 and n % 10 == 0:
       print(
             f"Step {n:4d} "
             f"t={t:.3f} "
             f"dW={dW:.4e}"
       )

uh.name = "solution"

# ============================================================
# Exact solution interpolation
# ============================================================

time.value = PETSc.ScalarType(T)

u_exact_expr = (
    ufl.exp(-time)
    * ufl.sin(ufl.pi * x[0])
    * ufl.sin(ufl.pi * x[1])
)

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
    print("Advection-Diffusion problem")
    print("===================================")
    print(f"Mesh : {nx} x {ny}")
    print(f"L2 error = {error_L2:.6e}")
    print()

# ============================================================
# Output
# ============================================================

with VTXWriter(
    domain.comm,
    "results/vtk/advection_diffusion_transient.bp",
    [uh]
) as vtx:
    vtx.write(0.0)

