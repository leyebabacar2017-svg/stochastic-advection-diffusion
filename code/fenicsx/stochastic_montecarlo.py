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

Nmc = 10

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

ndofs = V.dofmap.index_map.size_local

mean_solution = np.zeros(ndofs)

energy = []

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

for mc in range(Nmc):

    if domain.comm.rank == 0:
        print(
            f"\nMonte Carlo sample "
            f"{mc+1}/{Nmc}"
        )

    # ==========================================
    # Reset initial condition
    # ==========================================

    t = 0.0

    time.value = PETSc.ScalarType(0.0)

    u_n.interpolate(
        fem.Expression(
            ufl.sin(ufl.pi * x[0])
            * ufl.sin(ufl.pi * x[1]),
            V.element.interpolation_points
        )
    )

    # ==========================================
    # Time loop
    # ==========================================

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

        if (
            domain.comm.rank == 0
            and n % 10 == 0
        ):
            print(
                f"Step {n:4d} "
                f"t={t:.3f} "
                f"dW={dW:.4e}"
            )

    # ==========================================
    # Statistics for current realization
    # ==========================================

    mean_solution += uh.x.array

    energy_form = fem.form(
        ufl.inner(uh, uh) * ufl.dx
    )

    energy_mc = domain.comm.allreduce(
        fem.assemble_scalar(energy_form),
        op=MPI.SUM
    )

    energy.append(energy_mc)

uh.name = "solution"

mean_solution /= Nmc

u_mean = fem.Function(V)

u_mean.x.array[:] = mean_solution

u_mean.name = "mean_solution"

mean_energy = np.mean(energy)

std_energy = np.std(energy)

if domain.comm.rank == 0:

    print()
    print("===================================")
    print("Monte Carlo statistics")
    print("===================================")
    print(f"Nmc         = {Nmc}")
    print(f"Mean energy = {mean_energy:.6e}")
    print(f"Std energy  = {std_energy:.6e}")
    


# ============================================================
# Output
# ============================================================

import csv
import os

os.makedirs(
    "results/errors",
    exist_ok=True
)

with open(
    "results/errors/stochastic_statistics.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "Nmc",
        "mean_energy",
        "std_energy"
    ])

    writer.writerow([
        Nmc,
        mean_energy,
        std_energy
    ])
    
os.makedirs(
    "results/vtk",
    exist_ok=True
)

with VTXWriter(
    domain.comm,
    "results/vtk/mean_solution.bp",
    [u_mean]
) as vtx:
    vtx.write(0.0)