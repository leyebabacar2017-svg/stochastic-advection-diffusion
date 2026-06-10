import dolfinx
from mpi4py import MPI

print("FEniCSx successfully loaded")
print("MPI size =", MPI.COMM_WORLD.size)
