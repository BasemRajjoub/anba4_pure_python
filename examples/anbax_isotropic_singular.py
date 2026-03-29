"""ANBA4 isotropic cross-section example using singular solver (scikit-fem port)."""
import numpy as np
from skfem import MeshTri
from anba4 import AnbaxSingular, IsotropicMaterial

E, nu, rho = 1.0, 0.33, 1.0
mesh = MeshTri.init_tensor(np.linspace(-0.5, 0.5, 11), np.linspace(-0.5, 0.5, 11))
n_elem = mesh.nelements
materials = np.zeros(n_elem, dtype=int)
plane_orientations = np.full(n_elem, 90.0)
fiber_orientations = np.zeros(n_elem)

mat1 = IsotropicMaterial([E, nu], rho)
anba = AnbaxSingular(mesh, 2, [mat1], materials, plane_orientations, fiber_orientations)

print("Stiffness matrix (singular solver):")
stiff = anba.compute()
print(stiff)

print("\nMass matrix (singular solver):")
mass = anba.inertia()
print(mass)
