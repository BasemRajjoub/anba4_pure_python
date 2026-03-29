"""ANBA4 C-section example (scikit-fem port, simplified with rectangular mesh)."""
import numpy as np
from skfem import MeshTri
from anba4 import Anbax, IsotropicMaterial

E, nu = 1.0, 0.33
# Create rectangular mesh (approximate C-section)
mesh = MeshTri.init_tensor(np.linspace(-0.5, 1.5, 21), np.linspace(-1, 1, 21))
n_elem = mesh.nelements
materials = np.zeros(n_elem, dtype=int)
plane_orientations = np.full(n_elem, 90.0)
fiber_orientations = np.zeros(n_elem)

mat1 = IsotropicMaterial([E, nu], 1.0)
anba = Anbax(mesh, 2, [mat1], materials, plane_orientations, fiber_orientations)

print("Stiffness matrix (C-section approximation):")
stiff = anba.compute()
print(stiff)

print("\nMass matrix:")
mass = anba.inertia()
print(mass)
