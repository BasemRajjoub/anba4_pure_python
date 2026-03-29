"""ANBA4 multi-material with hole #2 (scikit-fem port)."""
import numpy as np
from skfem import MeshTri
from anba4 import Anbax, IsotropicMaterial

mesh = MeshTri.init_tensor(np.linspace(-10, 10, 21), np.linspace(-10, 10, 21))
n_elem = mesh.nelements
materials = np.zeros(n_elem, dtype=int)
plane_orientations = np.zeros(n_elem)
fiber_orientations = np.zeros(n_elem)

centroids = mesh.p[:, mesh.t].mean(axis=1)
materials[centroids[1] <= 0] = 1
materials[(centroids[0] >= -2) & (centroids[0] <= 2) & (centroids[1] >= -6) & (centroids[1] <= 6)] = 2

mat1 = IsotropicMaterial([80000, 0.3], 1.0)
mat2 = IsotropicMaterial([40000, 0.3], 1.0)
mat3 = IsotropicMaterial([80, 0.3], 1.0)

anba = Anbax(mesh, 1, [mat1, mat2, mat3], materials, plane_orientations, fiber_orientations)
stiff = anba.compute()
print("Stiffness (multimat2):", np.diag(stiff))
