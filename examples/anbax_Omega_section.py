"""ANBA4 Omega-section example (scikit-fem port, simplified)."""
import numpy as np
from skfem import MeshTri
from anba4 import Anbax, IsotropicMaterial

E, nu = 1.0, 0.33
mesh = MeshTri.init_tensor(np.linspace(-2, 2, 25), np.linspace(-1, 1.5, 25))
n_elem = mesh.nelements
materials = np.zeros(n_elem, dtype=int)
plane_orientations = np.full(n_elem, 90.0)
fiber_orientations = np.zeros(n_elem)

mat1 = IsotropicMaterial([E, nu], 1.0)
anba = Anbax(mesh, 2, [mat1], materials, plane_orientations, fiber_orientations)

stiff = anba.compute()
print("Stiffness (Omega):", np.diag(stiff))
