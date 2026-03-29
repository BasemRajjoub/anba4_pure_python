"""ANBA4 rotation multimat example (scikit-fem port)."""
import numpy as np
from skfem import MeshTri
from anba4 import Anbax, IsotropicMaterial, OrthotropicMaterial

props = np.array([
    [140.0e9, 10.0e9, 10.0e9],
    [6.0e9, 6.0e9, 6.0e9],
    [0.4, 0.3, 0.4]
])

mesh = MeshTri.init_tensor(np.linspace(-0.5, 0.5, 15), np.linspace(-0.5, 0.5, 5))
n_elem = mesh.nelements
materials = np.zeros(n_elem, dtype=int)
fiber_orientations = np.tile([20, -70], (n_elem // 2 + 1))[:n_elem]
plane_orientations = np.full(n_elem, 23.0)  # 23 deg rotation

mat1 = OrthotropicMaterial(props, 1600.0)
mat2 = IsotropicMaterial([140.0e9/100, 0.3], 1.0)
anba = Anbax(mesh, 1, [mat1, mat2], materials, plane_orientations, fiber_orientations, 1e9)

stiff = anba.compute()
print("Stiffness (rotation_multimat):", np.diag(stiff))

if __name__ == '__test__':
    # Expected FEniCS values (approximate)
    pass
