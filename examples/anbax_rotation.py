"""ANBA4 rotation example with orthotropic composite (scikit-fem port)."""
import numpy as np
from skfem import MeshTri
from anba4 import Anbax, OrthotropicMaterial

# Orthotropic material (carbon fiber)
# props layout: [[E_xx, E_yy, E_zz], [G_yz, G_xz, G_xy], [nu_zy, nu_zx, nu_xy]]
props = np.array([
    [140.0e9, 10.0e9, 10.0e9],
    [6.0e9, 6.0e9, 6.0e9],
    [0.4, 0.3, 0.4]
])

mesh = MeshTri.init_tensor(np.linspace(-0.5, 0.5, 15), np.linspace(-0.5, 0.5, 5))
n_elem = mesh.nelements

# 16-ply layup with alternating +20/-70 deg fiber orientations
materials = np.zeros(n_elem, dtype=int)
fiber_orientations = np.tile([20, -70], (n_elem // 2 + 1))[:n_elem]
plane_orientations = np.zeros(n_elem)

mat1 = OrthotropicMaterial(props, 1600.0)
anba = Anbax(mesh, 1, [mat1], materials, plane_orientations, fiber_orientations, 1e9)

stiff = anba.compute()
mass = anba.inertia()
print("Stiffness diagonal:", np.diag(stiff))
print("Mass diagonal:", np.diag(mass))
