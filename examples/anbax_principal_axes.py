"""ANBA4 principal axes example (scikit-fem port)."""
import numpy as np
from skfem import MeshTri
from anba4 import Anbax, IsotropicMaterial, decouple_stiffness, principal_axes_rotation_angle

E, nu = 1.0, 0.33
mesh = MeshTri.init_tensor(np.linspace(-0.5, 1.5, 21), np.linspace(-1, 1, 21))

# Rotate mesh coordinates
rot_angle = 30.0 / 180.0 * np.pi
cr, sr = np.cos(rot_angle), np.sin(rot_angle)
rot_tensor = np.array([[cr, -sr], [sr, cr]])
mesh.p[0:2, :] = (rot_tensor @ mesh.p[0:2, :]) + np.array([[3], [1]])

n_elem = mesh.nelements
materials = np.zeros(n_elem, dtype=int)
plane_orientations = np.full(n_elem, 90.0)
fiber_orientations = np.zeros(n_elem)

mat1 = IsotropicMaterial([E, nu], 1.0)
anba = Anbax(mesh, 2, [mat1], materials, plane_orientations, fiber_orientations)

stiff = anba.compute()
decoupled = decouple_stiffness(stiff)
angle = principal_axes_rotation_angle(decoupled)
print(f"Principal axes rotation angle: {angle / np.pi * 180.0:.2f} degrees")
print("Stiffness:", np.diag(stiff))
