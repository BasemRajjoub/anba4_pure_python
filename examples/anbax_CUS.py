"""ANBA4 CUS composite box beam example (scikit-fem port, simplified)."""
import numpy as np
from skfem import MeshTri
from anba4 import AnbaxSingular, OrthotropicMaterial

# Carbon fiber properties
props = np.array([
    [140.0e9, 10.0e9, 10.0e9],
    [6.0e9, 6.0e9, 6.0e9],
    [0.4, 0.3, 0.4]
])

# Create rectangular mesh (approximate box cross-section)
mesh = MeshTri.init_tensor(np.linspace(-0.476, 0.476, 15), np.linspace(-0.265, 0.265, 10))
n_elem = mesh.nelements

materials = np.zeros(n_elem, dtype=int)
fiber_orientations = np.zeros(n_elem)
plane_orientations = np.zeros(n_elem)

# Assign different plane orientations per region
centroids = mesh.p[:, mesh.t].mean(axis=1)
for i, (x, y) in enumerate(centroids.T):
    if x > 0.3:  # right wall
        plane_orientations[i] = 0
    elif x < -0.3:  # left wall
        plane_orientations[i] = 180
    elif y > 0.2:  # top wall
        plane_orientations[i] = 90
    else:  # bottom wall
        plane_orientations[i] = 270

mat1 = OrthotropicMaterial(props, 1600.0)
anba = AnbaxSingular(mesh, 2, [mat1], materials, plane_orientations, fiber_orientations, 1.0)

stiff = anba.compute()
print("Stiffness (CUS box):", np.diag(stiff))
