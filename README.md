# ANBA4-skfem

A FEniCS-free port of ANBA4 for cross-section beam analysis using scikit-fem.

## Description

ANBA4-skfem computes 6x6 stiffness and mass matrices of composite beam cross sections based on the generalized eigenvector theory from Morandini et al. (2010) "Characteristic behavior of prismatic anisotropic beam via generalized eigenvectors".

This port replaces FEniCS with scikit-fem for FEM assembly and scipy for linear algebra, enabling Windows compatibility without requiring FEniCS installation.

## Features

- Compute 6x6 stiffness matrix (EA, GJ, EI22, EI33, GA22, GA33)
- Compute 6x6 mass matrix
- Stress and strain field recovery
- Support for isotropic and orthotropic materials
- Triangular and quadrilateral meshes
- Linear and quadratic elements

## Installation

```bash
pip install numpy scipy scikit-fem
```

## Usage

```python
import numpy as np
from skfem import MeshTri
from anba4 import Anbax, IsotropicMaterial

# Create mesh
mesh = MeshTri.init_tensor(
    np.linspace(-0.5, 0.5, 6),
    np.linspace(-0.5, 0.5, 6)
)

# Define material (E=1, nu=0.33, rho=1.0)
mat = IsotropicMaterial([1.0, 0.33], rho=1.0)

# Material assignments for each element
materials = np.zeros(mesh.nelements, dtype=int)
plane_orientations = np.full(mesh.nelements, 90.0)
fiber_orientations = np.zeros(mesh.nelements)

# Create analyzer
anba = Anbax(mesh, 1, [mat], materials, plane_orientations, fiber_orientations)

# Compute stiffness and mass
K = anba.compute()
M = anba.inertia()

print(f"EA (axial): {K[2,2]:.6f}")
print(f"GJ (torsion): {K[5,5]:.6f}")
print(f"EI22 (bending x2): {K[3,3]:.6f}")
print(f"EI33 (bending x3): {K[4,4]:.6f}")

# Compute stress field for axial load
force = [1.0, 0.0, 0.0]  # F1=1 (axial)
moment = [0.0, 0.0, 0.0]
stress = anba.stress_field(force, moment)
```

## Stiffness Matrix Convention

The 6x6 stiffness matrix uses the following convention:
- Row/Col 0: Shear V2 (force in x2 direction)
- Row/Col 1: Shear V3 (force in x3 direction)
- Row/Col 2: Axial N (force in x1 direction)
- Row/Col 3: Bending M2 (moment about x2)
- Row/Col 4: Bending M3 (moment about x3)
- Row/Col 5: Torsion T (moment about x1)

## Voigt Notation

Stress and strain vectors use ANBA ordering:
- [σ11, σ22, σ33, σ23, σ13, σ12]

## License

GNU General Public License v3

## Credits

Original ANBA4:
- Copyright (C) 2018 Marco Morandini
- https://github.com/manuelma/anba4

scikit-fem port:
- Copyright (C) 2024-2026 Basem Rajjoub
