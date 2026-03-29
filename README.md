# ANBA4

ANBA4 computes the 6x6 stiffness and mass matrices of arbitrarily complex composite beam cross sections.

## Description

ANBA4-skfem is a FEniCS-free port of ANBA4 using scikit-fem for FEM assembly and scipy for linear algebra, enabling Windows compatibility without requiring FEniCS installation.

The method is based on the generalized eigenvector theory from Morandini et al. (2010). ANBA4 has been verified against the commercial solver VABS and validated against experimental measurements.

## Theory

The theory of ANBA4 is described in this work (and references therein):

**Morandini, M., Chierichetti, M., & Mantegazza, P.** (2010).
"Characteristic behavior of prismatic anisotropic beam via generalized eigenvectors."
*International Journal of Solids and Structures*, 47(10), 1327-1337.
https://doi.org/10.1016/j.ijsolstr.2010.01.017

ANBA4 has recently been verified against the commercial solver VABS and validated against experimental measurements:

**Feil, R., Pflumm, T., Bortolotti, P., & Morandini, M.** (2020).
"A cross-sectional aeroelastic analysis and structural optimization tool for slender composite structures."
*Composite Structures*, 253, 112755.
https://doi.org/10.1016/j.compstruct.2020.112755

## Features

- Compute 6x6 stiffness matrix (EA, GJ, EI22, EI33, GA22, GA33)
- Compute 6x6 mass matrix
- Stress and strain field recovery
- Support for isotropic and orthotropic materials
- Triangular and quadrilateral meshes
- Linear and quadratic elements
- Cross-platform compatibility (Windows, Linux, macOS)

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

## About This Port

This implementation replaces the original FEniCS/PETSc backend with scikit-fem and scipy, making ANBA4 available on Windows without requiring complex C++ build tools or FEniCS installation. The numerical algorithms and physical methods remain unchanged from the original.

**Key improvements:**
- Works natively on Windows, Linux, and macOS
- Pure Python implementation (except scikit-fem itself)
- Pip-installable with no C++ compiler required
- Fully compatible with modern Python ecosystems
- Identical numerical results to original ANBA4

## License

GNU General Public License v3 - see COPYING file for details.

## Credits

**Original ANBA4:**
- Copyright (C) 2018 Marco Morandini
- https://github.com/ANBA4/anba4

**scikit-fem port:**
- Copyright (C) 2024-2026 Basem Rajjoub

**Theory and validation:**
- Marco Morandini, Maria Chierichetti, Paolo Mantegazza (theory)
- Roland Feil, Tobias Pflumm, Pietro Bortolotti, Marco Morandini (VABS verification and experimental validation)
