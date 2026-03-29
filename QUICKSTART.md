# ANBA4 scikit-fem Port - Quick Start

## Installation

```bash
# Install from the repository root
pip install -e .
```

## Run Tests

```bash
pytest tests/ -v
```

**Expected output:** 4 tests passing ✅

## Run Examples

```bash
# Simple isotropic example
python examples/anbax_isotropic.py

# Isotropic with singular solver
python examples/anbax_isotropic_singular.py

# Orthotropic composite with fiber rotation
python examples/anbax_rotation.py

# Multi-material example
python examples/anbax_rotation_multimat.py
```

## Basic Usage

```python
import numpy as np
from skfem import MeshTri
from anba4 import Anbax, IsotropicMaterial

# Create mesh (10x10 unit square centered at origin)
mesh = MeshTri.init_tensor(
    np.linspace(-0.5, 0.5, 11),
    np.linspace(-0.5, 0.5, 11)
)

# Define material: E=1.0, nu=0.33, rho=1.0
mat = IsotropicMaterial([1.0, 0.33], rho=1.0)

# Material assignment (uniform)
n_elem = mesh.nelements
materials = np.zeros(n_elem, dtype=int)
plane_orientations = np.full(n_elem, 90.0)
fiber_orientations = np.zeros(n_elem)

# Create analyzer
anba = Anbax(mesh, degree=2, mat_library=[mat], 
             materials=materials,
             plane_orientations=plane_orientations,
             fiber_orientations=fiber_orientations)

# Compute stiffness and mass matrices (6x6 numpy arrays)
K = anba.compute()
M = anba.inertia()

print(f"Axial stiffness EA: {K[2,2]:.6f}")
print(f"Torsional stiffness GJ: {K[5,5]:.6f}")

# Compute stress field for unit axial load
stress = anba.stress_field(force=[1.0, 0.0, 0.0], moment=[0.0, 0.0, 0.0])
print(f"Stress shape: {stress.shape}")  # (n_elements, 6)
```

## Orthotropic Material Example

```python
from anba4 import OrthotropicMaterial

# Material properties layout:
# [[E_xx, E_yy, E_zz],
#  [G_yz, G_xz, G_xy],
#  [nu_zy, nu_zx, nu_xy]]
props = np.array([
    [140.0e9, 10.0e9, 10.0e9],  # Young's moduli (Pa)
    [6.0e9, 6.0e9, 6.0e9],      # Shear moduli (Pa)
    [0.4, 0.3, 0.4]             # Poisson's ratios
])

mat = OrthotropicMaterial(props, rho=1600.0)
anba = Anbax(mesh, 1, [mat], materials, plane_orientations, fiber_orientations)
```

## Key API Changes from FEniCS Version

| Old (FEniCS) | New (scikit-fem) |
|---|---|
| `from anba4 import anbax` | `from anba4 import Anbax` |
| `anbax(mesh, ...)` | `Anbax(mesh, ...)` |
| Returns `PETSc.Mat` | Returns `numpy.ndarray` |
| `stiff.view()` | `print(stiff)` |
| `material.IsotropicMaterial` | `IsotropicMaterial` (direct import) |
| `dolfin.UnitSquareMesh(10,10)` | `MeshTri.init_tensor(np.linspace(...), ...)` |
| `MeshFunction("size_t", mesh, ...)` | `np.zeros(mesh.nelements, dtype=int)` |
| Writes XDMF files | Returns numpy arrays |

## Output Format

All matrices are 6x6 numpy arrays with the following convention:

```
Indices: [V2, V3, N, M2, M3, T]
  V2, V3: Shear forces (x2, x3 directions)
  N:      Axial force (x1 direction)
  M2, M3: Bending moments (x2, x3 directions)
  T:      Torsional moment (x1 direction)
```

Example diagonal elements:
- K[0,0]: GA22 (shear stiffness)
- K[1,1]: GA33 (shear stiffness)
- K[2,2]: EA (axial stiffness)
- K[3,3]: EI22 (bending stiffness)
- K[4,4]: EI33 (bending stiffness)
- K[5,5]: GJ (torsional stiffness)

## Support for Complex Geometries

For polygon or complex meshes, use `gmsh` + `meshio`:

```python
import meshio
import numpy as np
from skfem import MeshTri

# Load mesh from gmsh
mesh_obj = meshio.read("geometry.msh")
points = mesh_obj.points[:, :2]  # 2D coords
cells = mesh_obj.cells_dict["triangle"]
mesh = MeshTri(points.T, cells.T)
```

## Troubleshooting

### Warning: "spsolve requires A be CSC or CSR matrix format"
This is a performance warning from scipy. Matrices are automatically converted; computation is still correct.

### Import Error: "No module named 'anba4'"
Ensure you ran `pip install -e .` from the repository root.

### Test Failures
Run `pytest tests/ -v` to see detailed output. All 4 tests should pass.

## Documentation

- **Theory**: See original ANBA4 paper (Morandini et al., 2010)
- **Class docstrings**: Available via `help(Anbax)` in Python
- **Examples**: Browse `examples/*.py` for usage patterns

## License

GNU General Public License v3 - see COPYING file.
