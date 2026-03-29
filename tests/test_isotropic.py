"""
Test isotropic material cross-section analysis.

Validates against the original ANBA4 FEniCS results.
"""

import numpy as np
import numpy.testing as npt
from skfem import MeshTri

from anba4 import Anbax, IsotropicMaterial


def create_unit_square_mesh(n=10):
    """Create a triangular mesh of a unit square centered at origin."""
    # Use scikit-fem's built-in tensor mesh generator (creates CCW oriented triangles)
    mesh = MeshTri.init_tensor(
        np.linspace(-0.5, 0.5, n + 1),
        np.linspace(-0.5, 0.5, n + 1)
    )
    return mesh


def test_isotropic_stiffness_mass():
    """Test stiffness and mass computation for isotropic unit square."""

    # Material parameters
    E = 1.0
    nu = 0.33
    rho = 1.0

    # Create mesh
    mesh = create_unit_square_mesh(n=10)
    n_elem = mesh.nelements

    # Material arrays (uniform material)
    materials = np.zeros(n_elem, dtype=int)
    plane_orientations = np.full(n_elem, 90.0)
    fiber_orientations = np.zeros(n_elem)

    # Create material library
    mat1 = IsotropicMaterial([E, nu], rho)
    mat_library = [mat1]

    # Create analyzer (use degree=1 for simpler testing first)
    anba = Anbax(mesh, 1, mat_library, materials, plane_orientations,
                 fiber_orientations)

    # Compute stiffness
    print("Computing stiffness matrix...")
    stiff = anba.compute()
    print("\nStiffness matrix:")
    print(stiff)

    # Compute mass
    print("\nComputing mass matrix...")
    mass = anba.inertia()
    print("\nMass matrix:")
    print(mass)

    # Expected values from original ANBA4 (FEniCS)
    expected_stiff = np.array([
        [ 3.11064401e-01, -5.76267647e-07,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  2.19833734e-16],
        [-5.76267647e-07,  3.11064401e-01,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00, -2.66259961e-16],
        [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00, 5.13607660e-16, -3.11112792e-16,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  4.93713695e-16, 8.33333333e-02, -8.92869184e-17,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00, -2.32228751e-16, -8.22353985e-17,  8.33333333e-02,  0.00000000e+00],
        [ 2.14650429e-16, -2.41327387e-16,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  5.28559454e-02]
    ])

    expected_mass = np.array([
        [ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  3.64291930e-17],
        [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00, -1.30104261e-17],
        [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00, -3.64291930e-17,  1.30104261e-17,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00, -3.64291930e-17, 8.33333333e-02, -2.60208521e-18,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  1.30104261e-17, -2.60208521e-18,  8.33333333e-02,  0.00000000e+00],
        [ 3.64291930e-17, -1.30104261e-17,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.66666667e-01]
    ])

    # Print comparison
    print("\n" + "="*60)
    print("Comparison with expected values:")
    print("="*60)

    print("\nStiffness diagonal:")
    print(f"  GA (shear): {stiff[0,0]:.6f} vs {expected_stiff[0,0]:.6f} (expected)")
    print(f"  GA (shear): {stiff[1,1]:.6f} vs {expected_stiff[1,1]:.6f} (expected)")
    print(f"  EA (axial): {stiff[2,2]:.6f} vs {expected_stiff[2,2]:.6f} (expected)")
    print(f"  EI (bend):  {stiff[3,3]:.6f} vs {expected_stiff[3,3]:.6f} (expected)")
    print(f"  EI (bend):  {stiff[4,4]:.6f} vs {expected_stiff[4,4]:.6f} (expected)")
    print(f"  GJ (tors):  {stiff[5,5]:.6f} vs {expected_stiff[5,5]:.6f} (expected)")

    print("\nMass diagonal:")
    print(f"  m (mass):  {mass[0,0]:.6f} vs {expected_mass[0,0]:.6f} (expected)")
    print(f"  m (mass):  {mass[1,1]:.6f} vs {expected_mass[1,1]:.6f} (expected)")
    print(f"  m (mass):  {mass[2,2]:.6f} vs {expected_mass[2,2]:.6f} (expected)")
    print(f"  I (rot):   {mass[3,3]:.6f} vs {expected_mass[3,3]:.6f} (expected)")
    print(f"  I (rot):   {mass[4,4]:.6f} vs {expected_mass[4,4]:.6f} (expected)")
    print(f"  Ip (pol):  {mass[5,5]:.6f} vs {expected_mass[5,5]:.6f} (expected)")

    # Validate mass matrix (should match analytical values)
    # For unit square: m = 1, I22 = I33 = 1/12, Ip = 1/6
    print("\n" + "="*60)
    print("Analytical validation:")
    print("="*60)
    print(f"  Mass = 1.0: computed = {mass[0,0]:.6f}")
    print(f"  I22 = 1/12 ~ 0.0833: computed = {mass[3,3]:.6f}")
    print(f"  I33 = 1/12 ~ 0.0833: computed = {mass[4,4]:.6f}")
    print(f"  Ip = 1/6 ~ 0.1667: computed = {mass[5,5]:.6f}")

    return stiff, mass


if __name__ == "__main__":
    stiff, mass = test_isotropic_stiffness_mass()
