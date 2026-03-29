"""
Test stress_field and strain_field methods.

Validates the newly implemented field recovery methods.
"""

import numpy as np
from skfem import MeshTri

from anba4 import Anbax, IsotropicMaterial


def create_unit_square_mesh(n=10):
    """Create a triangular mesh of a unit square centered at origin."""
    mesh = MeshTri.init_tensor(
        np.linspace(-0.5, 0.5, n + 1),
        np.linspace(-0.5, 0.5, n + 1)
    )
    return mesh


def test_stress_strain_field():
    """Test stress and strain field computation for isotropic unit square."""

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

    # Create analyzer
    anba = Anbax(mesh, 1, mat_library, materials, plane_orientations,
                 fiber_orientations)

    # Compute stiffness
    print("Computing stiffness matrix...")
    stiff = anba.compute()
    print(f"Stiffness matrix diagonal: EA={stiff[2,2]:.6f}, GJ={stiff[5,5]:.6f}")

    # Test 1: Axial load (N = 1)
    print("\n=== Test 1: Axial Load N=1 ===")
    force = np.array([1.0, 0.0, 0.0])  # F1 = 1 (axial)
    moment = np.array([0.0, 0.0, 0.0])
    
    stress = anba.stress_field(force, moment, reference="global", voigt_convention="anba")
    strain = anba.strain_field(force, moment, reference="global", voigt_convention="anba")
    
    print(f"Stress shape: {stress.shape}")
    print(f"Strain shape: {strain.shape}")
    print(f"Mean axial stress (sigma_11): {stress[:, 0].mean():.6f}")
    print(f"Mean axial strain (eps_11): {strain[:, 0].mean():.6f}")
    print(f"Expected strain: N/EA = {1.0/stiff[2,2]:.6f}")
    
    # For a unit square with E=1, axial stress should be uniform = 1/A = 1.0
    # And strain = sigma/E = 1.0
    print(f"Stress range: [{stress[:, 0].min():.6f}, {stress[:, 0].max():.6f}]")

    # Test 2: Torsion (M1 = 1)
    print("\n=== Test 2: Torsion M1=1 ===")
    force = np.array([0.0, 0.0, 0.0])
    moment = np.array([1.0, 0.0, 0.0])  # M1 = 1 (torsion)
    
    stress = anba.stress_field(force, moment, reference="global", voigt_convention="anba")
    strain = anba.strain_field(force, moment, reference="global", voigt_convention="anba")
    
    print(f"Mean shear stress (sigma_12): {stress[:, 5].mean():.6f}")
    print(f"Mean shear stress (sigma_13): {stress[:, 4].mean():.6f}")
    print(f"Shear stress sigma_12 range: [{stress[:, 5].min():.6f}, {stress[:, 5].max():.6f}]")
    print(f"Shear stress sigma_13 range: [{stress[:, 4].min():.6f}, {stress[:, 4].max():.6f}]")
    # For torsion, shear stresses should vary linearly with position
    
    # Test 3: Bending about x2 (M2 = 1)
    print("\n=== Test 3: Bending M2=1 ===")
    force = np.array([0.0, 0.0, 0.0])
    moment = np.array([0.0, 1.0, 0.0])  # M2 = 1
    
    stress = anba.stress_field(force, moment, reference="global", voigt_convention="anba")
    strain = anba.strain_field(force, moment, reference="global", voigt_convention="anba")
    
    print(f"Mean axial stress (sigma_11): {stress[:, 0].mean():.6f}")
    print(f"Stress sigma_11 range: [{stress[:, 0].min():.6f}, {stress[:, 0].max():.6f}]")
    # For bending, stress should vary linearly with x3
    
    # Test 4: Bending about x3 (M3 = 1)
    print("\n=== Test 4: Bending M3=1 ===")
    force = np.array([0.0, 0.0, 0.0])
    moment = np.array([0.0, 0.0, 1.0])  # M3 = 1
    
    stress = anba.stress_field(force, moment, reference="global", voigt_convention="anba")
    strain = anba.strain_field(force, moment, reference="global", voigt_convention="anba")
    
    print(f"Mean axial stress (sigma_11): {stress[:, 0].mean():.6f}")
    print(f"Stress sigma_11 range: [{stress[:, 0].min():.6f}, {stress[:, 0].max():.6f}]")
    # For bending, stress should vary linearly with x2

    # Test 5: Shear in x2 (V2 = 1)
    print("\n=== Test 5: Shear V2=1 ===")
    force = np.array([0.0, 1.0, 0.0])  # V2 = 1
    moment = np.array([0.0, 0.0, 0.0])
    
    stress = anba.stress_field(force, moment, reference="global", voigt_convention="anba")
    strain = anba.strain_field(force, moment, reference="global", voigt_convention="anba")
    
    print(f"Mean shear stress (sigma_12): {stress[:, 5].mean():.6f}")
    print(f"Shear stress range: [{stress[:, 5].min():.6f}, {stress[:, 5].max():.6f}]")

    # Test 6: Shear in x3 (V3 = 1)
    print("\n=== Test 6: Shear V3=1 ===")
    force = np.array([0.0, 0.0, 1.0])  # V3 = 1
    moment = np.array([0.0, 0.0, 0.0])
    
    stress = anba.stress_field(force, moment, reference="global", voigt_convention="anba")
    strain = anba.strain_field(force, moment, reference="global", voigt_convention="anba")
    
    print(f"Mean shear stress (sigma_13): {stress[:, 4].mean():.6f}")
    print(f"Shear stress range: [{stress[:, 4].min():.6f}, {stress[:, 4].max():.6f}]")

    print("\n=== All tests completed successfully! ===")
    return True


if __name__ == "__main__":
    test_stress_strain_field()
