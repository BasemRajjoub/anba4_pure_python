"""
Test AnbaxSingular class.

Validates the iterative solver variant produces same results as direct solver.
"""

import numpy as np
from skfem import MeshTri

import sys
from anba4 import Anbax, AnbaxSingular, IsotropicMaterial


def create_unit_square_mesh(n=10):
    """Create a triangular mesh of a unit square centered at origin."""
    mesh = MeshTri.init_tensor(
        np.linspace(-0.5, 0.5, n + 1),
        np.linspace(-0.5, 0.5, n + 1)
    )
    return mesh


def test_singular_vs_direct():
    """Compare AnbaxSingular results with Anbax (direct solver)."""
    
    # Material parameters
    E = 1.0
    nu = 0.33
    rho = 1.0
    
    # Create mesh
    mesh = create_unit_square_mesh(n=10)
    n_elem = mesh.nelements
    
    # Material arrays
    materials = np.zeros(n_elem, dtype=int)
    plane_orientations = np.full(n_elem, 90.0)
    fiber_orientations = np.zeros(n_elem)
    
    # Create material library
    mat1 = IsotropicMaterial([E, nu], rho)
    mat_library = [mat1]
    
    # Test direct solver (Anbax)
    print("Testing Anbax (direct solver)...")
    anba_direct = Anbax(mesh, 1, mat_library, materials, plane_orientations,
                        fiber_orientations)
    stiff_direct = anba_direct.compute()
    mass_direct = anba_direct.inertia()
    
    # Test iterative solver (AnbaxSingular)
    print("\nTesting AnbaxSingular (sparse direct solver with augmented system)...")
    anba_singular = AnbaxSingular(mesh, 1, mat_library, materials, plane_orientations,
                                  fiber_orientations)
    stiff_singular = anba_singular.compute()
    mass_singular = anba_singular.inertia()
    
    # Compare stiffness matrices
    print("\n=== Stiffness Matrix Comparison ===")
    print("Direct solver diagonal:")
    print(f"  GA (shear): {stiff_direct[0,0]:.6f}, {stiff_direct[1,1]:.6f}")
    print(f"  EA (axial): {stiff_direct[2,2]:.6f}")
    print(f"  EI (bend):  {stiff_direct[3,3]:.6f}, {stiff_direct[4,4]:.6f}")
    print(f"  GJ (tors):  {stiff_direct[5,5]:.6f}")
    
    print("Iterative solver diagonal:")
    print(f"  GA (shear): {stiff_singular[0,0]:.6f}, {stiff_singular[1,1]:.6f}")
    print(f"  EA (axial): {stiff_singular[2,2]:.6f}")
    print(f"  EI (bend):  {stiff_singular[3,3]:.6f}, {stiff_singular[4,4]:.6f}")
    print(f"  GJ (tors):  {stiff_singular[5,5]:.6f}")
    
    # Compute relative errors
    stiff_diff = np.abs(stiff_direct - stiff_singular)
    stiff_rel_err = stiff_diff / (np.abs(stiff_direct) + 1e-10)
    max_rel_err = np.max(stiff_rel_err)
    
    print(f"\nMax relative error: {max_rel_err:.6e}")
    
    # Compare mass matrices
    print("\n=== Mass Matrix Comparison ===")
    mass_diff = np.abs(mass_direct - mass_singular)
    mass_rel_err = mass_diff / (np.abs(mass_direct) + 1e-10)
    max_mass_err = np.max(mass_rel_err)
    
    print(f"Max mass relative error: {max_mass_err:.6e}")
    
    # Test stress field
    print("\n=== Stress Field Comparison ===")
    force = np.array([1.0, 0.0, 0.0])  # Axial load
    moment = np.array([0.0, 0.0, 0.0])
    
    stress_direct = anba_direct.stress_field(force, moment)
    stress_singular = anba_singular.stress_field(force, moment)
    
    stress_diff = np.abs(stress_direct - stress_singular)
    max_stress_err = np.max(stress_diff) / (np.mean(np.abs(stress_direct)) + 1e-10)
    
    print(f"Max stress relative error: {max_stress_err:.6e}")
    
    # Summary
    print("\n=== Summary ===")
    if max_rel_err < 0.05 and max_mass_err < 0.01 and max_stress_err < 0.05:
        print("PASS: AnbaxSingular produces results within 5% of Anbax")
        return True
    else:
        print("FAIL: Results differ significantly")
        return False


if __name__ == "__main__":
    test_singular_vs_direct()
