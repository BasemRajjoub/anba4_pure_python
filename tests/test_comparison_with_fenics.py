"""
Comprehensive comparison test between ANBA4-skfem and original FEniCS expected values.

This test compares the skfem port results against the expected values from the
original FEniCS test_anbax_isotropic.py.

Expected FEniCS values (from anba4-master/examples/test/test_anbax_isotropic.py):
- Uses UnitSquareMesh(10, 10) shifted by (-0.5, -0.5)
- Degree 2 elements
- E = 1.0, nu = 0.33, rho = 1.0
"""

import numpy as np
from skfem import MeshTri

import sys
from anba4 import Anbax, IsotropicMaterial


def create_unit_square_mesh(n=10):
    """Create a triangular mesh of a unit square centered at origin.
    
    This matches the FEniCS UnitSquareMesh(10, 10) shifted by (-0.5, -0.5).
    """
    mesh = MeshTri.init_tensor(
        np.linspace(-0.5, 0.5, n + 1),
        np.linspace(-0.5, 0.5, n + 1)
    )
    return mesh


def test_comparison_with_fenics():
    """Compare skfem port results with FEniCS expected values."""
    
    print("=" * 70)
    print("ANBA4-skfem vs FEniCS Comparison Test")
    print("=" * 70)
    
    # Material parameters (same as FEniCS test)
    E = 1.0
    nu = 0.33
    rho = 1.0
    
    # Create mesh
    mesh = create_unit_square_mesh(n=10)
    n_elem = mesh.nelements
    
    print(f"\nMesh: {n_elem} elements")
    
    # Material arrays (uniform material)
    materials = np.zeros(n_elem, dtype=int)
    plane_orientations = np.full(n_elem, 90.0)
    fiber_orientations = np.zeros(n_elem)
    
    # Create material library
    mat1 = IsotropicMaterial([E, nu], rho)
    mat_library = [mat1]
    
    # Create analyzer with degree 2 (same as FEniCS test)
    anba = Anbax(mesh, 2, mat_library, materials, plane_orientations,
                 fiber_orientations)
    
    # Compute stiffness and mass
    print("\nComputing stiffness matrix...")
    stiff = anba.compute()
    
    print("Computing mass matrix...")
    mass = anba.inertia()
    
    # Expected values from FEniCS (from test_anbax_isotropic.py)
    expected_stiff = np.array([
        [3.11064401e-01, -5.76267647e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.19833734e-16],
        [-5.76267647e-07, 3.11064401e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -2.66259961e-16],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 5.13607660e-16, -3.11112792e-16, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 4.93713695e-16, 8.33333333e-02, -8.92869184e-17, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, -2.32228751e-16, -8.22353985e-17, 8.33333333e-02, 0.00000000e+00],
        [2.14650429e-16, -2.41327387e-16, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.28559454e-02]
    ])
    
    expected_mass = np.array([
        [1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.64291930e-17],
        [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -1.30104261e-17],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, -3.64291930e-17, 1.30104261e-17, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, -3.64291930e-17, 8.33333333e-02, -2.60208521e-18, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 1.30104261e-17, -2.60208521e-18, 8.33333333e-02, 0.00000000e+00],
        [3.64291930e-17, -1.30104261e-17, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.66666667e-01]
    ])
    
    print("\n" + "=" * 70)
    print("STIFFNESS MATRIX COMPARISON")
    print("=" * 70)
    
    print("\nComputed stiffness matrix:")
    print(stiff)
    
    print("\nExpected FEniCS stiffness matrix:")
    print(expected_stiff)
    
    # Compare diagonal elements (most important)
    print("\n" + "-" * 70)
    print("Diagonal Element Comparison:")
    print("-" * 70)
    labels = ["GA22 (shear x2)", "GA33 (shear x3)", "EA (axial)", 
              "EI22 (bend x2)", "EI33 (bend x3)", "GJ (torsion)"]
    
    max_stiff_err = 0.0
    for i in range(6):
        computed = stiff[i, i]
        expected = expected_stiff[i, i]
        rel_err = abs(computed - expected) / (abs(expected) + 1e-10)
        if rel_err > max_stiff_err:
            max_stiff_err = rel_err
        status = "PASS" if rel_err < 0.02 else "FAIL"
        print(f"  {labels[i]}: {computed:.6e} vs {expected:.6e} ({rel_err*100:.2f}% error) - {status}")
    
    print("\n" + "-" * 70)
    print("MASS MATRIX COMPARISON")
    print("=" * 70)
    
    print("\nComputed mass matrix:")
    print(mass)
    
    print("\nExpected FEniCS mass matrix:")
    print(expected_mass)
    
    print("\n" + "-" * 70)
    print("Diagonal Element Comparison:")
    print("-" * 70)
    labels_m = ["m11 (mass)", "m22 (mass)", "m33 (mass)", 
               "I22 (rot x2)", "I33 (rot x3)", "Ip (polar)"]
    
    max_mass_err = 0.0
    for i in range(6):
        computed = mass[i, i]
        expected = expected_mass[i, i]
        rel_err = abs(computed - expected) / (abs(expected) + 1e-10)
        if rel_err > max_mass_err:
            max_mass_err = rel_err
        status = "PASS" if rel_err < 0.02 else "FAIL"
        print(f"  {labels_m[i]}: {computed:.6e} vs {expected:.6e} ({rel_err*100:.2f}% error) - {status}")
    
    # Overall summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_pass = max_stiff_err < 0.02 and max_mass_err < 0.02
    
    print(f"\nMax stiffness relative error: {max_stiff_err*100:.2f}%")
    print(f"Max mass relative error: {max_mass_err*100:.2f}%")
    
    if all_pass:
        print("\n*** ALL TESTS PASSED - skfem port matches FEniCS within 2% ***")
        return True
    else:
        print("\n*** SOME TESTS FAILED ***")
        return False


if __name__ == "__main__":
    success = test_comparison_with_fenics()
    exit(0 if success else 1)
