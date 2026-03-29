"""ANBA4 isotropic cross-section example (scikit-fem port)."""
import numpy as np
from skfem import MeshTri
from anba4 import Anbax, IsotropicMaterial

E, nu, rho = 1.0, 0.33, 1.0
mesh = MeshTri.init_tensor(np.linspace(-0.5, 0.5, 11), np.linspace(-0.5, 0.5, 11))
n_elem = mesh.nelements
materials = np.zeros(n_elem, dtype=int)
plane_orientations = np.full(n_elem, 90.0)
fiber_orientations = np.zeros(n_elem)

mat1 = IsotropicMaterial([E, nu], rho)
anba = Anbax(mesh, 2, [mat1], materials, plane_orientations, fiber_orientations)

print("Stiffness matrix:")
stiff = anba.compute()
print(stiff)

print("\nMass matrix:")
mass = anba.inertia()
print(mass)

if __name__ == '__test__':
    expected_stiff = np.array([
        [ 3.11064401e-01, -5.76267647e-07,  0., 0., 0., 2.19833734e-16],
        [-5.76267647e-07,  3.11064401e-01,  0., 0., 0., -2.66259961e-16],
        [ 0., 0.,  1.00000000e+00, 5.13607660e-16, -3.11112792e-16, 0.],
        [ 0., 0.,  4.93713695e-16, 8.33333333e-02, -8.92869184e-17, 0.],
        [ 0., 0., -2.32228751e-16, -8.22353985e-17,  8.33333333e-02, 0.],
        [ 2.14650429e-16, -2.41327387e-16,  0., 0., 0.,  5.28559454e-02]])
    expected_mass = np.array([
        [ 1., 0., 0., 0., 0., 3.64291930e-17],
        [ 0., 1., 0., 0., 0., -1.30104261e-17],
        [ 0., 0., 1., -3.64291930e-17, 1.30104261e-17, 0.],
        [ 0., 0., -3.64291930e-17, 8.33333333e-02, -2.60208521e-18, 0.],
        [ 0., 0., 1.30104261e-17, -2.60208521e-18,  8.33333333e-02, 0.],
        [ 3.64291930e-17, -1.30104261e-17,  0., 0., 0.,  1.66666667e-01]])
    np.testing.assert_almost_equal(stiff, expected_stiff, decimal=6)
    np.testing.assert_almost_equal(mass, expected_mass, decimal=6)
