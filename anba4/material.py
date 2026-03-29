"""
Pure Python material module for ANBA4-skfem.

Replaces the C++ FEniCS-based material.cpp from the original ANBA4.

Original ANBA4:
    Copyright (C) 2018 Marco Morandini
    https://github.com/manuelma/anba4

scikit-fem port:
    Copyright (C) 2024-2026 Basem Rajjoub

Licensed under GNU GPL v3 - see COPYING file for details.
"""

import numpy as np
from abc import ABC, abstractmethod


class Material(ABC):
    """Base class for materials."""

    def __init__(self, rho: float = 0.0):
        self.rho = rho
        self._transform_matrix = np.zeros((6, 6))
        self._modulus = np.zeros((6, 6))
        self._rotated_stress_modulus = np.zeros((6, 6))

    @abstractmethod
    def compute_elastic_modulus(self, alpha: float, beta: float) -> np.ndarray:
        """Compute the elastic modulus matrix rotated by plane/fiber angles."""
        pass

    @abstractmethod
    def compute_rotated_stress_elastic_modulus(self, alpha: float, beta: float) -> np.ndarray:
        """Compute the rotated stress elastic modulus."""
        pass

    def transformation_matrix(self, alpha: float, beta: float) -> np.ndarray:
        """
        Compute the 6x6 transformation matrix for stress/strain rotation.

        Parameters
        ----------
        alpha : float
            Fiber plane orientation angle in degrees
        beta : float
            Fiber orientation angle in degrees

        Returns
        -------
        T : ndarray (6, 6)
            Transformation matrix
        """
        pi180 = np.pi / 180.0

        # alpha -> fiber plane orientation; beta -> fiber orientation
        sn_a = -np.sin(alpha * pi180)
        cn_a = np.cos(alpha * pi180)
        sn_b = -np.sin(beta * pi180)
        cn_b = np.cos(beta * pi180)

        T = np.zeros((6, 6))

        T[0, 0] = cn_a * cn_a * cn_b * cn_b
        T[0, 1] = sn_a * sn_a
        T[0, 2] = cn_a * cn_a * sn_b * sn_b
        T[0, 3] = -2.0 * cn_a * sn_a * sn_b
        T[0, 4] = -2.0 * cn_a * cn_a * sn_b * cn_b
        T[0, 5] = 2.0 * cn_a * sn_a * cn_b

        T[1, 0] = sn_a * sn_a * cn_b * cn_b
        T[1, 1] = cn_a * cn_a
        T[1, 2] = sn_a * sn_a * sn_b * sn_b
        T[1, 3] = 2.0 * cn_a * sn_a * sn_b
        T[1, 4] = -2.0 * sn_a * sn_a * sn_b * cn_b
        T[1, 5] = -2.0 * cn_a * sn_a * cn_b

        T[2, 0] = sn_b * sn_b
        T[2, 2] = cn_b * cn_b
        T[2, 4] = 2.0 * cn_b * sn_b

        T[3, 0] = -sn_a * sn_b * cn_b
        T[3, 2] = sn_a * sn_b * cn_b
        T[3, 3] = cn_a * cn_b
        T[3, 4] = -sn_a * cn_b * cn_b + sn_a * sn_b * sn_b
        T[3, 5] = cn_a * sn_b

        T[4, 0] = cn_a * sn_b * cn_b
        T[4, 2] = -cn_a * sn_b * cn_b
        T[4, 3] = sn_a * cn_b
        T[4, 4] = -cn_a * sn_b * sn_b + cn_a * cn_b * cn_b
        T[4, 5] = sn_a * sn_b

        T[5, 0] = -sn_a * cn_a * cn_b * cn_b
        T[5, 1] = cn_a * sn_a
        T[5, 2] = -sn_a * cn_a * sn_b * sn_b
        T[5, 3] = -cn_a * cn_a * sn_b + sn_a * sn_a * sn_b
        T[5, 4] = 2.0 * sn_a * sn_b * cn_a * cn_b
        T[5, 5] = cn_a * cn_a * cn_b - sn_a * sn_a * cn_b

        self._transform_matrix = T
        return T


class IsotropicMaterial(Material):
    """Isotropic material definition."""

    def __init__(self, props, rho: float = 0.0):
        """
        Initialize isotropic material.

        Parameters
        ----------
        props : array-like
            [E, nu] - Young's modulus and Poisson's ratio
        rho : float
            Material density
        """
        super().__init__(rho)

        if hasattr(props, '__len__'):
            E = props[0]
            nu = props[1]
        else:
            raise ValueError("props must be [E, nu]")

        G = E / (2 * (1 + nu))
        delta = E / (1 + nu) / (1 - 2 * nu)
        diag = (1 - nu) * delta
        off_diag = nu * delta

        self._local_modulus = np.zeros((6, 6))
        self._local_modulus[0, 0] = diag
        self._local_modulus[0, 1] = off_diag
        self._local_modulus[0, 2] = off_diag
        self._local_modulus[1, 0] = off_diag
        self._local_modulus[1, 1] = diag
        self._local_modulus[1, 2] = off_diag
        self._local_modulus[2, 0] = off_diag
        self._local_modulus[2, 1] = off_diag
        self._local_modulus[2, 2] = diag
        self._local_modulus[3, 3] = G
        self._local_modulus[4, 4] = G
        self._local_modulus[5, 5] = G

    def compute_elastic_modulus(self, alpha: float, beta: float) -> np.ndarray:
        """For isotropic material, modulus is independent of orientation."""
        return self._local_modulus.copy()

    def compute_rotated_stress_elastic_modulus(self, alpha: float, beta: float) -> np.ndarray:
        """Compute rotated stress modulus."""
        T = self.transformation_matrix(alpha, beta)
        return self._local_modulus @ T.T


class OrthotropicMaterial(Material):
    """Orthotropic material definition."""

    def __init__(self, props, rho: float = 0.0):
        """
        Initialize orthotropic material.

        Parameters
        ----------
        props : ndarray (3, 3)
            Material properties matrix:
            [[E_xx, E_yy, E_zz],
             [G_yz, G_xz, G_xy],
             [nu_zy, nu_zx, nu_xy]]
        rho : float
            Material density
        """
        super().__init__(rho)

        props = np.asarray(props)

        e_xx = props[0, 0]
        e_yy = props[0, 1]
        e_zz = props[0, 2]
        g_yz = props[1, 0]
        g_xz = props[1, 1]
        g_xy = props[1, 2]
        nu_zy = props[2, 0]
        nu_zx = props[2, 1]
        nu_xy = props[2, 2]

        # Calculate the other 3 Poisson ratios
        nu_yx = e_yy * nu_xy / e_xx
        nu_xz = e_xx * nu_zx / e_zz
        nu_yz = e_yy * nu_zy / e_zz

        self._local_modulus = np.zeros((6, 6))

        delta = (1.0 - nu_xy*nu_yx - nu_yz*nu_zy - nu_xz*nu_zx
                 - 2.0*nu_yx*nu_zy*nu_xz) / (e_xx * e_yy * e_zz)

        self._local_modulus[0, 0] = (1.0 - nu_yz*nu_zy) / (e_yy * e_zz * delta)
        self._local_modulus[0, 1] = (nu_xy + nu_zy*nu_xz) / (e_xx * e_zz * delta)
        self._local_modulus[0, 2] = (nu_xz + nu_xy*nu_yz) / (e_xx * e_yy * delta)

        self._local_modulus[1, 0] = self._local_modulus[0, 1]
        self._local_modulus[1, 1] = (1.0 - nu_xz*nu_zx) / (e_xx * e_zz * delta)
        self._local_modulus[1, 2] = (nu_yz + nu_yx*nu_xz) / (e_xx * e_yy * delta)

        self._local_modulus[2, 0] = self._local_modulus[0, 2]
        self._local_modulus[2, 1] = self._local_modulus[1, 2]
        self._local_modulus[2, 2] = (1.0 - nu_xy*nu_yx) / (e_xx * e_yy * delta)

        self._local_modulus[3, 3] = g_yz
        self._local_modulus[4, 4] = g_xz
        self._local_modulus[5, 5] = g_xy

    def compute_elastic_modulus(self, alpha: float, beta: float) -> np.ndarray:
        """Compute elastic modulus rotated by plane/fiber angles."""
        T = self.transformation_matrix(alpha, beta)
        return T @ self._local_modulus @ T.T

    def compute_rotated_stress_elastic_modulus(self, alpha: float, beta: float) -> np.ndarray:
        """Compute rotated stress modulus."""
        T = self.transformation_matrix(alpha, beta)
        return self._local_modulus @ T.T
