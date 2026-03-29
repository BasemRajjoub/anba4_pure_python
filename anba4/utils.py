"""
Utility functions for ANBA4-skfem.

Original ANBA4:
    Copyright (C) 2018 Marco Morandini
    https://github.com/manuelma/anba4

scikit-fem port:
    Copyright (C) 2024-2026 Basem Rajjoub

Licensed under GNU GPL v3 - see COPYING file for details.
"""

import numpy as np


def compute_shear_center(stiff_matrix):
    """
    Compute shear center from 6x6 stiffness matrix.

    Parameters
    ----------
    stiff_matrix : ndarray (6, 6)
        Beam stiffness matrix

    Returns
    -------
    shear_center : list [x2, x3]
        Shear center coordinates
    """
    stiff_matrix = np.asarray(stiff_matrix)
    K1 = stiff_matrix[:3, :3]
    K3 = stiff_matrix[:3, 3:]
    Y = np.linalg.solve(K1, -K3)
    return [-Y[1, 2], Y[0, 2]]


def compute_tension_center(stiff_matrix):
    """
    Compute tension center from 6x6 stiffness matrix.

    Parameters
    ----------
    stiff_matrix : ndarray (6, 6)
        Beam stiffness matrix

    Returns
    -------
    tension_center : list [x2, x3]
        Tension center coordinates
    """
    stiff_matrix = np.asarray(stiff_matrix)
    K1 = stiff_matrix[:3, :3]
    K3 = stiff_matrix[:3, 3:]
    Y = np.linalg.solve(K1, -K3)
    return [Y[2, 1], -Y[2, 0]]


def compute_mass_center(mass_matrix):
    """
    Compute mass center from 6x6 mass matrix.

    Parameters
    ----------
    mass_matrix : ndarray (6, 6)
        Beam mass matrix

    Returns
    -------
    mass_center : list [x2, x3]
        Mass center coordinates
    """
    mass_matrix = np.asarray(mass_matrix)
    M1 = mass_matrix[:3, :3]
    M3 = mass_matrix[:3, 3:]
    Y = np.linalg.solve(M1, -M3)
    return [Y[2, 1], -Y[2, 0]]


def decouple_stiffness(stiff_matrix):
    """
    Decouple the 6x6 stiffness matrix to remove off-diagonal coupling.

    Parameters
    ----------
    stiff_matrix : ndarray (6, 6)
        Beam stiffness matrix

    Returns
    -------
    decoupled : ndarray (6, 6)
        Decoupled stiffness matrix
    """
    K = np.asarray(stiff_matrix).copy()
    K1 = K[:3, :3]
    K3 = K[:3, 3:]
    Y = np.linalg.solve(K1, -K3)

    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))
    TL = np.block([[I3, Z3], [Y.T, I3]])
    TR = np.block([[I3, Y], [Z3, I3]])

    return TL @ K @ TR


def principal_axes_rotation_angle(decoupled_stiff_matrix):
    """
    Compute principal axes rotation angle from decoupled stiffness matrix.

    Parameters
    ----------
    decoupled_stiff_matrix : ndarray (6, 6)
        Decoupled beam stiffness matrix

    Returns
    -------
    angle : float
        Principal axes rotation angle in radians
    """
    K = np.asarray(decoupled_stiff_matrix)
    K3 = K[3:, 3:]
    w3, v3 = np.linalg.eig(K3)

    if np.abs(v3[0, 0]) < np.abs(v3[0, 1]):
        angle = np.arccos(v3[0, 0])
    else:
        angle = -np.arcsin(v3[0, 1])

    return angle
