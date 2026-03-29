"""
Voigt notation utilities for ANBA4-skfem.

Stress and strain vector ordering (ANBA convention):
s11, s22, s33, s23, s13, s12

Original ANBA4:
    Copyright (C) 2018 Marco Morandini
    https://github.com/manuelma/anba4

scikit-fem port:
    Copyright (C) 2024-2026 Basem Rajjoub

Licensed under GNU GPL v3 - see COPYING file for details.
"""

import numpy as np


def stress_vector_to_tensor(sv):
    """
    Transform a contracted stress vector to stress tensor.

    Parameters
    ----------
    sv : array-like (6,) or (..., 6)
        Stress vector in order: s11, s22, s33, s23, s13, s12

    Returns
    -------
    st : ndarray (3, 3) or (..., 3, 3)
        Symmetric stress tensor
    """
    sv = np.asarray(sv)
    if sv.ndim == 1:
        return np.array([
            [sv[0], sv[5], sv[4]],
            [sv[5], sv[1], sv[3]],
            [sv[4], sv[3], sv[2]]
        ])
    else:
        # Batch case: sv has shape (..., 6)
        shape = sv.shape[:-1]
        st = np.zeros(shape + (3, 3))
        st[..., 0, 0] = sv[..., 0]
        st[..., 1, 1] = sv[..., 1]
        st[..., 2, 2] = sv[..., 2]
        st[..., 1, 2] = sv[..., 3]
        st[..., 2, 1] = sv[..., 3]
        st[..., 0, 2] = sv[..., 4]
        st[..., 2, 0] = sv[..., 4]
        st[..., 0, 1] = sv[..., 5]
        st[..., 1, 0] = sv[..., 5]
        return st


def stress_tensor_to_vector(st):
    """
    Transform a stress tensor to contracted stress vector.

    Parameters
    ----------
    st : ndarray (3, 3) or (..., 3, 3)
        Symmetric stress tensor

    Returns
    -------
    sv : ndarray (6,) or (..., 6)
        Stress vector in order: s11, s22, s33, s23, s13, s12
    """
    st = np.asarray(st)
    if st.ndim == 2:
        return np.array([st[0, 0], st[1, 1], st[2, 2], st[1, 2], st[0, 2], st[0, 1]])
    else:
        shape = st.shape[:-2]
        sv = np.zeros(shape + (6,))
        sv[..., 0] = st[..., 0, 0]
        sv[..., 1] = st[..., 1, 1]
        sv[..., 2] = st[..., 2, 2]
        sv[..., 3] = st[..., 1, 2]
        sv[..., 4] = st[..., 0, 2]
        sv[..., 5] = st[..., 0, 1]
        return sv


def stress_tensor_to_paraview_vector(st):
    """
    Transform a stress tensor to ParaView-compatible stress vector.

    Parameters
    ----------
    st : ndarray (3, 3) or (..., 3, 3)
        Symmetric stress tensor

    Returns
    -------
    sv : ndarray (6,) or (..., 6)
        Stress vector in ParaView order: s11, s22, s33, s12, s23, s13
    """
    st = np.asarray(st)
    if st.ndim == 2:
        return np.array([st[0, 0], st[1, 1], st[2, 2], st[0, 1], st[1, 2], st[0, 2]])
    else:
        shape = st.shape[:-2]
        sv = np.zeros(shape + (6,))
        sv[..., 0] = st[..., 0, 0]
        sv[..., 1] = st[..., 1, 1]
        sv[..., 2] = st[..., 2, 2]
        sv[..., 3] = st[..., 0, 1]
        sv[..., 4] = st[..., 1, 2]
        sv[..., 5] = st[..., 0, 2]
        return sv


def strain_vector_to_tensor(ev):
    """
    Transform an engineering strain vector to strain tensor.

    Parameters
    ----------
    ev : array-like (6,) or (..., 6)
        Engineering strain vector: e11, e22, e33, 2*e23, 2*e13, 2*e12

    Returns
    -------
    et : ndarray (3, 3) or (..., 3, 3)
        Symmetric strain tensor
    """
    ev = np.asarray(ev)
    if ev.ndim == 1:
        return np.array([
            [ev[0], 0.5*ev[5], 0.5*ev[4]],
            [0.5*ev[5], ev[1], 0.5*ev[3]],
            [0.5*ev[4], 0.5*ev[3], ev[2]]
        ])
    else:
        shape = ev.shape[:-1]
        et = np.zeros(shape + (3, 3))
        et[..., 0, 0] = ev[..., 0]
        et[..., 1, 1] = ev[..., 1]
        et[..., 2, 2] = ev[..., 2]
        et[..., 1, 2] = 0.5 * ev[..., 3]
        et[..., 2, 1] = 0.5 * ev[..., 3]
        et[..., 0, 2] = 0.5 * ev[..., 4]
        et[..., 2, 0] = 0.5 * ev[..., 4]
        et[..., 0, 1] = 0.5 * ev[..., 5]
        et[..., 1, 0] = 0.5 * ev[..., 5]
        return et


def strain_tensor_to_vector(et):
    """
    Transform a strain tensor to engineering strain vector.

    Parameters
    ----------
    et : ndarray (3, 3) or (..., 3, 3)
        Symmetric strain tensor

    Returns
    -------
    ev : ndarray (6,) or (..., 6)
        Engineering strain: e11, e22, e33, 2*e23, 2*e13, 2*e12
    """
    et = np.asarray(et)
    if et.ndim == 2:
        return np.array([et[0, 0], et[1, 1], et[2, 2],
                        2.0*et[1, 2], 2.0*et[0, 2], 2.0*et[0, 1]])
    else:
        shape = et.shape[:-2]
        ev = np.zeros(shape + (6,))
        ev[..., 0] = et[..., 0, 0]
        ev[..., 1] = et[..., 1, 1]
        ev[..., 2] = et[..., 2, 2]
        ev[..., 3] = 2.0 * et[..., 1, 2]
        ev[..., 4] = 2.0 * et[..., 0, 2]
        ev[..., 5] = 2.0 * et[..., 0, 1]
        return ev


def strain_tensor_to_paraview_vector(et):
    """
    Transform a strain tensor to ParaView-compatible engineering strain vector.

    Parameters
    ----------
    et : ndarray (3, 3) or (..., 3, 3)
        Symmetric strain tensor

    Returns
    -------
    ev : ndarray (6,) or (..., 6)
        Engineering strain in ParaView order: e11, e22, e33, 2*e12, 2*e23, 2*e13
    """
    et = np.asarray(et)
    if et.ndim == 2:
        return np.array([et[0, 0], et[1, 1], et[2, 2],
                        2.0*et[0, 1], 2.0*et[1, 2], 2.0*et[0, 2]])
    else:
        shape = et.shape[:-2]
        ev = np.zeros(shape + (6,))
        ev[..., 0] = et[..., 0, 0]
        ev[..., 1] = et[..., 1, 1]
        ev[..., 2] = et[..., 2, 2]
        ev[..., 3] = 2.0 * et[..., 0, 1]
        ev[..., 4] = 2.0 * et[..., 1, 2]
        ev[..., 5] = 2.0 * et[..., 0, 2]
        return ev
