"""
ANBA4-skfem: Cross-section beam analysis using scikit-fem.

A FEniCS-free port of ANBA4 using scikit-fem for FEM assembly
and scipy for linear algebra, enabling Windows compatibility.

Original ANBA4:
    Copyright (C) 2018 Marco Morandini
    https://github.com/manuelma/anba4

scikit-fem port:
    Copyright (C) 2024-2026 Basem Rajjoub

Licensed under GNU GPL v3 - see COPYING file for details.
"""

from .anbax import Anbax
from .anbax_singular import AnbaxSingular
from .material import Material, IsotropicMaterial, OrthotropicMaterial
from .voigt import (
    stress_vector_to_tensor, stress_tensor_to_vector,
    strain_vector_to_tensor, strain_tensor_to_vector,
    stress_tensor_to_paraview_vector, strain_tensor_to_paraview_vector
)
from .utils import (
    compute_shear_center, compute_tension_center, compute_mass_center,
    decouple_stiffness, principal_axes_rotation_angle
)

__all__ = [
    'Anbax', 'AnbaxSingular',
    'Material', 'IsotropicMaterial', 'OrthotropicMaterial',
    'stress_vector_to_tensor', 'stress_tensor_to_vector',
    'strain_vector_to_tensor', 'strain_tensor_to_vector',
    'stress_tensor_to_paraview_vector', 'strain_tensor_to_paraview_vector',
    'compute_shear_center', 'compute_tension_center', 'compute_mass_center',
    'decouple_stiffness', 'principal_axes_rotation_angle',
]

__version__ = '0.1.0'
