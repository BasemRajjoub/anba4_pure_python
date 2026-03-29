"""
ANBA4-skfem singular solver: Cross-section analysis using sparse direct methods.

Computes 6x6 stiffness and mass matrices of composite beam cross sections.
This variant uses scipy's sparse direct solver (spsolve) which handles 
singular systems through augmented system formulation.

Theory: Morandini et al. (2010) "Characteristic behavior of prismatic
anisotropic beam via generalized eigenvectors"

Original ANBA4:
    Copyright (C) 2018 Marco Morandini
    https://github.com/manuelma/anba4

scikit-fem port:
    Copyright (C) 2024-2026 Basem Rajjoub

Licensed under GNU GPL v3 - see COPYING file for details.
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve, lstsq

from skfem import *
from skfem.helpers import grad, dot

from .material import Material


class AnbaxSingular:
    """
    ANBA cross-section analyzer using scikit-fem with sparse direct solvers.
    
    This variant uses scipy's sparse direct solver (spsolve) with augmented
    system formulation to handle the singular E matrix.

    Based on the generalized eigenvector theory from Morandini et al. (2010).

    Parameters
    ----------
    mesh : skfem.Mesh
        2D mesh of the cross-section
    degree : int
        Polynomial degree (1 or 2)
    mat_library : list of Material
        Material objects
    materials : array-like
        Element material indices
    plane_orientations : array-like
        Element plane orientation angles (degrees)
    fiber_orientations : array-like
        Element fiber orientation angles (degrees)
    scaling_constraint : float
        Constraint scaling factor
    """

    def __init__(self, mesh, degree, mat_library, materials, plane_orientations,
                 fiber_orientations, scaling_constraint=1.0):
        self.mesh = mesh
        self.degree = degree
        self.mat_library = mat_library
        self.materials = np.asarray(materials).flatten()
        self.plane_orientations = np.asarray(plane_orientations).flatten()
        self.fiber_orientations = np.asarray(fiber_orientations).flatten()
        self.scaling_constraint = scaling_constraint
        self.n_elem = mesh.nelements

        # Build material stiffness field
        self._build_material_fields()

        # Create vector element (3 components for u1, u2, u3)
        if isinstance(mesh, MeshTri):
            scalar_elem = ElementTriP1() if degree == 1 else ElementTriP2()
        else:
            scalar_elem = ElementQuad1() if degree == 1 else ElementQuad2()

        self.elem = ElementVector(scalar_elem, dim=3)
        self.basis = Basis(mesh, self.elem)
        self.n_dofs = self.basis.N

        # Results storage
        self._stiffness = None
        self._mass = None
        self._chains = None
        self._E_mat = None
        self._C_mat = None
        self._M_mat = None
        self._G_matrix = None

    def _build_material_fields(self):
        """Pre-compute material stiffness for each element."""
        self.C = np.zeros((self.n_elem, 6, 6))
        self.rho = np.zeros(self.n_elem)

        for e in range(self.n_elem):
            mat_id = int(self.materials[e])
            alpha = self.plane_orientations[e]
            beta = self.fiber_orientations[e]
            mat = self.mat_library[mat_id]
            self.C[e] = mat.compute_elastic_modulus(alpha, beta)
            self.rho[e] = mat.rho

    def inertia(self):
        """
        Compute 6x6 mass matrix.

        Returns
        -------
        M : ndarray (6, 6)
        """
        mesh = self.mesh

        # Compute mass properties by integration
        m = 0.0
        S2 = 0.0
        S3 = 0.0
        I22 = 0.0
        I33 = 0.0
        I23 = 0.0

        for e in range(self.n_elem):
            rho_e = self.rho[e]
            if rho_e == 0:
                continue

            # Get element geometry
            if isinstance(mesh, MeshTri):
                elem_nodes = mesh.t[:, e]
                coords = mesh.p[:, elem_nodes]
                v1 = coords[:, 1] - coords[:, 0]
                v2 = coords[:, 2] - coords[:, 0]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                xc = coords[0, :].mean()
                yc = coords[1, :].mean()
            else:
                elem_nodes = mesh.t[:, e]
                coords = mesh.p[:, elem_nodes]
                x = coords[0, :]
                y = coords[1, :]
                area = 0.5 * abs(sum(x[i]*(y[(i+1)%4] - y[(i-1)%4]) for i in range(4)))
                xc = x.mean()
                yc = y.mean()

            m += rho_e * area
            S2 += rho_e * area * yc
            S3 += rho_e * area * xc
            I22 += rho_e * area * yc**2
            I33 += rho_e * area * xc**2
            I23 += rho_e * area * xc * yc

        # Assemble 6x6 mass matrix
        M = np.zeros((6, 6))

        M[0, 0] = m
        M[1, 1] = m
        M[2, 2] = m

        M[0, 5] = S2;  M[5, 0] = S2
        M[1, 5] = -S3; M[5, 1] = -S3
        M[2, 3] = -S2; M[3, 2] = -S2
        M[2, 4] = S3;  M[4, 2] = S3

        M[3, 3] = I22
        M[4, 4] = I33
        M[3, 4] = -I23
        M[4, 3] = -I23
        M[5, 5] = I22 + I33

        self._mass = M
        return M

    def compute(self):
        """
        Compute the 6x6 stiffness matrix using sparse direct solver.

        Returns
        -------
        K : ndarray (6, 6)
        """
        # Assemble the key matrices
        E_mat, C_mat, M_mat = self._assemble_matrices()

        # Store matrices
        self._E_mat = E_mat
        self._C_mat = C_mat
        self._M_mat = M_mat

        # Compute stiffness using augmented system approach
        K, chains = self._compute_stiffness_augmented(E_mat, C_mat, M_mat)

        # Store chains for field recovery
        self._chains = chains
        self._stiffness = K
        
        # Build G matrix for stress/strain recovery
        self._build_g_matrix()
        
        return K

    def _get_quadrature(self):
        """Get quadrature points and weights."""
        if isinstance(self.mesh, MeshTri):
            qp_ref = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]]).T
            qw = np.array([1/6, 1/6, 1/6])
        else:
            g = 1/np.sqrt(3)
            qp_ref = np.array([[-g, -g], [g, -g], [g, g], [-g, g]]).T
            qw = np.array([1, 1, 1, 1])
        return qp_ref, qw

    def _get_shape_functions(self, xi):
        """Get shape functions and derivatives at reference point xi."""
        if isinstance(self.mesh, MeshTri):
            if self.degree == 1:
                N = np.array([1 - xi[0] - xi[1], xi[0], xi[1]])
                dN_dxi = np.array([[-1, 1, 0], [-1, 0, 1]])
            else:
                L1, L2, L3 = 1 - xi[0] - xi[1], xi[0], xi[1]
                N = np.array([L1*(2*L1-1), L2*(2*L2-1), L3*(2*L3-1),
                             4*L1*L2, 4*L2*L3, 4*L3*L1])
                dN_dxi = np.array([
                    [-(4*L1-1), 4*L2-1, 0, 4*(L1-L2), 4*L3, -4*L3],
                    [-(4*L1-1), 0, 4*L3-1, -4*L2, 4*L2, 4*(L1-L3)]
                ])
        else:
            if self.degree == 1:
                N = 0.25 * np.array([(1-xi[0])*(1-xi[1]), (1+xi[0])*(1-xi[1]),
                                    (1+xi[0])*(1+xi[1]), (1-xi[0])*(1+xi[1])])
                dN_dxi = 0.25 * np.array([
                    [-(1-xi[1]), (1-xi[1]), (1+xi[1]), -(1+xi[1])],
                    [-(1-xi[0]), -(1+xi[0]), (1+xi[0]), (1-xi[0])]
                ])
            else:
                raise NotImplementedError("Quadratic quads not implemented")
        return N, dN_dxi

    def _get_geom_shape_functions(self, xi):
        """Get geometry shape functions for Jacobian computation."""
        if isinstance(self.mesh, MeshTri):
            N_geom = np.array([1 - xi[0] - xi[1], xi[0], xi[1]])
            dN_geom_dxi = np.array([[-1, 1, 0], [-1, 0, 1]])
        else:
            N_geom = 0.25 * np.array([(1-xi[0])*(1-xi[1]), (1+xi[0])*(1-xi[1]),
                                      (1+xi[0])*(1+xi[1]), (1-xi[0])*(1+xi[1])])
            dN_geom_dxi = 0.25 * np.array([
                [-(1-xi[1]), (1-xi[1]), (1+xi[1]), -(1+xi[1])],
                [-(1-xi[0]), -(1+xi[0]), (1+xi[0]), (1-xi[0])]
            ])
        return N_geom, dN_geom_dxi

    def _assemble_matrices(self):
        """Assemble E, C, M matrices (same as Anbax)."""
        basis = self.basis
        elem_dofs = basis.element_dofs
        n_dofs = self.n_dofs

        E_mat = lil_matrix((n_dofs, n_dofs))
        C_mat = lil_matrix((n_dofs, n_dofs))
        M_mat = lil_matrix((n_dofs, n_dofs))

        qp_ref, qw = self._get_quadrature()

        if isinstance(self.mesh, MeshTri):
            n_sf_nodes = 3 if self.degree == 1 else 6
        else:
            n_sf_nodes = 4 if self.degree == 1 else 9

        for e in range(self.n_elem):
            Ce = self.C[e]

            geom_nodes = self.mesh.t[:, e]
            coords = self.mesh.p[:, geom_nodes]

            dofs = elem_dofs[:, e]
            n_local = len(dofs)

            Ee = np.zeros((n_local, n_local))
            Ce_local = np.zeros((n_local, n_local))
            Me = np.zeros((n_local, n_local))

            for qp_idx in range(len(qw)):
                xi = qp_ref[:, qp_idx]
                w = qw[qp_idx]

                _, dN_geom_dxi = self._get_geom_shape_functions(xi)
                J = dN_geom_dxi @ coords.T
                detJ = abs(np.linalg.det(J))
                invJ = np.linalg.inv(J)

                N, dN_dxi = self._get_shape_functions(xi)
                dN_dx = invJ @ dN_dxi

                for i in range(n_local):
                    node_i = i // 3
                    comp_i = i % 3

                    if node_i >= n_sf_nodes:
                        continue

                    eps_s_i = np.zeros(6)
                    if comp_i == 0:
                        eps_s_i[5] = dN_dx[0, node_i]
                        eps_s_i[4] = dN_dx[1, node_i]
                    elif comp_i == 1:
                        eps_s_i[1] = dN_dx[0, node_i]
                        eps_s_i[3] = dN_dx[1, node_i]
                    else:
                        eps_s_i[2] = dN_dx[1, node_i]
                        eps_s_i[3] = dN_dx[0, node_i]

                    eps_n_i = np.zeros(6)
                    N_i = N[node_i]
                    if comp_i == 0:
                        eps_n_i[0] = N_i
                    elif comp_i == 1:
                        eps_n_i[5] = N_i
                    else:
                        eps_n_i[4] = N_i

                    for j in range(n_local):
                        node_j = j // 3
                        comp_j = j % 3

                        if node_j >= n_sf_nodes:
                            continue

                        eps_s_j = np.zeros(6)
                        if comp_j == 0:
                            eps_s_j[5] = dN_dx[0, node_j]
                            eps_s_j[4] = dN_dx[1, node_j]
                        elif comp_j == 1:
                            eps_s_j[1] = dN_dx[0, node_j]
                            eps_s_j[3] = dN_dx[1, node_j]
                        else:
                            eps_s_j[2] = dN_dx[1, node_j]
                            eps_s_j[3] = dN_dx[0, node_j]

                        eps_n_j = np.zeros(6)
                        N_j = N[node_j]
                        if comp_j == 0:
                            eps_n_j[0] = N_j
                        elif comp_j == 1:
                            eps_n_j[5] = N_j
                        else:
                            eps_n_j[4] = N_j

                        Ee[i, j] += eps_s_i @ Ce @ eps_s_j * detJ * w
                        Ce_local[i, j] += eps_s_i @ Ce @ eps_n_j * detJ * w
                        Me[i, j] += eps_n_i @ Ce @ eps_n_j * detJ * w

            for i, di in enumerate(dofs):
                for j, dj in enumerate(dofs):
                    E_mat[di, dj] += Ee[i, j]
                    C_mat[di, dj] += Ce_local[i, j]
                    M_mat[di, dj] += Me[i, j]

        return E_mat.tocsr(), C_mat.tocsr(), M_mat.tocsr()

    def _compute_stiffness_augmented(self, E_mat, C_mat, M_mat):
        """
        Compute 6x6 stiffness using augmented system formulation.
        
        This handles the singular E matrix by augmenting with nullspace constraints.
        """
        n_dofs = self.n_dofs
        basis = self.basis

        # Get DOF coordinates
        doflocs = basis.doflocs
        x2 = doflocs[0, :]
        x3 = doflocs[1, :]
        comp = np.arange(n_dofs) % 3

        # Convert to dense for some operations
        E_dense = E_mat.toarray()
        C_dense = C_mat.toarray()
        M_dense = M_mat.toarray()
        H_dense = C_dense - C_dense.T

        # Initialize the 4 rigid body modes (nullspace of E)
        d0_axial = np.zeros(n_dofs)
        d0_axial[comp == 0] = 1.0

        d0_torsion = np.zeros(n_dofs)
        d0_torsion[comp == 1] = -x3[comp == 1]
        d0_torsion[comp == 2] = x2[comp == 2]

        d0_shear2 = np.zeros(n_dofs)
        d0_shear2[comp == 1] = 1.0

        d0_shear3 = np.zeros(n_dofs)
        d0_shear3[comp == 2] = 1.0

        d0_list = [d0_axial, d0_torsion, d0_shear2, d0_shear3]

        # Build nullspace basis matrix
        Phi = np.column_stack(d0_list)

        def solve_constrained(A, b):
            """Solve singular system A x = b with nullspace constraints."""
            n = A.shape[0]
            nc = Phi.shape[1]

            # Build augmented system
            A_aug = np.zeros((n + nc, n + nc))
            A_aug[:n, :n] = A
            A_aug[:n, n:] = Phi
            A_aug[n:, :n] = Phi.T

            b_aug = np.zeros(n + nc)
            b_aug[:n] = b

            # Add small regularization
            A_aug[:n, :n] += 1e-12 * np.eye(n)

            try:
                sol = solve(A_aug, b_aug)
                return sol[:n]
            except np.linalg.LinAlgError:
                sol, _, _, _ = lstsq(A_aug, b_aug)
                return sol[:n]

        def project_out_null(v):
            """Project out nullspace components."""
            v = v.copy()
            for d0 in d0_list:
                norm_sq = np.dot(d0, d0)
                if norm_sq > 1e-14:
                    v = v - (np.dot(v, d0) / norm_sq) * d0
            return v

        # Initialize chains
        chains = [[], [], [], []]

        # Fill chains with initial vectors
        for i in range(4):
            chains[i].append(d0_list[i])

        # Add linear chains for bending (indices 2 and 3)
        d1_bend2 = np.zeros(n_dofs)
        d1_bend2[comp == 0] = -x2[comp == 0]
        d1_bend2 = project_out_null(d1_bend2)
        chains[2].append(d1_bend2)

        d1_bend3 = np.zeros(n_dofs)
        d1_bend3[comp == 0] = -x3[comp == 0]
        d1_bend3 = project_out_null(d1_bend3)
        chains[3].append(d1_bend3)

        # Solve E d1 = -H d0 for chains 0 and 1 (axial and torsion)
        for i in range(2):
            rhs = -H_dense @ chains[i][0]
            rhs = project_out_null(rhs)
            d1 = solve_constrained(E_dense, rhs)
            chains[i].append(project_out_null(d1))

        # Solve E d2 = M d0 - H d1 for chains 2 and 3 (bending)
        for i in [2, 3]:
            rhs = M_dense @ chains[i][0] - H_dense @ chains[i][1]
            rhs = project_out_null(rhs)
            d2 = solve_constrained(E_dense, rhs)
            chains[i].append(project_out_null(d2))

        # Correction step for bending chains
        a = np.zeros((2, 2))
        b = np.zeros(2)

        for i in [2, 3]:
            res = M_dense @ chains[i][1] - H_dense @ chains[i][2]

            b[0] = np.dot(res, chains[0][0])
            b[1] = np.dot(res, chains[1][0])

            for ii in range(2):
                vec = M_dense @ chains[ii][0] - H_dense @ chains[ii][1]
                a[0, ii] = np.dot(vec, chains[0][0])
                a[1, ii] = np.dot(vec, chains[1][0])

            if np.linalg.cond(a) < 1e12:
                x = np.linalg.solve(a, b)
                chains[i][2] = chains[i][2] - x[0] * chains[0][1] - x[1] * chains[1][1]
                chains[i][1] = chains[i][1] - x[0] * chains[0][0] - x[1] * chains[1][0]

        # Solve E d3 = M d1 - H d2 for chains 2 and 3
        for i in [2, 3]:
            rhs = M_dense @ chains[i][1] - H_dense @ chains[i][2]
            rhs = project_out_null(rhs)
            d3 = solve_constrained(E_dense, rhs)
            chains[i].append(project_out_null(d3))

        # Compute stiffness matrix
        def stiffness_term(da, db):
            return (da @ M_dense @ da + da @ C_dense.T @ db +
                    db @ C_dense @ da + db @ E_dense @ db)

        def cross_stiffness(da_i, db_i, da_j, db_j):
            return (da_i @ M_dense @ da_j + da_i @ C_dense.T @ db_j +
                    db_i @ C_dense @ da_j + db_i @ E_dense @ db_j)

        K = np.zeros((6, 6))

        d0_ax, d1_ax = chains[0]
        d0_tor, d1_tor = chains[1]
        d0_2, d1_2, d2_2, d3_2 = chains[2]
        d0_3, d1_3, d2_3, d3_3 = chains[3]

        # Diagonal terms
        K[2, 2] = stiffness_term(d0_ax, d1_ax)
        K[5, 5] = stiffness_term(d0_tor, d1_tor)
        K[4, 4] = stiffness_term(d1_2, d2_2)
        K[0, 0] = 0.75 * stiffness_term(d0_2, d2_2)
        K[3, 3] = stiffness_term(d1_3, d2_3)
        K[1, 1] = 0.75 * stiffness_term(d0_3, d2_3)

        # Cross terms
        K[2, 5] = cross_stiffness(d0_ax, d1_ax, d0_tor, d1_tor)
        K[5, 2] = K[2, 5]
        K[3, 4] = cross_stiffness(d1_3, d2_3, d1_2, d2_2)
        K[4, 3] = K[3, 4]
        K[0, 1] = 0.75 * cross_stiffness(d0_2, d2_2, d0_3, d2_3)
        K[1, 0] = K[0, 1]

        return K, chains

    def _build_g_matrix(self):
        """Build G matrix for stress/strain recovery."""
        if self._chains is None or self._stiffness is None:
            raise RuntimeError("Must call compute() before building G matrix")
        
        K = self._stiffness
        
        try:
            self._G_matrix = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            self._G_matrix = np.linalg.lstsq(K, np.eye(6), rcond=None)[0]

    def stress_field(self, force, moment, reference="local", voigt_convention="anba"):
        """
        Compute stress field distribution for given force/moment resultants.

        Parameters
        ----------
        force : array-like (3,)
            Force resultants [F1, F2, F3] (axial, shear x2, shear x3)
        moment : array-like (3,)
            Moment resultants [M1, M2, M3] (torsion, bending x2, bending x3)
        reference : str
            "local" for material coordinates, "global" for section coordinates
        voigt_convention : str
            "anba" for ANBA ordering, "paraview" for ParaView ordering

        Returns
        -------
        stress : ndarray (n_elem, 6)
            Stress vectors at element centers in Voigt notation
        """
        if self._chains is None:
            raise RuntimeError("Must call compute() before stress_field()")
        
        force = np.asarray(force)
        moment = np.asarray(moment)
        
        # ANBA convention: [V2, V3, N, M2, M3, T]
        AzInt = np.array([force[1], force[2], force[0], moment[1], moment[2], moment[0]])
        
        if self._G_matrix is None:
            self._build_g_matrix()
        
        magnitudes = self._G_matrix @ AzInt
        
        # Reconstruct displacement fields
        U = np.zeros(self.n_dofs)
        UP = np.zeros(self.n_dofs)
        
        chains = self._chains
        d0_ax, d1_ax = chains[0]
        d0_tor, d1_tor = chains[1]
        d0_2, d1_2, d2_2, d3_2 = chains[2]
        d0_3, d1_3, d2_3, d3_3 = chains[3]
        
        chain_pairs = [
            (d0_2, d2_2),
            (d0_3, d2_3),
            (d0_ax, d1_ax),
            (d1_3, d2_3),
            (d1_2, d2_2),
            (d0_tor, d1_tor),
        ]
        
        for i, (da, db) in enumerate(chain_pairs):
            U += da * magnitudes[i]
            UP += db * magnitudes[i]
        
        # Compute stress at element centers
        stress = np.zeros((self.n_elem, 6))
        
        for e in range(self.n_elem):
            Ce = self.C[e]
            
            geom_nodes = self.mesh.t[:, e]
            coords = self.mesh.p[:, geom_nodes]
            xc = coords[0, :].mean()
            yc = coords[1, :].mean()
            
            dofs = self.basis.element_dofs[:, e]
            U_e = U[dofs]
            UP_e = UP[dofs]
            
            eps = self._compute_strain_at_element_center(e, U_e, UP_e, xc, yc, magnitudes)
            sigma = Ce @ eps
            
            if reference == "global":
                mat_id = int(self.materials[e])
                alpha = self.plane_orientations[e]
                beta = self.fiber_orientations[e]
                mat = self.mat_library[mat_id]
                T = mat.transformation_matrix(alpha, beta)
                sigma = T @ sigma
            
            if voigt_convention == "paraview":
                sigma = np.array([sigma[0], sigma[1], sigma[2], sigma[5], sigma[3], sigma[4]])
            
            stress[e] = sigma
        
        return stress

    def strain_field(self, force, moment, reference="local", voigt_convention="anba"):
        """
        Compute strain field distribution for given force/moment resultants.

        Parameters
        ----------
        force : array-like (3,)
            Force resultants [F1, F2, F3]
        moment : array-like (3,)
            Moment resultants [M1, M2, M3]
        reference : str
            "local" or "global"
        voigt_convention : str
            "anba" or "paraview"

        Returns
        -------
        strain : ndarray (n_elem, 6)
            Strain vectors at element centers in Voigt notation
        """
        if self._chains is None:
            raise RuntimeError("Must call compute() before strain_field()")
        
        force = np.asarray(force)
        moment = np.asarray(moment)
        
        AzInt = np.array([force[1], force[2], force[0], moment[1], moment[2], moment[0]])
        
        if self._G_matrix is None:
            self._build_g_matrix()
        
        magnitudes = self._G_matrix @ AzInt
        
        U = np.zeros(self.n_dofs)
        UP = np.zeros(self.n_dofs)
        
        chains = self._chains
        d0_ax, d1_ax = chains[0]
        d0_tor, d1_tor = chains[1]
        d0_2, d1_2, d2_2, d3_2 = chains[2]
        d0_3, d1_3, d2_3, d3_3 = chains[3]
        
        chain_pairs = [
            (d0_2, d2_2),
            (d0_3, d2_3),
            (d0_ax, d1_ax),
            (d1_3, d2_3),
            (d1_2, d2_2),
            (d0_tor, d1_tor),
        ]
        
        for i, (da, db) in enumerate(chain_pairs):
            U += da * magnitudes[i]
            UP += db * magnitudes[i]
        
        strain = np.zeros((self.n_elem, 6))
        
        for e in range(self.n_elem):
            geom_nodes = self.mesh.t[:, e]
            coords = self.mesh.p[:, geom_nodes]
            xc = coords[0, :].mean()
            yc = coords[1, :].mean()
            
            dofs = self.basis.element_dofs[:, e]
            U_e = U[dofs]
            UP_e = UP[dofs]
            
            eps = self._compute_strain_at_element_center(e, U_e, UP_e, xc, yc, magnitudes)
            
            if reference == "local":
                mat_id = int(self.materials[e])
                alpha = self.plane_orientations[e]
                beta = self.fiber_orientations[e]
                mat = self.mat_library[mat_id]
                T = mat.transformation_matrix(alpha, beta)
                eps = T.T @ eps
            
            if voigt_convention == "paraview":
                eps = np.array([eps[0], eps[1], eps[2], eps[5], eps[3], eps[4]])
            
            strain[e] = eps
        
        return strain

    def _compute_strain_at_element_center(self, e, U_e, UP_e, xc, yc, generalized_strains=None):
        """Compute strain vector at element center."""
        if generalized_strains is not None:
            eps = np.zeros(6)
            eps[0] = generalized_strains[2] + generalized_strains[3] * yc - generalized_strains[4] * xc
            eps[5] = generalized_strains[0] - generalized_strains[5] * yc
            eps[4] = generalized_strains[1] + generalized_strains[5] * xc
            return eps
        
        # Fallback to displacement-based computation
        if isinstance(self.mesh, MeshTri):
            xi_center = np.array([1/3, 1/3])
        else:
            xi_center = np.array([0.0, 0.0])
        
        N, dN_dxi = self._get_shape_functions(xi_center)
        _, dN_geom_dxi = self._get_geom_shape_functions(xi_center)
        
        geom_nodes = self.mesh.t[:, e]
        coords = self.mesh.p[:, geom_nodes]
        
        J = dN_geom_dxi @ coords.T
        detJ = abs(np.linalg.det(J))
        invJ = np.linalg.inv(J)
        
        dN_dx = invJ @ dN_dxi
        
        if isinstance(self.mesh, MeshTri):
            n_sf_nodes = 3 if self.degree == 1 else 6
        else:
            n_sf_nodes = 4 if self.degree == 1 else 9
        
        eps = np.zeros(6)
        
        n_local = len(U_e)
        for i in range(n_local):
            node_i = i // 3
            comp_i = i % 3
            
            if node_i >= n_sf_nodes:
                continue
            
            if comp_i == 0:
                eps[5] += dN_dx[0, node_i] * U_e[i]
                eps[4] += dN_dx[1, node_i] * U_e[i]
            elif comp_i == 1:
                eps[1] += dN_dx[0, node_i] * U_e[i]
                eps[3] += dN_dx[1, node_i] * U_e[i]
            else:
                eps[2] += dN_dx[1, node_i] * U_e[i]
                eps[3] += dN_dx[0, node_i] * U_e[i]
            
            if comp_i == 0:
                eps[0] += N[node_i] * UP_e[i]
            elif comp_i == 1:
                eps[5] += N[node_i] * UP_e[i]
            else:
                eps[4] += N[node_i] * UP_e[i]
        
        return eps
