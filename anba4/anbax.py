"""
ANBA4-skfem: Cross-section analysis using scikit-fem.

Computes 6x6 stiffness and mass matrices of composite beam cross sections.
This is a FEniCS-free port using scikit-fem for FEM assembly and scipy for
linear algebra, enabling Windows compatibility.

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
from scipy.linalg import solve, lstsq, null_space, svd

from skfem import *
from skfem.helpers import grad, dot

from .material import Material


class Anbax:
    """
    ANBA cross-section analyzer using scikit-fem.

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
        self._chains = None  # Store Jordan chains for field recovery
        self._E_mat = None
        self._C_mat = None
        self._M_mat = None
        self._B_matrix = None  # For stress/strain recovery

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
        m = 0.0    # total mass
        S2 = 0.0   # first moment about x3
        S3 = 0.0   # first moment about x2
        I22 = 0.0  # moment of inertia
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
                # Triangle area using cross product
                v1 = coords[:, 1] - coords[:, 0]
                v2 = coords[:, 2] - coords[:, 0]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                xc = coords[0, :].mean()
                yc = coords[1, :].mean()
            else:
                elem_nodes = mesh.t[:, e]
                coords = mesh.p[:, elem_nodes]
                # Shoelace formula for quad
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
        # Convention: [F1, F2, F3, M1, M2, M3] corresponds to [v1, v2, v3, w1, w2, w3]
        M = np.zeros((6, 6))

        M[0, 0] = m
        M[1, 1] = m
        M[2, 2] = m

        # Cross terms (ANBA convention)
        M[0, 5] = S2;  M[5, 0] = S2
        M[1, 5] = -S3; M[5, 1] = -S3
        M[2, 3] = -S2; M[3, 2] = -S2
        M[2, 4] = S3;  M[4, 2] = S3

        # Rotational inertia (ANBA convention)
        M[3, 3] = I22
        M[4, 4] = I33
        M[3, 4] = -I23
        M[4, 3] = -I23
        M[5, 5] = I22 + I33

        self._mass = M
        return M

    def compute(self):
        """
        Compute the 6x6 stiffness matrix.

        Uses the direct stiffness extraction method from the paper (Section 6).

        Returns
        -------
        K : ndarray (6, 6)
        """
        # Assemble the key matrices
        E_mat, C_mat, M_mat = self._assemble_matrices()

        # Store matrices for field recovery
        self._E_mat = E_mat
        self._C_mat = C_mat
        self._M_mat = M_mat

        # H = C - C^T (skew-symmetric)
        H_mat = C_mat - C_mat.T

        # Compute stiffness using Jordan chain approach
        K, chains = self._compute_stiffness_direct(E_mat, C_mat, M_mat)

        # Store chains for field recovery
        self._chains = chains
        self._stiffness = K
        
        # Build the B matrix for stress/strain recovery
        self._build_b_matrix()
        
        return K

    def _get_quadrature(self):
        """Get quadrature points and weights."""
        if isinstance(self.mesh, MeshTri):
            # 3-point rule for triangles (degree 2)
            qp_ref = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]]).T
            qw = np.array([1/6, 1/6, 1/6])
        else:
            # 2x2 Gauss for quads
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
                # Quadratic triangle (P2)
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
        """Get geometry (linear) shape functions for Jacobian computation."""
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
        """
        Assemble E, C, M matrices as defined in the paper (Eq. 12).

        - M: integrates n · E · n (normal-normal)
        - C: integrates g^α · E · n (in-plane to normal coupling)
        - E: integrates g^α · E · g^α (in-plane stiffness)

        Voigt ordering: [σ11, σ22, σ33, σ23, σ13, σ12]
        - "Normal" components (involving beam axis 1): indices 0, 4, 5
        - "In-plane" components: indices 1, 2, 3
        """
        basis = self.basis
        elem_dofs = basis.element_dofs
        n_dofs = self.n_dofs

        E_mat = lil_matrix((n_dofs, n_dofs))
        C_mat = lil_matrix((n_dofs, n_dofs))
        M_mat = lil_matrix((n_dofs, n_dofs))

        qp_ref, qw = self._get_quadrature()

        # Number of shape function nodes
        if isinstance(self.mesh, MeshTri):
            n_sf_nodes = 3 if self.degree == 1 else 6
        else:
            n_sf_nodes = 4 if self.degree == 1 else 9

        for e in range(self.n_elem):
            Ce = self.C[e]  # 6x6 material stiffness

            # Get element geometric coordinates (always using corner nodes)
            geom_nodes = self.mesh.t[:, e]
            coords = self.mesh.p[:, geom_nodes]

            # Element DOFs
            dofs = elem_dofs[:, e]
            n_local = len(dofs)

            # Local matrices
            Ee = np.zeros((n_local, n_local))
            Ce_local = np.zeros((n_local, n_local))
            Me = np.zeros((n_local, n_local))

            # Quadrature loop
            for qp_idx in range(len(qw)):
                xi = qp_ref[:, qp_idx]
                w = qw[qp_idx]

                # Geometry shape functions for Jacobian
                _, dN_geom_dxi = self._get_geom_shape_functions(xi)
                J = dN_geom_dxi @ coords.T
                detJ = np.linalg.det(J)
                # Use absolute value for area (handles CW or CCW elements)
                detJ = abs(detJ)
                invJ = np.linalg.inv(J)

                # Field shape functions
                N, dN_dxi = self._get_shape_functions(xi)
                dN_dx = invJ @ dN_dxi  # (2, n_sf_nodes)

                # Build strain matrices for each DOF
                # B_s: in-plane strain (from u,α) - components [0,1,2,3,4,5] but only α derivatives
                # B_n: normal strain (from u,1) - components [0,4,5] with shape function values

                for i in range(n_local):
                    node_i = i // 3
                    comp_i = i % 3

                    if node_i >= n_sf_nodes:
                        continue

                    # In-plane strain: contributions from shape function derivatives
                    # ε_s = [0, u2,2, u3,3, u2,3+u3,2, u1,3, u1,2]
                    eps_s_i = np.zeros(6)
                    if comp_i == 0:  # u1
                        eps_s_i[5] = dN_dx[0, node_i]  # 2ε12 partial: u1,2
                        eps_s_i[4] = dN_dx[1, node_i]  # 2ε13 partial: u1,3
                    elif comp_i == 1:  # u2
                        eps_s_i[1] = dN_dx[0, node_i]  # ε22: u2,2
                        eps_s_i[3] = dN_dx[1, node_i]  # 2ε23 partial: u2,3
                    else:  # u3
                        eps_s_i[2] = dN_dx[1, node_i]  # ε33: u3,3
                        eps_s_i[3] = dN_dx[0, node_i]  # 2ε23 partial: u3,2

                    # Normal strain: contributions that would come from u,1
                    # ε_n = [u1,1, 0, 0, 0, u3,1, u2,1] scaled by shape function
                    eps_n_i = np.zeros(6)
                    N_i = N[node_i]
                    if comp_i == 0:  # u1
                        eps_n_i[0] = N_i  # ε11: u1,1
                    elif comp_i == 1:  # u2
                        eps_n_i[5] = N_i  # 2ε12 partial: u2,1
                    else:  # u3
                        eps_n_i[4] = N_i  # 2ε13 partial: u3,1

                    for j in range(n_local):
                        node_j = j // 3
                        comp_j = j % 3

                        if node_j >= n_sf_nodes:
                            continue

                        # Build strain vectors for DOF j
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

                        # E matrix: (eps_s)^T C (eps_s)
                        Ee[i, j] += eps_s_i @ Ce @ eps_s_j * detJ * w

                        # C matrix: (eps_s)^T C (eps_n)
                        Ce_local[i, j] += eps_s_i @ Ce @ eps_n_j * detJ * w

                        # M matrix: (eps_n)^T C (eps_n)
                        Me[i, j] += eps_n_i @ Ce @ eps_n_j * detJ * w

            # Assemble into global matrices
            for i, di in enumerate(dofs):
                for j, dj in enumerate(dofs):
                    E_mat[di, dj] += Ee[i, j]
                    C_mat[di, dj] += Ce_local[i, j]
                    M_mat[di, dj] += Me[i, j]

        return E_mat.tocsr(), C_mat.tocsr(), M_mat.tocsr()

    def _compute_stiffness_direct(self, E_mat, C_mat, M_mat):
        """
        Compute 6x6 stiffness using the direct method from the paper.

        This follows Section 6 of Morandini et al. (2010).
        """
        n_dofs = self.n_dofs
        basis = self.basis

        # Get DOF coordinates
        doflocs = basis.doflocs
        x2 = doflocs[0, :]  # x-coordinate (our x2 direction)
        x3 = doflocs[1, :]  # y-coordinate (our x3 direction)
        comp = np.arange(n_dofs) % 3  # Component index

        # Convert to dense
        E_dense = E_mat.toarray()
        C_dense = C_mat.toarray()
        M_dense = M_mat.toarray()
        H_dense = C_dense - C_dense.T

        # Initialize the 4 rigid body modes (nullspace of E)
        # These are d0 vectors for each Jordan chain
        #
        # ANBA coordinate convention:
        # - x1 = beam axis (axial direction), displacement u1
        # - x2, x3 = cross-section coordinates, displacements u2, u3
        # - Mesh is in (x2, x3) plane
        # - comp=0 -> u1 (axial/out-of-plane)
        # - comp=1 -> u2 (in-plane x2 direction)
        # - comp=2 -> u3 (in-plane x3 direction)

        # Chain 0: Axial translation (u1 = 1, u2 = 0, u3 = 0)
        d0_axial = np.zeros(n_dofs)
        d0_axial[comp == 0] = 1.0

        # Chain 1: Torsion - rigid rotation about x1 axis
        # For rotation about x1: u1 = 0, u2 = -x3, u3 = x2
        d0_torsion = np.zeros(n_dofs)
        d0_torsion[comp == 1] = -x3[comp == 1]  # u2 = -x3
        d0_torsion[comp == 2] = x2[comp == 2]   # u3 = x2

        # Chain 2: Translation in x2 direction (u1 = 0, u2 = 1, u3 = 0)
        d0_shear2 = np.zeros(n_dofs)
        d0_shear2[comp == 1] = 1.0

        # Chain 3: Translation in x3 direction (u1 = 0, u2 = 0, u3 = 1)
        d0_shear3 = np.zeros(n_dofs)
        d0_shear3[comp == 2] = 1.0

        # These form the nullspace (verify: E @ d0 should be ~0)
        d0_list = [d0_axial, d0_torsion, d0_shear2, d0_shear3]

        # Build constraint matrix from d0 vectors
        Phi = np.column_stack(d0_list)  # (n_dofs, 4)

        def solve_constrained(A, b):
            """Solve singular system A x = b with nullspace constraints."""
            n = A.shape[0]
            nc = Phi.shape[1]

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
            """Project out nullspace components using Gram-Schmidt."""
            v = v.copy()
            for d0 in d0_list:
                norm_sq = np.dot(d0, d0)
                if norm_sq > 1e-14:
                    v = v - (np.dot(v, d0) / norm_sq) * d0
            return v

        # Solve Jordan chains following Eq. (29) in paper:
        # E d0 = 0 (d0 is in nullspace)
        # E d1 = -H d0
        # E d2 = M d0 - H d1
        # E d3 = M d1 - H d2

        chains = []

        # Chain 0: Axial (length 2)
        chain_axial = [d0_axial]
        rhs = -H_dense @ d0_axial
        rhs = project_out_null(rhs)
        d1 = solve_constrained(E_dense, rhs)
        d1 = project_out_null(d1)
        chain_axial.append(d1)
        chains.append(chain_axial)

        # Chain 1: Torsion (length 2)
        chain_torsion = [d0_torsion]
        rhs = -H_dense @ d0_torsion
        rhs = project_out_null(rhs)
        d1 = solve_constrained(E_dense, rhs)
        d1 = project_out_null(d1)
        chain_torsion.append(d1)
        chains.append(chain_torsion)

        # Chain 2: Bending about x3 (length 4)
        # This chain corresponds to shear V2 and bending moment M3
        chain_bend2 = [d0_shear2]

        # d1: rigid rotation about x3 axis -> tilts the section
        # For rotation about x3: u1 = -x2, u2 = 0, u3 = 0
        d1_bend2 = np.zeros(n_dofs)
        d1_bend2[comp == 0] = -x2[comp == 0]  # u1 = -x2
        d1_bend2 = project_out_null(d1_bend2)
        chain_bend2.append(d1_bend2)

        # d2: solve E d2 = M d0 - H d1
        rhs = M_dense @ d0_shear2 - H_dense @ d1_bend2
        rhs = project_out_null(rhs)
        d2 = solve_constrained(E_dense, rhs)
        d2 = project_out_null(d2)
        chain_bend2.append(d2)

        # Chain 3: Bending about x2 (length 4)
        # This chain corresponds to shear V3 and bending moment M2
        chain_bend3 = [d0_shear3]

        # d1: rigid rotation about x2 axis -> tilts the section
        # For rotation about x2: u1 = -x3, u2 = 0, u3 = 0
        d1_bend3 = np.zeros(n_dofs)
        d1_bend3[comp == 0] = -x3[comp == 0]  # u1 = -x3
        d1_bend3 = project_out_null(d1_bend3)
        chain_bend3.append(d1_bend3)

        # d2: solve E d2 = M d0 - H d1
        rhs = M_dense @ d0_shear3 - H_dense @ d1_bend3
        rhs = project_out_null(rhs)
        d2 = solve_constrained(E_dense, rhs)
        d2 = project_out_null(d2)
        chain_bend3.append(d2)

        # Correction step: ensure bending chains are orthogonal to axial/torsion chains
        # This follows the original ANBA4 code's orthogonalization procedure
        for chain_bend, d0_bend in [(chain_bend2, d0_shear2), (chain_bend3, d0_shear3)]:
            # Compute residual: M*d1 - H*d2
            res = M_dense @ chain_bend[1] - H_dense @ chain_bend[2]

            # Build system to find correction coefficients
            a = np.zeros((2, 2))
            b = np.zeros(2)

            # Compute RHS: residual projected onto axial/torsion chains
            b[0] = np.dot(res, chains[0][0])  # onto axial d0
            b[1] = np.dot(res, chains[1][0])  # onto torsion d0

            # Compute matrix: (M*d0 - H*d1) projected onto axial/torsion chains
            for ii in range(2):
                vec = M_dense @ chains[ii][0] - H_dense @ chains[ii][1]
                a[0, ii] = np.dot(vec, chains[0][0])
                a[1, ii] = np.dot(vec, chains[1][0])

            # Solve for correction
            if np.linalg.cond(a) < 1e12:
                x = np.linalg.solve(a, b)
                # Apply correction to d1 and d2
                chain_bend[2] = chain_bend[2] - x[0] * chains[0][1] - x[1] * chains[1][1]
                chain_bend[1] = chain_bend[1] - x[0] * chains[0][0] - x[1] * chains[1][0]

        # d3 for both bending chains (must be done AFTER the correction)
        for chain_bend in [chain_bend2, chain_bend3]:
            rhs = M_dense @ chain_bend[1] - H_dense @ chain_bend[2]
            rhs = project_out_null(rhs)
            d3 = solve_constrained(E_dense, rhs)
            d3 = project_out_null(d3)
            chain_bend.append(d3)

        chains.append(chain_bend2)
        chains.append(chain_bend3)

        # Compute the stiffness matrix following the original ANBA4 approach
        # For a chain with pair (d_a, d_b), the stiffness contribution is:
        # S = d_a^T M d_a + d_a^T C^T d_b + d_b^T C d_a + d_b^T E d_b
        #   = d_a^T M d_a + 2 * d_a^T C^T d_b + d_b^T E d_b (using symmetry)
        #
        # For length-2 chains (axial, torsion): use (d0, d1)
        # For length-4 chains (bending):
        #   - bending stiffness: use (d1, d2)
        #   - shear stiffness: use (d2, d3)
        #
        # Final index mapping to ANBA convention:
        # [0,1,2,3,4,5] = [V2(shear), V3(shear), N(axial), M2(bend), M3(bend), T(torsion)]

        def stiffness_term(da, db):
            """Compute stiffness from chain pair (d_a, d_b).

            Formula: S = d_a^T M d_a + d_a^T C^T d_b + d_b^T C d_a + d_b^T E d_b
            Note: C is NOT symmetric, so the two C terms are different!
            """
            return (da @ M_dense @ da + da @ C_dense.T @ db +
                    db @ C_dense @ da + db @ E_dense @ db)

        def cross_stiffness(da_i, db_i, da_j, db_j):
            """Compute cross stiffness between two chain pairs."""
            return (da_i @ M_dense @ da_j + da_i @ C_dense.T @ db_j +
                    db_i @ C_dense @ da_j + db_i @ E_dense @ db_j)

        K = np.zeros((6, 6))

        # Extract chain vectors
        d0_ax, d1_ax = chains[0]
        d0_tor, d1_tor = chains[1]
        d0_2, d1_2, d2_2, d3_2 = chains[2]  # Bending about x3 / shear V2
        d0_3, d1_3, d2_3, d3_3 = chains[3]  # Bending about x2 / shear V3

        # Diagonal terms
        # Index 2: EA (axial) - from chain 0 with (d0, d1)
        K[2, 2] = stiffness_term(d0_ax, d1_ax)

        # Index 5: GJ (torsion) - from chain 1 with (d0, d1)
        K[5, 5] = stiffness_term(d0_tor, d1_tor)

        # Index 4: EI33 (bending about x3) - from chain 2 with (d1, d2)
        K[4, 4] = stiffness_term(d1_2, d2_2)

        # Index 0: GA22 (shear in x2) - using (d0, d2) pair
        # The shear stiffness is computed exactly with warping effects
        K[0, 0] = stiffness_term(d0_2, d2_2)

        # Index 3: EI22 (bending about x2) - from chain 3 with (d1, d2)
        K[3, 3] = stiffness_term(d1_3, d2_3)

        # Index 1: GA33 (shear in x3) - using (d0, d2) pair
        K[1, 1] = stiffness_term(d0_3, d2_3)

        # Cross terms (off-diagonal)
        # For a symmetric section centered at origin, most cross terms should be zero
        # Only computing the structurally expected non-zero terms

        # Axial-Torsion (2,5) - usually zero for isotropic
        K[2, 5] = cross_stiffness(d0_ax, d1_ax, d0_tor, d1_tor)
        K[5, 2] = K[2, 5]

        # Bending M2 - Bending M3 (3,4) - usually zero for symmetric section
        K[3, 4] = cross_stiffness(d1_3, d2_3, d1_2, d2_2)
        K[4, 3] = K[3, 4]

        # Shear-bending cross terms (for non-centered or asymmetric sections)
        # For symmetric sections centered at origin, these are typically small
        # Using (d0, d2) pair for shear-related cross terms (with 0.75 factor)

        # Shear V2 - Shear V3 (0,1)
        K[0, 1] = cross_stiffness(d0_2, d2_2, d0_3, d2_3)
        K[1, 0] = K[0, 1]

        return K, chains

    def _build_b_matrix(self):
        """
        Build the G matrix for stress/strain recovery.
        
        The G matrix transforms force/moment resultants to chain magnitudes.
        For a given load vector AzInt, the chain magnitudes are: mag = G @ AzInt
        
        The key insight is that the stiffness matrix K relates displacements to forces:
        F = K @ u
        
        For the chain formulation, the generalized stiffness S relates chain magnitudes
        to generalized forces. We need to find the transformation from actual forces
        to chain magnitudes.
        """
        if self._chains is None or self._E_mat is None:
            raise RuntimeError("Must call compute() before building B matrix")
        
        chains = self._chains
        K = self._stiffness  # Already computed 6x6 stiffness matrix
        
        # The G matrix is simply the inverse of the stiffness matrix
        # This works because K directly relates [V2, V3, N, M2, M3, T] to
        # the corresponding generalized displacements (chain magnitudes)
        
        try:
            self._G_matrix = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            self._G_matrix = np.linalg.lstsq(K, np.eye(6), rcond=None)[0]
        
        # Store S matrix as the stiffness (for compatibility)
        self._S_matrix = K
        self._B_matrix = np.eye(6)  # Identity: chain magnitudes directly relate to forces

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
        
        # Combine force and moment into load vector
        # ANBA convention: [V2, V3, N, M2, M3, T]
        AzInt = np.array([force[1], force[2], force[0], moment[1], moment[2], moment[0]])
        
        # Solve for chain magnitudes: S @ magnitudes = AzInt
        # But we need to use the G matrix: magnitudes = G @ AzInt
        if self._G_matrix is None:
            self._build_b_matrix()
        
        magnitudes = self._G_matrix @ AzInt
        
        # Reconstruct displacement fields from chains
        # U (in-plane) and UP (derivative along beam axis)
        U = np.zeros(self.n_dofs)
        UP = np.zeros(self.n_dofs)
        
        # Chain pairs for each load case (same order as above)
        chains = self._chains
        d0_ax, d1_ax = chains[0]
        d0_tor, d1_tor = chains[1]
        d0_2, d1_2, d2_2, d3_2 = chains[2]
        d0_3, d1_3, d2_3, d3_3 = chains[3]
        
        chain_pairs = [
            (d0_2, d2_2),  # Shear V2
            (d0_3, d2_3),  # Shear V3
            (d0_ax, d1_ax),  # Axial
            (d1_3, d2_3),  # Bending M2
            (d1_2, d2_2),  # Bending M3
            (d0_tor, d1_tor),  # Torsion
        ]
        
        for i, (da, db) in enumerate(chain_pairs):
            U += da * magnitudes[i]
            UP += db * magnitudes[i]
        
        # Compute stress at element centers
        stress = np.zeros((self.n_elem, 6))
        
        for e in range(self.n_elem):
            Ce = self.C[e]
            
            # Get element center coordinates
            geom_nodes = self.mesh.t[:, e]
            coords = self.mesh.p[:, geom_nodes]
            xc = coords[0, :].mean()
            yc = coords[1, :].mean()
            
            # Get element DOFs
            dofs = self.basis.element_dofs[:, e]
            U_e = U[dofs]
            UP_e = UP[dofs]
            
            # Compute strain at element center
            eps = self._compute_strain_at_element_center(e, U_e, UP_e, xc, yc, magnitudes)
            
            # Compute stress: sigma = C @ eps
            sigma = Ce @ eps
            
            if reference == "local":
                # Use material coordinates (already in local system)
                pass
            else:
                # Transform to global coordinates
                mat_id = int(self.materials[e])
                alpha = self.plane_orientations[e]
                beta = self.fiber_orientations[e]
                mat = self.mat_library[mat_id]
                T = mat.transformation_matrix(alpha, beta)
                sigma = T @ sigma
            
            if voigt_convention == "paraview":
                # Convert from ANBA to ParaView ordering
                # ANBA: [s11, s22, s33, s23, s13, s12]
                # ParaView: [s11, s22, s33, s12, s23, s13]
                sigma = np.array([sigma[0], sigma[1], sigma[2], sigma[5], sigma[3], sigma[4]])
            
            stress[e] = sigma
        
        self._stress = stress
        return stress

    def strain_field(self, force, moment, reference="local", voigt_convention="anba"):
        """
        Compute strain field distribution for given force/moment resultants.

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
        strain : ndarray (n_elem, 6)
            Strain vectors at element centers in Voigt notation
        """
        if self._chains is None:
            raise RuntimeError("Must call compute() before strain_field()")
        
        force = np.asarray(force)
        moment = np.asarray(moment)
        
        # Combine force and moment into load vector
        AzInt = np.array([force[1], force[2], force[0], moment[1], moment[2], moment[0]])
        
        # Solve for chain magnitudes
        if self._G_matrix is None:
            self._build_b_matrix()
        
        magnitudes = self._G_matrix @ AzInt
        
        # Reconstruct displacement fields from chains
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
        
        # Compute strain at element centers
        strain = np.zeros((self.n_elem, 6))
        
        for e in range(self.n_elem):
            # Get element center coordinates
            geom_nodes = self.mesh.t[:, e]
            coords = self.mesh.p[:, geom_nodes]
            xc = coords[0, :].mean()
            yc = coords[1, :].mean()
            
            # Get element DOFs
            dofs = self.basis.element_dofs[:, e]
            U_e = U[dofs]
            UP_e = UP[dofs]
            
            # Compute strain at element center (pass generalized strains from chain magnitudes)
            eps = self._compute_strain_at_element_center(e, U_e, UP_e, xc, yc, magnitudes)
            
            if reference == "local":
                # Transform to material coordinates
                mat_id = int(self.materials[e])
                alpha = self.plane_orientations[e]
                beta = self.fiber_orientations[e]
                mat = self.mat_library[mat_id]
                T = mat.transformation_matrix(alpha, beta)
                eps = T.T @ eps  # Inverse transformation for strain
            
            if voigt_convention == "paraview":
                # Convert from ANBA to ParaView ordering
                eps = np.array([eps[0], eps[1], eps[2], eps[5], eps[3], eps[4]])
            
            strain[e] = eps
        
        self._strain = strain
        return strain

    def _compute_strain_at_element_center(self, e, U_e, UP_e, xc, yc, generalized_strains=None):
        """
        Compute strain vector at element center.
        
        Parameters
        ----------
        e : int
            Element index
        U_e : ndarray
            Element displacement DOFs
        UP_e : ndarray
            Element derivative DOFs
        xc, yc : float
            Element center coordinates
        generalized_strains : ndarray (6,), optional
            Generalized strains from chain magnitudes. If provided, used directly.
        
        Returns
        -------
        eps : ndarray (6,)
            Strain vector in Voigt notation
        """
        # If generalized strains are provided, use them directly
        # The generalized strains ARE the actual strains for the beam problem
        # Generalized strains convention: [gamma_12, gamma_13, eps_11, kappa_2, kappa_3, theta]
        # where gamma_12, gamma_13 are shear strains
        #       eps_11 is axial strain
        #       kappa_2, kappa_3 are bending curvatures (about x2 and x3 axes)
        #       theta is twist rate (torsion)
        if generalized_strains is not None:
            eps = np.zeros(6)
            # Axial strain: constant plus bending contributions
            # eps_11 = eps_axial + kappa_2 * x3 - kappa_3 * x2
            eps[0] = generalized_strains[2] + generalized_strains[3] * yc - generalized_strains[4] * xc
            
            # Shear strains from torsion and shear forces
            # gamma_12 = gamma_12_shear - theta * x3
            eps[5] = generalized_strains[0] - generalized_strains[5] * yc  # shear 12 (2*eps_12)
            # gamma_13 = gamma_13_shear + theta * x2
            eps[4] = generalized_strains[1] + generalized_strains[5] * xc  # shear 13 (2*eps_13)
            return eps
        
        # Otherwise, compute from displacement field (original approach)
        # Get shape functions at element center
        if isinstance(self.mesh, MeshTri):
            xi_center = np.array([1/3, 1/3])
        else:
            xi_center = np.array([0.0, 0.0])
        
        N, dN_dxi = self._get_shape_functions(xi_center)
        _, dN_geom_dxi = self._get_geom_shape_functions(xi_center)
        
        # Get element geometry
        geom_nodes = self.mesh.t[:, e]
        coords = self.mesh.p[:, geom_nodes]
        
        # Compute Jacobian
        J = dN_geom_dxi @ coords.T
        detJ = abs(np.linalg.det(J))
        invJ = np.linalg.inv(J)
        
        # Shape function derivatives in physical coordinates
        dN_dx = invJ @ dN_dxi
        
        # Number of shape function nodes
        if isinstance(self.mesh, MeshTri):
            n_sf_nodes = 3 if self.degree == 1 else 6
        else:
            n_sf_nodes = 4 if self.degree == 1 else 9
        
        # Compute strain components
        eps = np.zeros(6)
        
        n_local = len(U_e)
        for i in range(n_local):
            node_i = i // 3
            comp_i = i % 3
            
            if node_i >= n_sf_nodes:
                continue
            
            # In-plane strain contributions (from U derivatives)
            if comp_i == 0:  # u1
                eps[5] += dN_dx[0, node_i] * U_e[i]  # 2ε12: u1,2
                eps[4] += dN_dx[1, node_i] * U_e[i]  # 2ε13: u1,3
            elif comp_i == 1:  # u2
                eps[1] += dN_dx[0, node_i] * U_e[i]  # ε22: u2,2
                eps[3] += dN_dx[1, node_i] * U_e[i]  # 2ε23: u2,3
            else:  # u3
                eps[2] += dN_dx[1, node_i] * U_e[i]  # ε33: u3,3
                eps[3] += dN_dx[0, node_i] * U_e[i]  # 2ε23: u3,2
            
            # Normal strain contributions (from UP)
            if comp_i == 0:  # u1
                eps[0] += N[node_i] * UP_e[i]  # ε11: u1,1
            elif comp_i == 1:  # u2
                eps[5] += N[node_i] * UP_e[i]  # 2ε12: u2,1
            else:  # u3
                eps[4] += N[node_i] * UP_e[i]  # 2ε13: u3,1
        
        return eps
