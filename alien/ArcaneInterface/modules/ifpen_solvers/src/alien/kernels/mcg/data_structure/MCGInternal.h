// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
#pragma once

//! Internal struct for MCG implementation
/*! Separate data from header;
 *  can be only included by LinearSystem and LinearSolver
 */

#include <chrono>

#if defined(__x86_64__) || defined(__amd64__)

#include <x86intrin.h>

#endif

#include "Common/index.h"
#include "Common/Utils/UniqueKey.h"
#include "BCSR/BCSRMatrix.h"
#include "BCSRgpu/BCSRgpuMatrix.h"
#include "MCGSolver/LinearSystem/BVector.h"
#include "MCGSolver/GPULinearSystem/GPUBVector.h"
#include "Precond/PrecondEquation.h"

#include <alien/kernels/mcg/MCGPrecomp.h>

BEGIN_MCGINTERNAL_NAMESPACE


    class MatrixInternal {
    public:
        typedef MCGSolver::CSRProfile<MCGSolver::Int32SparseIndex> ProfileType;
        typedef MCGSolver::BCSRMatrix<double, MCGSolver::Int32SparseIndex> MatrixType;

        bool m_elliptic_split_tag = false;
        std::shared_ptr<MCGSolver::BVector<MCGSolver::Equation::eType>> m_equation_type;

        MCGSolver::UniqueKey m_key;
        std::shared_ptr<MatrixType> m_matrix;

        std::vector<int> m_elem_perm;

        MatrixInternal() = default;

        ~MatrixInternal() = default;
    };

    class VectorInternal {
    public:
        VectorInternal(int nrow, int block_size)
                : m_bvector(std::make_shared<MCGSolver::BVector<double>>(nrow, block_size)) {}

        MCGSolver::UniqueKey m_key;
        std::shared_ptr<MCGSolver::BVector<double>> m_bvector;
    };

    class GpuMatrixInternal {
    public:
        typedef MCGSolver::CSRgpuProfile<MCGSolver::Int32SparseIndex> ProfileType;
        typedef MCGSolver::BCSRgpuMatrix<double, MCGSolver::Int32SparseIndex> MatrixType;

        bool m_elliptic_split_tag = false;
        std::shared_ptr<MCGSolver::BVector<MCGSolver::Equation::eType>> m_equation_type;

        MCGSolver::UniqueKey m_key;
        std::shared_ptr<MatrixType> m_matrix;

        std::vector<int> m_elem_perm;

        GpuMatrixInternal() = default;

        ~GpuMatrixInternal() = default;
    };

    class GpuVectorInternal {
    public:
        GpuVectorInternal(int nrow, int block_size)
                : m_bvector(std::make_shared<MCGSolver::GPUBVector<double>>(nrow, block_size)) {}

        MCGSolver::UniqueKey m_key;
        std::shared_ptr<MCGSolver::GPUBVector<double>> m_bvector;
    };

END_MCGINTERNAL_NAMESPACE
