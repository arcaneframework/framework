﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
#pragma once

//! Internal struct for MCG implementation
/*! Separate data from header;
 *  can be only included by LinearSystem and LinearSolver
 */

#include <Common/index.h>
#include <MCGSolver/LinearSystem/LinearSystem.h>
#include <MCGSolver/GPULinearSystem/GPULinearSystem.h>
#include <Precond/PrecondEquation.h>
#include <Solvers/SolverProperty.h>

#include <alien/kernels/mcg/MCGPrecomp.h>
#include <alien/kernels/mcg/data_structure/MemoryDomain.h>

BEGIN_MCGINTERNAL_NAMESPACE

/*---------------------------------------------------------------------------*/

template<typename NumT,eMemoryDomain Domain>
struct MatVecTypeGen
{};

template<typename NumT>
struct MatVecTypeGen<NumT,eMemoryDomain::CPU>
{
  using MatrixType = MCGSolver::BCSRMatrix<NumT,MCGSolver::Int32SparseIndex>;
  using VectorType = MCGSolver::BVector<NumT>;
};

template<typename NumT>
struct MatVecTypeGen<NumT,eMemoryDomain::GPU>
{
  using MatrixType = MCGSolver::BCSRgpuMatrix<NumT,MCGSolver::Int32SparseIndex>;
  using VectorType = MCGSolver::GPUBVector<NumT>;
};

template<typename NumT,eMemoryDomain Domain>
class MatrixInternal
{
 public:
  using MatrixType = typename MatVecTypeGen<NumT,Domain>::MatrixType;

  bool m_elliptic_split_tag = false;
  std::shared_ptr<MCGSolver::BVector<MCGSolver::Equation::eType>> m_equation_type;

  MCGSolver::UniqueKey m_key;
  std::shared_ptr<MatrixType> m_matrix;

  std::vector<int> m_elem_perm;
};

template<typename NumT,eMemoryDomain Domain>
class VectorInternal
{
 public:
  using VectorType = typename MatVecTypeGen<NumT,Domain>::VectorType;
  VectorInternal(int nrow, int block_size)
  : m_vector(new VectorType(nrow, block_size))
  {}

  MCGSolver::UniqueKey m_key;
  std::shared_ptr<VectorType> m_vector;
};

}
