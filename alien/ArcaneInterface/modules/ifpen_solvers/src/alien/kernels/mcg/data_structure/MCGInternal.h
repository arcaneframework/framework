// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ALIEN_MCGIMPL_MCGINTERNAL_H
#define ALIEN_MCGIMPL_MCGINTERNAL_H

//! Internal struct for MCG implementation
/*! Separate data from header;
 *  can be only included by LinearSystem and LinearSolver
 */

#include <chrono>
#if defined(__x86_64__) || defined(__amd64__)
#include <x86intrin.h>
#endif

#include <Common/index.h>
#include <MCGSolver/LinearSystem/LinearSystem.h>
#include <Precond/PrecondEquation.h>

#include <alien/kernels/mcg/MCGPrecomp.h>

BEGIN_MCGINTERNAL_NAMESPACE

//! Check parallel feature for MCG
inline void
checkParallel(bool)
{
  // This behaviour may be changed when Parallel MCG will be plugged
}

/*---------------------------------------------------------------------------*/
class UniqueKey
{
 public:
  UniqueKey()
  : m_rand(std::rand())
  ,
#if defined(__x86_64__) || defined(__amd64__)
  m_ts(_rdtsc())
#else
  m_ts(std::chrono::system_clock::now())
#endif

  {}

  bool operator==(const UniqueKey& k) const
  {
    return m_rand == k.m_rand && m_ts == k.m_ts;
  }

  bool operator!=(const UniqueKey& k) const { return !operator==(k); }

 private:
  int m_rand = 0;
#if defined(__x86_64__) || defined(__amd64__)
  uint64_t m_ts;
#else
  std::chrono::time_point<std::chrono::system_clock> m_ts;
#endif
};

class MatrixInternal
{
 public:
  using MatrixType = MCGSolver::BCSRMatrix<double,MCGSolver::Int32SparseIndex>;

  bool m_elliptic_split_tag = false;
  std::shared_ptr<MCGSolver::BVector<MCGSolver::Equation::eType>> m_equation_type;

  UniqueKey m_key;
  std::shared_ptr<MatrixType> m_matrix;

  std::vector<int> m_elem_perm;
};

class VectorInternal
{
 public:
  VectorInternal(int nrow, int block_size)
  : m_vector(new MCGSolver::BVector<double>(nrow, block_size))
  {}

  UniqueKey m_key;
  std::shared_ptr<MCGSolver::BVector<double>> m_vector;
};

END_MCGINTERNAL_NAMESPACE
