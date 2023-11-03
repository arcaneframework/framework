﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
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

#include <BCSR/BCSRMatrix.h>
#include <MCGSolver/BVector.h>
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
  typedef MCGSolver::CSRProfile<int, int> ProfileType;
  typedef MCGSolver::BCSRMatrix<double,double,int,int> MatrixType;

  bool m_elliptic_split_tag = false;
  MCGSolver::BVector<MCGSolver::Equation::eType>* m_equation_type = nullptr;

  UniqueKey m_key;
  std::shared_ptr<MatrixType> m_matrix[2][2] = { { nullptr, nullptr },
    { nullptr, nullptr } };

  std::vector<int> m_elem_perm;

  MatrixInternal() {}

  ~MatrixInternal() { delete m_equation_type; }
};

class VectorInternal
{
 public:
  VectorInternal(int nrow, int block_size)
  : m_bvector(nrow, block_size)
  {}

  UniqueKey m_key;
  MCGSolver::BVector<double> m_bvector;
};

class CompositeVectorInternal
{
 public:
  CompositeVectorInternal(const std::vector<std::pair<int, int>>& composite_info)
  {
    m_bvector.reserve(composite_info.size());

    for (const auto& p : composite_info) {
      m_bvector.emplace_back(p.first, p.second);
    }
  }

  UniqueKey m_key;
  std::vector<MCGSolver::BVector<double>> m_bvector;
};

END_MCGINTERNAL_NAMESPACE
