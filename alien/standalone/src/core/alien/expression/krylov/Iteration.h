// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#pragma once

#include <ostream>
#include <vector>

namespace Alien
{

template <typename AlgebraT>
class Iteration
{
 public:
  // clang-format off
  typedef AlgebraT                         AlgebraType;
  typedef typename AlgebraType::Matrix     MatrixType;
  typedef typename AlgebraType::Vector     VectorType;
  typedef typename MatrixType::ValueType   ValueType;
  typedef typename AlgebraType::FutureType FutureType;
  // clang-format on
  Iteration(AlgebraType& algebra,
            VectorType const& b,
            ValueType tol,
            int max_iter,
            ITraceMng* trace_mng = nullptr)
  : m_algebra(algebra)
  , m_max_iteration(max_iter)
  , m_tol(tol)
  , m_iter(0)
  , m_value(0)
  , m_f_value(m_value)
  , m_trace_mng(trace_mng)
  {
    m_algebra.dot(b, b, m_f_value);
    m_nrm2_b = m_f_value.get();
    if (m_trace_mng)
      m_trace_mng->info() << "STOP CRITERIA NORME B = " << m_nrm2_b;
    m_criteria_value = m_tol * m_tol * m_nrm2_b;
    m_sqrt_nrm2_b = std::sqrt(m_nrm2_b);
    m_value = m_criteria_value + 1;
    if (m_nrm2_b == 0)
      m_status = true;
    else
      m_status = false;
  }

  virtual ~Iteration()
  {}

  bool nullRhs() const
  {
    return m_nrm2_b == 0.;
  }

  bool first() const
  {
    return m_iter == 0;
  }

  bool stop(VectorType const& r)
  {
    if (m_iter >= m_max_iteration)
      return true;
    m_algebra.dot(r, r, m_f_value);
    m_status = m_f_value.get() < m_criteria_value;
    return m_status;
  }

  void operator++()
  {
    if (m_trace_mng) {
      m_trace_mng->info() << "================================";
      m_trace_mng->info() << "iteration (" << m_iter << ") criteria = " << getValue();
    }
    ++m_iter;
  }

  ValueType getValue() const
  {
    if (m_sqrt_nrm2_b == 0)
      return 0.;
    else
      return std::sqrt(m_value) / m_sqrt_nrm2_b;
  }

  int operator()() const
  {
    return m_iter;
  }

  bool getStatus() const
  {
    return m_status;
  }

 private:
  // clang-format off
  AlgebraType& m_algebra;
  int          m_max_iteration  = 0;
  ValueType    m_tol            = 0.;
  int          m_iter           = 0;
  ValueType    m_value          = 0.;
  FutureType   m_f_value;
  ValueType    m_criteria_value = 0.;
  ValueType    m_value_init     = 0.;
  ValueType    m_nrm2_b         = 0.;
  ValueType    m_sqrt_nrm2_b    = 0.;
  bool         m_status         = false;
  ITraceMng*   m_trace_mng      = nullptr;
  // clang-format on
};

} // namespace Alien
