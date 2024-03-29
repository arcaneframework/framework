/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * bicgs.h
 *
 *  Created on: Dec 1, 2021
 *      Author: gratienj
 */

#pragma once

#include <ostream>
#include <vector>

namespace Alien
{

template <typename AlgebraT>
class BiCGStab
{
 public:
  // clang-format off
  typedef AlgebraT                         AlgebraType;
  typedef typename AlgebraType::Matrix     MatrixType;
  typedef typename AlgebraType::Vector     VectorType;
  typedef typename MatrixType::ValueType   ValueType;
  typedef typename AlgebraType::FutureType FutureType;
  // clang-format on

  class Iteration
  {
   public:
    Iteration(AlgebraType& algebra,
              VectorType const& b,
              ValueType tol,
              int max_iter,
              ITraceMng* trace_mng = nullptr)
    : m_algebra(algebra)
    , m_value(0)
    , m_f_value(m_value)
    , m_max_iteration(max_iter)
    , m_tol(tol)
    , m_iter(0)
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
      if (m_trace_mng)
        m_trace_mng->info() << "iteration (" << m_iter << ") criteria = " << getValue();
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

  BiCGStab(AlgebraType& algebra, ITraceMng* trace_mng = nullptr)
  : m_algebra(algebra)
  , m_trace_mng(trace_mng)
  {}

  virtual ~BiCGStab()
  {}

  void setOutputLevel(int level)
  {
    m_output_level = level;
  }

  template <typename PrecondT, typename iterT>
  int solve(PrecondT& precond, iterT& iter, MatrixType const& A,
            VectorType const& b, VectorType& x)
  {
    if (iter.nullRhs())
      return 0;
    ValueType rho(0), rho1(0), alpha(0), beta(0), omega(0);
    VectorType p, phat, s, shat, t, v, r, r0;

    m_algebra.allocate(AlgebraType::resource(A), p, phat, s, shat, t, v, r, r0);

    // SEQ0
    //  r = b - A * x;
    m_algebra.copy(b, r);
    m_algebra.mult(A, x, r0);
    m_algebra.axpy(-1., r0, r);

    // rtilde = r
    m_algebra.copy(r, r0);
    m_algebra.copy(r, p);
    rho1 = m_algebra.dot(r, r0);
    if (m_output_level > 1)
      _print(0, "Seq 0", "rho1", rho1);

    /*
       phat = solve(M, p);
       v = A * phat;
       gamma = dot(r0, v);
       alpha = rho_1 / gamma;
       s = r - alpha * v;
     */
    // SEQ1
    m_algebra.exec(precond, p, phat);
    m_algebra.mult(A, phat, v);
    alpha = m_algebra.dot(v, r0);
    if (alpha == 0)
      throw typename AlgebraType::NullValueException("alpha");
    alpha = rho1 / alpha;
    m_algebra.copy(r, s);
    m_algebra.axpy(-alpha, v, s);
    if (m_output_level > 1)
      _print(0, "Seq 1", "alpha", alpha);

    if (iter.stop(s)) {
      ++iter;
      m_algebra.axpy(alpha, phat, x);
      m_algebra.free(p, phat, s, shat, t, v, r, r0);
      return 0;
    }

    // SEQ 2
    m_algebra.exec(precond, s, shat);
    m_algebra.mult(A, shat, t);
    omega = m_algebra.dot(t, s);
    beta = m_algebra.dot(t, t);

    if (beta == 0) {
      if (iter.stop(r)) {
        ++iter;
        m_algebra.axpy(alpha, phat, x);
        m_algebra.free(p, phat, s, shat, t, v, r, r0);
        return 0;
      }
      else
        throw typename AlgebraType::NullValueException("beta");
    }
    omega = omega / beta;
    if (m_output_level > 1)
      _print(iter(), "Seq 2", "beta", beta, "alpha", alpha, "rho1", rho1);

    // SEQ 3
    m_algebra.axpy(omega, shat, x);
    m_algebra.axpy(alpha, phat, x);
    m_algebra.copy(s, r);
    m_algebra.axpy(-omega, t, r);

    rho = rho1;
    ++iter;
    if (m_output_level > 1)
      _print(iter(), "Seq 3", "beta", beta, "alpha", alpha, "rho1", rho1);

    while (!iter.stop(r)) {
      //SEQ4
      rho1 = m_algebra.dot(r, r0);
      beta = (rho1 / rho) * (alpha / omega);
      m_algebra.axpy(-omega, v, p);
      m_algebra.scal(beta, p);
      m_algebra.axpy(1., r, p);
      if (m_output_level > 1)
        _print(iter(), "Seq 4", "beta", beta, "alpha", alpha, "rho1", rho1);

      if (rho == 0)
        throw typename AlgebraType::NullValueException("rho");

      //m_algebra.process (seq1);
      m_algebra.exec(precond, p, phat);
      m_algebra.mult(A, phat, v);
      alpha = m_algebra.dot(v, r0);
      if (alpha == 0)
        throw typename AlgebraType::NullValueException("alpha");
      else
        alpha = rho1 / alpha;

      m_algebra.copy(r, s);
      m_algebra.axpy(-alpha, v, s);

      if (m_output_level > 1)
        _print(iter(), "Seq 1", "alpha", alpha);
      if (iter.stop(s)) {
        m_algebra.axpy(alpha, phat, x);
        m_algebra.free(p, phat, s, shat, t, v, r, r0);
        return 0;
      }

      //m_algebra.process (seq2);
      m_algebra.exec(precond, s, shat);
      m_algebra.mult(A, shat, t);
      omega = m_algebra.dot(t, s);
      beta = m_algebra.dot(t, t);

      if (m_output_level > 1)
        _print(iter(), "Seq 2", "beta", beta, "alpha", alpha, "rho1", rho1, "omega", omega);
      if (beta == 0) {
        if (iter.stop(s)) {
          m_algebra.axpy(alpha, phat, x);
          m_algebra.free(p, phat, s, shat, t, v, r, r0);
          return 0;
        }
        throw typename AlgebraType::NullValueException("beta");
      }
      else
        omega = omega / beta;

      //m_algebra.process (seq3);
      m_algebra.axpy(omega, shat, x);
      m_algebra.axpy(alpha, phat, x);
      m_algebra.copy(s, r);
      m_algebra.axpy(-omega, t, r);

      rho = rho1;

      ++iter;
      if (m_output_level > 1)
        _print(iter(), "end loop", "beta", beta, "alpha", alpha, "rho", rho);
    }

    m_algebra.free(p, phat, s, shat, t, v, r, r0);

    return 0;
  }

  template <typename PrecondT, typename iterT>
  int solve2(PrecondT& precond, iterT& iter, MatrixType const& A,
             VectorType const& b, VectorType& x)
  {
    if (iter.nullRhs())
      return 0;
    // clang-format off
    ValueType  rho (0),   rho1 (0),    alpha (0),     beta (0),    gamma (0),     omega (0);
    FutureType frho(rho), frho1(rho1), falpha(alpha), fbeta(beta), fgamma(gamma), fomega(omega) ;
    VectorType p, phat, s, shat, t, v, r, r0;
    // clang-format on

    m_algebra.allocate(AlgebraType::resource(A), p, phat, s, shat, t, v, r, r0);

    // SEQ0
    //  r = b - A * x;
    m_algebra.copy(b, r);
    m_algebra.mult(A, x, r0);
    m_algebra.axpy(-1., r0, r);

    // rtilde = r
    m_algebra.copy(r, r0);
    m_algebra.copy(r, p);
    m_algebra.dot(r, r0, frho1);
    if (m_output_level > 1)
      _print(0, "Seq 0", "rho1", rho1);

    /*
           phat = solve(M, p);
           v = A * phat;
           gamma = dot(r0, v);
           alpha = rho_1 / gamma;
           s = r - alpha * v;
         */
    // SEQ1
    m_algebra.exec(precond, p, phat);
    m_algebra.mult(A, phat, v);
    m_algebra.dot(v, r0, falpha);
    if (falpha.get() == 0)
      throw typename AlgebraType::NullValueException("alpha");
    alpha = frho1.get() / alpha;

    m_algebra.copy(r, s);
    m_algebra.axpy(-alpha, v, s);
    if (m_output_level > 1)
      _print(0, "Seq 1", "alpha", alpha);

    if (iter.stop(s)) {
      ++iter;
      m_algebra.axpy(alpha, phat, x);
      m_algebra.free(p, phat, s, shat, t, v, r, r0);
      return 0;
    }

    // SEQ 2
    m_algebra.exec(precond, s, shat);
    m_algebra.mult(A, shat, t);
    m_algebra.dot(t, s, fomega);
    m_algebra.dot(t, t, fbeta);
    if (fbeta.get() == 0) {
      if (iter.stop(r)) {
        ++iter;
        m_algebra.axpy(alpha, phat, x);
        m_algebra.free(p, phat, s, shat, t, v, r, r0);
        return 0;
      }
      else
        throw typename AlgebraType::NullValueException("beta");
    }
    omega = fomega.get() / beta;
    if (m_output_level > 1)
      _print(iter(), "Seq 2", "beta", beta, "alpha", alpha, "rho1", rho1);

    // SEQ 3
    m_algebra.axpy(omega, shat, x);
    m_algebra.axpy(alpha, phat, x);
    m_algebra.copy(s, r);
    m_algebra.axpy(-omega, t, r);

    rho = rho1;
    ++iter;
    if (m_output_level > 1)
      _print(iter(), "Seq 3", "beta", beta, "alpha", alpha, "rho1", rho1);

    while (!iter.stop(r)) {
      //SEQ4
      /*
            beta = (rho_1 / rho_2) * (alpha / omega);
            p = r + beta * (p - omega * v);
          */
      m_algebra.dot(r, r0, frho1);
      beta = (frho1.get() / rho) * (alpha / omega);
      m_algebra.axpy(-omega, v, p);
      m_algebra.scal(beta, p);
      m_algebra.axpy(1., r, p);
      if (m_output_level > 1)
        _print(iter(), "Seq 4", "beta", beta, "alpha", alpha, "rho1", rho1);

      if (rho == 0)
        throw typename AlgebraType::NullValueException("rho");

      //m_algebra.process (seq1);
      m_algebra.exec(precond, p, phat);
      m_algebra.mult(A, phat, v);
      m_algebra.dot(v, r0, falpha);
      if (falpha.get() == 0)
        throw typename AlgebraType::NullValueException("alpha");
      else
        alpha = rho1 / alpha;

      m_algebra.copy(r, s);
      m_algebra.axpy(-alpha, v, s);
      if (m_output_level > 1)
        _print(iter(), "Seq 1", "alpha", alpha);

      if (iter.stop(s)) {
        m_algebra.axpy(alpha, phat, x);
        m_algebra.free(p, phat, s, shat, t, v, r, r0);
        return 0;
      }

      //m_algebra.process (seq2);
      m_algebra.exec(precond, s, shat);
      m_algebra.mult(A, shat, t);
      m_algebra.dot(t, s, fomega);
      m_algebra.dot(t, t, fbeta);
      if (m_output_level > 1)
        _print(iter(), "Seq 2", "beta", beta, "alpha", alpha, "rho1", rho1, "omega", omega);
      if (fbeta.get() == 0) {
        if (iter.stop(s)) {
          m_algebra.axpy(alpha, phat, x);
          m_algebra.free(p, phat, s, shat, t, v, r, r0);
          return 0;
        }
        throw typename AlgebraType::NullValueException("beta");
      }
      else
        omega = fomega.get() / beta;

      //m_algebra.process (seq3);
      m_algebra.axpy(omega, shat, x);
      m_algebra.axpy(alpha, phat, x);
      m_algebra.copy(s, r);
      m_algebra.axpy(-omega, t, r);

      rho = rho1;

      ++iter;
      if (m_output_level > 1)
        _print(iter(), "end loop", "beta", beta, "alpha", alpha, "rho", rho);
    }

    m_algebra.free(p, phat, s, shat, t, v, r, r0);

    return 0;
  }

 private:
  void
  _print(int iter, std::string const& msg, std::string const& label0,
         ValueType value0)
  {
    if (m_trace_mng) {
      m_trace_mng->info() << msg;
      m_trace_mng->info() << "Iterate: " << iter << " " << label0 << " "
                          << value0;
    }
  }

  void
  _print(int iter, std::string const& msg, std::string const& label0,
         ValueType value0, std::string const& label1, ValueType value1)
  {
    if (m_trace_mng) {
      _print(iter, msg, label0, value0);
      m_trace_mng->info() << "Iterate: " << iter << " " << label1 << " "
                          << value1;
    }
  }

  void
  _print(int iter, std::string const& msg, std::string const& label0,
         ValueType value0, std::string const& label1, ValueType value1,
         std::string const& label2, ValueType value2)
  {
    if (m_trace_mng) {
      _print(iter, msg, label0, value0, label1, value1);
      m_trace_mng->info() << "Iterate: " << iter << " " << label2 << " "
                          << value2;
    }
  }

  void
  _print(int iter, std::string const& msg, std::string const& label0,
         ValueType value0, std::string const& label1, ValueType value1,
         std::string const& label2, ValueType value2,
         std::string const& label3, ValueType value3)
  {
    if (m_trace_mng) {
      _print(iter, msg, label0, value0, label1, value1, label2, value2);
      m_trace_mng->info() << "Iterate: " << iter << " " << label3 << " "
                          << value3;
    }
  }

  AlgebraType& m_algebra;
  ITraceMng* m_trace_mng = nullptr;
  int m_output_level = 0;
};
} // namespace Alien
