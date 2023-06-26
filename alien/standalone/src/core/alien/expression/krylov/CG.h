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
 * cg.h
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
class CG
{
 public:
  // clang-format off
  typedef AlgebraT                         AlgebraType;
  typedef typename AlgebraType::Matrix     MatrixType;
  typedef typename AlgebraType::Vector     VectorType;
  typedef typename MatrixType::ValueType   ValueType;
  typedef typename AlgebraType::FutureType FutureType;
  // clang-format on

  CG(AlgebraType& algebra, ITraceMng* trace_mng = nullptr)
  : m_algebra(algebra)
  , m_trace_mng(trace_mng)
  {}

  virtual ~CG()
  {}

  void setOutputLevel(int level)
  {
    m_output_level = level;
  }

  template <typename PrecondT, typename iterT>
  int solve(PrecondT& precond,
            iterT& iter,
            MatrixType const& A,
            VectorType const& b,
            VectorType& x)
  {
    if (iter.nullRhs())
      return 0;
    ValueType rho(0), rho1(0), alpha(0);
    VectorType p, z, q, r;

    m_algebra.allocate(AlgebraType::resource(A), p, z, q, r);

    // SEQ0
    //  r = b - A * x;
    m_algebra.copy(b, r);
    m_algebra.mult(A, x, p);
    m_algebra.axpy(-1., p, r);

    // SEQ1
    /*
     * z = solve(M,r)
     * p = z
     * q = A p
     * rho1 = dot(r,z)
     * alpha = dot(p,q)
     * alpha = rho1/alpha
     * x += alpha*p
     * r += alpha*q
     */
    m_algebra.exec(precond, r, z);
    rho1 = m_algebra.dot(r, z);
    m_algebra.copy(z, p);
    m_algebra.mult(A, p, q);
    alpha = m_algebra.dot(p, q);
    if (m_output_level > 1)
      _print(0, "Seq 1", "rho", rho1, "alpha", alpha);
    if (alpha == 0) {
      if (iter.stop(r)) {
        ++iter;
        m_algebra.free(p, z, q, r);
        return 0;
      }
      else
        throw typename AlgebraType::NullValueException("alpha");
    }
    alpha = rho1 / alpha;
    m_algebra.axpy(alpha, p, x);
    m_algebra.axpy(-alpha, q, r);
    rho = rho1;
    ++iter;

    while (!iter.stop(r)) {
      // SEQ2
      /*
       * z = solve(M,r)
       * rho1 = dot(r,z)
       * alpha = rho1/rho
       * p = z + alpha*p
       * q = A*p
       * alpha = dot(q,p)
       * alpha = rho1/alpha
       * x += alpha*p
       * r -= alpha*q
       */
      m_algebra.exec(precond, r, z);
      rho1 = m_algebra.dot(r, z);
      alpha = rho1 / rho;
      m_algebra.axpy(alpha, p, z);
      m_algebra.copy(z, p);
      m_algebra.mult(A, p, q);
      alpha = m_algebra.dot(q, p);
      if (alpha == 0) {
        if (iter.stop(r)) {
          ++iter;
          m_algebra.free(p, z, q, r);
          return 0;
        }
        else
          throw typename AlgebraType::NullValueException("alpha");
      }
      alpha = rho1 / alpha;
      m_algebra.axpy(alpha, p, x);
      m_algebra.axpy(-alpha, q, r);
      rho = rho1;
      ++iter;
    }

    m_algebra.free(p, z, q, r);

    return 0;
  }

  template <typename PrecondT, typename iterT>
  int solve2(PrecondT& precond,
             iterT& iter,
             MatrixType const& A,
             VectorType const& b,
             VectorType& x)
  {

    if (iter.nullRhs())
      return 0;
    ValueType rho(0), rho1(0), alpha(0);
    FutureType frho(rho), frho1(rho1), falpha(alpha);
    VectorType p, z, q, r;

    m_algebra.allocate(AlgebraType::resource(A), p, z, q, r);

    // SEQ0
    //  r = b - A * x;
    m_algebra.copy(b, r);
    m_algebra.mult(A, x, p);
    m_algebra.axpy(-1., p, r);

    // SEQ1
    /*
     * z = solve(M,r)
     * p = z
     * q = A p
     * rho1 = dot(r,p)
     * alpha = dot(p,q)
     * alpha = rho1/alpha
     * x += alpha*p
     * r += alpha*q
     */
    m_algebra.exec(precond, r, z);
    m_algebra.copy(z, p);
    m_algebra.mult(A, p, q);
    m_algebra.dot(r, p, frho1);
    m_algebra.dot(p, q, falpha);
    if (falpha.get() == 0) {
      if (iter.stop(r)) {
        ++iter;
        m_algebra.free(p, z, q, r);
        return 0;
      }
      else
        throw typename AlgebraType::NullValueException("alpha");
    }
    alpha = frho1.get() / alpha;
    m_algebra.axpy(alpha, p, x);
    m_algebra.axpy(-alpha, q, r);
    rho = rho1;
    ++iter;

    while (!iter.stop(r)) {

      /*
       * z = solve(M,r)
       * rho1 = dot(r,z)
       * alpha = rho1/rho
       * p = z + alpha*p
       * q = A*p
       * alpha = dot(q,p)
       * alpha = rho1/alpha
       * x += alpha*p
       * r -= alpha*q
       */
      m_algebra.exec(precond, r, z);
      m_algebra.dot(r, z, frho1);
      alpha = frho1.get() / rho;
      m_algebra.axpy(alpha, p, z);
      m_algebra.copy(z, p);
      m_algebra.mult(A, p, q);
      m_algebra.dot(p, q, falpha);
      if (falpha.get() == 0) {
        if (iter.stop(r)) {
          ++iter;
          m_algebra.free(p, z, q, r);
          return 0;
        }
        else
          throw typename AlgebraType::NullValueException("alpha");
      }
      alpha = rho1 / alpha;
      m_algebra.axpy(alpha, p, x);
      m_algebra.axpy(-alpha, q, r);
      rho = rho1;
      ++iter;
    }

    m_algebra.free(p, z, q, r);

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
