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
#pragma once

#include <random>
#include <iostream>

/**
 *
 * G = (I-omega*A)
 * A-1 =omega*(I-G)-1=omega*(I+G+G2+...)
 */
namespace Alien
{

template <typename AlgebraT>
class NeumannPolyPreconditioner
{
 public:
  // clang-format off
  typedef AlgebraT                        AlgebraType ;
  typedef typename AlgebraType::Matrix    MatrixType;
  typedef typename AlgebraType::Vector    VectorType;
  typedef typename MatrixType::ValueType  ValueType;
  // clang-format on

  NeumannPolyPreconditioner(AlgebraType& algebra,
                            MatrixType const& matrix,
                            ValueType factor,
                            int polynome_order,
                            int factor_max_iter,
                            ITraceMng* trace_mng = nullptr)
  : m_algebra(algebra)
  , m_matrix(matrix)
  , m_factor(factor)
  , m_polynome_order(polynome_order)
  , m_factor_max_iter(factor_max_iter)
  , m_trace_mng(trace_mng)
  {
  }

  virtual ~NeumannPolyPreconditioner()
  {
    m_algebra.free(m_y);
    m_algebra.free(m_yy);
  };

  void setOutputLevel(int level)
  {
    m_output_level = level;
  }

  void init()
  {
    m_algebra.allocate(AlgebraType::resource(m_matrix), m_y, m_yy);
    m_algebra.assign(m_y, 0.);
    m_algebra.assign(m_yy, 0.);

    if (m_factor == 0) {
      computeFactor();
    }
    if (m_trace_mng) {
      m_trace_mng->info() << "Neunmann Preconditioner Info :";
      m_trace_mng->info() << "Polynome Degree       : " << m_polynome_order;
      m_trace_mng->info() << "Neunmann Factor       : " << m_factor;
      m_trace_mng->info() << "Power Method Max Iter : " << m_factor_max_iter;
    }
  }

  void solve(const VectorType& x, //! input
             VectorType& y //! output
  )
  {

    m_algebra.copy(x, y);
    for (int i = 0; i < m_polynome_order; ++i) {
      // yy = G*y
      evalG(y, m_y);
      //yy = x + yy = x + Gy
      m_algebra.axpy(1., x, m_y);
      //y = yy = x + G*y
      m_algebra.copy(m_y, y);
    }
    // y=factor*Pn(x)
    m_algebra.scal(m_factor, y);
  }

  void solve(AlgebraType& algebra,
             VectorType const& x,
             VectorType& y) const
  {

    algebra.copy(x, y);
    for (int i = 0; i < m_polynome_order; ++i) {
      // yy = G*y
      evalG(algebra, y, m_y);
      //yy = x + yy = x + Gy
      algebra.axpy(1., x, m_y);
      //y = yy = x + G*y
      algebra.copy(m_y, y);
    }
    // y=factor*Pn(x)
    algebra.scal(m_factor, y);
  }

  void evalG(AlgebraType& algebra,
             const VectorType& x, //! input
             VectorType& y) const
  {
    // yy = A.x
    algebra.mult(m_matrix, x, m_yy);
    algebra.copy(x, y);
    // y+= -factor * yy
    algebra.axpy(-m_factor, m_yy, y);
  }

  void evalG(const VectorType& x, //! input
             VectorType& y)
  {
    // yy = A.x
    m_algebra.mult(m_matrix, x, m_yy);
    m_algebra.copy(x, y);
    // y+= -factor * yy
    m_algebra.axpy(-m_factor, m_yy, y);
  }

  void computeFactor()
  {
    VectorType x;
    m_algebra.allocate(AlgebraType::resource(m_matrix), x);

    //m_algebra.assign(x,1.) ;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1., 1.);
    /*
      for (std::size_t i = 0; i < x.getAllocSize(); ++i)
      {
        x[i] =dis(gen);
      }*/
    m_algebra.assign(x, [&](auto i) {
      return dis(gen);
    });

    m_algebra.mult(m_matrix, x, m_y);
    ValueType norme2_k = m_algebra.norm2(m_y);
    for (int i = 0; i < m_factor_max_iter; ++i) {
      m_algebra.copy(m_y, x);
      m_algebra.mult(m_matrix, x, m_y);
      ValueType norme2 = m_algebra.norm2(m_y);
      m_factor = norme2 / norme2_k;
      norme2_k = norme2;
    }
    m_factor = 1. / (m_factor);

    if (m_trace_mng)
      m_trace_mng->info() << "Poly Factor : " << m_factor;

    m_algebra.free(x);
  }

 private:
  // clang-format off
  AlgebraType& m_algebra;

  //! matrix a preconditioner
  MatrixType const& m_matrix ;

  //! facteur d'acceleration
  ValueType m_factor    = 0.;
  int m_polynome_order  = 3 ;
  int m_factor_max_iter = 10 ;

  mutable VectorType m_y;
  mutable VectorType m_yy;

  ITraceMng* m_trace_mng    = nullptr;
  int        m_output_level = 0 ;
  // clang-format on
};

} // namespace Alien
