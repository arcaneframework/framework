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

template <typename AlgebraT, bool saad = false>
class ChebyshevPreconditioner
{
 public:
  // clang-format off
  static const bool                       m_use_saad_algo = saad;
  typedef AlgebraT                        AlgebraType;
  typedef typename AlgebraType::Matrix    MatrixType;
  typedef typename AlgebraType::Vector    VectorType;
  typedef typename MatrixType::ValueType  ValueType;
  // clang-format on

  ChebyshevPreconditioner(AlgebraType& algebra,
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
  {}

  virtual ~ChebyshevPreconditioner()
  {
    m_algebra.free(m_inv_diag, m_y, m_r);
    if (m_use_saad_algo) {
      m_algebra.free(m_w);
    }
    else {
      m_algebra.free(m_p, m_z);
    }
  };

  void setOutputLevel(int level)
  {
    m_output_level = level;
  }

  void init()
  {
    m_algebra.allocate(AlgebraType::resource(m_matrix), m_y, m_r);
    m_algebra.assign(m_y, 0.);
    m_algebra.assign(m_r, 0.);
    if (m_use_saad_algo) {
      m_algebra.allocate(AlgebraType::resource(m_matrix), m_w);
      m_algebra.assign(m_w, 0.);
    }
    else {
      m_algebra.allocate(AlgebraType::resource(m_matrix), m_p, m_z);
      m_algebra.assign(m_p, 0.);
      m_algebra.assign(m_z, 0.);
    }

    m_algebra.allocate(AlgebraType::resource(m_matrix), m_inv_diag);
    m_algebra.assign(m_inv_diag, 1.);
    m_algebra.computeInvDiag(m_matrix, m_inv_diag);

    if (m_beta == 0) {
      computeBeta();
    }

    if (m_alpha == 0) {
      computeAlpha();
    }
    if (m_factor != 0.)
      m_alpha = m_beta / m_factor;

    m_delta = (m_beta - m_alpha) / 2;
    m_theta = (m_beta + m_alpha) / 2;

    if (m_trace_mng) {
      m_trace_mng->info() << "CHEBYSHEV PRECONDITIONER";
      m_trace_mng->info() << "Polynome Degree       : " << m_polynome_order;
      m_trace_mng->info() << "User EigenValue Ratio : " << m_factor;
      m_trace_mng->info() << "Power Method Max Iter : " << m_factor_max_iter;
      m_trace_mng->info() << "ALPHA                 : " << m_alpha;
      m_trace_mng->info() << "BETA                  : " << m_beta;
      m_trace_mng->info() << "EigenValue Ratio      : " << m_beta / m_alpha;
      m_trace_mng->info() << "THETA                 : " << m_theta;
      m_trace_mng->info() << "DELTA                 : " << m_delta;
    }
  }

  void computeBeta()
  {
    VectorType x;
    m_algebra.allocate(AlgebraType::resource(m_matrix), x);
    std::random_device rd;
    //std::mt19937 gen(rd());
    std::mt19937 gen;
    std::uniform_real_distribution<> dis(-1., 1.);

    m_algebra.assign(x, [&](std::size_t i) {
      return dis(gen);
    });
    /*
      for (std::size_t i = 0; i < x.getAllocSize(); ++i)
      {
        x[i] =dis(gen);
      }*/

    //m_algebra.assign(x,1.) ;
    ValueType norm = m_algebra.norm2(x);
    m_algebra.scal(1 / norm, x);

    for (int i = 0; i < m_factor_max_iter; ++i) {
      m_algebra.mult(m_matrix, x, m_y);
      m_algebra.pointwiseMult(m_inv_diag, m_y, m_y);
      ValueType xAx = m_algebra.dot(m_y, x);
      ValueType xdx = m_algebra.dot(x, x);
      ValueType norme = m_algebra.norm2(m_y);
      m_algebra.scal(1 / norme, m_y);
      m_algebra.copy(m_y, x);
      m_beta = xAx / xdx;
      if (m_trace_mng && m_output_level > 1)
        m_trace_mng->info() << "Iter(" << i << ") : beta ratio=" << m_beta;
    }
    if (m_trace_mng)
      m_trace_mng->info() << "Max eigen value : " << m_beta;

    m_algebra.free(x);
  }

  void computeAlpha()
  {

    VectorType x;
    m_algebra.allocate(AlgebraType::resource(m_matrix), x);

    std::random_device rd;
    //std::mt19937 gen(rd());
    std::mt19937 gen;
    std::uniform_real_distribution<> dis(-1., 1.);
    /*
      for (std::size_t i = 0; i < x.getAllocSize(); ++i)
      {
        x[i] =dis(gen);
      }*/

    m_algebra.assign(x, [&](std::size_t i) {
      return dis(gen);
    });
    //m_algebra.assign(x,1.) ;

    ValueType norm = m_algebra.norm2(x);
    m_algebra.scal(1 / norm, x);

    for (int i = 0; i < m_factor_max_iter; ++i) {
      m_algebra.mult(m_matrix, x, m_y);
      m_algebra.pointwiseMult(m_inv_diag, m_y, m_y);
      m_algebra.axpy(-m_beta, x, m_y);

      ValueType xAx = m_algebra.dot(m_y, x);
      ValueType xdx = m_algebra.dot(x, x);
      m_alpha = xAx / xdx;
      if (m_trace_mng && m_output_level > 1)
        m_trace_mng->info() << "Iter(" << i << ") : alpha ratio=" << m_alpha;

      ValueType norme = m_algebra.norm2(m_y);
      m_algebra.scal(1 / norme, m_y);
      m_algebra.copy(m_y, x);
    }
    m_alpha += m_beta;
    if (m_trace_mng)
      m_trace_mng->info() << "Min eigen value : " << m_alpha;

    m_algebra.free(x);
  }

  void solve1(const VectorType& x, //! input
              VectorType& y //! output
  ) const
  {
    /*
     * r       = invD(b - Ax)
     * s       = theta/delta
     * rho_old = 1/sigma
     * d       = 1/theta * r
     * do k=0,k<degree
     *   y = y + d
     *   r = r - invD.A.d
     *   rho = (2*sigma -rho_old)^-1
     *   d   = rho*rho_old d + 2*rho/delta r
     *   rho_old = rho
     * enddo
     */
    //m_algebra.copy(x,y) ;
    //m_algebra.mult(m_matrix,x,m_y) ;
    //m_algebra.axpy(-1.,m_y,m_r) ;
    //m_algebra.copy(x,m_r) ;
    m_algebra.xyz(m_inv_diag, x, m_r);
    m_trace_mng->info() << "||R||" << m_algebra.norm2(m_r);

    ValueType sigma = m_theta / m_delta;
    ValueType rho_old = 1. / sigma;
    ValueType rho = rho_old;
    m_algebra.copy(m_r, m_w);
    m_algebra.scal(1 / m_theta, m_w);
    m_algebra.copy(m_w, y);

    for (int k = 1; k < m_polynome_order; ++k) {
      // R = R +invD*A.W
      m_algebra.mult(m_matrix, m_w, m_y);
      m_algebra.xyz(m_inv_diag, m_y, m_y);
      m_algebra.axpy(-1., m_y, m_r);
      m_trace_mng->info() << k << " "
                          << "||R||" << m_algebra.norm2(m_r);

      rho = 1. / (2 * sigma - rho_old);

      m_algebra.scal(rho * rho_old, m_w);
      m_algebra.axpy(2 * rho / m_delta, m_r, m_w);
      rho_old = rho;

      m_algebra.axpy(1., m_w, y);
    }
  }

  void solve2(VectorType const& x, //! input
              VectorType& y //! output
  ) const
  {

    const ValueType d = (m_beta + m_alpha) / 2; // Ifpack2 calls this theta
    const ValueType c = (m_beta - m_alpha) / 2; // Ifpack2 calls this 1/delta

    m_algebra.copy(x, y);

    m_algebra.mult(m_matrix, y, m_y);
    m_algebra.copy(x, m_r);
    m_algebra.axpy(-1., m_y, m_r);

    //solve (Z, D_inv, R); // z = D_inv * R, that is, D \ R.
    //m_algebra.copy(m_r,m_z) ;
    m_algebra.xyz(m_inv_diag, m_r, m_z);

    m_algebra.copy(m_z, m_p);
    ValueType alpha = 2 / d;

    //X.update (alpha, P, one); // X = X + alpha*P
    m_algebra.axpy(alpha, m_p, y);

    for (int i = 1; i < m_polynome_order; ++i) {
      //computeResidual (R, B, A, X); // R = B - A*X
      m_algebra.mult(m_matrix, y, m_y);
      m_algebra.copy(x, m_r);
      m_algebra.axpy(-1., m_y, m_r);

      m_trace_mng->info() << i << " "
                          << "||R||" << m_algebra.norm2(m_r);

      //solve (Z, D_inv, R); // z = D_inv * R, that is, D \ R.
      //m_algebra.copy(m_r,m_z) ;
      m_algebra.xyz(m_inv_diag, m_r, m_z);

      //beta = (c * alpha / two)^2;
      //const ST sqrtBeta = c * alpha / two;
      //beta = sqrtBeta * sqrtBeta;
      ValueType beta = alpha * (c / 2.) * (c / 2.);
      alpha = 1. / (d - beta);

      //P.update (one, Z, beta); // P = Z + beta*P
      m_algebra.scal(beta, m_p);
      m_algebra.axpy(1., m_z, m_p);

      //X.update (alpha, P, one); // X = X + alpha*P
      m_algebra.axpy(alpha, m_p, y);
      // If we compute the residual here, we could either do R = B -
      // A*X, or R = R - alpha*A*P.  Since we choose the former, we
      // can move the computeResidual call to the top of the loop.
    }
  }

  void solve(const VectorType& x, //! input
             VectorType& y //! output
  ) const
  {
    if (m_use_saad_algo)
      solve1(x, y);
    else
      solve2(x, y);
  }

  void _algo1(AlgebraType& algebra,
              VectorType const& x,
              VectorType& y) const
  {
    /*
     * r       = b -Ax
     * s       = theta/delta
     * rho_old = 1/sigma
     * d       = 1/theta * r
     * do k=0,k<degree
     *   y = y + d
     *   r = r - Ad
     *   rho = (2sigma1 -rho_old)^-1
     *   d   = rho*rho_old d + 2*rho/delta r
     *   rho_old = rho
     * enddo
     *
     *
     *
     *
     alpha = lambdaMax / eigRatio;
     beta = boostFactor_ * lambdaMax;
     delta = two / (beta - alpha);
     theta = (beta + alpha) / two;
     s1 = theta * delta;

     // W := (1/theta)*D_inv*B and X := 0 + W.

     // The rest of the iterations.
     ST rhok = one / s1;
     ST rhokp1, dtemp1, dtemp2;
     for (int deg = 1; deg < numIters; ++deg)
     {
     rhokp1 = one / (two * s1 - rhok);
     dtemp1 = rhokp1 * rhok;
     dtemp2 = two * rhokp1 * delta;
     rhok = rhokp1;
     // W := dtemp2*D_inv*(B - A*X) + dtemp1*W.
     // X := X + W
     }
     *
     *
     */

    //m_algebra.copy(x,m_r) ;
    m_algebra.pointwiseMult(m_inv_diag, x, m_r);

    ValueType sigma = m_theta / m_delta;
    ValueType rho_old = 1. / sigma;
    ValueType rho = rho_old;
    m_algebra.copy(m_r, m_w);
    m_algebra.scal(1 / m_theta, m_w);
    m_algebra.copy(m_w, y);

    m_trace_mng->info() << "sigma  =" << sigma;
    m_trace_mng->info() << "rho    =" << rho;
    m_trace_mng->info() << "theta  =" << m_theta;

    for (int k = 1; k < m_polynome_order; ++k) {
      m_algebra.mult(m_matrix, m_w, m_y);
      m_algebra.pointwiseMult(m_inv_diag, m_y, m_y);
      m_algebra.axpy(-1., m_y, m_r);

      rho = 1. / (2 * sigma - rho_old);
      m_algebra.scal(rho * rho_old, m_w);
      m_algebra.axpy(2 * rho / m_delta, m_r, m_w);

      //m_algebra.barrier(0,y) ;
      m_algebra.axpy(1., m_w, y);

      rho_old = rho;
    }
  }

  void _algo2(AlgebraType& algebra,
              VectorType const& x,
              VectorType& y) const
  {
    const ValueType d = (m_beta + m_alpha) / 2;
    const ValueType c = (m_beta - m_alpha) / 2;

    m_algebra.copy(x, y);

    m_algebra.mult(m_matrix, y, m_y);
    m_algebra.copy(x, m_r);
    m_algebra.axpy(-1., m_y, m_r);

    m_algebra.pointwiseMult(m_inv_diag, m_r, m_z); // z = D_inv * R, that is, D \ R.

    m_algebra.copy(m_z, m_p);

    ValueType alpha = 2 / d;
    m_algebra.axpy(alpha, m_p, y); // X = X + alpha*P

    for (int i = 1; i < m_polynome_order; ++i) {
      // R = B - A*X
      m_algebra.mult(m_matrix, y, m_y);
      m_algebra.copy(x, m_r);
      m_algebra.axpy(-1., m_y, m_r);

      // z = D_inv * R, that is, D \ R.
      m_algebra.pointwiseMult(m_inv_diag, m_r, m_z);

      //beta = (c * alpha / two)^2;
      //const ST sqrtBeta = c * alpha / two;
      //beta = sqrtBeta * sqrtBeta;
      ValueType beta = alpha * (c / 2.) * (c / 2.);
      alpha = 1. / (d - beta);

      // P = Z + beta*P
      m_algebra.scal(beta, m_p);
      m_algebra.axpy(1., m_z, m_p);

      // X = X + alpha*P
      m_algebra.axpy(alpha, m_p, y);
    }
  }

  void solve(AlgebraType& algebra,
             VectorType const& x,
             VectorType& y) const
  {
    if (m_use_saad_algo)
      _algo1(algebra, x, y);
    else
      _algo2(algebra, x, y);
  }

 private:
  AlgebraType& m_algebra;

  //! matrix a preconditioner
  MatrixType const& m_matrix;

  //! facteur d'acceleration
  // clang-format off
  ValueType m_factor = 0.;
  ValueType m_alpha  = 0.;
  ValueType m_beta   = 0.;
  ValueType m_theta  = 0.;
  ValueType m_delta  = 0.;
  // clang-format on

  int m_polynome_order;

  int m_factor_max_iter;

  VectorType m_inv_diag;

  mutable VectorType m_y;
  mutable VectorType m_r;
  mutable VectorType m_w;
  mutable VectorType m_p;
  mutable VectorType m_z;

  ITraceMng* m_trace_mng = nullptr;
  int m_output_level = 0;
};

} // namespace Alien
