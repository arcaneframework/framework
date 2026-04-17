// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * This file is based on the work on AMGCL library (version march 2026)
 * which can be found at https://github.com/ddemidov/amgcl.
 *
 * Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>
 * SPDX-License-Identifier: MIT
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <gtest/gtest.h>

#include <vector>
#include <random>
#include <boost/multi_array.hpp>

#include "arccore/alina/QRFactorizationImpl.h"
#include "arccore/alina/ValueTypeInterface.h"
#include "arccore/alina/ValueTypeComplex.h"
#include "arccore/alina/StaticMatrix.h"

using namespace Arcane;

template <class T>
struct make_random
{
  static T get()
  {
    static std::mt19937 gen;
    static std::uniform_real_distribution<T> rnd;

    return rnd(gen);
  }
};

template <class T>
T random()
{
  return make_random<T>::get();
}

template <class T>
struct make_random<std::complex<T>>
{
  static std::complex<T> get()
  {
    return std::complex<T>(random<T>(), random<T>());
  }
};

template <class T, int N, int M>
struct make_random<Alina::StaticMatrix<T, N, M>>
{
  typedef Alina::StaticMatrix<T, N, M> matrix;
  static matrix get()
  {
    matrix A = Alina::math::zero<matrix>();
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < M; ++j)
        A(i, j) = make_random<T>::get();
    return A;
  }
};

template <class value_type, Alina::detail::storage_order order>
void qr_factorize(int n, int m)
{
  std::cout << "factorize " << n << " " << m << std::endl;
  typedef typename std::conditional<order == Alina::detail::row_major,
                                    boost::c_storage_order,
                                    boost::fortran_storage_order>::type ma_storage_order;

  boost::multi_array<value_type, 2> A0(boost::extents[n][m], ma_storage_order());

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      A0[i][j] = random<value_type>();

  boost::multi_array<value_type, 2> A = A0;

  Alina::detail::QRFactorization<value_type> qr;

  qr.factorize(n, m, A.data(), order);

  // Check that A = QR
  int p = std::min(n, m);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      value_type sum = Alina::math::zero<value_type>();

      for (int k = 0; k < p; ++k)
        sum += qr.Q(i, k) * qr.R(k, j);

      sum -= A0[i][j];

      ASSERT_NEAR(Alina::math::norm(sum), 0.0, 1e-8);
    }
  }
}

template <class value_type, Alina::detail::storage_order order>
void qr_solve(int n, int m)
{
  std::cout << "solve " << n << " " << m << std::endl;
  typedef typename std::conditional<order == Alina::detail::row_major,
                                    boost::c_storage_order,
                                    boost::fortran_storage_order>::type ma_storage_order;

  typedef typename Alina::math::rhs_of<value_type>::type rhs_type;

  boost::multi_array<value_type, 2> A0(boost::extents[n][m], ma_storage_order());

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      A0[i][j] = random<value_type>();

  boost::multi_array<value_type, 2> A = A0;

  Alina::detail::QRFactorization<value_type> qr;

  std::vector<rhs_type> f0(n, Alina::math::constant<rhs_type>(1));
  std::vector<rhs_type> f = f0;

  std::vector<rhs_type> x(m);

  qr.solve(n, m, A.data(), f.data(), x.data(), order);

  std::vector<rhs_type> Ax(n);
  for (int i = 0; i < n; ++i) {
    rhs_type sum = Alina::math::zero<rhs_type>();
    for (int j = 0; j < m; ++j)
      sum += A0[i][j] * x[j];

    Ax[i] = sum;

    if (n < m) {
      ASSERT_NEAR(Alina::math::norm(sum - f0[i]), 0.0, 1e-8);
    }
  }

  if (n >= m) {
    for (int i = 0; i < m; ++i) {
      rhs_type sumx = Alina::math::zero<rhs_type>();
      rhs_type sumf = Alina::math::zero<rhs_type>();

      for (int j = 0; j < n; ++j) {
        sumx += Alina::math::adjoint(A0[j][i]) * Ax[j];
        sumf += Alina::math::adjoint(A0[j][i]) * f0[j];
      }

      rhs_type delta = sumx - sumf;

      ASSERT_NEAR(Alina::math::norm(delta), 0.0, 1e-8);
    }
  }
}

TEST(alina_test_qr, test_qr_factorize)
{
  const int shape[][2] = {
    { 3, 3 },
    { 3, 5 },
    { 5, 3 },
    { 5, 5 }
  };

  const int n = sizeof(shape) / sizeof(shape[0]);

  for (int i = 0; i < n; ++i) {
    qr_factorize<double, Alina::detail::row_major>(shape[i][0], shape[i][1]);
    qr_factorize<double, Alina::detail::col_major>(shape[i][0], shape[i][1]);
    qr_factorize<std::complex<double>, Alina::detail::row_major>(shape[i][0], shape[i][1]);
    qr_factorize<std::complex<double>, Alina::detail::col_major>(shape[i][0], shape[i][1]);
    qr_factorize<Alina::StaticMatrix<double, 2, 2>, Alina::detail::row_major>(shape[i][0], shape[i][1]);
    qr_factorize<Alina::StaticMatrix<double, 2, 2>, Alina::detail::col_major>(shape[i][0], shape[i][1]);
  }
}

TEST(alina_test_qr, test_qr_solve)
{
  const int shape[][2] = {
    { 3, 3 },
    { 3, 5 },
    { 5, 3 },
    { 5, 5 }
  };

  const int n = sizeof(shape) / sizeof(shape[0]);

  for (int i = 0; i < n; ++i) {
    qr_solve<double, Alina::detail::row_major>(shape[i][0], shape[i][1]);
    qr_solve<double, Alina::detail::col_major>(shape[i][0], shape[i][1]);
    qr_solve<std::complex<double>, Alina::detail::row_major>(shape[i][0], shape[i][1]);
    qr_solve<std::complex<double>, Alina::detail::col_major>(shape[i][0], shape[i][1]);
    qr_solve<Alina::StaticMatrix<double, 2, 2>, Alina::detail::row_major>(shape[i][0], shape[i][1]);
    qr_solve<Alina::StaticMatrix<double, 2, 2>, Alina::detail::col_major>(shape[i][0], shape[i][1]);
  }
}

TEST(alina_test_qr, qr_issue_39)
{
  boost::multi_array<double, 2> A0(boost::extents[2][2]);
  A0[0][0] = 1e+0;
  A0[0][1] = 1e+0;
  A0[1][0] = 1e-8;
  A0[1][1] = 1e+0;

  boost::multi_array<double, 2> A = A0;

  Alina::detail::QRFactorization<double> qr;

  qr.factorize(2, 2, A.data());

  // Check that A = QR
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      double sum = 0;
      for (int k = 0; k < 2; ++k)
        sum += qr.Q(i, k) * qr.R(k, j);

      sum -= A0[i][j];

      ASSERT_NEAR(sum, 0.0, 1e-8);
    }
  }
}
