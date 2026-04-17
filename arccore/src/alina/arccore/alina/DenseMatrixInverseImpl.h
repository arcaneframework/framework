// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DenseMatrixInverseImpl.h                                    (C) 2000-2026 */
/*                                                                           */
/* Compute the inverse of a dense matrix.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_DENSEMATRIXINVERSEIMPL_H
#define ARCCORE_ALINA_DENSEMATRIXINVERSEIMPL_H
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

#include <algorithm>
#include <cassert>
#include <numeric>
#include "arccore/alina/AlinaUtils.h"
#include "arccore/alina/ValueTypeInterface.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina::detail
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief  Inverse of a dense matrix.
 */
template <typename value_type> static void
inverse(int n, value_type* A, value_type* t, int* p)
{
  std::iota(p, p + n, 0);

  // Perform LU-factorization of A in-place
  for (int col = 0; col < n; ++col) {
    // Find pivot element
    int pivot_i = col;
    using mag_type = typename math::scalar_of<value_type>::type;
    mag_type pivot_mag = math::zero<mag_type>();
    for (int i = col; i < n; ++i) {
      int row = p[i];
      mag_type mag = math::norm(A[row * n + col]);
      if (mag > pivot_mag) {
        pivot_mag = mag;
        pivot_i = i;
      }
    }
    std::swap(p[col], p[pivot_i]);
    int pivot_row = p[col];
    // We have found pivot element, perform Gauss elimination
    value_type d = math::inverse(A[pivot_row * n + col]);
    assert(!math::is_zero(d));
    for (int i = col + 1; i < n; ++i) {
      int row = p[i];
      A[row * n + col] *= d;
      for (int j = col + 1; j < n; ++j)
        A[row * n + j] -= A[row * n + col] * A[pivot_row * n + j];
    }
    A[pivot_row * n + col] = d;
  }

  // Invert identity matrix in-place to get the solution.
  for (int k = 0; k < n; ++k) {
    // Lower triangular solve:
    for (int i = 0; i < n; ++i) {
      int row = p[i];
      value_type b = (row == k) ? math::identity<value_type>() : math::zero<value_type>();
      for (int j = 0; j < i; ++j)
        b -= A[row * n + j] * t[j * n + k];
      t[i * n + k] = b;
    }

    // Upper triangular solve:
    for (int i = n; i-- > 0;) {
      int row = p[i];
      for (int j = i + 1; j < n; ++j)
        t[i * n + k] -= A[row * n + j] * t[j * n + k];
      t[i * n + k] *= A[row * n + i];
    }
  }

  std::copy(t, t + n * n, A);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina::detail

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
