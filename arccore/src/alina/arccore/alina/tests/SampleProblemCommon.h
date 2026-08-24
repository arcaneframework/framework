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

#ifndef TESTS_SAMPLE_PROBLEM_HPP
#define TESTS_SAMPLE_PROBLEM_HPP

#include "arccore/alina/ValueTypeInterface.h"

#include <vector>
#include <iostream>

// Generates matrix for poisson problem in a unit cube.
template <typename ValueType, typename ColType, typename PtrType, typename RhsType> PtrType
sample_problem(Arcane::Int32 n,
               std::vector<ValueType>& val,
               std::vector<ColType>& col,
               std::vector<PtrType>& ptr,
               std::vector<RhsType>& rhs,
               double anisotropy = 1.0)
{
  // In sequential, the global size never exceed 2^31.
  PtrType n3 = n * n * n;

  ptr.clear();
  col.clear();
  val.clear();
  rhs.clear();

  ptr.reserve(n3 + 1);
  col.reserve(n3 * 7);
  val.reserve(n3 * 7);
  rhs.reserve(n3);

  const auto one = Arcane::Alina::math::identity<ValueType>();

  double hx = 1;
  double hy = hx * anisotropy;
  double hz = hy * anisotropy;

  ptr.push_back(0);
  for (ColType k = 0, idx = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i, ++idx) {
        if (k > 0) {
          col.push_back(idx - n * n);
          val.push_back(-1.0 / (hz * hz) * one);
        }

        if (j > 0) {
          col.push_back(idx - n);
          val.push_back(-1.0 / (hy * hy) * one);
        }

        if (i > 0) {
          col.push_back(idx - 1);
          val.push_back(-1.0 / (hx * hx) * one);
        }

        col.push_back(idx);
        val.push_back((2 / (hx * hx) + 2 / (hy * hy) + 2 / (hz * hz)) * one);

        if (i + 1 < n) {
          col.push_back(idx + 1);
          val.push_back(-1.0 / (hx * hx) * one);
        }

        if (j + 1 < n) {
          col.push_back(idx + n);
          val.push_back(-1.0 / (hy * hy) * one);
        }

        if (k + 1 < n) {
          col.push_back(idx + n * n);
          val.push_back(-1.0 / (hz * hz) * one);
        }

        rhs.push_back(Arcane::Alina::math::constant<RhsType>(1.0));
        ptr.push_back(static_cast<PtrType>(col.size()));
      }
    }
  }

  return n3;
}

//---------------------------------------------------------------------------
// Generates a distributed matrix for poisson problem in a unit cube
template <typename ValueType, typename ColType, typename PtrType, typename RhsType> PtrType
sample_problem_distributed(int comm_rank, int comm_size,
                           Arcane::Int32 n, int block_size,
                           std::vector<PtrType>& ptr,
                           std::vector<ColType>& col,
                           std::vector<ValueType>& val,
                           std::vector<RhsType>& rhs)
{
  // We use `PtrType` for multiplication because the global size may exceed
  // the maximum value of Int32
  PtrType nx = n;
  PtrType n3 = nx * nx * nx;

  PtrType chunk = (n3 + comm_size - 1) / comm_size;
  if (chunk % block_size != 0) {
    chunk += block_size - chunk % block_size;
  }
  PtrType row_beg = std::min(n3, chunk * comm_rank);
  PtrType row_end = std::min(n3, row_beg + chunk);
  chunk = row_end - row_beg;

  ptr.clear();
  ptr.reserve(chunk + 1);
  col.clear();
  col.reserve(chunk * 7);
  val.clear();
  val.reserve(chunk * 7);

  rhs.resize(chunk);
  std::fill(rhs.begin(), rhs.end(), 1.0);

  const double h2i = (n - 1) * (n - 1);
  ptr.push_back(0);

  for (PtrType idx = row_beg; idx < row_end; ++idx) {
    PtrType k = idx / (n * n);
    PtrType j = (idx / n) % n;
    PtrType i = idx % n;

    if (k > 0) {
      col.push_back(idx - n * n);
      val.push_back(-h2i);
    }

    if (j > 0) {
      col.push_back(idx - n);
      val.push_back(-h2i);
    }

    if (i > 0) {
      col.push_back(idx - 1);
      val.push_back(-h2i);
    }

    col.push_back(idx);
    val.push_back(6 * h2i);

    if (i + 1 < n) {
      col.push_back(idx + 1);
      val.push_back(-h2i);
    }

    if (j + 1 < n) {
      col.push_back(idx + n);
      val.push_back(-h2i);
    }

    if (k + 1 < n) {
      col.push_back(idx + n * n);
      val.push_back(-h2i);
    }

    ptr.push_back(col.size());
  }

  return chunk;
}

#endif
