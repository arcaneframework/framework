// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BlockCSRMatrix.h                                            (C) 2000-2026 */
/*                                                                           */
/* Sparse matrix in block-CSR format.                         .              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_BLOCKCSRMATRIX_H
#define ARCCORE_ALINA_BLOCKCSRMATRIX_H
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

#include "arccore/alina/AlinaUtils.h"

#include <vector>
#include <algorithm>
#include <numeric>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sparse matrix in Block CSR format.
 *
 * \param V Value type.
 * \param C Column number type.
 * \param P Index type.
 */
template <typename V, typename C, typename P>
struct BlockCSRMatrix
{
  typedef V value_type;
  typedef V val_type;
  typedef C col_type;
  typedef P ptr_type;

  size_t block_size;
  size_t nrows, ncols;
  size_t brows, bcols;

  std::vector<ptr_type> ptr;
  std::vector<col_type> col;
  std::vector<val_type> val;

  /*!
   * \brief Converts matrix in CRS format to Block CRS format.
   *
   * \param A          Input matrix.
   * \param block_size Block size.
   *
   * \note Input matrix dimensions are *not* required to be divisible by
   * block_size.
   */
  template <class Matrix>
  BlockCSRMatrix(const Matrix& A, size_t block_size)
  : block_size(block_size)
  , nrows(backend::nbRow(A))
  , ncols(backend::nbColumn(A))
  , brows((nrows + block_size - 1) / block_size)
  , bcols((ncols + block_size - 1) / block_size)
  , ptr(brows + 1, 0)
  {
    std::vector<ptrdiff_t> marker(bcols, -1);

    // Count number of nonzeros in block matrix.
    for (ptr_type ib = 0; ib < static_cast<ptr_type>(brows); ++ib) {
      ptr_type ia = ib * block_size;

      for (size_t k = 0; k < block_size && ia < static_cast<ptr_type>(nrows); ++k, ++ia) {
        for (auto a = backend::row_begin(A, ia); a; ++a) {
          col_type cb = a.col() / block_size;

          if (marker[cb] != static_cast<col_type>(ib)) {
            marker[cb] = static_cast<col_type>(ib);
            ++ptr[ib + 1];
          }
        }
      }
    }

    {
      std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());
      col.resize(ptr.back());
      val.resize(ptr.back() * block_size * block_size, 0);
    }

    std::fill(marker.begin(), marker.end(), -1);

    // Fill the block matrix.
    for (ptr_type ib = 0; ib < static_cast<ptr_type>(brows); ++ib) {
      ptr_type ia = ib * block_size;
      ptr_type row_beg = ptr[ib];
      ptr_type row_end = row_beg;

      for (size_t k = 0; k < block_size && ia < static_cast<ptr_type>(nrows); ++k, ++ia) {
        for (auto a = backend::row_begin(A, ia); a; ++a) {
          col_type cb = a.col() / block_size;
          col_type cc = a.col() % block_size;
          val_type va = a.value();

          if (marker[cb] < row_beg) {
            marker[cb] = row_end;
            col[row_end] = cb;
            val[block_size * (block_size * row_end + k) + cc] = va;
            ++row_end;
          }
          else {
            val[block_size * (block_size * marker[cb] + k) + cc] = va;
          }
        }
      }
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
