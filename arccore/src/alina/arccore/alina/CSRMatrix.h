// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CSRMatrix.h                                                 (C) 2000-2026 */
/*                                                                           */
/* Sparse matrix stored in CSR (Compressed Sparse Row) format.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_CSRMATRIX_H
#define ARCCORE_ALINA_CSRMATRIX_H
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

#include "arccore/alina/AlinaGlobal.h"
#include "arccore/alina/AlinaUtils.h"

// A supprimer
#include "arccore/alina/BackendInterface.h"

#include <cstddef>
#include <numeric>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

//! Array for internal CSRMatrix fields
template <typename DataType>
class CSRArray
{
 public:

  using val_type = DataType;

  CSRArray() = default;
  CSRArray(const CSRArray&) = delete;
  CSRArray& operator=(const CSRArray&) = delete;
  ~CSRArray()
  {
    // Do not free memory here because we are not the owner
    // if own_data is false. The CSRMatrix will handle that.
  }

 public:

  val_type& operator[](Int64 i) { return ptr[i]; }
  const val_type& operator[](Int64 i) const { return ptr[i]; }
  operator val_type*() { return ptr; }
  operator const val_type*() const { return ptr; }
  val_type* operator+(Int64 i) { return ptr + i; }
  const val_type* operator+(Int64 i) const { return ptr + i; }
  val_type* data() { return ptr; }
  const val_type* data() const { return ptr; }

 public:

  //! Set the new size. WARNING: this method do not handle the delete of the current value
  void resize(size_t new_size)
  {
    ptr = new val_type[new_size];
  }
  void reset()
  {
    delete[] ptr;
    ptr = nullptr;
  }
  void setPointerZeroCopy(val_type* new_ptr)
  {
    ptr = new_ptr;
  }

 private:

  val_type* ptr = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sparse matrix stored in CSR (Compressed Sparse Row) format.
 */
template <typename val_t, typename col_t, typename ptr_t>
struct CSRMatrix
{
  typedef val_t value_type;
  typedef val_t val_type;
  typedef col_t col_type;
  typedef ptr_t ptr_type;

 private:
  size_t m_nb_row = 0;
 public:
  size_t ncols = 0;
private:
  size_t m_nb_non_zero = 0;
public:
  CSRArray<ptr_type> ptr;
  CSRArray<col_type> col;
  CSRArray<val_type> val;
  bool own_data = true;

  [[nodiscard]] size_t nbRow() const noexcept { return m_nb_row; }
  void setNbRow(size_t v) { m_nb_row = v; }

  [[nodiscard]] size_t nbNonZero() const noexcept { return m_nb_non_zero; }
  void setNbNonZero(size_t v) { m_nb_non_zero = v; }

 public:

  CSRMatrix() = default;

  template <class PtrRange, class ColRange, class ValRange>
  CSRMatrix(size_t nrows, size_t ncols, const PtrRange& ptr_range, const ColRange& col_range, const ValRange& val_range)
  : m_nb_row(nrows)
  , ncols(ncols)
  {
    ARCCORE_ALINA_TIC("CSR copy");
    precondition(static_cast<ptrdiff_t>(nrows + 1) == std::distance(std::begin(ptr_range), std::end(ptr_range)),
                 "ptr_range has wrong size in crs constructor");

    m_nb_non_zero = ptr_range[nrows];

    precondition(static_cast<ptrdiff_t>(m_nb_non_zero) == std::distance(std::begin(col_range), std::end(col_range)),
                 "col_range has wrong size in crs constructor");

    precondition(static_cast<ptrdiff_t>(m_nb_non_zero) == std::distance(std::begin(val_range), std::end(val_range)),
                 "val_range has wrong size in crs constructor");

    ptr.resize(nrows + 1);
    col.resize(m_nb_non_zero);
    val.resize(m_nb_non_zero);

    ptr[0] = ptr_range[0];
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(nrows); ++i) {
      ptr[i + 1] = ptr_range[i + 1];
      for (auto j = ptr_range[i]; j < ptr_range[i + 1]; ++j) {
        col[j] = col_range[j];
        val[j] = val_range[j];
      }
    }
    ARCCORE_ALINA_TOC("CSR copy");
  }

  // TODO: A supprimer. Mettre cela dans une function externe pour ne pas dépendre de backend
  template <class Matrix>
  CSRMatrix(const Matrix& A)
  : m_nb_row(backend::nbRow(A))
  , ncols(backend::nbColumn(A))
  {
    ARCCORE_ALINA_TIC("CSR copy");
    ptr.resize(m_nb_row + 1);
    ptr[0] = 0;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(m_nb_row); ++i) {
      int row_width = 0;
      for (auto a = backend::row_begin(A, i); a; ++a)
        ++row_width;
      ptr[i + 1] = row_width;
    }

    m_nb_non_zero = scan_row_sizes();
    col.resize(m_nb_non_zero);
    val.resize(m_nb_non_zero);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(m_nb_row); ++i) {
      ptr_type row_head = ptr[i];
      for (auto a = backend::row_begin(A, i); a; ++a) {
        col[row_head] = a.col();
        val[row_head] = a.value();

        ++row_head;
      }
    }
    ARCCORE_ALINA_TOC("CSR copy");
  }

  CSRMatrix(const CSRMatrix& other)
  : m_nb_row(other.m_nb_row)
  , ncols(other.ncols)
  , m_nb_non_zero(other.m_nb_non_zero)
  {
    if (other.ptr && other.col && other.val) {
      ptr.resize(m_nb_row + 1);
      col.resize(m_nb_non_zero);
      val.resize(m_nb_non_zero);

      ptr[0] = other.ptr[0];
#pragma omp parallel for
      for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(m_nb_row); ++i) {
        ptr[i + 1] = other.ptr[i + 1];
        for (ptr_type j = other.ptr[i]; j < other.ptr[i + 1]; ++j) {
          col[j] = other.col[j];
          val[j] = other.val[j];
        }
      }
    }
  }

  CSRMatrix(CSRMatrix&& other) noexcept
  : m_nb_row(other.m_nb_row)
  , ncols(other.ncols)
  , m_nb_non_zero(other.m_nb_non_zero)
  , ptr(other.ptr)
  , col(other.col)
  , val(other.val)
  , own_data(other.own_data)
  {
    other.m_nbRow = 0;
    other.ncols = 0;
    other.m_nb_non_zero = 0;
    other.ptr = 0;
    other.col = 0;
    other.val = 0;
  }

  CSRMatrix& operator=(const CSRMatrix& other)
  {
    free_data();

    m_nb_row = other.m_nb_row;
    ncols = other.ncols;
    m_nb_non_zero = other.m_nb_non_zero;

    if (other.ptr && other.col && other.val) {
      ptr = new ptr_type[m_nb_row + 1];
      col = new col_type[m_nb_non_zero];
      val = new val_type[m_nb_non_zero];

      ptr[0] = other.ptr[0];
#pragma omp parallel for
      for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(m_nb_row); ++i) {
        ptr[i + 1] = other.ptr[i + 1];
        for (ptr_type j = other.ptr[i]; j < other.ptr[i + 1]; ++j) {
          col[j] = other.col[j];
          val[j] = other.val[j];
        }
      }
    }

    return *this;
  }

  CSRMatrix& operator=(CSRMatrix&& other) noexcept
  {
    std::swap(m_nb_row, other.m_nb_row);
    std::swap(ncols, other.ncols);
    std::swap(m_nb_non_zero, other.m_nb_non_zero);
    std::swap(ptr, other.ptr);
    std::swap(col, other.col);
    std::swap(val, other.val);
    std::swap(own_data, other.own_data);

    return *this;
  }

  void free_data()
  {
    if (own_data) {
      ptr.reset();
      col.reset();
      val.reset();
    }
  }

  void set_size(size_t n, size_t m, bool clean_ptr = false)
  {
    precondition(!ptr, "matrix data has already been allocated!");

    m_nb_row = n;
    ncols = m;

    ptr.resize(m_nb_row + 1);

    if (clean_ptr) {
      ptr[0] = 0;
#pragma omp parallel for
      for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(m_nb_row); ++i)
        ptr[i + 1] = 0;
    }
  }

  ptr_type scan_row_sizes()
  {
    std::partial_sum(ptr.data(), ptr.data() + m_nb_row + 1, ptr.data());
    return ptr[m_nb_row];
  }

  void set_nonzeros()
  {
    set_nonzeros(ptr[m_nb_row]);

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(m_nb_row); ++i) {
      ptrdiff_t row_beg = ptr[i];
      ptrdiff_t row_end = ptr[i + 1];
      for (ptrdiff_t j = row_beg; j < row_end; ++j) {
        col[j] = 0;
        val[j] = math::zero<val_type>();
      }
    }
  }

  void set_nonzeros(size_t n, bool need_values = true)
  {
    precondition(!col && !val, "matrix data has already been allocated!");

    m_nb_non_zero = n;

    col.resize(m_nb_non_zero);

    if (need_values)
      val.resize(m_nb_non_zero);
  }

  ~CSRMatrix()
  {
    free_data();
  }

  class row_iterator
  {
   public:

    row_iterator(const col_type* col, const col_type* end, const val_type* val)
    : m_col(col)
    , m_end(end)
    , m_val(val)
    {}

    operator bool() const
    {
      return m_col < m_end;
    }

    row_iterator& operator++()
    {
      ++m_col;
      ++m_val;
      return *this;
    }

    col_type col() const
    {
      return *m_col;
    }

    val_type value() const
    {
      return *m_val;
    }

   private:

    const col_type* m_col = nullptr;
    const col_type* m_end = nullptr;
    const val_type* m_val = nullptr;
  };

  row_iterator row_begin(size_t row) const
  {
    ptr_type p = ptr[row];
    ptr_type e = ptr[row + 1];
    return row_iterator(col + p, col + e, val + p);
  }

  size_t bytes() const
  {
    if (own_data) {
      return sizeof(ptr_type) * (m_nb_row + 1) + sizeof(col_type) * m_nb_non_zero + sizeof(val_type) * m_nb_non_zero;
    }
    else {
      return 0;
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
