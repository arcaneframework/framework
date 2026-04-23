// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CSRMatrixView.h                                             (C) 2000-2026 */
/*                                                                           */
/* View of a sparse matrix stored in CSR (Compressed Sparse Row) format.     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ALINA_CSRMATRIXVIEW_H
#define ARCCORE_ALINA_CSRMATRIXVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/alina/AlinaGlobal.h"

#include "arccore/base/Span.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Index in the RowColumn list of a CSR Matrix.
 */
template <typename IndexType_>
class CSRRowColumnIndex
{
 public:

  using IndexType = IndexType_;

 public:

  CSRRowColumnIndex() = default;
  explicit constexpr ARCCORE_HOST_DEVICE CSRRowColumnIndex(IndexType index)
  : m_index(index)
  {}

 public:

  [[nodiscard]] constexpr ARCCORE_HOST_DEVICE IndexType value() const { return m_index; }
  constexpr ARCCORE_HOST_DEVICE operator IndexType() const { return m_index; }

 private:

  IndexType m_index = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Represents an iterator over the columns of a CSRRow.
 */
template <typename IndexType_>
class CSRRowColumnIterator
{
  template <typename T> friend class CSRRow;
  using IndexType = IndexType_;

 public:

  CSRRowColumnIterator() = default;

 private:

  explicit constexpr CSRRowColumnIterator(IndexType index)
  : m_index(index)
  {}

 public:

  constexpr CSRRowColumnIndex<IndexType> operator*() const { return CSRRowColumnIndex(m_index); }
  constexpr void operator++() { ++m_index; }

  friend constexpr bool
  operator!=(const CSRRowColumnIterator& lhs, const CSRRowColumnIterator& rhs)
  {
    return lhs.m_index != rhs.m_index;
  }
  constexpr bool isValid() const { return m_index != (-1); }

 private:

  IndexType m_index = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Represents a row of a CSR Matrix.
 */
template <typename IndexType_>
class CSRRow
{
  template <typename V, typename C, typename R>
  friend class CSRMatrixView;
  template <typename T>
  friend class CSRRowRangeIterator;

 public:

  using IndexType = IndexType_;
  using IteratorType = CSRRowColumnIterator<IndexType>;

 public:

  CSRRow() = default;

 private:

  constexpr CSRRow(IndexType begin, IndexType end)
  : m_begin(begin)
  , m_end(end)
  {}

 public:

  constexpr IteratorType begin() const { return IteratorType(m_begin); }
  constexpr IteratorType end() const { return IteratorType(m_end); }

 private:

  IndexType m_begin = -1;
  IndexType m_end = -1;
};

template <typename T, size_t IntType>
class SpanChooser;

template<typename T>
class SpanChooser<T,4>
{
 public:
  using SpanType = SmallSpan<T>;
};

template <typename T>
class SpanChooser<T, 8>
{
 public:

  using SpanType = Span<T>;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Iterator over rows of a CSR Matrix.
 */
template <typename IndexType_>
class CSRRowRangeIterator
{
  template <typename V, typename C, typename R>
  friend class CSRMatrixView;
  template <typename T>
  friend class CSRRowRange;

 public:

  using IndexType = IndexType_;
  using ThatClass = CSRRowRangeIterator<IndexType>;

 public:

  //! Sentinel to detect end of iteration
  class Sentinel
  {
    friend CSRRowRangeIterator<IndexType>;

   public:

    explicit Sentinel(IndexType v)
    : m_end(v)
    {}

   private:

    IndexType m_end = -1;
  };

 public:

  CSRRowRangeIterator() = default;

 private:

  constexpr CSRRowRangeIterator(IndexType* row_indexes, IndexType index)
  : m_row_indexes(row_indexes)
  , m_index(index)
  {}

 public:

  constexpr CSRRow<IndexType> operator*() const
  {
    return CSRRow(m_row_indexes[m_index], m_row_indexes[m_index + 1]);
  }
  constexpr ThatClass& operator++()
  {
    ++m_index;
    return (*this);
  }
  constexpr ThatClass operator++(int)
  {
    return ThatClass(m_row_indexes, m_index++);
  }
  friend bool operator!=(const CSRRowRangeIterator& a, const Sentinel& b)
  {
    return a.m_index != b.m_end;
  }

 private:

  IndexType* m_row_indexes = nullptr;
  IndexType m_index = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Represents a range of rows of a CSR Matrix.
 */
template <typename IndexType_>
class CSRRowRange
{
  template <typename ValueType_, typename ColumnType_, typename RowIndexType_>
  friend class CSRMatrixView;

 public:

  using IndexType = IndexType_;
  using IteratorType = CSRRowRangeIterator<IndexType>;
  using SentinelType = IteratorType::Sentinel;

 public:

  CSRRowRange() = default;

 private:

  constexpr ARCCORE_HOST_DEVICE CSRRowRange(IndexType* row_indexes, IndexType begin, IndexType end)
  : m_row_indexes(row_indexes)
  , m_begin(begin)
  , m_end(end)
  {}

 public:

  constexpr IteratorType begin() const
  {
    return IteratorType(m_row_indexes, m_begin);
  }
  constexpr SentinelType end() const
  {
    return SentinelType(m_end);
  }

 private:

  IndexType* m_row_indexes = nullptr;
  IndexType m_begin = -1;
  IndexType m_end = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sparse matrix stored in CSR (Compressed Sparse Row) format.
 */
template <typename ValueType_, typename ColumnType_, typename RowIndexType_>
class CSRMatrixView
{
 public:

  typedef ValueType_ value_type;
  typedef ValueType_ val_type;
  typedef ColumnType_ col_type;
  using ptr_type = RowIndexType_;
  using RowIndexType = RowIndexType_;
  using ColumnSpanType = SpanChooser<ColumnType_,sizeof(RowIndexType_)>::SpanType;
  using ValueSpanType = SpanChooser<ValueType_,sizeof(RowIndexType_)>::SpanType;

 public:

  CSRMatrixView() = default;

  CSRMatrixView(Int32 nb_row, RowIndexType nb_non_zero, ptr_type* ptr_range, col_type* col_range, val_type* val_range)
  : m_values(val_range)
  , m_row_indexes(ptr_range)
  , m_columns(col_range)
  , m_nb_row(nb_row)
  , m_nb_non_zero(nb_non_zero)
  {
  }

 public:

  //! Number of row
  constexpr Int32 nbRow() const noexcept { return m_nb_row; }
  //! Number of non-zero in the matrix
  constexpr RowIndexType nbNonZero() const noexcept { return m_nb_non_zero; }

  //! Number of non-zero for the row \a row
  constexpr Int32 nbNonZeroForRow(Int32 row) const
  {
    ARCCORE_CHECK_AT(row, m_nb_row);
    return m_row_indexes[row + 1] - m_row_indexes[row];
  }

  SmallSpan<RowIndexType> rowIndexes() const noexcept { return { m_row_indexes, m_nb_row + 1 }; }
  ColumnSpanType columns() const noexcept { return { m_columns, m_nb_non_zero }; }
  ValueSpanType values() const noexcept { return { m_values, m_nb_non_zero }; }

 public:

  [[nodiscard]] constexpr CSRRow<RowIndexType> rowRange(Int32 row) const
  {
    auto begin = m_row_indexes[row];
    auto end = m_row_indexes[row + 1];
    return { begin, end };
  }
  //! Range of all rows of the matrix
  constexpr CSRRowRange<RowIndexType> rows() const
  {
    return CSRRowRange<RowIndexType>(m_row_indexes, 0, m_nb_row);
  }
  //! Range of rows from interval [begin,begin+size[
  constexpr CSRRowRange<RowIndexType> subrows(RowIndexType begin, RowIndexType size) const
  {
    return CSRRowRange<RowIndexType>(m_row_indexes, begin, begin + size);
  }

 public:

  constexpr ptr_type ptr(Int32 i) const
  {
    ARCCORE_CHECK_AT(i, m_nb_row + 1);
    return m_row_indexes[i];
  }
  constexpr col_type col(RowIndexType i) const
  {
    ARCCORE_CHECK_AT(i, m_nb_non_zero);
    return m_columns[i];
  }
  constexpr val_type val(RowIndexType i) const
  {
    ARCCORE_CHECK_AT(i, m_nb_non_zero);
    return m_values[i];
  }

  ptr_type& ptr(Int32 i)
  {
    ARCCORE_CHECK_AT(i, m_nb_row + 1);
    return m_row_indexes[i];
  }
  col_type& col(RowIndexType i)
  {
    ARCCORE_CHECK_AT(i, m_nb_non_zero);
    return m_columns[i];
  }
  val_type& val(RowIndexType i)
  {
    ARCCORE_CHECK_AT(i, m_nb_non_zero);
    return m_values[i];
  }

  //! Value of the matrix for the given RowColumnIndex \a rc_index
  constexpr val_type& value(CSRRowColumnIndex<RowIndexType> rc_index) const
  {
    return m_values[rc_index];
  }
  //! Value of the matrix for the given RowColumnIndex \a rc_index
  constexpr col_type column(CSRRowColumnIndex<RowIndexType> rc_index) const
  {
    return m_columns[rc_index];
  }

 private:

  val_type* m_values = nullptr;
  RowIndexType* m_row_indexes = nullptr;
  col_type* m_columns = nullptr;
  Int32 m_nb_row = 0;
  RowIndexType m_nb_non_zero = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
