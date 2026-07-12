// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumMatrixDataView.h                                         (C) 2000-2026 */
/*                                                                           */
/* Specific implementation of DataView(Setter/Getter) for NumMatrix.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_NUMMATRIXDATAVIEW_H
#define ARCANE_UTILS_NUMMATRIXDATAVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumMatrix.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Read-only view for a NumMatrix<DataType_,Row,Column>.
 */
template <typename DataType_, int Row, int Column>
class NumMatrixDataViewGetter
{
 public:

  //! Type of the matrix
  using NumMatrixType = NumMatrix<DataType_, Row, Column>;
  //! Accessor for the matrix
  using AccessorReturnType = const NumMatrixType&;
  //! Accessor for one element of the matrix
  using MatrixElemenAccessor = DataViewGetter<DataType_>;

 public:

  explicit ARCCORE_HOST_DEVICE NumMatrixDataViewGetter(const NumMatrixType* ptr)
  : m_ptr(ptr)
  {}

 public:

  static constexpr ARCCORE_HOST_DEVICE AccessorReturnType build(const NumMatrixType* ptr)
  {
    return { *ptr };
  }

 public:

  constexpr operator AccessorReturnType() const noexcept { return *m_ptr; }

 private:

  const NumMatrixType* m_ptr = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Mutable view for a NumMatrix<DataType_,Row,Column>.
 */
template <typename DataType_, int Row, int Column>
class NumMatrixDataViewGetterSetter
: public DataViewGetterSetter<NumMatrix<DataType_, Row, Column>>
{
  using BaseClass = DataViewGetterSetter<NumMatrix<DataType_, Row, Column>>;

 public:

  //! Type of the matrix
  using NumMatrixType = NumMatrix<DataType_, Row, Column>;
  //! Accessor for one element of the matrix
  using MatrixElemenAccessor = DataViewGetterSetter<DataType_>;

 public:

  explicit ARCCORE_HOST_DEVICE NumMatrixDataViewGetterSetter(NumMatrixType* ptr)
  : BaseClass(ptr)
  {}

 public:

  NumMatrixDataViewGetterSetter& operator=(const NumMatrixType& v)
  {
    BaseClass::operator=(v);
    return (*this);
  }

  void fill(const DataType_& v)
  {
    this->m_ptr->fill(v);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
