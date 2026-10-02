// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumVectorDataView.h                                         (C) 2000-2026 */
/*                                                                           */
/* Specific implementation of DataView(Setter/Getter) for NumVector.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_NUMVECTORDATAVIEW_H
#define ARCCORE_COMMON_NUMVECTORDATAVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/NumVector.h"

#include "arccore/common/DataView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Read-only view for a NumVector<DataType_, Size_>.
 */
template <typename DataType_, int Size_>
class NumVectorDataViewGetter
{
 public:

  //! Number of elements of the vector
  static constexpr int Size = Size_;
  //! Type of the vector
  using NumVectorType = NumVector<DataType_, Size>;
  //! Accessor for the vector
  using AccessorReturnType = const NumVectorType&;
  //! Accessor for one element of the vector
  using VectorElemenAccessor = DataViewGetter<DataType_>;

 public:

  explicit ARCCORE_HOST_DEVICE NumVectorDataViewGetter(const NumVectorType* ptr)
  : m_ptr(ptr)
  {}

 public:

  static constexpr ARCCORE_HOST_DEVICE AccessorReturnType build(const NumVectorType* ptr)
  {
    return { *ptr };
  }

 public:

  constexpr operator AccessorReturnType() const noexcept { return *m_ptr; }

 private:

  const NumVectorType* m_ptr = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Mutable view for a NumVector<DataType_,Size>.
 */
template <typename DataType_, int Size_>
class NumVectorDataViewGetterSetter
: public DataViewGetterSetter<NumVector<DataType_, Size_>>
{
  using BaseClass = DataViewGetterSetter<NumVector<DataType_, Size_>>;

 public:

  //! Number of elements of the vector
  static constexpr int Size = Size_;
  //! Type of the vector
  using NumVectorType = NumVector<DataType_, Size>;
  //! Accessor for one element of the vector
  using VectorElemenAccessor = DataViewGetterSetter<DataType_>;

 public:

  explicit ARCCORE_HOST_DEVICE NumVectorDataViewGetterSetter(NumVectorType* ptr)
  : BaseClass(ptr)
  {}

 public:

  NumVectorDataViewGetterSetter& operator=(const NumVectorType& v)
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
