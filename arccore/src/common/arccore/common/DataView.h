// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataView.h                                                  (C) 2000-2026 */
/*                                                                           */
/* Views of variable data.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_DATAVIEW_H
#define ARCCORE_COMMON_DATAVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file DataView.h
 *
 * This file contains the type declarations for managing
 * views for accelerators.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{
class AtomicImpl;
}
namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Class for accessing an element of a read view.
 */
template <typename DataType>
class DataViewGetter
{
 public:

  using ValueType = const DataType;
  using AccessorReturnType = const DataType;
  static ARCCORE_HOST_DEVICE AccessorReturnType build(const DataType* ptr)
  {
    return { *ptr };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Class for accessing an element of a write view.
 */
template <typename DataType>
class DataViewSetter
{
  // To access m_ptr;
  friend class DataViewGetterSetter<DataType>;

 public:

  using ValueType = DataType;
  using AccessorReturnType = DataViewSetter<DataType>;

 public:

  explicit ARCCORE_HOST_DEVICE DataViewSetter(DataType* ptr)
  : m_ptr(ptr)
  {}
  ARCCORE_HOST_DEVICE DataViewSetter(const DataViewSetter<DataType>& v)
  : m_ptr(v.m_ptr)
  {}
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator=(const DataType& v)
  {
    *m_ptr = v;
    return (*this);
  }
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator=(const DataViewSetter<DataType>& v)
  {
    // Attention: you must update the value and not the pointer
    // otherwise the code like a = b with 'a' and 'b' two instances of this
    // class will not work.
    *m_ptr = *(v.m_ptr);
    return (*this);
  }
  static ARCCORE_HOST_DEVICE AccessorReturnType build(DataType* ptr)
  {
    return AccessorReturnType(ptr);
  }

 public:

  // Binary arithmetic operators
  // +=
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator+=(const DataType& v)
  {
    *m_ptr = (*m_ptr) + v;
    return (*this);
  }
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator+=(const DataViewSetter<DataType>& v)
  {
    *m_ptr = (*m_ptr) + *(v.m_ptr);
    return (*this);
  }

  // -=
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator-=(const DataType& v)
  {
    *m_ptr = (*m_ptr) - v;
    return (*this);
  }
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator-=(const DataViewSetter<DataType>& v)
  {
    *m_ptr = (*m_ptr) - *(v.m_ptr);
    return (*this);
  }

  // *=
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator*=(const DataType& v)
  {
    *m_ptr = (*m_ptr) * v;
    return (*this);
  }
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator*=(const DataViewSetter<DataType>& v)
  {
    *m_ptr = (*m_ptr) * *(v.m_ptr);
    return (*this);
  }

  // /=
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator/=(const DataType& v)
  {
    *m_ptr = (*m_ptr) / v;
    return (*this);
  }
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator/=(const DataViewSetter<DataType>& v)
  {
    *m_ptr = (*m_ptr) / *(v.m_ptr);
    return (*this);
  }

 public:

  template <typename X = DataType, typename ComponentDataType = decltype(X::x)>
  ARCCORE_HOST_DEVICE void setX(ComponentDataType value)
  {
    m_ptr->x = value;
  }
  template <typename X = DataType, typename ComponentDataType = decltype(X::y)>
  ARCCORE_HOST_DEVICE void setY(ComponentDataType value)
  {
    m_ptr->y = value;
  }
  template <typename X = DataType, typename ComponentDataType = decltype(X::z)>
  ARCCORE_HOST_DEVICE void setZ(ComponentDataType value)
  {
    m_ptr->z = value;
  }

  ARCCORE_HOST_DEVICE void setXX(Real value) requires(requires() { DataType::x.x; })
  {
    m_ptr->x.x = value;
  }
  ARCCORE_HOST_DEVICE void setYX(Real value) requires(requires() { DataType::y.x; })
  {
    m_ptr->y.x = value;
  }
  ARCCORE_HOST_DEVICE void setZX(Real value) requires(requires() { DataType::z.x; })
  {
    m_ptr->z.x = value;
  }

  ARCCORE_HOST_DEVICE void setXY(Real value) requires(requires() { DataType::x.y; })
  {
    m_ptr->x.y = value;
  }
  ARCCORE_HOST_DEVICE void setYY(Real value) requires(requires() { DataType::y.y; })
  {
    m_ptr->y.y = value;
  }
  ARCCORE_HOST_DEVICE void setZY(Real value) requires(requires() { DataType::z.y; })
  {
    m_ptr->z.y = value;
  }

  ARCCORE_HOST_DEVICE void setXZ(Real value) requires(requires() { DataType::x.z; })
  {
    m_ptr->x.z = value;
  }
  ARCCORE_HOST_DEVICE void setYZ(Real value) requires(requires() { DataType::y.z; })
  {
    m_ptr->y.z = value;
  }
  ARCCORE_HOST_DEVICE void setZZ(Real value) requires(requires() { DataType::z.z; })
  {
    m_ptr->z.z = value;
  }

  /*!
   * \brief Applies the operator operator[] on the type.
   *
   * The operation is only valid if DataType::operator[](Int32) exists.
   * \return a DataViewSetter on the returned value of operator()(Int32).
   */
  ARCCORE_HOST_DEVICE auto operator[](Int32 index)
  requires(requires() { std::declval<const DataType>()(0); })
  {
    using SubscriptType = decltype(std::declval<const DataType>()[0]);
    return DataViewSetter<SubscriptType>(&m_ptr->operator[](index));
  }

 private:

  DataType* m_ptr = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Class for accessing an element of a read/write view.
 *
 * This class extends the functionalities of DataViewSetter by adding
 * the possibility to access the data value.
 */
template <typename DataType>
class DataViewGetterSetter
: public DataViewSetter<DataType>
{
  using BaseType = DataViewSetter<DataType>;
  using BaseType::m_ptr;
  friend class Arcane::Accelerator::Impl::AtomicImpl;

  // Add friend for specific views which need access to m_ptr.
  template <typename DataType_, int Row, int Column>
  friend class NumMatrixDataViewGetterSetter;

 public:

  using ValueType = DataType;
  using AccessorReturnType = DataViewGetterSetter<DataType>;

 public:

  explicit ARCCORE_HOST_DEVICE DataViewGetterSetter(DataType* ptr)
  : BaseType(ptr)
  {}
  ARCCORE_HOST_DEVICE DataViewGetterSetter(const DataViewGetterSetter& v)
  : BaseType(v)
  {}
  //! Operator to convert to the underlying type for read-only operation
  ARCCORE_HOST_DEVICE operator DataType() const
  {
    return *m_ptr;
  }
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator=(const DataViewGetterSetter<DataType>& v)
  {
    BaseType::operator=(v);
    return (*this);
  }
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator=(const DataType& v)
  {
    BaseType::operator=(v);
    return (*this);
  }
  static ARCCORE_HOST_DEVICE AccessorReturnType build(DataType* ptr)
  {
    return AccessorReturnType(ptr);
  }

  /*!
   * \brief Applies, if it exists, the operator operator[](Int32) on the type.
   *
   * \return a DataViewGetterSetter on the returned value of operator[](Int32).
   */
  ARCCORE_HOST_DEVICE auto operator[](Int32 index)
  requires(requires() { std::declval<const DataType>()[0]; })
  {
    using DataTypeReturnType = decltype(std::declval<const DataType>()[0]);
    return DataViewGetterSetter<DataTypeReturnType>(&m_ptr->operator[](index));
  }

  /*!
   * \brief Applies, if it exists, the operator operator()(Int32) on the type.
   *
   * \return a DataViewGetterSetter on the returned value of operator()(Int32).
   */
  constexpr ARCCORE_HOST_DEVICE auto operator()(Int32 i0)
  requires(requires() { std::declval<const DataType>()(0); })
  {
    using DataTypeReturnType = decltype(std::declval<const DataType>()(0));
    return DataViewGetterSetter<DataTypeReturnType>(&m_ptr->operator()(i0));
  }

  /*!
   * \brief Applies, if it exists, the operator operator()(Int32,Int32) on DataType.
   *
   * \return a DataViewGetterSetter on the returned value of operator()(Int32,Int32).
   */
  constexpr ARCCORE_HOST_DEVICE auto operator()(Int32 i0, Int32 i1)
  requires(requires() { std::declval<const DataType>()(0, 0); })
  {
    using DataTypeReturnType = decltype(std::declval<const DataType>()(0, 0));
    return DataViewGetterSetter<DataTypeReturnType>(&m_ptr->operator()(i0, i1));
  }
  friend std::ostream& operator<<(std::ostream& o, const DataViewGetterSetter& v)
  {
    o << *(v.m_ptr);
    return o;
  }

 private:

  //! Address of the data. Valid only for simple types (i.e. not Real3)
  constexpr ARCCORE_HOST_DEVICE DataType* _address() const { return m_ptr; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
