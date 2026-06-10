// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array3View.h                                                (C) 2000-2025 */
/*                                                                           */
/* View of a 3D array.                                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAY3VIEW_H
#define ARCCORE_BASE_ARRAY3VIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Array2View.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 * \brief View for a 3D array.
 *
 * A 3D view can be created from a classic array (Array)
 * as follows:
 \code
 UniqueArray<Int32> a(5*7*9);
 Array3View<Int32> view(a.unguardedBasePointer(),5,7,9);
 view[3][4][5] = 2;
 view.setItem(4,5,6, 1); // Sets the value 1 to the element view[4][5][6].
 \endcode
 For performance reasons, it is preferable to access elements
 via operator()()
 */
template <class DataType>
class Array3View
{
 public:

  constexpr Array3View(DataType* ptr, Integer dim1_size, Integer dim2_size, Integer dim3_size)
  : m_ptr(ptr)
  , m_dim1_size(dim1_size)
  , m_dim2_size(dim2_size)
  , m_dim3_size(dim3_size)
  , m_dim23_size(dim2_size * dim3_size)
  {
  }
  constexpr Array3View()
  : m_ptr(0)
  , m_dim1_size(0)
  , m_dim2_size(0)
  , m_dim3_size(0)
  , m_dim23_size(0)
  {
  }

 public:

  constexpr Integer dim1Size() const { return m_dim1_size; }
  constexpr Integer dim2Size() const { return m_dim2_size; }
  constexpr Integer dim3Size() const { return m_dim3_size; }
  constexpr Integer totalNbElement() const { return m_dim1_size * m_dim23_size; }

 public:

  constexpr Array2View<DataType> operator[](Integer i)
  {
    ARCCORE_CHECK_AT(i, m_dim1_size);
    return Array2View<DataType>(m_ptr + (m_dim23_size * i), m_dim2_size, m_dim3_size);
  }
  constexpr ConstArray2View<DataType> operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i, m_dim1_size);
    return ConstArray2View<DataType>(m_ptr + (m_dim23_size * i), m_dim2_size, m_dim3_size);
  }
  constexpr DataType item(Integer i, Integer j, Integer k) const
  {
    ARCCORE_CHECK_AT3(i, j, k, m_dim1_size, m_dim2_size, m_dim3_size);
    return m_ptr[(m_dim23_size * i) + m_dim3_size * j + k];
  }
  constexpr const DataType& operator()(Integer i, Integer j, Integer k) const
  {
    ARCCORE_CHECK_AT3(i, j, k, m_dim1_size, m_dim2_size, m_dim3_size);
    return m_ptr[(m_dim23_size * i) + m_dim3_size * j + k];
  }
  constexpr DataType& operator()(Integer i, Integer j, Integer k)
  {
    ARCCORE_CHECK_AT3(i, j, k, m_dim1_size, m_dim2_size, m_dim3_size);
    return m_ptr[(m_dim23_size * i) + m_dim3_size * j + k];
  }
#ifdef ARCCORE_HAS_MULTI_SUBSCRIPT
  constexpr const DataType& operator[](Integer i, Integer j, Integer k) const
  {
    ARCCORE_CHECK_AT3(i, j, k, m_dim1_size, m_dim2_size, m_dim3_size);
    return m_ptr[(m_dim23_size * i) + m_dim3_size * j + k];
  }
  constexpr DataType& operator[](Integer i, Integer j, Integer k)
  {
    ARCCORE_CHECK_AT3(i, j, k, m_dim1_size, m_dim2_size, m_dim3_size);
    return m_ptr[(m_dim23_size * i) + m_dim3_size * j + k];
  }
#endif
  constexpr DataType setItem(Integer i, Integer j, Integer k, const DataType& value)
  {
    ARCCORE_CHECK_AT3(i, j, k, m_dim1_size, m_dim2_size, m_dim3_size);
    m_ptr[(m_dim23_size * i) + m_dim3_size * j + k] = value;
  }

 public:

  /*!
   * \brief Pointer to the allocated memory.
   */
  inline DataType* unguardedBasePointer()
  {
    return m_ptr;
  }

 private:

  DataType* m_ptr;
  Integer m_dim1_size; //!< Size of the 1st dimension
  Integer m_dim2_size; //!< Size of the 2nd dimension
  Integer m_dim3_size; //!< Size of the 3rd dimension
  Integer m_dim23_size; //!< dim2 * dim3
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 * \brief View for a constant 3D array.
 */
template <class DataType>
class ConstArray3View
{
 public:

  constexpr ConstArray3View(const DataType* ptr, Integer dim1_size, Integer dim2_size, Integer dim3_size)
  : m_ptr(ptr)
  , m_dim1_size(dim1_size)
  , m_dim2_size(dim2_size)
  , m_dim3_size(dim3_size)
  , m_dim23_size(dim2_size * dim3_size)
  {
  }
  constexpr ConstArray3View()
  : m_ptr(0)
  , m_dim1_size(0)
  , m_dim2_size(0)
  , m_dim3_size(0)
  , m_dim23_size(0)
  {
  }

 public:

  constexpr Integer dim1Size() const { return m_dim1_size; }
  constexpr Integer dim2Size() const { return m_dim2_size; }
  constexpr Integer dim3Size() const { return m_dim3_size; }
  constexpr Integer totalNbElement() const { return m_dim1_size * m_dim23_size; }

 public:

  constexpr ConstArray2View<DataType> operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i, m_dim1_size);
    return ConstArray2View<DataType>(m_ptr + (m_dim23_size * i), m_dim2_size, m_dim3_size);
  }
  constexpr DataType item(Integer i, Integer j, Integer k) const
  {
    ARCCORE_CHECK_AT3(i, j, k, m_dim1_size, m_dim2_size, m_dim3_size);
    return m_ptr[(m_dim23_size * i) + m_dim3_size * j + k];
  }
  constexpr const DataType& operator()(Integer i, Integer j, Integer k) const
  {
    ARCCORE_CHECK_AT3(i, j, k, m_dim1_size, m_dim2_size, m_dim3_size);
    return m_ptr[(m_dim23_size * i) + m_dim3_size * j + k];
  }
#ifdef ARCCORE_HAS_MULTI_SUBSCRIPT
  constexpr const DataType& operator[](Integer i, Integer j, Integer k) const
  {
    ARCCORE_CHECK_AT3(i, j, k, m_dim1_size, m_dim2_size, m_dim3_size);
    return m_ptr[(m_dim23_size * i) + m_dim3_size * j + k];
  }
#endif

 public:

  //! Pointer to the allocated memory.
  constexpr inline const DataType* unguardedBasePointer() const { return m_ptr; }

  //! Pointer to the allocated memory.
  constexpr const DataType* data() const { return m_ptr; }

 private:

  const DataType* m_ptr;
  Integer m_dim1_size; //!< Size of the 1st dimension
  Integer m_dim2_size; //!< Size of the 2nd dimension
  Integer m_dim3_size; //!< Size of the 3rd dimension
  Integer m_dim23_size; //!< dim2 * dim3
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
