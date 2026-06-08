// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array2View.h                                                (C) 2000-2025 */
/*                                                                           */
/* View of a 2D array.                                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAY2VIEW_H
#define ARCCORE_BASE_ARRAY2VIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"
#include "arccore/base/TraceInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 *
 * \brief Mutable view for a 2D array.
 *
 * Like any view, an instance of this class is only valid as long as
 * the container it originates from does not change its number of elements.
 */
template<class DataType>
class Array2View
{
  friend class SmallSpan2<DataType>;
  friend class SmallSpan2<const DataType>;
  friend class Span2<DataType>;
  friend class Span2<const DataType>;
 public:
  friend class ConstArray2View<DataType>;
 public:
  //! Creates a 2D view of dimension [\a dim1_size][\a dim2_size]
  constexpr Array2View(DataType* ptr,Integer dim1_size,Integer dim2_size)
  : m_ptr(ptr), m_dim1_size(dim1_size), m_dim2_size(dim2_size) {}
  //! Creates an empty 2D view.
  constexpr Array2View() : m_ptr(nullptr), m_dim1_size(0), m_dim2_size(0) {}
 public:
  //! Number of elements in the first dimension
  constexpr Integer dim1Size() const { return m_dim1_size; }
  //! Number of elements in the second dimension
  constexpr Integer dim2Size() const { return m_dim2_size; }
  //! Total number of elements.
  constexpr Integer totalNbElement() const { return m_dim1_size*m_dim2_size; }
 public:
  constexpr ArrayView<DataType> operator[](Integer i)
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    return ArrayView<DataType>(m_dim2_size,m_ptr + (m_dim2_size*i));
  }
  constexpr ConstArrayView<DataType> operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    return ConstArrayView<DataType>(m_dim2_size,m_ptr + (m_dim2_size*i));
  }
  //! Value of element [\a i][\a j]
  constexpr DataType item(Integer i,Integer j) const
  {
    ARCCORE_CHECK_AT2(i,j,m_dim1_size,m_dim2_size);
    return m_ptr[(m_dim2_size*i) + j];
  }
  //! Positions element [\a i][\a j] to \a value
  constexpr DataType setItem(Integer i,Integer j,const DataType& value)
  {
    ARCCORE_CHECK_AT2(i,j,m_dim1_size,m_dim2_size);
    m_ptr[(m_dim2_size*i) + j] = value;
  }
  //! Value of element [\a i][\a j]
  constexpr const DataType operator()(Integer i,Integer j) const
  {
    ARCCORE_CHECK_AT2(i,j,m_dim1_size,m_dim2_size);
    return m_ptr[(m_dim2_size*i) + j];
  }
  //! Value of element [\a i][\a j]
  constexpr DataType& operator()(Integer i,Integer j)
  {
    ARCCORE_CHECK_AT2(i,j,m_dim1_size,m_dim2_size);
    return m_ptr[(m_dim2_size*i) + j];
  }
#ifdef ARCCORE_HAS_MULTI_SUBSCRIPT
  //! Value of element [\a i][\a j]
  constexpr const DataType operator[](Integer i,Integer j) const
  {
    ARCCORE_CHECK_AT2(i,j,m_dim1_size,m_dim2_size);
    return m_ptr[(m_dim2_size*i) + j];
  }
  //! Value of element [\a i][\a j]
  constexpr DataType& operator[](Integer i,Integer j)
  {
    ARCCORE_CHECK_AT2(i,j,m_dim1_size,m_dim2_size);
    return m_ptr[(m_dim2_size*i) + j];
  }
#endif

 public:

  /*!
   * \brief Pointer to the allocated memory.
   */
  constexpr inline DataType* unguardedBasePointer()
  { return m_ptr; }

  /*!
   * \brief Pointer to the allocated memory.
   */
  constexpr inline DataType* data() { return m_ptr; }
 private:
  DataType* m_ptr;
  Integer m_dim1_size;
  Integer m_dim2_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 * \brief View for a constant 2D array.
 */
template<class DataType>
class ConstArray2View
{
  friend class SmallSpan2<const DataType>;
  friend class Span2<const DataType>;
 public:
  constexpr ConstArray2View(const DataType* ptr,Integer dim1_size,Integer dim2_size)
  : m_ptr(ptr), m_dim1_size(dim1_size), m_dim2_size(dim2_size)
  {
  }
  constexpr ConstArray2View(const Array2View<DataType>& rhs)
  : m_ptr(rhs.m_ptr), m_dim1_size(rhs.m_dim1_size), m_dim2_size(rhs.m_dim2_size)
  {
  }
  constexpr ConstArray2View() : m_ptr(0), m_dim1_size(0), m_dim2_size(0)
  {
  }
 public:
  constexpr Integer dim1Size() const { return m_dim1_size; }
  constexpr Integer dim2Size() const { return m_dim2_size; }
  constexpr Integer totalNbElement() const { return m_dim1_size*m_dim2_size; }
 public:
  constexpr ConstArrayView<DataType> operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i,m_dim1_size);
    return ConstArrayView<DataType>(m_dim2_size,m_ptr + (m_dim2_size*i));
  }
  constexpr DataType item(Integer i,Integer j) const
  {
    ARCCORE_CHECK_AT2(i,j,m_dim1_size,m_dim2_size);
    return m_ptr[(m_dim2_size*i) + j];
  } 
  //! Value of element [\a i][\a j]
  constexpr const DataType operator()(Integer i,Integer j) const
  {
    ARCCORE_CHECK_AT2(i,j,m_dim1_size,m_dim2_size);
    return m_ptr[(m_dim2_size*i) + j];
  }
#ifdef ARCCORE_HAS_MULTI_SUBSCRIPT
  //! Value of element [\a i][\a j]
  constexpr const DataType operator[](Integer i,Integer j) const
  {
    ARCCORE_CHECK_AT2(i,j,m_dim1_size,m_dim2_size);
    return m_ptr[(m_dim2_size*i) + j];
  }
#endif

 public:

  //! Pointer to the allocated memory.
  constexpr const DataType* unguardedBasePointer() const { return m_ptr; }

  //! Pointer to the allocated memory.
  constexpr const DataType* data() const { return m_ptr; }

 private:

  const DataType* m_ptr;
  Integer m_dim1_size;
  Integer m_dim2_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
