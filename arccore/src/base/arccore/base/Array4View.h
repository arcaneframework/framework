// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array4View.h                                                (C) 2000-2025 */
/*                                                                           */
/* View of a 4D array.                                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAY4VIEW_H
#define ARCCORE_BASE_ARRAY4VIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Array3View.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 * \brief View for a 4D array.
 *
 * This class allows obtaining a 4D view from a contiguous memory area,
 * such as the one obtained via the Array class.
 *
 * The view can be used like a classic C array, for example:
 * \code
 * Array4View<Real> a;
 * a[0][1][2][3] = 5.0;
 * \endcode
 *
 * However, it is preferable to directly use the item() or setItem() methods
 * (or the operator()) to access an element of the array for reading or
 * writing.
 */
template <class DataType>
class Array4View
{
 public:

  //! Constructs a view
  constexpr Array4View(DataType* ptr, Integer dim1_size, Integer dim2_size,
                       Integer dim3_size, Integer dim4_size)
  : m_ptr(ptr)
  , m_dim1_size(dim1_size)
  , m_dim2_size(dim2_size)
  , m_dim3_size(dim3_size)
  , m_dim4_size(dim4_size)
  , m_dim34_size(dim3_size * dim4_size)
  , m_dim234_size(m_dim34_size * dim2_size)
  {
  }
  //! Constructs an empty view
  constexpr Array4View()
  : m_ptr(0)
  , m_dim1_size(0)
  , m_dim2_size(0)
  , m_dim3_size(0)
  , m_dim4_size(0)
  , m_dim34_size(0)
  , m_dim234_size(0)
  {
  }

 public:

  //! Value of the first dimension
  constexpr Integer dim1Size() const { return m_dim1_size; }
  //! Value of the second dimension
  constexpr Integer dim2Size() const { return m_dim2_size; }
  //! Value of the third dimension
  constexpr Integer dim3Size() const { return m_dim3_size; }
  //! Value of the fourth dimension
  constexpr Integer dim4Size() const { return m_dim4_size; }
  //! Total number of elements in the array
  constexpr Integer totalNbElement() const { return m_dim1_size * m_dim234_size; }

 public:

  constexpr Array3View<DataType> operator[](Integer i)
  {
    ARCCORE_CHECK_AT(i, m_dim1_size);
    return Array3View<DataType>(m_ptr + (m_dim234_size * i), m_dim2_size, m_dim3_size, m_dim4_size);
  }
  constexpr ConstArray3View<DataType> operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i, m_dim1_size);
    return ConstArray3View<DataType>(m_ptr + (m_dim234_size * i), m_dim2_size, m_dim3_size, m_dim4_size);
  }
  //! Value for element \a i,j,k,l
  constexpr DataType item(Integer i, Integer j, Integer k, Integer l) const
  {
    ARCCORE_CHECK_AT4(i, j, k, l, m_dim1_size, m_dim2_size, m_dim3_size, m_dim4_size);
    return m_ptr[(m_dim234_size * i) + m_dim34_size * j + m_dim4_size * k + l];
  }
  //! Value for element \a i,j,k,l
  constexpr const DataType& operator()(Integer i, Integer j, Integer k, Integer l) const
  {
    ARCCORE_CHECK_AT4(i, j, k, l, m_dim1_size, m_dim2_size, m_dim3_size, m_dim4_size);
    return m_ptr[(m_dim234_size * i) + m_dim34_size * j + m_dim4_size * k + l];
  }
  //! Value for element \a i,j,k,l
  constexpr DataType& operator()(Integer i, Integer j, Integer k, Integer l)
  {
    ARCCORE_CHECK_AT4(i, j, k, l, m_dim1_size, m_dim2_size, m_dim3_size, m_dim4_size);
    return m_ptr[(m_dim234_size * i) + m_dim34_size * j + m_dim4_size * k + l];
  }
#ifdef ARCCORE_HAS_MULTI_SUBSCRIPT
  //! Value for element \a i,j,k,l
  constexpr const DataType& operator[](Integer i, Integer j, Integer k, Integer l) const
  {
    ARCCORE_CHECK_AT4(i, j, k, l, m_dim1_size, m_dim2_size, m_dim3_size, m_dim4_size);
    return m_ptr[(m_dim234_size * i) + m_dim34_size * j + m_dim4_size * k + l];
  }
  //! Value for element \a i,j,k,l
  constexpr DataType& operator[](Integer i, Integer j, Integer k, Integer l)
  {
    ARCCORE_CHECK_AT4(i, j, k, l, m_dim1_size, m_dim2_size, m_dim3_size, m_dim4_size);
    return m_ptr[(m_dim234_size * i) + m_dim34_size * j + m_dim4_size * k + l];
  }
#endif
  //! Sets the value for element \a i,j,k,l
  constexpr void setItem(Integer i, Integer j, Integer k, Integer l, const DataType& value)
  {
    ARCCORE_CHECK_AT4(i, j, k, l, m_dim1_size, m_dim2_size, m_dim3_size, m_dim4_size);
    m_ptr[(m_dim234_size * i) + m_dim34_size * j + m_dim4_size * k + l] = value;
  }

 public:

  /*!
   * \brief Pointer to the first element of the array.
   */
  constexpr inline DataType* unguardedBasePointer() { return m_ptr; }

  /*!
   * \brief Pointer to the first element of the array.
   */
  constexpr inline DataType* data() { return m_ptr; }

 private:

  DataType* m_ptr;
  Integer m_dim1_size; //!< Size of the 1st dimension
  Integer m_dim2_size; //!< Size of the 2nd dimension
  Integer m_dim3_size; //!< Size of the 3rd dimension
  Integer m_dim4_size; //!< Size of the 4th dimension
  Integer m_dim34_size; //!< dim3 * dim4
  Integer m_dim234_size; //!< dim2 * dim3 * dim4
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 * \brief Constant view for a 4D array
 */
template <class DataType>
class ConstArray4View
{
 public:

  constexpr ConstArray4View(DataType* ptr, Integer dim1_size, Integer dim2_size,
                            Integer dim3_size, Integer dim4_size)
  : m_ptr(ptr)
  , m_dim1_size(dim1_size)
  , m_dim2_size(dim2_size)
  , m_dim3_size(dim3_size)
  , m_dim4_size(dim4_size)
  , m_dim34_size(dim3_size * dim4_size)
  , m_dim234_size(m_dim34_size * dim2_size)
  {
  }
  constexpr ConstArray4View()
  : m_ptr(nullptr)
  , m_dim1_size(0)
  , m_dim2_size(0)
  , m_dim3_size(0)
  , m_dim4_size(0)
  , m_dim34_size(0)
  , m_dim234_size(0)
  {
  }

 public:

  constexpr Integer dim1Size() const { return m_dim1_size; }
  constexpr Integer dim2Size() const { return m_dim2_size; }
  constexpr Integer dim3Size() const { return m_dim3_size; }
  constexpr Integer dim4Size() const { return m_dim4_size; }
  constexpr Integer totalNbElement() const { return m_dim1_size * m_dim234_size; }

 public:

  constexpr ConstArray3View<DataType> operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i, m_dim1_size);
    return ConstArray3View<DataType>(m_ptr + (m_dim234_size * i), m_dim2_size, m_dim3_size, m_dim4_size);
  }
  //! Value for element \a i,j,k,l
  constexpr const DataType& operator()(Integer i, Integer j, Integer k, Integer l) const
  {
    ARCCORE_CHECK_AT4(i, j, k, l, m_dim1_size, m_dim2_size, m_dim3_size, m_dim4_size);
    return m_ptr[(m_dim234_size * i) + m_dim34_size * j + m_dim4_size * k + l];
  }
#ifdef ARCCORE_HAS_MULTI_SUBSCRIPT
  //! Value for element \a i,j,k,l
  constexpr const DataType& operator[](Integer i, Integer j, Integer k, Integer l) const
  {
    ARCCORE_CHECK_AT4(i, j, k, l, m_dim1_size, m_dim2_size, m_dim3_size, m_dim4_size);
    return m_ptr[(m_dim234_size * i) + m_dim34_size * j + m_dim4_size * k + l];
  }
#endif
  constexpr DataType item(Integer i, Integer j, Integer k, Integer l) const
  {
    ARCCORE_CHECK_AT4(i, j, k, l, m_dim1_size, m_dim2_size, m_dim3_size, m_dim4_size);
    return m_ptr[(m_dim234_size * i) + m_dim34_size * j + m_dim4_size * k + l];
  }

 public:

  //! Pointer to the allocated memory.
  constexpr inline const DataType* unguardedBasePointer() { return m_ptr; }

  //! Pointer to the allocated memory.
  constexpr inline const DataType* data() { return m_ptr; }

 private:

  const DataType* m_ptr;
  Integer m_dim1_size; //!< Size of the 1st dimension
  Integer m_dim2_size; //!< Size of the 2nd dimension
  Integer m_dim3_size; //!< Size of the 3rd dimension
  Integer m_dim4_size; //!< Size of the 4th dimension
  Integer m_dim34_size; //!< dim3 * dim4
  Integer m_dim234_size; //!< dim2 * dim3 * dim4
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
