// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MultiArray2View.h                                           (C) 2000-2025 */
/*                                                                           */
/* View of a 2D array with multiple sizes.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MULTIARRAY2VIEW_H
#define ARCANE_UTILS_MULTIARRAY2VIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/ArrayView.h"

#include <type_traits>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Modifiable view on a MultiArray2.
 */
template <class DataType>
class MultiArray2View
{
 public:

  //! View on the array \a buf
  MultiArray2View(ArrayView<DataType> buf, ConstArrayView<Int32> indexes, ConstArrayView<Int32> sizes)
  : m_buffer(buf)
  , m_indexes(indexes)
  , m_sizes(sizes)
  {}
  //! Empty view
  MultiArray2View() = default;

 public:

  //! Number of elements in the first dimension.
  Int32 dim1Size() const { return m_sizes.size(); }

  /*!
   * \brief Number of elements in the first dimension.
   * \deprecated Use dim1Size() instead.
   */
  ARCANE_DEPRECATED_122 Int32 size() const { return dim1Size(); }

  //! Number of elements in the second dimension
  ConstArrayView<Int32> dim2Sizes() const { return m_sizes; }

  //! Total number of elements in the array.
  Int32 totalNbElement() const { return m_buffer.size(); }

 public:

  //! The i-th element of the array
  ArrayView<DataType> operator[](Int32 i)
  {
    return ArrayView<DataType>(this->m_sizes[i], &this->m_buffer[this->m_indexes[i]]);
  }
  //! The i-th element of the array
  ConstArrayView<DataType> operator[](Int32 i) const
  {
    return ConstArrayView<DataType>(this->m_sizes[i], this->m_buffer.data() + (this->m_indexes[i]));
  }

 private:

  ArrayView<DataType> m_buffer;
  ConstArrayView<Int32> m_indexes;
  ConstArrayView<Int32> m_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Constant view on a MultiArray2.
 */
template <class DataType>
class ConstMultiArray2View
{
 private:

  friend class MultiArray2<DataType>;

 public:

  //! View on the array \a buf
  ConstMultiArray2View(ConstArrayView<DataType> buf, ConstArrayView<Int32> indexes,
                       ConstArrayView<Int32> sizes)
  : m_buffer(buf)
  , m_indexes(indexes)
  , m_sizes(sizes)
  {}

  //! Empty view
  ConstMultiArray2View() = default;

 public:

  //! Number of elements in the first dimension.
  Int32 dim1Size() const { return m_sizes.size(); }

  /*!
   * \brief Number of elements in the first dimension.
   * \deprecated Use dim1Size() instead.
   */
  ARCANE_DEPRECATED_122 Int32 size() const { return dim1Size(); }

  //! Number of elements in the second dimension
  ConstArrayView<Int32> dim2Sizes() const { return m_sizes; }

  //! Total number of elements in the array.
  Int32 totalNbElement() const { return m_buffer.size(); }

 public:

  //! The i-th element of the array
  ConstArrayView<DataType> operator[](Int32 i) const
  {
    return ConstArrayView<DataType>(this->m_sizes[i], this->m_buffer.data() + (this->m_indexes[i]));
  }

 private:

  ConstArrayView<DataType> m_buffer;
  ConstArrayView<Int32> m_indexes;
  ConstArrayView<Int32> m_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief View on a MultiArray2.
 *
 * Instances of this class are created by calling MultiArray2::span()
 * or MultiArray2::constSpan().
 */
template <class DataType>
class JaggedSmallSpan
{
 private:

  friend class MultiArray2<std::remove_cv_t<DataType>>;

 public:

  //! Empty view
  JaggedSmallSpan() = default;

 private:

  //! View on the array \a buf
  JaggedSmallSpan(SmallSpan<DataType> buf, SmallSpan<const Int32> indexes,
                  SmallSpan<const Int32> sizes)
  : m_buffer(buf)
  , m_indexes(indexes)
  , m_sizes(sizes)
  {}

 public:

  //! Number of elements in the first dimension.
  constexpr ARCCORE_HOST_DEVICE Int32 dim1Size() const { return m_sizes.size(); }

  //! Number of elements in the second dimension
  constexpr ARCCORE_HOST_DEVICE SmallSpan<const Int32> dim2Sizes() const { return m_sizes; }

  //! Total number of elements in the array.
  constexpr ARCCORE_HOST_DEVICE Int32 totalNbElement() const { return m_buffer.size(); }

 public:

  //! The i-th element of the array
  constexpr ARCCORE_HOST_DEVICE SmallSpan<DataType> operator[](Int32 i) const
  {
    return m_buffer.subSpan(m_indexes[i], m_sizes[i]);
  }

 private:

  SmallSpan<DataType> m_buffer;
  SmallSpan<const Int32> m_indexes;
  SmallSpan<const Int32> m_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
