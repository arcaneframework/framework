// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MultiArray2.h                                               (C) 2000-2025 */
/*                                                                           */
/* Multi-sized 2D Array.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MULTIARRAY2_H
#define ARCANE_UTILS_MULTIARRAY2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/MultiArray2View.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Collection
 * \brief Base class for multi-sized 2D arrays.
 *
 * This class manages 2D arrays where the number of elements in the
 * second dimension is variable.
 * For example:
 * \code
 *  UniqueArray<Int32> sizes(3); // Array with 3 elements
 *  sizes[0] = 1; sizes[1] = 2; sizes[2] = 4;
 *  // Constructs the array with sizes as dimensions
 *  MultiArray2<Int32> v(sizes);
 *  info() << " size=" << v.dim1Size(); // displays 3
 *  info() << " size[0]=" << v[0].size(); // displays 1
 *  info() << " size[1]=" << v[1].size(); // displays 2
 *  info() << " size[2]=" << v[2].size(); // displays 4
 * \endcode
 *
 * \note Indices are stored using the Int32 type.
 * The total number of elements in the array is therefore limited to 2^31
 *
 * It is possible to resize (via the resize() method) the
 * array while keeping its values, but for performance reasons, these
 * resizes apply to the entire array (it is not possible
 * to resize only a single element, for example v[5].resize(3)).
 * 
 * Like Array and Array2, instances of this class are not copyable or assignable. To get this functionality, you must
 * use the SharedMultiArray2 class for reference semantics
 * or UniqueMultiArray2 for value semantics.
 */
template <typename DataType>
class MultiArray2
{
 public:

  using ConstReferenceType = typename UniqueArray<DataType>::ConstReferenceType;
  using ThatClass = MultiArray2<DataType>;

 public:

  MultiArray2() = default;
  // TODO: Make accessible only to UniqueMultiArray2 or SharedMultiArray2
  explicit MultiArray2(ConstArrayView<Int32> sizes)
  {
    _resize(sizes);
  }

 public:

  MultiArray2(const ThatClass& rhs) = delete;
  ThatClass& operator=(const ThatClass& rhs) = delete;

 protected:

  /*!
   * \brief Copy constructor.
   * Temporary method to be removed once the copy constructor and copy operator
   * are deleted.
   */
  MultiArray2(const MultiArray2<DataType>& rhs, bool do_clone)
  : m_buffer(do_clone ? rhs.m_buffer.clone() : rhs.m_buffer)
  , m_indexes(do_clone ? rhs.m_indexes.clone() : rhs.m_indexes)
  , m_sizes(do_clone ? rhs.m_sizes.clone() : rhs.m_sizes)
  {
  }
  explicit MultiArray2(ConstMultiArray2View<DataType> aview)
  : m_buffer(aview.m_buffer)
  , m_indexes(aview.m_indexes)
  , m_sizes(aview.m_sizes)
  {
  }
  explicit MultiArray2(const MemoryAllocationOptions& allocation_options)
  : m_buffer(allocation_options)
  , m_indexes(allocation_options)
  , m_sizes(allocation_options)
  {}
  // TODO: Make accessible only to UniqueMultiArray2 or SharedMultiArray2
  MultiArray2(const MemoryAllocationOptions& allocation_options, ConstArrayView<Int32> sizes)
  : MultiArray2(allocation_options)
  {
    _resize(sizes);
  }

 public:

  ArrayView<DataType> operator[](Integer i)
  {
    return ArrayView<DataType>(m_sizes[i], m_buffer.data() + (m_indexes[i]));
  }
  ConstArrayView<DataType> operator[](Integer i) const
  {
    return ConstArrayView<DataType>(m_sizes[i], m_buffer.data() + (m_indexes[i]));
  }

 public:

  //! Total number of elements
  Int32 totalNbElement() const { return m_buffer.size(); }

  //! Clears the array elements.
  void clear()
  {
    m_buffer.clear();
    m_indexes.clear();
    m_sizes.clear();
  }
  //! Fills the array elements with the value \a v
  void fill(const DataType& v)
  {
    m_buffer.fill(v);
  }
  DataType& at(Integer i, Integer j)
  {
    return m_buffer[m_indexes[i] + j];
  }
  ConstReferenceType at(Integer i, Integer j) const
  {
    return m_buffer[m_indexes[i] + j];
  }
  void setAt(Integer i, Integer j, ConstReferenceType v)
  {
    return m_buffer.setAt(m_indexes[i] + j, v);
  }

 public:

  //! Number of elements following the first dimension
  Int32 dim1Size() const { return m_indexes.size(); }

  //! Array of the number of elements following the second dimension
  ConstArrayView<Int32> dim2Sizes() const { return m_sizes; }

  //! Conversion operator to a mutable view
  operator MultiArray2View<DataType>()
  {
    return view();
  }

  //! Conversion operator to a constant view.
  operator ConstMultiArray2View<DataType>() const
  {
    return constView();
  }

  //! Mutable view of the array
  MultiArray2View<DataType> view()
  {
    return MultiArray2View<DataType>(m_buffer, m_indexes, m_sizes);
  }

  //! Constant view of the array
  ConstMultiArray2View<DataType> constView() const
  {
    return ConstMultiArray2View<DataType>(m_buffer, m_indexes, m_sizes);
  }

  //! Mutable view of the array
  JaggedSmallSpan<DataType> span()
  {
    return { m_buffer.smallSpan(), m_indexes, m_sizes };
  }

  //! Constant view of the array
  JaggedSmallSpan<const DataType> span() const
  {
    return { m_buffer, m_indexes, m_sizes };
  }

  //! Constant view of the array
  JaggedSmallSpan<const DataType> constSpan() const
  {
    return { m_buffer.constSmallSpan(), m_indexes, m_sizes };
  }

  //! View of the array as a 1D array
  ArrayView<DataType> viewAsArray()
  {
    return m_buffer.view();
  }

  //! View of the array as a 1D array
  ConstArrayView<DataType> viewAsArray() const
  {
    return m_buffer.constView();
  }

  //! Resizes the array with new sizes \a new_sizes
  void resize(ConstArrayView<Int32> new_sizes)
  {
    if (new_sizes.empty()) {
      clear();
    }
    else
      _resize(new_sizes);
  }

 protected:

  ConstArrayView<DataType> _value(Integer i) const
  {
    return ConstArrayView<DataType>(m_sizes[i], m_buffer.data() + m_indexes[i]);
  }

 protected:

  void _resize(ConstArrayView<Int32> ar)
  {
    Integer size1 = ar.size();
    // Calculates the total number of elements
    // TODO: Check that we do not exceed the max value of an Int32
    Integer total_size = 0;
    for (Integer i = 0; i < size1; ++i)
      total_size += ar[i];

    // If the total number of elements does not change, check
    // if the resize is necessary
    if (total_size == totalNbElement() && size1 == m_indexes.size()) {
      bool is_same = true;
      for (Integer i = 0; i < size1; ++i)
        if (m_sizes[i] != ar[i]) {
          is_same = false;
          break;
        }
      if (is_same)
        return;
    }

    Integer old_size1 = m_indexes.size();

    SharedArray<DataType> new_buffer(m_buffer.allocationOptions(), total_size);

    // Copies the values from the old array to the new one.
    if (old_size1 > size1)
      old_size1 = size1;
    Integer index = 0;
    for (Integer i = 0; i < old_size1; ++i) {
      Integer size2 = ar[i];
      Integer old_size2 = m_sizes[i];
      if (old_size2 > size2)
        old_size2 = size2;
      ConstArrayView<DataType> cav(_value(i));
      for (Integer j = 0; j < old_size2; ++j)
        new_buffer[index + j] = cav[j];
      index += size2;
    }
    m_buffer = new_buffer;

    m_indexes.resize(size1);
    m_sizes.resize(size1);
    for (Integer i2 = 0, index2 = 0; i2 < size1; ++i2) {
      Integer size2 = ar[i2];
      m_indexes[i2] = index2;
      m_sizes[i2] = size2;
      index2 += size2;
    }
  }

 protected:

  void _copy(const MultiArray2<DataType>& rhs, bool do_clone)
  {
    m_buffer = do_clone ? rhs.m_buffer.clone() : rhs.m_buffer;
    m_indexes = do_clone ? rhs.m_indexes.clone() : rhs.m_indexes;
    m_sizes = do_clone ? rhs.m_sizes.clone() : rhs.m_sizes;
  }
  void _copy(ConstMultiArray2View<DataType> aview)
  {
    m_buffer = aview.m_buffer;
    m_indexes = aview.m_indexes;
    m_sizes = aview.m_sizes;
  }

 private:

  //! Array of Values
  SharedArray<DataType> m_buffer;
  //! Array of indices in \a m_buffer of the first element of the second dimension
  SharedArray<Int32> m_indexes;
  //! Array of sizes of the second dimension
  SharedArray<Int32> m_sizes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 * \brief Multi-sized 2D array with reference semantics.
 */
template <typename DataType>
class SharedMultiArray2
: public MultiArray2<DataType>
{
 public:

  using ThatClass = SharedMultiArray2<DataType>;

 public:

  SharedMultiArray2() = default;
  explicit SharedMultiArray2(ConstArrayView<Int32> sizes)
  : MultiArray2<DataType>(sizes)
  {}
  SharedMultiArray2(ConstMultiArray2View<DataType> view)
  : MultiArray2<DataType>(view)
  {}
  SharedMultiArray2(const SharedMultiArray2<DataType>& rhs)
  : MultiArray2<DataType>(rhs, false)
  {}
  SharedMultiArray2(const UniqueMultiArray2<DataType>& rhs);

 public:

  ThatClass& operator=(const ThatClass& rhs)
  {
    if (&rhs != this)
      this->_copy(rhs, false);
    return (*this);
  }
  void operator=(ConstMultiArray2View<DataType> view)
  {
    this->_copy(view);
  }
  ThatClass& operator=(const UniqueMultiArray2<DataType>& rhs);
  void operator=(const MultiArray2<DataType>& rhs) = delete;

 public:

  //! Clones the array
  SharedMultiArray2<DataType> clone() const
  {
    return SharedMultiArray2<DataType>(this->constView());
  }

 private:

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 * \brief Multi-sized 2D array with value semantics.
 */
template <typename DataType>
class UniqueMultiArray2
: public MultiArray2<DataType>
{
 public:

  using ThatClass = UniqueMultiArray2<DataType>;

 public:

  UniqueMultiArray2() = default;
  explicit UniqueMultiArray2(ConstArrayView<Int32> sizes)
  : MultiArray2<DataType>(sizes)
  {}
  explicit UniqueMultiArray2(IMemoryAllocator* allocator)
  : UniqueMultiArray2(MemoryAllocationOptions(allocator))
  {}
  explicit UniqueMultiArray2(const MemoryAllocationOptions& allocation_options)
  : MultiArray2<DataType>(allocation_options)
  {}
  UniqueMultiArray2(const MemoryAllocationOptions& allocation_options,
                    ConstArrayView<Int32> sizes)
  : MultiArray2<DataType>(allocation_options, sizes)
  {}
  UniqueMultiArray2(ConstMultiArray2View<DataType> view)
  : MultiArray2<DataType>(view)
  {}
  UniqueMultiArray2(const SharedMultiArray2<DataType>& rhs)
  : MultiArray2<DataType>(rhs, true)
  {}
  UniqueMultiArray2(const UniqueMultiArray2<DataType>& rhs)
  : MultiArray2<DataType>(rhs, true)
  {}

 public:

  ThatClass& operator=(const SharedMultiArray2<DataType>& rhs)
  {
    this->_copy(rhs, true);
    return (*this);
  }
  ThatClass& operator=(ConstMultiArray2View<DataType> view)
  {
    // TODO: Check that \a view is not in this array
    this->_copy(view);
    return (*this);
  }
  ThatClass& operator=(const UniqueMultiArray2<DataType>& rhs)
  {
    if (&rhs != this)
      this->_copy(rhs, true);
    return (*this);
  }
  ThatClass& operator=(const MultiArray2<DataType>& rhs) = delete;

 public:

  //! Clones the array
  UniqueMultiArray2<DataType> clone() const
  {
    return UniqueMultiArray2<DataType>(this->constView());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> SharedMultiArray2<DataType>::
SharedMultiArray2(const UniqueMultiArray2<DataType>& rhs)
: MultiArray2<DataType>(rhs, true)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> SharedMultiArray2<DataType>& SharedMultiArray2<DataType>::
operator=(const UniqueMultiArray2<DataType>& rhs)
{
  this->_copy(rhs, true);
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
