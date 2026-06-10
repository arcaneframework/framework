// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayView.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Types defining C array views.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAYVIEW_H
#define ARCCORE_BASE_ARRAYVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayRange.h"
#include "arccore/base/ArrayViewCommon.h"
#include "arccore/base/BaseTypes.h"

#include <cstddef>
#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T> class ConstArrayView;
template <typename T> class ConstIterT;
template <typename T> class IterT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 * \brief Modifiable view of an array of type \a T.
 *
 This template class allows accessing and using an array of elements of type
 \a T in the same way as a standard C array. It also maintains the size of the
 array. The size() function allows knowing the number of elements in the
 array, and the operator operator[]() allows accessing a given element.

 It is guaranteed that all elements of the view are consecutive in memory.

 This class does not manage any memory; the associated container manages it.
 Possible containers provided by %Arccore are the classes Array,
 UniqueArray or SharedArray. A view is only valid as long as the associated
 container is not reallocated.
 Similarly, the constructor and the copy operator only copy the pointers
 without reallocating memory. Therefore, they must be used with caution.

 If %Arccore is compiled in check mode (ARCCORE_CHECK is defined), accesses
 via the operator operator[]() are checked, and an
 IndexOutOfRangeException exception is thrown if an array overflow occurs.
 By attaching a debug session to the process, it is possible to see the call
 stack at the time of the overflow.

 Here are some usage examples:

 \code
 Real t[5];
 ArrayView<Real> a(t,5); // Manages an array of 5 reals.
 Integer i = 3;
 Real v = a[2]; // Assigns to the value of the 2nd element
 a[i] = 5; // Assigns the value 5 to the 3rd element
 \endcode

 It is also possible to access the array elements using
 iterators in the same way as with STL containers.

 The following example creates an iterator \e i on the array \e a and iterates
 over the entire array (method i()) and displays the elements:

 \code
 * for( Real v : a )
 *   cout << v;
 \endcode

 The following example calculates the sum of the first 3 elements of the array:

 \code
 * Real sum = 0.0;
 * for( Real v : a.subView(0,3) )
 *   sum += v;
 \endcode

*/
template <class T>
class ArrayView
{
  template <typename T2, Int64 Extent> friend class Span;
  template <typename T2, Int32 Extent> friend class SmallSpan;

 public:

  using ThatClass = ArrayView<T>;

  //! Type of the array elements
  typedef T value_type;
  //! Pointer type of an array element
  typedef value_type* pointer;
  //! Constant pointer type of an array element
  typedef const value_type* const_pointer;
  //! Type of the iterator over an array element
  typedef ArrayIterator<pointer> iterator;
  //! Type of the constant iterator over an array element
  typedef ArrayIterator<const_pointer> const_iterator;
  //! Reference type of an array element
  typedef value_type& reference;
  //! Constant reference type of an array element
  typedef const value_type& const_reference;
  //! Type indexing the array
  typedef Integer size_type;
  //! Type of a distance between array iterator elements
  typedef std::ptrdiff_t difference_type;

  //! Type of an iterator over the entire array
  typedef IterT<ArrayView<T>> iter;
  //! Type of a constant iterator over the entire array
  typedef ConstIterT<ArrayView<T>> const_iter;

  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

 public:

  //! Constructs an empty view.
  constexpr ArrayView() noexcept
  : m_size(0)
  , m_ptr(nullptr)
  {}

  //! Copy constructor from another view
  ArrayView(const ArrayView<T>& from) = default;

  //! Constructs a view over a memory region starting at \a ptr and
  // containing \a asize elements.
  constexpr ArrayView(Integer asize, pointer ptr) noexcept
  : m_size(asize)
  , m_ptr(ptr)
  {}

  //! Constructs a view over a memory region starting at \a ptr and
  //! containing \a asize elements.
  template <std::size_t N>
  constexpr ArrayView(std::array<T, N>& v)
  : m_size(arccoreCheckArraySize(v.size()))
  , m_ptr(v.data())
  {}

  //! Copy assignment operator
  ArrayView<T>& operator=(const ArrayView<T>& from) = default;

  template <std::size_t N>
  constexpr ArrayView<T>& operator=(std::array<T, N>& from)
  {
    m_size = arccoreCheckArraySize(from.size());
    m_ptr = from.data();
    return (*this);
  }

 public:

  //! Constructs a view over a memory region starting at \a ptr and
  // containing \a asize elements.
  static constexpr ThatClass create(pointer ptr, Integer asize) noexcept
  {
    return ThatClass(asize, ptr);
  }

 public:

  /*!
   * \brief i-th element of the array.
   *
   * In \a check mode, checks for overflows.
   */
  constexpr reference operator[](Integer i)
  {
    ARCCORE_CHECK_AT(i, m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-th element of the array.
   *
   * In \a check mode, checks for overflows.
   */
  constexpr const_reference operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i, m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-th element of the array.
   *
   * In \a check mode, checks for overflows.
   */
  constexpr reference operator()(Integer i)
  {
    ARCCORE_CHECK_AT(i, m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-th element of the array.
   *
   * In \a check mode, checks for overflows.
   */
  constexpr const_reference operator()(Integer i) const
  {
    ARCCORE_CHECK_AT(i, m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-th element of the array.
   *
   * In \a check mode, checks for overflows.
   */
  constexpr const_reference item(Integer i) const
  {
    ARCCORE_CHECK_AT(i, m_size);
    return m_ptr[i];
  }

  /*!
   * \brief Sets the i-th element of the array.
   *
   * In \a check mode, checks for overflows.
   */
  constexpr void setItem(Integer i, const_reference v)
  {
    ARCCORE_CHECK_AT(i, m_size);
    m_ptr[i] = v;
  }

  //! Returns the size of the array
  constexpr Integer size() const noexcept { return m_size; }
  //! Number of elements in the array
  constexpr Integer length() const noexcept { return m_size; }

  //! Iterator to the first element of the array.
  constexpr iterator begin() noexcept { return iterator(m_ptr); }
  //! Iterator to the first element after the end of the array.
  constexpr iterator end() noexcept { return iterator(m_ptr + m_size); }
  //! Constant iterator to the first element of the array.
  constexpr const_iterator begin() const noexcept { return const_iterator(m_ptr); }
  //! Constant iterator to the first element after the end of the array.
  constexpr const_iterator end() const noexcept { return const_iterator(m_ptr + m_size); }
  //! Reverse iterator to the first element of the array.
  constexpr reverse_iterator rbegin() noexcept { return std::make_reverse_iterator(end()); }
  //! Reverse iterator to the first element of the array.
  constexpr const_reverse_iterator rbegin() const noexcept { return std::make_reverse_iterator(end()); }
  //! Reverse iterator to the first element after the end of the array.
  constexpr reverse_iterator rend() noexcept { return std::make_reverse_iterator(begin()); }
  //! Reverse iterator to the first element after the end of the array.
  constexpr const_reverse_iterator rend() const noexcept { return std::make_reverse_iterator(begin()); }

 public:

  //! Iteration range from the first to the last element.
  ARCCORE_DEPRECATED_REASON("Y2023: Use begin()/end() instead")
  ArrayRange<pointer> range()
  {
    return ArrayRange<pointer>(m_ptr, m_ptr + m_size);
  }
  //! Iteration range from the first to the last element.
  ARCCORE_DEPRECATED_REASON("Y2023: Use begin()/end() instead")
  ArrayRange<const_pointer> range() const
  {
    return ArrayRange<const_pointer>(m_ptr, m_ptr + m_size);
  }

 public:

  //! Address of the index-th element
  constexpr pointer ptrAt(Integer index)
  {
    ARCCORE_CHECK_AT(index, m_size);
    return m_ptr + index;
  }

  //! Address of the index-th element
  constexpr const_pointer ptrAt(Integer index) const
  {
    ARCCORE_CHECK_AT(index, m_size);
    return m_ptr + index;
  }

  // Element at index \a i. Always checks for overflows
  constexpr const_reference at(Integer i) const
  {
    arccoreCheckAt(i, m_size);
    return m_ptr[i];
  }

  // Sets the element at index \a i. Always checks for overflows
  void setAt(Integer i, const_reference value)
  {
    arccoreCheckAt(i, m_size);
    m_ptr[i] = value;
  }

  //! Fills the array with the value \a o
  void fill(const T& o) noexcept
  {
    for (Integer i = 0, n = m_size; i < n; ++i)
      m_ptr[i] = o;
  }

  /*!
   * \brief Constant view of this view.
   */
  constexpr ConstArrayView<T> constView() const noexcept
  {
    return ConstArrayView<T>(m_size, m_ptr);
  }

  /*!
   * \brief Sub-view starting from element \a abegin
   * and containing \a asize elements.
   *
   * If (\a abegin+ \a asize) is greater than the size of the array,
   * the view is truncated to this size, potentially returning an empty view.
   */
  constexpr ArrayView<T> subView(Integer abegin, Integer asize) noexcept
  {
    if (abegin >= m_size)
      return ArrayView<T>();
    asize = _min(asize, m_size - abegin);
    return ArrayView<T>(asize, m_ptr + abegin);
  }

  /*!
   * \brief Sub-view starting from element \a abegin
   * and containing \a asize elements.
   *
   * If (\a abegin+ \a asize) is greater than the size of the array,
   * the view is truncated to this size, potentially returning an empty view.
   */
  constexpr ThatClass subPart(Integer abegin, Integer asize) noexcept
  {
    return subView(abegin, asize);
  }

  /*!
   * \brief Constant sub-view starting from
   * element `abegin` and containing `asize` elements.
   *
   * If (\a abegin+ \a asize) is greater than the size of the array,
   * the view is truncated to this size, potentially returning an empty view.
   */
  constexpr ConstArrayView<T> subConstView(Integer abegin, Integer asize) const noexcept
  {
    if (abegin >= m_size)
      return ConstArrayView<T>();
    asize = _min(asize, m_size - abegin);
    return ConstArrayView<T>(asize, m_ptr + abegin);
  }

  //! Sub-view corresponding to the interval \a index over \a nb_interval
  constexpr ArrayView<T> subViewInterval(Integer index, Integer nb_interval)
  {
    return impl::subViewInterval<ThatClass>(*this, index, nb_interval);
  }

  //! Sub-view corresponding to the interval \a index over \a nb_interval
  constexpr ThatClass subPartInterval(Integer index, Integer nb_interval)
  {
    return impl::subViewInterval<ThatClass>(*this, index, nb_interval);
  }

  /*!
   * \brief Copies the array \a copy_array into the instance.
   *
   * Since no memory allocation is performed, the
   * number of elements in \a copy_array must be less than or equal to the
   * current number of elements. If it is smaller, the elements of the
   * current array located at the end of the array remain unchanged.
   */
  template <class U>
  void copy(const U& copy_array)
  {
    auto copy_size = copy_array.size();
    const_pointer copy_begin = copy_array.data();
    pointer to_ptr = m_ptr;
    Integer n = m_size;
    if (copy_size < m_size)
      n = (Integer)copy_size;
    for (Integer i = 0; i < n; ++i)
      to_ptr[i] = copy_begin[i];
  }

  //! Returns \a true if the array is empty (zero dimension)
  constexpr bool empty() const noexcept { return m_size == 0; }
  //! \a true if the array contains the element with value \a v
  bool contains(const_reference v) const
  {
    for (Integer i = 0; i < m_size; ++i) {
      if (m_ptr[i] == v)
        return true;
    }
    return false;
  }

 public:

  void setArray(const ArrayView<T>& v) noexcept
  {
    m_ptr = v.m_ptr;
    m_size = v.m_size;
  }

  /*!
   * \brief Pointer to the beginning of the view.
   *
   * \warning Accesses via the returned pointer cannot be
   * verified by Arcane, unlike accesses via
   * operator[](): no overflow check is possible,
   * even in verification mode.
   */
  constexpr pointer unguardedBasePointer() noexcept { return m_ptr; }

  /*!
   * \brief Constant pointer to the start of the view.
   *
   * \warning Accesses via the returned pointer cannot be
   * verified by Arcane, unlike accesses via
   * operator[](): no overflow check is possible,
   * even in verification mode.
   */
  constexpr const_pointer unguardedBasePointer() const noexcept { return m_ptr; }

  /*!
   * \brief Pointer to the start of the view.
   *
   * \warning Accesses via the returned pointer cannot be
   * verified by Arcane, unlike accesses via
   * operator[](): no overflow check is possible,
   * even in verification mode.
   */
  constexpr const_pointer data() const noexcept { return m_ptr; }

  /*!
   * \brief Constant pointer to the start of the view.
   *
   * \warning Accesses via the returned pointer cannot be
   * verified by Arcane, unlike accesses via
   * operator[](): no overflow check is possible,
   * even in verification mode.
   */
  constexpr pointer data() noexcept { return m_ptr; }

 public:

  friend inline bool operator==(const ArrayView<T>& rhs, const ArrayView<T>& lhs)
  {
    return impl::areEqual(rhs, lhs);
  }

  friend inline bool operator!=(const ArrayView<T>& rhs, const ArrayView<T>& lhs)
  {
    return !(rhs == lhs);
  }

  friend std::ostream& operator<<(std::ostream& o, const ArrayView<T>& val)
  {
    impl::dumpArray(o, val, 500);
    return o;
  }

 protected:

  /*!
   * \brief Returns a pointer to the array.
   *
   * This method is identical to unguardedBasePointer() (i.e.: it will be
   * necessary to consider deleting it)
   */
  constexpr pointer _ptr() noexcept { return m_ptr; }
  /*!
   * \brief Returns a pointer to the array
   *
   * This method is identical to unguardedBasePointer() (i.e.: it will be
   * necessary to consider deleting it)
   */
  constexpr const_pointer _ptr() const noexcept { return m_ptr; }

  /*!
   * \brief Modifies the pointer and size of the array.
   *
   * It is up to the derived class to verify the consistency between the pointer
   * allocated and the given dimension.
   */
  void _setArray(pointer v, Integer s) noexcept
  {
    m_ptr = v;
    m_size = s;
  }

  /*!
   * \brief Modifies the pointer to the start of the array.
   *
   * It is up to the derived class to verify the consistency between the pointer
   * allocated and the given dimension.
   */
  void _setPtr(pointer v) noexcept { m_ptr = v; }

  /*!
   * \brief Modifies the size of the array.
   *
   * It is up to the derived class to verify the consistency between the pointer
   * allocated and the given dimension.
   */
  void _setSize(Integer s) noexcept { m_size = s; }

 private:

  Integer m_size; //!< Number of elements
  pointer m_ptr; //!< Pointer to the array

 private:

  static constexpr Integer _min(Integer a, Integer b) noexcept
  {
    return ((a < b) ? a : b);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 * \brief Constant view of an array of type \a T.
 *
 * This class functions the same way as ArrayView with the only
 * difference being that the elements of the array cannot be modified.
 */
template <class T>
class ConstArrayView
{
  friend class Span<T>;
  friend class Span<const T>;
  friend class SmallSpan<T>;
  friend class SmallSpan<const T>;

 public:

  using ThatClass = ConstArrayView<T>;

  //! Type of the array elements
  typedef T value_type;
  //! Constant pointer type of an array element
  typedef const value_type* const_pointer;
  //! Constant iterator type over an array element
  typedef ArrayIterator<const_pointer> const_iterator;
  //! Constant reference type of an array element
  typedef const value_type& const_reference;
  //! Type indexing the array
  typedef Integer size_type;
  //! Type of a distance between array iterator elements
  typedef std::ptrdiff_t difference_type;

  using const_value_type = typename std::add_const_t<value_type>;

  //! Type of a constant iterator over the entire array
  typedef ConstIterT<ConstArrayView<T>> const_iter;

  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

 public:

  //! Constructs an empty array.
  constexpr ConstArrayView() noexcept
  : m_size(0)
  , m_ptr(nullptr)
  {}
  //! Constructs an array with \a s elements
  constexpr ConstArrayView(Integer s, const_pointer ptr) noexcept
  : m_size(s)
  , m_ptr(ptr)
  {}
  //! Copy constructor.
  ConstArrayView(const ConstArrayView<T>& from) = default;
  /*!
   * \brief Copy constructor.
   * \warning Only the pointer is copied. No memory copy is performed.
   */
  constexpr ConstArrayView(const ArrayView<T>& from) noexcept
  : m_size(from.size())
  , m_ptr(from.data())
  {}

  //! Creation from a std::array
  template <std::size_t N, typename X, typename = std::enable_if_t<std::is_same_v<X, const_value_type>>>
  constexpr ConstArrayView(const std::array<X, N>& v)
  : m_size(arccoreCheckArraySize(v.size()))
  , m_ptr(v.data())
  {}

  /*!
   * \brief Copy assignment operator.
   * \warning Only the pointer is copied. No memory copy is performed.
   */
  ConstArrayView<T>& operator=(const ConstArrayView<T>& from) = default;

  /*!
   * \brief Copy assignment operator.
   * \warning Only the pointer is copied. No memory copy is performed.
   */
  constexpr ConstArrayView<T>& operator=(const ArrayView<T>& from)
  {
    m_size = from.size();
    m_ptr = from.data();
    return (*this);
  }

  //! Copy assignment operator
  template <std::size_t N, typename X, typename = std::enable_if_t<std::is_same_v<X, const_value_type>>>
  constexpr ConstArrayView<T>& operator=(const std::array<X, N>& from)
  {
    m_size = arccoreCheckArraySize(from.size());
    m_ptr = from.data();
    return (*this);
  }

 public:

  //! Constructs a view over a memory region starting at \a ptr and
  //! containing \a asize elements.
  static constexpr ThatClass create(const_pointer ptr, Integer asize) noexcept
  {
    return ThatClass(asize, ptr);
  }

 public:

  /*!
   * \brief Sub-view (constant) starting from element \a abegin and
   * containing \a asize elements.
   *
   * If `(abegin+asize)` is greater than the array size,
   * the view is truncated to that size, potentially returning an empty view.
   */
  constexpr ConstArrayView<T> subView(Integer abegin, Integer asize) const noexcept
  {
    if (abegin >= m_size)
      return ConstArrayView<T>();
    asize = _min(asize, m_size - abegin);
    return ConstArrayView<T>(asize, m_ptr + abegin);
  }

  /*!
   * \brief Sub-view (constant) starting from element \a abegin and
   * containing \a asize elements.
   *
   * If `(abegin+asize)` is greater than the array size,
   * the view is truncated to that size, potentially returning an empty view.
   */
  constexpr ThatClass subPart(Integer abegin, Integer asize) const noexcept
  {
    return subView(abegin, asize);
  }

  /*!
   * \brief Sub-view (constant) starting from element \a abegin and
   * containing \a asize elements.
   *
   * If `(abegin+asize)` is greater than the array size,
   * the view is truncated to that size, potentially returning an empty view.
   */
  constexpr ConstArrayView<T> subConstView(Integer abegin, Integer asize) const noexcept
  {
    return subView(abegin, asize);
  }

  //! Sub-view corresponding to the interval \a index over \a nb_interval
  constexpr ConstArrayView<T> subViewInterval(Integer index, Integer nb_interval) const
  {
    return impl::subViewInterval<ThatClass>(*this, index, nb_interval);
  }

  //! Sub-view corresponding to the interval \a index over \a nb_interval
  constexpr ThatClass subPartInterval(Integer index, Integer nb_interval) const
  {
    return impl::subViewInterval<ThatClass>(*this, index, nb_interval);
  }

  //! Address of the index-th element
  constexpr const_pointer ptrAt(Integer index) const
  {
    ARCCORE_CHECK_AT(index, m_size);
    return m_ptr + index;
  }

  /*!
   * \brief i-th element of the array.
   *
   * In \a check mode, checks for overflows.
   */
  constexpr const_reference operator[](Integer i) const
  {
    ARCCORE_CHECK_AT(i, m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-th element of the array.
   *
   * In \a check mode, checks for overflows.
   */
  constexpr const_reference operator()(Integer i) const
  {
    ARCCORE_CHECK_AT(i, m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-th element of the array.
   *
   * In `check` mode, checks for overflows.
   */
  constexpr const_reference item(Integer i) const
  {
    ARCCORE_CHECK_AT(i, m_size);
    return m_ptr[i];
  }

  //! Number of elements in the array
  constexpr Integer size() const noexcept { return m_size; }
  //! Number of elements in the array
  constexpr Integer length() const noexcept { return m_size; }
  //! Iterator over the first element of the array.
  constexpr const_iterator begin() const noexcept { return const_iterator(m_ptr); }
  //! Iterator over the first element after the end of the array.
  constexpr const_iterator end() const noexcept { return const_iterator(m_ptr + m_size); }
  //! Reverse iterator over the first element of the array.
  constexpr const_reverse_iterator rbegin() const noexcept { return std::make_reverse_iterator(end()); }
  //! Reverse iterator over the first element after the end of the array.
  constexpr const_reverse_iterator rend() const noexcept { return std::make_reverse_iterator(begin()); }
  //! \a true if the array is empty (size()==0)
  constexpr bool empty() const noexcept { return m_size == 0; }
  //! \a true if the array contains the element of value \a v
  bool contains(const_reference v) const
  {
    for (Integer i = 0; i < m_size; ++i) {
      if (m_ptr[i] == v)
        return true;
    }
    return false;
  }
  void setArray(const ConstArrayView<T>& v) noexcept
  {
    m_ptr = v.m_ptr;
    m_size = v.m_size;
  }

  /*!
   * \brief Pointer to the allocated memory.
   *
   * \warning Accesses via the returned pointer cannot be
   * verified by Arcane, unlike accesses via
   * operator[](): no overflow check is possible,
   * even in verification mode.
   */
  constexpr const_pointer unguardedBasePointer() const noexcept { return m_ptr; }

  /*!
   * \brief Pointer to the allocated memory.
   *
   * \warning Accesses via the returned pointer cannot be
   * verified by Arcane, unlike accesses via
   * operator[](): no overflow check is possible,
   * even in verification mode.
   */
  constexpr const_pointer data() const noexcept { return m_ptr; }

  //! Iteration range from the first to the last element.
  ARCCORE_DEPRECATED_REASON("Y2023: Use begin()/end() instead")
  ArrayRange<const_pointer> range() const
  {
    return ArrayRange<const_pointer>(m_ptr, m_ptr + m_size);
  }

 public:

  friend inline bool operator==(const ConstArrayView<T>& rhs, const ConstArrayView<T>& lhs)
  {
    return Arcane::impl::areEqual(rhs, lhs);
  }

  friend inline bool operator!=(const ConstArrayView<T>& rhs, const ConstArrayView<T>& lhs)
  {
    return !(rhs == lhs);
  }

  friend std::ostream& operator<<(std::ostream& o, const ConstArrayView<T>& val)
  {
    Arcane::impl::dumpArray(o, val, 500);
    return o;
  }

 private:

  Integer m_size; //!< Number of elements
  const_pointer m_ptr; //!< Pointer to the start of the array

 private:

  static constexpr Integer _min(Integer a, Integer b) noexcept
  {
    return ((a < b) ? a : b);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Displays the values of array \a val to the stream \a o.
 *
 * If \a max_print is positive, at most \a max_print values
 * are displayed. If the array size is greater than
 * \a max_print, the first and last
 * (max_print/2) elements are displayed.
 */
template <typename T> inline void
dumpArray(std::ostream& o, ConstArrayView<T> val, int max_print)
{
  impl::dumpArray(o, val, max_print);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
