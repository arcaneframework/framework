// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Span.h                                                      (C) 2000-2025 */
/*                                                                           */
/* Views on C arrays.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_SPAN_H
#define ARCCORE_BASE_SPAN_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"

#include <type_traits>
#include <optional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// To indicate that Span<T>::view() returns an ArrayView
// and Span<const T>::view() returns a ConstArrayView.
template <typename T>
class ViewTypeT
{
 public:

  using view_type = ArrayView<T>;
};
template <typename T>
class ViewTypeT<const T>
{
 public:

  using view_type = ConstArrayView<T>;
};

//! To have the type (SmallSpan or Span) depending on the size (Int32 or Int64)
template <typename T, typename SizeType>
class SpanTypeFromSize;

template <typename T>
class SpanTypeFromSize<T, Int32>
{
 public:

  using SpanType = SmallSpan<T>;
};

template <typename T>
class SpanTypeFromSize<T, Int64>
{
 public:

  using SpanType = Span<T>;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to store the size of a SpanImpl.
 *
 * If Extent != DynExtent, it is not necessary to keep
 * track of the number of elements in a field of the instance.
 */
template <typename SizeType>
class DynamicExtentStorage
{
  template <typename T, typename SpanSizeType, SpanSizeType SpanExtent>
  friend class ::Arcane::SpanImpl;

 public:

  explicit constexpr DynamicExtentStorage(SizeType s) noexcept
  : m_size(s)
  {}

 public:

  constexpr SizeType size() const noexcept { return m_size; }

 private:

  SizeType m_size;
};

class ARCCORE_BASE_EXPORT ExtentStorageBase
{
 public:

  static void _throwBadSize [[noreturn]] (Int64 wanted_size, Int64 expected_size);
};

//! Specialization for the compile-time known number of elements
template <typename SizeType, SizeType FixedExtent>
class ExtentStorage
{
  template <typename T, typename SpanSizeType, SpanSizeType SpanExtent>
  friend class ::Arcane::SpanImpl;

 public:

  explicit constexpr ExtentStorage([[maybe_unused]] SizeType s) noexcept
  {
#if defined(ARCCORE_CHECK) && !defined(ARCCORE_DEVICE_CODE)
    if (s != FixedExtent)
      ExtentStorageBase::_throwBadSize(s, FixedExtent);
#endif
  }
  ExtentStorage() = default;

 public:

  constexpr SizeType size() const noexcept { return FixedExtent; }

 private:

  static constexpr SizeType m_size = FixedExtent;
};

//! Specialization for the dynamic number of elements
template <>
class ExtentStorage<Int32, DynExtent>
: public DynamicExtentStorage<Int32>
{
  using BaseClass = DynamicExtentStorage<Int32>;

 public:

  explicit constexpr ExtentStorage(Int32 s) noexcept
  : BaseClass(s)
  {}
};

//! Specialization for the dynamic number of elements
template <>
class ExtentStorage<Int64, DynExtent>
: public DynamicExtentStorage<Int64>
{
  using BaseClass = DynamicExtentStorage<Int64>;

 public:

  explicit constexpr ExtentStorage(Int64 s) noexcept
  : BaseClass(s)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 * \brief View of an array of elements of type \a T.
 *
 * This class should not be used directly. Span or SmallSpan must be used instead.
 *
 * The view is non-modifiable if the template argument is of type 'const T'.
 * This class allows accessing and using an array of elements of type \a T
 * in the same way as a standard C array. \a SizeType is the
 * type used to store the number of elements in the array. This can
 * be 'Int32' or 'Int64'.
 *
 * If \a Extent is different from DynExtent (the default), the size is
 * variable; otherwise, it is fixed and has the value \a Extent.
 */
template <typename T, typename SizeType, SizeType Extent>
class SpanImpl
{
  using ExtentStorageType = Impl::ExtentStorage<SizeType, Extent>;

 public:

  using ThatClass = SpanImpl<T, SizeType, Extent>;
  using SubSpanType = SpanImpl<T, SizeType, DynExtent>;
  using size_type = SizeType;
  using ElementType = T;
  using element_type = ElementType;
  using value_type = typename std::remove_cv_t<ElementType>;
  using const_value_type = typename std::add_const_t<value_type>;
  using index_type = SizeType;
  using difference_type = SizeType;
  using pointer = ElementType*;
  using const_pointer = const ElementType*;
  using reference = ElementType&;
  using const_reference = const ElementType&;
  using iterator = ArrayIterator<pointer>;
  using const_iterator = ArrayIterator<const_pointer>;
  using view_type = typename Impl::ViewTypeT<ElementType>::view_type;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  //! Indicates if 'X' or 'const X' can be converted to 'T'
  template <typename X>
  using is_same_const_type = std::enable_if_t<std::is_same_v<X, T> || std::is_same_v<std::add_const_t<X>, T>>;

  static constexpr bool IsDynamic = (Extent == DynExtent);

 public:

  //! Constructs an empty view.
  constexpr ARCCORE_HOST_DEVICE SpanImpl() noexcept
  : m_ptr(nullptr)
  , m_size(0)
  {}

  //! Copy constructor from another view
  // For a Span<const T>, it is allowed to construct from a Span<T>
  template <typename X, SizeType XExtent, typename = std::enable_if_t<std::is_same_v<const X, T>>>
  constexpr ARCCORE_HOST_DEVICE SpanImpl(const SpanImpl<X, SizeType, XExtent>& from) noexcept
  : m_ptr(from.data())
  , m_size(from.size())
  {}

  template <SizeType XExtent>
  constexpr ARCCORE_HOST_DEVICE SpanImpl(const SpanImpl<T, SizeType, XExtent>& from) noexcept
  : m_ptr(from.data())
  , m_size(from.size())
  {}

  //! Constructs a view on a memory region starting at \a ptr and
  //! containing \a asize elements.
  constexpr ARCCORE_HOST_DEVICE SpanImpl(pointer ptr, SizeType asize) noexcept
  : m_ptr(ptr)
  , m_size(asize)
  {}

  //! Constructs a view from a std::array
  template <std::size_t N, typename X, typename = is_same_const_type<X>>
  constexpr ARCCORE_HOST_DEVICE SpanImpl(std::array<X, N>& from)
  : m_ptr(from.data())
  , m_size(ArraySizeChecker<SizeType>::check(from.size()))
  {}

  //! Constructs a view from a pointer with a fixed size
  explicit constexpr ARCCORE_HOST_DEVICE SpanImpl(T* ptr) requires(!IsDynamic)
  : m_ptr(ptr)
  {}

  //! Copy assignment operator
  template <std::size_t N, typename X, typename = is_same_const_type<X>>
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator=(std::array<X, N>& from)
  {
    m_ptr = from.data();
    m_size = ExtentStorageType(ArraySizeChecker<SizeType>::check(from.size()));
    return (*this);
  }

 public:

  //! Constructs a view on a memory region starting at \a ptr and
  // containing \a asize elements.
  static constexpr ThatClass create(pointer ptr, SizeType asize) noexcept
  {
    return ThatClass(ptr, asize);
  }

 public:

  /*!
   * \brief i-th element of the array.
   *
   * In \a check mode, bounds checking is performed.
   */
  constexpr ARCCORE_HOST_DEVICE reference operator[](SizeType i) const
  {
    ARCCORE_CHECK_AT(i, m_size.m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-th element of the array.
   *
   * In \a check mode, bounds checking is performed.
   */
  constexpr ARCCORE_HOST_DEVICE reference operator()(SizeType i) const
  {
    ARCCORE_CHECK_AT(i, m_size.m_size);
    return m_ptr[i];
  }

  /*!
   * \brief i-th element of the array.
   *
   * In \a check mode, bounds checking is performed.
   */
  constexpr ARCCORE_HOST_DEVICE reference item(SizeType i) const
  {
    ARCCORE_CHECK_AT(i, m_size.m_size);
    return m_ptr[i];
  }

  /*!
   * \brief Sets the i-th element of the array.
   *
   * In \a check mode, bounds checking is performed.
   */
  constexpr ARCCORE_HOST_DEVICE void setItem(SizeType i, const_reference v) noexcept
  {
    ARCCORE_CHECK_AT(i, m_size.m_size);
    m_ptr[i] = v;
  }

  //! Returns the size of the array
  constexpr ARCCORE_HOST_DEVICE SizeType size() const noexcept { return m_size.m_size; }
  //! Returns the size of the array in bytes
  constexpr ARCCORE_HOST_DEVICE SizeType sizeBytes() const noexcept
  {
    // TODO: always return an Int64
    return static_cast<SizeType>(m_size.m_size * sizeof(value_type));
  }
  //! Number of elements in the array
  constexpr ARCCORE_HOST_DEVICE SizeType length() const noexcept { return m_size.m_size; }

  /*!
   * \brief Iterator for the first element of the array.
   */
  constexpr ARCCORE_HOST_DEVICE iterator begin() const noexcept { return iterator(m_ptr); }
  /*!
   * \brief Iterator for the element after the end of the array.
   */
  constexpr ARCCORE_HOST_DEVICE iterator end() const noexcept { return iterator(m_ptr + m_size.m_size); }
  //! Reverse iterator for the first element of the array.
  constexpr ARCCORE_HOST_DEVICE reverse_iterator rbegin() const noexcept { return std::make_reverse_iterator(end()); }
  //! Reverse iterator for the element after the end of the array.
  constexpr ARCCORE_HOST_DEVICE reverse_iterator rend() const noexcept { return std::make_reverse_iterator(begin()); }

 public:

  //! Iteration range from the first to the last element.
  ARCCORE_DEPRECATED_REASON("Y2023: Use begin()/end() instead")
  ArrayRange<pointer> range() const
  {
    return ArrayRange<pointer>(m_ptr, m_ptr + m_size.m_size);
  }

 public:

  //! Address of the index-th element
  constexpr ARCCORE_HOST_DEVICE pointer ptrAt(SizeType index) const
  {
    ARCCORE_CHECK_AT(index, m_size.m_size);
    return m_ptr + index;
  }

  // Element at index \a i. Always checks for bounds.
  constexpr ARCCORE_HOST_DEVICE reference at(SizeType i) const
  {
    arccoreCheckAt(i, m_size.m_size);
    return m_ptr[i];
  }

  // Sets the element at index \a i. Always checks for bounds.
  constexpr ARCCORE_HOST_DEVICE void setAt(SizeType i, const_reference value)
  {
    arccoreCheckAt(i, m_size.m_size);
    m_ptr[i] = value;
  }

  //! Fills the array with the value \a o
  ARCCORE_HOST_DEVICE inline void fill(T o)
  {
    for (SizeType i = 0, n = m_size.m_size; i < n; ++i)
      m_ptr[i] = o;
  }

  /*!
   * \brief Constant view of this view.
   */
  constexpr view_type smallView()
  {
    Integer s = arccoreCheckArraySize(m_size.m_size);
    return view_type(s, m_ptr);
  }

  /*!
   * \brief Constant view of this view.
   */
  constexpr ConstArrayView<value_type> constSmallView() const
  {
    Integer s = arccoreCheckArraySize(m_size.m_size);
    return ConstArrayView<value_type>(s, m_ptr);
  }

  /*!
   * \brief Sub-view starting from element \a abegin
   * and containing \a asize elements.
   *
   * If `(abegin+asize` is greater than the size of the array,
   * the view is truncated to this size, potentially returning an empty view.
   */
  constexpr ARCCORE_HOST_DEVICE SubSpanType subSpan(SizeType abegin, SizeType asize) const
  {
    if (abegin >= m_size.m_size)
      return {};
    asize = _min(asize, m_size.m_size - abegin);
    return { m_ptr + abegin, asize };
  }

  /*!
   * \brief Sub-view starting from element \a abegin and containing \a asize elements.
   * \sa subSpan()
   */
  constexpr ARCCORE_HOST_DEVICE SubSpanType subPart(SizeType abegin, SizeType asize) const
  {
    return subSpan(abegin, asize);
  }

  /*!
   * \brief Sub-view starting from element \a abegin
   * and containing \a asize elements.
   *
   * If `(abegin+asize)` is greater than the size of the array,
   * the view is truncated to this size, potentially returning an empty view.
   */
  ARCCORE_DEPRECATED_REASON("Y2023: use subSpan() instead")
  constexpr SubSpanType subView(SizeType abegin, SizeType asize) const
  {
    return subSpan(abegin, asize);
  }

  //! For C++20 compatibility
  constexpr ARCCORE_HOST_DEVICE SubSpanType subspan(SizeType abegin, SizeType asize) const
  {
    return subSpan(abegin, asize);
  }

  //! Sub-view corresponding to the interval \a index over \a nb_interval
  ARCCORE_DEPRECATED_REASON("Y2023: use subSpanInterval() instead")
  constexpr SubSpanType subViewInterval(SizeType index, SizeType nb_interval) const
  {
    return impl::subViewInterval<ThatClass>(*this, index, nb_interval);
  }

  //! Sub-view corresponding to the interval \a index over \a nb_interval
  constexpr SubSpanType subSpanInterval(SizeType index, SizeType nb_interval) const
  {
    return impl::subViewInterval<ThatClass>(*this, index, nb_interval);
  }

  //! Sub-view corresponding to the interval \a index over \a nb_interval
  constexpr SubSpanType subPartInterval(SizeType index, SizeType nb_interval) const
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
  template <class U> ARCCORE_HOST_DEVICE void copy(const U& copy_array)
  {
    Int64 n = copy_array.size();
    Int64 size_as_int64 = m_size.m_size;
    arccoreCheckAt(n, size_as_int64 + 1);
    const_pointer copy_begin = copy_array.data();
    pointer to_ptr = m_ptr;
    // We are sure that \a fits into a 'SizeType' because it is smaller
    // than \a m_size
    SizeType n_as_sizetype = static_cast<SizeType>(n);
    for (SizeType i = 0; i < n_as_sizetype; ++i)
      to_ptr[i] = copy_begin[i];
  }

  //! Returns \a true if the array is empty (zero dimension)
  constexpr ARCCORE_HOST_DEVICE bool empty() const noexcept { return m_size.m_size == 0; }
  //! Returns \a true if the array contains the element with value \a v
  ARCCORE_HOST_DEVICE bool contains(const_reference v) const
  {
    for (SizeType i = 0; i < m_size.m_size; ++i) {
      if (m_ptr[i] == v)
        return true;
    }
    return false;
  }

  /*!
   * /brief Position of the first element with value \a v
   *
   * /param v The value to find.
   * /return The position of the first element with value \a v if present, std::nullopt otherwise.
   */
  std::optional<SizeType> findFirst(const_reference v) const
  {
    for (SizeType i = 0; i < m_size.m_size; ++i) {
      if (m_ptr[i] == v)
        return i;
    }
    return std::nullopt;
  }

 public:

  constexpr ARCCORE_HOST_DEVICE void setArray(const ArrayView<T>& v) noexcept
  {
    m_ptr = v.m_ptr;
    m_size = v.m_size;
  }
  constexpr ARCCORE_HOST_DEVICE void setArray(const Span<T>& v) noexcept
  {
    m_ptr = v.m_ptr;
    m_size = v.m_size;
  }

  /*!
   * \brief Pointer to the start of the view.
   *
   * \warning Accesses via the returned pointer cannot be
   * checked by Arcane, unlike accesses via
   * operator[](): no overflow checking is possible,
   * even in check mode.
   */
  constexpr ARCCORE_HOST_DEVICE pointer data() const noexcept { return m_ptr; }

  //! Equality operator (valid if T is const but not X)
  template <typename X, SizeType Extent2, typename = std::enable_if_t<std::is_same_v<X, value_type>>> friend bool
  operator==(const SpanImpl<T, SizeType, Extent>& rhs, const SpanImpl<X, SizeType, Extent2>& lhs)
  {
    return impl::areEqual(SpanImpl<T, SizeType>(rhs), SpanImpl<T, SizeType>(lhs));
  }

  //! Inequality operator (valid if T is const but not X)
  template <typename X, SizeType Extent2, typename = std::enable_if_t<std::is_same_v<X, value_type>>> friend bool
  operator!=(const SpanImpl<T, SizeType, Extent>& rhs, const SpanImpl<X, SizeType, Extent2>& lhs)
  {
    return !operator==(rhs, lhs);
  }

  //! Equality operator
  template <SizeType Extent2> friend bool
  operator==(const SpanImpl<T, SizeType, Extent>& rhs, const SpanImpl<T, SizeType, Extent2>& lhs)
  {
    return impl::areEqual(SpanImpl<T, SizeType>(rhs), SpanImpl<T, SizeType>(lhs));
  }

  //! Inequality operator
  template <SizeType Extent2> friend bool
  operator!=(const SpanImpl<T, SizeType, Extent>& rhs, const SpanImpl<T, SizeType, Extent2>& lhs)
  {
    return !operator==(rhs, lhs);
  }

  friend inline std::ostream& operator<<(std::ostream& o, const ThatClass& val)
  {
    impl::dumpArray(o, Span<const T, DynExtent>(val.data(), val.size()), 500);
    return o;
  }

 protected:

  /*!
   * \brief Modifies the array pointer and size.
   *
   * It is up to the derived class to verify the consistency between the allocated pointer
   * and the given dimension.
   */
  constexpr void _setArray(pointer v, SizeType s) noexcept
  {
    m_ptr = v;
    m_size = s;
  }

  /*!
   * \brief Modifies the array start pointer.
   *
   * It is up to the derived class to verify the consistency between the allocated pointer
   * and the given dimension.
   */
  constexpr void _setPtr(pointer v) noexcept { m_ptr = v; }

  /*!
   * \brief Modifies the array size.
   *
   * It is up to the derived class to verify the consistency between the allocated pointer
   * and the given dimension.
   */
  constexpr void _setSize(SizeType s) noexcept { m_size = ExtentStorageType(s); }

 private:

  pointer m_ptr; //!< Pointer to the array
  //! Number of elements in the array
  ARCCORE_NO_UNIQUE_ADDRESS ExtentStorageType m_size;

 private:

  static constexpr SizeType _min(SizeType a, SizeType b)
  {
    return ((a < b) ? a : b);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 * \brief View of an array of elements of type \a T.
 *
 * The view is non-modifiable if the template argument is of type 'const T'.
 Cette class allows accessing and using an array of elements of type \a T in
 the same way as a standard C array. It is similar to ArrayView, except that
 the number of elements is stored as an 'Int64' and can therefore exceed 2GB.
 It is designed to be similar to the C++20 std::span class.
*/
template <typename T, Int64 Extent>
class Span
: public SpanImpl<T, Int64, Extent>
{
 public:

  using ThatClass = Span<T, Extent>;
  using BaseClass = SpanImpl<T, Int64, Extent>;
  using size_type = Int64;
  using value_type = typename BaseClass::value_type;
  using pointer = typename BaseClass::pointer;
  template <typename X>
  using is_same_const_type = std::enable_if_t<std::is_same_v<X, T> || std::is_same_v<std::add_const_t<X>, T>>;
  static constexpr bool IsDynamic = (Extent == DynExtent);

 public:

  //! Constructs an empty view.
  Span() = default;
  //! Copy constructor from another view
  constexpr ARCCORE_HOST_DEVICE Span(const ArrayView<value_type>& from) noexcept
  : BaseClass(from.m_ptr, from.m_size)
  {}
  // Constructor from a ConstArrayView. This is only allowed
  // if T is const.
  template <typename X, typename = std::enable_if_t<std::is_same_v<X, value_type>>>
  constexpr ARCCORE_HOST_DEVICE Span(const ConstArrayView<X>& from) noexcept
  : BaseClass(from.m_ptr, from.m_size)
  {}
  // For a Span<const T>, we are allowed to construct from a Span<T>
  template <typename X, Int64 XExtent, typename = std::enable_if_t<std::is_same_v<const X, T>>>
  constexpr ARCCORE_HOST_DEVICE Span(const Span<X, XExtent>& from) noexcept
  : BaseClass(from)
  {}
  // For a Span<const T>, we are allowed to construct from a SmallSpan<T>
  template <typename X, Int32 XExtent, typename = std::enable_if_t<std::is_same_v<const X, T>>>
  constexpr ARCCORE_HOST_DEVICE Span(const SmallSpan<X, XExtent>& from) noexcept
  : BaseClass(from.data(), from.size())
  {}
  template <Int64 XExtent>
  constexpr ARCCORE_HOST_DEVICE Span(const SpanImpl<T, Int64, XExtent>& from) noexcept
  : BaseClass(from)
  {}
  template <Int32 XExtent>
  constexpr ARCCORE_HOST_DEVICE Span(const SpanImpl<T, Int32, XExtent>& from) noexcept
  : BaseClass(from.data(), from.size())
  {}

  //! Constructs a view on a memory area starting at \a ptr and containing
  //! \a asize elements.
  constexpr ARCCORE_HOST_DEVICE Span(pointer ptr, Int64 asize) noexcept
  : BaseClass(ptr, asize)
  {}

  //! Constructs a view from a std::array.
  template <std::size_t N, typename X, typename = is_same_const_type<X>>
  constexpr ARCCORE_HOST_DEVICE Span(std::array<X, N>& from) noexcept
  : BaseClass(from.data(), from.size())
  {}

  //! Constructs a view from a pointer with a fixed size
  explicit constexpr ARCCORE_HOST_DEVICE Span(T* ptr) requires(!IsDynamic)
  : BaseClass(ptr)
  {}

  //! Copy assignment operator
  template <std::size_t N, typename X, typename = is_same_const_type<X>>
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator=(std::array<X, N>& from) noexcept
  {
    this->_setPtr(from.data());
    this->_setSize(from.size());
    return (*this);
  }

 public:

  //! Constructs a view on a memory area starting at \a ptr and
  // containing \a asize elements.
  static constexpr ThatClass create(pointer ptr, size_type asize) noexcept
  {
    return ThatClass(ptr, asize);
  }

 public:

  /*!
   * \brief Sub-view starting from element \a abegin
   * and containing \a asize elements.
   *
   * If `(abegin+asize` is greater than the size of the array,
   * the view is truncated to this size, potentially returning an empty view.
   */
  constexpr ARCCORE_HOST_DEVICE Span<T, DynExtent> subspan(Int64 abegin, Int64 asize) const
  {
    return BaseClass::subspan(abegin, asize);
  }

  /*!
   * \brief Sub-view starting from element \a abegin
   * and containing \a asize elements.
   *
   * If `(abegin+asize)` is greater than the size of the array,
   * the view is truncated to this size, potentially returning an empty view.
   */
  constexpr ARCCORE_HOST_DEVICE Span<T, DynExtent> subSpan(Int64 abegin, Int64 asize) const
  {
    return BaseClass::subSpan(abegin, asize);
  }

  /*!
   * \brief Sub-view starting from element \a abegin
   * and containing \a asize elements.
   *
   * If `(abegin+asize)` is greater than the size of the array,
   * the view is truncated to this size, potentially returning an empty view.
   */
  constexpr ARCCORE_HOST_DEVICE Span<T, DynExtent> subPart(Int64 abegin, Int64 asize) const
  {
    return BaseClass::subPart(abegin, asize);
  }

  //! Sub-view corresponding to the interval \a index over \a nb_interval
  constexpr ARCCORE_HOST_DEVICE Span<T, DynExtent> subSpanInterval(Int64 index, Int64 nb_interval) const
  {
    return impl::subViewInterval<ThatClass>(*this, index, nb_interval);
  }

  //! Sub-view corresponding to the interval \a index over \a nb_interval
  constexpr ARCCORE_HOST_DEVICE Span<T, DynExtent> subPartInterval(Int64 index, Int64 nb_interval) const
  {
    return impl::subViewInterval<ThatClass>(*this, index, nb_interval);
  }

  /*!
   * \brief Sub-view starting from element \a abegin
   * and containing \a asize elements.
   *
   * If `(abegin+asize)` is greater than the size of the array,
   * the view is truncated to this size, potentially returning an empty view.
   */
  ARCCORE_DEPRECATED_REASON("Y2023: use subSpan() instead")
  constexpr ARCCORE_HOST_DEVICE Span<T> subView(Int64 abegin, Int64 asize) const
  {
    return subspan(abegin, asize);
  }

  //! Sub-view corresponding to the interval \a index over \a nb_interval
  ARCCORE_DEPRECATED_REASON("Y2023: use subSpanInterval() instead")
  constexpr ARCCORE_HOST_DEVICE Span<T> subViewInterval(Int64 index, Int64 nb_interval) const
  {
    return impl::subViewInterval<ThatClass>(*this, index, nb_interval);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 * \brief View of an array of elements of type \a T.
 *
 * The view is non-modifiable if the template argument is of type 'const T'.
 *
 * This class allows accessing and using an array of elements of type \a T
 * in the same way as a standard C array. It is similar to Span, except
 * that the number of elements is stored as an 'Int32'.
 *
 * \note To be valid, the number of bytes associated with the view
 * (sizeBytes()) must also fit within an \a Int32.
 */
template <typename T, Int32 Extent>
class SmallSpan
: public SpanImpl<T, Int32, Extent>
{
 public:

  using ThatClass = SmallSpan<T, Extent>;
  using BaseClass = SpanImpl<T, Int32, Extent>;
  using size_type = Int32;
  using value_type = typename BaseClass::value_type;
  using pointer = typename BaseClass::pointer;
  template <typename X>
  using is_same_const_type = std::enable_if_t<std::is_same_v<X, T> || std::is_same_v<std::add_const_t<X>, T>>;
  static constexpr bool IsDynamic = (Extent == DynExtent);

 public:

  //! Constructs an empty view.
  SmallSpan() = default;

  //! Copy constructor from another view
  constexpr ARCCORE_HOST_DEVICE SmallSpan(const ArrayView<value_type>& from) noexcept
  : BaseClass(from.m_ptr, from.m_size)
  {}

  // Constructor from a ConstArrayView. This is only allowed
  // if T is const.
  template <typename X, typename = std::enable_if_t<std::is_same<X, value_type>::value>>
  constexpr ARCCORE_HOST_DEVICE SmallSpan(const ConstArrayView<X>& from) noexcept
  : BaseClass(from.m_ptr, from.m_size)
  {}

  // For a Span<const T>, we are allowed to construct from a Span<T>
  template <typename X, typename = std::enable_if_t<std::is_same<X, value_type>::value>>
  constexpr ARCCORE_HOST_DEVICE SmallSpan(const SmallSpan<X>& from) noexcept
  : BaseClass(from)
  {}

  template <Int32 XExtent>
  constexpr ARCCORE_HOST_DEVICE SmallSpan(const SpanImpl<T, Int32, XExtent>& from) noexcept
  : BaseClass(from)
  {}

  //! Constructs a view over a memory region starting at \a ptr and
  //! containing \a asize elements.
  constexpr ARCCORE_HOST_DEVICE SmallSpan(pointer ptr, Int32 asize) noexcept
  : BaseClass(ptr, asize)
  {}

  template <std::size_t N, typename X, typename = is_same_const_type<X>>
  constexpr ARCCORE_HOST_DEVICE SmallSpan(std::array<X, N>& from)
  : BaseClass(from)
  {}

  //! Constructs a view from a pointer with a fixed size
  explicit constexpr ARCCORE_HOST_DEVICE SmallSpan(T* ptr) requires(!IsDynamic)
  : BaseClass(ptr)
  {}

  //! Copy assignment operator
  template <std::size_t N, typename X, typename = is_same_const_type<X>>
  constexpr ARCCORE_HOST_DEVICE ThatClass& operator=(std::array<X, N>& from)
  {
    BaseClass::operator=(from);
    return (*this);
  }

 public:

  //! Constructs a view over a memory region starting at \a ptr and
  // containing \a asize elements.
  static constexpr ThatClass create(pointer ptr, size_type asize) noexcept
  {
    return ThatClass(ptr, asize);
  }

 public:

  /*!
   * \brief Sub-view starting from element \a abegin
   * and containing \a asize elements.
   *
   * If `(abegin+asize` is greater than the array size,
   * the view is truncated to that size, possibly returning an empty view.
   */
  constexpr ARCCORE_HOST_DEVICE SmallSpan<T, DynExtent> subspan(Int32 abegin, Int32 asize) const
  {
    return BaseClass::subspan(abegin, asize);
  }

  /*!
   * \brief Sub-view starting from element \a abegin
   * and containing \a asize elements.
   *
   * If `(abegin+asize)` is greater than the array size,
   * the view is truncated to that size, possibly returning an empty view.
   */
  constexpr ARCCORE_HOST_DEVICE SmallSpan<T, DynExtent> subSpan(Int32 abegin, Int32 asize) const
  {
    return BaseClass::subSpan(abegin, asize);
  }

  /*!
   * \brief Sub-view starting from element \a abegin
   * and containing \a asize elements.
   *
   * If `(abegin+asize)` is greater than the array size,
   * the view is truncated to that size, possibly returning an empty view.
   */
  constexpr ARCCORE_HOST_DEVICE SmallSpan<T, DynExtent> subPart(Int32 abegin, Int32 asize) const
  {
    return BaseClass::subSpan(abegin, asize);
  }

  //! Sub-view corresponding to the interval \a index over \a nb_interval
  constexpr ARCCORE_HOST_DEVICE SmallSpan<T, DynExtent> subSpanInterval(Int32 index, Int32 nb_interval) const
  {
    return impl::subViewInterval<ThatClass>(*this, index, nb_interval);
  }

  //! Sub-view corresponding to the interval \a index over \a nb_interval
  constexpr ARCCORE_HOST_DEVICE ThatClass subPartInterval(Int32 index, Int32 nb_interval) const
  {
    return subSpanInterval(index, nb_interval);
  }

  /*!
   * \brief Sub-view starting from element \a abegin
   * and containing \a asize elements.
   *
   * If `(abegin+asize)` is greater than the array size,
   * the view is truncated to that size, possibly returning an empty view.
   */
  ARCCORE_DEPRECATED_REASON("Y2023: use subPart() instead")
  constexpr ARCCORE_HOST_DEVICE SmallSpan<T> subView(Int32 abegin, Int32 asize) const
  {
    return subspan(abegin, asize);
  }

  //! Sub-view corresponding to the interval \a index over \a nb_interval
  ARCCORE_DEPRECATED_REASON("Y2023: use subPartInterval() instead")
  constexpr ARCCORE_HOST_DEVICE SmallSpan<T> subViewInterval(Int32 index, Int32 nb_interval) const
  {
    return impl::subViewInterval<ThatClass>(*this, index, nb_interval);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Prints the values of the array \a val to the stream \a o.
 *
 * If \a max_print is positive, at most \a max_print values
 * are printed. If the array size is greater than
 * \a max_print, then the first (max_print/2) and last
 * elements are printed.
 */
template <typename T, typename SizeType> inline void
dumpArray(std::ostream& o, SpanImpl<const T, SizeType> val, int max_print)
{
  impl::dumpArray(o, val, max_print);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Extracts a sub-array from a list of indices.
 *
 * Fills \a result with the values from the array \a values
 * corresponding to the indices \a indexes.
 *
 * \pre results.size() >= indexes.size();
 */
template <typename DataType, typename IntegerType, typename SizeType> inline void
_sampleSpan(SpanImpl<const DataType, SizeType> values,
            SpanImpl<const IntegerType, SizeType> indexes,
            SpanImpl<DataType, SizeType> result)
{
  const Int64 result_size = indexes.size();
  [[maybe_unused]] const Int64 my_size = values.size();
  const DataType* ptr = values.data();
  for (Int64 i = 0; i < result_size; ++i) {
    IntegerType index = indexes[i];
    ARCCORE_CHECK_AT(index, my_size);
    result[i] = ptr[index];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Extracts a sub-array from a list of indices.
 *
 * Fills \a result with the values from the array \a values
 * corresponding to the indices \a indexes.
 *
 * \pre results.size() >= indexes.size();
 */
template <typename DataType> inline void
sampleSpan(Span<const DataType> values, Span<const Int64> indexes, Span<DataType> result)
{
  _sampleSpan<DataType, Int64, Int64>(values, indexes, result);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Extracts a sub-array from a list of indices.
 *
 * The result is stored in \a result, whose size must be at least
 * equal to that of \a indexes.
 */
template <typename DataType> inline void
sampleSpan(Span<const DataType> values, Span<const Int32> indexes, Span<DataType> result)
{
  _sampleSpan<DataType, Int32, Int64>(values, indexes, result);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Converts the view into an array of non-modifiable bytes.
 */
template <typename DataType, typename SizeType, SizeType Extent>
inline typename Impl::SpanTypeFromSize<const std::byte, SizeType>::SpanType
asBytes(const SpanImpl<DataType, SizeType, Extent>& s)
{
  return { reinterpret_cast<const std::byte*>(s.data()), s.sizeBytes() };
}

/*!
 * \brief Converts the view into an array of non-modifiable bytes.
 */
template <typename DataType>
inline SmallSpan<const std::byte>
asBytes(const ArrayView<DataType>& s)
{
  return asBytes(SmallSpan<DataType>(s));
}

/*!
 * \brief Converts the view into an array of non-modifiable bytes.
 */
template <typename DataType>
inline SmallSpan<const std::byte>
asBytes(const ConstArrayView<DataType>& s)
{
  return asBytes(SmallSpan<const DataType>(s));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Converts the view into an array of modifiable bytes.
 *
 * This method is only accessible if \a DataType is not `const`.
 */
template <typename DataType, typename SizeType, SizeType Extent,
          typename std::enable_if_t<!std::is_const<DataType>::value, int> = 0>
inline typename Impl::SpanTypeFromSize<std::byte, SizeType>::SpanType
asWritableBytes(const SpanImpl<DataType, SizeType, Extent>& s)
{
  return { reinterpret_cast<std::byte*>(s.data()), s.sizeBytes() };
}

/*!
 * \brief Converts the view into an array of modifiable bytes.
 *
 * This method is only accessible if \a DataType is not `const`.
 */
template <typename DataType> inline SmallSpan<std::byte>
asWritableBytes(const ArrayView<DataType>& s)
{
  return asWritableBytes(SmallSpan<DataType>(s));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{

  template <typename ByteType, typename DataType, Int64 Extent> inline Span<DataType>
  asSpanInternal(Span<ByteType, Extent> bytes)
  {
    Int64 size = bytes.size();
    if (size == 0)
      return {};
    static constexpr Int64 data_type_size = static_cast<Int64>(sizeof(DataType));
    static_assert(data_type_size > 0, "Bad datatype size");
    ARCCORE_ASSERT((size % data_type_size) == 0, ("Size is not a multiple of sizeof(DataType)"));
    auto* ptr = reinterpret_cast<DataType*>(bytes.data());
    return { ptr, size / data_type_size };
  }

  template <typename ByteType, typename DataType, Int32 Extent> inline SmallSpan<DataType>
  asSmallSpanInternal(SmallSpan<ByteType, Extent> bytes)
  {
    Int32 size = bytes.size();
    if (size == 0)
      return {};
    static constexpr Int32 data_type_size = static_cast<Int32>(sizeof(DataType));
    static_assert(data_type_size > 0, "Bad datatype size");
    ARCCORE_ASSERT((size % data_type_size) == 0, ("Size is not a multiple of sizeof(DataType)"));
    auto* ptr = reinterpret_cast<DataType*>(bytes.data());
    return { ptr, size / data_type_size };
  }

} // namespace impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Converts a Span<std::byte> into a Span<DataType>.
 * \pre bytes.size() % sizeof(DataType) == 0;
 */
template <typename DataType, Int64 Extent> inline Span<DataType>
asSpan(Span<std::byte, Extent> bytes)
{
  return impl::asSpanInternal<std::byte, DataType, Extent>(bytes);
}

/*!
 * \brief Converts a Span<std::byte> into a Span<const DataType>.
 * \pre bytes.size() % sizeof(DataType) == 0;
 */
template <typename DataType, Int64 Extent> inline Span<const DataType>
asSpan(Span<const std::byte, Extent> bytes)
{
  return impl::asSpanInternal<const std::byte, const DataType, Extent>(bytes);
}

/*!
 * \brief Converts a SmallSpan<std::byte> into a SmallSpan<DataType>.
 * \pre bytes.size() % sizeof(DataType) == 0;
 */
template <typename DataType, Int32 Extent> inline SmallSpan<DataType>
asSmallSpan(SmallSpan<std::byte, Extent> bytes)
{
  return impl::asSmallSpanInternal<std::byte, DataType, Extent>(bytes);
}

/*!
 * \brief Converts a SmallSpan<const std::byte> into a SmallSpan<const DataType>.
 * \pre bytes.size() % sizeof(DataType) == 0;
 */
template <typename DataType, Int32 Extent> inline SmallSpan<const DataType>
asSmallSpan(SmallSpan<const std::byte, Extent> bytes)
{
  return impl::asSmallSpanInternal<const std::byte, const DataType, Extent>(bytes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Returns a Span associated with std::array.
 */
template <typename DataType, size_t SizeType> inline Span<DataType, SizeType>
asSpan(std::array<DataType, SizeType>& s)
{
  Int64 size = static_cast<Int64>(s.size());
  return { s.data(), size };
}

/*!
 * \brief Returns a SmallSpan associated with std::array.
 */
template <typename DataType, size_t SizeType> inline SmallSpan<DataType, SizeType>
asSmallSpan(std::array<DataType, SizeType>& s)
{
  Int32 size = static_cast<Int32>(s.size());
  return { s.data(), size };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Writes the content of \a bytes to the stream \a ostr in binary format.
 *
 * This is equivalent to calling ostr.write(bytes.data(),bytes.size());
 */
extern "C++" ARCCORE_BASE_EXPORT void
binaryWrite(std::ostream& ostr, const Span<const std::byte>& bytes);

/*!
 * \brief Reads the content of \a bytes from the stream \a istr in binary format.
 *
 * This is equivalent to calling istr.read(bytes.data(),bytes.size());
 */
extern "C++" ARCCORE_BASE_EXPORT void
binaryRead(std::istream& istr, const Span<std::byte>& bytes);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
using Arcane::asBytes;
using Arcane::asSmallSpan;
using Arcane::asSpan;
using Arcane::asWritableBytes;
using Arcane::binaryRead;
using Arcane::binaryWrite;
using Arcane::sampleSpan;
} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
