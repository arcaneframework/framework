// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayIterator.h                                             (C) 2000-2026 */
/*                                                                           */
/* Iterator over Arrays, ArrayView, ConstArrayView, ...                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAYITERATOR_H
#define ARCCORE_BASE_ARRAYITERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include <type_traits>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Impl
{
//! Traits for iterators for arrays. This need to be specialized
template <typename T>
struct ArrayIteratorTraits;

//! Specialization for pointer.
template <typename T>
struct ArrayIteratorTraits<T*>
{
  using value_type = std::remove_cv_t<T>;
  using difference_type = std::ptrdiff_t;
  using reference = T&;
  using pointer = T*;
};

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Iterator over Arccore array classes.
 *
 * This iterator is used for Array, ArrayView, and ConstArrayView classes.
 *
 * It is of type std::random_access_iterator_tag.
 */
template <typename Iterator_>
class ArrayIterator
{
 private:

  // For the case where C++14 is not supported.
  template <bool B, class XX = void>
  using Iterator_enable_if_t = typename std::enable_if<B, XX>::type;

 protected:

  Iterator_ m_ptr;

  using TraitsType_ = Arcane::Impl::ArrayIteratorTraits<Iterator_>;

 public:

  //typedef typename std::random_access_iterator_tag iterator_category;
  //using iterator_category = std::random_access_iterator_tag;
  using value_type = TraitsType_::value_type;
  using difference_type = TraitsType_::difference_type;
  using reference = TraitsType_::reference;
  using pointer = TraitsType_::pointer;

 public:

  constexpr ARCCORE_HOST_DEVICE ArrayIterator() noexcept : m_ptr(Iterator_()) {}

  constexpr ARCCORE_HOST_DEVICE explicit ArrayIterator(const Iterator_& i) noexcept
  : m_ptr(i) {}

  // Allow iterator to const_iterator conversion
  template <typename X, typename = Iterator_enable_if_t<std::is_same<X, value_type*>::value>>
  constexpr ARCCORE_HOST_DEVICE ArrayIterator(const ArrayIterator<X>& iter) noexcept
  : m_ptr(iter.base()) {}

  // Forward iterator requirements
  constexpr ARCCORE_HOST_DEVICE reference operator*() const noexcept { return *m_ptr; }
  constexpr ARCCORE_HOST_DEVICE pointer operator->() const noexcept { return m_ptr; }
  constexpr ARCCORE_HOST_DEVICE ArrayIterator& operator++() noexcept
  {
    ++m_ptr;
    return *this;
  }
  constexpr ARCCORE_HOST_DEVICE ArrayIterator operator++(int) noexcept { return ArrayIterator(m_ptr++); }

  // Bidirectional iterator requirements
  constexpr ARCCORE_HOST_DEVICE ArrayIterator& operator--() noexcept
  {
    --m_ptr;
    return *this;
  }
  constexpr ARCCORE_HOST_DEVICE ArrayIterator operator--(int) noexcept { return ArrayIterator(m_ptr--); }

  // Random access iterator requirements
  constexpr ARCCORE_HOST_DEVICE reference operator[](difference_type n) const noexcept { return m_ptr[n]; }
  constexpr ARCCORE_HOST_DEVICE ArrayIterator& operator+=(difference_type n) noexcept
  {
    m_ptr += n;
    return *this;
  }
  constexpr ARCCORE_HOST_DEVICE ArrayIterator operator+(difference_type n) const noexcept { return ArrayIterator(m_ptr + n); }
  constexpr ARCCORE_HOST_DEVICE ArrayIterator& operator-=(difference_type n) noexcept
  {
    m_ptr -= n;
    return *this;
  }
  constexpr ARCCORE_HOST_DEVICE ArrayIterator operator-(difference_type n) const noexcept { return ArrayIterator(m_ptr - n); }

  constexpr ARCCORE_HOST_DEVICE const Iterator_& base() const noexcept { return m_ptr; }
};

// Forward iterator requirements
template <typename I1, typename I2> constexpr ARCCORE_HOST_DEVICE inline bool
operator==(const ArrayIterator<I1>& lhs, const ArrayIterator<I2>& rhs) noexcept
{
  return lhs.base() == rhs.base();
}

template <typename I> constexpr ARCCORE_HOST_DEVICE inline bool
operator==(const ArrayIterator<I>& lhs, const ArrayIterator<I>& rhs) noexcept
{
  return lhs.base() == rhs.base();
}

template <typename I1, typename I2> constexpr ARCCORE_HOST_DEVICE inline bool
operator!=(const ArrayIterator<I1>& lhs, const ArrayIterator<I2>& rhs) noexcept
{
  return lhs.base() != rhs.base();
}

template <typename I> constexpr ARCCORE_HOST_DEVICE inline bool
operator!=(const ArrayIterator<I>& lhs, const ArrayIterator<I>& rhs) noexcept
{
  return lhs.base() != rhs.base();
}

// Random access iterator requirements
template <typename I1, typename I2> constexpr ARCCORE_HOST_DEVICE inline bool
operator<(const ArrayIterator<I1>& lhs, const ArrayIterator<I2>& rhs) noexcept
{
  return lhs.base() < rhs.base();
}

template <typename I> constexpr ARCCORE_HOST_DEVICE inline bool
operator<(const ArrayIterator<I>& lhs, const ArrayIterator<I>& rhs) noexcept
{
  return lhs.base() < rhs.base();
}

template <typename I1, typename I2> constexpr ARCCORE_HOST_DEVICE inline bool
operator>(const ArrayIterator<I1>& lhs, const ArrayIterator<I2>& rhs) noexcept
{
  return lhs.base() > rhs.base();
}

template <typename I> constexpr ARCCORE_HOST_DEVICE inline bool
operator>(const ArrayIterator<I>& lhs, const ArrayIterator<I>& rhs) noexcept
{
  return lhs.base() > rhs.base();
}

template <typename I1, typename I2> constexpr ARCCORE_HOST_DEVICE inline bool
operator<=(const ArrayIterator<I1>& lhs, const ArrayIterator<I2>& rhs) noexcept
{
  return lhs.base() <= rhs.base();
}

template <typename I> constexpr ARCCORE_HOST_DEVICE inline bool
operator<=(const ArrayIterator<I>& lhs, const ArrayIterator<I>& rhs) noexcept
{
  return lhs.base() <= rhs.base();
}

template <typename I1, typename I2> constexpr ARCCORE_HOST_DEVICE inline bool
operator>=(const ArrayIterator<I1>& lhs, const ArrayIterator<I2>& rhs) noexcept
{
  return lhs.base() >= rhs.base();
}

template <typename I> constexpr ARCCORE_HOST_DEVICE inline bool
operator>=(const ArrayIterator<I>& lhs, const ArrayIterator<I>& rhs) noexcept
{
  return lhs.base() >= rhs.base();
}

// _GLIBCXX_RESOLVE_LIB_DEFECTS
// According to the resolution of DR179 not only the various comparison
// operators but also operator- must accept mixed iterator/const_iterator
// parameters.
template <typename I1, typename I2>
#if __cplusplus >= 201103L
// DR 685.
constexpr ARCCORE_HOST_DEVICE inline auto
operator-(const ArrayIterator<I1>& lhs, const ArrayIterator<I2>& rhs) noexcept
->decltype(lhs.base() - rhs.base())
#else
constexpr inline typename ArrayIterator<I1>::difference_type
operator-(const ArrayIterator<I1>& lhs, const ArrayIterator<I2>& rhs)
#endif
{
  return lhs.base() - rhs.base();
}

template <typename I> constexpr ARCCORE_HOST_DEVICE inline typename ArrayIterator<I>::difference_type
operator-(const ArrayIterator<I>& lhs, const ArrayIterator<I>& rhs) noexcept
{
  return lhs.base() - rhs.base();
}

template <typename I> constexpr ARCCORE_HOST_DEVICE inline ArrayIterator<I>
operator+(typename ArrayIterator<I>::difference_type n,
          const ArrayIterator<I>& i) noexcept
{
  return ArrayIterator<I>(i.base() + n);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
