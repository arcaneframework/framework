// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayIterator.h                                             (C) 2000-2025 */
/*                                                                           */
/* Itérateur sur les Array, ArrayView, ConstArrayView, ...                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAYITERATOR_H
#define ARCCORE_BASE_ARRAYITERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include <iterator>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Itérateur sur les classes tableau de Arccore.
 *
 * Cet itérateur est utilisé pour les classes Array, ArrayView et ConstArrayView.
 *
 * Il est du type std::random_access_iterator_tag.
 */
template <typename Iterator_>
class ArrayIterator
{
 private:

  // Pour le cas où on ne supporte pas le C++14.
  template< bool B, class XX = void >
  using Iterator_enable_if_t = typename std::enable_if<B,XX>::type;

 protected:

  Iterator_ m_ptr;

  using TraitsType_ = std::iterator_traits<Iterator_>;

 public:

  typedef typename std::random_access_iterator_tag iterator_category;
  typedef typename TraitsType_::value_type value_type;
  typedef typename TraitsType_::difference_type difference_type;
  typedef typename TraitsType_::reference reference;
  typedef typename TraitsType_::pointer pointer;

 public:

  constexpr ARCCORE_HOST_DEVICE ArrayIterator() ARCCORE_NOEXCEPT : m_ptr(Iterator_()) {}

  constexpr ARCCORE_HOST_DEVICE explicit ArrayIterator(const Iterator_& i) ARCCORE_NOEXCEPT
  : m_ptr(i) {}

  // Allow iterator to const_iterator conversion
  template<typename X,typename = Iterator_enable_if_t<std::is_same<X,value_type*>::value> >
  constexpr ARCCORE_HOST_DEVICE ArrayIterator(const ArrayIterator<X>& iter) ARCCORE_NOEXCEPT
  : m_ptr(iter.base()) { }

  // Forward iterator requirements
  constexpr ARCCORE_HOST_DEVICE reference operator*() const ARCCORE_NOEXCEPT { return *m_ptr; }
  constexpr ARCCORE_HOST_DEVICE pointer operator->() const ARCCORE_NOEXCEPT { return m_ptr; }
  constexpr ARCCORE_HOST_DEVICE ArrayIterator& operator++() ARCCORE_NOEXCEPT { ++m_ptr; return *this; }
  constexpr ARCCORE_HOST_DEVICE ArrayIterator operator++(int) ARCCORE_NOEXCEPT { return ArrayIterator(m_ptr++); }

  // Bidirectional iterator requirements
  constexpr ARCCORE_HOST_DEVICE ArrayIterator& operator--() ARCCORE_NOEXCEPT { --m_ptr; return *this; }
  constexpr ARCCORE_HOST_DEVICE ArrayIterator operator--(int) ARCCORE_NOEXCEPT { return ArrayIterator(m_ptr--); }

  // Random access iterator requirements
  constexpr ARCCORE_HOST_DEVICE reference operator[](difference_type n) const ARCCORE_NOEXCEPT { return m_ptr[n]; }
  constexpr ARCCORE_HOST_DEVICE ArrayIterator& operator+=(difference_type n) ARCCORE_NOEXCEPT { m_ptr += n; return *this; }
  constexpr ARCCORE_HOST_DEVICE ArrayIterator operator+(difference_type n) const ARCCORE_NOEXCEPT { return ArrayIterator(m_ptr+n); }
  constexpr ARCCORE_HOST_DEVICE ArrayIterator& operator-=(difference_type n) ARCCORE_NOEXCEPT { m_ptr -= n; return *this; }
  constexpr ARCCORE_HOST_DEVICE ArrayIterator operator-(difference_type n) const ARCCORE_NOEXCEPT { return ArrayIterator(m_ptr-n); }

  constexpr ARCCORE_HOST_DEVICE const Iterator_& base() const ARCCORE_NOEXCEPT { return m_ptr; }
};

// Forward iterator requirements
template<typename I1, typename I2> constexpr ARCCORE_HOST_DEVICE inline bool
operator==(const ArrayIterator<I1>& lhs,const ArrayIterator<I2>& rhs) ARCCORE_NOEXCEPT
{ return lhs.base() == rhs.base(); }

template<typename I> constexpr ARCCORE_HOST_DEVICE inline bool
operator==(const ArrayIterator<I>& lhs,const ArrayIterator<I>& rhs)  ARCCORE_NOEXCEPT
{ return lhs.base() == rhs.base(); }

template<typename I1, typename I2> constexpr ARCCORE_HOST_DEVICE inline bool
operator!=(const ArrayIterator<I1>& lhs,const ArrayIterator<I2>& rhs) ARCCORE_NOEXCEPT
{ return lhs.base() != rhs.base(); }

template<typename I> constexpr ARCCORE_HOST_DEVICE inline bool
operator!=(const ArrayIterator<I>& lhs,const ArrayIterator<I>& rhs) ARCCORE_NOEXCEPT
{ return lhs.base() != rhs.base(); }

// Random access iterator requirements
template<typename I1, typename I2> constexpr ARCCORE_HOST_DEVICE inline bool
operator<(const ArrayIterator<I1>& lhs,const ArrayIterator<I2>& rhs) ARCCORE_NOEXCEPT
{ return lhs.base() < rhs.base(); }

template<typename I> constexpr ARCCORE_HOST_DEVICE inline bool
operator<(const ArrayIterator<I>& lhs,const ArrayIterator<I>& rhs) ARCCORE_NOEXCEPT
{ return lhs.base() < rhs.base(); }

template<typename I1, typename I2> constexpr ARCCORE_HOST_DEVICE inline bool
operator>(const ArrayIterator<I1>& lhs,const ArrayIterator<I2>& rhs) ARCCORE_NOEXCEPT
{ return lhs.base() > rhs.base(); }

template<typename I> constexpr ARCCORE_HOST_DEVICE inline bool
operator>(const ArrayIterator<I>& lhs,const ArrayIterator<I>& rhs) ARCCORE_NOEXCEPT
{ return lhs.base() > rhs.base(); }

template<typename I1, typename I2> constexpr ARCCORE_HOST_DEVICE inline bool
operator<=(const ArrayIterator<I1>& lhs,const ArrayIterator<I2>& rhs) ARCCORE_NOEXCEPT
{ return lhs.base() <= rhs.base(); }

template<typename I> constexpr ARCCORE_HOST_DEVICE inline bool
operator<=(const ArrayIterator<I>& lhs,const ArrayIterator<I>& rhs) ARCCORE_NOEXCEPT
{ return lhs.base() <= rhs.base(); }

template<typename I1, typename I2> constexpr ARCCORE_HOST_DEVICE inline bool
operator>=(const ArrayIterator<I1>& lhs,const ArrayIterator<I2>& rhs) ARCCORE_NOEXCEPT
{ return lhs.base() >= rhs.base(); }

template<typename I> constexpr ARCCORE_HOST_DEVICE inline bool
operator>=(const ArrayIterator<I>& lhs,const ArrayIterator<I>& rhs) ARCCORE_NOEXCEPT
{ return lhs.base() >= rhs.base(); }

// _GLIBCXX_RESOLVE_LIB_DEFECTS
// According to the resolution of DR179 not only the various comparison
// operators but also operator- must accept mixed iterator/const_iterator
// parameters.
template<typename I1, typename I2>
#if __cplusplus >= 201103L
// DR 685.
constexpr ARCCORE_HOST_DEVICE inline auto
operator-(const ArrayIterator<I1>& lhs,const ArrayIterator<I2>& rhs) ARCCORE_NOEXCEPT
  -> decltype(lhs.base() - rhs.base())
#else
  constexpr inline typename ArrayIterator<I1>::difference_type
  operator-(const ArrayIterator<I1>& lhs,const ArrayIterator<I2>& rhs)
#endif
{ return lhs.base() - rhs.base(); }

template<typename I> constexpr ARCCORE_HOST_DEVICE inline typename ArrayIterator<I>::difference_type
operator-(const ArrayIterator<I>& lhs,const ArrayIterator<I>& rhs) ARCCORE_NOEXCEPT
{ return lhs.base() - rhs.base(); }

template<typename I> constexpr ARCCORE_HOST_DEVICE inline ArrayIterator<I>
operator+(typename ArrayIterator<I>::difference_type n,
          const ArrayIterator<I>& i) ARCCORE_NOEXCEPT
{ return ArrayIterator<I>(i.base() + n); }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
