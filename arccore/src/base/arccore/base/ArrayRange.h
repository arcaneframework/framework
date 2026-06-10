// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayRange.h                                                (C) 2000-2025 */
/*                                                                           */
/* Interval over Array, ArrayView, ConstArrayView, ...                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAYRANGE_H
#define ARCCORE_BASE_ARRAYRANGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayIterator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interval over %Arccore array classes.
 *
 * This class is used to adapt array classes to STL iterators. It provides
 * methods such as begin()/end().
 */
template <typename T>
class ArrayRange
{
 public:

  typedef std::iterator_traits<T> _TraitsType;

 public:

  typedef typename _TraitsType::value_type value_type;
  typedef typename _TraitsType::difference_type difference_type;
  typedef typename _TraitsType::reference reference;
  typedef typename _TraitsType::pointer pointer;
  typedef const value_type* const_pointer;
  //! Type of the iterator for an element of the array
  typedef ArrayIterator<pointer> iterator;
  //! Type of the constant iterator for an element of the array
  typedef ArrayIterator<const_pointer> const_iterator;

 public:

  //! Constructs an empty range.
  ArrayRange() ARCCORE_NOEXCEPT : m_begin(nullptr)
  , m_end(nullptr)
  {}
  //! Constructs a range going from \a abegin to \a aend.
  ArrayRange(pointer abegin, pointer aend) ARCCORE_NOEXCEPT : m_begin(abegin)
  , m_end(aend)
  {}

 public:

  //! Returns an iterator to the first element of the array
  iterator begin() { return iterator(m_begin); }
  //! Returns an iterator to the first element after the end of the array
  iterator end() { return iterator(m_end); }
  //! Returns a constant iterator to the first element of the array
  const_iterator begin() const { return const_iterator(m_begin); }
  //! Returns a constant iterator to the first element after the end of the array
  const_iterator end() const { return const_iterator(m_end); }

  //! Pointer to the underlying array.
  value_type* data() { return m_begin; }
  //! Constant pointer to the underlying array.
  const value_type* data() const { return m_begin; }
  //! Indicates if the array is empty.
  bool empty() const { return m_end == m_begin; }

 private:

  T m_begin;
  T m_end;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
