// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayRange.h                                                (C) 2000-2026 */
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
 protected:

  using TraitsType_ = Arcane::Impl::ArrayIteratorTraits<T>;

 public:

  using value_type = TraitsType_::value_type;
  using difference_type = TraitsType_::difference_type;
  using reference = TraitsType_::reference;
  using pointer = TraitsType_::pointer;

  using const_pointer = const value_type*;
  //! Type of the iterator for an element of the array
  using iterator = ArrayIterator<pointer>;
  //! Type of the constant iterator for an element of the array
  using const_iterator = ArrayIterator<const_pointer>;

 public:

  //! Constructs an empty range.
  ArrayRange() = default;

  //! Constructs a range going from \a abegin to \a aend.
  ArrayRange(pointer abegin, pointer aend) noexcept
  : m_begin(abegin)
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

  T m_begin = nullptr;
  T m_end  = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
