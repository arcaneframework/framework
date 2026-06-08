// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Iterator.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Iterators (obsolete).                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ITERATOR_H
#define ARCCORE_BASE_ITERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include <utility>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Iteration interval.
 * \ingroup Collection
 This class manages an iteration interval with a beginning and an end. It
 allows for the simple construction of an iterator pair to iterate
 over the entire container.
*/
template <class IT, class R, class P, class V>
class IteratorBase
{
 public:

  IteratorBase(IT b, IT e)
  : m_begin(std::move(b))
  , m_end(std::move(e))
  {}

  void operator++() { ++m_begin; }
  void operator--() { --m_begin; }
  R operator*() const { return *m_begin; }
  V operator->() const { return &(*m_begin); }
  bool notEnd() const { return m_begin != m_end; }
  bool operator()() const { return notEnd(); }
  IT current() const { return m_begin; }
  IT end() const { return m_end; }

 private:

  IT m_begin; //!< Iterator over the current element
  IT m_end; //!< Iterator over the end of the container.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Iterator interval
 * \ingroup Collection
 This class manages an iterator pair allowing modification of the
 elements of the container.
 */
template <class T>
class IterT
: public IteratorBase<typename T::iterator,
                      typename T::reference, typename T::pointer, typename T::value_type*>
{
 public:

  typedef typename T::iterator iterator;
  typedef typename T::reference reference;
  typedef typename T::pointer pointer;
  typedef typename T::value_type value_type;
  typedef IteratorBase<iterator, reference, pointer, value_type*> Base;

  IterT(T& t)
  : Base(t.begin(), t.end())
  {}
  IterT(iterator b, iterator e)
  : Base(b, e)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Constant iterator interval
 * \ingroup Collection
 *
 This class manages an iterator pair that does not allow
 modification of the elements of the container.
 */
template <class T>
class ConstIterT
: public IteratorBase<typename T::const_iterator,
                      typename T::const_reference, typename T::const_pointer, const typename T::value_type*>
{
 public:

  typedef typename T::const_iterator const_iterator;
  typedef typename T::const_reference const_reference;
  typedef typename T::const_pointer const_pointer;
  typedef typename T::value_type value_type;
  typedef IteratorBase<const_iterator, const_reference, const_pointer, const value_type*> Base;

  ConstIterT(const T& t)
  : Base(t.begin(), t.end())
  {}
  ConstIterT(const_iterator b, const_iterator e)
  : Base(b, e)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
