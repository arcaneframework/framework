// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ReferenceCounter.h                                          (C) 2000-2025 */
/*                                                                           */
/* Encapsulation of a pointer with a reference counter.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_REFERENCECOUNTER_H
#define ARCCORE_BASE_REFERENCECOUNTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/CheckedPointer.h"
#include "arccore/base/RefDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Encapsulation of a pointer with a reference counter.
 *
 * This class holds a pointer of a type that must implement
 * the following methods:
 * - addReference() to add a reference
 * - removeReference() to remove a reference.
 *
 * Unlike std::shared_ptr, the reference counter is managed
 * internally by the type *T*.
 * This class performs no action based on the reference counter value.
 * the eventual destruction of the object when the reference counter reaches
 * zero is handled by the object itself.
 */
template<class T>
class ReferenceCounter
: public CheckedPointer<T>
{
 public:

  //! Type of the base class
  typedef CheckedPointer<T> BaseClass;

  using BaseClass::m_value;

 public:

  //! Constructs an instance without a reference
  ReferenceCounter() : BaseClass(nullptr) {}
  //! Constructs an instance referencing \a t
  explicit ReferenceCounter(T* t) : BaseClass(nullptr) { _changeValue(t); }
  //! Constructs a reference referencing \a from
  ReferenceCounter(const ReferenceCounter<T>& from)
  : BaseClass(nullptr) { _changeValue(from.m_value); }

  //! Copy operator
  ReferenceCounter<T>& operator=(const ReferenceCounter<T>& from)
  {
    _changeValue(from.m_value);
    return (*this);
  }

  //! Assigns the value \a new_value to the instance
  ReferenceCounter<T>& operator=(T* new_value)
  {
    _changeValue(new_value);
    return (*this);
  }

  //! Destructor. Decrements the reference counter of the pointed object
  ~ReferenceCounter() { _removeRef(); }

 private:
	
  //! Removes a reference to the encapsulated object if not null
  void _removeRef()
  {
    if (m_value)
      ReferenceCounterAccessor<T>::removeReference(m_value);
  }
  //! Changes the referenced object to \a new_value
  void _changeValue(T* new_value)
  {
    if (m_value==new_value)
      return;
    // Always add first in case the new value
    // and the old value are from the same instance.
    if (new_value)
      ReferenceCounterAccessor<T>::addReference(new_value);
    _removeRef();
    m_value = new_value;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
