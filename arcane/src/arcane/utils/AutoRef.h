// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AutoRef.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Encapsulation of a pointer with a reference counter.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_AUTOREF_H
#define ARCANE_UTILS_AUTOREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ptr.h"
#include "arccore/base/AutoRef2.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Encapsulation of a pointer with a reference counter.
 *
 * This class encapsulates a pointer of a type that implements the methods
 * of the abstract class ISharedReference (the template parameter does not
 * need to derive from this class) and increments (_addRef()) or decrements
 * (_removeRef()) the reference counter of the pointed element during
 * successive assignments. This class does not perform any action based
 * on the value of the reference counter; the eventual destruction of the object
 * when the reference counter reaches zero is handled by the object itself.
 */
template <class T>
class AutoRefT
: public PtrT<T>
{
 public:

  //! Base class type
  using BaseClass = PtrT<T>;

  using BaseClass::m_value;

 public:

  //! Constructs an instance without a reference
  AutoRefT() = default;

  //! Constructs an instance referencing \a t
  explicit AutoRefT(T* t)
  : BaseClass()
  {
    _changeValue(t);
  }

  //! Constructs a reference referencing \a from
  AutoRefT(const AutoRefT<T>& from)
  : BaseClass()
  {
    _changeValue(from.m_value);
  }

  //! Copy operator
  AutoRefT<T>& operator=(const AutoRefT<T>& from)
  {
    _changeValue(from.m_value);
    return (*this);
  }

  //! Assigns the value \a new_value to the instance
  AutoRefT<T>& operator=(T* new_value)
  {
    _changeValue(new_value);
    return (*this);
  }

  //! Destructor. Decrements the reference counter of the pointed object
  ~AutoRefT() { _removeRef(); }

 private:

  //! Adds a reference to the encapsulated object if not null
  void _addRef()
  {
    if (m_value)
      m_value->addRef();
  }

  //! Removes a reference to the encapsulated object if not null
  void _removeRef()
  {
    if (m_value)
      m_value->removeRef();
  }

  //! Changes the referenced object to \a new_value
  void _changeValue(T* new_value)
  {
    if (m_value == new_value)
      return;
    _removeRef();
    m_value = new_value;
    _addRef();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
