// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AutoRef2.h                                                  (C) 2000-2026 */
/*                                                                           */
/* Encapsulation of a pointer with a reference counter.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_AUTOREF2_H
#define ARCCORE_BASE_AUTOREF2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

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
 * need to derive from this class) and increments (addRef()) or decrements
 * (removeRef()) the reference counter of the pointed element during
 * successive assignments. This class performs no action based
 * on the reference counter value; the eventual destruction of the object
 * when the reference counter reaches zero is handled by the object itself.
 */
template <class T>
class AutoRef2
{
 public:

  using ThatClass = AutoRef2<T>;

 public:

  //! Constructs an instance without a reference
  AutoRef2() = default;
  //! Constructs an instance referencing \a t
  explicit AutoRef2(T* t)
  {
    _changeValue(t);
  }
  //! Constructs a reference referencing \a from
  AutoRef2(const ThatClass& from)
  {
    _changeValue(from.m_value);
  }
  //! Constructs a reference referencing \a from
  AutoRef2(ThatClass&& from) noexcept
  : m_value(from.m_value)
  {
    from.m_value = nullptr;
  }

  //! Copy operator
  ThatClass& operator=(const ThatClass& from)
  {
    _changeValue(from.m_value);
    return (*this);
  }
  //! Move operator
  ThatClass& operator=(ThatClass&& from) noexcept
  {
    _removeRef();
    m_value = from.m_value;
    from.m_value = nullptr;
    return (*this);
  }

  //! Assigns the value \a new_value to the instance
  ThatClass& operator=(T* new_value)
  {
    _changeValue(new_value);
    return (*this);
  }

  //! Destructor. Decrements the reference counter of the pointed object
  ~AutoRef2() { _removeRef(); }

  //! Returns the object referenced by the instance
  T* operator->() const
  {
    ARCCORE_CHECK_PTR(m_value);
    return m_value;
  }

  //! Returns the object referenced by the instance
  inline T& operator*() const
  {
    ARCCORE_CHECK_PTR(m_value);
    return *m_value;
  }

  //! Returns the object referenced by the instance
  T* get() const { return m_value; }

  bool isNull() const { return !m_value; }
  operator bool() const { return m_value; }

  friend bool operator==(const ThatClass& a, const ThatClass& b)
  {
    return a.get() == b.get();
  }
  friend bool operator!=(const ThatClass& a, const ThatClass& b)
  {
    return a.get() != b.get();
  }

 private:

  //! Adds a reference to the encapsulated object if not null
  void _addRef()
  {
    if (m_value)
      m_value->addRef();
  }
  //! Removes a reference to the encapsulated object if not null
  void _removeRef() noexcept
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

 private:

  T* m_value = nullptr; //!< Pointer to the referenced object
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
