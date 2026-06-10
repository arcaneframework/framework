// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CheckedPointer.h                                            (C) 2000-2025 */
/*                                                                           */
/* Classes encapsulating a pointer allowing usage checking.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_CHECKEDPOINTER_H
#define ARCCORE_BASE_CHECKEDPOINTER_H
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
 * \brief Encapsulation of a pointer.
 *
 * This class does nothing special other than encapsulating a
 * pointer of any type. It serves as a base class for other
 * classes that provide more advanced features like AutoRefT.
 *
 * To prevent accidental copies, the copy constructor and
 * copy operators are protected.
 *
 * In debug mode, it checks that a null pointer is not accessed.
 *
 * The template parameter does not need to be defined. This class can therefore
 * be instantiated for an opaque type.
 */
template <class T>
class CheckedPointer
{
 protected:

  //! Copy operator
  const CheckedPointer<T>& operator=(const CheckedPointer<T>& from)
  {
    m_value = from.m_value;
    return (*this);
  }

  template <typename T2>
  const CheckedPointer<T>& operator=(const CheckedPointer<T2>& from)
  {
    m_value = from.get();
    return (*this);
  }

  //! Assigns the value \a new_value to the instance
  const CheckedPointer<T>& operator=(T* new_value)
  {
    m_value = new_value;
    return (*this);
  }

  //! Constructs a reference referring to \a from
  CheckedPointer(const CheckedPointer<T>& from)
  : m_value(from.m_value)
  {}

  //! Constructs a reference referring to \a from
  template <typename T2>
  CheckedPointer(const CheckedPointer<T2>& from)
  : m_value(from.m_value)
  {}

 public:

  //! Constructs an instance without a reference
  CheckedPointer()
  : m_value(nullptr)
  {}

  //! Constructs an instance referring to \a t
  explicit CheckedPointer(T* t)
  : m_value(t)
  {}

 public:

  explicit operator bool() const { return get() != nullptr; }

 public:

  //! Returns the object referenced by the instance
  inline T* operator->() const
  {
#ifdef ARCCORE_CHECK
    if (!m_value)
      arccoreNullPointerError();
#endif
    return m_value;
  }

  //! Returns the object referenced by the instance
  inline T& operator*() const
  {
#ifdef ARCCORE_CHECK
    if (!m_value)
      arccoreNullPointerError();
#endif
    return *m_value;
  }

  /*!
   * \brief Returns the object referenced by the instance
   *
   * \warning In general, caution must be used when using this
   * function and the returned pointer should not be retained.
   */
  inline T* get() const
  {
    return m_value;
  }

  inline bool isNull() const
  {
    return (!m_value);
  }

  /*!
   * \brief Compares the objects referenced by \a v1 and \a v2
   * The comparison is done pointer by pointer.
   * \retval true if they are equal
   * \retval false otherwise
   */
  template <typename T2> friend bool
  operator==(const CheckedPointer<T>& v1, const CheckedPointer<T2>& v2)
  {
    return v1.get() == v2.get();
  }

  /*!
   * \brief Compares the objects referenced by \a v1 and \a v2
   * The comparison is done pointer by pointer.
   * \retval false if they are equal
   * \retval true otherwise
   */
  template <typename T2> friend bool
  operator!=(const CheckedPointer<T>& v1, const CheckedPointer<T2>& v2)
  {
    return v1.get() != v2.get();
  }

 protected:

  T* m_value; //!< Pointer to the referenced object
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
