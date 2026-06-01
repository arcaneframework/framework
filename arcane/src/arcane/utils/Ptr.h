// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Ptr.h                                                       (C) 2000-2024 */
/*                                                                           */
/* Classes encapsulating various pointers.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_PTR_H
#define ARCANE_UTILS_PTR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Core
 * \brief Encapsulation of a pointer.
 *
 * This class does nothing special other than encapsulating a
 * pointer of any type. It serves as a base class for other
 * classes that provide more advanced features like AutoRefT.
 *
 * To avoid accidental copies, the copy constructor and
 * copy operators are protected.
 *
 * In debug mode, checks that we do not access a null pointer.
 *
 * The template parameter does not need to be defined. This class can therefore
 * be instantiated for an opaque type.
 */
template <class T>
class PtrT
{
 protected:

  //! Copy operator
  PtrT<T>& operator=(const PtrT<T>& from)
  {
    m_value = from.m_value;
    return (*this);
  }

  template <typename T2>
  PtrT<T>& operator=(const PtrT<T2>& from)
  {
    m_value = from.get();
    return (*this);
  }

  //! Assigns the value \a new_value to the instance
  PtrT<T>& operator=(T* new_value)
  {
    m_value = new_value;
    return (*this);
  }

  //! Constructs a reference referring to \a from
  PtrT(const PtrT<T>& from)
  : m_value(from.m_value)
  {}

  //! Constructs a reference referring to \a from
  template <typename T2>
  PtrT(const PtrT<T2>& from)
  : m_value(from.m_value)
  {}

 public:

  //! Constructs an instance without a reference
  PtrT() = default;

  //! Constructs an instance referring to \a t
  explicit PtrT(T* t)
  : m_value(t)
  {}

  virtual ~PtrT() = default;

 public:
 public:

  //! Returns the object referenced by the instance
  inline T* operator->() const
  {
#ifdef ARCANE_CHECK
    if (!m_value)
      arcaneNullPointerError();
#endif
    return m_value;
  }

  //! Returns the object referenced by the instance
  inline T& operator*() const
  {
#ifdef ARCANE_CHECK
    if (!m_value)
      arcaneNullPointerError();
#endif
    return *m_value;
  }

  /*!
   * \brief Returns the object referenced by the instance
   *
   * \warning In general, caution must be exercised when using this
   * function and the returned pointer should not be retained.
   */
  T* get() const { return m_value; }

  bool isNull() const { return !m_value; }

 protected:

  T* m_value = nullptr; //!< Pointer to the referenced object
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Compares the objects referenced by \a v1 and \a v2
 *
 * The comparison is done pointer by pointer.
 * \retval true if they are equal
 * \retval false otherwise
 */
template <typename T1, typename T2> inline bool
operator==(const PtrT<T1>& v1, const PtrT<T2>& v2)
{
  return v1.get() == v2.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Compares the objects referenced by \a v1 and \a v2
 * The comparison is done pointer by pointer.
 * \retval false if they are equal
 * \retval true otherwise
 */
template <typename T1, typename T2> inline bool
operator!=(const PtrT<T1>& v1, const PtrT<T2>& v2)
{
  return v1.get() != v2.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
