// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ScopedPtr.h                                                 (C) 2000-2006 */
/*                                                                           */
/* Encapsulation of a pointer that is automatically destroyed.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_SCOPEDPTR_H
#define ARCANE_UTILS_SCOPEDPTR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ptr.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Core
 * \brief Encapsulation of an automatically destructing pointer.
 *
 This class encapsulates a pointer to an object that will be destroyed (via
 the delete operator) when the instance of this class goes out of scope.

 This class is useful to ensure that an object is deallocated even if an exception occurs.

 \since 0.4.40
 \author Gilles Grospellier
 \date 16/07/2001
 */
template<class T>
class ScopedPtrT
: public PtrT<T>
{
 public:

  //! Base class type
  typedef PtrT<T> BaseClass;

 public:

  //! Constructs an instance without a reference
  ScopedPtrT() : BaseClass(0) {}

  //! Constructs an instance referencing t
  explicit ScopedPtrT(T* t) : BaseClass(t) {}

  //! Destroys the referenced object.
  ~ScopedPtrT() { delete this->m_value; }

 public:
  
  //! Copy operator
  const ScopedPtrT<T>& operator=(const ScopedPtrT<T>& from)
    {
      if (this!=&from){
        delete this->m_value;
        BaseClass::operator=(from);
      }
      return (*this);
    }

  //! Assigns the value new_value to the instance
  const ScopedPtrT<T>& operator=(T* new_value)
    {
      if (this->m_value!=new_value){
        delete this->m_value;
        this->m_value = new_value;
      }
      return (*this);
    }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
