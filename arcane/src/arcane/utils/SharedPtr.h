// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedPtr.h                                                 (C) 2000-2022 */
/*                                                                           */
/* Encapsulation d'un pointeur avec compteur de référence.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_SHAREDPTR_H
#define ARCANE_UTILS_SHAREDPTR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ptr.h"
#include "arcane/utils/Atomic.h"

#include <iostream>

/*
 * Différent de AutoRef où le pointeur doit référencer un objet satisfaisant
 * un contrat d'utilisation similaire à l'interface ISharedReference
 * (autre implémentation : std::(tr1::)|boost::shared_ptr
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class RefCounter
{
 public:

  RefCounter()
  : m_counter(1)
  {}
  void addRef() { ++m_counter; }
  void removeRef() { --m_counter; }
  Int32 refCount() const { return m_counter.load(); }

 private:

  std::atomic<Int32> m_counter;
};

template <typename T>
class SharedPtrT : public PtrT<T>
{
 private:

  typedef PtrT<T> BaseClass;

 public:

  ~SharedPtrT() { reset(); }

  SharedPtrT()
  : BaseClass(0)
  , m_counter(0)
  , m_free(true)
  {}

  //Constructeur de copie
  SharedPtrT(const SharedPtrT<T>& ptr)
  : PtrT<T>(0)
  {
    _copy(ptr.get(), ptr.refCountPtr(), ptr.explicitDelete());
  }

  // N'est pas utilise quand T2 = T
  template <typename T2>
  SharedPtrT(const SharedPtrT<T2>& ptr)
  : PtrT<T>(0)
  {
    _copy(ptr.get(), ptr.refCountPtr(), ptr.explicitDelete());
  }

  // Constructeurs explicites
  template <typename T2>
  explicit SharedPtrT(T2* t, bool tofree = true)
  : BaseClass(t)
  , m_counter(new RefCounter())
  , m_free(tofree)
  {}

  // Constructeurs pour faire un "dynamic_cast"
  template <typename T2>
  explicit SharedPtrT(const SharedPtrT<T2>& ptr, bool)
  {
    _copy(dynamic_cast<T*>(ptr.get()), ptr.refCountPtr(), ptr.explicitDelete());
  }

  SharedPtrT<T>& operator=(const SharedPtrT<T>& ptr)
  {
    reset();
    _copy(ptr.get(), ptr.refCountPtr(), ptr.explicitDelete());
    return (*this);
  }

  // N'est pas utilise quand T2 = T
  template <typename T2>
  SharedPtrT<T>& operator=(const SharedPtrT<T2>& ptr)
  {
    reset();
    _copy(ptr.get(), ptr.refCountPtr(), ptr.explicitDelete());
    return (*this);
  }

  bool isUnique() const { return (m_counter->refCount() == 1); }
  Int32 refCount() const { return (m_counter) ? m_counter->refCount() : 0; }
  bool isUsed() const { return (m_counter != NULL); }
  void reset()
  {
    if (!m_counter)
      return;
    m_counter->removeRef();
    if (m_counter->refCount() == 0) {
      if (m_free)
        delete BaseClass::m_value;
      delete m_counter;
    }
    m_counter = 0;
    BaseClass::m_value = 0;
  }

  RefCounter* refCountPtr() const { return m_counter; }
  bool explicitDelete() const { return m_free; }

 private:

  void _copy(T* ptr, RefCounter* ref, bool free)
  {
    BaseClass::operator=(ptr);
    m_counter = ref;
    m_free = free;
    if (m_counter != 0)
      m_counter->addRef();
  }

  RefCounter* m_counter;
  bool m_free;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
SharedPtrT<T> SPtr(T* ptr)
{
  return SharedPtrT<T>(ptr);
}

template <typename T2, typename T>
SharedPtrT<T2> SPtr_dynamic_cast(const SharedPtrT<T>& src)
{
  return SharedPtrT<T2>(src, true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
