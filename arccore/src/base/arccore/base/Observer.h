// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Observer.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Observer.                                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_OBSERVER_H
#define ARCCORE_BASE_OBSERVER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/IObserver.h"
#include "arccore/base/IObservable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Base class of an observer.
 */
class ARCCORE_BASE_EXPORT AbstractObserver
: public IObserver
{
 public:

  AbstractObserver() = default;
  ~AbstractObserver() override;

 public:

  //! Attaches to the observable \a obs
  void attachToObservable(IObservable* obs) override;

  //! Detaches from the observable
  void detach() override;

 private:

  IObservable* m_observable = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Core
 * \brief Observer for a type T
 */
template <class T>
class ObserverT
: public AbstractObserver
{
 public:

  typedef void (T::*Func0Ptr)(); //!< Type of the member function pointer
  typedef void (T::*Func1Ptr)(const IObservable&); //!< Type of the member function pointer

 public:

  //! Constructor
  ObserverT(T* object, Func1Ptr funcptr)
  : m_object(object)
  , m_function0(nullptr)
  , m_function1(funcptr)
  {}
  ObserverT(T* object, Func0Ptr funcptr)
  : m_object(object)
  , m_function0(funcptr)
  , m_function1(nullptr)
  {}

 public:
 public:

  //! Executes the associated method
  void observerUpdate(IObservable* iob) override
  {
    if (m_function1)
      (m_object->*m_function1)(*iob);
    if (m_function0)
      (m_object->*m_function0)();
  }

 private:

  T* m_object; //!< Associated object.
  Func0Ptr m_function0; //!< Pointer to the associated method.
  Func1Ptr m_function1; //!< Pointer to the associated method.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
