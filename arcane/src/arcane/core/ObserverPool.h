// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ObserverPool.h                                              (C) 2000-2025 */
/*                                                                           */
/* List of observers.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_OBSERVERPOOL_H
#define ARCANE_CORE_OBSERVERPOOL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"

#include "arcane/core/IObservable.h"
#include "arcane/core/Observer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief List of observers
 *
 * This class allows managing a list of observers and ensuring
 * their destruction when the observer's target object is destroyed.
 */
class ARCANE_CORE_EXPORT ObserverPool
{
 public:

  typedef Collection<IObserver*> ObserverCollection;

 public:

  //! Constructor
  ObserverPool() {}
  ~ObserverPool(); //!< Frees resources

 public:

  //! Adds an observer
  template <class T> inline void
  addObserver(T* obj, void (T::*func)(const IObservable&), IObservable* oba)
  {
    IObserver* obs = new ObserverT<T>(obj, func);
    oba->attachObserver(obs);
    m_observers.add(obs);
  }

  //! Adds an observer
  template <class T> inline void
  addObserver(T* obj, void (T::*func)(), IObservable* oba)
  {
    IObserver* obs = new ObserverT<T>(obj, func);
    oba->attachObserver(obs);
    m_observers.add(obs);
  }

  //! List of observers
  ObserverCollection observers() { return m_observers; }

  //! Detaches all observers (also detaches them in the process)
  void detachAll();

 private:

  List<IObserver*> m_observers; //!< List of observers
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
