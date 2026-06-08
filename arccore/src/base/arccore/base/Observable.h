// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Observable.h                                                (C) 2000-2025 */
/*                                                                           */
/* Observable.                                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_OBSERVABLE_H
#define ARCCORE_BASE_OBSERVABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/IObservable.h"
#include "arccore/base/IObserver.h"
#include "arccore/base/CoreArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Core
 * \brief Base class of an observable.
 *
 * An observable cannot be copied.
 */
class ARCCORE_BASE_EXPORT Observable
: public IObservable
{
 public:

  ~Observable() override; //!< Releases resources

 public:

  Observable()
  : m_is_destroyed(false)
  {}

 public:

  Observable(const Observable& rhs) = delete;
  void operator=(const Observable& rhs) = delete;

 public:

  void attachObserver(IObserver* obs) override;
  void detachObserver(IObserver* obs) override;
  void notifyAllObservers() override;
  bool hasObservers() const override;
  void detachAllObservers() override;

 protected:

  void _detachAllObservers();

 private:

  bool m_is_destroyed;
  Impl::CoreArray<IObserver*> m_observers;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Observable that automatically calls
 * IObservable::detachAllObservers() in the destructor.
 */
class ARCCORE_BASE_EXPORT AutoDetachObservable
: public Observable
{
 public:

  AutoDetachObservable()
  : Observable()
  {}
  ~AutoDetachObservable();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
