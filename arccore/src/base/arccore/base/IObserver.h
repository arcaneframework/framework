// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IObserver.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Observer interface.                                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_IOBSERVER_H
#define ARCCORE_BASE_IOBSERVER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Observer interface.
 *
 * This interface represents the concept of an observer as defined in the
 * Design Pattern.
 * An observer is attached to an observable (IObservable) via the
 * IObservable::attachObserver() method and detached by
 * IObservable::detachObserver(). The observable notifies it of a change
 * by calling the observerUpdate() method.
 *
 * An observer can only be attached to one observable at a time
 *
 * The methods of this class must only be called
 * by IObservable and never directly by the user.
 */
class ARCCORE_BASE_EXPORT IObserver
{
 protected:

  IObserver() {}

 public:

  virtual ~IObserver() {} //!< Releases resources

 public:

  //! \brief Notification coming from the observable \a oba.
  virtual void observerUpdate(IObservable*) = 0;

 public:

  //! Attaches to the observable \a obs
  virtual void attachToObservable(IObservable* obs) = 0;

  //! Detaches from the observable
  virtual void detach() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
