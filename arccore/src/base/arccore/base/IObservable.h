// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IObservable.h                                               (C) 2000-2025 */
/*                                                                           */
/* Interface of an observable.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_IOBSERVABLE_H
#define ARCCORE_BASE_IOBSERVABLE_H
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
 * \ingroup Core
 * \brief Interface of an observable.
 *
 * An observable is an object that maintains a list of observers
 * (IObserver) and allows them to be notified of an event via the
 * notifyAllObserver() method.
 *
 * An observer is added to the list of observers by the attachObserver()
 * method and removed from this same list by detachObserver().
 *
 * The list of attached observers is ordered, and notifications occur in the
 * order of the list elements. If the same observer is present multiple times, it will be notified as many times as it is present.
 *
 * \warning It is essential to remove the observers associated with an
 * observable by calling detachAllObservers() before destroying it.
 *
 * \sa IObserver
 */
class ARCCORE_BASE_EXPORT IObservable
{
 public:

  virtual ~IObservable() {} //!< Frees resources

 public:

  static IObservable* createDefault();

 public:

  /*!
   * \brief Attaches the observer \a obs to this observable.
   *
   * It is possible to attach an observer more than once.
   */
  virtual void attachObserver(IObserver* obs) = 0;

  /*!
   * \brief Detaches the observer \a obs from this observable.
   *
   * If the observer \a obs is not present, nothing happens. If it is
   * present multiple times, the last occurrence is deleted.
   */
  virtual void detachObserver(IObserver* obs) = 0;

  /*!
   * \brief Notifies all observers.
   *
   * For each attached observer, calls IObserver::observerUpdate().
   */
  virtual void notifyAllObservers() = 0;

  //! True if observers are attached to this observable.
  virtual bool hasObservers() const = 0;

  /*!
   * \brief Detaches all observers associated with this instance.
   */
  virtual void detachAllObservers() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
