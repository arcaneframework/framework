// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITimerMng.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Interface of a timer manager.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITIMERMNG_H
#define ARCANE_CORE_ITIMERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a timer manager.
 *
 * This manager is used exclusively by the architecture's timers (Timer)
 * and must not be used directly.
 *
 * A timer uses the beginTimer() method to indicate 
 * to this manager that it wishes to start a time measurement, and
 * the endTimer() method to indicate that the measurement is finished and
 * to obtain the time elapsed since the call to beginTimer(). It is also
 * possible to obtain the elapsed time without stopping the timer by calling
 * the getTime() function.
 *
 * Timers of the same type nest within each other and must respect
 * the stack principle for calls to beginTimer() and endTimer(): the
 * timer calling endTimer() must be the last one to have called beginTimer().
 *
 * The type of time used is determined by Timer::type(). It is either
 * CPU time or real time.
 *
 */
class ITimerMng
{
 public:

  /*!
   * \brief Frees resources.
   * \pre !hasTimer()
   */
  virtual ~ITimerMng() = default;

 public:

  /*!
   * \brief Attaches the timer \a timer to this manager.
   *
   * \pre !\a timer
   * \pre !hasTimer(\a timer)
   * \post hasTimer(\a timer)
   */
  virtual void beginTimer(Timer* timer) = 0;

  /*!
   * \brief Releases the timer \a timer.
   *
   * \return the time elapsed since the call to beginTimer().
   *
   * \pre !\a timer
   * \pre hasTimer(\a timer)
   * \post !hasTimer(\a timer)
   */
  virtual Real endTimer(Timer* timer) = 0;

  /*!
   * \brief Time elapsed since the last call to beginTimer().
   *
   * \pre !\a timer
   * \pre hasTimer(\a timer)
   */
  virtual Real getTime(Timer* timer) = 0;

  /*!
   * \brief Indicates if the timer \a timer is registered.
   *
   * \pre !\a timer
   * \deprecated This function will eventually be removed. Do not use it.
   */
  ARCCORE_DEPRECATED_2019("Do not use this method")
  virtual bool hasTimer(Timer* timer) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
