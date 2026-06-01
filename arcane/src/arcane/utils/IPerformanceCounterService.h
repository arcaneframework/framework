// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IPerformanceCounterService.h                                (C) 2000-2022 */
/*                                                                           */
/* Interface of a service for accessing performance counters.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IPERFORMANCECOUNTERSERVICE_H
#define ARCANE_UTILS_IPERFORMANCECOUNTERSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a service for accessing performance counters.
 */
class ARCANE_UTILS_EXPORT IPerformanceCounterService
{
 public:

  //! Minimum size of the view for getCounters()
  static const int MIN_COUNTER_SIZE = 8;

 public:

  virtual ~IPerformanceCounterService() = default;

 public:

  //! Initializes the service.
  virtual void initialize() = 0;

  /*!
   * \brief Starts tracking performance counters.
   * \pre isStarted()==false.
   * \post isStarted()==true.
   */
  virtual void start() = 0;

  /*!
   * \brief Stops tracking performance counters.
   * \pre isStarted()==true.
   * \post isStarted()==false.
   */
  virtual void stop() = 0;

  //! Indicates if the service has started (start() has been called)
  virtual bool isStarted() const = 0;

  /*!
   * \brief Retrieves the current values of the counters.
   *
   * This method must only be called if isStarted() is true.
   *
   * If \a do_substract is \a false, fills \a counters with the
   * current values of the counters. If \a do_substract is \a true,
   * fills counters with the difference between the current values and those
   * in \a counters during the call.
   *
   \code
   * Int64ArrayView counters = ...;
   * IPerformanceCounterService* p = ...;
   * p->getCounters(counters,false);
   * ... // Operation.
   * p->getCounters(counters,true);
   * info() << "Nb cycle=" << counters[0].
   \endcode
   *
   * The counter at index 0 is always the number of cycles. \a counters
   * must have enough elements to provide at least MIN_COUNTER_SIZE counters.
   *
   * \retval the number of counters provided.
   * \pre isStarted()==true
   */
  virtual Int32 getCounters(Int64ArrayView counters, bool do_substract) = 0;

  /*!
   * \brief Value of the counter for the number of CPU cycles.
   *
   * \pre isStarted()==true   
   */
  virtual Int64 getCycles() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
