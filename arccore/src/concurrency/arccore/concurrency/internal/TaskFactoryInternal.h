// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TaskFactoryInternal.h                                       (C) 2000-2025 */
/*                                                                           */
/* Internal Arcane API for 'TaskFactory'.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_INTERNAL_TASKFACTORYINTERNAL_H
#define ARCCORE_CONCURRENCY_INTERNAL_TASKFACTORYINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/ConcurrencyGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Internal Arcane API for 'TaskFactory'.
 */
class ARCCORE_CONCURRENCY_EXPORT TaskFactoryInternal
{
 public:

  //! Adds an observer for thread creation.
  static void addThreadCreateObserver(IObserver* o);

  //! Removes an observer for thread creation.
  static void removeThreadCreateObserver(IObserver* o);

  //! Notifies all observers of thread creation
  static void notifyThreadCreated();

 public:

  static void setImplementation(ITaskImplementation* task_impl);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
