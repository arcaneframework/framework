// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IAsyncQueue.h                                               (C) 2000-2019 */
/*                                                                           */
/* Asynchronous queue allowing the exchange of information between threads.  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_THREAD_IASYNCQUEUE_H
#define ARCANE_PARALLEL_THREAD_IASYNCQUEUE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Asynchronous queue allowing the exchange of information between threads
 */
class IAsyncQueue
{
 public:
  virtual ~IAsyncQueue() = default;
 public:
  //! Adds \a v to the queue.
  virtual void push(void* v) =0;
  /*!
   * \brief Retrieves the first value from the queue and blocks if there are none.
   */
  virtual void* pop() =0;
  /*!
   * \brief Retrieves the first value if available. Returns `nullptr` otherwise.
   */
  virtual void* tryPop() =0;
 public:
  static IAsyncQueue* createQueue();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
