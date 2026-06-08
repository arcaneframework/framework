// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IThreadBarrier.h                                            (C) 2000-2026 */
/*                                                                           */
/* Interface of a barrier with threads.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_CONCURRENCY_ITHREADBARRIER_H
#define ARCCORE_CONCURRENCY_ITHREADBARRIER_H
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
 * \brief Interface of a barrier between threads.
 *
 * Once created (via IThreadImplementation::createBarrier()),
 * the barrier must be initialized
 * via init() for \a n threads. Then, each thread must
 * call the wait() method to wait until all
 * other threads reach this same point.
 * The barrier can be used multiple times.
 * To destroy the barrier, you must call destroy(). This also frees
 * the instance which should no longer be used.
 */
class ARCCORE_CONCURRENCY_EXPORT IThreadBarrier
{
 protected:

  virtual ~IThreadBarrier() = default;

 public:

  //! Initializes the barrier for \a nb_thread.
  virtual void init(Integer nb_thread) =0;

  //! Destroys the barrier.
  virtual void destroy() =0;

  //! Blocks and waits until all threads call this method.
  virtual void wait() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
