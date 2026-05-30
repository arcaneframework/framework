// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneThreadMisc.h                                          (C) 2000-2021 */
/*                                                                           */
/* Various functions for threads.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_THREAD_ARCANETHREADMISC_H
#define ARCANE_PARALLEL_THREAD_ARCANETHREADMISC_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/thread/ArcaneThread.h"

#include <thread>

#if defined(__x86_64__)
#include <immintrin.h> // For _mm_pause()
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Uses the CPU 'pause' instruction if possible.
 *
 * This instruction is used to indicate to the CPU that we want to wait while
 * avoiding the consumption of unnecessary resources.
 */
inline void
arcaneDoCPUPause(Int32 count)
{
  bool is_done = false;
#if defined(__x86_64__)
  while (count > 0) {
    _mm_pause();
    --count;
  }
  is_done = true;
#elif defined(__aarch64__)
  while (count > 0) {
    __asm__ __volatile__("yield" ::: "memory");
    --count;
  }
  is_done = true;
#else
  ARCANE_UNUSED(count);
#endif
  if (!is_done)
    std::this_thread::yield();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
