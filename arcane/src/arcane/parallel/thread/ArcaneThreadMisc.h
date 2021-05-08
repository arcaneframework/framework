// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneThreadMisc.h                                          (C) 2000-2021 */
/*                                                                           */
/* Fonctions diverses pour les threads.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_THREAD_ARCANETHREADMISC_H
#define ARCANE_PARALLEL_THREAD_ARCANETHREADMISC_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/thread/ArcaneThread.h"

#include <thread>

#if defined(__x86_64__)
#include <immintrin.h> // Pour _mm_pause()
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Utilise l'instruction 'pause' du CPU si possible.
 *
 * Cette instruction sert à indiquer au CPU qu'on souhaite attendre en
 * évitant de consommer des ressources inutiles.
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
#elif defined (__aarch64__)
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

