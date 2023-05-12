// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorCoreGlobalInternal.h                             (C) 2000-2023 */
/*                                                                           */
/* Déclarations générales pour le support des accélérateurs.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_INTERNAL_ACCELERATORCOREGLOBALINTERNAL_H
#define ARCANE_ACCELERATOR_CORE_INTERNAL_ACCELERATORCOREGLOBALINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indique si on utilise le runtime CUDA
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT bool isUsingCUDARuntime();

//! Positionne l'utilisation du runtime CUDA
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void setUsingCUDARuntime(bool v);

//! Récupère l'implémentation CUDA de RunQueue (peut être nulle)
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
IRunnerRuntime*
getCUDARunQueueRuntime();

//! Positionne l'implémentation CUDA de RunQueue.
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void setCUDARunQueueRuntime(IRunnerRuntime* v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indique si on utilise le runtime HIP
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT bool isUsingHIPRuntime();

//! Positionne l'utilisation du runtime HIP
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void setUsingHIPRuntime(bool v);

//! Récupère l'implémentation HIP de RunQueue (peut être nulle)
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
IRunnerRuntime*
getHIPRunQueueRuntime();

//! Positionne l'implémentation HIP de RunQueue.
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void setHIPRunQueueRuntime(IRunnerRuntime* v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Récupère l'implémentation Séquentielle de RunQueue
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
IRunnerRuntime*
getSequentialRunQueueRuntime();

//! Récupère l'implémentation Thread de RunQueue
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
IRunnerRuntime*
getThreadRunQueueRuntime();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
