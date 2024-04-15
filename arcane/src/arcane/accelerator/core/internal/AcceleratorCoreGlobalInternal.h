// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorCoreGlobalInternal.h                             (C) 2000-2024 */
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
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT IRunnerRuntime*
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
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT IRunnerRuntime*
getHIPRunQueueRuntime();

//! Positionne l'implémentation HIP de RunQueue.
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void setHIPRunQueueRuntime(IRunnerRuntime* v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indique si on utilise le runtime SYCL
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT bool isUsingSYCLRuntime();

//! Positionne l'utilisation du runtime SYCL
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void setUsingSYCLRuntime(bool v);

//! Récupère l'implémentation SYCL de RunQueue (peut être nulle)
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT IRunnerRuntime*
getSYCLRunQueueRuntime();

//! Positionne l'implémentation SYCL de RunQueue.
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT void setSYCLRunQueueRuntime(IRunnerRuntime* v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Récupère l'implémentation accélérateur courante (peut être nulle).
 *
 * Le pointeur retourné est nul si aucun runtime accélérateur n'est positionné.
 * Si isUsingCUDARuntime() est vrai, retourne le runtime associé à CUDA.
 * Si isUsingHIPRuntime() est vrai retourne le runtime associé à HIP.
 */
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT IRunnerRuntime*
getAcceleratorRunnerRuntime();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Récupère l'implémentation Séquentielle de RunQueue
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT IRunnerRuntime*
getSequentialRunQueueRuntime();

//! Récupère l'implémentation Thread de RunQueue
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT IRunnerRuntime*
getThreadRunQueueRuntime();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Classe statique fournissant des fonctions internes à Arcane.
class ARCANE_ACCELERATOR_CORE_EXPORT RuntimeStaticInfo
{
 public:

  static ePointerAccessibility
  getPointerAccessibility(eExecutionPolicy policy, const void* ptr, PointerAttribute* ptr_attr);

  static void
  checkPointerIsAcccessible(eExecutionPolicy policy, const void* ptr,
                            const char* name, const TraceInfo& ti);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
