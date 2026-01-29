// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorCoreGlobalInternal.h                             (C) 2000-2025 */
/*                                                                           */
/* Déclarations générales pour le support des accélérateurs.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_INTERNAL_ACCELERATORCOREGLOBALINTERNAL_H
#define ARCCORE_COMMON_ACCELERATOR_INTERNAL_ACCELERATORCOREGLOBALINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indique si on utilise le runtime CUDA
extern "C++" ARCCORE_COMMON_EXPORT bool isUsingCUDARuntime();

//! Positionne l'utilisation du runtime CUDA
extern "C++" ARCCORE_COMMON_EXPORT void setUsingCUDARuntime(bool v);

//! Récupère l'implémentation CUDA de RunQueue (peut être nulle)
extern "C++" ARCCORE_COMMON_EXPORT IRunnerRuntime*
getCUDARunQueueRuntime();

//! Positionne l'implémentation CUDA de RunQueue.
extern "C++" ARCCORE_COMMON_EXPORT void setCUDARunQueueRuntime(IRunnerRuntime* v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indique si on utilise le runtime HIP
extern "C++" ARCCORE_COMMON_EXPORT bool isUsingHIPRuntime();

//! Positionne l'utilisation du runtime HIP
extern "C++" ARCCORE_COMMON_EXPORT void setUsingHIPRuntime(bool v);

//! Récupère l'implémentation HIP de RunQueue (peut être nulle)
extern "C++" ARCCORE_COMMON_EXPORT IRunnerRuntime*
getHIPRunQueueRuntime();

//! Positionne l'implémentation HIP de RunQueue.
extern "C++" ARCCORE_COMMON_EXPORT void setHIPRunQueueRuntime(IRunnerRuntime* v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indique si on utilise le runtime SYCL
extern "C++" ARCCORE_COMMON_EXPORT bool isUsingSYCLRuntime();

//! Positionne l'utilisation du runtime SYCL
extern "C++" ARCCORE_COMMON_EXPORT void setUsingSYCLRuntime(bool v);

//! Récupère l'implémentation SYCL de RunQueue (peut être nulle)
extern "C++" ARCCORE_COMMON_EXPORT IRunnerRuntime*
getSYCLRunQueueRuntime();

//! Positionne l'implémentation SYCL de RunQueue.
extern "C++" ARCCORE_COMMON_EXPORT void setSYCLRunQueueRuntime(IRunnerRuntime* v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Récupère l'implémentation accélérateur courante (peut être nulle).
 *
 * Le pointeur retourné est nul si aucun runtime accélérateur n'est positionné.
 * Si isUsingCUDARuntime() est vrai, retourne le runtime associé à CUDA.
 * Si isUsingHIPRuntime() est vrai retourne le runtime associé à HIP.
 */
extern "C++" ARCCORE_COMMON_EXPORT IRunnerRuntime*
getAcceleratorRunnerRuntime();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Récupère l'implémentation Séquentielle de RunQueue
extern "C++" ARCCORE_COMMON_EXPORT IRunnerRuntime*
getSequentialRunQueueRuntime();

//! Récupère l'implémentation Thread de RunQueue
extern "C++" ARCCORE_COMMON_EXPORT IRunnerRuntime*
getThreadRunQueueRuntime();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Affiche l'UUID d'un accélérateur
extern "C++" ARCCORE_COMMON_EXPORT void
printUUID(std::ostream& o, char bytes[16]);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Classe statique fournissant des fonctions internes à Arcane.
class ARCCORE_COMMON_EXPORT RuntimeStaticInfo
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
/*!
 * \brief Initialise \a runner avec les informations de \a acc_info.
 *
 * Cette fonction appelle runner.setAsCurrentDevice() après
 * l'initialisation.
 */
extern "C++" ARCCORE_COMMON_EXPORT void
arccoreInitializeRunner(Runner& runner, ITraceMng* tm,
                        const AcceleratorRuntimeInitialisationInfo& acc_info);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
