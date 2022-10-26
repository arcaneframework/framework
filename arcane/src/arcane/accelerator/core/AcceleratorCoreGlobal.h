// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorCoreGlobal.h                                     (C) 2000-2022 */
/*                                                                           */
/* Déclarations générales pour le support des accélérateurs.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_ACCELERATORCOREGLOBAL_H
#define ARCANE_ACCELERATOR_CORE_ACCELERATORCOREGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_arcane_accelerator_core
#define ARCANE_ACCELERATOR_CORE_EXPORT ARCANE_EXPORT
#else
#define ARCANE_ACCELERATOR_CORE_EXPORT ARCANE_IMPORT
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IAcceleratorMng;
class Runner;
class RunQueue;
class RunCommand;
class RunQueueEvent;
class AcceleratorRuntimeInitialisationInfo;
class RunQueueBuildInfo;
class MemoryCopyArgs;
class MemoryPrefetchArgs;
class DeviceId;
class DeviceInfo;
class IDeviceInfoList;
enum class eMemoryAdvice;

namespace impl
{
class IRunnerRuntime;
// typedef pour compatibilité avec anciennes versions (octobre 2022)
using IRunQueueRuntime = IRunnerRuntime;
class IRunQueueStream;
class RunCommandImpl;
class IReduceMemoryImpl;
class ReduceMemoryImpl;
class RunQueueImpl;
class IRunQueueEventImpl;
class RunCommandLaunchInfo;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Politique d'exécution pour un Runner.
 */
enum class eExecutionPolicy
{
  //! Aucune politique d'exécution
  None,
  //! Politique d'exécution séquentielle
  Sequential,
  //! Politique d'exécution multi-thread
  Thread,
  //! Politique d'exécution utilisant l'environnement CUDA
  CUDA,
  //! Politique d'exécution utilisant l'environnement HIP
  HIP
};

//! Affiche le nom de la politique d'exécution
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
std::ostream& operator<<(std::ostream& o,eExecutionPolicy exec_policy);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Politique des opératations de réduction sur les accélérateurs
 */
enum class eDeviceReducePolicy
{
  //! Utilise des opérations atomiques entre les blocs
  Atomic = 1,
  //! Utilise un noyau de calcul avec une synchronisations entre les blocs.
  Grid = 2
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT IReduceMemoryImpl*
internalGetOrCreateReduceMemoryImpl(RunCommand* command);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indique si \a exec_policy correspond à un accélérateur
inline bool
isAcceleratorPolicy(eExecutionPolicy exec_policy)
{
  return exec_policy==eExecutionPolicy::CUDA || exec_policy==eExecutionPolicy::HIP;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indique si on utilise le runtime CUDA
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
bool isUsingCUDARuntime();

//! Positionne l'utilisation du runtime CUDA
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
void setUsingCUDARuntime(bool v);

//! Récupère l'implémentation CUDA de RunQueue (peut être nulle)
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
IRunnerRuntime* getCUDARunQueueRuntime();

//! Positionne l'implémentation CUDA de RunQueue.
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
void setCUDARunQueueRuntime(IRunnerRuntime* v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indique si on utilise le runtime HIP
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
bool isUsingHIPRuntime();

//! Positionne l'utilisation du runtime HIP
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
void setUsingHIPRuntime(bool v);

//! Récupère l'implémentation HIP de RunQueue (peut être nulle)
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
IRunnerRuntime* getHIPRunQueueRuntime();

//! Positionne l'implémentation HIP de RunQueue.
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
void setHIPRunQueueRuntime(IRunnerRuntime* v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Récupère l'implémentation Séquentielle de RunQueue
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
IRunnerRuntime* getSequentialRunQueueRuntime();

//! Récupère l'implémentation Thread de RunQueue
extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
IRunnerRuntime* getThreadRunQueueRuntime();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
